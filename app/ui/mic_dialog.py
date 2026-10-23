"""Проверка микрофона: устройство, живой уровень, запись, прослушивание, распознавание."""

from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from app.asr import audio
from app.asr.recorder import MicRecorder
from app.asr.registry import get_asr_engine
from app.config import Settings
from app.tts import TTSService
from app.ui.level_meter import LevelMeter
from app.ui.workers import PlayFileWorker, TranscribeWorker

RECORD_SECONDS = 3.0


class MicDialog(QDialog):
    """Проверяем микрофон до того, как включать распознавание в чате."""

    def __init__(self, settings: Settings, tts: TTSService, parent=None) -> None:
        super().__init__(parent)
        self.settings = settings
        self._tts = tts
        self._samples = None
        self._play_worker: PlayFileWorker | None = None
        self._transcribe_worker: TranscribeWorker | None = None

        self.setWindowTitle("Проверка микрофона")
        self.setMinimumWidth(560)
        self.setStyleSheet(parent.styleSheet() if parent else "")

        self._recorder = MicRecorder(self)
        self._recorder.level.connect(self._on_level)
        self._recorder.failed.connect(self._on_failed)

        self._build_ui()
        self._start_monitor()

    # --- интерфейс -------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        self.device_combo = QComboBox()
        self.device_combo.addItems(MicRecorder.devices() or ["микрофон не найден"])
        index = self.device_combo.findText(self.settings.asr.input_device)
        if index >= 0:
            self.device_combo.setCurrentIndex(index)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        root.addWidget(QLabel("Устройство:"))
        root.addWidget(self.device_combo)

        meter_row = QHBoxLayout()
        self.meter = LevelMeter(bars=16)
        meter_row.addWidget(QLabel("Уровень:"))
        meter_row.addWidget(self.meter)
        meter_row.addStretch()
        root.addLayout(meter_row)

        self.status = QLabel("Говорите — полоски должны реагировать на голос.")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        buttons = QHBoxLayout()
        self.record_button = QPushButton(f"Записать {RECORD_SECONDS:.0f} с")
        self.record_button.clicked.connect(self._record)
        self.play_button = QPushButton("Прослушать")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self._play)
        self.transcribe_button = QPushButton("Распознать")
        self.transcribe_button.setEnabled(False)
        self.transcribe_button.clicked.connect(self._transcribe)
        buttons.addWidget(self.record_button)
        buttons.addWidget(self.play_button)
        buttons.addWidget(self.transcribe_button)
        buttons.addStretch()
        root.addLayout(buttons)

        self.result = QPlainTextEdit()
        self.result.setReadOnly(True)
        self.result.setPlaceholderText("Здесь появится распознанный текст")
        self.result.setMinimumHeight(90)
        root.addWidget(self.result)

        hint = QLabel(
            "Запись идёт 3 секунды. Если полоски не двигаются — выберите другое "
            "устройство или проверьте разрешение Windows на доступ к микрофону."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#6b7280;")
        root.addWidget(hint)

        close_buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_buttons.rejected.connect(self.reject)
        close_buttons.button(QDialogButtonBox.StandardButton.Close).setText("Закрыть")
        root.addWidget(close_buttons)

    # --- микрофон --------------------------------------------------------
    def _start_monitor(self) -> bool:
        self._recorder.stop()
        if not self._recorder.start(self.device_combo.currentText()):
            self.meter.set_state("idle")
            return False
        self.meter.set_state("ready")
        self.status.setText(f"Слушаю: {self._recorder.device_name}")
        return True

    def _on_device_changed(self, _index: int) -> None:
        name = self.device_combo.currentText()
        if name and name != "микрофон не найден":
            self.settings.asr.input_device = name
        self._samples = None
        self.play_button.setEnabled(False)
        self.transcribe_button.setEnabled(False)
        self.result.clear()
        self._start_monitor()

    def _on_level(self, value: float) -> None:
        self.meter.set_level(value)

    def _on_failed(self, message: str) -> None:
        self.status.setText(message)
        self.meter.set_state("idle")

    def _record(self) -> None:
        if not self._recorder.is_open and not self._start_monitor():
            return
        self._samples = None
        self.result.clear()
        self.record_button.setEnabled(False)
        self.play_button.setEnabled(False)
        self.transcribe_button.setEnabled(False)
        self._recorder.begin_capture()
        self.meter.set_state("recording")
        self.status.setText(f"Говорите… осталось {RECORD_SECONDS:.0f} с")
        QTimer.singleShot(int(RECORD_SECONDS * 1000), self._finish_record)

    def _finish_record(self) -> None:
        samples = self._recorder.end_capture()
        rate = self._recorder.rate
        self.meter.set_state("ready")
        self.record_button.setEnabled(True)

        if samples.size == 0:
            self.status.setText("Ничего не записалось — проверьте микрофон")
            return

        data = audio.to_mono_16k(samples, rate, 1)
        self._samples = data
        seconds = data.size / float(audio.TARGET_RATE)
        self.status.setText(
            f"Записано {seconds:.1f} с, пик {audio.peak(data):.2f}, "
            f"громкость {audio.rms(data):.3f}"
        )
        has_audio = data.size > 0
        self.play_button.setEnabled(has_audio)
        self.transcribe_button.setEnabled(has_audio)

    # --- прослушивание и распознавание -----------------------------------
    def _play(self) -> None:
        if self._samples is None or self._samples.size == 0:
            return
        path = Path(tempfile.gettempdir()) / "mlvh_mic_check.wav"
        try:
            audio.write_wav(path, self._samples, audio.TARGET_RATE)
        except OSError as exc:
            self.status.setText(f"Не удалось сохранить запись: {exc}")
            return

        self._stop_play_worker()
        worker = PlayFileWorker(self._tts, path, self.settings.tts.playback_volume, self)
        worker.finished_ok.connect(lambda: self.status.setText("Прослушали запись"))
        worker.failed.connect(self.status.setText)
        worker.finished.connect(self._forget_play_worker)
        worker.finished.connect(worker.deleteLater)
        self._play_worker = worker
        worker.start()

    def _transcribe(self) -> None:
        if self._samples is None or self._samples.size == 0:
            return

        engine = get_asr_engine(self.settings.asr.engine)
        info = engine.probe()
        if not info.available:
            self.status.setText(f"{info.title} не установлен: {info.hint}")
            return

        self._stop_transcribe_worker()
        worker = TranscribeWorker(engine, self._samples, self.settings.asr, self)
        worker.note.connect(self.status.setText)
        worker.done.connect(self._on_transcribed)
        worker.failed.connect(self.status.setText)
        worker.finished.connect(self._forget_transcribe_worker)
        worker.finished.connect(worker.deleteLater)
        self._transcribe_worker = worker
        self.transcribe_button.setEnabled(False)
        worker.start()

    def _on_transcribed(self, text: str) -> None:
        self.transcribe_button.setEnabled(True)
        text = text.strip()
        if not text:
            self.status.setText("Тишина или неразборчиво — попробуйте ещё раз")
            return
        self.result.setPlainText(text)
        self.status.setText("Распознано")

    # --- остановка потоков ----------------------------------------------
    def _stop_play_worker(self) -> None:
        worker = self._play_worker
        if worker is None:
            return
        try:
            if worker.isRunning():
                self._tts.stop()
                worker.wait(2000)
        except RuntimeError:
            pass

    def _stop_transcribe_worker(self) -> None:
        worker = self._transcribe_worker
        if worker is None:
            return
        try:
            if worker.isRunning():
                worker.requestInterruption()
                if not worker.wait(2000):
                    worker.terminate()
                    worker.wait(500)
        except RuntimeError:
            pass

    def _forget_play_worker(self) -> None:
        self._play_worker = None

    def _forget_transcribe_worker(self) -> None:
        self._transcribe_worker = None

    def closeEvent(self, event) -> None:
        self._recorder.stop()
        self._stop_play_worker()
        self._stop_transcribe_worker()
        super().closeEvent(event)
