"""Проверка микрофона: устройство, живой уровень, запись, прослушивание, распознавание."""

from __future__ import annotations

import tempfile
import time
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
from app.ui.workers import PlayFileWorker, PrepareWorker, TranscribeWorker

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
        self._prepare_worker: PrepareWorker | None = None
        self._engine_ready = False
        self._fallback_tried = False
        self._started_at = 0.0
        self._seconds_left = RECORD_SECONDS

        self.setWindowTitle("Проверка микрофона")
        self.setMinimumWidth(560)
        self.setStyleSheet(parent.styleSheet() if parent else "")

        self._recorder = MicRecorder(self)
        self._recorder.level.connect(self._on_level)
        self._recorder.failed.connect(self._on_failed)

        self._countdown = QTimer(self)
        self._countdown.setInterval(200)
        self._countdown.timeout.connect(self._tick_recording)

        self._progress = QTimer(self)
        self._progress.setInterval(500)
        self._progress.timeout.connect(self._tick_progress)

        self._build_ui()
        self._start_monitor()
        QTimer.singleShot(200, self._warm_up)

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
        self.status.setStyleSheet("color:#cbd5e1;")
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
        self.result.setPlaceholderText(
            "Шаг 3: нажмите «Распознать» — здесь появится текст. "
            "Первый запуск грузит модель, это занимает до полуминуты."
        )
        self.result.setMinimumHeight(90)
        root.addWidget(self.result)

        hint = QLabel(
            "Как это работает: 1) выберите устройство и говорите — полоски "
            "двигаются; 2) «Записать 3 с» и говорите; 3) «Прослушать» — "
            "убедиться, что запись слышно; 4) «Распознать» — проверить, что "
            "движок понимает речь. Если полоски стоят — другое устройство или "
            "разрешение Windows на микрофон."
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
        self._set_status(f"Слушаю: {self._recorder.device_name}. Говорите — полоски должны двигаться.")
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
        self._set_status(message, "#f87171")
        self.meter.set_state("idle")

    # --- прогрев модели --------------------------------------------------
    def _warm_up(self) -> None:
        """Грузим модель заранее: иначе первый «Распознать» выглядит зависшим."""
        engine = get_asr_engine(self.settings.asr.engine)
        info = engine.probe()
        if not info.available:
            self._set_status(f"{info.title} не установлен: {info.hint}", "#f87171")
            self.transcribe_button.setEnabled(False)
            return

        self._set_status(
            "Загружаю модель распознавания… Первый раз это до полуминуты, "
            "потом — мгновенно.",
            "#fbbf24",
        )
        self.transcribe_button.setEnabled(False)

        worker = PrepareWorker(engine, self.settings.asr, self)
        worker.ready.connect(self._on_engine_ready)
        worker.failed.connect(self._on_engine_failed)
        worker.finished.connect(self._forget_prepare_worker)
        worker.finished.connect(worker.deleteLater)
        self._prepare_worker = worker
        worker.start()

    def _on_engine_ready(self, device: str) -> None:
        self._engine_ready = True
        self._set_status(
            f"Модель готова ({device}). Запишите 3 секунды и нажмите «Распознать».",
            "#34d399",
        )
        if self._samples is not None and self._samples.size:
            self.transcribe_button.setEnabled(True)

    def _on_engine_failed(self, message: str) -> None:
        self._engine_ready = False
        self._set_status(f"Модель не загрузилась: {message}", "#f87171")
        self.result.setPlainText(f"Ошибка модели: {message}")

    def _forget_prepare_worker(self) -> None:
        self._prepare_worker = None

    def _set_status(self, text: str, color: str = "#cbd5e1") -> None:
        self.status.setText(text)
        self.status.setStyleSheet(f"color:{color};")

    def _tick_recording(self) -> None:
        left = RECORD_SECONDS - (time.monotonic() - self._started_at)
        self._seconds_left = max(0.0, left)
        self._set_status(f"Идёт запись… осталось {self._seconds_left:.1f} с", "#ef4444")

    def _tick_progress(self) -> None:
        elapsed = time.monotonic() - self._started_at
        if not self._engine_ready:
            self._set_status(
                f"Загружаю модель… {elapsed:.0f} с. Первый запуск — до полуминуты.",
                "#fbbf24",
            )
        else:
            self._set_status(f"Распознаю… {elapsed:.0f} с", "#fbbf24")

    # --- запись ----------------------------------------------------------
    def _record(self) -> None:
        if not self._recorder.is_open and not self._start_monitor():
            return
        self._samples = None
        self.result.clear()
        self._fallback_tried = False
        self.record_button.setEnabled(False)
        self.play_button.setEnabled(False)
        self.transcribe_button.setEnabled(False)
        self._recorder.begin_capture()
        self.meter.set_state("recording")
        self._started_at = time.monotonic()
        self._countdown.start()
        self._tick_recording()
        QTimer.singleShot(int(RECORD_SECONDS * 1000), self._finish_record)

    def _finish_record(self) -> None:
        self._countdown.stop()
        samples = self._recorder.end_capture()
        rate = self._recorder.rate
        self.meter.set_state("ready")
        self.record_button.setEnabled(True)

        if samples.size == 0:
            self._set_status(
                "Ничего не записалось. Проверьте, что микрофон не занят другой "
                "программой и разрешён в настройках Windows.",
                "#f87171",
            )
            return

        data = audio.to_mono_16k(samples, rate, 1)
        self._samples = data
        seconds = data.size / float(audio.TARGET_RATE)
        volume = audio.rms(data)
        self._set_status(
            f"Записано {seconds:.1f} с, громкость {volume:.3f}. "
            "Послушайте запись или нажмите «Распознать».",
            "#34d399" if volume > 0.004 else "#f87171",
        )
        if volume <= 0.004:
            self.result.setPlainText(
                "Запись почти тихая: похоже, микрофон не слышит голос. "
                "Проверьте устройство и уровень в Windows."
            )
        has_audio = data.size > 0
        self.play_button.setEnabled(has_audio)
        self.transcribe_button.setEnabled(has_audio and self._engine_ready)

    # --- прослушивание и распознавание -----------------------------------
    def _play(self) -> None:
        if self._samples is None or self._samples.size == 0:
            return
        path = Path(tempfile.gettempdir()) / "mlvh_mic_check.wav"
        try:
            audio.write_wav(path, self._samples, audio.TARGET_RATE)
        except OSError as exc:
            self._set_status(f"Не удалось сохранить запись: {exc}", "#f87171")
            return

        self._stop_play_worker()
        worker = PlayFileWorker(self._tts, path, self.settings.tts.playback_volume, self)
        worker.finished_ok.connect(lambda: self._set_status("Прослушали запись", "#34d399"))
        worker.failed.connect(lambda message: self._set_status(message, "#f87171"))
        worker.finished.connect(self._forget_play_worker)
        worker.finished.connect(worker.deleteLater)
        self._play_worker = worker
        worker.start()

    def _transcribe(self) -> None:
        if self._samples is None or self._samples.size == 0:
            self._set_status("Сначала запишите 3 секунды", "#fbbf24")
            return

        engine = get_asr_engine(self.settings.asr.engine)
        info = engine.probe()
        if not info.available:
            self._set_status(f"{info.title} не установлен: {info.hint}", "#f87171")
            return

        self._stop_transcribe_worker()
        self._fallback_tried = False
        worker = TranscribeWorker(engine, self._samples, self.settings.asr, self)
        worker.done.connect(self._on_transcribed)
        worker.failed.connect(self._on_transcribe_failed)
        worker.finished.connect(self._forget_transcribe_worker)
        worker.finished.connect(worker.deleteLater)
        self._transcribe_worker = worker

        self._started_at = time.monotonic()
        self._progress.start()
        self._tick_progress()
        self.transcribe_button.setEnabled(False)
        self.record_button.setEnabled(False)
        worker.start()

    def _on_transcribed(self, text: str) -> None:
        self._progress.stop()
        self.transcribe_button.setEnabled(True)
        self.record_button.setEnabled(True)

        text = text.strip()
        if text:
            self.result.setPlainText(text)
            self._set_status(f"Распознано: {text}", "#34d399")
            return

        # Whisper на тихой или короткой записи иногда возвращает пусто —
        # пробуем лёгкий Vosk, если он установлен. Ровно один раз.
        fallback = get_asr_engine("vosk")
        if (
            not self._fallback_tried
            and self.settings.asr.engine != "vosk"
            and fallback.probe().available
        ):
            self._fallback_tried = True
            self._set_status("Whisper не разобрал — пробую Vosk…", "#fbbf24")
            self._start_fallback(fallback)
            return

        self.result.setPlainText(
            "Распознать не удалось: движок не нашёл речи. Запишите ещё раз "
            "погромче и ближе к микрофону."
        )
        self._set_status("Тишина или неразборчиво — попробуйте ещё раз", "#f87171")

    def _start_fallback(self, engine) -> None:
        worker = TranscribeWorker(engine, self._samples, self.settings.asr, self)
        worker.done.connect(self._on_transcribed)
        worker.failed.connect(self._on_transcribe_failed)
        worker.finished.connect(self._forget_transcribe_worker)
        worker.finished.connect(worker.deleteLater)
        self._transcribe_worker = worker
        worker.start()

    def _on_transcribe_failed(self, message: str) -> None:
        self._progress.stop()
        self.transcribe_button.setEnabled(True)
        self.record_button.setEnabled(True)
        self._set_status(f"Ошибка распознавания: {message}", "#f87171")
        self.result.setPlainText(f"Ошибка распознавания:\n{message}")

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
        self._countdown.stop()
        self._progress.stop()
        self._recorder.stop()
        self._stop_play_worker()
        self._stop_transcribe_worker()
        worker = self._prepare_worker
        if worker is not None:
            try:
                if worker.isRunning():
                    worker.wait(3000)
            except RuntimeError:
                pass
        super().closeEvent(event)
