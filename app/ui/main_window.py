"""Главное окно MLVH: чат слева, настройки голоса справа."""

from __future__ import annotations

import threading
from html import escape
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from app.config import (
    AVAILABLE_MODELS,
    BASE_DIR,
    FALLBACK_VOICES,
    Settings,
    get_api_key,
    load_settings,
    mask_key,
    save_api_key,
    save_settings,
)
from app.deepseek_client import DeepSeekClient, DeepSeekError
from app.engines.registry import engine_infos, get_engine
from app.engines.silero_engine import SPEAKERS as SILERO_SPEAKERS
from app.personas import (
    Persona,
    is_builtin_persona,
    new_persona,
    persona_by_id,
)
from app.tts import SpeechPipeline, TTSService, voice_label
from app.ui.persona_dialog import PersonaDialog
from app.ui.workers import ChatWorker, SpeakWorker, SpeechSignals, VoiceListWorker

QSS = """
QMainWindow, QWidget { background: #111827; color: #e5e7eb; }
QLabel#Title { font-size: 22px; font-weight: 700; color: #f9fafb; }
QLabel#Subtitle { color: #9ca3af; }
QLabel#KeyOk { color: #34d399; }
QLabel#KeyMissing { color: #f87171; }
QTextBrowser#Chat {
    background: #0b1220; border: 1px solid #1f2937; border-radius: 10px;
    padding: 10px; font-size: 14px;
}
QGroupBox {
    border: 1px solid #1f2937; border-radius: 10px; margin-top: 14px; padding-top: 8px;
    font-weight: 600; color: #cbd5e1;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QLineEdit {
    background: #0b1220; border: 1px solid #1f2937; border-radius: 8px;
    padding: 8px 10px; font-size: 14px;
}
QLineEdit:focus { border-color: #3b82f6; }
QPushButton {
    background: #1f2937; border: 1px solid #334155; border-radius: 8px;
    padding: 7px 12px; color: #e5e7eb;
}
QPushButton:hover { background: #273449; }
QPushButton:disabled { color: #64748b; border-color: #1f2937; }
QPushButton#Primary { background: #2563eb; border-color: #2563eb; color: white; font-weight: 600; }
QPushButton#Primary:hover { background: #1d4ed8; }
QComboBox, QSpinBox {
    background: #0b1220; border: 1px solid #1f2937; border-radius: 8px; padding: 5px 8px;
}
QSlider::groove:horizontal { height: 4px; background: #1f2937; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 14px; margin: -6px 0; border-radius: 7px; background: #3b82f6;
}
QCheckBox { color: #cbd5e1; }
QStatusBar { color: #9ca3af; }
"""


class MainWindow(QMainWindow):
    """Окно: слева переписка, справа настройки голоса."""

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self.settings: Settings = settings or load_settings()
        self.tts = TTSService()

        self.history: list[dict] = []
        self._pending_user_index: int | None = None
        self._last_reply = ""
        self._reply_parts: list[str] = []
        self._voices: list[dict] = []
        self._body_fmt = QTextCharFormat()
        self._chat_worker: ChatWorker | None = None
        self._speak_worker: SpeakWorker | None = None
        self._voice_worker: VoiceListWorker | None = None
        self._pipeline: SpeechPipeline | None = None

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)
        self._save_timer.timeout.connect(self._persist)

        # Озвучка работает в фоновых потоках, поэтому в GUI сообщаем через сигналы.
        self._speech_signals = SpeechSignals(self)
        self._speech_signals.done.connect(lambda: self.statusBar().showMessage("Готов", 3000))
        self._speech_signals.error.connect(self._on_speak_failed)
        self._speech_signals.note.connect(self._on_speak_note)
        self._speech_signals.prepare_done.connect(self._on_prepare_done)
        self._speech_signals.prepare_failed.connect(self._on_prepare_failed)

        self.setWindowTitle("MLVH — My Little Voice Helper")
        self.resize(1040, 700)
        self.setStyleSheet(QSS)

        self._build_ui()
        self._populate_engines()
        self._populate_personas()
        self._apply_settings()
        self._refresh_key_state()
        self._load_voices()

        # Локальные модели грузятся заранее: Chatterbox ~20 с, Silero ~4 с.
        if self.settings.tts.engine in ("chatterbox", "silero"):
            self._warm_up_engine(self.settings.tts.engine)

    # --- сборка интерфейса ---------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)
        root.addLayout(self._build_header())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_chat_panel())
        splitter.addWidget(self._build_sidebar())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([700, 330])
        root.addWidget(splitter, 1)
        root.addLayout(self._build_input_row())

        self.statusBar().showMessage("Готов")
        self.tokens_label = QLabel("токены: —")
        self.statusBar().addPermanentWidget(self.tokens_label)

    def _build_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        title = QLabel("MLVH")
        title.setObjectName("Title")
        subtitle = QLabel("My Little Voice Helper")
        subtitle.setObjectName("Subtitle")
        row.addWidget(title)
        row.addWidget(subtitle)
        row.addStretch()

        row.addWidget(QLabel("Модель:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(AVAILABLE_MODELS)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        row.addWidget(self.model_combo)

        self.key_state = QLabel("")
        row.addWidget(self.key_state)

        self.key_button = QPushButton("Ввести ключ")
        self.key_button.clicked.connect(self._ask_api_key)
        row.addWidget(self.key_button)
        return row

    def _build_chat_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self.chat = QTextBrowser()
        self.chat.setObjectName("Chat")
        self.chat.setOpenExternalLinks(True)
        self.chat.setHtml(f'<p style="color:#94a3b8;">{escape(self._welcome_text())}</p>')
        layout.addWidget(self.chat, 1)
        return panel

    def _build_input_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Спросите что-нибудь… (Enter — отправить)")
        self.input.returnPressed.connect(self._send)

        self.send_button = QPushButton("Отправить")
        self.send_button.setObjectName("Primary")
        self.send_button.clicked.connect(self._send)

        self.replay_button = QPushButton("Повторить")
        self.replay_button.setToolTip("Озвучить последний ответ ещё раз")
        self.replay_button.clicked.connect(self._replay)

        self.stop_button = QPushButton("Стоп")
        self.stop_button.clicked.connect(self._stop_all)

        row.addWidget(self.input, 1)
        row.addWidget(self.send_button)
        row.addWidget(self.replay_button)
        row.addWidget(self.stop_button)
        return row

    def _build_sidebar(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(380)

        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(6)

        persona_box = QGroupBox("Характер")
        persona_layout = QVBoxLayout(persona_box)
        self.persona_combo = QComboBox()
        self.persona_combo.setMinimumWidth(200)
        self.persona_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.persona_combo.setMinimumContentsLength(14)
        self.persona_combo.currentIndexChanged.connect(self._on_persona_changed)
        persona_layout.addWidget(self.persona_combo)

        persona_buttons = QHBoxLayout()
        self.add_persona_button = QPushButton("Добавить")
        self.add_persona_button.setToolTip("Создать свой характер")
        self.add_persona_button.clicked.connect(self._add_persona)
        self.edit_persona_button = QPushButton("Изменить…")
        self.edit_persona_button.clicked.connect(self._edit_persona)
        persona_buttons.addWidget(self.add_persona_button)
        persona_buttons.addWidget(self.edit_persona_button)
        persona_layout.addLayout(persona_buttons)

        persona_buttons_2 = QHBoxLayout()
        self.delete_persona_button = QPushButton("Удалить")
        self.delete_persona_button.setToolTip("Удалить свой характер")
        self.delete_persona_button.clicked.connect(self._delete_persona)
        self.reset_persona_button = QPushButton("Сброс")
        self.reset_persona_button.setToolTip("Вернуть встроенный характер")
        self.reset_persona_button.clicked.connect(self._reset_persona)
        persona_buttons_2.addWidget(self.delete_persona_button)
        persona_buttons_2.addWidget(self.reset_persona_button)
        persona_layout.addLayout(persona_buttons_2)
        layout.addWidget(persona_box)

        voice_box = QGroupBox("Голос")
        voice_layout = QVBoxLayout(voice_box)

        voice_layout.addWidget(QLabel("Движок:"))
        self.engine_combo = QComboBox()
        self.engine_combo.setMinimumWidth(200)
        self.engine_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.engine_combo.setMinimumContentsLength(14)
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        voice_layout.addWidget(self.engine_combo)

        self.engine_hint = QLabel("")
        self.engine_hint.setWordWrap(True)
        self.engine_hint.setStyleSheet("color:#6b7280;")
        voice_layout.addWidget(self.engine_hint)

        reference_row = QHBoxLayout()
        self.reference_button = QPushButton("Образец голоса…")
        self.reference_button.setToolTip(
            "Файл .wav/.mp3 с голосом-образцом для Chatterbox. "
            "Если не выбрать, образец будет взят из текущего Edge-голоса."
        )
        self.reference_button.clicked.connect(self._pick_reference)
        self.reference_label = QLabel("из Edge-голоса")
        self.reference_label.setStyleSheet("color:#6b7280;")
        self.reference_label.setWordWrap(True)
        reference_row.addWidget(self.reference_button)
        voice_layout.addLayout(reference_row)
        voice_layout.addWidget(self.reference_label)

        voice_layout.addWidget(QLabel("Голос:"))
        self.voice_combo = QComboBox()
        self.voice_combo.setMinimumWidth(200)
        self.voice_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.voice_combo.setMinimumContentsLength(14)
        self.voice_combo.currentIndexChanged.connect(self._on_voice_changed)
        voice_layout.addWidget(self.voice_combo)

        self.all_voices_check = QCheckBox("Все голоса")
        self.all_voices_check.setToolTip("Показать все голоса Edge, а не только русские")
        self.all_voices_check.toggled.connect(self._on_all_voices)
        voice_layout.addWidget(self.all_voices_check)

        self.rate_slider = self._add_slider(voice_layout, "Темп", -50, 50, 0, " %", self._on_rate)
        self.volume_slider = self._add_slider(voice_layout, "Громкость", -50, 50, 0, " %", self._on_volume)
        self.pitch_slider = self._add_slider(voice_layout, "Тон", -50, 50, 0, " Гц", self._on_pitch)
        self.play_slider = self._add_slider(voice_layout, "Плеер", 0, 100, 90, " %", self._on_play_volume)
        self.exaggeration_slider = self._add_slider(
            voice_layout, "Живость", 0, 100, 50, " %", self._on_exaggeration
        )
        self.exaggeration_slider.setToolTip(
            "Выразительность локального движка Chatterbox. На Edge не влияет."
        )
        self.cfg_slider = self._add_slider(
            voice_layout, "Похожесть", 0, 100, 50, " %", self._on_cfg
        )
        self.cfg_slider.setToolTip(
            "Насколько голос держится образца (cfg_weight у Chatterbox). "
            "Выше — чётче и ближе к образцу, ниже — свободнее интонация."
        )

        self.silero_row = QHBoxLayout()
        self.silero_label = QLabel("Голос Silero:")
        self.silero_label.setFixedWidth(78)
        self.silero_combo = QComboBox()
        for key, title in SILERO_SPEAKERS.items():
            self.silero_combo.addItem(title, key)
        self.silero_combo.currentIndexChanged.connect(self._on_silero_speaker)
        self.silero_row.addWidget(self.silero_label)
        self.silero_row.addWidget(self.silero_combo, 1)
        voice_layout.addLayout(self.silero_row)

        chars_row = QHBoxLayout()
        chars_row.addWidget(QLabel("Читать:"))
        self.chars_spin = QSpinBox()
        self.chars_spin.setRange(200, 10000)
        self.chars_spin.setSingleStep(100)
        self.chars_spin.setSuffix(" симв.")
        self.chars_spin.valueChanged.connect(self._on_max_chars)
        chars_row.addWidget(self.chars_spin)
        chars_row.addStretch()
        voice_layout.addLayout(chars_row)

        self.speak_check = QCheckBox("Озвучивать ответы")
        self.speak_check.toggled.connect(self._on_speak_enabled)
        voice_layout.addWidget(self.speak_check)

        self.test_button = QPushButton("Проверить голос")
        self.test_button.clicked.connect(self._test_voice)
        voice_layout.addWidget(self.test_button)
        layout.addWidget(voice_box)

        dialog_box = QGroupBox("Диалог")
        dialog_layout = QVBoxLayout(dialog_box)
        self.thinking_check = QCheckBox("Режим рассуждений")
        self.thinking_check.setToolTip("Модель сначала обдумывает ответ — медленнее, но глубже")
        self.thinking_check.toggled.connect(self._on_thinking)
        dialog_layout.addWidget(self.thinking_check)
        clear_button = QPushButton("Очистить чат")
        clear_button.clicked.connect(self._clear_chat)
        dialog_layout.addWidget(clear_button)
        layout.addWidget(dialog_box)

        hint = QLabel(
            "Голос — бесплатный Edge TTS (нужен интернет).\n"
            "Ответы — DeepSeek, нужен ключ с platform.deepseek.com."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#6b7280;")
        layout.addWidget(hint)
        layout.addStretch()

        scroll.setWidget(holder)
        return scroll

    def _add_slider(
        self,
        layout: QVBoxLayout,
        title: str,
        minimum: int,
        maximum: int,
        value: int,
        suffix: str,
        on_change,
    ) -> QSlider:
        """Строка «подпись — ползунок — значение»."""
        row = QHBoxLayout()
        label = QLabel(title)
        label.setFixedWidth(78)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)

        value_label = QLabel("")
        value_label.setFixedWidth(56)
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        def fmt(raw: int) -> str:
            sign = "+" if raw > 0 else ""
            return f"{sign}{raw}{suffix}"

        value_label.setText(fmt(value))
        slider.valueChanged.connect(lambda raw: (value_label.setText(fmt(raw)), on_change(raw)))

        row.addWidget(label)
        row.addWidget(slider, 1)
        row.addWidget(value_label)
        layout.addLayout(row)
        return slider

    # --- настройки ------------------------------------------------------
    def _apply_settings(self) -> None:
        model = self.settings.model
        tts = self.settings.tts

        for widget in (self.model_combo, self.speak_check, self.thinking_check, self.persona_combo):
            widget.blockSignals(True)
        self.model_combo.setCurrentText(
            model.name if model.name in AVAILABLE_MODELS else AVAILABLE_MODELS[0]
        )
        persona_index = self.persona_combo.findData(self.settings.active_persona)
        if persona_index >= 0:
            self.persona_combo.setCurrentIndex(persona_index)
        self.speak_check.setChecked(bool(tts.enabled))
        self.thinking_check.setChecked(bool(model.thinking))
        for widget in (self.model_combo, self.speak_check, self.thinking_check, self.persona_combo):
            widget.blockSignals(False)

        self.rate_slider.setValue(int(tts.rate))
        self.volume_slider.setValue(int(tts.volume))
        self.pitch_slider.setValue(int(tts.pitch))
        self.play_slider.setValue(int(round(tts.playback_volume * 100)))
        self.exaggeration_slider.setValue(int(round(tts.chatterbox_exaggeration * 100)))
        self.cfg_slider.setValue(int(round(tts.chatterbox_cfg * 100)))
        silero_index = self.silero_combo.findData(tts.silero_speaker)
        if silero_index >= 0:
            self.silero_combo.blockSignals(True)
            self.silero_combo.setCurrentIndex(silero_index)
            self.silero_combo.blockSignals(False)
        self.chars_spin.setValue(int(tts.max_speak_chars))

        engine_index = self.engine_combo.findData(tts.engine)
        if engine_index >= 0 and engine_index != self.engine_combo.currentIndex():
            self.engine_combo.blockSignals(True)
            self.engine_combo.setCurrentIndex(engine_index)
            self.engine_combo.blockSignals(False)
        self._refresh_engine_ui()

    def _touch(self) -> None:
        """Настройки меняются часто — пишем файл с небольшой задержкой."""
        self._save_timer.start()

    def _persist(self) -> None:
        try:
            save_settings(self.settings)
        except OSError as exc:
            self.statusBar().showMessage(f"Не удалось сохранить настройки: {exc}", 5000)

    def _on_model_changed(self, name: str) -> None:
        self.settings.model.name = name
        self._touch()

    def _on_rate(self, value: int) -> None:
        self.settings.tts.rate = int(value)
        self._touch()

    def _on_volume(self, value: int) -> None:
        self.settings.tts.volume = int(value)
        self._touch()

    def _on_pitch(self, value: int) -> None:
        self.settings.tts.pitch = int(value)
        self._touch()

    def _on_play_volume(self, value: int) -> None:
        self.settings.tts.playback_volume = value / 100
        self._touch()

    def _on_max_chars(self, value: int) -> None:
        self.settings.tts.max_speak_chars = int(value)
        self._touch()

    def _on_speak_enabled(self, enabled: bool) -> None:
        self.settings.tts.enabled = bool(enabled)
        self._touch()

    def _on_thinking(self, enabled: bool) -> None:
        self.settings.model.thinking = bool(enabled)
        self._touch()

    # --- голоса ---------------------------------------------------------
    def _populate_engines(self) -> None:
        """Список движков: недоступные показываем, но выбрать не даём."""
        self._engine_infos = {info.id: info for info in engine_infos()}
        self.engine_combo.blockSignals(True)
        self.engine_combo.clear()
        for info in self._engine_infos.values():
            self.engine_combo.addItem(info.label(), info.id)
        index = self.engine_combo.findData(self.settings.tts.engine)
        self.engine_combo.setCurrentIndex(index if index >= 0 else 0)
        self.engine_combo.blockSignals(False)
        self._refresh_engine_ui()

    def _refresh_engine_ui(self) -> None:
        """Подсказка и доступность полей под выбранный движок."""
        engine_id = self.settings.tts.engine
        info = self._engine_infos.get(engine_id)

        if info is None:
            self.engine_hint.setText("")
            return
        self.engine_hint.setText(info.hint.replace("\n", " "))

        is_local = engine_id == "chatterbox"
        is_silero = engine_id == "silero"
        self.reference_button.setEnabled(is_local and info.available)
        self.exaggeration_slider.setEnabled(is_local)
        self.cfg_slider.setEnabled(is_local)

        # Поля Silero видны только для его движка.
        for widget in (self.silero_label, self.silero_combo):
            widget.setVisible(is_silero)

        if not is_local:
            self.reference_label.setText("Образец нужен только для Chatterbox")
        elif self.settings.tts.chatterbox_reference:
            self.reference_label.setText(Path(self.settings.tts.chatterbox_reference).name)
        else:
            self.reference_label.setText("из Edge-голоса (создастся автоматически)")

    def _on_engine_changed(self, index: int) -> None:
        if index < 0:
            return
        engine_id = str(self.engine_combo.itemData(index))
        info = self._engine_infos.get(engine_id)

        if info is not None and not info.available:
            QMessageBox.information(
                self,
                "Движок не установлен",
                f"{info.title} не установлен.\n\n{info.hint}",
            )
            # Остаёмся на Edge, чтобы озвучка не пропала.
            self.engine_combo.blockSignals(True)
            back = self.engine_combo.findData(self.settings.tts.engine)
            self.engine_combo.setCurrentIndex(back if back >= 0 else 0)
            self.engine_combo.blockSignals(False)
            return

        self.settings.tts.engine = engine_id
        self._touch()
        self._refresh_engine_ui()

        if engine_id in ("chatterbox", "silero"):
            self._warm_up_engine(engine_id)

    def _warm_up_engine(self, engine_id: str = "chatterbox") -> None:
        """Грузим локальную модель заранее, чтобы первый ответ не ждал."""
        engine = get_engine(engine_id)
        info = engine.probe()
        if not info.available:
            return
        state_key = f"_warm_state_{engine_id}"
        if getattr(self, state_key, None) in ("loading", "ready"):
            return

        setattr(self, state_key, "loading")
        self.statusBar().showMessage(f"Загружаю модель {info.title}…")

        def work() -> None:
            try:
                engine.prepare(self.settings.tts)
                setattr(self, state_key, "ready")
                self._speech_signals.prepare_done.emit(f"Модель {info.title} готова")
            except Exception as exc:
                setattr(self, state_key, "failed")
                self._speech_signals.prepare_failed.emit(str(exc))

        threading.Thread(target=work, daemon=True).start()

    def _pick_reference(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Образец голоса для Chatterbox",
            str(BASE_DIR / "samples"),
            "Аудио (*.wav *.mp3 *.flac *.ogg)",
        )
        if not path:
            return
        self.settings.tts.chatterbox_reference = path
        self._touch()
        self.reference_label.setText(Path(path).name)
        self.statusBar().showMessage(f"Образец голоса: {Path(path).name}", 5000)

    def _on_exaggeration(self, value: int) -> None:
        self.settings.tts.chatterbox_exaggeration = value / 100
        self._touch()

    def _on_cfg(self, value: int) -> None:
        self.settings.tts.chatterbox_cfg = value / 100
        self._touch()

    def _on_silero_speaker(self, index: int) -> None:
        if index < 0:
            return
        speaker = self.silero_combo.itemData(index)
        if speaker:
            self.settings.tts.silero_speaker = str(speaker)
            self._touch()

    def _on_voice_changed(self, index: int) -> None:
        if index < 0:
            return
        data = self.voice_combo.itemData(index)
        if data:
            self.settings.tts.voice = str(data)
            self._touch()

    def _on_all_voices(self, _checked: bool) -> None:
        self._load_voices()

    def _load_voices(self) -> None:
        if self._is_running(self._voice_worker):
            return
        self.voice_combo.blockSignals(True)
        self.voice_combo.clear()
        self.voice_combo.addItem("загружаю…", self.settings.tts.voice)
        self.voice_combo.blockSignals(False)

        prefix = "" if self.all_voices_check.isChecked() else "ru"
        worker = VoiceListWorker(prefix, self)
        worker.loaded.connect(self._populate_voices)
        worker.failed.connect(self._on_voices_failed)
        worker.finished.connect(self._forget_voice_worker)
        worker.finished.connect(worker.deleteLater)
        self._voice_worker = worker
        worker.start()

    def _populate_voices(self, voices: list[dict]) -> None:
        self._voices = voices
        self.voice_combo.blockSignals(True)
        self.voice_combo.clear()
        for voice in voices:
            self.voice_combo.addItem(voice_label(voice), voice.get("ShortName"))

        index = self.voice_combo.findData(self.settings.tts.voice)
        if index < 0:
            self.voice_combo.addItem(self.settings.tts.voice, self.settings.tts.voice)
            index = self.voice_combo.count() - 1
        self.voice_combo.setCurrentIndex(index)
        self.voice_combo.blockSignals(False)
        self.statusBar().showMessage(f"Голосов доступно: {len(voices)}", 4000)

    def _on_voices_failed(self, message: str) -> None:
        self.voice_combo.blockSignals(True)
        self.voice_combo.clear()
        for name in FALLBACK_VOICES:
            self.voice_combo.addItem(name, name)
        index = self.voice_combo.findData(self.settings.tts.voice)
        self.voice_combo.setCurrentIndex(index if index >= 0 else 0)
        self.voice_combo.blockSignals(False)
        self.statusBar().showMessage(f"Список голосов не загрузился: {message}", 8000)

    # --- характеры ------------------------------------------------------
    def _populate_personas(self) -> None:
        self.persona_combo.blockSignals(True)
        self.persona_combo.clear()
        for persona in self.settings.personas:
            self.persona_combo.addItem(persona.label(), persona.id)
        index = self.persona_combo.findData(self.settings.active_persona)
        self.persona_combo.setCurrentIndex(index if index >= 0 else 0)
        self.persona_combo.blockSignals(False)
        self._sync_persona_buttons()

    def _sync_persona_buttons(self) -> None:
        persona = self.settings.persona()
        is_builtin = persona is not None and is_builtin_persona(persona.id)
        self.reset_persona_button.setEnabled(bool(is_builtin))
        self.delete_persona_button.setEnabled(persona is not None and not is_builtin)

    def _on_persona_changed(self, index: int) -> None:
        if index < 0:
            return
        persona = persona_by_id(self.settings.personas, str(self.persona_combo.itemData(index)))
        if persona is not None:
            self._activate_persona(persona)

    def _activate_persona(self, persona: Persona, announce: bool = True) -> None:
        """Переключаемся на характер: промпт, голос и ползунки — из его пресета."""
        self.settings.apply_persona(persona)
        self._apply_settings()
        self._touch()
        self._sync_persona_buttons()

        # Смена характера — это другой собеседник, старую историю не тянем.
        self.history.clear()
        self._pending_user_index = None
        self._last_reply = ""

        if announce and persona.greeting:
            self._insert_line(f"{persona.name}: ", persona.greeting, "#c4b5fd", "#e5e7eb")
            if self.settings.tts.enabled:
                self._speak(persona.greeting)
        self.statusBar().showMessage(f"Характер: {persona.name}", 4000)

    def _edit_persona(self) -> None:
        persona = self.settings.persona()
        if persona is None:
            return

        dialog = PersonaDialog(persona, self._voices, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        updated = dialog.result_persona()
        if updated is None:
            return

        self.settings.save_persona(updated)
        self._populate_personas()
        self._activate_persona(updated, announce=False)
        self.statusBar().showMessage(f"Характер «{updated.name}» сохранён", 4000)

    def _add_persona(self) -> None:
        """Создаём новый характер на основе заготовки."""
        template = new_persona(self.settings.personas, self.settings.tts.voice)
        dialog = PersonaDialog(template, self._voices, self, create=True)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        created = dialog.result_persona()
        if created is None:
            return

        self.settings.save_persona(created)
        self._populate_personas()
        self._activate_persona(created)
        self._persist()
        self.statusBar().showMessage(f"Характер «{created.name}» создан", 4000)

    def _delete_persona(self) -> None:
        """Удаляем свой характер; встроенные не трогаем."""
        persona = self.settings.persona()
        if persona is None or is_builtin_persona(persona.id):
            return

        answer = QMessageBox.question(
            self,
            "MLVH",
            f"Удалить характер «{persona.name}»? Действие необратимо.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self.settings.personas = [item for item in self.settings.personas if item.id != persona.id]
        fallback = persona_by_id(self.settings.personas, "helper") or self.settings.personas[0]
        self._populate_personas()
        self._activate_persona(fallback, announce=False)
        self._persist()
        self.statusBar().showMessage(f"Характер «{persona.name}» удалён", 4000)

    def _reset_persona(self) -> None:
        persona = self.settings.persona()
        if persona is None:
            return
        if not is_builtin_persona(persona.id):
            QMessageBox.information(
                self, "MLVH", "Этот характер создан вручную — сбрасывать нечего."
            )
            return

        answer = QMessageBox.question(
            self,
            "MLVH",
            f"Вернуть встроенные настройки характера «{persona.name}»?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        restored = self.settings.reset_persona(persona.id)
        self._populate_personas()
        self._activate_persona(restored, announce=False)
        self.statusBar().showMessage(f"Характер «{restored.name}» восстановлен", 4000)

    # --- ключ -----------------------------------------------------------
    def _refresh_key_state(self) -> None:
        key = get_api_key()
        if key:
            self.key_state.setText(f"ключ: {mask_key(key)}")
            self.key_state.setObjectName("KeyOk")
            self.key_button.setText("Сменить ключ")
        else:
            self.key_state.setText("ключ не задан")
            self.key_state.setObjectName("KeyMissing")
            self.key_button.setText("Ввести ключ")
        self.key_state.style().unpolish(self.key_state)
        self.key_state.style().polish(self.key_state)

    def _ask_api_key(self) -> None:
        key, ok = QInputDialog.getText(
            self,
            "Ключ DeepSeek",
            "Вставьте ключ с platform.deepseek.com/api_keys.\n"
            "Он сохранится в .env рядом с программой.",
            QLineEdit.EchoMode.Password,
            "",
        )
        if not ok or not key.strip():
            return
        try:
            save_api_key(key)
        except OSError as exc:
            QMessageBox.warning(self, "MLVH", f"Не удалось записать .env: {exc}")
            return
        self._refresh_key_state()
        self.statusBar().showMessage("Ключ сохранён в .env", 5000)

    # --- чат ------------------------------------------------------------
    def _assistant_name(self) -> str:
        """Имя собеседника для префикса в чате: активный характер, иначе MLVH."""
        persona = self.settings.persona()
        if persona is not None and persona.name.strip():
            return persona.name.strip()
        return "MLVH"

    def _welcome_text(self) -> str:
        """Приветствие при старте: из характера, если оно задано."""
        persona = self.settings.persona()
        if persona is not None and persona.greeting.strip():
            return persona.greeting.strip()
        return (
            f"Привет! Это {self._assistant_name()}. Спросите что-нибудь — "
            "отвечу текстом и голосом."
        )

    def _insert_line(self, prefix: str, text: str, prefix_color: str, body_color: str) -> None:
        cursor = self.chat.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertBlock()

        prefix_format = QTextCharFormat()
        prefix_format.setFontWeight(QFont.Weight.Bold)
        prefix_format.setForeground(QColor(prefix_color))
        cursor.insertText(prefix, prefix_format)

        body_format = QTextCharFormat()
        body_format.setForeground(QColor(body_color))
        cursor.insertText(text, body_format)

        self.chat.setTextCursor(cursor)
        self._scroll_to_end()

    def _scroll_to_end(self) -> None:
        bar = self.chat.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _begin_assistant(self) -> None:
        """Начинаем новый абзац для ответа: «Имя характера: », дальше — поток текста."""
        self._reply_parts = []
        self._body_fmt = QTextCharFormat()
        self._body_fmt.setForeground(QColor("#e5e7eb"))

        cursor = self.chat.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertBlock()

        prefix_format = QTextCharFormat()
        prefix_format.setFontWeight(QFont.Weight.Bold)
        prefix_format.setForeground(QColor("#a5b4fc"))
        cursor.insertText(f"{self._assistant_name()}: ", prefix_format)

        self.chat.setTextCursor(cursor)
        self._scroll_to_end()

    def _on_chunk(self, text: str) -> None:
        self._reply_parts.append(text)
        if self._pipeline is not None:
            self._pipeline.push(text)
        cursor = self.chat.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text, self._body_fmt)
        self.chat.setTextCursor(cursor)
        self._scroll_to_end()

    def _on_reasoning(self, text: str) -> None:
        """Рассуждения вслух не читаем — только короткая строка статуса."""
        tail = text.strip().replace("\n", " ")[-70:]
        self.statusBar().showMessage(f"Рассуждает: …{tail}")

    def _send(self) -> None:
        if self._is_running(self._chat_worker):
            return
        text = self.input.text().strip()
        if not text:
            return

        key = get_api_key()
        if not key:
            self._prompt_for_key()
            return

        self.input.clear()
        self._insert_line("Вы: ", text, "#93c5fd", "#e5e7eb")
        self.history.append({"role": "user", "content": text})
        self._pending_user_index = len(self.history) - 1

        messages = [{"role": "system", "content": self.settings.system_prompt}]
        messages += self.history[-self.settings.history_turns * 2:]

        model = self.settings.model
        max_tokens = model.max_tokens
        if model.thinking:
            # Рассуждения тоже тратят токены — иначе ответ обрезается или приходит пустым.
            max_tokens = max(max_tokens, 2000)
        try:
            client = DeepSeekClient(
                api_key=key,
                model=model.name,
                temperature=model.temperature,
                max_tokens=max_tokens,
                thinking=model.thinking,
            )
        except DeepSeekError as exc:
            self._drop_pending_question()
            self._show_error(str(exc))
            return

        self._begin_assistant()
        self._start_pipeline()
        worker = ChatWorker(client, messages, self)
        worker.chunk.connect(self._on_chunk)
        worker.reasoning.connect(self._on_reasoning)
        worker.note.connect(lambda text: self.statusBar().showMessage(text))
        worker.finished_ok.connect(self._on_reply_done)
        worker.failed.connect(self._on_reply_failed)
        worker.finished.connect(self._forget_chat_worker)
        worker.finished.connect(worker.deleteLater)
        self._chat_worker = worker

        self._set_busy(True)
        self.statusBar().showMessage("Думаю…")
        worker.start()

    def _prompt_for_key(self) -> None:
        answer = QMessageBox.question(
            self,
            "MLVH",
            "Для ответов нужен ключ DeepSeek — без него модель недоступна.\n"
            "Ввести ключ сейчас?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._ask_api_key()

    def _set_busy(self, busy: bool) -> None:
        self.send_button.setEnabled(not busy)
        self.input.setEnabled(not busy)
        if not busy:
            self.input.setFocus()

    def _on_reply_done(self, text: str, usage: dict) -> None:
        self._set_busy(False)
        reply = text.strip()
        self._last_reply = reply

        if reply:
            self.history.append({"role": "assistant", "content": reply})
            self._pending_user_index = None
        else:
            self._drop_pending_question()
            self._on_chunk("(пустой ответ)")

        if usage:
            self.tokens_label.setText(
                f"токены: {usage.get('total_tokens', 0)} "
                f"(вход {usage.get('prompt_tokens', 0)} / выход {usage.get('completion_tokens', 0)})"
            )
        if usage.get("truncated"):
            self._on_chunk(" …")
            self.statusBar().showMessage(
                "Ответ обрезан лимитом токенов — увеличьте max_tokens в config.json", 8000
            )

        self.statusBar().showMessage("Готов", 3000)
        if reply:
            self._finish_pipeline()
        else:
            self._stop_pipeline()

    def _on_reply_failed(self, message: str) -> None:
        self._set_busy(False)
        self._stop_pipeline()
        self._drop_pending_question()
        self._show_error(message)
        self.statusBar().showMessage("Ошибка запроса", 5000)

    def _drop_pending_question(self) -> None:
        """Убираем из истории вопрос, на который модель так и не ответила.

        Иначе следующий запрос уйдёт с двумя сообщениями user подряд.
        """
        index = self._pending_user_index
        self._pending_user_index = None
        if index is None or not (0 <= index < len(self.history)):
            return
        if self.history[index].get("role") == "user":
            del self.history[index]

    def _show_error(self, message: str) -> None:
        self._insert_line("Ошибка: ", message, "#f87171", "#fca5a5")

    # --- озвучка --------------------------------------------------------
    def _start_pipeline(self) -> None:
        """Готовим потоковую озвучку: она начнёт говорить, не дожидаясь конца ответа."""
        self._stop_pipeline()
        if not self.settings.tts.enabled:
            return

        # Локальная модель грузится долго: греем заранее, пока пользователь печатает.
        if self.settings.tts.engine in ("chatterbox", "silero"):
            self._warm_up_engine(self.settings.tts.engine)

        self._pipeline = SpeechPipeline(
            self.tts,
            self.settings.tts,
            on_done=self._speech_signals.done.emit,
            on_error=self._speech_signals.error.emit,
            on_note=self._speech_signals.note.emit,
        )
        self._pipeline.start()
        self.statusBar().showMessage("Думаю и говорю…")

    def _stop_pipeline(self, wait: bool = False) -> None:
        if self._pipeline is None:
            return
        self._pipeline.stop()
        if wait:
            self._pipeline.wait(2.0)
        self._pipeline = None

    def _finish_pipeline(self) -> None:
        if self._pipeline is not None:
            self._pipeline.finish()

    def _speak(self, text: str) -> None:
        """Озвучка готового текста целиком: повтор, приветствие, проверка голоса.

        Конвейер текущего ответа не трогаем: иначе озвучка начинается заново
        и снова ждёт прогрева модели.
        """
        if self._pipeline is not None and self._pipeline.is_running:
            self._pipeline.stop()

        if self._is_running(self._speak_worker):
            self.tts.stop()
            self._speak_worker.wait(2000)

        worker = SpeakWorker(self.tts, text, self.settings.tts, self)
        worker.started.connect(lambda: self.statusBar().showMessage("Говорю…"))
        worker.finished_ok.connect(lambda: self.statusBar().showMessage("Готов", 3000))
        worker.failed.connect(self._on_speak_failed)
        worker.finished.connect(self._forget_speak_worker)
        worker.finished.connect(worker.deleteLater)
        self._speak_worker = worker
        worker.start()

    def _on_speak_failed(self, message: str) -> None:
        self.statusBar().showMessage(f"Озвучка: {message}", 8000)

    def _on_speak_note(self, message: str) -> None:
        """Предупреждения озвучки — в статус и в чат, чтобы не терялись молча."""
        self.statusBar().showMessage(message, 8000)
        self._insert_line("Озвучка: ", message, "#fbbf24", "#fcd34d")

    def _on_prepare_done(self, message: str) -> None:
        self.statusBar().showMessage(message, 5000)

    def _on_prepare_failed(self, message: str) -> None:
        self.statusBar().showMessage("Модель не загрузилась", 8000)
        QMessageBox.warning(self, "Chatterbox", message)

    def _test_voice(self) -> None:
        self._speak(f"Привет! Это {self._assistant_name()}. Голос настроен и готов к работе.")

    def _replay(self) -> None:
        if not self._last_reply:
            self.statusBar().showMessage("Пока нечего повторять", 3000)
            return
        self._speak(self._last_reply)

    @staticmethod
    def _is_running(worker) -> bool:
        """Проверяем поток, не падая на уже удалённом C++ объекте."""
        if worker is None:
            return False
        try:
            return worker.isRunning()
        except RuntimeError:
            return False

    def _stop_all(self) -> None:
        if self._is_running(self._chat_worker):
            self._chat_worker.cancel()
        self._stop_pipeline()
        self.tts.stop()
        self._set_busy(False)
        self.statusBar().showMessage("Остановлено", 3000)

    def _clear_chat(self) -> None:
        self.history.clear()
        self._pending_user_index = None
        self._last_reply = ""
        self.chat.clear()
        self.chat.setHtml(
            f'<p style="color:#94a3b8;">Новый диалог. {escape(self._assistant_name())} слушает.</p>'
        )
        self.tokens_label.setText("токены: —")
        self.statusBar().showMessage("История очищена", 3000)

    def _forget_chat_worker(self) -> None:
        self._chat_worker = None

    def _forget_speak_worker(self) -> None:
        self._speak_worker = None

    def _forget_voice_worker(self) -> None:
        self._voice_worker = None

    def _stop_thread(self, worker, timeout_ms: int) -> None:
        """Дожидаемся потока, а если он завис — завершаем принудительно.

        Qt нельзя уничтожать вместе с работающим QThread: процесс падает
        с кодом 0xC0000409 уже после закрытия окна.
        """
        if not self._is_running(worker):
            return
        worker.requestInterruption()
        if not worker.wait(timeout_ms):
            worker.terminate()
            worker.wait(1000)

    def closeEvent(self, event) -> None:
        self._stop_all()
        self._stop_pipeline(wait=True)
        self._stop_thread(self._chat_worker, 3000)
        self._persist()
        self.tts.stop()
        self._stop_thread(self._speak_worker, 2000)
        self._stop_thread(self._voice_worker, 3000)
        TTSService.clear_cache()
        super().closeEvent(event)


