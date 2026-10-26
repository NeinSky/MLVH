"""Главное окно MLVH: чат слева, настройки голоса справа."""

from __future__ import annotations

import os
import threading
import time
from html import escape
from pathlib import Path

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QKeySequence, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
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

from app import __version__
from app.agent import (
    AgentLimits,
    Workspace,
    build_agent_prompt,
    build_messages,
    validate_root,
)
from app.asr import audio
from app.asr.global_hotkey import GlobalHotkey
from app.asr.hotkeys import normalize as normalize_hotkey
from app.asr.hotkeys import parse as parse_hotkey
from app.asr.recorder import MicRecorder
from app.asr.registry import asr_engine_infos, get_asr_engine
from app.asr.wake import WakeWorker
from app.asr.wakeword import strip_wake_word
from app.asr.whisper_engine import MODELS as WHISPER_MODELS
from app.config import (
    ASR_MODES,
    ASR_MODE_TITLES,
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
from app.ui.level_meter import LevelMeter
from app.ui.mic_dialog import MicDialog
from app.ui.persona_dialog import PersonaDialog
from app.ui.workers import (
    AgentWorker,
    ChatWorker,
    PrepareWorker,
    SpeakWorker,
    SpeechSignals,
    TranscribeWorker,
    VoiceListWorker,
)

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
QListWidget {
    background: #0b1220; border: 1px solid #1f2937; border-radius: 8px; padding: 4px;
}
QListWidget::item { padding: 3px 4px; }
QListWidget::item:selected { background: #1f2937; color: #f9fafb; }
QSlider::groove:horizontal { height: 4px; background: #1f2937; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 14px; margin: -6px 0; border-radius: 7px; background: #3b82f6;
}
QCheckBox { color: #cbd5e1; }
QStatusBar { color: #9ca3af; }
QPushButton#Mic { min-width: 32px; padding: 5px 6px; font-size: 16px; }
QPushButton#Mic[state="recording"] { background: #b91c1c; border-color: #ef4444; }
QPushButton#Mic[state="busy"] { background: #b45309; border-color: #f59e0b; }
"""


def _format_tokens(value: int) -> str:
    """1_000_000 → «1M», 128_000 → «128K» — для строки статуса."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1000:
        return f"{value // 1000}K"
    return str(value)


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
        self._reply_started = True
        self._voices: list[dict] = []
        self._body_fmt = QTextCharFormat()
        self._chat_worker: ChatWorker | AgentWorker | None = None
        self._speak_worker: SpeakWorker | None = None
        self._voice_worker: VoiceListWorker | None = None
        self._pipeline: SpeechPipeline | None = None

        # --- микрофон и распознавание ---
        self._recording = False
        self._transcribe_worker: TranscribeWorker | None = None
        self._wake: WakeWorker | None = None
        self._asr_infos: dict = {}
        self._hotkey_text = normalize_hotkey(self.settings.asr.hotkey)
        # Запись начата удержанием клавиши: ждём отпускания, а не модификаторов.
        self._hotkey_down = False
        self._record_started_at = 0.0
        self._prepare_worker = None
        self._asr_warm_pending = False
        self._echo_guard_until = 0.0
        self._transcribe_started_at = 0.0
        self._closing = False

        self._recorder = MicRecorder(self)
        self._recorder.level.connect(self._on_mic_level)
        self._recorder.failed.connect(self._on_mic_failed)

        self._global_hotkey = GlobalHotkey(self)
        self._global_hotkey.pressed.connect(self._toggle_recording)

        self._max_record_timer = QTimer(self)
        self._max_record_timer.setSingleShot(True)
        self._max_record_timer.timeout.connect(self._stop_recording)

        self._record_timer = QTimer(self)
        self._record_timer.setInterval(500)
        self._record_timer.timeout.connect(self._tick_recording)

        self._wake_restarts = 0
        self._mic_dead_ticks = 0
        self._wake_pending = None   # поток, который ещё завершается после stop()
        self._mic_watchdog_timer = QTimer(self)
        self._mic_watchdog_timer.setInterval(2000)
        self._mic_watchdog_timer.timeout.connect(self._mic_watchdog)
        self._mic_watchdog_timer.start()

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

        self.setWindowTitle(f"MLVH {__version__} — My Little Voice Helper")
        self.resize(1040, 700)
        self.setStyleSheet(QSS)

        self._build_ui()
        self._populate_engines()
        self._populate_asr_engines()
        self._populate_personas()
        self._apply_settings()
        self._refresh_key_state()
        self._load_voices()

        # Хоткей в окне ловим фильтром событий: так работает и удержание.
        QApplication.instance().installEventFilter(self)
        self._apply_global_hotkey(announce=False)
        QTimer.singleShot(0, self._apply_mic_mode)

        # Модель распознавания грузим в фоне: иначе первый голосовой ввод
        # выглядит зависшим на 10 секунд.
        if self.settings.asr.mode != "off":
            QTimer.singleShot(1200, self._warm_up_asr)

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

        self.mic_button = QPushButton("🎤")
        self.mic_button.setObjectName("Mic")
        self.mic_button.setToolTip("Записать голосом")
        self.mic_button.clicked.connect(self._toggle_recording)

        self.mic_meter = LevelMeter(bars=6)

        self.send_button = QPushButton("Отправить")
        self.send_button.setObjectName("Primary")
        self.send_button.clicked.connect(self._send)

        self.replay_button = QPushButton("Повторить")
        self.replay_button.setToolTip("Озвучить последний ответ ещё раз")
        self.replay_button.clicked.connect(self._replay)

        self.stop_button = QPushButton("Стоп")
        self.stop_button.clicked.connect(self._stop_all)

        row.addWidget(self.mic_button)
        row.addWidget(self.mic_meter)
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

        mic_box = QGroupBox("Микрофон")
        mic_layout = QVBoxLayout(mic_box)

        self.mic_mode_combo = QComboBox()
        for mode in ASR_MODES:
            self.mic_mode_combo.addItem(ASR_MODE_TITLES[mode], mode)
        self.mic_mode_combo.setToolTip(
            "Без микрофона — голосовой ввод выключен.\n"
            "По нажатию — запись только по клавише.\n"
            "Слушать микрофон — ждать спецслово (микрофон открыт постоянно)."
        )
        self.mic_mode_combo.currentIndexChanged.connect(self._on_mic_mode_changed)
        mic_layout.addWidget(self.mic_mode_combo)

        self.asr_engine_combo = QComboBox()
        self.asr_engine_combo.setMinimumWidth(200)
        self.asr_engine_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.asr_engine_combo.setMinimumContentsLength(14)
        self.asr_engine_combo.currentIndexChanged.connect(self._on_asr_engine_changed)
        mic_layout.addWidget(self.asr_engine_combo)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Модель:"))
        self.asr_model_combo = QComboBox()
        self.asr_model_combo.addItems(WHISPER_MODELS)
        self.asr_model_combo.currentTextChanged.connect(self._on_asr_model_changed)
        model_row.addWidget(self.asr_model_combo, 1)
        mic_layout.addLayout(model_row)

        hotkey_row = QHBoxLayout()
        hotkey_row.addWidget(QLabel("Клавиша:"))
        self.hotkey_edit = QKeySequenceEdit()
        self.hotkey_edit.setMaximumSequenceLength(1)
        self.hotkey_edit.setToolTip("Нажмите комбинацию с модификатором, например Ctrl+Space")
        self.hotkey_edit.editingFinished.connect(self._on_hotkey_changed)
        hotkey_row.addWidget(self.hotkey_edit, 1)
        mic_layout.addLayout(hotkey_row)

        self.global_hotkey_check = QCheckBox("Регистрировать нажатия глобально")
        self.global_hotkey_check.setToolTip(
            "Работает, даже когда окно MLVH не в фокусе.\n"
            "Глобально Windows отдаёт только нажатие, поэтому запись "
            "включается и выключается повторным нажатием."
        )
        self.global_hotkey_check.toggled.connect(self._on_global_hotkey)
        mic_layout.addWidget(self.global_hotkey_check)

        self.auto_send_check = QCheckBox("Отправлять сразу после распознавания")
        self.auto_send_check.toggled.connect(self._on_auto_send)
        mic_layout.addWidget(self.auto_send_check)

        wake_row = QHBoxLayout()
        wake_row.addWidget(QLabel("Спецслово:"))
        self.wake_word_edit = QLineEdit()
        self.wake_word_edit.setPlaceholderText("элви")
        self.wake_word_edit.editingFinished.connect(self._on_wake_word)
        wake_row.addWidget(self.wake_word_edit, 1)
        mic_layout.addLayout(wake_row)

        self.mic_test_button = QPushButton("Проверить микрофон…")
        self.mic_test_button.clicked.connect(self._open_mic_test)
        mic_layout.addWidget(self.mic_test_button)

        self.asr_hint = QLabel("")
        self.asr_hint.setWordWrap(True)
        self.asr_hint.setStyleSheet("color:#6b7280;")
        mic_layout.addWidget(self.asr_hint)
        layout.addWidget(mic_box)

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

        agent_box = QGroupBox("Агент")
        agent_layout = QVBoxLayout(agent_box)

        self.agent_read_check = QCheckBox("Чтение файлов")
        self.agent_read_check.setToolTip(
            "Разрешить агенту читать файлы в указанных папках.\n"
            "Записи, удаления и запуска команд нет — только чтение."
        )
        self.agent_read_check.toggled.connect(self._on_agent_read)
        agent_layout.addWidget(self.agent_read_check)

        self.agent_roots = QListWidget()
        self.agent_roots.setMaximumHeight(92)
        self.agent_roots.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.agent_roots.currentRowChanged.connect(lambda _row: self._sync_agent_buttons())
        agent_layout.addWidget(self.agent_roots)

        agent_buttons = QHBoxLayout()
        self.add_agent_root_button = QPushButton("Добавить папку…")
        self.add_agent_root_button.clicked.connect(self._add_agent_root)
        self.remove_agent_root_button = QPushButton("Удалить")
        self.remove_agent_root_button.clicked.connect(self._remove_agent_root)
        agent_buttons.addWidget(self.add_agent_root_button)
        agent_buttons.addWidget(self.remove_agent_root_button)
        agent_layout.addLayout(agent_buttons)

        rounds_row = QHBoxLayout()
        rounds_row.addWidget(QLabel("Шагов:"))
        self.agent_rounds_spin = QSpinBox()
        self.agent_rounds_spin.setRange(1, 2_147_483_647)
        self.agent_rounds_spin.setToolTip(
            "Сколько раз агент может обратиться к файлам за один вопрос.\n"
            "По умолчанию 20; можно задать любое положительное число.\n"
            "Повторные вызовы, 300 с и 800 000 символов ограничены отдельно —\n"
            "при большом числе шагов их можно поднять в config.json."
        )
        self.agent_rounds_spin.valueChanged.connect(self._on_agent_rounds)
        rounds_row.addWidget(self.agent_rounds_spin)
        rounds_row.addStretch()
        agent_layout.addLayout(rounds_row)

        self.agent_hint = QLabel("")
        self.agent_hint.setWordWrap(True)
        self.agent_hint.setStyleSheet("color:#6b7280;")
        agent_layout.addWidget(self.agent_hint)
        layout.addWidget(agent_box)

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

        for widget in (
            self.model_combo,
            self.speak_check,
            self.thinking_check,
            self.persona_combo,
            self.agent_read_check,
            self.agent_rounds_spin,
        ):
            widget.blockSignals(True)
        self.model_combo.setCurrentText(
            model.name if model.name in AVAILABLE_MODELS else AVAILABLE_MODELS[0]
        )
        persona_index = self.persona_combo.findData(self.settings.active_persona)
        if persona_index >= 0:
            self.persona_combo.setCurrentIndex(persona_index)
        self.speak_check.setChecked(bool(tts.enabled))
        self.thinking_check.setChecked(bool(model.thinking))
        self.agent_read_check.setChecked(bool(self.settings.agent.read_enabled))
        self.agent_rounds_spin.setValue(int(self.settings.agent.max_rounds))
        for widget in (
            self.model_combo,
            self.speak_check,
            self.thinking_check,
            self.persona_combo,
            self.agent_read_check,
            self.agent_rounds_spin,
        ):
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

        asr = self.settings.asr
        mic_widgets = (
            self.mic_mode_combo,
            self.asr_engine_combo,
            self.asr_model_combo,
            self.hotkey_edit,
            self.global_hotkey_check,
            self.auto_send_check,
            self.wake_word_edit,
        )
        for widget in mic_widgets:
            widget.blockSignals(True)

        mode_index = self.mic_mode_combo.findData(asr.mode)
        self.mic_mode_combo.setCurrentIndex(mode_index if mode_index >= 0 else 0)
        asr_index = self.asr_engine_combo.findData(asr.engine)
        if asr_index >= 0:
            self.asr_engine_combo.setCurrentIndex(asr_index)
        if asr.model in WHISPER_MODELS:
            self.asr_model_combo.setCurrentText(asr.model)
        self.hotkey_edit.setKeySequence(QKeySequence(asr.hotkey))
        self.global_hotkey_check.setChecked(bool(asr.global_hotkey))
        self.auto_send_check.setChecked(bool(asr.auto_send))
        self.wake_word_edit.setText(asr.wake_word)

        for widget in mic_widgets:
            widget.blockSignals(False)
        self._refresh_mic_ui()

        self._populate_agent_roots()
        self._sync_agent_buttons()
        self._refresh_agent_ui()

    def _touch(self) -> None:
        """Настройки меняются часто — пишем файл с небольшой задержкой."""
        self._save_timer.start()

    def _persist(self) -> None:
        try:
            save_settings(self.settings)
        except OSError as exc:
            self.statusBar().showMessage(f"Не удалось сохранить настройки: {exc}", 5000)

    # --- агент ----------------------------------------------------------
    def _populate_agent_roots(self) -> None:
        self.agent_roots.clear()
        for root in self.settings.agent.roots:
            item = QListWidgetItem(root)
            item.setToolTip(root)
            self.agent_roots.addItem(item)

    def _sync_agent_buttons(self) -> None:
        self.remove_agent_root_button.setEnabled(self.agent_roots.currentRow() >= 0)

    def _refresh_agent_ui(self) -> None:
        agent = self.settings.agent
        if not agent.read_enabled:
            self.agent_hint.setText("Агент выключен: включите «Чтение файлов».")
            return
        if not agent.roots:
            self.agent_hint.setText("Чтение включено, но нет ни одной рабочей папки.")
            return

        available = 0
        missing: list[str] = []
        for raw in agent.roots:
            resolved, error = validate_root(raw)
            if resolved is None:
                missing.append(f"{raw} ({error})")
            else:
                available += 1

        if available:
            text = f"Чтение: папок — {available}. Агент видит только их."
            if missing:
                text += "\nНедоступно: " + "; ".join(missing)
            self.agent_hint.setText(text)
        else:
            self.agent_hint.setText("Рабочие папки недоступны: " + "; ".join(missing))

    def _on_agent_read(self, enabled: bool) -> None:
        self.settings.agent.read_enabled = bool(enabled)
        self._touch()
        self._refresh_agent_ui()

    def _on_agent_rounds(self, value: int) -> None:
        self.settings.agent.max_rounds = int(value)
        self._touch()

    def _add_agent_root(self) -> None:
        roots = self.settings.agent.roots
        start = roots[-1] if roots else str(Path.home())
        folder = QFileDialog.getExistingDirectory(
            self, "Рабочая папка агента (только чтение)", start
        )
        if not folder:
            return

        resolved, error = validate_root(folder)
        if resolved is None:
            QMessageBox.warning(self, "Агент", f"Эту папку нельзя добавить: {error}")
            return

        key = os.path.normcase(str(resolved))
        existing = {os.path.normcase(str(Path(raw))) for raw in roots}
        if key in existing:
            self.statusBar().showMessage("Эта папка уже добавлена", 3000)
            return

        roots.append(str(resolved))
        self._populate_agent_roots()
        self.agent_roots.setCurrentRow(self.agent_roots.count() - 1)
        self._sync_agent_buttons()
        self._refresh_agent_ui()
        self._touch()
        self.statusBar().showMessage(f"Папка добавлена: {resolved}", 4000)

    def _remove_agent_root(self) -> None:
        row = self.agent_roots.currentRow()
        if row < 0 or row >= len(self.settings.agent.roots):
            return

        removed = self.settings.agent.roots.pop(row)
        self._populate_agent_roots()
        if self.settings.agent.roots:
            self.agent_roots.setCurrentRow(
                min(row, self.agent_roots.count() - 1)
            )
        self._sync_agent_buttons()
        self._refresh_agent_ui()
        self._touch()
        self.statusBar().showMessage(f"Папка убрана: {removed}", 3000)

    def _agent_workspace(self) -> Workspace | None:
        """Рабочие папки текущего запроса или None, если агент выключен."""
        agent = self.settings.agent
        if not agent.read_enabled or not agent.roots:
            return None

        workspace = Workspace(agent.roots, AgentLimits.from_settings(agent))
        if not workspace:
            self.statusBar().showMessage("Агент: нет доступных рабочих папок", 5000)
            return None
        if workspace.skipped:
            self.statusBar().showMessage(
                "Агент: пропущены недоступные папки — "
                + "; ".join(f"{path} ({why})" for path, why in workspace.skipped),
                6000,
            )
        return workspace

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

    # --- микрофон --------------------------------------------------------
    def _populate_asr_engines(self) -> None:
        """Список движков распознавания: недоступные видны, но не выбираются."""
        self._asr_infos = {info.id: info for info in asr_engine_infos()}
        self.asr_engine_combo.blockSignals(True)
        self.asr_engine_combo.clear()
        for info in self._asr_infos.values():
            self.asr_engine_combo.addItem(info.label(), info.id)
        index = self.asr_engine_combo.findData(self.settings.asr.engine)
        self.asr_engine_combo.setCurrentIndex(index if index >= 0 else 0)
        self.asr_engine_combo.blockSignals(False)

    def _refresh_mic_ui(self) -> None:
        """Подсказка, доступность кнопок и индикатора под текущий режим."""
        asr = self.settings.asr
        info = self._asr_infos.get(asr.engine)
        available = bool(info and info.available)
        enabled = asr.mode != "off"

        self.asr_engine_combo.setEnabled(enabled)
        self.asr_model_combo.setEnabled(enabled and asr.engine == "whisper")
        self.hotkey_edit.setEnabled(enabled)
        self.global_hotkey_check.setEnabled(enabled)
        self.auto_send_check.setEnabled(enabled)
        self.wake_word_edit.setEnabled(enabled and asr.mode == "wake")
        self.mic_test_button.setEnabled(enabled)
        self.mic_button.setEnabled(enabled)

        if not enabled:
            self.mic_button.setToolTip("Микрофон выключен: включите режим в панели «Микрофон»")
        elif not available:
            self.mic_button.setToolTip("Движок распознавания не установлен: .\\install_asr.ps1")
        elif asr.mode == "wake":
            self.mic_button.setToolTip(f"Слушаю спецслово «{asr.wake_word}»")
        else:
            self.mic_button.setToolTip(f"Записать голосом ({asr.hotkey})")

        if info is None:
            self.asr_hint.setText("")
        elif not info.available:
            self.asr_hint.setText(f"{info.title}: {info.reason}. {info.hint}")
        elif asr.mode == "wake":
            self.asr_hint.setText(
                f"Слушаю «{asr.wake_word}». Говорите вместе с вопросом: "
                f"«{asr.wake_word}, сколько времени?». Если имя не ловится — "
                "попробуйте слово попроще: «помощник»."
            )
        else:
            self.asr_hint.setText(info.hint.replace("\n", " "))

        if not self._recording and not self._is_running(self._transcribe_worker):
            if enabled and asr.mode == "wake" and self._wake is not None:
                self.mic_meter.set_state("listening")
            else:
                self.mic_meter.set_state("ready" if enabled else "idle")

    def _on_mic_mode_changed(self, index: int) -> None:
        if index < 0:
            return
        mode = str(self.mic_mode_combo.itemData(index))

        if mode == "wake":
            vosk = self._asr_infos.get("vosk")
            if vosk is None or not vosk.available:
                QMessageBox.information(
                    self,
                    "MLVH",
                    "Режим «Слушать микрофон» работает на лёгком движке Vosk.\n\n"
                    "Установите его: .\\install_asr.ps1, затем скачайте модель "
                    "(см. README). Пока включён режим «По нажатию».",
                )
                self.mic_mode_combo.blockSignals(True)
                back = self.mic_mode_combo.findData(self.settings.asr.mode)
                self.mic_mode_combo.setCurrentIndex(back if back >= 0 else 0)
                self.mic_mode_combo.blockSignals(False)
                return

        self.settings.asr.mode = mode
        self._wake_restarts = 0
        self._touch()
        self._apply_global_hotkey()
        self._refresh_mic_ui()
        self._apply_mic_mode()

    def _on_asr_engine_changed(self, index: int) -> None:
        if index < 0:
            return
        engine_id = str(self.asr_engine_combo.itemData(index))
        info = self._asr_infos.get(engine_id)

        if info is not None and not info.available:
            QMessageBox.information(
                self,
                "Движок не установлен",
                f"{info.title} не установлен.\n\n{info.hint}",
            )
            self.asr_engine_combo.blockSignals(True)
            back = self.asr_engine_combo.findData(self.settings.asr.engine)
            self.asr_engine_combo.setCurrentIndex(back if back >= 0 else 0)
            self.asr_engine_combo.blockSignals(False)
            return

        self.settings.asr.engine = engine_id
        self._touch()
        self._refresh_mic_ui()
        self._warm_up_asr()

    def _warm_up_asr(self) -> None:
        """Готовим модель заранее: первый «Распознать» не должен ждать загрузки."""
        info = self._asr_infos.get(self.settings.asr.engine)
        if info is None or not info.available:
            return
        if self._prepare_worker is not None and self._prepare_worker.isRunning():
            # Уже грузим другую модель — повторим после завершения.
            self._asr_warm_pending = True
            return

        self._asr_warm_pending = False
        engine = get_asr_engine(self.settings.asr.engine)
        worker = PrepareWorker(engine, self.settings.asr, self)
        worker.ready.connect(
            lambda device: self.statusBar().showMessage(
                f"Распознавание готово ({device})", 4000
            )
        )
        worker.failed.connect(
            lambda message: self.statusBar().showMessage(f"Распознавание: {message}", 8000)
        )
        worker.finished.connect(self._forget_prepare_worker)
        worker.finished.connect(worker.deleteLater)
        self._prepare_worker = worker
        worker.start()

    def _forget_prepare_worker(self) -> None:
        self._prepare_worker = None
        if self._asr_warm_pending:
            self._asr_warm_pending = False
            QTimer.singleShot(0, self._warm_up_asr)

    def _on_asr_model_changed(self, model: str) -> None:
        if not model:
            return
        changed = model != self.settings.asr.model
        self.settings.asr.model = model
        self._touch()
        # Новая модель грузится в фоне, иначе первый ввод ждёт ~10 секунд.
        if changed:
            self._warm_up_asr()

    def _on_auto_send(self, checked: bool) -> None:
        self.settings.asr.auto_send = bool(checked)
        self._touch()

    def _on_wake_word(self) -> None:
        word = self.wake_word_edit.text().strip().lower() or "элви"
        changed = word != self.settings.asr.wake_word
        self.wake_word_edit.setText(word)
        self.settings.asr.wake_word = word
        self._touch()
        self._refresh_mic_ui()

        # Слушатель запоминает слово при старте — перезапускаем его.
        if changed and self.settings.asr.mode == "wake" and self._wake is not None:
            self._stop_wake_listening()
            self._start_wake_listening()

    def _on_hotkey_changed(self) -> None:
        text = self.hotkey_edit.keySequence().toString(
            QKeySequence.SequenceFormat.PortableText
        )
        parsed = parse_hotkey(text)
        if parsed is None:
            QMessageBox.information(
                self,
                "MLVH",
                "Нужна комбинация с модификатором, например Ctrl+Space.",
            )
            self.hotkey_edit.setKeySequence(QKeySequence(self.settings.asr.hotkey))
            return

        self.settings.asr.hotkey = parsed.text()
        self._hotkey_text = normalize_hotkey(self.settings.asr.hotkey)
        self._touch()
        self._apply_global_hotkey()
        self._refresh_mic_ui()

    def _on_global_hotkey(self, checked: bool) -> None:
        self.settings.asr.global_hotkey = bool(checked)
        self._touch()
        self._apply_global_hotkey()

    def _apply_global_hotkey(self, announce: bool = True) -> None:
        """Регистрируем клавишу глобально или честно откатываемся в окно."""
        self._global_hotkey.unregister()
        if not self.settings.asr.global_hotkey or self.settings.asr.mode == "off":
            return

        ok, reason = self._global_hotkey.register(parse_hotkey(self.settings.asr.hotkey))
        if ok:
            if announce:
                self.statusBar().showMessage(
                    f"Глобальная клавиша: {self.settings.asr.hotkey}", 4000
                )
            return

        self.settings.asr.global_hotkey = False
        self._touch()
        self.global_hotkey_check.blockSignals(True)
        self.global_hotkey_check.setChecked(False)
        self.global_hotkey_check.blockSignals(False)
        if announce:
            QMessageBox.warning(
                self,
                "MLVH",
                f"Не удалось зарегистрировать клавишу глобально: {reason}.\n\n"
                "Она работает, пока окно MLVH активно.",
            )
        else:
            self.statusBar().showMessage(f"Глобальная клавиша недоступна: {reason}", 6000)

    def _open_mic_test(self) -> None:
        dialog = MicDialog(self.settings, self.tts, self)
        dialog.exec()
        self._touch()
        self._refresh_mic_ui()

    # --- режим «Слушать микрофон» ---------------------------------------
    def _apply_mic_mode(self) -> None:
        """Приводим микрофон к текущему режиму: слушаем спецслово или нет."""
        if self.settings.asr.mode == "wake":
            alive = (
                self._wake is not None
                and self._wake.isRunning()
                and self._recorder.is_streaming
            )
            if alive:
                self._set_mic_state("listening")
                return
            self._start_wake_listening()
        else:
            self._stop_wake_listening()

    def _start_wake_listening(self) -> None:
        if self._wake is not None:
            return

        info = self._asr_infos.get("vosk")
        if info is None or not info.available:
            self.statusBar().showMessage(
                "Для режима «Слушать микрофон» нужен Vosk: .\\install_asr.ps1", 8000
            )
            return
        if not self._recorder.start(self.settings.asr.input_device, stream=True):
            # Устройство могло быть занято: сторожевой таймер попробует снова.
            return

        worker = WakeWorker(self.settings.asr, self)
        worker.ready.connect(self._on_wake_ready)
        worker.heard.connect(self._on_wake_heard)
        worker.timeout.connect(self._on_wake_timeout)
        worker.detected.connect(self._on_wake_detected)
        worker.failed.connect(self._on_wake_failed)
        self._wake = worker
        self._recorder.chunk.connect(self._feed_wake)
        worker.start()
        self._set_mic_state("listening")

    def _on_wake_ready(self) -> None:
        self._wake_restarts = 0
        self.statusBar().showMessage(
            f"Слушаю спецслово «{self.settings.asr.wake_word}»", 4000
        )

    def _on_wake_timeout(self) -> None:
        """Сказали только спецслово и замолчали — возвращаемся к ожиданию."""
        if self.settings.asr.mode != "wake":
            return
        self._set_mic_state("listening")
        self.statusBar().showMessage(
            f"Не услышал вопрос. Скажите: «{self.settings.asr.wake_word}, ваш вопрос»", 5000
        )

    def _mic_watchdog(self) -> None:
        """Режим включён, но микрофон или слушатель умер — поднимаем заново.

        Без этого одно неудачное открытие устройства (занято другой программой,
        переключение режима, усыпление потока после озвучки) оставляло MLVH
        «глухим» до перезапуска.
        """
        if self.settings.asr.mode != "wake":
            return
        if self._is_running(self._transcribe_worker):
            return
        if self._wake_pending is not None:
            return  # предыдущий слушатель ещё завершается

        alive = self._wake is not None and self._wake.isRunning()
        if alive and self._recorder.is_streaming:
            # Поток захвата мог уснуть: считаем пустые циклы и перезапускаем.
            if self._recorder.is_active:
                self._mic_dead_ticks = 0
                return
            self._mic_dead_ticks += 1
            if self._mic_dead_ticks < 3:
                return
            self.statusBar().showMessage("Микрофон уснул — перезапускаю…", 4000)
        self._mic_dead_ticks = 0

        if self._wake_restarts > 20:
            self.statusBar().showMessage(
                "Микрофон не открывается. Проверьте устройство и режим «По нажатию».", 8000
            )
            return

        self._wake_restarts += 1
        self._stop_wake_listening()
        self._start_wake_listening()

    def _stop_wake_listening(self, wait_ms: int = 1500) -> None:
        worker = self._wake
        if worker is not None:
            try:
                self._recorder.chunk.disconnect(self._feed_wake)
            except (RuntimeError, TypeError):
                pass
            worker.stop()
            if worker.wait(wait_ms):
                self._wake = None
            else:
                # Не убиваем поток: он может держать блокировку загрузки модели,
                # и тогда все следующие загрузки повиснут навсегда. Отпускаем
                # его доработать и добираем позже.
                self._wake = None
                self._wake_pending = worker
                QTimer.singleShot(500, self._reap_wake_worker)

        if not self._recording:
            self._recorder.stop()
            self._set_mic_state("idle")

    def _reap_wake_worker(self) -> None:
        worker = self._wake_pending
        if worker is None:
            return
        if worker.wait(0):
            self._wake_pending = None
        else:
            QTimer.singleShot(500, self._reap_wake_worker)

    def _feed_wake(self, samples) -> None:
        """Куски аудио уходят в Vosk, но не пока MLVH говорит или распознаёт."""
        if self._wake is None:
            return
        if self.settings.asr.mute_while_speaking and self._is_speaking():
            return
        if self._is_running(self._transcribe_worker):
            # Распознавание не должно глушить микрофон навсегда: если движок
            # завис, через 45 секунд снова слушаем.
            if time.monotonic() - self._transcribe_started_at < 45:
                return
        self._wake.feed(samples)

    def _on_wake_heard(self) -> None:
        QApplication.beep()
        self.mic_meter.set_state("recording")
        self.statusBar().showMessage("Слушаю вопрос…")
    def _on_wake_detected(self, samples, rest: str) -> None:
        self.mic_meter.set_state("busy")
        self.statusBar().showMessage(f"Услышал: {rest}")

        data = audio.to_mono_16k(samples, audio.TARGET_RATE, 1)
        if data.size == 0:
            self.mic_meter.set_state("listening")
            return

        whisper = self._asr_infos.get("whisper")
        if whisper is not None and whisper.available:
            self._transcribe_samples(data)
        else:
            # Без whisper берём текст Vosk: грубее, но без лишней модели.
            self._on_transcribed(rest)

    def _on_wake_failed(self, message: str) -> None:
        self.statusBar().showMessage(message, 8000)
        self._stop_wake_listening()

        # Первые сбои лечим перезапуском (сторожевой таймер), а если Vosk
        # стабильно не поднимается — возвращаемся в «По нажатию».
        if self._wake_restarts >= 3:
            self._wake_restarts = 0
            self.settings.asr.mode = "push"
            self._touch()
            self._apply_settings()
            QMessageBox.warning(
                self,
                "MLVH",
                f"Режим «Слушать микрофон» не запустился: {message}\n\n"
                "Включён режим «По нажатию». Проверьте Vosk: .\\install_asr.ps1",
            )
        else:
            self._wake_restarts += 1

    def _idle_mic_state(self) -> None:
        """После распознавания возвращаемся в «слушаю» или в покой."""
        if self.settings.asr.mode == "wake" and self._wake is not None:
            self._set_mic_state("listening")
        else:
            self._set_mic_state("idle")

    # --- запись и распознавание -----------------------------------------
    def _set_mic_state(self, state: str) -> None:
        self.mic_meter.set_state(state)
        self.mic_button.setProperty("state", state)
        style = self.mic_button.style()
        style.unpolish(self.mic_button)
        style.polish(self.mic_button)

    def _on_mic_level(self, value: float) -> None:
        if self._recording or self.settings.asr.mode == "wake":
            self.mic_meter.set_level(value)

    def _on_mic_failed(self, message: str) -> None:
        self._recording = False
        self._max_record_timer.stop()
        self._set_mic_state("idle")
        self.statusBar().showMessage(message, 8000)

    def _is_speaking(self) -> bool:
        """Идёт ли звук из колонок (плюс полсекунды запаса на эхо).

        Раньше здесь проверялось «поток озвучки жив». Из-за этого микрофон
        глушился, пока существовал конвейер, — в том числе после «Стоп» и
        после того, как ответ давно дочитан.
        """
        if self.tts.is_playing:
            self._echo_guard_until = time.monotonic() + 0.5
            return True
        return time.monotonic() < self._echo_guard_until

    def _mic_problem(self) -> str:
        asr = self.settings.asr
        if asr.mode == "off":
            return "Микрофон выключен: включите режим в панели «Микрофон»"
        info = self._asr_infos.get(asr.engine)
        if info is None or not info.available:
            return "Движок распознавания не установлен: .\\install_asr.ps1"
        return ""

    def _toggle_recording(self) -> None:
        """Клик по кнопке и глобальная клавиша: нажал — пишу, нажал — стоп."""
        if self.settings.asr.mode == "wake":
            word = self.settings.asr.wake_word
            self.statusBar().showMessage(f"Скажите: «{word}, ваш вопрос»", 4000)
            return
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        if self._recording:
            return

        if self.settings.asr.mode == "wake":
            word = self.settings.asr.wake_word
            self.statusBar().showMessage(f"Скажите: «{word}, ваш вопрос»", 4000)
            return

        problem = self._mic_problem()
        if problem:
            self.statusBar().showMessage(problem, 6000)
            return

        if self.settings.asr.mute_while_speaking and self._is_speaking():
            self.statusBar().showMessage(
                "MLVH говорит — микрофон не слушаю. Нажмите ещё раз после ответа.", 5000
            )
            return

        if not self._recorder.start(self.settings.asr.input_device):
            return

        self._recorder.begin_capture()
        self._recording = True
        self._record_started_at = time.monotonic()
        self._max_record_timer.start(max(3, int(self.settings.asr.max_seconds)) * 1000)
        self._record_timer.start()
        self._set_mic_state("recording")
        self.statusBar().showMessage(self._recording_hint())

    def _recording_hint(self) -> str:
        if self.settings.asr.global_hotkey:
            return (
                f"Запись идёт. Нажмите {self.settings.asr.hotkey} ещё раз, "
                "чтобы остановить (Esc — отмена)"
            )
        return f"Запись идёт. Отпустите {self.settings.asr.hotkey} — распознаю (Esc — отмена)"

    def _tick_recording(self) -> None:
        if not self._recording:
            return
        elapsed = time.monotonic() - self._record_started_at
        limit = max(3, int(self.settings.asr.max_seconds))
        self.statusBar().showMessage(f"{self._recording_hint()} · {elapsed:.0f} из {limit} с")

    def _stop_recording(self) -> None:
        if not self._recording:
            return

        self._recording = False
        self._max_record_timer.stop()
        self._record_timer.stop()
        samples = self._recorder.end_capture()
        rate = self._recorder.rate
        self._recorder.stop()
        self._set_mic_state("idle")

        if samples.size == 0:
            self.statusBar().showMessage("Ничего не записалось — проверьте микрофон", 5000)
            return

        seconds = samples.size / float(rate or audio.TARGET_RATE)
        data = audio.trim_silence(audio.to_mono_16k(samples, rate, 1))
        if seconds < 0.3 or data.size == 0 or audio.rms(data) < 0.004:
            self.statusBar().showMessage("Не расслышал — попробуйте ещё раз", 4000)
            return

        self._transcribe_samples(data)

    def _cancel_recording(self) -> None:
        if not self._recording:
            return
        self._recording = False
        self._hotkey_down = False
        self._max_record_timer.stop()
        self._record_timer.stop()
        self._recorder.end_capture()
        self._recorder.stop()
        self._set_mic_state("idle")
        self.statusBar().showMessage("Запись отменена", 3000)

    def _transcribe_samples(self, samples) -> None:
        if self._is_running(self._transcribe_worker):
            return

        engine = get_asr_engine(self.settings.asr.engine)
        worker = TranscribeWorker(engine, samples, self.settings.asr, self)
        worker.note.connect(lambda text: self.statusBar().showMessage(text))
        worker.done.connect(self._on_transcribed)
        worker.failed.connect(self._on_transcribe_failed)
        worker.finished.connect(self._forget_transcribe_worker)
        worker.finished.connect(worker.deleteLater)
        self._transcribe_worker = worker

        self._transcribe_started_at = time.monotonic()
        self._set_mic_state("busy")
        self.statusBar().showMessage("Распознаю…")
        worker.start()

    def _on_transcribed(self, text: str) -> None:
        self._idle_mic_state()
        text = text.strip()

        # В режиме спецслова имя могло попасть в распознанный текст.
        if self.settings.asr.mode == "wake":
            stripped = strip_wake_word(text, self.settings.asr.wake_word)
            if stripped is not None:
                text = stripped.strip()

        if not text:
            self.statusBar().showMessage("Не расслышал — попробуйте ещё раз", 4000)
            return

        if self.input.text() and not self.input.text().endswith(" "):
            self.input.insert(" ")
        self.input.insert(text)
        self.input.setFocus()

        if self.settings.asr.auto_send:
            self._send()
        else:
            self.statusBar().showMessage("Распознано — проверьте и отправьте", 5000)

    def _on_transcribe_failed(self, message: str) -> None:
        self._idle_mic_state()
        self.statusBar().showMessage(f"Распознавание: {message}", 8000)

    def _forget_transcribe_worker(self) -> None:
        self._transcribe_worker = None

    def _matches_hotkey(self, event) -> bool:
        """Совпадает ли событие с комбинацией (модификаторы учитываются)."""
        if not self._hotkey_text:
            return False
        text = QKeySequence(event.keyCombination()).toString(
            QKeySequence.SequenceFormat.PortableText
        )
        return normalize_hotkey(text) == self._hotkey_text

    def _hotkey_key(self) -> Qt.Key:
        """Клавиша комбинации без модификаторов — для проверки отпускания."""
        sequence = QKeySequence(self.settings.asr.hotkey)
        if sequence.count() == 0:
            return Qt.Key.Key_unknown
        return sequence[0].key()

    def _is_hotkey_key(self, event) -> bool:
        """Отпускание самой клавиши.

        Windows и Qt присылают KeyRelease уже без модификаторов, если их
        отпустили первыми, поэтому здесь модификаторы не проверяем.
        """
        return self._hotkey_key() != Qt.Key.Key_unknown and event.key() == self._hotkey_key()

    def eventFilter(self, obj, event) -> bool:
        """Хоткей в окне: нажали — пишем, отпустили — распознаём."""
        if not self.settings.asr.global_hotkey and not self.hotkey_edit.hasFocus():
            if event.type() == QEvent.Type.KeyPress:
                if event.key() == Qt.Key.Key_Escape and self._recording:
                    self._cancel_recording()
                    return True
                if self._matches_hotkey(event):
                    if not event.isAutoRepeat() and not self._recording:
                        self._hotkey_down = True
                        self._start_recording()
                    return True
            elif event.type() == QEvent.Type.KeyRelease:
                if self._hotkey_down and self._is_hotkey_key(event):
                    self._hotkey_down = False
                    self._stop_recording()
                    return True
            elif event.type() in (
                QEvent.Type.WindowDeactivate,
                QEvent.Type.ApplicationDeactivate,
            ):
                # Отпускание могло уйти другому окну — не оставляем запись висеть.
                if self._hotkey_down:
                    self._hotkey_down = False
                    self._stop_recording()

        return super().eventFilter(obj, event)

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
        self._reply_started = True

    def _ensure_reply_started(self) -> None:
        """Ответ агента начинается после строк инструментов — префикс выводим лениво."""
        if not self._reply_started:
            self._begin_assistant()

    def _on_agent_tool(self, summary: str) -> None:
        self._insert_line("Агент: ", summary, "#34d399", "#a7f3d0")
        self.statusBar().showMessage(f"Агент: {summary}")

    def _on_chunk(self, text: str) -> None:
        self._ensure_reply_started()
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

        model = self.settings.model
        max_tokens = model.max_tokens
        if model.thinking:
            # Рассуждения тоже тратят токены — иначе ответ обрезается или приходит пустым.
            max_tokens = max(max_tokens, 2000)

        workspace = self._agent_workspace()
        limits = AgentLimits.from_settings(self.settings.agent) if workspace else None
        system_prompt = self.settings.system_prompt
        if workspace is not None and limits is not None:
            system_prompt = build_agent_prompt(system_prompt, workspace, limits)
        trimmed = build_messages(
            system_prompt,
            self.history,
            model.context_tokens,
            max_tokens,
        )
        if trimmed.dropped:
            self.statusBar().showMessage(
                f"Контекст: {trimmed.dropped} старых реплик не поместились "
                "и пропущены",
                5000,
            )

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

        self._start_pipeline()
        if workspace is not None and limits is not None:
            # Строки инструментов идут до ответа, поэтому «Имя: » выводим лениво.
            self._reply_started = False
            worker = AgentWorker(
                client,
                trimmed.messages,
                workspace,
                limits,
                context_tokens=model.context_tokens,
                max_tokens=max_tokens,
                parent=self,
            )
            worker.tool.connect(self._on_agent_tool)
        else:
            self._begin_assistant()
            worker = ChatWorker(client, trimmed.messages, self)
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

        if reply and not self._reply_started:
            # Текст не пришёл потоком (редкий случай) — показываем целиком.
            self._on_chunk(reply)

        if reply:
            self.history.append({"role": "assistant", "content": reply})
            self._pending_user_index = None
        else:
            self._drop_pending_question()
            self._on_chunk("(пустой ответ)")

        warning = ""
        if usage:
            extra = ""
            rounds = int(usage.get("rounds", 0) or 0)
            tool_calls = int(usage.get("tool_calls", 0) or 0)
            if rounds:
                extra += f" · шагов {rounds}"
            if tool_calls:
                extra += f", чтений {tool_calls}"
            self.tokens_label.setText(
                f"токены: {usage.get('total_tokens', 0)} "
                f"(вход {usage.get('prompt_tokens', 0)} / "
                f"выход {usage.get('completion_tokens', 0)})"
                f"{extra} · окно {_format_tokens(self.settings.model.context_tokens)}"
            )
            prompt_tokens = int(
                usage.get("max_prompt_tokens", usage.get("prompt_tokens", 0)) or 0
            )
            if prompt_tokens >= self.settings.model.context_tokens * 0.9:
                warning = "Контекст почти заполнен — начните новый диалог"
        if usage.get("truncated"):
            self._on_chunk(" …")
            warning = "Ответ обрезан лимитом токенов — увеличьте max_tokens в config.json"

        if warning:
            self.statusBar().showMessage(warning, 8000)
        else:
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
        self._cancel_recording()
        if self._is_running(self._chat_worker):
            self._chat_worker.cancel()
        self._stop_pipeline()
        self.tts.stop()
        self._echo_guard_until = 0.0
        # «Стоп» должен и озвучку остановить, и сразу вернуть микрофон к работе.
        if self._is_running(self._speak_worker):
            self._speak_worker.wait(1500)
        self._set_busy(False)
        self._set_mic_state("listening" if self.settings.asr.mode == "wake" else "idle")
        if not self._closing:
            self._apply_mic_mode()
        self.statusBar().showMessage("Остановлено", 3000)

    def _clear_chat(self) -> None:
        self.history.clear()
        self._pending_user_index = None
        self._last_reply = ""
        self._reply_started = True
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
        self._closing = True
        self._stop_all()
        self._stop_wake_listening(wait_ms=4000)
        pending = self._wake_pending
        if pending is not None:
            pending.wait(4000)
            self._wake_pending = None
        self._recorder.stop()
        self._global_hotkey.unregister()
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self._stop_pipeline(wait=True)
        self._stop_thread(self._chat_worker, 3000)
        self._stop_thread(self._transcribe_worker, 3000)
        self._stop_thread(self._prepare_worker, 3000)
        self._persist()
        self.tts.stop()
        self._stop_thread(self._speak_worker, 2000)
        self._stop_thread(self._voice_worker, 3000)
        TTSService.clear_cache()
        super().closeEvent(event)


