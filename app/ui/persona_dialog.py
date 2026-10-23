"""Диалог редактирования характера: имя, промпт и голос."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from app.personas import Persona
from app.tts import voice_label


class PersonaDialog(QDialog):
    """Правка одного характера. Возвращает новый Persona через result_persona()."""

    def __init__(
        self,
        persona: Persona,
        voices: list[dict],
        parent=None,
        create: bool = False,
    ) -> None:
        super().__init__(parent)
        self._persona = persona
        self._create = create
        self._result: Persona | None = None

        self.setWindowTitle("Новый характер" if create else f"Характер: {persona.name}")
        self.setMinimumWidth(560)
        self.setStyleSheet(parent.styleSheet() if parent else "")

        root = QVBoxLayout(self)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.name_edit = QLineEdit("" if create else persona.name)
        self.name_edit.setPlaceholderText("Например: Цундэрэ")
        form.addRow("Имя:", self.name_edit)

        self.emoji_edit = QLineEdit(persona.emoji)
        self.emoji_edit.setMaximumWidth(80)
        form.addRow("Значок:", self.emoji_edit)

        self.greeting_edit = QLineEdit(persona.greeting)
        form.addRow("Приветствие:", self.greeting_edit)

        self.prompt_edit = QPlainTextEdit(persona.prompt)
        self.prompt_edit.setMinimumHeight(150)
        form.addRow("Характер (промпт):", self.prompt_edit)

        self.voice_combo = QComboBox()
        self.voice_combo.setMinimumWidth(260)
        for voice in voices:
            self.voice_combo.addItem(voice_label(voice), voice.get("ShortName"))
        index = self.voice_combo.findData(persona.voice)
        if index < 0:
            self.voice_combo.addItem(persona.voice, persona.voice)
            index = self.voice_combo.count() - 1
        self.voice_combo.setCurrentIndex(index)
        form.addRow("Голос:", self.voice_combo)

        self.rate_slider, self.rate_label = self._slider_row(-50, 50, persona.rate, " %")
        form.addRow("Темп:", self._wrap(self.rate_slider, self.rate_label))

        self.pitch_slider, self.pitch_label = self._slider_row(-50, 50, persona.pitch, " Гц")
        form.addRow("Тон:", self._wrap(self.pitch_slider, self.pitch_label))

        self.chars_spin = QSpinBox()
        self.chars_spin.setRange(200, 10000)
        self.chars_spin.setSingleStep(100)
        self.chars_spin.setSuffix(" симв.")
        self.chars_spin.setValue(persona.max_speak_chars)
        form.addRow("Читать не длиннее:", self.chars_spin)

        root.addLayout(form)

        hint = QLabel(
            "Тон +30…+50 Гц делает голос «моложе» и мультяшнее; "
            "манеру задаёт в основном текст промпта."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#6b7280;")
        root.addWidget(hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        save_button = buttons.button(QDialogButtonBox.StandardButton.Save)
        if save_button is not None:
            save_button.setText("Создать" if create else "Сохранить")
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    @staticmethod
    def _slider_row(minimum: int, maximum: int, value: int, suffix: str) -> tuple[QSlider, QLabel]:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)

        label = QLabel()
        label.setFixedWidth(56)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        def fmt(raw: int) -> str:
            sign = "+" if raw > 0 else ""
            return f"{sign}{raw}{suffix}"

        label.setText(fmt(value))
        slider.valueChanged.connect(lambda raw: label.setText(fmt(raw)))
        return slider, label

    @staticmethod
    def _wrap(slider: QSlider, label: QLabel) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(slider, 1)
        row.addWidget(label)
        return row

    def _on_save(self) -> None:
        name = self.name_edit.text().strip() or self._persona.name
        self._result = self._persona.copy(
            name=name,
            emoji=self.emoji_edit.text().strip(),
            greeting=self.greeting_edit.text().strip(),
            prompt=self.prompt_edit.toPlainText().strip() or self._persona.prompt,
            voice=self.voice_combo.currentData() or self._persona.voice,
            rate=self.rate_slider.value(),
            pitch=self.pitch_slider.value(),
            max_speak_chars=self.chars_spin.value(),
        )
        self.accept()

    def result_persona(self) -> Persona | None:
        return self._result
