"""Диалог провайдера модели: DeepSeek или любой OpenAI-совместимый API.

DeepSeek — достаточно ключа. Чужой провайдер — нужны адрес API, алиас модели
и ключ; алиас после этого задаётся здесь, а не в выпадающем списке окна.
"""

from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from app.config import (
    AVAILABLE_MODELS,
    DEFAULT_KEY_ENV,
    DEFAULT_MODEL_ALIAS,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_KEY_ENV,
    ModelSettings,
    normalize_base_url,
    normalize_env_name,
)
from app.ui.workers import ProviderProbeWorker

THINKING_TITLES = (
    ("Авто: только DeepSeek", "auto"),
    ("Не отправлять", "none"),
    ("Всегда DeepSeek-формат", "deepseek"),
)


class ProviderDialog(QDialog):
    """Провайдер, алиас модели и ключ.

    Результат — `result_settings()`: dict с base_url, name, api_key_env,
    thinking_style и api_key (пустая строка — ключ не меняем).
    """

    def __init__(self, model: ModelSettings, parent=None) -> None:
        super().__init__(parent)
        self._model = model
        self._result: dict | None = None
        self._probe: ProviderProbeWorker | None = None

        self.setWindowTitle("Ключ и провайдер")
        self.setMinimumWidth(620)
        self.setStyleSheet(parent.styleSheet() if parent else "")

        root = QVBoxLayout(self)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.provider_combo = QComboBox()
        self.provider_combo.addItem("DeepSeek — нужен только ключ", "deepseek")
        self.provider_combo.addItem(
            "OpenAI-совместимый API — адрес, алиас модели и ключ", "custom"
        )
        self.provider_combo.setCurrentIndex(0 if model.is_deepseek() else 1)
        self.provider_combo.currentIndexChanged.connect(self._sync_fields)
        form.addRow("Провайдер:", self.provider_combo)

        self.base_url_edit = QLineEdit(model.base_url)
        self.base_url_edit.setPlaceholderText(
            "например: https://openrouter.ai/api/v1, http://localhost:11434/v1"
        )
        form.addRow("Адрес API:", self.base_url_edit)

        self.alias_edit = QLineEdit(model.name)
        self.alias_edit.setPlaceholderText(DEFAULT_MODEL_ALIAS)
        form.addRow("Алиас модели:", self.alias_edit)

        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Ключ:", self.key_edit)

        self.env_edit = QLineEdit(model.api_key_env)
        form.addRow("Переменная с ключом:", self.env_edit)

        self.thinking_combo = QComboBox()
        for title, value in THINKING_TITLES:
            self.thinking_combo.addItem(title, value)
        index = self.thinking_combo.findData(model.thinking_style)
        if index >= 0:
            self.thinking_combo.setCurrentIndex(index)
        form.addRow("Режим рассуждений:", self.thinking_combo)

        probe_row = QHBoxLayout()
        self.probe_button = QPushButton("Проверить подключение…")
        self.probe_button.clicked.connect(self._probe_connection)
        probe_row.addWidget(self.probe_button)
        self.probe_label = QLabel("")
        self.probe_label.setWordWrap(True)
        self.probe_label.setStyleSheet("color:#6b7280;")
        probe_row.addWidget(self.probe_label, 1)
        form.addRow("", probe_row)

        root.addLayout(form)

        self.hint = QLabel("")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color:#6b7280;")
        root.addWidget(self.hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        save_button = buttons.button(QDialogButtonBox.StandardButton.Save)
        if save_button is not None:
            save_button.setText("Сохранить")
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._sync_fields()

    # --- поля ------------------------------------------------------------
    def _is_custom(self) -> bool:
        return self.provider_combo.currentData() == "custom"

    def _key_env_name(self) -> str:
        entered = normalize_env_name(self.env_edit.text())
        if entered:
            return entered
        return DEFAULT_KEY_ENV if self._is_custom() else DEEPSEEK_KEY_ENV

    def _existing_key(self) -> str:
        return (os.getenv(self._key_env_name()) or "").strip()

    def _sync_fields(self) -> None:
        """DeepSeek — только ключ; чужой провайдер — ещё адрес и алиас."""
        custom = self._is_custom()
        self.base_url_edit.setEnabled(custom)
        if custom:
            self.base_url_edit.setPlaceholderText(
                "например: https://openrouter.ai/api/v1, http://localhost:11434/v1"
            )
            self.alias_edit.setPlaceholderText(
                "алиас у провайдера, например openai/gpt-4o-mini"
            )
            env_hint = DEFAULT_KEY_ENV
            self.hint.setText(
                "Ответы пойдут на указанный адрес. Алиас модели берётся отсюда, поэтому "
                "выпадающий список «Модель» в окне станет недоступен.\n"
                "Нужен OpenAI-совместимый API: OpenRouter, Groq, Ollama, LM Studio, "
                "vLLM или корпоративный шлюз."
            )
        else:
            self.base_url_edit.setText("")
            self.alias_edit.setPlaceholderText(DEFAULT_MODEL_ALIAS)
            env_hint = DEEPSEEK_KEY_ENV
            aliases = ", ".join(AVAILABLE_MODELS)
            self.hint.setText(
                "DeepSeek: достаточно ключа с platform.deepseek.com/api_keys. "
                f"Модель выбирается в окне ({aliases}); пустое имя — {DEFAULT_MODEL_ALIAS}."
            )
        self.env_edit.setPlaceholderText(env_hint)
        has_key = bool(self._existing_key())
        self.key_edit.setPlaceholderText(
            "ключ уже сохранён — оставьте пустым, чтобы не менять"
            if has_key
            else f"ключ {env_hint}"
        )
        self.probe_label.setText("")

    # --- проверка подключения --------------------------------------------
    def _probe_connection(self) -> None:
        if self._probe is not None and self._probe.isRunning():
            return
        base_url = (
            normalize_base_url(self.base_url_edit.text())
            if self._is_custom()
            else DEEPSEEK_BASE_URL
        )
        if not base_url:
            self.probe_label.setText("Сначала укажите адрес API.")
            return
        api_key = self.key_edit.text().strip() or self._existing_key()
        self.probe_label.setText("Проверяю…")
        self.probe_button.setEnabled(False)

        self._probe = ProviderProbeWorker(base_url, api_key, parent=self)
        self._probe.done.connect(self._on_probe_done)
        self._probe.failed.connect(self._on_probe_failed)
        self._probe.finished.connect(self._on_probe_finished)
        self._probe.start()

    def _on_probe_done(self, url: str, models: list) -> None:
        if not models:
            self.probe_label.setText(f"Подключение есть ({url}), но моделей нет.")
            return
        shown = ", ".join(str(item) for item in models[:5])
        more = f" и ещё {len(models) - 5}" if len(models) > 5 else ""
        self.probe_label.setText(f"Моделей: {len(models)}: {shown}{more}")

    def _on_probe_failed(self, message: str) -> None:
        self.probe_label.setText(message)

    def _on_probe_finished(self) -> None:
        worker, self._probe = self._probe, None
        self.probe_button.setEnabled(True)
        if worker is not None:
            worker.deleteLater()

    # --- сохранение ------------------------------------------------------
    def _on_save(self) -> None:
        custom = self._is_custom()
        base_url = normalize_base_url(self.base_url_edit.text()) if custom else ""
        alias = self.alias_edit.text().strip()
        if custom:
            if not base_url:
                QMessageBox.warning(
                    self,
                    "MLVH",
                    "Для своего провайдера нужен адрес API "
                    "(например https://openrouter.ai/api/v1).",
                )
                return
            if not alias:
                QMessageBox.warning(
                    self, "MLVH",
                    "Укажите алиас модели — так она называется у провайдера.",
                )
                return

        env_raw = self.env_edit.text().strip()
        env_name = normalize_env_name(env_raw)
        if env_raw and not env_name:
            QMessageBox.warning(
                self,
                "MLVH",
                "Имя переменной может состоять из латинских букв, цифр и «_». "
                "Например: OPENROUTER_API_KEY.",
            )
            return

        key = self.key_edit.text().strip()
        if not key and not self._existing_key():
            QMessageBox.warning(
                self,
                "MLVH",
                f"Вставьте ключ — он сохранится в .env как {self._key_env_name()}.",
            )
            return

        if not custom and not alias:
            alias = DEFAULT_MODEL_ALIAS

        if self._probe is not None and self._probe.isRunning():
            self._probe.wait(3000)
            self._probe = None

        self._result = {
            "base_url": base_url,
            "name": alias,
            "api_key_env": env_name,
            "thinking_style": self.thinking_combo.currentData() or "auto",
            "api_key": key,
        }
        self.accept()

    def result_settings(self) -> dict | None:
        return self._result

    def closeEvent(self, event) -> None:  # noqa: N802 — имя задаёт Qt
        if self._probe is not None and self._probe.isRunning():
            self._probe.wait(3000)
            self._probe = None
        super().closeEvent(event)