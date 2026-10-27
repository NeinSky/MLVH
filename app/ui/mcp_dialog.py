"""Диалог MCP-сервера: id, команда, аргументы, окружение и таймаут."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)


class McpServerDialog(QDialog):
    """Добавление или правка одного сервера. Возвращает dict через result_server()."""

    def __init__(
        self,
        server: dict | None = None,
        parent=None,
        create: bool = False,
    ) -> None:
        super().__init__(parent)
        self._create = create
        self._enabled = bool((server or {}).get("enabled", True))
        self._result: dict | None = None

        title = "Новый MCP-сервер" if create else "MCP-сервер"
        self.setWindowTitle(title)
        self.setMinimumWidth(560)
        self.setStyleSheet(parent.styleSheet() if parent else "")

        root = QVBoxLayout(self)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        data = server or {}
        self.id_edit = QLineEdit(str(data.get("id") or ""))
        self.id_edit.setPlaceholderText("например: filesystem")
        form.addRow("id:", self.id_edit)

        self.command_edit = QLineEdit(str(data.get("command") or ""))
        self.command_edit.setPlaceholderText("например: npx, python, node")
        form.addRow("Команда:", self.command_edit)

        self.args_edit = QPlainTextEdit("\n".join(data.get("args") or []))
        self.args_edit.setPlaceholderText(
            "По одному аргументу в строке:\n-y\nchrome-devtools-mcp@latest"
        )
        self.args_edit.setMinimumHeight(80)
        form.addRow("Аргументы:", self.args_edit)

        self.cwd_edit = QLineEdit(str(data.get("cwd") or ""))
        cwd_row = QHBoxLayout()
        cwd_row.addWidget(self.cwd_edit, 1)
        browse = QPushButton("Папка…")
        browse.clicked.connect(self._pick_cwd)
        cwd_row.addWidget(browse)
        form.addRow("Рабочая папка:", cwd_row)

        env_lines = [
            f"{key}={value}" for key, value in (data.get("env") or {}).items()
        ]
        self.env_edit = QPlainTextEdit("\n".join(env_lines))
        self.env_edit.setPlaceholderText("KEY=VALUE, по одному в строке (необязательно)")
        self.env_edit.setMinimumHeight(70)
        form.addRow("Окружение:", self.env_edit)

        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(0, 600)
        self.timeout_spin.setSuffix(" с")
        self.timeout_spin.setSpecialValueText("по умолчанию")
        self.timeout_spin.setValue(int(data.get("timeout") or 0))
        form.addRow("Таймаут вызова:", self.timeout_spin)

        root.addLayout(form)

        hint = QLabel(
            "MCP-сервер — отдельная программа на вашем компьютере. Она не ограничена "
            "папками агента и может менять данные: запускайте только те серверы, "
            "которым доверяете. Ключ DeepSeek серверу не передаётся.\n"
            "Пример: id «chrome-devtools», команда npx, аргументы «-y» и "
            "«chrome-devtools-mcp@latest»."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#6b7280;")
        root.addWidget(hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        save_button = buttons.button(QDialogButtonBox.StandardButton.Save)
        if save_button is not None:
            save_button.setText("Добавить" if create else "Сохранить")
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _pick_cwd(self) -> None:
        start = self.cwd_edit.text().strip() or ""
        folder = QFileDialog.getExistingDirectory(
            self, "Рабочая папка MCP-сервера", start
        )
        if folder:
            self.cwd_edit.setText(folder)

    def _read_env(self) -> dict[str, str] | None:
        env: dict[str, str] = {}
        for line in self.env_edit.toPlainText().splitlines():
            text = line.strip()
            if not text:
                continue
            if "=" not in text:
                QMessageBox.warning(
                    self,
                    "MCP",
                    f"Строка окружения без «=»: {text}\nНужно KEY=VALUE.",
                )
                return None
            key, value = text.split("=", 1)
            key = key.strip()
            if not key:
                QMessageBox.warning(self, "MCP", f"Пустое имя переменной: {text}")
                return None
            env[key] = value.strip()
        return env

    def _on_save(self) -> None:
        server_id = self.id_edit.text().strip()
        command = self.command_edit.text().strip()
        if not server_id:
            QMessageBox.warning(self, "MCP", "Укажите id сервера.")
            return
        if not command:
            QMessageBox.warning(self, "MCP", "Укажите команду запуска.")
            return

        env = self._read_env()
        if env is None:
            return

        args = [
            line.strip()
            for line in self.args_edit.toPlainText().splitlines()
            if line.strip()
        ]
        self._result = {
            "id": server_id,
            "command": command,
            "args": args,
            "env": env,
            "cwd": self.cwd_edit.text().strip(),
            "enabled": self._enabled,
            "timeout": int(self.timeout_spin.value()),
        }
        self.accept()

    def result_server(self) -> dict | None:
        return self._result
