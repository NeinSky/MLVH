"""Смоук окна без монитора: панель настроек влезает, режимы переключаются.

    .venv\\Scripts\\python.exe tools\\ui_smoke.py

Окно собирается offscreen, поэтому нужен только PySide6. Конфиг берётся
временный: `_touch()` в окне сохраняет настройки целиком, и обычный смоук
перезаписал бы настоящий `config.json` (так уже случалось).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}".rstrip())
        FAILED.append(name)


def main() -> int:
    import app.config as config

    # Настройки — во временный файл: настоящий config.json не трогаем.
    config.CONFIG_PATH = Path(tempfile.mkdtemp(prefix="mlvh_ui_")) / "config.json"

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QScrollArea

    application = QApplication.instance() or QApplication([])

    from app.ui.main_window import MainWindow

    window = MainWindow()
    window.show()
    application.processEvents()
    application.processEvents()

    print("окно:")
    check("окно собралось", window.width() > 0 and window.height() > 0)
    check("минимальная ширина разумная", window.minimumSizeHint().width() <= 1280,
          f"{window.minimumSizeHint().width()} px")

    print("боковая панель:")
    panel = window.findChildren(QScrollArea)[0]
    holder = panel.widget()
    needed = holder.minimumSizeHint().width()
    available = panel.viewport().width()
    check("содержимое влезает по ширине", needed <= available,
          f"нужно {needed}, есть {available}")
    check("горизонтального ползунка нет", not panel.horizontalScrollBar().isVisible())

    print("подписи:")
    clipped = [
        label.text()[:24]
        for label in holder.findChildren(QLabel)
        if label.text() and not label.wordWrap()
        and label.sizeHint().width() > label.width() + 1
    ]
    check("ничего не обрезано", not clipped, str(clipped[:3]))

    print("кнопки:")
    heights = {button.sizeHint().height() for button in window.findChildren(QPushButton)}
    check("кнопки компактные", max(heights) <= 24, str(sorted(heights)))

    print("режимы английского:")
    combo = window.english_mode_combo
    check("пять режимов", combo.count() == 5, str(combo.count()))
    hints = [combo.itemData(index, Qt.ItemDataRole.ToolTipRole) for index in range(combo.count())]
    check("у каждого режима есть подсказка", all(hints), str(hints[:2]))
    for mode in ("translit", "silero_en", "auto"):
        combo.setCurrentIndex(combo.findData(mode))
        application.processEvents()
        check(f"переключение в {mode}", window.settings.tts.english_mode == mode,
              window.settings.tts.english_mode)
        check(f"голос EN для {mode}",
              window.english_voice_combo.isEnabled() == (mode in ("auto", "edge")),
              str(window.english_voice_combo.isEnabled()))

    window.close()
    print("")
    if FAILED:
        print(f"Провалено: {len(FAILED)} — {', '.join(FAILED)}")
        return 1
    print("Окно в порядке.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
