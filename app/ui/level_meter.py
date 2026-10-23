"""Индикатор уровня микрофона: полоски, цвет зависит от состояния."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QWidget

STATE_COLORS = {
    "idle": "#374151",
    "ready": "#3b82f6",
    "recording": "#ef4444",
    "busy": "#f59e0b",
    "listening": "#8b5cf6",
}


class LevelMeter(QWidget):
    """Полоски уровня: заполнение по громкости, цвет — по состоянию микрофона."""

    def __init__(self, bars: int = 8, parent=None) -> None:
        super().__init__(parent)
        self._bars = max(1, bars)
        self._level = 0.0
        self._state = "idle"
        self._bar_width = 4
        self._gap = 2

        self.setFixedSize(
            self._bars * self._bar_width + (self._bars - 1) * self._gap + 2,
            18,
        )
        self.setToolTip("Уровень микрофона")

    def set_level(self, value: float) -> None:
        value = max(0.0, min(1.0, float(value)))
        if abs(value - self._level) < 0.01:
            return
        self._level = value
        self.update()

    def level(self) -> float:
        return self._level

    def set_state(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        if state == "idle":
            self._level = 0.0
        self.update()

    def state(self) -> str:
        return self._state

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        active = QColor(STATE_COLORS.get(self._state, STATE_COLORS["idle"]))
        idle = QColor("#1f2937")
        filled = int(round(self._level * self._bars))
        height = self.height() - 4

        painter.setPen(Qt.PenStyle.NoPen)
        for index in range(self._bars):
            painter.setBrush(active if index < filled else idle)
            rect = QRectF(
                1 + index * (self._bar_width + self._gap),
                2,
                self._bar_width,
                height,
            )
            painter.drawRoundedRect(rect, 1.5, 1.5)
        painter.end()
