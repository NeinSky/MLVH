"""Базовые типы движков озвучки: ошибка, описание, интерфейс."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import TTSSettings


class TTSError(RuntimeError):
    """Ошибка синтеза или воспроизведения речи."""


@dataclass
class EngineInfo:
    """Что показать в интерфейсе и что проверить перед запуском."""

    id: str
    title: str
    hint: str
    available: bool
    reason: str = ""

    def label(self) -> str:
        if self.available:
            return self.title
        return f"{self.title} — не установлен"


class TTSEngine:
    """Общий интерфейс движка: проверка доступности и синтез."""

    id: str = "base"
    title: str = "Движок"

    def probe(self) -> EngineInfo:
        raise NotImplementedError

    def synthesize(self, text: str, path: Path, settings: TTSSettings) -> Path:
        raise NotImplementedError

    def prepare(self) -> None:
        """Необязательная подготовка (например, загрузка модели)."""
