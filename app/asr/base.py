"""Базовые типы распознавания речи: ошибка, описание, интерфейс движка."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.config import ASRSettings


class ASRError(RuntimeError):
    """Ошибка распознавания речи."""


@dataclass
class ASRInfo:
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


class ASREngine:
    """Общий интерфейс движка: проверка доступности и распознавание.

    На вход движок получает float32-массив моно 16 кГц — это же, что отдаёт
    `app.asr.audio.to_mono_16k`, поэтому движки не думают о форматах.
    """

    id: str = "base"
    title: str = "Движок"

    def probe(self) -> ASRInfo:
        raise NotImplementedError

    def prepare(self, settings: ASRSettings) -> None:
        """Необязательная подготовка: загрузка модели (может занять секунды)."""

    def transcribe(self, samples: np.ndarray, settings: ASRSettings) -> str:
        raise NotImplementedError
