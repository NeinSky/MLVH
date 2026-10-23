"""Vosk: лёгкое офлайн-распознавание на CPU — для режима «Слушать микрофон»."""

from __future__ import annotations

import importlib.util
import json
import threading
from pathlib import Path

import numpy as np

from app.asr.audio import TARGET_RATE, as_float
from app.asr.base import ASREngine, ASRError, ASRInfo
from app.config import BASE_DIR, ASRSettings

MODEL_NAME = "vosk-model-small-ru-0.22"


def models_dir() -> Path:
    return BASE_DIR / "models" / "vosk"


def model_path() -> Path:
    return models_dir() / MODEL_NAME


class VoskEngine(ASREngine):
    """Vosk small-ru: ~45 МБ, быстро на CPU, без пунктуации."""

    id = "vosk"
    title = "Vosk (офлайн)"

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    def probe(self) -> ASRInfo:
        if importlib.util.find_spec("vosk") is None:
            return ASRInfo(
                id=self.id,
                title=self.title,
                hint="Установите движок: .\\install_asr.ps1",
                available=False,
                reason="не установлен пакет vosk",
            )
        if not model_path().is_dir():
            return ASRInfo(
                id=self.id,
                title=self.title,
                hint=f"Скачайте модель в {model_path()}: .\\install_asr.ps1",
                available=False,
                reason="нет модели",
            )
        return ASRInfo(
            id=self.id,
            title=self.title,
            hint="Лёгкая офлайн-модель для спецслова и коротких фраз.",
            available=True,
        )

    def prepare(self, settings: ASRSettings) -> None:
        self.load_model()

    def load_model(self):
        """Общая модель Vosk для всех потребителей.

        Модель грузится один раз и под блокировкой: одновременная загрузка из
        двух потоков (прогрев и слушатель спецслова) вешала приложение.
        """
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from vosk import Model
            except ImportError as exc:
                raise ASRError("vosk не установлен: .\\install_asr.ps1") from exc
            try:
                self._model = Model(str(model_path()))
            except Exception as exc:
                raise ASRError(f"Не удалось загрузить модель Vosk: {exc}") from exc
            return self._model

    def _load(self):
        return self.load_model()

    def transcribe(self, samples: np.ndarray, settings: ASRSettings) -> str:
        model = self._load()
        data = as_float(samples)
        if data.size == 0:
            return ""

        from vosk import KaldiRecognizer

        recognizer = KaldiRecognizer(model, TARGET_RATE)
        recognizer.AcceptWaveform((data * 32767.0).astype(np.int16).tobytes())
        try:
            return str(json.loads(recognizer.FinalResult()).get("text", "")).strip()
        except json.JSONDecodeError:
            return ""
