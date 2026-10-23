"""faster-whisper: локальное распознавание, CUDA с фолбэком на CPU."""

from __future__ import annotations

import importlib.util
import os
import threading
from pathlib import Path

import numpy as np

from app.asr.audio import as_float
from app.asr.base import ASREngine, ASRError, ASRInfo
from app.config import BASE_DIR, ASRSettings

MODELS = ("tiny", "base", "small", "medium", "large-v3")

# Подсказка модели: с ней она ставит знаки препинания и не уходит в английский.
PROMPT = "Вопрос или команда голосовому помощнику по-русски."


def models_dir() -> Path:
    return BASE_DIR / "models" / "whisper"


def _add_torch_dlls() -> None:
    """ctranslate2 ищет cuDNN рядом с собой; в Windows он лежит в torch/lib."""
    if os.name != "nt":
        return
    try:
        import torch

        lib = Path(torch.__file__).parent / "lib"
        if lib.is_dir():
            os.add_dll_directory(str(lib))
    except Exception:
        # Без torch работаем на CPU: это медленнее, но не смертельно.
        pass


class WhisperEngine(ASREngine):
    """faster-whisper (CTranslate2): лучшее качество на русском."""

    id = "whisper"
    title = "faster-whisper"

    def __init__(self) -> None:
        self._model = None
        self._key: tuple[str, str, str] | None = None
        self._lock = threading.Lock()
        self.device_name = ""

    def probe(self) -> ASRInfo:
        missing = [
            name
            for name in ("faster_whisper", "ctranslate2")
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            return ASRInfo(
                id=self.id,
                title=self.title,
                hint="Установите движок: .\\install_asr.ps1",
                available=False,
                reason="не установлен: " + ", ".join(missing),
            )
        return ASRInfo(
            id=self.id,
            title=self.title,
            hint=(
                "Локальное распознавание, работает офлайн. Модель скачается "
                f"при первом запуске в {models_dir()}."
            ),
            available=True,
        )

    def prepare(self, settings: ASRSettings) -> None:
        self._load(settings)

    def _load(self, settings: ASRSettings):
        key = (settings.model, settings.device, settings.compute_type)
        if self._model is not None and self._key == key:
            return self._model

        # Прогрев и распознавание идут в разных потоках. Две одновременные
        # загрузки CTranslate2 вешали приложение при смене модели.
        with self._lock:
            if self._model is not None and self._key == key:
                return self._model
            return self._load_locked(settings, key)

    def _load_locked(self, settings: ASRSettings, key: tuple[str, str, str]):
        _add_torch_dlls()
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ASRError("faster-whisper не установлен: .\\install_asr.ps1") from exc

        folder = models_dir()
        folder.mkdir(parents=True, exist_ok=True)

        # CUDA может не подняться (нет cuDNN, старая карта) — тогда CPU.
        attempts: list[tuple[str, str]] = []
        if settings.device == "cuda":
            attempts.append(("cuda", settings.compute_type or "int8_float16"))
            attempts.append(("cuda", "float16"))
        attempts.append(("cpu", "int8"))

        last_error: Exception | None = None
        for device, compute in attempts:
            try:
                self._model = WhisperModel(
                    settings.model,
                    device=device,
                    compute_type=compute,
                    download_root=str(folder),
                )
                self.device_name = f"{device}/{compute}"
                self._key = key
                return self._model
            except Exception as exc:
                last_error = exc
        raise ASRError(f"Не удалось загрузить модель «{settings.model}»: {last_error}")

    def transcribe(self, samples: np.ndarray, settings: ASRSettings) -> str:
        model = self._load(settings)
        data = as_float(samples)
        if data.size == 0:
            return ""

        try:
            segments, _info = model.transcribe(
                data,
                language=settings.language or "ru",
                task="transcribe",
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
                initial_prompt=PROMPT,
                without_timestamps=True,
            )
            return " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as exc:
            raise ASRError(f"Ошибка распознавания: {exc}") from exc
