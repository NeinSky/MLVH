"""Silero v5_ru — локальный русский движок с автоматическими ударениями.

Модель маленькая (десятки МБ) и работает быстро даже на CPU. Интонация ровная,
зато произношение и ударения — лучшие из локальных вариантов.
Лицензия CC BY-NC 4.0: не для коммерческого использования.
"""

from __future__ import annotations

import importlib.util
import logging
import ssl
import threading
import urllib.request
from pathlib import Path

from app.config import BASE_DIR, TTSSettings
from app.engines.base import EngineInfo, TTSEngine, TTSError

LANGUAGE = "ru"
MODEL_URL = "https://models.silero.ai/models/tts/ru/v5_ru.pt"
MODEL_DIR = BASE_DIR / "models" / "silero"
MODEL_PATH = MODEL_DIR / "v5_ru.pt"

log = logging.getLogger(__name__)

INSTALL_HINT = (
    "Движок Silero не установлен.\n\n"
    "В папке проекта:  .\\install_silero.ps1\n"
    "или вручную:      pip install silero\n\n"
    "Модель (~60 МБ) скачается при первом озвучивании."
)

SPEAKERS = {
    "baya": "Бая (жен.)",
    "kseniya": "Ксения (жен.)",
    "xenia": "Ксения (жен., старая)",
    "aidar": "Айдар (муж.)",
    "eugene": "Евгений (муж.)",
}


def _download_model() -> Path:
    """Качаем модель в папку проекта.

    У models.silero.ai просрочен сертификат, поэтому при ошибке проверки
    повторяем без неё — файл кладём только по HTTPS и предупреждаем в лог.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = MODEL_PATH.with_suffix(".part")

    def fetch(context: ssl.SSLContext) -> None:
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "MLVH"})
        with urllib.request.urlopen(request, timeout=120, context=context) as response:
            with open(tmp, "wb") as handle:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    handle.write(block)

    def is_cert_error(exc: BaseException) -> bool:
        """urllib заворачивает SSL-ошибку в URLError — разбираем причину."""
        reason = getattr(exc, "reason", exc)
        return isinstance(reason, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(exc)

    try:
        fetch(ssl.create_default_context())
    except Exception as exc:
        if not is_cert_error(exc):
            tmp.unlink(missing_ok=True)
            raise TTSError(f"Не удалось скачать модель Silero: {exc}") from exc
        log.warning(
            "У models.silero.ai просрочен сертификат — качаю модель без проверки цепочки"
        )
        try:
            fetch(ssl._create_unverified_context())
        except Exception as retry_exc:
            tmp.unlink(missing_ok=True)
            raise TTSError(f"Не удалось скачать модель Silero: {retry_exc}") from retry_exc

    if not tmp.exists() or tmp.stat().st_size < 1_000_000:
        tmp.unlink(missing_ok=True)
        raise TTSError("Модель Silero скачалась не полностью — попробуйте ещё раз.")

    tmp.replace(MODEL_PATH)
    return MODEL_PATH


class SileroEngine(TTSEngine):
    """Быстрый локальный синтез с правильными ударениями."""

    id = "silero"
    title = "Silero v5 (локально)"

    _model = None
    _device = None
    _model_lock = threading.Lock()
    _ready = threading.Event()

    def probe(self) -> EngineInfo:
        missing = [
            name for name in ("torch", "silero") if importlib.util.find_spec(name) is None
        ]
        if missing:
            return EngineInfo(
                id=self.id,
                title=self.title,
                hint=INSTALL_HINT,
                available=False,
                reason="Не установлено: " + ", ".join(missing),
            )
        return EngineInfo(
            id=self.id,
            title=self.title,
            hint=(
                "Локально, быстро, с автоматическими ударениями. "
                "Голоса: Бая, Ксения, Айдар, Евгений. Интонация ровнее, чем у Chatterbox."
            ),
            available=True,
        )

    # --- модель ---------------------------------------------------------
    def prepare(self, settings: TTSSettings | None = None) -> None:
        self._get_model()

    def _get_model(self):
        if self._ready.is_set():
            return self._model

        with self._model_lock:
            if self._model is not None:
                self._ready.set()
                return self._model

            try:
                import torch
                from torch import package
            except ImportError as exc:
                raise TTSError(INSTALL_HINT) from exc

            try:
                if not MODEL_PATH.exists():
                    _download_model()
                # v5 распространяется как torch.package, а не как jit-архив.
                importer = package.PackageImporter(str(MODEL_PATH))
                model = importer.load_pickle("tts_models", "model")
            except TTSError:
                raise
            except Exception as exc:
                raise TTSError(
                    f"Не удалось загрузить модель Silero: {exc}\n"
                    f"Файл модели: {MODEL_PATH}\n"
                    "Если файл повреждён, удалите его — он скачается заново."
                ) from exc

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                model.to(self._device)
            except Exception:
                self._device = "cpu"
            self._model = model
            self._ready.set()
            return self._model

    def wait_ready(self, timeout: float = 120.0) -> bool:
        return self._ready.wait(timeout)

    @property
    def device(self) -> str:
        return self._device or "не загружено"

    # --- синтез ---------------------------------------------------------
    def synthesize(self, text: str, path: Path, settings: TTSSettings) -> Path:
        model = self._get_model()

        try:
            import torchaudio
        except ImportError as exc:
            raise TTSError(INSTALL_HINT) from exc

        speaker = settings.silero_speaker or "baya"
        if speaker not in SPEAKERS:
            speaker = "baya"

        try:
            audio = model.apply_tts(
                text=text,
                speaker=speaker,
                sample_rate=settings.silero_sample_rate,
                put_accent=settings.silero_accent,
                put_yo=settings.silero_yo,
            )
        except Exception as exc:
            raise TTSError(f"Silero не смог озвучить текст: {exc}") from exc

        out = path.with_suffix(".wav")
        try:
            tensor = audio.unsqueeze(0) if audio.dim() == 1 else audio
            torchaudio.save(str(out), tensor.cpu(), settings.silero_sample_rate)
        except Exception as exc:
            raise TTSError(f"Не удалось сохранить звук Silero: {exc}") from exc
        return out
