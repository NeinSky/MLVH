"""Silero v3_en — локальная английская модель для английских вставок.

Не отдельный движок в списке: пользователь выбирает не его, а режим
«Локальной английской моделью» в настройках английских вставок. Нужна, когда
интернета нет, а английскую фразу транслитом читать уже неприятно.

Модель 2019 года на LJSpeech (57 МБ, CC BY-NC 4.0): разборчиво, но интонация
плоская — Edge звучит живее, поэтому по умолчанию выбран он.
"""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path

from app.config import BASE_DIR, TTSSettings
from app.engines.base import EngineInfo, TTSEngine, TTSError
from app.engines.silero_engine import _download_url

MODEL_URL = "https://models.silero.ai/models/tts/en/v3_en.pt"
MODEL_DIR = BASE_DIR / "models" / "silero"
MODEL_PATH = MODEL_DIR / "v3_en.pt"

DEFAULT_SPEAKER = "en_0"

INSTALL_HINT = (
    "Английская модель Silero не скачана.\n\n"
    "Она нужна только для режима «Локальной английской моделью». "
    "Файл (~57 МБ) скачается при первом использовании."
)


class SileroEnEngine(TTSEngine):
    """Английский синтез Silero: тот же интерфейс, что у остальных движков."""

    id = "silero_en"
    title = "Silero v3_en (локально)"

    _instance: "SileroEnEngine | None" = None
    _model = None
    _device = None
    _model_lock = threading.Lock()
    _ready = threading.Event()

    @classmethod
    def instance(cls) -> "SileroEnEngine":
        """Одна модель на процесс: вторая копия съела бы ещё 58 МБ видеопамяти."""
        if cls._instance is None:
            cls._instance = SileroEnEngine()
        return cls._instance

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
        return EngineInfo(id=self.id, title=self.title, hint=INSTALL_HINT, available=True)

    def wait_ready(self, timeout: float = 120.0) -> bool:
        return self._ready.wait(timeout)

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
                raise TTSError("Не установлен torch: .\\install_silero.ps1") from exc

            try:
                if not MODEL_PATH.exists():
                    MODEL_DIR.mkdir(parents=True, exist_ok=True)
                    _download_url(MODEL_URL, MODEL_PATH)
                model = package.PackageImporter(str(MODEL_PATH)).load_pickle("tts_models", "model")
            except TTSError:
                raise
            except Exception as exc:
                raise TTSError(
                    f"Не удалось загрузить английскую модель Silero: {exc}\n"
                    f"Файл модели: {MODEL_PATH}"
                ) from exc

            # Внимание: `TTSModelMultiAcc_v3.to()` ничего не возвращает — результат
            # присваивать нельзя, иначе модель станет None.
            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                model.to(device)
            except Exception:
                device = "cpu"
                model.to(device)
            self._device = device
            self._model = model
            self._ready.set()
            return self._model

    def synthesize(self, text: str, path: Path, settings: TTSSettings) -> Path:
        model = self._get_model()
        try:
            import torchaudio
        except ImportError as exc:
            raise TTSError("Не установлен torchaudio: .\\install_silero.ps1") from exc

        try:
            audio = model.apply_tts(text=text, speaker=DEFAULT_SPEAKER, sample_rate=48000)
        except Exception as exc:
            raise TTSError(f"Silero v3_en не смог озвучить текст: {exc}") from exc

        out = path.with_suffix(".wav")
        try:
            tensor = audio.unsqueeze(0) if audio.dim() == 1 else audio
            torchaudio.save(str(out), tensor.cpu(), 48000)
        except Exception as exc:
            raise TTSError(f"Не удалось сохранить звук Silero v3_en: {exc}") from exc
        return out
