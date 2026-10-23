"""Chatterbox Multilingual — опциональный локальный движок (MIT, русский, эмоции).

Модуль намеренно не импортирует torch/chatterbox на уровне файла: если пакеты
не установлены, MLVH продолжает работать на Edge TTS, а здесь просто
возвращается «не установлен» с инструкцией.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import threading
from pathlib import Path

from app.config import BASE_DIR, TTSSettings
from app.engines.base import EngineInfo, TTSEngine, TTSError
from app.engines.edge_engine import EdgeEngine

MODELS_DIR = BASE_DIR / "models"
REFERENCE_DIR = MODELS_DIR / "references"
LANGUAGE = "ru"

log = logging.getLogger(__name__)

INSTALL_HINT = (
    "Локальный движок не установлен.\n\n"
    "В папке проекта:  .\\install_chatterbox.ps1\n"
    "или вручную:      pip install -r requirements-chatterbox.txt\n\n"
    "Модель (~2 ГБ) скачается при первом озвучивании."
)


class ChatterboxEngine(TTSEngine):
    """Локальный синтез с эмоциями и клонированием голоса по образцу."""

    id = "chatterbox"
    title = "Chatterbox (локально)"

    _model = None
    _device = None
    _model_lock = threading.Lock()
    _ready = threading.Event()
    _conds_key: tuple | None = None

    # --- доступность ----------------------------------------------------
    def probe(self) -> EngineInfo:
        missing = [
            name
            for name in ("torch", "torchaudio", "chatterbox", "librosa")
            if importlib.util.find_spec(name) is None
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
                "Локально, без интернета. Эмоции и клонирование голоса по образцу.\n"
                "Первый запуск грузит модель (~2 ГБ), дальше — быстро."
            ),
            available=True,
        )

    # --- модель ---------------------------------------------------------
    def prepare(self, settings: TTSSettings | None = None) -> None:
        """Загружаем модель заранее (можно звать из фонового потока).

        Если переданы настройки — заодно считаем conditionals и прогоняем
        короткий синтез: первый настоящий ответ не платит за разогрев CUDA.
        """
        model = self._get_model()
        if settings is None:
            return
        try:
            reference = self._reference(settings)
            self._ensure_conditionals(model, reference, settings)
            model.generate(
                "Проверка.",
                language_id=LANGUAGE,
                exaggeration=settings.chatterbox_exaggeration,
                cfg_weight=settings.chatterbox_cfg,
            )
        except Exception as exc:  # прогрев не критичен для работы
            log.warning("Прогрев Chatterbox не удался: %s", exc)

    def _get_model(self):
        """Модель грузится один раз; второй поток ждёт, а не грузит её заново."""
        if self._ready.is_set():
            return self._model

        with self._model_lock:
            if self._model is not None:
                self._ready.set()
                return self._model

            # Модели держим внутри проекта: папку models можно перенести целиком.
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))

            try:
                import torch
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            except ImportError as exc:
                raise TTSError(INSTALL_HINT) from exc

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                self._model = ChatterboxMultilingualTTS.from_pretrained(device=self._device)
            except Exception as exc:
                raise TTSError(
                    f"Не удалось загрузить модель Chatterbox: {exc}\n"
                    "Проверьте интернет (модель качается один раз) и место на диске."
                ) from exc
            self._ready.set()
            return self._model

    def wait_ready(self, timeout: float = 120.0) -> bool:
        """Ждём, пока модель загрузится (если её греет другой поток)."""
        return self._ready.wait(timeout)

    @property
    def device(self) -> str:
        return self._device or "не загружено"

    # --- образец голоса -------------------------------------------------
    def _reference(self, settings: TTSSettings) -> Path:
        """Путь к образцу голоса. Если не задан — делаем образец из Edge-голоса."""
        custom = (settings.chatterbox_reference or "").strip()
        if custom:
            path = Path(custom)
            if path.exists():
                return path
            raise TTSError(f"Файл образца голоса не найден: {custom}")

        REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        name = f"edge_{settings.voice}_r{settings.rate}_p{settings.pitch}.mp3"
        cached = REFERENCE_DIR / name
        if cached.exists() and cached.stat().st_size > 0:
            return cached

        # Берём тембр из Edge: получаем образец и уже его клонируем локально.
        sample_text = (
            "Привет! Это образец моего голоса. "
            "Я говорю спокойно, чтобы было слышно каждое слово."
        )
        edge_settings = TTSSettings(
            voice=settings.voice,
            rate=settings.rate,
            volume=settings.volume,
            pitch=settings.pitch,
        )
        EdgeEngine().synthesize(sample_text, cached, edge_settings)
        return cached

    def _ensure_conditionals(self, model, reference: Path, settings: TTSSettings) -> None:
        """Conditionals зависят от образца и выразительности — считаем один раз."""
        key = (str(reference), round(float(settings.chatterbox_exaggeration), 3))
        if self._conds_key == key:
            return
        model.prepare_conditionals(
            str(reference),
            exaggeration=settings.chatterbox_exaggeration,
        )
        self._conds_key = key

    def _apply_precision(self, model, settings: TTSSettings) -> None:
        """FP16 на GPU: быстрее и меньше памяти, качество почти не страдает."""
        if self._device != "cuda" or not settings.chatterbox_fp16:
            return
        if getattr(self, "_fp16_applied", False):
            return
        try:
            model.t3.half()
            model.s3gen.half()
            self._fp16_applied = True
            log.info("Chatterbox переведён в FP16")
        except Exception as exc:
            log.warning("FP16 не применился: %s", exc)

    def _watermark_off(self, model) -> None:
        """На потоковых кусках метка не нужна: файл сразу удаляется."""
        if getattr(self, "_wm_disabled", False):
            return
        try:
            from perth import DummyWatermarker

            model.watermarker = DummyWatermarker()
            self._wm_disabled = True
        except Exception as exc:
            log.warning("Не удалось отключить водяной знак: %s", exc)

    # --- синтез ---------------------------------------------------------
    def synthesize(self, text: str, path: Path, settings: TTSSettings) -> Path:
        # Модель грузится при первом обращении; в окне она греется заранее,
        # но если пришли сюда напрямую — прогреваем сейчас.
        model = self._get_model()
        reference = self._reference(settings)

        try:
            import torchaudio
        except ImportError as exc:
            raise TTSError(INSTALL_HINT) from exc

        self._apply_precision(model, settings)
        if not settings.chatterbox_watermark:
            self._watermark_off(model)

        try:
            self._ensure_conditionals(model, reference, settings)
            wav = self._generate(model, text, settings)
        except TTSError:
            raise
        except Exception as exc:
            # FP16 иногда не дружит с conditionals — откатываемся на FP32.
            if getattr(self, "_fp16_applied", False):
                log.warning("Синтез в FP16 не удался (%s), откат на FP32", exc)
                self._drop_fp16(model)
                try:
                    self._ensure_conditionals(model, reference, settings)
                    wav = self._generate(model, text, settings)
                except Exception as retry_exc:
                    raise TTSError(
                        f"Chatterbox не смог озвучить текст: {retry_exc}"
                    ) from retry_exc
            else:
                raise TTSError(f"Chatterbox не смог озвучить текст: {exc}") from exc

        out = path.with_suffix(".wav")
        try:
            torchaudio.save(str(out), wav, model.sr)
        except Exception as exc:
            raise TTSError(f"Не удалось сохранить звук Chatterbox: {exc}") from exc
        return out

    def _generate(self, model, text: str, settings: TTSSettings):
        return model.generate(
            text,
            language_id=LANGUAGE,
            exaggeration=settings.chatterbox_exaggeration,
            cfg_weight=settings.chatterbox_cfg,
            temperature=settings.chatterbox_temperature,
            repetition_penalty=settings.chatterbox_repetition_penalty,
        )

    def _drop_fp16(self, model) -> None:
        try:
            model.t3.float()
            model.s3gen.float()
        except Exception:
            pass
        self._fp16_applied = False
        self._conds_key = None
