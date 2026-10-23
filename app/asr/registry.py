"""Реестр движков распознавания: единственное место, где они перечислены."""

from __future__ import annotations

from app.asr.base import ASREngine, ASRError, ASRInfo
from app.asr.vosk_engine import VoskEngine
from app.asr.whisper_engine import WhisperEngine

DEFAULT_ASR_ENGINE = "whisper"

_ENGINES: dict[str, ASREngine] = {
    WhisperEngine.id: WhisperEngine(),
    VoskEngine.id: VoskEngine(),
}


def engine_ids() -> list[str]:
    return list(_ENGINES)


def get_asr_engine(engine_id: str) -> ASREngine:
    """Движок по id; неизвестный id — это whisper, чтобы ввод не ломался."""
    return _ENGINES.get(engine_id) or _ENGINES[DEFAULT_ASR_ENGINE]


def asr_engine_infos() -> list[ASRInfo]:
    """Описания всех движков для интерфейса."""
    infos: list[ASRInfo] = []
    for engine in _ENGINES.values():
        try:
            infos.append(engine.probe())
        except Exception as exc:  # проверка не должна ронять окно
            infos.append(
                ASRInfo(
                    id=engine.id,
                    title=engine.title,
                    hint="",
                    available=False,
                    reason=f"Ошибка проверки: {exc}",
                )
            )
    return infos
