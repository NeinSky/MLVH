"""Реестр движков: единственное место, где они перечислены."""

from __future__ import annotations

from app.engines.base import EngineInfo, TTSEngine, TTSError
from app.engines.chatterbox_engine import ChatterboxEngine
from app.engines.edge_engine import EdgeEngine
from app.engines.silero_engine import SileroEngine

DEFAULT_ENGINE = "edge"

_ENGINES: dict[str, TTSEngine] = {
    EdgeEngine.id: EdgeEngine(),
    ChatterboxEngine.id: ChatterboxEngine(),
    SileroEngine.id: SileroEngine(),
}


def engine_ids() -> list[str]:
    return list(_ENGINES)


def get_engine(engine_id: str) -> TTSEngine:
    """Движок по id; неизвестный id — это Edge, чтобы озвучка не ломалась."""
    return _ENGINES.get(engine_id) or _ENGINES[DEFAULT_ENGINE]


def engine_infos() -> list[EngineInfo]:
    """Описания всех движков для интерфейса."""
    infos = []
    for engine in _ENGINES.values():
        try:
            infos.append(engine.probe())
        except Exception as exc:  # проверка не должна ронять окно
            infos.append(
                EngineInfo(
                    id=engine.id,
                    title=engine.title,
                    hint="",
                    available=False,
                    reason=f"Ошибка проверки: {exc}",
                )
            )
    return infos
