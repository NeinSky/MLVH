"""Голосовые движки MLVH.

Edge TTS работает всегда (нужен интернет). Chatterbox — опциональный локальный
движок: если его пакеты не установлены, MLVH просто показывает «не установлен».
"""

from app.engines.base import EngineInfo, TTSEngine, TTSError

__all__ = ["EngineInfo", "TTSEngine", "TTSError"]
