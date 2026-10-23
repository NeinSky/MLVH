"""Edge TTS — облачный движок Microsoft, работает без установки моделей."""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.config import TTSSettings
from app.engines.base import EngineInfo, TTSEngine, TTSError


class EdgeEngine(TTSEngine):
    """Бесплатный облачный синтез: быстро, но без эмоций и стилей."""

    id = "edge"
    title = "Edge TTS (облако)"

    def probe(self) -> EngineInfo:
        return EngineInfo(
            id=self.id,
            title=self.title,
            hint="Бесплатно, нужен интернет. Голоса Svetlana, Dmitry и другие.",
            available=True,
        )

    def synthesize(self, text: str, path: Path, settings: TTSSettings) -> Path:
        try:
            import edge_tts
        except ImportError as exc:  # pragma: no cover - зависимость ставится всегда
            raise TTSError("Не установлен edge-tts: pip install edge-tts") from exc

        async def run() -> None:
            communicate = edge_tts.Communicate(
                text,
                settings.voice,
                rate=f"{settings.rate:+d}%",
                volume=f"{settings.volume:+d}%",
                pitch=f"{settings.pitch:+d}Hz",
            )
            await communicate.save(str(path))

        asyncio.run(run())
        return path
