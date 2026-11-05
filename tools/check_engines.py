"""Проверка движков: какие установлены, что с GPU и как читается английский.

    python tools/check_engines.py
    python tools/check_engines.py --live
    python tools/check_engines.py --live --mode edge

`--live` синтезирует пробную фразу (без проигрывания) и показывает, на какие
части она разложилась и каким движком каждая прочитана.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SAMPLE = "Курс Python и Docker — это база, а GitHub нужен для pull request."


def show_engines() -> None:
    try:
        import torch

        print(f"torch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU: не найдена (локальные движки будут медленнее)")
    except ImportError:
        print("torch: не установлен — локальные движки недоступны")

    from app.engines.registry import engine_infos

    print("")
    print("Движки:")
    for info in engine_infos():
        status = "готов" if info.available else info.reason
        print(f"  {info.id:<12} {info.title:<24} {status}")


def live(mode: str = "") -> int:
    """Синтез пробной фразы по частям: что куда ушло и сколько заняло."""
    from app.config import load_settings
    from app.speech_parts import build_parts
    from app.tts import TTSService, SpeechPipeline, clean_for_speech

    settings = load_settings()
    if mode:
        settings.tts.english_mode = mode
    print("")
    print(f"Пробная фраза: {SAMPLE}")
    print(f"Режим английского: {settings.tts.english_mode}, движок: {settings.tts.engine}")

    cleaned = clean_for_speech(SAMPLE, normalize_numbers=True)
    parts = build_parts(cleaned, settings.tts)
    for part in parts:
        print(f"  {part.lang}: {part.text.strip()}")

    folder = Path(tempfile.mkdtemp(prefix="mlvh_check_"))
    tts = TTSService()
    pipeline = SpeechPipeline(
        tts, settings.tts, on_note=lambda note: print("  примечание:", note)
    )
    for index, part in enumerate(parts):
        started = time.time()
        try:
            paths = pipeline._synthesize_part(part, folder / f"part_{index}.mp3")
        except Exception as exc:            # проверка не должна падать
            print(f"  {part.lang}: ошибка — {exc}")
            continue
        spent = time.time() - started
        names = ", ".join(path.name for path in paths) or "нет файла"
        print(f"  {part.lang}: {spent:.2f} с → {names}")

    tts.clear_cache()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Проверка движков MLVH")
    parser.add_argument("--live", action="store_true", help="синтезировать пробную фразу")
    parser.add_argument("--mode", default="", help="english_mode: auto/translit/edge/silero_en/off")
    args = parser.parse_args(argv)

    show_engines()
    if args.live:
        return live(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

