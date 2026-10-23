"""Проверка движков распознавания: что установлено и где модели."""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    from app.asr.registry import asr_engine_infos
    from app.asr.vosk_engine import model_path as vosk_model
    from app.asr.whisper_engine import models_dir

    print("Движки распознавания:")
    for info in asr_engine_infos():
        status = "готов" if info.available else info.reason
        print(f"  {info.id:<12} {info.title:<20} {status}")

    print("")
    print(f"Модели whisper: {models_dir()}")
    print(f"Модель Vosk:    {vosk_model()} {'есть' if vosk_model().is_dir() else 'нет'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
