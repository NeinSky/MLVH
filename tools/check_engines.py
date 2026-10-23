"""Проверка движков: какие установлены и что с GPU."""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
