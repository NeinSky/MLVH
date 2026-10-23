"""Предзагрузка моделей Chatterbox в папку проекта (для офлайн-переноса).

Запуск:  .venv\\Scripts\\python.exe tools\\download_models.py

После этого папку models\\ можно скопировать на другую машину вместе с проектом:
MLVH ищет модели именно там (переменная HF_HOME выставляется автоматически).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
HF_HOME = MODELS_DIR / "hf"

os.environ.setdefault("HF_HOME", str(HF_HOME))


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} ТБ"


def main() -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Нет huggingface_hub. Сначала установите движок:")
        print("  .\\install_chatterbox.ps1")
        return 1

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Скачиваю модель Chatterbox в {HF_HOME} …")
    try:
        path = snapshot_download(repo_id="ResembleAI/chatterbox")
    except Exception as exc:
        print(f"Не удалось скачать модель: {exc}")
        return 1

    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    print(f"Готово: {path}")
    print(f"Размер модели: {human_size(total)}")
    print("")
    print("Чтобы перенести на другую машину, скопируйте папки:")
    print(f"  {MODELS_DIR}   (модели и образцы голоса)")
    print("  .venv            (установленные пакеты — только для той же ОС и Python)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
