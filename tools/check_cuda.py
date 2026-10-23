"""Проверка CUDA: какой torch стоит и видит ли он видеокарту."""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    try:
        import torch
    except ImportError:
        print("torch не установлен.")
        return 1

    print(f"torch: {torch.__version__}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Видеопамять: {total:.1f} ГБ")
        return 0

    if "+cpu" in torch.__version__:
        print("Это CPU-сборка. Для GPU поставьте CUDA-версию (install_torch_cuda.ps1).")
    else:
        print("Сборка с CUDA, но видеокарта не найдена — проверьте драйвер NVIDIA.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
