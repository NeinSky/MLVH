# Установка локального движка Chatterbox (опционально).
# MLVH работает и без него: Edge TTS остаётся движком по умолчанию.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Не найдено окружение .venv. Сначала: python -m venv .venv"
    exit 1
}

Write-Host "Устанавливаю Chatterbox (torch ~2.5 ГБ, займёт время)..."
& $python -m pip install --upgrade pip

# torch с PyPI — CPU-сборка. Если есть NVIDIA, ставим CUDA-вариант.
$hasNvidia = $null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue)
if ($hasNvidia) {
    Write-Host "Найдена NVIDIA — ставлю torch с поддержкой CUDA (cu124)..."
    & $python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    if ($LASTEXITCODE -ne 0) {
        Write-Host "CUDA-сборка не встала, откатываюсь на CPU-версию."
        & $python -m pip install torch==2.6.0 torchaudio==2.6.0
    }
} else {
    Write-Host "NVIDIA не найдена — будет CPU-синтез (медленнее)."
}

& $python -m pip install -r requirements-chatterbox.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "Установка не удалась. Смотрите сообщения выше."
    exit 1
}

Write-Host ""
Write-Host "Проверяю движок..."
& $python -c "import sys; sys.path.insert(0, '.'); from app.engines.registry import engine_infos; [print(' ', i.id, '—', 'готов' if i.available else i.reason) for i in engine_infos()]"

Write-Host ""
Write-Host "Готово. Теперь в окне MLVH: Голос → Движок → «Chatterbox (локально)»."
Write-Host "Модель (~2 ГБ) скачается при первом озвучивании и ляжет в папку models\."
