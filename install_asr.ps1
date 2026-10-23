# Установка распознавания речи (голосовой ввод).
# Опционально: без этого MLVH работает как раньше, микрофон просто выключен.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Не найдено окружение .venv. Сначала: python -m venv .venv"
    exit 1
}

Write-Host "Устанавливаю faster-whisper и Vosk..."
& $python -m pip install -r requirements-asr.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "Установка не удалась."
    exit 1
}

Write-Host ""
Write-Host "Проверяю движки..."
& $python tools\check_asr.py

Write-Host ""
Write-Host "Готово. Модель whisper скачается при первом распознавании."
Write-Host "Для режима «Слушать микрофон» скачайте модель Vosk small-ru:"
Write-Host "  https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip"
Write-Host "  распакуйте в models\vosk\vosk-model-small-ru-0.22"
