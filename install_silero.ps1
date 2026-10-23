# Установка локального движка Silero (опционально).
# Даёт правильные русские ударения; torch уже стоит вместе с Chatterbox.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Не найдено окружение .venv. Сначала: python -m venv .venv"
    exit 1
}

Write-Host "Устанавливаю Silero..."
& $python -m pip install -r requirements-silero.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "Установка не удалась."
    exit 1
}

Write-Host ""
Write-Host "Проверяю движок (модель ~139 МБ скачается при первом синтезе)..."
& $python tools\check_engines.py
