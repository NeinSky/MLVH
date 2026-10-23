# Офлайн-установка CUDA-сборки torch из папки wheels\.
# Нужно, когда интернет медленный: файлы качаются вручную, ставим локально.
#
# Что положить в wheels\:
#   torch-2.6.0+cu124-cp313-cp313-win_amd64.whl        (~2.4 ГБ)
#   torchaudio-2.6.0+cu124-cp313-cp313-win_amd64.whl   (~4 МБ)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

$python = ".\.venv\Scripts\python.exe"
$wheels = Join-Path (Get-Location) "wheels"

$torch = Get-ChildItem $wheels -Filter "torch-2.6.0*cu124*cp313*win_amd64.whl" -ErrorAction SilentlyContinue
$audio = Get-ChildItem $wheels -Filter "torchaudio-2.6.0*cu124*cp313*win_amd64.whl" -ErrorAction SilentlyContinue

if (-not $torch -or -not $audio) {
    Write-Host "В папке wheels\ нет нужных файлов. Ожидались:"
    Write-Host "  torch-2.6.0+cu124-cp313-cp313-win_amd64.whl"
    Write-Host "  torchaudio-2.6.0+cu124-cp313-cp313-win_amd64.whl"
    exit 1
}

Write-Host "Ставлю torch из локальных файлов (без интернета)..."
& $python -m pip install --no-index --find-links $wheels --force-reinstall --no-deps `
    "torch==2.6.0+cu124" "torchaudio==2.6.0+cu124"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Установка не удалась."
    exit 1
}

Write-Host ""
Write-Host "Проверяю CUDA..."
& $python (Join-Path (Get-Location) "tools\check_cuda.py")
