# ============================================================
# build_staging.ps1
# ------------------------------------------------------------
# Готовит dist\app\ для упаковки в установщик Inno Setup:
#   * берёт ТОЛЬКО Consumer-сборку (QtCTReconstructionConsumer.exe);
#   * исключает Producer (QtCTReconstruction.exe);
#   * подтягивает Qt-runtime (DLL + platforms/qml/... через windeployqt);
#   * подтягивает CUDA-DLL (cudart, cufft) из CUDA Toolkit;
#   * чистит build-мусор (.pdb/.exp/.lib/.ilk и CMake-каталоги).
#
# Renaming Consumer.exe -> CTReconstruction.exe сам скрипт НЕ делает —
# это происходит на этапе установки (см. installer.iss, секция [Files]).
# Так dist\app\ остаётся узнаваемым и пригодным для повторного использования.
#
# Использование:
#   pwsh -ExecutionPolicy Bypass -File dist\build_staging.ps1
#   pwsh -ExecutionPolicy Bypass -File dist\build_staging.ps1 -Rebuild
# ============================================================
[CmdletBinding()]
param(
    # Сначала вызвать cmake --build (иначе используем уже собранный exe).
    [switch]$Rebuild,

    # Путь к CMake-build-папке Release (можно переопределить).
    [string]$BuildDir = "$PSScriptRoot\..\out\build\release",

    # Каталог установки Qt (для windeployqt).
    [string]$QtBin = "C:\Qt\6.11.0\msvc2022_64\bin",

    # Каталог CUDA Toolkit с DLL.
    [string]$CudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin"
)

$ErrorActionPreference = 'Stop'

# --- 0. Опциональная пересборка ---------------------------------------------
if ($Rebuild) {
    Write-Host "==> cmake --build (Release, Consumer-only)..." -ForegroundColor Cyan
    & cmake --build $BuildDir --config Release --target QtCTReconstructionConsumer
    if ($LASTEXITCODE -ne 0) { throw "CMake build failed" }
}

# --- 1. Подготовка пустого staging-каталога ---------------------------------
$Staging = Join-Path $PSScriptRoot 'app'
Write-Host "==> Cleaning staging: $Staging" -ForegroundColor Cyan
if (Test-Path $Staging) { Remove-Item $Staging -Recurse -Force }
New-Item $Staging -ItemType Directory | Out-Null

# --- 2. Копируем только Consumer.exe (Producer оставляем за бортом) ---------
$ConsumerExe = Join-Path $BuildDir 'QtCTReconstructionConsumer.exe'
if (-not (Test-Path $ConsumerExe)) {
    throw "Consumer .exe не найден: $ConsumerExe`nЗапусти с -Rebuild или собери проект вручную."
}
Write-Host "==> Copying Consumer.exe" -ForegroundColor Cyan
Copy-Item $ConsumerExe $Staging

# --- 3. windeployqt: разворачиваем Qt-runtime вокруг Consumer.exe ----------
$WinDeployQt = Join-Path $QtBin 'windeployqt.exe'
if (-not (Test-Path $WinDeployQt)) {
    throw "windeployqt не найден: $WinDeployQt"
}
$QmlSrc = Join-Path $PSScriptRoot '..\QtCTReconstruction'
Write-Host "==> windeployqt --release" -ForegroundColor Cyan
& $WinDeployQt `
    --release `
    --qmldir $QmlSrc `
    --no-translations `
    --no-system-d3d-compiler `
    --no-compiler-runtime `
    (Join-Path $Staging 'QtCTReconstructionConsumer.exe')
if ($LASTEXITCODE -ne 0) { throw "windeployqt failed" }

# --- 4. CUDA Runtime DLLs (cudart, cufft) -----------------------------------
if (Test-Path $CudaBin) {
    Write-Host "==> Copying CUDA runtime DLLs" -ForegroundColor Cyan
    Get-ChildItem $CudaBin -Filter 'cudart64_*.dll' | Copy-Item -Destination $Staging
    Get-ChildItem $CudaBin -Filter 'cufft64_*.dll'  | Copy-Item -Destination $Staging
} else {
    Write-Warning "CUDA bin не найден ($CudaBin) — CUDA-бэкенды у пользователя работать НЕ будут."
}

# --- 5. Иконка рядом с .exe (для setWindowIcon в рантайме) ------------------
$IconPng = Join-Path $PSScriptRoot '..\resources\icon.png'
if (Test-Path $IconPng) {
    Copy-Item $IconPng $Staging
}

# --- 6. Чистим build-мусор, если он зачем-то попал --------------------------
Get-ChildItem $Staging -Recurse -Include '*.pdb','*.exp','*.lib','*.ilk' -ErrorAction Ignore |
    Remove-Item -Force

# --- 7. Удаляем Producer .exe, если случайно скопировался -------------------
$ProducerExe = Join-Path $Staging 'QtCTReconstruction.exe'
if (Test-Path $ProducerExe) {
    Write-Host "==> Removing Producer.exe (Consumer-only build)" -ForegroundColor Cyan
    Remove-Item $ProducerExe -Force
}

# --- 8. Итоговая статистика -------------------------------------------------
$size = (Get-ChildItem $Staging -Recurse | Measure-Object Length -Sum).Sum / 1MB
$files = (Get-ChildItem $Staging -Recurse -File).Count
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Staging готов: $Staging" -ForegroundColor Green
Write-Host ("  Файлов: {0}, размер: {1:N1} МБ" -f $files, $size) -ForegroundColor Green
Write-Host ""
Write-Host "Следующий шаг — собрать установщик:" -ForegroundColor Yellow
Write-Host '  & "C:\Program Files (x86)\Inno Setup 6\iscc.exe" dist\installer.iss' -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Green
