$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$venvPath = Join-Path $projectRoot ".venv313"
$localPython = Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\python.exe"

if (Test-Path $localPython) {
    $pythonExe = $localPython
} else {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        throw "Python executable not found. Install 64-bit Python 3.13 first."
    }
    $pythonExe = $pythonCommand.Source
}

Write-Host "Using Python:" $pythonExe
& $pythonExe -c "import sys,struct; print(sys.version); print(f'{struct.calcsize(\"P\")*8}-bit')"

if (-not (Test-Path $venvPath)) {
    & $pythonExe -m venv $venvPath
}

$venvPython = Join-Path $venvPath "Scripts\python.exe"

& $venvPython -m pip install --upgrade pip setuptools wheel --default-timeout 120 --retries 10
& $venvPython -m pip install --only-binary=:all: --default-timeout 120 --retries 10 -r requirements.txt

Write-Host ""
Write-Host "Core dependencies installed successfully."
Write-Host "Activate with:"
Write-Host "  .\.venv313\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Optional extras for SHAP/XGBoost/LightGBM/CatBoost:"
Write-Host "  .\.venv313\Scripts\python.exe -m pip install --only-binary=:all: --default-timeout 120 --retries 10 -r requirements-optional.txt"
