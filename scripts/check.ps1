# One-command check on Windows (equivalent of `make check`, no make required).
#
#   powershell -ExecutionPolicy Bypass -File .\scripts\check.ps1
#
# On a fresh clone it creates .venv with Python >= 3.11 and installs the pinned
# dependencies from requirements.lock (needs internet once); afterwards it runs offline.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$venvPy = Join-Path $root ".venv\Scripts\python.exe"

function Find-BasePython {
    # Python >= 3.11: prefer the py launcher's 3.11, then any python/python3 on PATH
    $candidates = @(@("py", "-3.11"), @("python"), @("python3"), @("py", "-3"))
    foreach ($c in $candidates) {
        $exe = $c[0]
        $pre = @($c | Select-Object -Skip 1)
        if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
        $ok = & $exe @pre -c "import sys; print(int(sys.version_info >= (3, 11)))" 2>$null
        if ($LASTEXITCODE -eq 0 -and "$ok".Trim() -eq "1") { return ,$c }
    }
    return $null
}

Push-Location $root
try {
    if (-not (Test-Path $venvPy)) {
        $base = Find-BasePython
        if ($null -eq $base) {
            Write-Host "Python 3.11+ not found. Install it from https://www.python.org/downloads/ and run this script again." -ForegroundColor Red
            exit 1
        }
        Write-Host "==> creating .venv with: $($base -join ' ')"
        $exe = $base[0]; $pre = @($base | Select-Object -Skip 1)
        & $exe @pre -m venv .venv
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPy)) { Write-Host "could not create .venv" -ForegroundColor Red; exit 1 }
    }
    & $venvPy -c "import pytest" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "==> installing pinned dependencies from requirements.lock"
        & $venvPy -m pip install --disable-pip-version-check -q -r requirements.lock
        if ($LASTEXITCODE -ne 0) { Write-Host "pip install failed (internet is needed once for this step)" -ForegroundColor Red; exit 1 }
    }
    $env:PYTHONPATH = Join-Path $root "src"
    & $venvPy -m audit_env check
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
