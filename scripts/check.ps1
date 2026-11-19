# Windows equivalent of `make check` (no make required).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $root "src"
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
Push-Location $root
try { & $py -m audit_env check; exit $LASTEXITCODE } finally { Pop-Location }
