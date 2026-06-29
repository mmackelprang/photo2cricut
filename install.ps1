<#
install.ps1 -- set up photo2cricut in a local virtual environment and smoke-test it.

Usage:
  ./install.ps1            # creates .venv, installs, runs a smoke test
  ./install.ps1 -Dev       # also installs pytest + cairosvg and runs tests

If you hit an execution-policy error, run this once in the same PowerShell session:
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#>
[CmdletBinding()]
param([switch]$Dev)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
Write-Host ">> Using interpreter: $(& $python --version)"

if (-not (Test-Path ".venv")) {
    Write-Host ">> Creating virtual environment (.venv)"
    & $python -m venv .venv
}

$activate = Join-Path ".venv" "Scripts/Activate.ps1"
. $activate

python -m pip install --upgrade pip | Out-Null

Write-Host ">> Installing photo2cricut"
if ($Dev) { pip install -e ".[dev]" } else { pip install -e . }

Write-Host ">> Smoke test: generate test image -> convert -> validate"
New-Item -ItemType Directory -Force -Path "examples" | Out-Null
photo2cricut-makeimg examples/test_portrait.jpg
photo2cricut examples/test_portrait.jpg examples/test_portrait.svg --method xdog --width-in 8
photo2cricut-validate examples/test_portrait.svg

if ($Dev) {
    Write-Host ">> Running test suite"
    pytest -q
}

Write-Host ""
Write-Host ">> Done. Activate the environment with:  .\.venv\Scripts\Activate.ps1"
Write-Host ">> Then convert your own photo:          photo2cricut my_photo.jpg out.svg"
