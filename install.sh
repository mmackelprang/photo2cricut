#!/usr/bin/env bash
# install.sh -- set up photo2cricut in a local virtual environment and smoke-test it.
# Usage:  ./install.sh            (creates .venv, installs, runs a smoke test)
#         ./install.sh --dev      (also installs pytest + cairosvg and runs tests)
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
DEV=0
[[ "${1:-}" == "--dev" ]] && DEV=1

echo ">> Using interpreter: $($PYTHON --version 2>&1)"

if [[ ! -d .venv ]]; then
  echo ">> Creating virtual environment (.venv)"
  "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null

echo ">> Installing photo2cricut"
if [[ $DEV -eq 1 ]]; then
  pip install -e ".[dev]"
else
  pip install -e .
fi

echo ">> Smoke test: generate test image -> convert -> validate"
mkdir -p examples
photo2cricut-makeimg examples/test_portrait.jpg
photo2cricut examples/test_portrait.jpg examples/test_portrait.svg --method xdog --width-in 8
photo2cricut-validate examples/test_portrait.svg

if [[ $DEV -eq 1 ]]; then
  echo ">> Running test suite"
  pytest -q
fi

echo ""
echo ">> Done. Activate the environment with:  source .venv/bin/activate"
echo ">> Then convert your own photo:          photo2cricut my_photo.jpg out.svg"
