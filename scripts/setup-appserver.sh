#!/usr/bin/env bash
# Set up the photo2coloringbook [gpu] backend on a CUDA host (e.g. appserver).
# Handles hosts with no system pip / no python3-venv ensurepip / no sudo:
# bootstraps pip via get-pip.py into a --without-pip venv.
#
# Usage:  scripts/setup-appserver.sh [venv_dir]   (default: .venv-gpu)
set -euo pipefail
cd "$(dirname "$0")/.."
ENVDIR="${1:-.venv-gpu}"

python3 -m venv --without-pip "$ENVDIR"
# shellcheck disable=SC1091
. "$ENVDIR/bin/activate"
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python /tmp/get-pip.py

# Torch with the CUDA 12.1 wheels (NOT on PyPI -- needs the dedicated index).
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install controlnet_aux==0.0.10 rembg==2.0.76 onnxruntime==1.27.0
pip install -e '.[coloringbook]'   # base deps + reportlab/Pillow (book.py needs reportlab)

echo ">> Verifying CUDA + GPU..."
python -c "import torch; print('cuda', torch.cuda.is_available(), \
  torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
echo ">> Done. Generate a book with:"
echo ">>   photo2coloringbook <photos_dir> book.pdf --backend contour --bg auto"
