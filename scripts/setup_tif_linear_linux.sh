#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <conda-env-name> [torch-index-url]"
  exit 1
fi

ENV_NAME="$1"
TORCH_INDEX_URL="${2:-}"

eval "$(conda shell.bash hook)"
conda create -n "${ENV_NAME}" python=3.11 -y
conda activate "${ENV_NAME}"

cd "${REPO_ROOT}"

python -m pip install --upgrade pip

if [[ -n "${TORCH_INDEX_URL}" ]]; then
  python -m pip install torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"
else
  echo "Skip torch install. Pass a CUDA wheel index URL as the second argument if needed."
  echo "Example: bash $0 ${ENV_NAME} https://download.pytorch.org/whl/cu124"
fi

python -m pip install -r requirements.txt
python -m pip install pillow tifffile omegaconf torchmetrics "numpy<2" "scipy>=1.13,<1.14"
python -m pip install -e .

python - <<'PY'
import importlib
mods = ["PIL", "tifffile", "omegaconf", "torchmetrics", "numpy", "scipy", "dinov3"]
for mod in mods:
    importlib.import_module(mod)
try:
    importlib.import_module("torch")
    importlib.import_module("torchvision")
    print("PyTorch check passed.")
except Exception as exc:
    print(f"PyTorch check skipped or failed: {exc}")
print("Environment check passed.")
PY
