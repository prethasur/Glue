#!/usr/bin/env bash
set -euo pipefail

# Run from <REPO_ROOT>.
# NCI module names can change. Adjust these lines for your project allocation.
module purge || true
# module load python3/3.11.0
# module load cuda/12.1.1
# module load openmpi/4.1.5

ENV_NAME="Glue"
ENV_DIR=".venvs/${ENV_NAME}"

if command -v conda >/dev/null 2>&1; then
  if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -y -n "${ENV_NAME}" python=3.11
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
else
  python3 -m venv "${ENV_DIR}"
  # shellcheck disable=SC1091
  source "${ENV_DIR}/bin/activate"
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python - <<'PY'
import importlib.util
import torch

required = ["torch", "torchvision", "pandas", "pyarrow", "tifffile", "PIL"]
optional = ["rasterio"]
for name in required:
    print(f"{name}: {importlib.util.find_spec(name) is not None}")
for name in optional:
    print(f"{name} optional: {importlib.util.find_spec(name) is not None}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
PY
