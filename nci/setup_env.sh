#!/usr/bin/env bash
set -euo pipefail

# Run from /scratch/yo13/ps8396/Glue on Gadi.
module purge
module load pytorch/1.10.0

python3 -m venv .venvs/Glue --system-site-packages
# shellcheck disable=SC1091
source .venvs/Glue/bin/activate

python -m pip install --upgrade pip
python -m pip install pandas numpy tifffile scikit-learn pillow
python -m pip install torchvision==0.11.1 --no-deps

python - <<'PY'
import importlib.util
import torch

required = ["torch", "torchvision", "pandas", "numpy", "tifffile", "PIL", "sklearn"]
for name in required:
    print(f"{name}: {importlib.util.find_spec(name) is not None}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
PY
