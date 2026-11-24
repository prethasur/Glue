#!/usr/bin/env bash
# Shared body of the training PBS jobs. Called by run_train_*.pbs with MODE set.
# Required environment (pass with `qsub -v`):
#   GLUE_ROOT     repository checkout on Gadi
#   SPLIT_DIR     directory with the split CSVs (train_seen.csv, ...)
#   DATA_ROOT     root used to resolve band paths (portable subset root)
#   OUTPUT_DIR    run output directory (created if missing)
#   PRETRAINED    true | false   (false = random frozen backbone, a negative control)
# Optional: PRETRAINED_WEIGHTS (local ResNet-50 state_dict; compute nodes have no internet),
#   SEED, EPOCHS, BATCH_SIZE, LR, LORA_RANK, LORA_ALPHA, LORA_TARGET_LAYERS, NUM_WORKERS,
#   VENV (default: $GLUE_ROOT/.venvs/Glue), TORCH_MODULE (default: pytorch/1.10.0).
set -euo pipefail

: "${MODE:?MODE must be set by the calling PBS script}"
: "${GLUE_ROOT:?set GLUE_ROOT via qsub -v}"
: "${SPLIT_DIR:?set SPLIT_DIR via qsub -v}"
: "${DATA_ROOT:?set DATA_ROOT via qsub -v}"
: "${OUTPUT_DIR:?set OUTPUT_DIR via qsub -v}"
: "${PRETRAINED:?set PRETRAINED=true or PRETRAINED=false via qsub -v}"

SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LORA_RANK="${LORA_RANK:-4}"
LORA_ALPHA="${LORA_ALPHA:-8.0}"
LORA_TARGET_LAYERS="${LORA_TARGET_LAYERS:-layer4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VENV="${VENV:-${GLUE_ROOT}/.venvs/Glue}"
TORCH_MODULE="${TORCH_MODULE:-pytorch/1.10.0}"

cd "${GLUE_ROOT}"
mkdir -p "${OUTPUT_DIR}"

module purge
module load "${TORCH_MODULE}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

extra=()
if [[ -n "${PRETRAINED_WEIGHTS:-}" ]]; then
  extra+=(--pretrained_weights "${PRETRAINED_WEIGHTS}")
fi

python -m src.training.train_baseline \
  --mode "${MODE}" \
  --split_dir "${SPLIT_DIR}" \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --pretrained "${PRETRAINED}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_target_layers "${LORA_TARGET_LAYERS}" \
  --device cuda \
  --num_workers "${NUM_WORKERS}" \
  "${extra[@]}"
