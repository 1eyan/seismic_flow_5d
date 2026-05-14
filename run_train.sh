#!/bin/bash
# FPM V3 training — queryctx mode (trace_axis model)
# Usage:
#   bash run_train.sh
#   H5_FILE=/path/to/irregular.h5 DATASET_NEIGHBORS_TRAIN=/path/to/train_pool.npz bash run_train.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# ---- GPU ----
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_GPUS="${NUM_GPUS:-2}"

# ---- Training ----
MODEL_NAME="${MODEL_NAME:-trace_axis}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-200}"
SEED="${SEED:-515}"
DATA_TYPE="${DATA_TYPE:-df_field1031_5d}"

# ---- Model ----
GEOM_MODE="${GEOM_MODE:-relative}"
USE_MISSING_EMBEDDING="${USE_MISSING_EMBEDDING:-false}"
USE_PHYS_OMEGA="${USE_PHYS_OMEGA:-true}"

# ---- Data ----
H5_FILE="${H5_FILE:-${ROOT_DIR}/data/raw5d_data.h5}"
H5_FILE_REGULAR="${H5_FILE_REGULAR:-${ROOT_DIR}/data/reg5dbin_label.h5}"
DATASET_NEIGHBORS_TRAIN="${DATASET_NEIGHBORS_TRAIN:-${ROOT_DIR}/data/train_pool_idx_2d.npz}"

# ---- Queryctx ----
TRAIN_NUM_QUERY="${TRAIN_NUM_QUERY:-32}"
TRAIN_CONTEXT_SIZE="${TRAIN_CONTEXT_SIZE:-}"
PATCH_BETA="${PATCH_BETA:-0.3}"
FORCE_ANCHOR_QUERY="${FORCE_ANCHOR_QUERY:-false}"
TRACE_SORT_KEYS="${TRACE_SORT_KEYS:-rx,ry,sx,sy}"

TIME_PS="${TIME_PS:-1256}"
TRACE_PS="${TRACE_PS:-128}"
USE_P_SCALE="${USE_P_SCALE:-false}"

echo "======================================"
echo "FPM V3 Training — queryctx + trace_axis"
echo "GPU: ${CUDA_VISIBLE_DEVICES}  |  Num: ${NUM_GPUS}"
echo "Model: trace_axis  |  Batch: ${BATCH_SIZE}  |  LR: ${LR}  |  Epochs: ${EPOCHS}"
echo "H5 irregular: ${H5_FILE}"
echo "H5 regular:   ${H5_FILE_REGULAR}"
echo "neighbors:    ${DATASET_NEIGHBORS_TRAIN}"
echo "num_query: ${TRAIN_NUM_QUERY}  |  beta: ${PATCH_BETA}  |  sort: ${TRACE_SORT_KEYS}"
echo "phys_omega: ${USE_PHYS_OMEGA}"
echo "======================================"

cmd_args=(
  --model_name "${MODEL_NAME}"
  --batch_size "${BATCH_SIZE}"
  --lr "${LR}"
  --epochs "${EPOCHS}"
  --seed "${SEED}"
  --data_type "${DATA_TYPE}"
  --geom_mode "${GEOM_MODE}"
  --use_missing_embedding "${USE_MISSING_EMBEDDING}"
  --use_phys_omega "${USE_PHYS_OMEGA}"
  --use_p_scale "${USE_P_SCALE}"
  --time_ps "${TIME_PS}"
  --trace_ps "${TRACE_PS}"
  --h5File "${H5_FILE}"
  --h5File_regular "${H5_FILE_REGULAR}"
  --dataset_neighbors_train "${DATASET_NEIGHBORS_TRAIN}"
  --train_num_query "${TRAIN_NUM_QUERY}"
  --patch_beta "${PATCH_BETA}"
  --force_anchor_query "${FORCE_ANCHOR_QUERY}"
  --trace_sort_keys "${TRACE_SORT_KEYS}"
  --dataset_type queryctx
)

if [[ -n "${TRAIN_CONTEXT_SIZE}" ]]; then
  cmd_args+=(--train_context_size "${TRAIN_CONTEXT_SIZE}")
fi

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  accelerate launch --config_file "${ROOT_DIR}/accelerate_config.yaml" \
    "${ROOT_DIR}/train.py" "${cmd_args[@]}"
else
  python "${ROOT_DIR}/train.py" "${cmd_args[@]}"
fi

echo "Training done!"
