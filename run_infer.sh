#!/usr/bin/env bash
# FPM V3 inference — queryctx mode (trace_axis model)
# Usage:
#   bash run_infer.sh
#   CHECKPOINT=/path/to/model.pth MASK_SEGY=/path/to/mask.sgy bash run_infer.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# ---- GPU ----
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_GPUS="${NUM_GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-29502}"

# ---- Model checkpoint ----
CHECKPOINT="${CHECKPOINT:-/home/chengzhitong/5d_regular/seismic_transformer_5d/resultsFPM/trace_axis_datatype_df_field1031_5d_queryctx/checkpoints/model-5.pth}"

# ---- Data ----
H5_DIR="${H5_DIR:-/data/shared/测试数据/h5}"
H5_IRREGULAR="${H5_IRREGULAR:-${H5_DIR}/field1031_irregular.h5}"
H5_REGULAR="${H5_REGULAR:-${H5_DIR}/field1031_label.h5}"
H5_MASK="${H5_MASK:-${H5_DIR}/field1031_mask.h5}"
MASK_SEGY="${MASK_SEGY:-/data/shared/测试数据/mask_from_label.sgy}"
DATASET_NEIGHBORS_INFER="${DATASET_NEIGHBORS_INFER:-${H5_DIR}/patch_anchor_patch/infer_query_context.npz}"
LABEL_SEGY="${LABEL_SEGY:-}"

# ---- Output ----
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/gen_fill_results}"
OUTPUT_SEGY="${OUTPUT_SEGY:-${OUTPUT_DIR}/filled_missing.sgy}"
OUTPUT_RESIDUAL_SEGY="${OUTPUT_RESIDUAL_SEGY:-${OUTPUT_DIR}/residual.sgy}"

# ---- Inference params ----
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TIME_PS="${TIME_PS:-1256}"
TRACE_PS="${TRACE_PS:-128}"
HEADER_MODE="${HEADER_MODE:-fixed}"

# ---- Model params (must match training) ----
GEOM_MODE="${GEOM_MODE:-relative}"
USE_MISSING_EMBEDDING="${USE_MISSING_EMBEDDING:-false}"
USE_P_SCALE="${USE_P_SCALE:-false}"
USE_PHYS_OMEGA="${USE_PHYS_OMEGA:-true}"

SAMPLING_METHOD="${SAMPLING_METHOD:-ode}"
ODE_NUM_STEPS="${ODE_NUM_STEPS:-50}"
ODE_SAMPLING_METHOD="${ODE_SAMPLING_METHOD:-dopri5}"
ODE_ATOL="${ODE_ATOL:-1e-6}"
ODE_RTOL="${ODE_RTOL:-1e-3}"
SDE_NUM_STEPS="${SDE_NUM_STEPS:-250}"

VISUALIZE="${VISUALIZE:-false}"
VIS_BATCHES="${VIS_BATCHES:-0}"

mkdir -p "${OUTPUT_DIR}"

shared_args=(
  --checkpoint "${CHECKPOINT}"
  --h5_irregular "${H5_IRREGULAR}"
  --h5_regular "${H5_REGULAR}"
  --h5_mask "${H5_MASK}"
  --mask_path "${MASK_SEGY}"
  --dataset_neighbors_infer "${DATASET_NEIGHBORS_INFER}"
  --output_dir "${OUTPUT_DIR}"
  --output_segy "${OUTPUT_SEGY}"
  --output_residual_segy "${OUTPUT_RESIDUAL_SEGY}"
  --batch_size "${BATCH_SIZE}"
  --time_ps "${TIME_PS}"
  --trace_ps "${TRACE_PS}"
  --header_mode "${HEADER_MODE}"
  --geom_mode "${GEOM_MODE}"
  --use_missing_embedding "${USE_MISSING_EMBEDDING}"
  --use_p_scale "${USE_P_SCALE}"
  --use_phys_omega "${USE_PHYS_OMEGA}"
  --sampling_method "${SAMPLING_METHOD}"
  --ode_num_steps "${ODE_NUM_STEPS}"
  --ode_sampling_method "${ODE_SAMPLING_METHOD}"
  --ode_atol "${ODE_ATOL}"
  --ode_rtol "${ODE_RTOL}"
  --sde_num_steps "${SDE_NUM_STEPS}"
  --visualize "${VISUALIZE}"
  --vis_batches "${VIS_BATCHES}"
  --strict_fill
)

if [[ -n "${LABEL_SEGY}" ]]; then
  shared_args+=(--label_segy "${LABEL_SEGY}")
fi

echo "============================================================"
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  echo "FPM V3 Inference — queryctx (${NUM_GPUS} GPUs)"
else
  echo "FPM V3 Inference — queryctx (single GPU)"
fi
echo "checkpoint:    ${CHECKPOINT}"
echo "H5_DIR:        ${H5_DIR}"
echo "h5_irregular:  ${H5_IRREGULAR}"
echo "h5_regular:    ${H5_REGULAR}"
echo "h5_mask:       ${H5_MASK}"
echo "mask_segy:     ${MASK_SEGY}"
echo "neighbors_npz: ${DATASET_NEIGHBORS_INFER}"
echo "output_segy:   ${OUTPUT_SEGY}"
echo "device:        ${DEVICE}"
echo "batch_size:    ${BATCH_SIZE}"
echo "phys_omega:    ${USE_PHYS_OMEGA}"
echo "visualize:     ${VISUALIZE}"
echo "============================================================"

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    "${ROOT_DIR}/infer_cli.py" \
    --device "${DEVICE}" \
    "${shared_args[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/infer.stdout.log"
else
  python "${ROOT_DIR}/infer_cli.py" \
    --device "${DEVICE}" \
    "${shared_args[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/infer.stdout.log"
fi

echo "Inference done!"
