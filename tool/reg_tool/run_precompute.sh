#!/usr/bin/env bash
#
# run_precompute.sh — 控制 precompute_anchor_patch_v2.py
#
# 用法:
#   bash tool/reg_tool/run_precompute.sh              # 默认参数全量跑
#   bash tool/reg_tool/run_precompute.sh --skip-train # 仅推理
#   bash tool/reg_tool/run_precompute.sh --skip-infer # 仅训练
#   bash tool/reg_tool/run_precompute.sh --num-anchors 512 --k-patch 128 --seed 42

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/precompute_anchor_patch_v2.py"

BASE_DIR="${BASE_DIR:-/data/shared/测试数据/h5}"
RAW_H5="${RAW_H5:-${BASE_DIR}/field1031_irregular.h5}"
REGULAR_H5="${REGULAR_H5:-${BASE_DIR}/field1031_label.h5}"
TARGET_H5="${TARGET_H5:-${BASE_DIR}/field1031_mask.h5}"
GROUP_KEY="${GROUP_KEY:-1551}"
GROUP_KEY="${GROUP_KEY:-1551}"
PATCH_DIR="${PATCH_DIR:-${BASE_DIR}/anchor_patch_v2}"

# ── 训练参数 ──────────────────────────────────────────────────
NUM_ANCHORS="${NUM_ANCHORS:-7896}"
ANCHOR_STRIDE="${ANCHOR_STRIDE:-128}"
K_PATCH="${K_PATCH:-256}"
TOP_L="${TOP_L:-512}"
NUM_QUERY="${NUM_QUERY:-8}"
BETA="${BETA:-0.3}"
SEED="${SEED:-0}"
METRIC_WEIGHTS="${METRIC_WEIGHTS:-1.0,1.0,0.5,0.5}"
TRAIN_ANCHOR_SELECTOR="${TRAIN_ANCHOR_SELECTOR:-value_based_anchor_sampling}"
TRAIN_TRUSTED_SOURCE="${TRAIN_TRUSTED_SOURCE:-all}"

# ── 推理参数 ──────────────────────────────────────────────────
BLOCK_DIVISORS="${BLOCK_DIVISORS:-6,21,7,5}"
STRIDE_DIVISORS="${STRIDE_DIVISORS:-6,21,7,5}"
QUERY_MASK_MODE="${QUERY_MASK_MODE:-regular_true}"
MAX_QUERY_PER_PATCH="${MAX_QUERY_PER_PATCH:-128}"
INFER_GPU_DEVICE="${INFER_GPU_DEVICE:-cuda:5}"
GPU_QUERY_CHUNK_SIZE="${GPU_QUERY_CHUNK_SIZE:-128}"

# ── 拆分逗号分隔的多值参数 ────────────────────────────────────
IFS=',' read -ra _block_divs  <<< "${BLOCK_DIVISORS}"
IFS=',' read -ra _stride_divs <<< "${STRIDE_DIVISORS}"

# ── 构建参数列表 ──────────────────────────────────────────────
ARGS=(
    --base-dir              "${BASE_DIR}"
    --raw-h5                "${RAW_H5}"
    --target-h5             "${TARGET_H5}"
    --regular-h5            "${REGULAR_H5}"
    --group-key             "${GROUP_KEY}"
    --patch-dir             "${PATCH_DIR}"
    --anchor-stride         "${ANCHOR_STRIDE}"
    --k-patch               "${K_PATCH}"
    --num-query             "${NUM_QUERY}"
    --beta                  "${BETA}"
    --seed                  "${SEED}"
    --metric-weights        "${METRIC_WEIGHTS}"
    --train-anchor-selector "${TRAIN_ANCHOR_SELECTOR}"
    --train-trusted-source  "${TRAIN_TRUSTED_SOURCE}"
    --block-divisors        "${_block_divs[@]}"
    --stride-divisors       "${_stride_divs[@]}"
    --query-mask-mode       "${QUERY_MASK_MODE}"
    --max-query-per-patch   "${MAX_QUERY_PER_PATCH}"
    --gpu-query-chunk-size  "${GPU_QUERY_CHUNK_SIZE}"
    --infer-gpu-device      "${INFER_GPU_DEVICE}"
    --no-infer-use-gpu
    --skip-infer
)

[[ -n "${NUM_ANCHORS}" ]] && ARGS+=(--num-anchors "${NUM_ANCHORS}")
[[ -n "${TOP_L}" ]]       && ARGS+=(--top-l "${TOP_L}")

# 透传用户额外参数
ARGS+=("$@")

echo "============================================"
echo "PY_SCRIPT = ${PY_SCRIPT}"
echo "BASE_DIR  = ${BASE_DIR}"
echo "PATCH_DIR = ${PATCH_DIR}"
echo "CMD_ARGS  = ${ARGS[*]}"
echo "============================================"

python "${PY_SCRIPT}" "${ARGS[@]}"
