#!/usr/bin/env bash
#
# run_core.sh — 控制 core.py 的 shell 入口 (对齐 precompute_anchor_patch_v2)
#
# 用法:
#   bash tool/reg_tool/run_core.sh                              # anchor_patch 模式 (推荐使用 --enable-auto-params)
#   bash tool/reg_tool/run_core.sh binning                      # binning 模式
#   bash tool/reg_tool/run_core.sh binning+csg                  # binning → csg 链式
#   bash tool/reg_tool/run_core.sh binning+crg                  # binning → crg 链式
#   bash tool/reg_tool/run_core.sh kdtree                       # kdtree 模式
#   bash tool/reg_tool/run_core.sh csg                          # csg 模式
#   bash tool/reg_tool/run_core.sh crg                          # crg 模式
#   bash tool/reg_tool/run_core.sh anchor_patch --enable-auto-params --infer-use-gpu --train-knn-use-gpu

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/core.py"

# ── 从命令行取 mode ──────────────────────────────────────────
MODE="${MODE:-anchor_patch}"
if [[ $# -gt 0 && "$1" != -* ]]; then
    MODE="$1"
    shift
fi

# ── 路径默认值 ────────────────────────────────────────────────
BASE_DIR="${BASE_DIR:-/data/shared/测试数据/h5}"
RAW_H5="${RAW_H5:-${BASE_DIR}/field1031_irregular.h5}"
REGULAR_H5="${REGULAR_H5:-${BASE_DIR}/field1031_label.h5}"
TARGET_H5="${TARGET_H5:-${BASE_DIR}/field1031_mask.h5}"
GROUP_KEY="${GROUP_KEY:-1551}"
PATCH_DIR="${PATCH_DIR:-}"

# ── 训练参数 ──────────────────────────────────────────────────
NUM_ANCHORS="${NUM_ANCHORS:-}"
K_PATCH="${K_PATCH:-256}"
TOP_L="${TOP_L:-512}"
NUM_QUERY="${NUM_QUERY:-8}"
BETA="${BETA:-0.3}"
SEED="${SEED:-0}"
METRIC_WEIGHTS="${METRIC_WEIGHTS:-1.0,1.0,0.5,0.5}"
TRAIN_ANCHOR_SELECTOR="${TRAIN_ANCHOR_SELECTOR:-value_based_anchor_sampling}"
TRAIN_TRUSTED_SOURCE="${TRAIN_TRUSTED_SOURCE:-all}"
ANCHOR_STRIDE="${ANCHOR_STRIDE:-128}"
POOL_SIZE="${POOL_SIZE:-}"

# ── 推理 (4D block) 参数 ──────────────────────────────────────
BLOCK_DIVISORS="${BLOCK_DIVISORS:-6,21,7,5}"
STRIDE_DIVISORS="${STRIDE_DIVISORS:-6,21,7,5}"
QUERY_MASK_MODE="${QUERY_MASK_MODE:-regular_true}"
MAX_QUERY_PER_PATCH="${MAX_QUERY_PER_PATCH:-128}"
GPU_QUERY_CHUNK_SIZE="${GPU_QUERY_CHUNK_SIZE:-128}"

# ── GPU 参数 ──────────────────────────────────────────────────
# 推理 context 选择距离计算上 GPU (torch batch matmul)
INFER_USE_GPU="${INFER_USE_GPU:-true}"
INFER_GPU_DEVICE="${INFER_GPU_DEVICE:-cuda:5}"

# 训练锚点选择: value_based_anchor_sampling 的 kNN 距离矩阵上 GPU
# 数据集小时(trusted≤VALUE_KNN_FULL_MATRIX_MAX_N)一次性全矩阵，大时分批
TRAIN_KNN_USE_GPU="${TRAIN_KNN_USE_GPU:-true}"
# 贪心抑制循环也上 GPU (收益较小，默认关；run_precompute.sh 默认开)
TRAIN_SUPPRESSION_USE_GPU="${TRAIN_SUPPRESSION_USE_GPU:-true}"
# GPU kNN 分批行数 (距离矩阵 [candidates, trusted] 显存受限时)
VALUE_KNN_GPU_BATCH_ROWS="${VALUE_KNN_GPU_BATCH_ROWS:-512}"
# 可信任点数阈值: ≤此值一次性全矩阵 GPU 计算，>此值分批
VALUE_KNN_FULL_MATRIX_MAX_N="${VALUE_KNN_FULL_MATRIX_MAX_N:-4096}"

# ── 开关 ──────────────────────────────────────────────────────
ENABLE_AUTO_PARAMS="${ENABLE_AUTO_PARAMS:-fasle}"
AUTO_PARAMS_ANCHOR_STRIDE="${AUTO_PARAMS_ANCHOR_STRIDE:-128}"
RAW_KEY_AGGREGATE="${RAW_KEY_AGGREGATE:-mean}"
GREEDY_FILL_UNCOVERED="${GREEDY_FILL_UNCOVERED:-true}"
SKIP_TRAIN="${SKIP_TRAIN:-false}"
SKIP_INFER="${SKIP_INFER:-true}"

# ── 构建 ARGS ─────────────────────────────────────────────────
ARGS=(
    "${MODE}"
    --base_dir              "${BASE_DIR}"
    --raw_h5                "${RAW_H5}"
    --regular_h5            "${REGULAR_H5}"
    --target_h5             "${TARGET_H5}"
    --group_key             "${GROUP_KEY}"
    --k_patch               "${K_PATCH}"
    --top_l                 "${TOP_L}"
    --num_query             "${NUM_QUERY}"
    --beta                  "${BETA}"
    --seed                  "${SEED}"
    --metric_weights        "${METRIC_WEIGHTS}"
    --train-anchor-selector "${TRAIN_ANCHOR_SELECTOR}"
    --train-trusted-source  "${TRAIN_TRUSTED_SOURCE}"
    --anchor-stride         "${ANCHOR_STRIDE}"
    --query-mask-mode       "${QUERY_MASK_MODE}"
    --max-query-per-patch   "${MAX_QUERY_PER_PATCH}"
    --gpu-query-chunk-size  "${GPU_QUERY_CHUNK_SIZE}"
    --infer-gpu-device      "${INFER_GPU_DEVICE}"
    --auto-params-anchor-stride "${AUTO_PARAMS_ANCHOR_STRIDE}"
    --raw_key_aggregate     "${RAW_KEY_AGGREGATE}"
    --value-knn-gpu-batch-rows "${VALUE_KNN_GPU_BATCH_ROWS}"
    --value-knn-full-matrix-max-n "${VALUE_KNN_FULL_MATRIX_MAX_N}"
)

# 可选数值参数 (非空才传递)
[[ -n "${NUM_ANCHORS}" ]]      && ARGS+=(--num_anchors "${NUM_ANCHORS}")
[[ -n "${PATCH_DIR}" ]]        && ARGS+=(--patch-dir "${PATCH_DIR}")
[[ -n "${POOL_SIZE}" ]]        && ARGS+=(--pool-size "${POOL_SIZE}")
[[ -n "${BLOCK_DIVISORS}" ]]   && ARGS+=(--block-divisors $(echo "${BLOCK_DIVISORS}" | tr ',' ' '))
[[ -n "${STRIDE_DIVISORS}" ]]  && ARGS+=(--stride-divisors $(echo "${STRIDE_DIVISORS}" | tr ',' ' '))

# 可选布尔标记
[[ "${ENABLE_AUTO_PARAMS}" == "true" ]]            && ARGS+=(--enable-auto-params)
[[ "${INFER_USE_GPU}" == "true" ]]                 && ARGS+=(--infer-use-gpu)
[[ "${TRAIN_KNN_USE_GPU}" == "true" ]]             && ARGS+=(--train-knn-use-gpu)
[[ "${TRAIN_SUPPRESSION_USE_GPU}" == "true" ]]     && ARGS+=(--train-suppression-use-gpu)
[[ "${GREEDY_FILL_UNCOVERED}" == "true" ]]         && ARGS+=(--greedy-fill-uncovered)
[[ "${SKIP_TRAIN}" == "true" ]]                    && ARGS+=(--skip-train)
[[ "${SKIP_INFER}" == "true" ]]                    && ARGS+=(--skip-infer)

# 透传用户额外参数
ARGS+=("$@")

echo "============================================"
echo "PY_SCRIPT = ${PY_SCRIPT}"
echo "MODE      = ${MODE}"
echo "BASE_DIR  = ${BASE_DIR}"
echo "CMD_ARGS  = ${ARGS[*]}"
echo "============================================"

python "${PY_SCRIPT}" "${ARGS[@]}"
