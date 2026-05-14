#!/usr/bin/env bash
#
# run_core.sh — 控制 core.py 的 shell 入口
#
# 用法:
#   bash reg_tool/run_core.sh                           # anchor_patch 模式
#   bash reg_tool/run_core.sh binning                   # binning 模式
#   bash reg_tool/run_core.sh kdtree                    # kdtree 模式
#   bash reg_tool/run_core.sh csg                       # csg 模式
#   bash reg_tool/run_core.sh crg                       # crg 模式
#   bash reg_tool/run_core.sh anchor_patch --num-anchors 1024 --k-patch 128

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/core.py"

# ── 从命令行取 mode（第一个非 option 参数），否则用默认值 ──────────
MODE="${MODE:-anchor_patch}"
if [[ $# -gt 0 && "$1" != -* ]]; then
    MODE="$1"
    shift
fi

# ── 路径默认值 ────────────────────────────────────────────────
BASE_DIR="${BASE_DIR:-${SCRIPT_DIR}/../h5/dongfang}"
RAW_H5="${RAW_H5:-${BASE_DIR}/raw5d_data1104.h5}"
REGULAR_H5="${REGULAR_H5:-${BASE_DIR}/reg5dbin_label1031.h5}"
TARGET_H5="${TARGET_H5:-${BASE_DIR}/reg5dbin_label1031_binning.h5}"
GROUP_KEY="${GROUP_KEY:-1551}"

# ── anchor_patch 参数 ─────────────────────────────────────────
NUM_ANCHORS="${NUM_ANCHORS:-2048}"
K_PATCH="${K_PATCH:-64}"
TOP_L="${TOP_L:-128}"
NUM_QUERY="${NUM_QUERY:-8}"
BETA="${BETA:-0.3}"
SEED="${SEED:-0}"
METRIC_WEIGHTS="${METRIC_WEIGHTS:-1,1,0.5,0.5}"
GRID_NX="${GRID_NX:-0}"
GRID_NY="${GRID_NY:-0}"
BLOCK_BX="${BLOCK_BX:-16}"
BLOCK_BY="${BLOCK_BY:-16}"
STRIDE_SX="${STRIDE_SX:-8}"
STRIDE_SY="${STRIDE_SY:-8}"

# ── 构建参数列表 ──────────────────────────────────────────────
ARGS=(
    "${MODE}"
    --base_dir      "${BASE_DIR}"
    --raw_h5        "${RAW_H5}"
    --regular_h5    "${REGULAR_H5}"
    --target_h5     "${TARGET_H5}"
    --group_key     "${GROUP_KEY}"
    --num_anchors   "${NUM_ANCHORS}"
    --k_patch       "${K_PATCH}"
    --top_l         "${TOP_L}"
    --num_query     "${NUM_QUERY}"
    --beta          "${BETA}"
    --seed          "${SEED}"
    --metric_weights "${METRIC_WEIGHTS}"
    --grid_nx       "${GRID_NX}"
    --grid_ny       "${GRID_NY}"
    --block_bx      "${BLOCK_BX}"
    --block_by      "${BLOCK_BY}"
    --stride_sx     "${STRIDE_SX}"
    --stride_sy     "${STRIDE_SY}"
)

# 透传用户额外参数
ARGS+=("$@")

echo "============================================"
echo "PY_SCRIPT = ${PY_SCRIPT}"
echo "MODE      = ${MODE}"
echo "BASE_DIR  = ${BASE_DIR}"
echo "CMD_ARGS  = ${ARGS[*]}"
echo "============================================"

python "${PY_SCRIPT}" "${ARGS[@]}"
