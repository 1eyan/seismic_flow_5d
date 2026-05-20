#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/Segy2H5.py"

# ---- SEG-Y config preset ----
SEGY_CONFIG="${SEGY_CONFIG:-field1031}"

python "${PY_SCRIPT}" \
    --irr /data/shared/测试数据/raw5d_data_updated/raw5d_data1104.sgy \
    --label /data/shared/测试数据/reg_pku_1031/reg_pku_1030/reg5dbin_label1031.sgy \
    --mask /data/shared/测试数据/mask_from_label.sgy \
    --dataset-name field1031 \
    --config "${SEGY_CONFIG}" \
    --mode fixed
