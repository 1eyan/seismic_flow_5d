#!/usr/bin/env bash
# Generate irregular/mask/label data pairs from a SEG-Y file for 5D training.
# Byte positions are loaded from segy_config.yaml (preset selected via SEGY_CONFIG).
# No separate JSON file needed.
#
# Usage:
#   bash run_mk_irr_mask.sh
#   INPUT_SGY=/path/to/file.sgy MISSING_RATIO=0.4 bash run_mk_irr_mask.sh
#   SEGY_CONFIG=field1031 INPUT_SGY=... bash run_mk_irr_mask.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# ---- Required ----
INPUT_SGY="${INPUT_SGY:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

# ---- Config preset (must exist in config/segy_config.yaml) ----
SEGY_CONFIG="${SEGY_CONFIG:-a002}"

# ---- Mask params ----
MISSING_RATIO="${MISSING_RATIO:-0.3}"
MASK_DOMAIN="${MASK_DOMAIN:-receiver}"   # receiver | shot | both | 4d

if [[ -z "${INPUT_SGY}" ]]; then
  echo "[ERROR] INPUT_SGY is required."
  echo "Usage: INPUT_SGY=/path/to/data.sgy bash run_mk_irr_mask.sh"
  echo ""
  echo "Optional env vars:"
  echo "  SEGY_CONFIG   preset name (default: a002)"
  echo "  MISSING_RATIO fraction to drop (default: 0.3)"
  echo "  MASK_DOMAIN   receiver | shot | both | 4d (default: receiver)"
  exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  # Default: same dir as input, in a train/ subfolder
  OUTPUT_DIR="$(dirname "${INPUT_SGY}")/train"
fi

echo "============================================================"
echo "mk_irr_mask — generate irregular/mask/label from SEG-Y"
echo "input:       ${INPUT_SGY}"
echo "output_dir:  ${OUTPUT_DIR}"
echo "segy_config: ${SEGY_CONFIG}"
echo "miss_ratio:  ${MISSING_RATIO}"
echo "mask_domain: ${MASK_DOMAIN}"
echo "============================================================"

export INPUT_SGY
export OUTPUT_DIR
export MISSING_RATIO
export SEGY_CONFIG
export MASK_DOMAIN

python "${ROOT_DIR}/utils/mk_irr_mask.py"

echo ""
echo "Done! Outputs in ${OUTPUT_DIR}"
echo "  - *new.sgy/h5       (label — full data)"
echo "  - *irr_*.sgy/h5     (irregular — kept traces only)"
echo "  - *mask_*.sgy/h5    (mask — missing traces zeroed)"
echo "  - *shot_xy_cut.dat  (shot coords for obs system QC)"
echo "  - *rcvs_xy_cut.dat  (receiver coords for obs system QC)"
