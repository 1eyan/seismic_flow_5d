# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Seismic 5D interpolation using Flow Matching with a Transformer (DiT) backbone. Given irregularly-sampled seismic traces (shot/receiver coordinates + time), the model reconstructs missing traces on a regular grid. The "query-context" paradigm: observed traces serve as "context" to condition generation of "query" (missing) traces.

## Shell Scripts (primary entry points)

- **Training**: `bash run_train.sh` — multi-GPU by default via `accelerate launch`. Override via env vars:
  `H5_FILE=... DATASET_NEIGHBORS_TRAIN=... NUM_GPUS=1 bash run_train.sh`
- **Inference**: `bash run_infer.sh` — multi-GPU via `torchrun`. Override checkpoint/data paths via env vars.
- **Data prep (SEG-Y to H5)**: 
  - Single/multi-file: `python tool/convert_tool/batch_segy2h5.py` (uses `segyPairs` from `dataset_config.py`)
  - Triple-file (irr+mask+label → 3 H5s): `python tool/convert_tool/Segy2H5.py --irr ... --mask ... --label ... --dataset-name field1031`
    - Auto-creates `h5/` subdirectory under the common root of the 3 SEG-Y files
    - Outputs: `<dataset_name>_irregular.h5`, `<dataset_name>_mask.h5`, `<dataset_name>_label.h5`
- **Preprocessing (grid/patches)**: `bash tool/reg_tool/run_precompute.sh` then `bash tool/reg_tool/run_core.sh`

## Key Architecture

### Data flow
```
SEG-Y → [tool/convert_tool] → H5 files → [tool/reg_tool] → .npz patch indices
                                                    ↓
                                          train.py (DatasetH5_all_queryctx)
                                                    ↓
                                          FlowMatchingModel(SeisDiTRopeV2)
                                                    ↓
                                          infer_cli.py → filled SEG-Y
```

### Model (`model/seisdit_trace_axis.py`)

**SeisDiTRopeV2** is a DiT-style U-Net where trace dimension is the sequence axis for attention:
- **Tokenizer**: Two 2D convs process `[data, mask]` channels, then fuse via 1x1 conv
- **Encoder**: ResBlock stack with AdaTimeModulation, time-axis downsampling
- **Bottleneck**: Stack of `DiTBlockTrace` — Transformer blocks with `TraceAxisAttention2D` (global attention over traces) + RoPE positional encoding
- **Decoder**: ResBlock stack with skip connections from encoder, time-axis upsampling
- **Conditioning**: 4D coordinates `(sx, sy, rx, ry)` condition the model two ways:
  1. RoPE encoding on attention keys/queries (via `SegmentedRoPEExpCached` in `model/rope.py`)
  2. adaLN modulation — geometry is Fourier-embedded through `Geomlp` and added to time embedding

### Flow Matching (`fpm.py` + `transport/`)

- Default: **Linear path** with **velocity prediction** (`x_t = t*x1 + (1-t)*x0`; model predicts `u_t = x1 - x0`)
- `Transport.training_losses()` computes MSE between model output and the true vector field
- `FlowMatchingModel.sample()` runs ODE/SDE integration (via `torchdiffeq` or Euler-Maruyama) conditioned on context traces and coordinates
- Path types: Linear (`ICPlan`), GVP (`GVPCPlan`), VP (`VPCPlan`)
- Prediction types: `VELOCITY` (default), `NOISE`, `SCORE`

### Dataset (`dataset/dataset.py`)

**DatasetH5_all_queryctx** has two modes:
- **train_pool**: Random anchor query → `diverse_topk` selects diverse context traces around it
- **infer_query_context**: Precomputed grid-query indices + observation context indices for inference

### Coordinate normalization (`utils/coord_utils.py`)

Spatial coords are normalized to `[-1, 1]` via global min-max. When `--use_phys_omega` is enabled, RoPE base frequencies are auto-computed from physical grid step sizes using Nyquist sampling theory.

### SEG-Y I/O (`utils/segy_utils.py`)

Reads/writes SEG-Y traces. Builds lookup tables from trace headers (shot_line, shot_stake, recv_line, recv_stake) to file positions. `fill_segy()` writes predicted traces into a mask SEG-Y to produce the completed output.

### SEG-Y Configuration (`config/segy_config.py` + `config/segy_config.yaml`)

All SEG-Y header byte positions are centralized in `config/segy_config.yaml`, organized as named presets per dataset. **To switch datasets, select the preset before importing other modules:**

```python
from config import segy_config
segy_config.load_config("field1031")   # or "sw06", "segc3"
```

Or set the env var `SEGY_CONFIG=preset_name` before launching any script:
```bash
SEGY_CONFIG=segc3 python tool/convert_tool/batch_segy2h5.py
```

**Available presets:**
- `field1031` — standard SEG-Y REV 1 (shot_line=17, shot_stake=21, recv_line=61, recv_stake=65, coords at 73/77/81/85)
- `sw06` — alternative format (shot_line=221, shot_stake=225, recv_stake=229)
- `segc3` — self_computed mode, coordinate-only (only shot_x/y, rec_x/y at 73/77/81/85); line/stake numbers are computed from scaled coordinates, not read from headers

**Modules affected:** `utils/segy_utils.py`, `tool/convert_tool/Segy2H5.py`, `infer_cli.py`, `dataset/dataset.py` — all import `KEY_COLUMNS`, `SEGY_BYTE_POS`/`BYTE_POS`, `COORD_COL`, `SORT_KEYS` from `config.segy_config`.

## Package Structure

```
├── train.py / infer_cli.py    — CLI entry points (relative imports from root package)
├── infer.py / fpm.py          — inference engine + flow matching wrapper
│
├── utils/                     — utility modules (coord_utils, sampler_utils, segy_utils)
├── dataset/                   — PyTorch Dataset classes (DatasetH5_all_queryctx, DatasetH5Interp)
├── config/                    — SEG-Y byte positions + dataset argument parser
├── model/                     — Transformer/DiT architecture + RoPE
├── transport/                 — Flow matching transport + ODE/SDE integrators
└── tool/                      — offline data-processing tools (convert_tool, reg_tool)
```

Relative imports use `..` to cross package boundaries (e.g. `dataset/dataset.py` imports from `..utils.sampler_utils` and `..config.segy_config`).

## Key training hyperparameters

- `--train_num_query` (16-32): query traces per sample
- `--trace_ps` (128): total traces per patch (query + context)
- `--time_ps` (1256): time samples per trace
- `--patch_beta` (0.3): diversity weight for context selection
- `--geom_mode` (relative/source/receiver): which coordinate pairs to use for conditioning
- `--use_phys_omega` (true/false): auto-compute RoPE base frequencies from grid geometry
