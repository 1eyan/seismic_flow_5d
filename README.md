# seismic_transformer_5d

Seismic 5D interpolation using Flow Matching with a DiT (Diffusion Transformer) backbone.

Given irregularly-sampled seismic traces (source/receiver coordinates + time), the model reconstructs missing traces on a regular grid. Uses a **query-context** paradigm: observed traces serve as "context" to condition the generation of "query" (missing/predicted) traces.

## Architecture

```
Input: [data(2D), mask(2D)]   ── Tokenizer (2× Conv2d + 1×1 Conv) ──→
                                
Encoder (ResBlock × AdaTimeModulation, time-axis downsampling) ──→

Bottleneck (DiTBlockTrace × N)                              
  ├─ TraceAxisAttention2D   (global attention over traces)
  ├─ SegmentedRoPEExpCached (4D coordinate encoding)        
  └─ adaLN_modulation       (geometry + time conditioning)   
                                
Decoder (ResBlock × AdaTimeModulation, time-axis upsampling + skip connections) ──→

FlowMatchingModel                                                       
  ├─ Path: ICPlan (Linear), GVP, or VP                                 
  ├─ Prediction: velocity (default), score, noise                      
  ├─ Training: MSE between model(xt, t, cond) and true ut              
  └─ Inference: ODE/SDE integration from noise → signal                
```

**Data flow:**

```
SEG-Y files  ──[Segy2H5]──→  H5 (irregular) + H5 (regular grid)
                                    │
                     ┌──────────────┼──────────────┐
                     ▼              ▼               ▼
               binning (core.py)   csg/crg        kdtree
                     │              │               │
                     ▼              ▼               ▼
              train_pool_idx_2d.npz   /   csg_train_pool.npz   /   kdtree neighbors
              infer_query_context.npz
                                    │
                     ┌──────────────┘
                     ▼
          DatasetH5_all_queryctx (two-H5 queryctx)
          DatasetH5Interp        (single-H5 interpolation)
                     │
                     ▼
               train.py   /   infer_cli.py
                     │
                     ▼
          SeisDiTRopeV2 → FlowMatchingModel
                     │
                     ▼
          filled SEG-Y (interpolation output)
```

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** `torch>=2.0 numpy>=1.21 h5py>=3.0 accelerate>=0.20 torchdiffeq>=0.2 timm>=0.9 einops>=0.7 matplotlib>=3.5 segyio>=1.9 tqdm>=4.60 tensorboard>=2.10 pyyaml>=6.0`

## Project Structure

```
├── run_train.sh                    # Training launch script (single/multi-GPU)
├── run_infer.sh                    # Inference launch script (single/multi-GPU)
├── train.py                        # Training entry point
├── infer_cli.py                    # Inference + SEG-Y fill entry point
├── infer.py                        # Inference engine (batching, flow_sample)
├── fpm.py                          # FlowMatchingModel wrapper
│
├── utils/                          # Utility modules
│   ├── coord_utils.py              # Coordinate normalization, RoPE omega computation
│   ├── sampler_utils.py            # diverse_topk, weighted_sqdist NumPy utilities
│   └── segy_utils.py               # SEG-Y I/O (read/write headers + data)
│
├── dataset/                        # Dataset classes
│   ├── dataset.py                  # DatasetH5_all_queryctx (primary dataset)
│   └── dataset_interp.py           # DatasetH5Interp (single-H5 interpolation)
│
├── config/                         # Configuration
│   ├── data_config.py              # Dataset argument parser
│   ├── segy_config.py              # SEG-Y byte position presets (YAML loader)
│   └── segy_config.yaml            # Byte position definitions per dataset
│
├── model/                          # Model architecture
│   ├── seisdit_trace_axis.py       # SeisDiTRopeV2 U-Net (~1980 lines)
│   └── rope.py                     # SegmentedRoPEExpCached positional encoding
│
├── transport/                      # Flow matching transport
│   ├── transport.py                # Transport class + Sampler
│   ├── path.py                     # ICPlan (Linear), GVPCPlan, VPCPlan
│   ├── integrators.py              # ODE/SDE integrators
│   └── utils.py                    # EasyDict, mean_flat
│
└── tool/                           # CLI tools
    ├── convert_tool/
    │   ├── Segy2H5.py              # Core SEG-Y → H5 converter
    │   ├── batch_segy2h5.py        # Parallel multi-file converter
    │   └── dataset_config.py       # SEG-Y pair configuration
    │
    └── reg_tool/
        ├── core.py                 # H5 I/O, binning, kdtree, csg/crg, anchor_patch
        ├── patch_sampler.py        # Anchor selection, diverse_topk, grid blocks
        ├── anchor_selector.py      # FPS, facility location, value-based anchor selection
        ├── precompute_anchor_patch_v2.py  # Production precompute CLI
        ├── auto_params.py          # Auto-compute hyperparameters from observation system
        ├── run_precompute.sh       # Shell entry for precompute_anchor_patch_v2.py
        └── run_core.sh             # Shell entry for core.py (multi-mode)
```

---

## Quick Start

### Step 1: SEG-Y → H5 conversion

```bash
# Triple-file mode (irregular + mask + label)
python tool/convert_tool/Segy2H5.py \
    --irr irregular.sgy \
    --mask mask.sgy \
    --label label.sgy \
    --dataset-name my_dataset

# Single/multi-file mode (from segyPairs in dataset_config.py)
python tool/convert_tool/batch_segy2h5.py --num-workers 4
```

**Outputs:** H5 files with groups containing `data` (traces), `sx/sy/rx/ry` (coordinates), `shot_line/shot_stake/recv_line/recv_stake` (keys), and optional `mask`.

### Step 2: Precomputation (patch index generation)

```bash
# Auto-compute hyperparameters from your observation system
python tool/reg_tool/auto_params.py \
    --raw-h5 my_irregular.h5 \
    --regular-h5 my_regular.h5 \
    --output-format both

# Run precomputation using auto-computed params
python tool/reg_tool/core.py anchor_patch \
    --raw_h5 my_irregular.h5 \
    --regular_h5 my_regular.h5 \
    --enable-auto-params

# Or use the shell wrapper (for precompute_anchor_patch_v2.py)
RAW_H5=my_irregular.h5 REGULAR_H5=my_regular.h5 \
    bash tool/reg_tool/run_precompute.sh
```

**Precomputation modes:**

| Mode | Command | Use Case |
|------|---------|----------|
| `anchor_patch` | `core.py anchor_patch` | Standard 2D anchor-based patch sampling |
| `binning` | `core.py binning` | Map irregular traces to regular grid, create mask |
| `binning+csg` | `core.py binning+csg` | Chain: binning → per-shot gather patches |
| `binning+crg` | `core.py binning+crg` | Chain: binning → per-receiver gather patches |
| `kdtree` | `core.py kdtree` | KDTree-based spatial neighborhood patches |
| `precompute_anchor_patch_v2.py` | `run_precompute.sh` | Production 4D block-based pipeline |

**Output files (in `patch/` or `patchV2/`):**

| File | Mode | Description |
|------|------|-------------|
| `train_pool_idx_2d.npz` | train | `pool_idx_2d` + `anchor_idx`: candidate trace pools per anchor |
| `infer_query_context.npz` | infer | `grid_query_idx_list` + `context_idx_list`: query/context index pairs |
| `csg_train_pool.npz` | csg | `pool_idx` (object array): one gather per row |
| `crg_train_pool.npz` | crg | Same for receiver gathers |
| `coord_norm_stats.npz` | all | Min/max/mean/std of observed and grid coordinates |

### Step 3: Training

```bash
# Multi-GPU (default: 2 GPUs)
H5_FILE=my_irregular.h5 \
H5_FILE_REGULAR=my_regular.h5 \
DATASET_NEIGHBORS_TRAIN=patch/train_pool_idx_2d.npz \
    bash run_train.sh

# Single GPU
NUM_GPUS=1 \
H5_FILE=my_irregular.h5 \
H5_FILE_REGULAR=my_regular.h5 \
DATASET_NEIGHBORS_TRAIN=patch/train_pool_idx_2d.npz \
    bash run_train.sh

# Or direct Python
python train.py \
    --h5File my_irregular.h5 \
    --h5File_regular my_regular.h5 \
    --dataset_neighbors_train patch/train_pool_idx_2d.npz \
    --batch_size 2 --epochs 200
```

**Key training hyperparameters (env vars or CLI args):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 2 | Per-GPU batch size |
| `LR` | 1e-4 | AdamW learning rate |
| `EPOCHS` | 200 | Total epochs |
| `TRAIN_NUM_QUERY` | 32 | Query traces per sample |
| `TRACE_PS` | 128 | Total traces per patch (Q + K) |
| `TIME_PS` | 1256 | Time samples per trace |
| `PATCH_BETA` | 0.3 | Diversity weight for context selection |
| `GEOM_MODE` | relative | Coordinate mode: `relative`, `source`, `receiver` |
| `USE_PHYS_OMEGA` | true | Auto-compute RoPE base from grid step |
| `PATH_TYPE` | Linear | Flow path: `Linear`, `GVP`, `VP` |
| `PREDICTION` | velocity | Prediction target: `velocity`, `score`, `noise` |

**Outputs:**
- `resultsFPM/checkpoints/model-{epoch}.pth` — model checkpoints
- `resultsFPM/logs/training_config.json` — full training configuration (for inference)
- `resultsFPM/logs/coord_stats.json`, `rope_frequency_config.json` — coordinate state
- TensorBoard logs in `resultsFPM/logs/`

### Step 4: Inference (SEG-Y fill)

```bash
# Multi-GPU (default: 2 GPUs)
CHECKPOINT=resultsFPM/checkpoints/model-20.pth \
H5_IRREGULAR=my_irregular.h5 \
H5_REGULAR=my_regular.h5 \
H5_MASK=binned_output.h5 \
MASK_SEGY=mask_template.sgy \
DATASET_NEIGHBORS_INFER=patch/infer_query_context.npz \
    bash run_infer.sh

# Single GPU
NUM_GPUS=1 \
CHECKPOINT=resultsFPM/checkpoints/model-20.pth \
    bash run_infer.sh
```

**Outputs:**
- `gen_fill_results/filled_missing.sgy` — filled SEG-Y (missing traces reconstructed)
- `gen_fill_results/residual.sgy` — residual vs label (if `--label_segy` provided)
- `gen_fill_results/filled_missing_keys.csv` / `unfilled_missing_keys.csv` — per-key status
- `gen_fill_results/summary.json` — overall statistics

---

## Dataset Classes

### DatasetH5_all_queryctx (`dataset/dataset.py`)

Primary query-context dataset. Uses two H5 files:
- **`h5File`**: irregular (observed) traces → context source
- **`h5File_regular`**: regular grid traces → query source at inference, coordinate stats

**Constructor:**
```python
DatasetH5_all_queryctx(
    h5File="data/irregular.h5",
    h5File_regular="data/regular.h5",
    dataset_neighbors="patch/train_pool_idx_2d.npz",   # train
    # dataset_neighbors="patch/infer_query_context.npz", # infer
    train=True,
    train_num_query=32,
    trace_ps=128,
    time_ps=1256,
    patch_beta=0.3,
    trace_sort_keys=("rx", "ry", "sx", "sy"),
)
```

**Two modes (auto-detected from npz):**
- **`train_pool`**: Load pool indices per anchor, online select query traces randomly + context via `diverse_topk`
- **`infer_query_context`**: Load precomputed query (grid) + context (observed) index pairs

**Per-sample normalization:** clip at 99.5th percentile of context traces, divide by threshold.

### DatasetH5Interp (`dataset/dataset_interp.py`)

Single-H5 interpolation dataset. Both query and context come from the same binned grid.

```python
DatasetH5Interp(
    h5File="data/binned_grid.h5",
    dataset_neighbors="patch/train_pool_idx_2d.npz",   # or csg_train_pool.npz
    train=True,
    train_num_query=32,
    trace_ps=128,
    time_ps=1256,
)
```

**Precomputation format compatibility:**
- `pool_idx_2d` (binning/anchors/kdtree)
- `pool_idx` + `pool_key` (csg/crg object arrays)
- `grid_query_idx_list` + `context_idx_list` (inference)

**Inference semantics:** Query = missing grid positions (mask==0), Context = observed positions (mask==1).

---

## Hyperparameter Auto-Computation

The script `tool/reg_tool/auto_params.py` automatically computes optimal hyperparameters from the raw+regular H5 files:

```bash
python tool/reg_tool/auto_params.py \
    --raw-h5 my_irregular.h5 \
    --regular-h5 my_regular.h5 \
    --output-format shell    # shell env exports (for run_precompute.sh)
    --output-format json     # JSON output
    --target-block-volume 400  # target grid cells per inference block
```

**Auto-computed parameters:**

| Parameter | Formula |
|-----------|---------|
| `NUM_ANCHORS` | `N_obs // ANCHOR_STRIDE` (stride default 128) |
| `K_PATCH` | `32 + 192 × coverage_ratio`, clamped [32, 512] |
| `TOP_L` | `2 × K_PATCH` |
| `NUM_QUERY` | `min(K_PATCH // 4, 32)` |
| `BLOCK_DIVISORS` | `dim_i / target_volume^(1/4)` (4D) |
| `STRIDE_DIVISORS` | `BLOCK_DIVISORS / 2` (50% overlap) |
| `METRIC_WEIGHTS` | Normalized `[1, range_sx/range_sy, 0.5·range_sx/range_rx, 0.5·range_sx/range_ry]` |

The `--enable-auto-params` flag in `core.py`'s `anchor_patch` mode applies these computations directly.

---

## SEG-Y Configuration

SEG-Y header byte positions are centralized in `config/segy_config.yaml`. Three presets:

| Preset | Usage | Description |
|--------|-------|-------------|
| `field1031` | Standard SEG-Y REV 1 | shot_line=17, shot_stake=21, recv_line=61, recv_stake=65, coords at 73/77/81/85 |
| `sw06` | Alternative format | shot_line=221, shot_stake=225, recv_stake=229 |
| `segc3` | Self-computed mode | Coordinate-only (73/77/81/85); line/stake numbers computed from scaled coords |

**Switching presets:**
```bash
SEGY_CONFIG=segc3 python train.py ...
SEGY_CONFIG=sw06 python tool/convert_tool/batch_segy2h5.py
```

```python
from config import segy_config
segy_config.load_config("field1031")
bp = segy_config.get_byte_pos()
```

**Exported constants** (after config load):
- `KEY_COLUMNS = ("shot_line", "shot_stake", "recv_line", "recv_stake")`
- `COORD_COL = {"sx": 0, "sy": 1, "rx": 2, "ry": 3}`
- `TRACE_SORT_KEYS = ("rx", "ry", "sx", "sy")`

---

## Coordinate Normalization & RoPE

### Normalization

Coordinates are normalized globally to `[-1, 1]` using per-axis min/max from the **regular grid** data:

```
coord_norm = 2 × (coord - min) / (max - min) - 1
```

This normalization is used for both:
1. **diverse_topk context selection** (in `dataset/dataset.py` / `utils/sampler_utils.py`)
2. **RoPE positional encoding** (in `model/rope.py`)

Stats saved as `coord_stats.json` during training and loaded during inference for consistency.

### Physical Omega (`--use_phys_omega`)

When enabled, RoPE base frequencies are computed from physical grid steps using Nyquist sampling theory:

```
lambda_phys_x = 2 × min(grid_step_sx, grid_step_rx)    # minimum resolvable wavelength
omega_x = π × Lx / lambda_phys_x                        # for normalized coords
rope_base = (omega_x + omega_y) / 2
```

**Coordinate modes (`--geom_mode`):**

| Mode | Input to RoPE | Description |
|------|---------------|-------------|
| `relative` | (Δx, Δy, mx, my, offset, azimuth) | Relative geometry between source and receiver |
| `source` | (sx, sy) | Source coordinates only |
| `receiver` | (rx, ry) | Receiver coordinates only |

---

## Flow Matching

The model follows the **Flow Matching** framework with a **Linear** path (default):

**Path:** `x_t = t·x₁ + (1-t)·x₀`

**Training:** Sample `t ~ U[0,1]`, compute `u_t = x₁ - x₀` (velocity), model predicts `u_t` given `(x_t, x_cond, t, coords)`. Loss = MSE.

**Inference:** Start from `x₀ ~ N(0, I)`, integrate ODE `dx/dt = v(x, t, c)` from `t=0` to `t=1`.

**Path types (`--path_type`):**

| Path | Description |
|------|-------------|
| `Linear` (ICPlan) | Linear interpolation (default) |
| `GVP` (GVPCPlan) | Trigonometric interpolant |
| `VP` (VPCPlan) | Variance-preserving path |

**Sampling (`--sampling_method`):**

| Method | Parameters | Description |
|--------|------------|-------------|
| `ode` | `--ode_num_steps 50` (dopri5) | Adaptive ODE solver via torchdiffeq |
| `sde` | `--sde_num_steps 250` (Euler) | Euler-Maruyama SDE integration |

---

## The `tool/reg_tool/core.py` Modes

`core.py` is the central preprocessing hub. It supports these modes:

| Mode | Description | Output |
|------|-------------|--------|
| `anchor_patch` | Anchor-based 2D patch sampling | `train_pool_idx_2d.npz`, `infer_query_context.npz` |
| `binning` | Map irregular traces to regular grid | Binned H5 + mask |
| `binning+csg` | Binning → per-shot-gather patches | `csg_train_pool.npz` (object arrays) |
| `binning+crg` | Binning → per-receiver-gather patches | `crg_train_pool.npz` |
| `kdtree` | KDTree greedy covering → neighbor patches | `train_pool_idx_2d.npz` |

**Key functions for import:**
- `generate_binning_keys(info_f)` — 4D integer keys from headers
- `binning(raw_info, regular_info)` — Align irregular to regular grid with mean aggregation
- `gather(info_f, mode)` — Group traces by shot (CSG) or receiver (CRG)
- `build_grid_index_map_4d_from_coord_grid(coord_grid)` — Sparse 4D logical index map
- `make_query_mask(mode, regular_mask, n_grid)` — Inference query mask from regular mask

**Shell wrappers:**

```bash
# Production 4D pipeline
bash tool/reg_tool/run_precompute.sh

# Multi-mode pipeline (anchor_patch, binning, csg, crg, kdtree)
bash tool/reg_tool/run_core.sh [mode] [extra_args...]
```

---

## Environment Variable Reference

### `run_train.sh`

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `0,1` | GPU device IDs |
| `NUM_GPUS` | `2` | Number of GPUs |
| `H5_FILE` | `data/raw5d_data.h5` | Irregular (observed) H5 |
| `H5_FILE_REGULAR` | `data/reg5dbin_label.h5` | Regular grid H5 |
| `DATASET_NEIGHBORS_TRAIN` | `data/train_pool_idx_2d.npz` | Training pool npz |
| `BATCH_SIZE` | `2` | Per-GPU batch size |
| `LR` | `1e-4` | Learning rate |
| `EPOCHS` | `200` | Training epochs |
| `TRAIN_NUM_QUERY` | `32` | Query traces per sample |
| `TRACE_PS` | `128` | Total traces per patch |
| `TIME_PS` | `1256` | Time samples per trace |
| `PATCH_BETA` | `0.3` | Diversity weight |
| `GEOM_MODE` | `relative` | Coordinate mode |
| `USE_PHYS_OMEGA` | `true` | Auto-compute RoPE base |

### `run_infer.sh`

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `2,3` | GPU device IDs |
| `NUM_GPUS` | `2` | Number of GPUs |
| `CHECKPOINT` | `results/checkpoints/model-20.pth` | Model checkpoint |
| `H5_IRREGULAR` | `data/raw5d_data.h5` | Irregular H5 |
| `H5_REGULAR` | `data/reg5dbin_label.h5` | Regular H5 |
| `H5_MASK` | `data/binning.h5` | Mask H5 (from binning) |
| `MASK_SEGY` | `data/mask.sgy` | Template SEG-Y (zero=missing) |
| `DATASET_NEIGHBORS_INFER` | `data/infer_query_context.npz` | Inference npz |
| `OUTPUT_DIR` | `gen_fill_results` | Output directory |
| `OUTPUT_SEGY` | `filled_missing.sgy` | Output filled SEG-Y |
| `BATCH_SIZE` | `6` | Inference batch size |
| `SAMPLING_METHOD` | `ode` | Sampling (ode/sde) |
| `ODE_NUM_STEPS` | `50` | ODE steps |
| `SDE_NUM_STEPS` | `250` | SDE steps |

### `tool/reg_tool/run_core.sh`

| Variable | Default | Description |
|----------|---------|-------------|
| `MODE` | `anchor_patch` | Processing mode |
| `NUM_ANCHORS` | `2048` | Number of anchor points |
| `K_PATCH` | `64` | Context/patch size |
| `TOP_L` | `128` | Local candidate size |
| `GRID_NX/NY` | `0` | 2D grid dimensions |

### `tool/reg_tool/run_precompute.sh`

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_ANCHORS` | `7896` | Number of anchor points |
| `K_PATCH` | `256` | Context/patch size |
| `TOP_L` | `512` | Local candidate size |
| `BLOCK_DIVISORS` | `6,21,7,5` | 4D block divisors |
| `STRIDE_DIVISORS` | `6,21,7,5` | 4D stride divisors |
