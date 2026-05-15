#!/usr/bin/env python3
"""Auto-compute anchor-patch hyperparameters from observation system (raw + regular H5).

Reads raw (irregular) and regular (grid) H5 files, extracts coordinate statistics,
and computes sensible defaults for all precompute parameters.

Usage:
    python reg_tool/auto_params.py [--base-dir PATH] [--group-key KEY]
    python reg_tool/auto_params.py --raw-h5 raw.h5 --regular-h5 reg.h5  # explicit paths

Output format is directly usable as shell env-var exports for run_precompute.sh.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from h5py import File


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def typical_grid_step(arr: np.ndarray) -> Tuple[Optional[float], np.ndarray]:
    u = np.sort(np.unique(arr))
    if u.size < 2:
        return None, u
    d = np.diff(u)
    d = d[d > 1e-9]
    if d.size == 0:
        return None, u
    return float(np.median(d)), u


def compute_observation_stats(
    raw_group, regular_group
) -> Dict[str, Any]:
    """Extract all coordinate statistics from raw and regular H5 groups."""
    sx_raw = raw_group["sx"][:].astype(np.float64)
    sy_raw = raw_group["sy"][:].astype(np.float64)
    rx_raw = raw_group["rx"][:].astype(np.float64)
    ry_raw = raw_group["ry"][:].astype(np.float64)

    sx_reg = regular_group["sx"][:].astype(np.float64)
    sy_reg = regular_group["sy"][:].astype(np.float64)
    rx_reg = regular_group["rx"][:].astype(np.float64)
    ry_reg = regular_group["ry"][:].astype(np.float64)

    n_obs = int(sx_raw.shape[0])
    n_grid = int(sx_reg.shape[0])

    dsx, sx_u = typical_grid_step(sx_reg)
    dsy, sy_u = typical_grid_step(sy_reg)
    drx, rx_u = typical_grid_step(rx_reg)
    dry, ry_u = typical_grid_step(ry_reg)

    nsx = len(sx_u)
    nsy = len(sy_u)
    nrx = len(rx_u)
    nry = len(ry_u)

    range_sx = sx_u.max() - sx_u.min()
    range_sy = sy_u.max() - sy_u.min()
    range_rx = rx_u.max() - rx_u.min()
    range_ry = ry_u.max() - ry_u.min()

    coverage = n_obs / n_grid if n_grid > 0 else 1.0

    has_mask = "mask" in regular_group
    if has_mask:
        mask = regular_group["mask"][:].astype(bool).reshape(-1)
        n_trusted = int(mask.sum())
        n_missing = int((~mask).sum())
    else:
        n_trusted = n_grid
        n_missing = 0

    return {
        "n_obs": n_obs,
        "n_grid": n_grid,
        "n_trusted": n_trusted,
        "n_missing": n_missing,
        "coverage_ratio": float(coverage),
        "nsx": nsx,
        "nsy": nsy,
        "nrx": nrx,
        "nry": nry,
        "grid_step_sx": dsx,
        "grid_step_sy": dsy,
        "grid_step_rx": drx,
        "grid_step_ry": dry,
        "range_sx": float(range_sx),
        "range_sy": float(range_sy),
        "range_rx": float(range_rx),
        "range_ry": float(range_ry),
        "sx_min": float(sx_u.min()),
        "sx_max": float(sx_u.max()),
        "sy_min": float(sy_u.min()),
        "sy_max": float(sy_u.max()),
        "rx_min": float(rx_u.min()),
        "rx_max": float(rx_u.max()),
        "ry_min": float(ry_u.min()),
        "ry_max": float(ry_u.max()),
    }


def compute_anchor_params(stats: Dict[str, Any]) -> Dict[str, Any]:
    n_obs = stats["n_obs"]
    coverage = stats["coverage_ratio"]

    num_anchors = max(128, int(np.sqrt(n_obs) * 2.5)) if n_obs > 0 else 2048
    anchor_stride = max(1, n_obs // num_anchors)
    num_anchors = n_obs // anchor_stride

    return {
        "num_anchors": int(num_anchors),
        "anchor_stride": int(anchor_stride),
    }


def compute_patch_params(stats: Dict[str, Any]) -> Dict[str, Any]:
    n_grid = stats["n_grid"]
    nsx, nsy, nrx, nry = stats["nsx"], stats["nsy"], stats["nrx"], stats["nry"]
    coverage = stats["coverage_ratio"]

    avg_density = n_grid / max(1, nsx * nsy * nrx * nry) if nsx * nsy * nrx * nry > 0 else 1.0

    k_patch = int(np.clip(32 + 192 * coverage, 32, 512))
    top_l = 2 * k_patch
    num_query = max(1, min(k_patch // 4, 32))

    return {
        "k_patch": k_patch,
        "top_l": top_l,
        "num_query": num_query,
    }


def compute_block_params(stats: Dict[str, Any], target_block_volume: int = 400) -> Dict[str, Any]:
    nsx, nsy, nrx, nry = stats["nsx"], stats["nsy"], stats["nrx"], stats["nry"]
    n_grid = stats["n_grid"]
    dims_4d = (nsx, nsy, nrx, nry)

    total_cells = max(1, nsx * nsy * nrx * nry)
    avg_per_cell = n_grid / total_cells

    target_cells = int(target_block_volume) if avg_per_cell >= 1.0 else int(max(256, target_block_volume / max(avg_per_cell, 0.01)))

    vol_per_dim = target_cells ** (1.0 / 4.0)

    block_divisors = []
    for d in dims_4d:
        if d <= 0:
            block_divisors.append(1)
            continue
        div = max(1, int(round(d / max(1.0, vol_per_dim))))
        block_divisors.append(div)

    block_size = tuple(max(1, int(d) // int(b)) for d, b in zip(dims_4d, block_divisors))
    total_block_cells = int(np.prod(block_size)) if all(b > 0 for b in block_size) else 0

    stride_divisors = tuple(max(1, b // 2) for b in block_divisors)

    max_query_per_patch = min(256, max(32, total_block_cells))

    return {
        "grid_shape_4d": list(dims_4d),
        "block_divisors": list(block_divisors),
        "stride_divisors": list(stride_divisors),
        "block_size": list(block_size),
        "block_cells": total_block_cells,
        "max_query_per_patch": max_query_per_patch,
    }


def compute_metric_weights(stats: Dict[str, Any]) -> Tuple[float, float, float, float]:
    rsx = max(stats["range_sx"], 1e-6)
    rsy = max(stats["range_sy"], 1e-6)
    rrx = max(stats["range_rx"], 1e-6)
    rry = max(stats["range_ry"], 1e-6)

    w_sx = 1.0
    w_sy = rsx / rsy if rsy > 0 else 1.0
    w_rx = rsx / rrx * 0.5 if rrx > 0 else 0.5
    w_ry = rsx / rry * 0.5 if rry > 0 else 0.5

    w_sum = w_sx + w_sy + w_rx + w_ry
    if w_sum > 0:
        w_sx, w_sy, w_rx, w_ry = [v * 4.0 / w_sum for v in (w_sx, w_sy, w_rx, w_ry)]

    return (round(w_sx, 3), round(w_sy, 3), round(w_rx, 3), round(w_ry, 3))


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def format_shell_exports(params: Dict[str, Any]) -> str:
    """Format parameters as shell env-var exports for run_precompute.sh."""
    lines = []

    grid_shape = params["grid_shape_4d"]
    lines.append("# Grid dimensions: nsx nsy nrx nry")
    lines.append(f"#   {grid_shape[0]} x {grid_shape[1]} x {grid_shape[2]} x {grid_shape[3]}")

    lines.append("")
    lines.append("# ── Anchor / training ──")
    lines.append(f"NUM_ANCHORS={params['num_anchors']}")
    lines.append(f"ANCHOR_STRIDE={params['anchor_stride']}")
    lines.append(f"K_PATCH={params['k_patch']}")
    lines.append(f"TOP_L={params['top_l']}")
    lines.append(f"NUM_QUERY={params['num_query']}")
    lines.append(f"METRIC_WEIGHTS={','.join(str(w) for w in params['metric_weights'])}")

    lines.append("")
    lines.append("# ── Inference block / stride ──")
    bd = params["block_divisors"]
    sd = params["stride_divisors"]
    bs = params.get("block_size", [])
    bc = params.get("block_cells", "?")
    lines.append(f"BLOCK_DIVISORS={','.join(str(b) for b in bd)}")
    lines.append(f"STRIDE_DIVISORS={','.join(str(s) for s in sd)}")
    lines.append(f"MAX_QUERY_PER_PATCH={params['max_query_per_patch']}")
    lines.append(f"# block_size={','.join(str(b) for b in bs)}  cells={bc}")

    lines.append("")
    lines.append("# ── Observation summary ──")
    lines.append(f"# N_obs={params['n_obs']}  N_grid={params['n_grid']}")
    lines.append(f"# N_trusted={params['n_trusted']}  N_missing={params['n_missing']}")
    lines.append(f"# Coverage ratio={params['coverage_ratio']:.3f}")
    lines.append("")
    lines.append(f"# --- end of auto_params ---")

    return "\n".join(lines)


def format_json(params: Dict[str, Any]) -> str:
    return json.dumps(params, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Auto-compute anchor-patch precompute hyperparameters from observation system."
    )
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--raw-h5", type=str, default=None)
    parser.add_argument("--regular-h5", type=str, default=None)
    parser.add_argument("--group-key", type=str, default="1551")
    parser.add_argument(
        "--output-format",
        choices=["shell", "json", "both"],
        default="shell",
        help="output format (default: shell)",
    )
    parser.add_argument("--target-block-volume", type=int, default=400,
                        help="target grid cells per block (default: 400)")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    default_base = Path(__file__).resolve().parent.parent.parent / "seis_flow_data12V2" / "h5" / "dongfang"
    base_dir = Path(args.base_dir).expanduser().resolve() if args.base_dir else default_base
    raw_h5 = Path(args.raw_h5) if args.raw_h5 else base_dir / "raw5d_data1104.h5"
    regular_h5 = Path(args.regular_h5) if args.regular_h5 else base_dir / "reg5dbin_label1031.h5"

    if not raw_h5.exists():
        sys.exit(f"raw H5 not found: {raw_h5}")
    if not regular_h5.exists():
        sys.exit(f"regular H5 not found: {regular_h5}")

    print(f"raw_h5:    {raw_h5}", file=sys.stderr)
    print(f"regular_h5: {regular_h5}", file=sys.stderr)

    with File(raw_h5, "r") as f_raw, File(regular_h5, "r") as f_reg:
        raw_group = f_raw[args.group_key]
        regular_group = f_reg[args.group_key]

        stats = compute_observation_stats(raw_group, regular_group)

    anchor_params = compute_anchor_params(stats)
    patch_params = compute_patch_params(stats)
    block_params = compute_block_params(stats, target_block_volume=args.target_block_volume)
    metric_weights = compute_metric_weights(stats)

    params = {
        **stats,
        **anchor_params,
        **patch_params,
        **block_params,
        "metric_weights": metric_weights,
        "target_block_volume": args.target_block_volume,
    }

    if args.output_format in ("shell", "both"):
        print(format_shell_exports(params))
    if args.output_format in ("json", "both"):
        print(format_json(params))


if __name__ == "__main__":
    main()
