#!/usr/bin/env python3
"""Production script for ``anchor_patch_debug.ipynb`` — thin CLI over ``core.py``.

The heavy logic lives in ``core.py`` and ``patch_sampler.py``.  This file only
keeps the argument parser, lazy-dependency loading, and the ``main()``
orchestration loop.

Defaults mirror the notebook source:
- base dir: ``../h5/dongfang`` relative to this file
- patch dir: ``patchV2``
- train anchor selector: ``value_based_anchor_sampling``
- train trusted points: all raw observations
- infer query mask: regular ``mask == True``
- infer context filter: none
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple


# ── lazy imports (keep --help fast) ─────────────────────────────────


def load_runtime_dependencies() -> None:
    """Import heavy scientific dependencies after argparse has handled --help."""
    global np
    global File
    global generate_binning_keys
    global raw_obs_valid_mask_from_regular_trusted_mask
    global read_coord4, read_trace_data, read_regular_mask
    global resolve_block_tuple
    global build_grid_index_map_4d_from_coord_grid
    global make_query_mask
    global save_norm_stats
    global object_array, summarize_query_context
    global validate_outputs
    global normalize_coords
    global precompute_infer_patches_4d
    global precompute_train_patches_2d

    if "np" in globals():
        return

    import numpy as _np
    from h5py import File as _File

    try:
        from .core import (
            generate_binning_keys as _generate_binning_keys,
            raw_obs_valid_mask_from_regular_trusted_mask as _raw_obs_valid_mask,
            read_coord4 as _read_coord4,
            read_trace_data as _read_trace_data,
            read_regular_mask as _read_regular_mask,
            resolve_block_tuple as _resolve_block_tuple,
            build_grid_index_map_4d_from_coord_grid as _build_grid_index_map_4d,
            make_query_mask as _make_query_mask,
            save_norm_stats as _save_norm_stats,
            object_array as _object_array,
            summarize_query_context as _summarize_query_context,
            validate_outputs as _validate_outputs,
        )
        from .patch_sampler import (
            normalize_coords as _normalize_coords,
            precompute_infer_patches_4d as _precompute_infer_patches_4d,
            precompute_train_patches_2d as _precompute_train_patches_2d,
        )
    except ImportError:
        from core import (
            generate_binning_keys as _generate_binning_keys,
            raw_obs_valid_mask_from_regular_trusted_mask as _raw_obs_valid_mask,
            read_coord4 as _read_coord4,
            read_trace_data as _read_trace_data,
            read_regular_mask as _read_regular_mask,
            resolve_block_tuple as _resolve_block_tuple,
            build_grid_index_map_4d_from_coord_grid as _build_grid_index_map_4d,
            make_query_mask as _make_query_mask,
            save_norm_stats as _save_norm_stats,
            object_array as _object_array,
            summarize_query_context as _summarize_query_context,
            validate_outputs as _validate_outputs,
        )
        from patch_sampler import (
            normalize_coords as _normalize_coords,
            precompute_infer_patches_4d as _precompute_infer_patches_4d,
            precompute_train_patches_2d as _precompute_train_patches_2d,
        )

    np = _np
    File = _File
    generate_binning_keys = _generate_binning_keys
    raw_obs_valid_mask_from_regular_trusted_mask = _raw_obs_valid_mask
    read_coord4 = _read_coord4
    read_trace_data = _read_trace_data
    read_regular_mask = _read_regular_mask
    resolve_block_tuple = _resolve_block_tuple
    build_grid_index_map_4d_from_coord_grid = _build_grid_index_map_4d
    make_query_mask = _make_query_mask
    save_norm_stats = _save_norm_stats
    object_array = _object_array
    summarize_query_context = _summarize_query_context
    validate_outputs = _validate_outputs
    normalize_coords = _normalize_coords
    precompute_infer_patches_4d = _precompute_infer_patches_4d
    precompute_train_patches_2d = _precompute_train_patches_2d


# ── CLI helpers ─────────────────────────────────────────────────────


def default_base_dir() -> Path:
    return (Path(__file__).resolve().parent / ".." / "h5" / "dongfang").resolve()


def parse_metric_weights(text: str) -> Tuple[float, float, float, float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(values) != 4:
        raise argparse.ArgumentTypeError("metric weights must be four comma-separated numbers")
    return tuple(values)  # type: ignore[return-value]


def as_path(path: Optional[str], default: Path) -> Path:
    return Path(path).expanduser().resolve() if path else default.resolve()


# ── argument parser ─────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute the notebook V2 train pool and infer query/context patch files."
    )
    parser.add_argument("--base-dir", type=str, default=str(default_base_dir()))
    parser.add_argument("--raw-h5", type=str, default=None)
    parser.add_argument("--target-h5", type=str, default=None)
    parser.add_argument("--regular-h5", type=str, default=None)
    parser.add_argument("--group-key", type=str, default="1551")
    parser.add_argument("--patch-dir", type=str, default=None)
    parser.add_argument("--regular-mask-key", type=str, default="mask")

    parser.add_argument("--num-anchors", type=int, default=None)
    parser.add_argument("--anchor-stride", type=int, default=128)
    parser.add_argument("--k-patch", type=int, default=256)
    parser.add_argument("--top-l", type=int, default=None)
    parser.add_argument("--num-query", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metric-weights", type=parse_metric_weights, default=(1.0, 1.0, 0.5, 0.5))
    parser.add_argument(
        "--train-anchor-selector",
        choices=[
            "farthest_point_sampling",
            "facility_location_anchor_sampling",
            "value_based_anchor_sampling",
        ],
        default="value_based_anchor_sampling",
    )
    parser.add_argument(
        "--train-trusted-source",
        choices=["all", "regular_mask"],
        default="all",
        help="Notebook default is all; regular_mask uses raw rows whose 4D keys map to regular mask=True.",
    )
    parser.add_argument("--value-local-top-l", type=int, default=None)
    parser.add_argument("--value-suppression", choices=["subtractive", "multiplicative"], default="subtractive")
    parser.add_argument("--value-suppression-lambda", type=float, default=1.0)
    parser.add_argument("--value-score-tol", type=float, default=0.0)
    parser.add_argument("--value-knn-gpu-batch-rows", type=int, default=512)
    parser.add_argument("--value-knn-gpu-device", type=str, default="cuda:0")
    parser.add_argument("--value-knn-full-matrix-max-n", type=int, default=4096)
    parser.add_argument("--train-knn-use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-suppression-use-gpu", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--block-size", type=int, nargs=4, default=None)
    parser.add_argument("--stride", type=int, nargs=4, default=None)
    parser.add_argument("--block-divisors", type=int, nargs=4, default=(6, 21, 7, 5))
    parser.add_argument("--stride-divisors", type=int, nargs=4, default=(6, 21, 7, 5))
    parser.add_argument("--on-grid-collision", choices=["raise", "last"], default="raise")
    parser.add_argument(
        "--query-mask-mode",
        choices=["regular_true", "regular_false", "all", "none"],
        default="regular_true",
        help="Notebook default is regular_true because it passes regular mask.astype(bool).",
    )
    parser.add_argument(
        "--infer-obs-valid-source",
        choices=["none", "regular_mask"],
        default="none",
        help="Notebook default is none.",
    )
    parser.add_argument("--infer-top-l", type=int, default=None)
    parser.add_argument("--max-query-per-patch", type=int, default=64)
    parser.add_argument("--gpu-query-chunk-size", type=int, default=64)
    parser.add_argument("--infer-gpu-device", type=str, default="cuda:0")
    parser.add_argument("--infer-use-gpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-full-query-coverage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--greedy-fill-uncovered", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-infer", action="store_true")
    parser.add_argument("--save-legacy-anchor-files", action="store_true")
    parser.add_argument("--save-grid-index-map", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--summary-json", type=str, default=None)
    return parser


# ── main ────────────────────────────────────────────────────────────


def main() -> None:
    args = build_arg_parser().parse_args()
    load_runtime_dependencies()

    base_dir = as_path(args.base_dir, default_base_dir())
    raw_h5 = as_path(args.raw_h5, base_dir / "raw5d_data1104.h5")
    regular_h5 = as_path(args.regular_h5, base_dir / "reg5dbin_label1031.h5")
    target_h5 = args.target_h5
    patch_dir = as_path(args.patch_dir, base_dir / "patchV2")
    patch_dir.mkdir(parents=True, exist_ok=True)

    print("base_dir:", base_dir)
    print("raw_h5:", raw_h5)
    print("regular_h5:", regular_h5)
    print("patch_dir:", patch_dir)

    with File(raw_h5, "r") as f_raw, File(regular_h5, "r") as f_reg:
        raw_group = f_raw[args.group_key]
        regular_group = f_reg[args.group_key]
        print("raw keys:", list(raw_group.keys()))
        print("regular keys:", list(regular_group.keys()))

        trace_obs = read_trace_data(raw_group)
        coord_obs = read_coord4(raw_group)
        coord_grid = read_coord4(regular_group)
        regular_mask = read_regular_mask(
                regular_group, args.regular_mask_key, coord_grid.shape[0],
                target_h5, group_key=args.group_key,
            )
        print("regular mask true count:", int(regular_mask.sum()))

        raw_keys = generate_binning_keys(raw_group).astype(np.int64)
        reg_keys = generate_binning_keys(regular_group).astype(np.int64)

    if trace_obs.shape[0] != coord_obs.shape[0] or raw_keys.shape[0] != coord_obs.shape[0]:
        raise ValueError("raw data, raw coordinates, and raw binning keys must have the same length")
    if reg_keys.shape[0] != coord_grid.shape[0]:
        raise ValueError("regular coordinates and regular binning keys must have the same length")

    print("trace_obs shape:", trace_obs.shape)
    print("coord_obs shape:", coord_obs.shape)
    print("coord_grid shape:", coord_grid.shape)
    print("regular mask true count:", int(regular_mask.sum()))

    raw_obs_valid = raw_obs_valid_mask_from_regular_trusted_mask(
        raw_binning_keys=raw_keys,
        reg_binning_keys=reg_keys,
        regular_trusted_mask=regular_mask,
    )
    trusted_from_mask = np.flatnonzero(raw_obs_valid).astype(np.int64)
    np.save(patch_dir / "raw_obs_valid_mask.npy", raw_obs_valid)
    np.save(patch_dir / "trusted_row_indices_from_regular_mask.npy", trusted_from_mask)
    print("raw_obs_valid true count:", int(raw_obs_valid.sum()))

    coord_obs_norm, coord_grid_norm, norm_stats = normalize_coords(coord_obs, coord_grid)
    save_norm_stats(patch_dir / "coord_norm_stats.npz", norm_stats)
    np.save(patch_dir / "coord_obs_norm.npy", coord_obs_norm)
    np.save(patch_dir / "coord_grid_norm.npy", coord_grid_norm)
    print("coord_obs_norm range:", float(coord_obs_norm.min()), float(coord_obs_norm.max()))
    print("coord_grid_norm range:", float(coord_grid_norm.min()), float(coord_grid_norm.max()))

    grid_index_map_4d, grid_info = build_grid_index_map_4d_from_coord_grid(
        coord_grid_norm,
        on_collision=args.on_grid_collision,
    )
    dims_4d = (grid_info["nsx"], grid_info["nsy"], grid_info["nrx"], grid_info["nry"])
    valid_grid_cells = int(np.count_nonzero(grid_index_map_4d >= 0))
    if valid_grid_cells != coord_grid.shape[0] and args.on_grid_collision == "raise":
        raise ValueError("grid index map valid cell count does not match coord_grid rows")
    if args.save_grid_index_map:
        np.save(patch_dir / "grid_index_map_4d.npy", grid_index_map_4d)
        np.savez(
            patch_dir / "grid_index_map_4d_levels.npz",
            sx_levels=grid_info["sx_levels"],
            sy_levels=grid_info["sy_levels"],
            rx_levels=grid_info["rx_levels"],
            ry_levels=grid_info["ry_levels"],
        )
    print("grid_index_map_4d shape:", grid_index_map_4d.shape)
    print("grid_index_map_4d valid cells:", valid_grid_cells)

    metric_weights = list(args.metric_weights)
    k_patch = int(args.k_patch)
    top_l = int(args.top_l) if args.top_l is not None else k_patch + 128
    infer_top_l = int(args.infer_top_l) if args.infer_top_l is not None else k_patch * 2
    num_anchors = (
        int(args.num_anchors)
        if args.num_anchors is not None
        else max(1, int(trace_obs.shape[0]) // int(args.anchor_stride))
    )
    value_local_top_l = args.value_local_top_l
    if value_local_top_l is None:
        value_local_top_l = top_l

    summary: Dict[str, Any] = {
        "base_dir": str(base_dir),
        "raw_h5": str(raw_h5),
        "regular_h5": str(regular_h5),
        "patch_dir": str(patch_dir),
        "group_key": args.group_key,
        "n_obs": int(coord_obs.shape[0]),
        "n_grid": int(coord_grid.shape[0]),
        "grid_shape_4d": [int(x) for x in dims_4d],
        "regular_mask_true": int(regular_mask.sum()),
        "raw_obs_valid_true": int(raw_obs_valid.sum()),
        "k_patch": k_patch,
        "top_l": top_l,
        "infer_top_l": infer_top_l,
        "num_anchors": num_anchors,
        "metric_weights": metric_weights,
    }

    if not args.skip_train:
        if args.train_trusted_source == "all":
            trusted_idx = np.arange(coord_obs.shape[0], dtype=np.int64)
        else:
            trusted_idx = trusted_from_mask
        if trusted_idx.size == 0:
            raise ValueError("no trusted training observations are available")

        print("building train patches")
        print("train trusted source:", args.train_trusted_source, "count:", trusted_idx.size)
        train_pack = precompute_train_patches_2d(
            coord_obs_norm=coord_obs_norm,
            trace_obs=trace_obs,
            trusted_idx=trusted_idx,
            num_anchors=num_anchors,
            k_patch=k_patch,
            top_l=top_l,
            metric_weights=metric_weights,
            beta=args.beta,
            anchor_selector=args.train_anchor_selector,
            facility_nearest_l=top_l,
            value_local_top_l=value_local_top_l,
            value_suppression=args.value_suppression,
            value_suppression_lambda=args.value_suppression_lambda,
            value_score_tol=args.value_score_tol,
            value_knn_use_gpu=args.train_knn_use_gpu,
            value_knn_gpu_batch_rows=args.value_knn_gpu_batch_rows,
            value_knn_gpu_device=args.value_knn_gpu_device,
            value_knn_full_matrix_max_n=args.value_knn_full_matrix_max_n,
            value_suppression_use_gpu=args.train_suppression_use_gpu,
            num_query=args.num_query,
            seed=args.seed,
            pool_size=args.pool_size,
        )
        np.savez(
            patch_dir / "train_pool_idx_2d.npz",
            pool_idx_2d=train_pack["patch_idx_2d"],
            anchor_idx=train_pack["anchor_idx"],
        )
        if args.save_legacy_anchor_files:
            np.savez(patch_dir / "anchor_train_patch_idx_2d.npz", **{"0": train_pack["patch_idx_2d"]})
            np.savez(patch_dir / "anchor_train_context_idx_2d.npz", **{"0": train_pack["context_idx_2d"]})
            np.savez(patch_dir / "anchor_train_query_idx_2d.npz", **{"0": train_pack["query_idx_2d"]})
            np.save(patch_dir / "anchor_train_anchor_idx.npy", train_pack["anchor_idx"])
            np.save(patch_dir / "anchor_train_anchor_coord.npy", train_pack["anchor_coord"])
        print("train_pool_idx_2d:", train_pack["patch_idx_2d"].shape)
        summary["train_pool_shape"] = [int(x) for x in train_pack["patch_idx_2d"].shape]

    if not args.skip_infer:
        block_size = resolve_block_tuple(args.block_size, args.block_divisors, dims_4d, "block_size")
        stride = resolve_block_tuple(args.stride, args.stride_divisors, dims_4d, "stride")
        query_mask = make_query_mask(
            mode=args.query_mask_mode,
            regular_mask=regular_mask,
            n_grid=coord_grid.shape[0],
        )
        obs_valid_mask = None
        if args.infer_obs_valid_source == "regular_mask":
            obs_valid_mask = raw_obs_valid

        print("building infer patches")
        print("block_size:", block_size, "stride:", stride)
        print("query_mask_mode:", args.query_mask_mode)
        if query_mask is not None:
            print("query targets:", int(query_mask.sum()))
        print("infer obs_valid source:", args.infer_obs_valid_source)
        infer_pack = precompute_infer_patches_4d(
            coord_obs_norm=coord_obs_norm,
            coord_grid_norm=coord_grid_norm,
            grid_shape_4d=None,
            block_size=block_size,
            stride=stride,
            k_patch=k_patch,
            top_l=infer_top_l,
            metric_weights=metric_weights,
            beta=args.beta,
            grid_query_mask=query_mask,
            require_full_query_coverage=args.require_full_query_coverage,
            grid_index_map_4d=grid_index_map_4d,
            max_query_per_patch=args.max_query_per_patch,
            greedy_fill_uncovered=args.greedy_fill_uncovered,
            obs_valid_mask=obs_valid_mask,
            use_gpu=args.infer_use_gpu,
            gpu_device=args.infer_gpu_device,
            gpu_query_chunk_size=args.gpu_query_chunk_size,
        )
        query_list = infer_pack["grid_query_idx_list"]
        context_list = infer_pack["patch_idx_list"]
        input_missing_ratio = summarize_query_context(query_list, context_list)
        np.savez(
            patch_dir / "infer_query_context.npz",
            grid_query_idx_list=object_array(query_list),
            context_idx_list=object_array(context_list),
            block_id=infer_pack["block_id"],
            block_center_grid_idx=infer_pack["block_center_grid_idx"],
            anchor_grid_idx_list=object_array(infer_pack["anchor_grid_idx_list"]),
        )
        np.savez(
            patch_dir / "infer_query_context_stats.npz",
            patch_query_count=infer_pack["patch_query_count"],
            patch_context_count=infer_pack["patch_context_count"],
            patch_input_count=infer_pack["patch_input_count"],
            patch_query_ratio=infer_pack["patch_query_ratio"],
            input_missing_ratio=input_missing_ratio,
        )
        print("infer_query_context samples:", len(query_list))
        if len(query_list):
            print(
                "query count min/max/mean:",
                int(infer_pack["patch_query_count"].min()),
                int(infer_pack["patch_query_count"].max()),
                float(infer_pack["patch_query_count"].mean()),
            )
            print(
                "input missing ratio min/max/mean:",
                float(input_missing_ratio.min()),
                float(input_missing_ratio.max()),
                float(input_missing_ratio.mean()),
            )
        summary["infer_samples"] = int(len(query_list))
        summary["block_size"] = [int(x) for x in block_size]
        summary["stride"] = [int(x) for x in stride]
        summary["query_mask_mode"] = args.query_mask_mode

    validation = validate_outputs(
        patch_dir=patch_dir,
        n_obs=coord_obs.shape[0],
        n_grid=coord_grid.shape[0],
        check_train=not args.skip_train,
        check_infer=not args.skip_infer,
    )
    summary["validation"] = validation
    print("validation:", validation)

    summary_path = Path(args.summary_json).expanduser().resolve() if args.summary_json else patch_dir / "precompute_anchor_patch_v2_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("summary_json:", summary_path)
    print("All checks passed.")


if __name__ == "__main__":
    main()
