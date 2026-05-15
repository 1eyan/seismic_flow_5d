#!/usr/bin/env python3
"""FPM V3 inference CLI — queryctx mode.

Supports single-GPU and multi-GPU (via torchrun).

Usage:
    # Single GPU
    python infer_cli.py --checkpoint results/checkpoints/model-20.pth \
        --h5_irregular data/raw5d_data.h5 --h5_regular data/reg5dbin_label.h5 \
        --h5_mask data/reg5dbin_label_binning.h5 --mask_segy mask.sgy \
        --dataset_neighbors_infer data/infer_query_context.npz

    # Multi-GPU (torchrun)
    torchrun --nproc_per_node=2 infer_cli.py \
        --checkpoint ... --h5_irregular ... --dataset_neighbors_infer ...
"""

import argparse
import csv
import json
import logging
import os
import struct
import sys
import time
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist

from .dataset import DatasetH5_all_queryctx
from .utils import build_coord_config
from .model import SeisDiTRopeV2
from .fpm import FlowMatchingModel
from .infer import run_queryctx_inference, add_prediction, fit_trace
from .utils import (
    read_segy_headers,
    read_segy_data,
    write_segy_data,
    build_lookup,
)

try:
    from .config.segy_config import KEY_COLUMNS
except ImportError:
    from config.segy_config import KEY_COLUMNS

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kw):
        return iterable


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def none_or_str(value):
    if value is None or str(value).lower() in {"none", "null"}:
        return None
    return str(value)


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("queryctx_infer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "infer.log", encoding="utf-8"),
    ):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def setup_ddp() -> Tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    return rank, local_rank, world_size


def load_checkpoint(model: torch.nn.Module, path: str, strict: bool, logger: logging.Logger) -> None:
    ckpt = torch.load(path, map_location="cpu")
    state = (
        ckpt.get("model", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
        if isinstance(ckpt, dict) else ckpt
    )
    if any(k.startswith("module.") for k in state):
        state = OrderedDict((k.replace("module.", "", 1), v) for k, v in state.items())
    result = model.load_state_dict(state, strict=strict)
    logger.info("checkpoint loaded: missing=%s unexpected=%s",
                result.missing_keys, result.unexpected_keys)


def load_training_config(checkpoint_path: str) -> Dict[str, Any]:
    ckpt = Path(checkpoint_path)
    config_path = ckpt.parent.parent / "logs" / "training_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    config_path = ckpt.parent / "logs" / "training_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _first_of(*keys: str, cfg: dict, default=None):
    for k in keys:
        if k in cfg:
            return cfg[k]
    ds = cfg.get("dataset", {})
    for k in keys:
        if k in ds:
            return ds[k]
    dsa = cfg.get("dataset_args", {})
    for k in keys:
        if k in dsa:
            return dsa[k]
    return default


def save_reports(output_dir: Path, headers, written, unfilled, still_missing,
                 observed_changed, unmatched, summary) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    header_by_idx = {int(h["trace_idx"]): h["key"] for h in headers}
    for name, indices in (
        ("filled_missing_keys.csv", written),
        ("unfilled_missing_keys.csv", unfilled),
        ("still_missing_after_write_keys.csv", still_missing),
        ("observed_changed_keys.csv", observed_changed),
    ):
        with open(output_dir / name, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["trace_idx", *KEY_COLUMNS])
            for idx in indices:
                writer.writerow([idx, *header_by_idx.get(int(idx), ("", "", "", ""))])
    with open(output_dir / "unmatched_prediction_keys.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(KEY_COLUMNS)
        writer.writerows(unmatched)


def fill_segy(args, headers, missing_global, pred_sum, pred_count, logger,
              label_data=None) -> dict:
    mask_data = read_segy_data(args.mask_path)
    lookup = build_lookup(headers)
    out = mask_data.copy()
    ns = mask_data.shape[1]
    written, unmatched = set(), []

    for key, total in pred_sum.items():
        indices = lookup.get(key)
        if not indices:
            unmatched.append(key)
            continue
        trace = fit_trace(total / max(pred_count[key], 1), ns)
        wrote = False
        for trace_idx in indices:
            if missing_global[trace_idx]:
                out[trace_idx] = trace
                written.add(trace_idx)
                wrote = True
        if not wrote:
            unmatched.append(key)

    write_start = time.perf_counter()
    write_segy_data(args.mask_path, args.output_segy, out)
    writeback_seconds = time.perf_counter() - write_start

    written_sorted = sorted(written)
    missing_indices = set(np.flatnonzero(missing_global).tolist())
    unfilled = sorted(missing_indices - written)
    after = read_segy_data(args.output_segy)
    still_missing = np.flatnonzero(
        missing_global & np.all(np.abs(after) <= args.missing_eps, axis=1)
    ).tolist()
    observed_changed = np.flatnonzero(
        (~missing_global) & np.any(np.abs(after - mask_data) > args.missing_eps, axis=1)
    ).tolist()

    residual_stats = {}
    if label_data is not None:
        residual = np.zeros_like(out)
        for trace_idx in written_sorted:
            residual[trace_idx] = out[trace_idx] - label_data[trace_idx]
        write_segy_data(args.mask_path, args.output_residual_segy, residual)
        filled_residuals = residual[written_sorted]
        residual_stats = {
            "residual_max_abs": float(np.max(np.abs(filled_residuals))) if filled_residuals.size else 0.0,
            "residual_mean_abs": float(np.mean(np.abs(filled_residuals))) if filled_residuals.size else 0.0,
            "output_residual_segy": args.output_residual_segy,
        }
        logger.info("residual SEGY: %s | max_abs=%.6g mean_abs=%.6g",
                     args.output_residual_segy,
                     residual_stats["residual_max_abs"],
                     residual_stats["residual_mean_abs"])

    summary = {
        "key_columns": list(KEY_COLUMNS),
        "segy_traces": int(mask_data.shape[0]),
        "segy_samples": int(mask_data.shape[1]),
        "missing_total": int(len(missing_indices)),
        "written": int(len(written_sorted)),
        "unfilled": int(len(unfilled)),
        "still_missing_after_write": int(len(still_missing)),
        "observed_changed": int(len(observed_changed)),
        "prediction_keys": int(len(pred_sum)),
        "unmatched_prediction_keys": int(len(unmatched)),
        "writeback_seconds": round(writeback_seconds, 3),
        "output_segy": args.output_segy,
        **residual_stats,
    }

    save_reports(Path(args.output_dir), headers, written_sorted, unfilled,
                 still_missing, observed_changed, unmatched, summary)
    logger.info("writeback summary: %s", summary)

    if args.strict_fill and (unfilled or still_missing or observed_changed):
        raise RuntimeError(
            f"strict fill failed: unfilled={len(unfilled)} "
            f"still_missing={len(still_missing)} observed_changed={len(observed_changed)}"
        )
    return summary


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FPM V3 queryctx inference and SEGY fill")

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--h5_irregular", required=True)
    parser.add_argument("--h5_regular", required=True)
    parser.add_argument("--h5_mask", required=True)
    parser.add_argument("--mask_path", required=True,
                        help="SEGY with missing traces (zero traces are replaced)")
    parser.add_argument("--label_segy", default=None,
                        help="Ground truth SEGY for residual (optional)")
    parser.add_argument("--dataset_neighbors_infer", required=True,
                        help="infer_query_context.npz path")

    parser.add_argument("--output_dir", default="gen_fill_results")
    parser.add_argument("--output_segy", default=None)
    parser.add_argument("--output_residual_segy", default=None)

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--time_ps", type=int, default=1256)
    parser.add_argument("--trace_ps", type=int, default=128)
    parser.add_argument("--missing_eps", type=float, default=1e-10)
    parser.add_argument("--header_mode", choices=["fixed", "self_computed"], default="fixed")

    parser.add_argument("--strict_load", action="store_false", dest="strict_load")
    parser.add_argument("--strict_fill", action="store_true", default=False)

    parser.add_argument("--geom_mode", choices=["source", "receiver", "relative"], default="relative")
    parser.add_argument("--use_missing_embedding", type=str2bool, default=False)
    parser.add_argument("--use_p_scale", type=str2bool, default=False)
    parser.add_argument("--use_phys_omega", type=str2bool, default=True)

    parser.add_argument("--model_type", default="trace_axis")
    parser.add_argument("--pe_type", default="transformer")

    parser.add_argument("--path_type", choices=["Linear", "GVP", "VP"], default="Linear")
    parser.add_argument("--prediction", choices=["velocity", "score", "noise"], default="velocity")
    parser.add_argument("--loss_weight", type=none_or_str, default=None)
    parser.add_argument("--sampling_method", choices=["ode", "sde"], default="ode")
    parser.add_argument("--ode_sampling_method", default="dopri5")
    parser.add_argument("--ode_num_steps", type=int, default=50)
    parser.add_argument("--ode_atol", type=float, default=1e-6)
    parser.add_argument("--ode_rtol", type=float, default=1e-3)
    parser.add_argument("--sde_sampling_method", default="Euler")
    parser.add_argument("--sde_num_steps", type=int, default=250)

    parser.add_argument("--visualize", type=str2bool, default=False)
    parser.add_argument("--vis_batches", type=int, default=0)

    args = parser.parse_args()
    args.output_dir = str(Path(args.output_dir).resolve())
    args.output_segy = args.output_segy or str(Path(args.output_dir) / "filled_missing.sgy")
    args.output_residual_segy = args.output_residual_segy or str(Path(args.output_dir) / "residual.sgy")
    return args


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0

    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
        logger = setup_logger(Path(args.output_dir)) if is_main else logging.getLogger("infer")
        if not is_main:
            logger.setLevel(logging.WARNING)
    else:
        device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
        logger = setup_logger(Path(args.output_dir))

    if is_main:
        logger.info("args: %s", vars(args))
        logger.info("key_columns=%s world_size=%d", KEY_COLUMNS, world_size)

    total_start = time.perf_counter()

    # Read SEGY
    mask_data = read_segy_data(args.mask_path)
    headers = read_segy_headers(args.mask_path, args.header_mode)
    if len(headers) != mask_data.shape[0]:
        raise ValueError(f"header_count={len(headers)} != segy_traces={mask_data.shape[0]}")
    missing_global = np.all(np.abs(mask_data) <= args.missing_eps, axis=1)
    if is_main:
        logger.info("template SEGY: traces=%d samples=%d missing=%d",
                     mask_data.shape[0], mask_data.shape[1], int(missing_global.sum()))

    label_data = None
    if args.label_segy:
        label_data = read_segy_data(args.label_segy)
        if label_data.shape != mask_data.shape:
            raise ValueError(f"label shape {label_data.shape} != mask shape {mask_data.shape}")
        logger.info("label SEGY loaded: %s shape=%s", args.label_segy, label_data.shape)

    # Load training config from checkpoint for auto-detection
    train_cfg = load_training_config(args.checkpoint)
    if train_cfg and is_main:
        logger.info("loaded training config from checkpoint directory")

    # ---- Dataset ----
    use_p_scale = args.use_p_scale
    time_ps = args.time_ps
    if train_cfg:
        use_p_scale = bool(_first_of(
            "use_p_scale", cfg=train_cfg, default=use_p_scale
        ))
        tps = _first_of("time_ps", cfg=train_cfg)
        if tps is not None:
            time_ps = int(tps)

    logger.info("building DatasetH5_all_queryctx inference dataset")
    dataset = DatasetH5_all_queryctx(
        h5File=args.h5_irregular,
        h5File_regular=args.h5_regular,
        dataset_neighbors=args.dataset_neighbors_infer,
        train=False,
        use_p_scale=use_p_scale,
        time_ps=time_ps,
        trace_ps=args.trace_ps,
    )
    logger.info("queryctx dataset ready: samples=%d time_ps=%d", len(dataset), dataset.time_ps)

    # ---- Coord config ----
    coord_config = build_coord_config(dataset)
    if is_main:
        logger.info("coord_config: %s", coord_config.get("_summary", ""))

    if args.use_phys_omega and coord_config.get("omega_mode") != "disabled":
        omega = coord_config["omega"]
        rope_base = (omega["x"] + omega["y"]) / 2.0
        if is_main:
            logger.info("PhysOmega: rope_base=%.2f", rope_base)
    else:
        rope_base = 10000.0

    # ---- Model ----
    backbone = SeisDiTRopeV2(
        image_channels=2,
        n_channels=32,
        f_dict=None,
        num_layers=8,
        d_model=512,
        pe_type=args.pe_type,
        geom_mode=args.geom_mode,
        missing_focus_adapter=args.use_missing_embedding,
        coord_config=coord_config,
        rope_base=rope_base,
    ).to(device).eval()

    load_checkpoint(backbone, args.checkpoint, not args.strict_load, logger)
    if is_main:
        logger.info("model=SeisDiTRopeV2 params=%d device=%s",
                     sum(p.numel() for p in backbone.parameters()), device)

    fpm = FlowMatchingModel(
        model=backbone,
        trace_num=dataset.trace_ps,
        time_steps=dataset.time_ps,
        path_type=args.path_type,
        prediction=args.prediction,
        loss_weight=args.loss_weight,
        sample_num=1,
        device=device,
        sup_mode="all",
        use_coherence=False,
        sigma_obs=0.001,
        use_bayesian=False,
        sampling_method=args.sampling_method,
        ode_sampling_method=args.ode_sampling_method,
        ode_num_steps=args.ode_num_steps,
        ode_atol=args.ode_atol,
        ode_rtol=args.ode_rtol,
        sde_sampling_method=args.sde_sampling_method,
        sde_num_steps=args.sde_num_steps,
    ).eval()

    # ---- Inference ----
    pred_sum, pred_count, infer_stats = run_queryctx_inference(
        dataset=dataset,
        fpm=fpm,
        device=device,
        batch_size=args.batch_size,
        visualize=args.visualize,
        vis_dir=str(Path(args.output_dir) / "vis"),
        vis_max=args.vis_batches,
        progress=True,
        logger=logger,
    )

    # ---- DDP gather ----
    if world_size > 1:
        dist.barrier()
        gathered = [None] * world_size
        dist.all_gather_object(gathered, (pred_sum, dict(pred_count), infer_stats))
        if is_main:
            merged_sum, merged_count = {}, defaultdict(int)
            for ps, pc, _st in gathered:
                for k, v in ps.items():
                    merged_sum[k] = v if k not in merged_sum else merged_sum[k] + v
                for k, v in pc.items():
                    merged_count[k] += v
            pred_sum, pred_count = merged_sum, merged_count

    # ---- SEGY fill ----
    if is_main:
        summary = fill_segy(args, headers, missing_global, pred_sum, pred_count,
                            logger, label_data=label_data)
        summary.update(infer_stats)
        summary["num_gpus"] = world_size
        summary["total_seconds"] = round(time.perf_counter() - total_start, 3)
        (Path(args.output_dir) / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("done in %.2fs | output=%s", summary["total_seconds"], args.output_segy)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
