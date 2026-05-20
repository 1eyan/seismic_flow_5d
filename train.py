#!/usr/bin/env python3
"""FPM V3 training script — queryctx mode with trace_axis model.

Supports single-GPU and multi-GPU (via accelerate / torchrun).

Usage:
    # Single GPU
    python train.py --h5File data/irregular.h5 --h5File_regular data/regular.h5 \
        --dataset_neighbors_train data/train_pool_idx_2d.npz

    # Multi-GPU (accelerate)
    accelerate launch --config_file accelerate_config.yaml train.py \
        --h5File data/irregular.h5 --h5File_regular data/regular.h5 \
        --dataset_neighbors_train data/train_pool_idx_2d.npz
"""

import argparse
import gc
import json
import math
import os
import pathlib
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# -- internal (self-contained) --

from dataset import DatasetH5_all_queryctx
from config.data_config import get_parser
from config.segy_config import print_info as print_segy_config
from utils import build_coord_config
from model import SeisDiTRopeV2
from fpm import FlowMatchingModel


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def _batch_to_xy(batch):
    """Unify batch dict → (data, data_mask, rx, ry, sx, sy)."""
    if not isinstance(batch, dict):
        return None
    if "data" in batch:
        return (
            batch["data"],
            batch["masked_patch"],
            batch["rx_patch"],
            batch["ry_patch"],
            batch["sx_patch"],
            batch["sy_patch"],
        )
    if "x_gt" in batch:
        return (
            batch["x_gt"],
            batch["x_obs"],
            batch["gx"],
            batch["gy"],
            batch["sx"],
            batch["sy"],
        )
    return None


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    # Start from queryctx parser and add training-specific arguments
    parser = argparse.ArgumentParser(parents=[get_parser()], conflict_handler="resolve")

    parser.add_argument("--model_name", type=str, default="trace_axis")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=515)
    parser.add_argument("--data_type", type=str, default="df_field1031_5d")
    parser.add_argument("--geom_mode", type=str, default="relative",
                        choices=["source", "receiver", "relative"])
    parser.add_argument("--use_missing_embedding", type=str2bool, default=False)
    parser.add_argument("--pe_type", type=str, default="transformer")

    # RoPE
    parser.add_argument("--rope_base", type=float, default=10000.0,
                        help="RoPE base frequency (overridden when use_phys_omega=true)")

    # Flow matching
    parser.add_argument("--path_type", type=str, default="Linear",
                        choices=["Linear", "GVP", "VP"])
    parser.add_argument("--prediction", type=str, default="velocity",
                        choices=["velocity", "score", "noise"])
    parser.add_argument("--loss_weight", type=str, default=None,
                        choices=[None, "velocity", "likelihood"])
    parser.add_argument("--sampling_method", type=str, default="ode",
                        choices=["ode", "sde"])
    parser.add_argument("--ode_num_steps", type=int, default=50)
    parser.add_argument("--sde_num_steps", type=int, default=250)

    # Output
    parser.add_argument("--results_dir", type=str, default="./resultsFPM")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--accumulation_steps", type=int, default=4)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------


class Trainer:
    def __init__(
        self,
        fpm: FlowMatchingModel,
        results_folder: str,
        dl: DataLoader,
        val_dl: DataLoader,
        args: argparse.Namespace,
        accelerator: Accelerator,
        train_lr: float = 1e-4,
        epochs: int = 200,
        save_every: int = 10,
        accumulation_steps: int = 4,
        warmup_epochs: int = 5,
    ):
        self.fpm = fpm
        self.args = args
        self.accelerator = accelerator
        self.device = accelerator.device

        self.results_folder = Path(results_folder)
        self.ckp_folder = self.results_folder / "checkpoints"
        self.img_folder = self.results_folder / "images"
        self.log_folder = self.results_folder / "logs"

        if accelerator.is_main_process:
            for d in (self.results_folder, self.ckp_folder, self.img_folder, self.log_folder):
                d.mkdir(exist_ok=True, parents=True)
            self.writer = SummaryWriter(log_dir=str(self.log_folder))
            self.log_file = open(self.log_folder / "training_log.txt", "a")
        else:
            self.writer = None
            self.log_file = None

        self.dl = dl
        self.val_dl = val_dl

        self.train_epochs = epochs
        self.num_steps = len(self.dl)
        self.train_num_steps = self.train_epochs * self.num_steps
        self.save_every = save_every

        self.accumulation_steps = accumulation_steps
        self.base_lr = train_lr

        model_to_optimize = fpm.model if not hasattr(fpm.model, "module") else fpm.model.module

        self.opt = AdamW(
            [{"params": model_to_optimize.parameters(), "lr": train_lr}],
            lr=train_lr,
            betas=(0.9, 0.95),
            weight_decay=1e-4,
        )

        self.warmup_epochs = warmup_epochs
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt,
            T_max=max(1, self.train_epochs - self.warmup_epochs),
            eta_min=5e-5,
        )

        self.fpm.model, self.opt, self.dl, self.val_dl = accelerator.prepare(
            self.fpm.model, self.opt, self.dl, self.val_dl
        )

        if accelerator.is_main_process:
            self._save_training_config(args)
            self._log(f"Model: SeisDiTRopeV2")
            self._log(f"Train dataset: {len(self.dl.dataset)} samples")
            self._log(f"Val dataset:   {len(self.val_dl.dataset)} samples")
            self._log(f"Steps/epoch:   {self.num_steps}")
            self._log(f"Total steps:   {self.train_num_steps}")
            self._log(f"Batch size:    {args.batch_size}")
            self._log(f"Learning rate: {self.base_lr}")

    def _log(self, msg: str):
        if self.accelerator.is_main_process and self.log_file is not None:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_file.write(f"[{ts}] {msg}\n")
            self.log_file.flush()
            print(msg)

    def _save_training_config(self, args):
        cfg = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "training_args": {k: v for k, v in vars(args).items()},
            "model": {
                "class": "SeisDiTRopeV2",
                "total_params": sum(p.numel() for p in self.fpm.model.parameters()),
                "trainable_params": sum(
                    p.numel() for p in self.fpm.model.parameters() if p.requires_grad
                ),
            },
            "training": {
                "batch_size": args.batch_size,
                "learning_rate": self.base_lr,
                "epochs": self.train_epochs,
                "steps_per_epoch": self.num_steps,
            },
            "dataset": {
                "train_length": len(self.dl.dataset),
                "val_length": len(self.val_dl.dataset),
                "class": "DatasetH5_all_queryctx",
                "h5File": args.h5File,
                "h5File_regular": args.h5File_regular,
                "dataset_neighbors_train": args.dataset_neighbors_train,
                "train_num_query": args.train_num_query,
                "patch_beta": args.patch_beta,
                "trace_sort_keys": args.trace_sort_keys,
                "time_ps": args.time_ps,
                "trace_ps": args.trace_ps,
            },
        }

        ds = self.dl.dataset
        if hasattr(ds, "coord_stats") and ds.coord_stats:
            cfg["coord_stats"] = ds.coord_stats
            coord_config = build_coord_config(ds)
            cfg["coord_norm_mode"] = coord_config["coord_norm_mode"]
            cfg["omega"] = coord_config["omega"]

        with open(self.log_folder / "training_config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False, default=str)

    def save(self, milestone: int):
        if not self.accelerator.is_main_process:
            return
        unwrapped = self.accelerator.unwrap_model(self.fpm.model)
        path = self.ckp_folder / f"model-{milestone}.pth"
        torch.save({"model": unwrapped.state_dict()}, path)
        self._log(f"Saved checkpoint: {path}")

    def _compute_val_loss(self):
        self.fpm.model.eval()
        losses = []
        max_samples = min(50, len(self.val_dl))
        with torch.no_grad():
            for i, batch in enumerate(self.val_dl):
                if i >= max_samples:
                    break
                xy = _batch_to_xy(batch) if isinstance(batch, dict) else None
                if xy is not None:
                    data, data_mask, rx, ry, sx, sy = xy
                else:
                    data, data_mask, rx, ry, sx, sy, _, _ = batch

                data = data.unsqueeze(1).to(self.device)
                data_mask = data_mask.unsqueeze(1).to(self.device)
                rx, ry, sx, sy = (
                    rx.to(self.device), ry.to(self.device), sx.to(self.device), sy.to(self.device)
                )

                with self.accelerator.autocast():
                    loss = self.fpm(data, condL=(rx, ry, sx, sy), x_cond=data_mask)
                losses.append(loss.item())
        self.fpm.model.train()
        return sum(losses) / len(losses) if losses else float("nan")

    def train(self):
        with tqdm(
            range(self.train_epochs),
            total=self.train_epochs,
            desc="Training",
            disable=not self.accelerator.is_main_process,
        ) as pbar:
            for epoch in pbar:
                loss_list = []
                if hasattr(self.dl.sampler, "set_epoch"):
                    self.dl.sampler.set_epoch(epoch)

                self.fpm.model.train()
                self.opt.zero_grad(set_to_none=True)

                for idx, batch in enumerate(self.dl):
                    xy = _batch_to_xy(batch) if isinstance(batch, dict) else None
                    if xy is not None:
                        data, data_mask, rx, ry, sx, sy = xy
                    else:
                        data, data_mask, rx, ry, sx, sy, _, _ = batch

                    data = data.unsqueeze(1).to(self.device)
                    data_mask = data_mask.unsqueeze(1).to(self.device)
                    rx, ry, sx, sy = (
                        rx.to(self.device), ry.to(self.device),
                        sx.to(self.device), sy.to(self.device),
                    )

                    with self.accelerator.autocast():
                        loss = self.fpm(data, condL=(rx, ry, sx, sy), x_cond=data_mask)

                    loss = loss / self.accumulation_steps
                    self.accelerator.backward(loss)

                    do_step = ((idx + 1) % self.accumulation_steps == 0) or (
                        idx + 1 == len(self.dl)
                    )
                    if do_step:
                        self.accelerator.clip_grad_norm_(
                            self.fpm.parameters(), max_norm=1.0
                        )
                        self.opt.step()
                        self.opt.zero_grad(set_to_none=True)

                    loss_list.append(loss.item() * self.accumulation_steps)

                    if epoch == 0 and idx == 0 and self.accelerator.is_main_process:
                        self._log(f"First loss: {loss_list[-1]:.4f}")

                # LR schedule
                if epoch < self.warmup_epochs:
                    warmup_lr = self.base_lr * (epoch + 1) / self.warmup_epochs
                    for pg in self.opt.param_groups:
                        pg["lr"] = warmup_lr
                else:
                    self.lr_scheduler.step()

                avg_loss = sum(loss_list) / len(loss_list) if loss_list else float("nan")
                current_lr = self.opt.param_groups[0]["lr"]

                gc.collect()
                torch.cuda.empty_cache()

                val_loss = None
                if epoch % 10 == 0 or epoch == 0:
                    val_loss = self._compute_val_loss()

                if self.accelerator.is_main_process:
                    pbar.set_postfix(
                        epoch=epoch + 1, loss=avg_loss,
                        val_loss=val_loss if val_loss is not None else "N/A",
                        lr=current_lr,
                    )
                    self.writer.add_scalar("Loss/train", avg_loss, epoch)
                    self.writer.add_scalar("LR", current_lr, epoch)
                    if val_loss is not None:
                        self.writer.add_scalar("Loss/val", val_loss, epoch)

                if ((epoch + 1) % self.save_every == 0 or epoch == 0) and self.accelerator.is_main_process:
                    self.save((epoch + 1) // self.save_every)

                self.accelerator.wait_for_everyone()

    def __del__(self):
        if hasattr(self, "log_file") and self.log_file is not None:
            self.log_file.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
        kwargs_handlers=[ddp_kwargs],
    )

    device = accelerator.device
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    # Print active SEG-Y dataset config (rank 0 only)
    if rank == 0:
        print_segy_config()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ---- Dataset ----
    trace_sort_keys = tuple(args.trace_sort_keys.split(","))

    if rank != 0:
        _real_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
    try:
        dataset = DatasetH5_all_queryctx(
            h5File=args.h5File,
            h5File_regular=args.h5File_regular,
            dataset_neighbors=args.dataset_neighbors_train,
            train=True,
            train_num_query=args.train_num_query,
            train_context_size=args.train_context_size,
            patch_beta=args.patch_beta,
            force_anchor_query=args.force_anchor_query,
            #trace_sort_keys=trace_sort_keys,
            use_p_scale=args.use_p_scale,
            time_ps=args.time_ps,
            trace_ps=args.trace_ps,
            epoch_repeat=args.epoch_repeat,
        )
    finally:
        if rank != 0:
            sys.stdout = _real_stdout

    if rank == 0:
        print(f"[queryctx] time_ps={dataset.time_ps} trace_ps={dataset.trace_ps} "
              f"train_num_query={args.train_num_query} samples={len(dataset)}")

    # ---- Coord config ----
    coord_config = build_coord_config(dataset)
    if rank == 0:
        print(coord_config["_summary"])

    # ---- RoPE base ----
    if args.use_phys_omega and coord_config.get("omega_mode") != "disabled":
        omega = coord_config["omega"]
        rope_base = (omega["x"] + omega["y"]) / 2.0
        if rank == 0:
            print(f"[PhysOmega] rope_base={rope_base:.2f}")
    else:
        rope_base = args.rope_base

    # ---- DataLoaders ----
    train_sampler = DistributedSampler(dataset) if world_size > 1 else None
    val_sampler = DistributedSampler(dataset, shuffle=False) if world_size > 1 else None

    dl = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=5,
        sampler=train_sampler,
    )
    val_dl = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4, sampler=val_sampler
    )

    if rank == 0:
        print(f"dataset: {len(dataset)} samples, shape={dataset[0]['data'].shape}")
        print(f"dl: {len(dl)} batches, val_dl: {len(val_dl)} batches")

    # ---- Model ----
    model_unet = SeisDiTRopeV2(
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
    ).to(device)

    trace_ps_actual = dataset.trace_ps
    time_ps_actual = dataset.time_ps

    fpm = FlowMatchingModel(
        model=model_unet,
        trace_num=trace_ps_actual,
        time_steps=time_ps_actual,
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
        ode_num_steps=args.ode_num_steps,
        sde_num_steps=args.sde_num_steps,
    )

    res_dir = Path(args.results_dir) / f"{args.model_name}_datatype_{args.data_type}_queryctx"
    if accelerator.is_main_process:
        res_dir.mkdir(exist_ok=True, parents=True)

    trainer = Trainer(
        fpm=fpm,
        results_folder=str(res_dir),
        dl=dl,
        val_dl=val_dl,
        args=args,
        accelerator=accelerator,
        train_lr=args.lr,
        epochs=args.epochs,
        save_every=args.save_every,
        accumulation_steps=args.accumulation_steps,
    )

    trainer.train()


if __name__ == "__main__":
    main()
