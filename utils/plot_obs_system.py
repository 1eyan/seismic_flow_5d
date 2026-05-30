"""
Plot observation system (shot & receiver geometry) from mk_irr_mask output.

Reads {stem}-shot_xy_cut.dat and {stem}-rcvs_xy_cut.dat, draws:
  1. Shot + receiver scatter on one canvas (label / irr / mask comparison)
  2. Optional: line-connectivity overview

Usage:
  python utils/plot_obs_system.py /path/to/output_dir/ --stem A002-train-part1
  # or set env:
  OBS_DIR=/path/to OBS_STEM=A002-train-part1 python utils/plot_obs_system.py
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OBS_COLORS = {"label": "#2c7bb6", "irr": "#d7191c", "masked": "#cccccc"}


def _load_xy(path: str, label: str):
    """Load a 2-column int array; return empty if file missing."""
    if not os.path.isfile(path):
        print(f"[WARN] {label}: {path} not found, skipping")
        return None
    data = np.loadtxt(path, delimiter="\t", dtype=np.int32)
    # handle single-point file shape
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def plot_obs_system(shot_xy: np.ndarray, rcv_xy: np.ndarray,
                    stem: str, output_dir: str, suffix: str = ""):
    """Main observation system plot — 4 panels."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Observation System — {stem}{suffix}", fontsize=14, y=0.98)

    # Panel A: Shot distribution
    ax = axes[0, 0]
    ax.scatter(shot_xy[:, 0], shot_xy[:, 1], s=8, c=OBS_COLORS["label"],
               alpha=0.7, edgecolors="none", label=f"Shots ({len(shot_xy)})")
    ax.set_title("Shot Points")
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.legend(markerscale=2)
    ax.grid(True, alpha=0.3)

    # Panel B: Receiver distribution
    ax = axes[0, 1]
    ax.scatter(rcv_xy[:, 0], rcv_xy[:, 1], s=8, c=OBS_COLORS["irr"],
               alpha=0.7, edgecolors="none", label=f"Receivers ({len(rcv_xy)})")
    ax.set_title("Receiver Points")
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.legend(markerscale=2)
    ax.grid(True, alpha=0.3)

    # Panel C: Shot + Receiver overlay
    ax = axes[1, 0]
    ax.scatter(shot_xy[:, 0], shot_xy[:, 1], s=12, c=OBS_COLORS["label"],
               alpha=0.8, marker="^", label=f"Shots ({len(shot_xy)})")
    ax.scatter(rcv_xy[:, 0], rcv_xy[:, 1], s=6, c=OBS_COLORS["irr"],
               alpha=0.6, marker=".", label=f"Receivers ({len(rcv_xy)})")
    ax.set_title("Shot + Receiver Overlay")
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.legend(markerscale=2)
    ax.grid(True, alpha=0.3)

    # Panel D: Fold count histogram per receiver
    ax = axes[1, 1]
    if len(shot_xy) > 1 and len(rcv_xy) > 1:
        # simple bin-count: how many shots cover each receiver area
        x_min = min(shot_xy[:, 0].min(), rcv_xy[:, 0].min())
        x_max = max(shot_xy[:, 0].max(), rcv_xy[:, 0].max())
        y_min = min(shot_xy[:, 1].min(), rcv_xy[:, 1].min())
        y_max = max(shot_xy[:, 1].max(), rcv_xy[:, 1].max())
        bins = 40
        heat, xedges, yedges = np.histogram2d(
            rcv_xy[:, 0], rcv_xy[:, 1], bins=bins,
            range=[[x_min, x_max], [y_min, y_max]],
        )
        im = ax.imshow(heat.T, origin="lower", aspect="auto",
                       extent=[x_min, x_max, y_min, y_max], cmap="viridis")
        fig.colorbar(im, ax=ax, label="Receiver count per bin")
        ax.set_title("Receiver Density (2D histogram)")
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)

    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(output_dir, f"{stem}_obs_system{suffix}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Saved  {out_path}")
    plt.close(fig)


def plot_missing_comparison(shot_label: np.ndarray, shot_irr: np.ndarray,
                            rcv_label: np.ndarray, rcv_irr: np.ndarray,
                            stem: str, output_dir: str):
    """Side-by-side comparison: label vs irregular."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
    fig.suptitle(f"Observation System — {stem}  (label vs irregular)", fontsize=14, y=0.98)

    panels = [
        (0, 0, "Shots — label",  shot_label, OBS_COLORS["label"]),
        (0, 1, "Shots — irr",    shot_irr,   OBS_COLORS["irr"]),
        (1, 0, "Receivers — label", rcv_label, OBS_COLORS["label"]),
        (1, 1, "Receivers — irr",   rcv_irr,   OBS_COLORS["irr"]),
    ]
    for row, col, title, xy, color in panels:
        ax = axes[row, col]
        ax.scatter(xy[:, 0], xy[:, 1], s=6, c=color, alpha=0.6, edgecolors="none")
        ax.set_title(f"{title}  ({len(xy)})")
        ax.set_xlabel("Easting")
        ax.set_ylabel("Northing")
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(output_dir, f"{stem}_obs_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Saved  {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot observation system from mk_irr_mask output")
    parser.add_argument("dir", nargs="?", default=os.environ.get("OBS_DIR", "."),
                        help="Output directory containing *shot_xy_cut.dat (default: $OBS_DIR or .)")
    parser.add_argument("--stem", default=os.environ.get("OBS_STEM", ""),
                        help="File stem (e.g. A002-train-part1). If empty, auto-detect first *shot_xy_cut.dat")
    parser.add_argument("--out", default="",
                        help="Output image dir (default: same as input dir)")
    args = parser.parse_args()

    data_dir = args.dir
    out_dir = args.out or data_dir

    # auto-detect stem if not given
    stem = args.stem
    if not stem:
        candidates = [f for f in os.listdir(data_dir) if f.endswith("shot_xy_cut.dat")]
        if not candidates:
            print("[ERROR] No *shot_xy_cut.dat found in", data_dir)
            sys.exit(1)
        # use the first match, strip the "-shot_xy_cut.dat" suffix
        stem = candidates[0].rsplit("-shot_xy_cut.dat", 1)[0]
        print(f"[INFO] Auto-detected stem: {stem}")

    # label (full) geometry
    shot_file = os.path.join(data_dir, f"{stem}-shot_xy_cut.dat")
    rcv_file  = os.path.join(data_dir, f"{stem}-rcvs_xy_cut.dat")

    shot_label = _load_xy(shot_file, "shot_xy_cut")
    rcv_label  = _load_xy(rcv_file, "rcvs_xy_cut")

    if shot_label is None or rcv_label is None:
        print("[ERROR] Missing coordinate files, cannot plot")
        sys.exit(1)

    # main plot
    plot_obs_system(shot_label, rcv_label, stem, out_dir)

    # try load irr comparison if exists
    shot_irr_file = os.path.join(data_dir, f"{stem}-shot_xy_irr.dat")
    rcv_irr_file  = os.path.join(data_dir, f"{stem}-rcvs_xy_irr.dat")
    shot_irr = _load_xy(shot_irr_file, "shot_xy_irr")
    rcv_irr  = _load_xy(rcv_irr_file, "rcvs_xy_irr")
    if shot_irr is not None and rcv_irr is not None:
        plot_missing_comparison(shot_label, shot_irr, rcv_label, rcv_irr, stem, out_dir)

    print(f"[DONE] Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
