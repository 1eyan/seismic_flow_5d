#!/usr/bin/env python3
"""大规模观测下锚点与 patch 的 2D 可视化（分层抽样 + PCA + 凸包/散点上限）。

设计要点（数据量很大时）：
- 仅用随机子集拟合 2D PCA，再投影全体锚点与 patch 内点（矩阵一次性投影，不逐点 Python 循环）。
- 背景用 hexbin / 下采样散点表达观测密度，避免 ``N`` 个点全部 scatter。
- patch 边界用 2D 凸包（或点数过少时用线段/小圆），并限制绘制 patch 数量 + 每 patch 点数上限。
- 可选第二张子图：仅锚点 + 密度，用于快速总览。

依赖：numpy、matplotlib；凸包优先 ``scipy.spatial.ConvexHull``（无则退化为 patch 内最多 ``--fallback_scatter_per_patch`` 个点）。

用法示例::

    python visualize_anchor_patches.py --patch_dir /path/to/patch \\
        --coord_obs_norm /path/to/coord_obs_norm.npy \\
        --out_png overview.png

若 ``patch_dir`` 下已有 ``anchor_train_patch_idx_2d.npz`` 与 ``anchor_train_anchor_idx.npy``，可省略显式传 patch 路径。

**坐标约定**（与 ``coord_obs_norm`` 列顺序一致）：0,1 炮点 x,y；2,3 检波点 x,y。
使用 ``--axes I J`` 时直接取第 ``I``、``J`` 列做平面投影（无 PCA），例如 ``--axes 0 1`` 炮点平面，``--axes 2 3`` 检波点平面。
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# PCA（仅 numpy，在子集上拟合）
# -----------------------------------------------------------------------------


def fit_pca_2d(
    coords: np.ndarray,
    *,
    max_fit_samples: int,
    seed: int,
) -> Dict[str, Any]:
    """在 ``coords`` 的随机子集上拟合中心化 + 前两个主成分方向。"""
    n, d = coords.shape
    rng = np.random.default_rng(seed)
    if n <= max_fit_samples:
        fit_idx = np.arange(n)
    else:
        fit_idx = rng.choice(n, size=max_fit_samples, replace=False)
    X = coords[fit_idx].astype(np.float64, copy=False)
    mean = X.mean(axis=0)
    Xc = X - mean
    # 协方差特征分解（对称、实数）
    cov = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    W = evecs[:, order[:2]]
    return {"mean": mean, "W": W}


def project_2d(coords: np.ndarray, state: Dict[str, Any]) -> np.ndarray:
    X = coords.astype(np.float64, copy=False)
    return (X - state["mean"]) @ state["W"]


def slice_coord_2d(coords: np.ndarray, i: int, j: int) -> np.ndarray:
    """取四维坐标的两列作为平面 ``[N,2]``。"""
    if not (0 <= i < 4 and 0 <= j < 4):
        raise ValueError(f"axes 列号须在 [0,3] 内，得到 ({i}, {j})")
    if i == j:
        raise ValueError("axes 两列须不同")
    return np.asarray(coords[:, [i, j]], dtype=np.float64)


def default_axis_label(dim: int) -> str:
    """默认列名：炮点 0,1；检波点 2,3。"""
    names = ("shot_x", "shot_y", "recv_x", "recv_y")
    return names[dim]


# -----------------------------------------------------------------------------
# 凸包（可选 scipy）
# -----------------------------------------------------------------------------


def _hull_polygon_xy(xy: np.ndarray) -> Optional[np.ndarray]:
    """返回凸包顶点 ``[H,2]``（闭合不重复首点）；点不足 3 个返回 None。"""
    if xy.shape[0] < 3:
        return None
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(xy)
        return xy[hull.vertices]
    except Exception:
        return None


# -----------------------------------------------------------------------------
# 绘图
# -----------------------------------------------------------------------------


def _draw_patch_region(
    ax: Any,
    xy_patch: np.ndarray,
    *,
    color: Any,
    alpha_fill: float,
    alpha_edge: float,
    max_scatter: int,
    rng: np.random.Generator,
) -> None:
    """凸包填充；失败则最多 ``max_scatter`` 个散点。"""
    poly = _hull_polygon_xy(xy_patch)
    if poly is not None:
        poly_closed = np.vstack([poly, poly[:1]])
        ax.fill(
            poly_closed[:, 0],
            poly_closed[:, 1],
            facecolor=color,
            edgecolor=color,
            alpha=alpha_fill,
            linewidth=0.4,
            zorder=2,
        )
        ax.plot(
            poly_closed[:, 0],
            poly_closed[:, 1],
            color=color,
            alpha=alpha_edge,
            linewidth=0.6,
            zorder=3,
        )
        return
    if xy_patch.shape[0] == 2:
        ax.plot(xy_patch[:, 0], xy_patch[:, 1], color=color, alpha=alpha_edge, linewidth=0.8, zorder=3)
        return
    if xy_patch.shape[0] == 1:
        ax.scatter(
            xy_patch[:, 0],
            xy_patch[:, 1],
            s=4,
            c=[color],
            alpha=alpha_edge,
            zorder=3,
        )
        return
    n = xy_patch.shape[0]
    if n > max_scatter:
        sel = rng.choice(n, size=max_scatter, replace=False)
        xy_patch = xy_patch[sel]
    ax.scatter(
        xy_patch[:, 0],
        xy_patch[:, 1],
        s=2,
        c=[color],
        alpha=0.25,
        linewidths=0,
        zorder=2,
    )


def run_visualize(
    coord_obs_norm: np.ndarray,
    anchor_idx: np.ndarray,
    patch_idx_2d: np.ndarray,
    *,
    out_png: str,
    axes: Optional[Tuple[int, int]] = None,
    axis_xlabel: Optional[str] = None,
    axis_ylabel: Optional[str] = None,
    max_pca_fit: int = 100_000,
    max_hexbin_points: int = 400_000,
    gridsize: int = 80,
    max_patch_hulls: int = 400,
    max_points_per_patch_for_hull: int = 256,
    fallback_scatter_per_patch: int = 64,
    cmap_background: str = "bone",
    cmap_hulls: str = "tab20",
    alpha_hull_fill: float = 0.12,
    alpha_hull_edge: float = 0.35,
    anchor_size: float = 12.0,
    seed: int = 0,
    dpi: int = 150,
    figsize: Tuple[float, float] = (14.0, 12.0),
    title: Optional[str] = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    coord = np.asarray(coord_obs_norm, dtype=np.float32)
    if coord.ndim != 2 or coord.shape[1] != 4:
        raise ValueError(f"coord_obs_norm 需要 [N,4]，得到 {coord.shape}")

    anchors = np.asarray(anchor_idx, dtype=np.int64).reshape(-1)
    p2d = np.asarray(patch_idx_2d, dtype=np.int64)
    if p2d.ndim != 2 or p2d.shape[0] != anchors.size:
        raise ValueError("patch_idx_2d 行数须与 anchor 数量一致")

    n_obs = coord.shape[0]
    if np.any(anchors < 0) or np.any(anchors >= n_obs):
        raise ValueError("anchor_idx 越界")
    if np.any(p2d >= n_obs) and np.any(p2d >= 0):
        bad = p2d[p2d >= 0]
        if np.max(bad) >= n_obs:
            raise ValueError("patch_idx_2d 存在 >= N 的索引")

    rng = np.random.default_rng(seed)

    if axes is not None:
        ai, aj = int(axes[0]), int(axes[1])
        xy_all = slice_coord_2d(coord, ai, aj)
        xl = axis_xlabel or default_axis_label(ai)
        yl = axis_ylabel or default_axis_label(aj)
        plot_title = title or f"Anchors & patches (axes {ai},{aj})"
        supt = (
            "Direct 2D slice of coord columns; hulls subsampled (not all observations drawn)"
        )
        right_subtitle = "All anchors + density (same axes)"
    else:
        xl, yl = "PC1", "PC2"
        state = fit_pca_2d(coord, max_fit_samples=max_pca_fit, seed=seed)
        xy_all = project_2d(coord, state)
        plot_title = title or "Anchors & patches (PCA 2D)"
        supt = "PCA fit on a random subset; hulls subsampled (not all observations drawn)"
        right_subtitle = "All anchors + density (same PCA)"

    xy_anchor = xy_all[anchors]

    # 背景：hexbin 用子集，避免内存爆炸
    n_bg = min(n_obs, max_hexbin_points)
    if n_bg < n_obs:
        bg_idx = rng.choice(n_obs, size=n_bg, replace=False)
    else:
        bg_idx = np.arange(n_obs)
    xy_bg = xy_all[bg_idx]

    a = anchors.size
    hull_indices = np.arange(a)
    if a > max_patch_hulls:
        hull_indices = rng.choice(a, size=max_patch_hulls, replace=False)
        hull_indices.sort()

    tab = plt.get_cmap(cmap_hulls)
    norm = mcolors.Normalize(vmin=0, vmax=max(19, len(hull_indices) - 1))

    fig, axes = plt.subplots(1, 2, figsize=(figsize[0], figsize[1] * 0.55))

    # 左：密度 + 少量 patch 区域 + 锚点
    ax0 = axes[0]
    hb = ax0.hexbin(
        xy_bg[:, 0],
        xy_bg[:, 1],
        gridsize=gridsize,
        mincnt=1,
        cmap=cmap_background,
        bins="log",
        linewidths=0,
    )
    plt.colorbar(hb, ax=ax0, fraction=0.046, pad=0.04, label="log10(count+1)")

    for j, m in enumerate(hull_indices):
        row = p2d[m]
        idx = row[row >= 0]
        if idx.size == 0:
            continue
        if idx.size > max_points_per_patch_for_hull:
            sub = rng.choice(idx.size, size=max_points_per_patch_for_hull, replace=False)
            idx = idx[sub]
        xy_p = xy_all[idx]
        c = tab(norm(j % 20))
        _draw_patch_region(
            ax0,
            xy_p,
            color=c,
            alpha_fill=alpha_hull_fill,
            alpha_edge=alpha_hull_edge,
            max_scatter=fallback_scatter_per_patch,
            rng=rng,
        )

    ax0.scatter(
        xy_anchor[:, 0],
        xy_anchor[:, 1],
        s=anchor_size,
        c="red",
        marker="o",
        edgecolors="darkred",
        linewidths=0.4,
        label="anchors",
        zorder=5,
    )
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_title(
        f"{plot_title}\n(hexbin N={n_bg}; hulls drawn: {len(hull_indices)} of {a} patches)",
        fontsize=11,
    )
    ax0.set_xlabel(xl)
    ax0.set_ylabel(yl)
    ax0.grid(True, alpha=0.2)
    ax0.legend(loc="upper right", fontsize=8)

    # 右：仅锚点（全部）+ 轻量密度，便于看清锚点分布
    ax1 = axes[1]
    ax1.hexbin(
        xy_bg[:, 0],
        xy_bg[:, 1],
        gridsize=min(gridsize, 60),
        mincnt=1,
        cmap="Greys",
        bins="log",
        alpha=0.35,
        linewidths=0,
    )
    ax1.scatter(
        xy_anchor[:, 0],
        xy_anchor[:, 1],
        s=anchor_size * 0.85,
        c="crimson",
        marker="x",
        linewidths=0.6,
        label="anchors (all)",
        zorder=5,
    )
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title(right_subtitle, fontsize=11)
    ax1.set_xlabel(xl)
    ax1.set_ylabel(yl)
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc="upper right", fontsize=8)

    fig.suptitle(supt, fontsize=10, y=1.02)
    fig.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(out_png))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def _load_anchor_patch_default(patch_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    path_anchor = os.path.join(patch_dir, "anchor_train_anchor_idx.npy")
    path_patch = os.path.join(patch_dir, "anchor_train_patch_idx_2d.npz")
    if not os.path.isfile(path_anchor):
        raise FileNotFoundError(path_anchor)
    if not os.path.isfile(path_patch):
        raise FileNotFoundError(path_patch)
    anchor_idx = np.load(path_anchor)
    patch_2d = np.load(path_patch, allow_pickle=True)["0"]
    return anchor_idx, np.asarray(patch_2d, dtype=np.int64)


def main() -> None:
    p = argparse.ArgumentParser(description="锚点与 patch 的 2D 可视化（大规模友好）")
    p.add_argument("--patch_dir", type=str, default='/NAS/czt/mount/seis_flow_data12V2/h5/dongfang/patch', help="含 anchor_train_*.npy/npz 的目录")
    p.add_argument("--coord_obs_norm", type=str, default=None, help="[N,4] float32，默认 patch_dir/coord_obs_norm.npy")
    p.add_argument("--anchor_idx", type=str, default=None, help="覆盖默认 anchor_train_anchor_idx.npy")
    p.add_argument("--patch_idx_2d_npz", type=str, default=None, help="覆盖默认 anchor_train_patch_idx_2d.npz 中的数组，直接给 .npy [A,K] 也可")
    p.add_argument("--out_png", type=str, default='/NAS/czt/mount/seis_flow_data12V2/reg_tool/anchor_patches_visualization.png')
    p.add_argument(
        "--axes",
        type=int,
        nargs=2,
        metavar=("I", "J"),
        default=None,
        help="直接使用 coord 第 I、J 列作横纵轴（0..3），不 PCA。例：炮点 0 1，检波点 2 3",
    )
    p.add_argument("--xlabel", type=str, default=None, help="横轴标签（仅 --axes 时默认 shot_x/recv_x 等）")
    p.add_argument("--ylabel", type=str, default=None, help="纵轴标签")
    p.add_argument("--max_pca_fit", type=int, default=100_000)
    p.add_argument("--max_hexbin_points", type=int, default=400_0000)
    p.add_argument("--gridsize", type=int, default=80)
    p.add_argument("--max_patch_hulls", type=int, default=4000, help="最多绘制多少个 patch 的凸包（其余仍显示锚点）")
    p.add_argument("--max_points_per_patch_for_hull", type=int, default=256)
    p.add_argument("--fallback_scatter_per_patch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    if args.patch_dir is None and (args.anchor_idx is None or args.coord_obs_norm is None):
        raise SystemExit("需要 --patch_dir，或同时提供 --coord_obs_norm 与 --anchor_idx（及 patch 数组）")

    if args.patch_dir:
        patch_dir = os.path.abspath(args.patch_dir)
        path_coord = args.coord_obs_norm or os.path.join(patch_dir, "coord_obs_norm.npy")
        if not os.path.isfile(path_coord):
            raise FileNotFoundError(path_coord)
        coord = np.load(path_coord)

        anchor_idx, patch_2d = _load_anchor_patch_default(patch_dir)
        if args.anchor_idx is not None:
            anchor_idx = np.load(args.anchor_idx)
        if args.patch_idx_2d_npz is not None:
            if args.patch_idx_2d_npz.endswith(".npz"):
                patch_2d = np.load(args.patch_idx_2d_npz, allow_pickle=True)["0"]
            else:
                patch_2d = np.load(args.patch_idx_2d_npz)

        patch_2d = np.asarray(patch_2d, dtype=np.int64)
    else:
        coord = np.load(args.coord_obs_norm)
        anchor_idx = np.load(args.anchor_idx)
        if not args.patch_idx_2d_npz:
            raise SystemExit("无 --patch_dir 时必须指定 --patch_idx_2d_npz")
        if args.patch_idx_2d_npz.endswith(".npz"):
            patch_2d = np.load(args.patch_idx_2d_npz, allow_pickle=True)["0"]
        else:
            patch_2d = np.load(args.patch_idx_2d_npz)
        patch_2d = np.asarray(patch_2d, dtype=np.int64)

    axes_arg = None if args.axes is None else (int(args.axes[0]), int(args.axes[1]))

    run_visualize(
        coord,
        anchor_idx,
        patch_2d,
        out_png=args.out_png,
        axes=axes_arg,
        axis_xlabel=args.xlabel,
        axis_ylabel=args.ylabel,
        max_pca_fit=args.max_pca_fit,
        max_hexbin_points=args.max_hexbin_points,
        gridsize=args.gridsize,
        max_patch_hulls=args.max_patch_hulls,
        max_points_per_patch_for_hull=args.max_points_per_patch_for_hull,
        fallback_scatter_per_patch=args.fallback_scatter_per_patch,
        seed=args.seed,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
