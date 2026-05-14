"""Level-1 static anchor / patch quality metrics (coordinates only, no training).

Distance-heavy steps default to **GPU batched GEMM** (PyTorch) when available; NumPy CPU
fallback only when ``use_gpu=False`` or CUDA/MPS unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import tqdm

try:
    from .anchor_selector import _resolve_torch_device
    from .patch_sampler import (
        _check_coord_2d,
        _to_numpy,
        diverse_topk,
        normalize_coords,
        parse_metric_weights,
        top_l_neighbors,
    )
except ImportError:
    from anchor_selector import _resolve_torch_device
    from patch_sampler import (
        _check_coord_2d,
        _to_numpy,
        diverse_topk,
        normalize_coords,
        parse_metric_weights,
        top_l_neighbors,
    )

ArrayLike = Union[np.ndarray, Any]


def _maybe_tqdm(
    it: Iterable,
    *,
    show: bool,
    total: Optional[int] = None,
    desc: str = "",
    unit: str = "batch",
) -> Iterable:
    if not show:
        return it
    return tqdm.tqdm(it, total=total, desc=desc, unit=unit)


def _as_coord_obs_norm(coord_obs: ArrayLike, name: str = "coord_obs") -> np.ndarray:
    arr = _check_coord_2d(_to_numpy(coord_obs, np.float32), name)
    return arr


def _pad_patch_list_to_2d(
    patch_indices_list: Sequence[np.ndarray], pad_value: int = -1
) -> np.ndarray:
    """Pad ragged patch index lists to ``[A, K_max]`` (``pad_value`` for padding)."""
    if not patch_indices_list:
        return np.zeros((0, 0), dtype=np.int64)
    lens = [int(np.asarray(x, dtype=np.int64).size) for x in patch_indices_list]
    k_max = max(lens) if lens else 0
    a = len(patch_indices_list)
    out = np.full((a, k_max), pad_value, dtype=np.int64)
    for i, x in enumerate(patch_indices_list):
        p = np.asarray(x, dtype=np.int64).reshape(-1)
        if p.size:
            out[i, : p.size] = p
    return out


def _resolve_patch_idx_2d(
    anchor_idx: np.ndarray,
    coord_obs_norm: np.ndarray,
    *,
    patch_idx_2d: Optional[np.ndarray],
    patch_indices_list: Optional[Sequence[np.ndarray]],
    k_patch: int,
    top_l: int,
    metric_weights: Optional[Sequence[float]],
    beta: float,
    show_progress: bool,
) -> Tuple[np.ndarray, bool]:
    """Return ``patch_idx_2d`` shape ``[A, K]`` and ``built_from_anchor`` flag."""
    a = int(anchor_idx.size)
    if patch_idx_2d is not None:
        p2 = np.asarray(patch_idx_2d, dtype=np.int64)
        if p2.ndim != 2:
            raise ValueError(f"patch_idx_2d must be 2D [A, K], got {p2.shape}")
        if p2.shape[0] != a:
            raise ValueError(
                f"patch_idx_2d rows ({p2.shape[0]}) must match len(anchor_idx) ({a})"
            )
        return p2, False
    if patch_indices_list is not None:
        if len(patch_indices_list) != a:
            raise ValueError(
                f"len(patch_indices_list) ({len(patch_indices_list)}) must match len(anchor_idx) ({a})"
            )
        return _pad_patch_list_to_2d(patch_indices_list), False
    # Build from anchors (legacy)
    patch_list: List[np.ndarray] = []
    it = enumerate(anchor_idx)
    if show_progress:
        it = tqdm.tqdm(it, total=a, desc="build patch (per anchor)", unit="anchor")
    for _, aid in it:
        patch_list.append(
            build_patch_indices_from_anchor(
                int(aid),
                coord_obs_norm,
                k_patch=k_patch,
                top_l=top_l,
                metric_weights=metric_weights,
                beta=beta,
            )
        )
    return _pad_patch_list_to_2d(patch_list), True


def _patch_radius_from_2d(
    z: np.ndarray,
    anchor_idx: np.ndarray,
    patch_idx_2d: np.ndarray,
    *,
    use_gpu: bool,
    gpu_device: Optional[str],
) -> np.ndarray:
    """Per-anchor mean distance (embedding metric) from anchor to valid patch points."""
    import torch

    a, k = patch_idx_2d.shape
    if a != anchor_idx.size:
        raise ValueError("patch_idx_2d rows must match anchor_idx length")
    dev = _resolve_torch_device(gpu_device) if use_gpu else None
    if dev is not None:
        try:
            z_t = torch.as_tensor(z, dtype=torch.float64, device=dev)
            aid = torch.as_tensor(anchor_idx, dtype=torch.long, device=dev)
            pidx = torch.as_tensor(patch_idx_2d, dtype=torch.long, device=dev)
            valid = pidx >= 0
            safe = torch.where(valid, pidx, torch.zeros_like(pidx))
            z_a = z_t[aid]
            z_p = z_t[safe]
            d2 = torch.sum((z_p - z_a[:, None, :]) ** 2, dim=-1)
            dist = torch.sqrt(torch.clamp(d2, min=0.0))
            vf = valid.to(dtype=torch.float64)
            cnt = vf.sum(dim=-1)
            radii = torch.where(
                cnt > 0,
                (dist * vf).sum(dim=-1) / cnt,
                torch.zeros_like(cnt),
            )
            return radii.detach().cpu().numpy()
        except Exception as exc:
            warnings.warn(
                f"GPU patch radius failed ({exc}); using CPU NumPy.",
                UserWarning,
                stacklevel=2,
            )
    # CPU
    z_a = z[anchor_idx]
    valid = patch_idx_2d >= 0
    safe = np.where(valid, patch_idx_2d, 0)
    z_p = z[safe]
    d2 = np.sum((z_p - z_a[:, None, :]) ** 2, axis=-1)
    dist = np.sqrt(np.maximum(d2, 0.0))
    vf = valid.astype(np.float64)
    cnt = np.sum(vf, axis=-1)
    radii = np.where(cnt > 0, np.sum(dist * vf, axis=-1) / cnt, 0.0)
    return radii.astype(np.float64)


def _patch_diversity_from_2d(
    z: np.ndarray,
    patch_idx_2d: np.ndarray,
    *,
    use_gpu: bool,
    gpu_device: Optional[str],
) -> np.ndarray:
    """Per-patch mean pairwise distance in upper triangle (valid pairs only)."""
    import torch

    a, k = patch_idx_2d.shape
    dev = _resolve_torch_device(gpu_device) if use_gpu else None
    if dev is not None and k > 1:
        try:
            pidx = torch.as_tensor(patch_idx_2d, dtype=torch.long, device=dev)
            valid = pidx >= 0
            safe = torch.where(valid, pidx, torch.zeros_like(pidx))
            z_t = torch.as_tensor(z, dtype=torch.float64, device=dev)
            z_p = z_t[safe]
            d2 = torch.sum(
                (z_p[:, :, None, :] - z_p[:, None, :, :]) ** 2,
                dim=-1,
            )
            dist = torch.sqrt(torch.clamp(d2, min=0.0))
            triu = torch.triu(
                torch.ones((k, k), dtype=torch.bool, device=dev),
                diagonal=1,
            )
            pair_ok = valid[:, :, None] & valid[:, None, :] & triu[None, :, :]
            n_pairs = pair_ok.to(dtype=torch.float64).sum(dim=(-2, -1))
            divs = torch.where(
                n_pairs > 0,
                (dist * pair_ok.to(dtype=torch.float64)).sum(dim=(-2, -1)) / n_pairs,
                torch.zeros_like(n_pairs),
            )
            return divs.detach().cpu().numpy()
        except Exception as exc:
            warnings.warn(
                f"GPU patch diversity failed ({exc}); using CPU NumPy.",
                UserWarning,
                stacklevel=2,
            )
    # CPU
    divs = np.empty(a, dtype=np.float64)
    for m in range(a):
        pidx = np.asarray(patch_idx_2d[m], dtype=np.int64).reshape(-1)
        pidx = pidx[pidx >= 0]
        if pidx.size < 2:
            divs[m] = 0.0
            continue
        pts = z[pidx]
        d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
        iu = np.triu_indices(pidx.size, k=1)
        divs[m] = float(np.sqrt(np.maximum(d2[iu], 0.0)).mean())
    return divs


def _patch_overlap_from_2d(
    patch_idx_2d: np.ndarray,
    num_obs: int,
    *,
    max_pairs: Optional[int],
    rng: Optional[np.random.Generator],
    show_progress: bool,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Jaccard-style overlap: ``|P_i∩P_j|/min(|P_i|,|P_j|)`` via sparse ``M @ M.T``."""
    a, k = patch_idx_2d.shape
    if a < 2:
        return (
            {
                "mean_patch_overlap": float("nan"),
                "median_patch_overlap": float("nan"),
                "p95_patch_overlap": float("nan"),
                "max_patch_overlap": float("nan"),
            },
            np.zeros((0,), dtype=np.float64),
        )

    valid = patch_idx_2d >= 0
    rows = np.repeat(np.arange(a, dtype=np.int64), k)
    cols = patch_idx_2d.reshape(-1).astype(np.int64)
    vflat = valid.reshape(-1)
    rows = rows[vflat]
    cols = cols[vflat]
    keep = cols >= 0
    rows = rows[keep]
    cols = cols[keep]
    if cols.size and (np.max(cols) >= num_obs):
        raise ValueError(
            f"patch index max {int(np.max(cols))} >= num_obs ({num_obs})"
        )

    nnz = rows.size
    try:
        import scipy.sparse as sp
    except ImportError:
        sp = None
    if sp is not None and nnz > 0:
        M = sp.csr_matrix(
            (np.ones(nnz, dtype=np.float64), (rows, cols)),
            shape=(a, num_obs),
        )
        G = M @ M.T
        sizes = np.asarray(M.sum(axis=1)).ravel()
    else:
        if num_obs * a > 50_000_000:
            raise ImportError(
                "需要 scipy.sparse 以在大型观测集上计算 patch overlap（避免稠密 [A,N]）"
            )
        M = np.zeros((a, num_obs), dtype=np.float64)
        M[rows, cols] = 1.0
        G = M @ M.T
        sizes = np.asarray(M.sum(axis=1)).ravel()

    pairs = [(i, j) for i in range(a) for j in range(i + 1, a)]
    total = len(pairs)
    cap = total if max_pairs is None else min(max_pairs, total)
    if total > cap:
        rng = rng or np.random.default_rng(0)
        sel = rng.choice(total, size=cap, replace=False)
        pairs = [pairs[int(t)] for t in sel]
    vals = np.empty(len(pairs), dtype=np.float64)
    it = enumerate(pairs)
    if show_progress:
        it = tqdm.tqdm(it, total=len(pairs), desc="patch overlap (pairs)", unit="pair")
    for t, (ii, jj) in it:
        inter = float(G[ii, jj])
        denom = min(float(sizes[ii]), float(sizes[jj]))
        vals[t] = 0.0 if denom <= 0 else inter / denom
    summary = {
        "mean_patch_overlap": float(np.mean(vals)),
        "median_patch_overlap": float(np.median(vals)),
        "p95_patch_overlap": float(np.percentile(vals, 95)),
        "max_patch_overlap": float(np.max(vals)),
    }
    return summary, vals


def _embedding_z(coords: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    return coords.astype(np.float64) * np.sqrt(w, dtype=np.float64)


def _nearest_dist_obs_to_anchors_cpu(
    z: np.ndarray,
    z_norm2: np.ndarray,
    za: np.ndarray,
    za_norm2: np.ndarray,
    batch_rows: int = 2048,
    show_progress: bool = True,
) -> np.ndarray:
    n = z.shape[0]
    br = max(1, int(batch_rows))
    n_batch = (n + br - 1) // br
    out = np.empty(n, dtype=np.float64)
    for s in _maybe_tqdm(
        range(0, n, br),
        show=show_progress,
        total=n_batch,
        desc="coverage dist (CPU batches)",
        unit="batch",
    ):
        e = min(s + br, n)
        blk = z[s:e]
        d2 = z_norm2[s:e, None] + za_norm2[None, :] - 2.0 * (blk @ za.T)
        np.maximum(d2, 0.0, out=d2)
        out[s:e] = np.sqrt(np.min(d2, axis=1))
    return out


def _nearest_dist_obs_to_anchors_gpu(
    z: np.ndarray,
    z_norm2: np.ndarray,
    za: np.ndarray,
    za_norm2: np.ndarray,
    device: Any,
    batch_rows: int = 4096,
    show_progress: bool = True,
) -> np.ndarray:
    import torch

    z_t = torch.as_tensor(z, dtype=torch.float64, device=device)
    zn_t = torch.as_tensor(z_norm2, dtype=torch.float64, device=device)
    za_t = torch.as_tensor(za, dtype=torch.float64, device=device)
    zan_t = torch.as_tensor(za_norm2, dtype=torch.float64, device=device)
    n = z.shape[0]
    out = np.empty(n, dtype=np.float64)
    br = max(1, int(batch_rows))
    n_batch = (n + br - 1) // br
    for s in _maybe_tqdm(
        range(0, n, br),
        show=show_progress,
        total=n_batch,
        desc="coverage dist (GPU batches)",
        unit="batch",
    ):
        e = min(s + br, n)
        blk = z_t[s:e]
        d2 = zn_t[s:e, None] + zan_t[None, :] - 2.0 * (blk @ za_t.T)
        d2 = torch.clamp(d2, min=0.0)
        out[s:e] = torch.sqrt(torch.min(d2, dim=1).values).detach().cpu().numpy()
    return out


def _nearest_dist_obs_to_anchors(
    z: np.ndarray,
    z_norm2: np.ndarray,
    za: np.ndarray,
    za_norm2: np.ndarray,
    *,
    use_gpu: bool = True,
    gpu_device: Optional[str] = None,
    gpu_batch_rows: int = 4096,
    cpu_batch_rows: int = 2048,
    show_progress: bool = True,
) -> np.ndarray:
    if use_gpu:
        dev = _resolve_torch_device(gpu_device)
        if dev is not None:
            try:
                return _nearest_dist_obs_to_anchors_gpu(
                    z,
                    z_norm2,
                    za,
                    za_norm2,
                    dev,
                    batch_rows=gpu_batch_rows,
                    show_progress=show_progress,
                )
            except Exception as exc:
                warnings.warn(
                    f"GPU coverage distances failed ({exc}); using CPU NumPy.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            warnings.warn(
                "use_gpu=True but no CUDA/MPS device; using CPU NumPy.",
                UserWarning,
                stacklevel=2,
            )
    return _nearest_dist_obs_to_anchors_cpu(
        z,
        z_norm2,
        za,
        za_norm2,
        batch_rows=cpu_batch_rows,
        show_progress=show_progress,
    )


def _nearest_dist_each_anchor_to_others_cpu(
    za: np.ndarray,
    za_norm2: np.ndarray,
    batch_rows: int = 512,
    show_progress: bool = True,
) -> np.ndarray:
    a = za.shape[0]
    if a <= 1:
        return np.full(a, np.inf, dtype=np.float64)
    br = max(1, int(batch_rows))
    n_batch = (a + br - 1) // br
    nn = np.empty(a, dtype=np.float64)
    for i in _maybe_tqdm(
        range(0, a, br),
        show=show_progress,
        total=n_batch,
        desc="redundancy dist (CPU batches)",
        unit="batch",
    ):
        e = min(i + br, a)
        bi = za[i:e]
        d2 = za_norm2[i:e, None] + za_norm2[None, :] - 2.0 * (bi @ za.T)
        np.maximum(d2, 0.0, out=d2)
        for r in range(e - i):
            d2[r, i + r] = np.inf
        nn[i:e] = np.sqrt(np.min(d2, axis=1))
    return nn


def _nearest_dist_each_anchor_to_others_gpu(
    za: np.ndarray,
    za_norm2: np.ndarray,
    device: Any,
    batch_rows: int = 512,
    show_progress: bool = True,
) -> np.ndarray:
    import torch

    a = za.shape[0]
    if a <= 1:
        return np.full(a, np.inf, dtype=np.float64)
    za_t = torch.as_tensor(za, dtype=torch.float64, device=device)
    zan_t = torch.as_tensor(za_norm2, dtype=torch.float64, device=device)
    nn = np.empty(a, dtype=np.float64)
    br = max(1, int(batch_rows))
    n_batch = (a + br - 1) // br
    for i in _maybe_tqdm(
        range(0, a, br),
        show=show_progress,
        total=n_batch,
        desc="redundancy dist (GPU batches)",
        unit="batch",
    ):
        e = min(i + br, a)
        bi = za_t[i:e]
        d2 = zan_t[i:e, None] + zan_t[None, :] - 2.0 * (bi @ za_t.T)
        d2 = torch.clamp(d2, min=0.0)
        rows = torch.arange(e - i, device=device)
        cols = torch.arange(i, e, device=device, dtype=torch.long)
        d2[rows, cols] = float("inf")
        nn[i:e] = torch.sqrt(torch.min(d2, dim=1).values).detach().cpu().numpy()
    return nn


def _nearest_dist_each_anchor_to_others(
    za: np.ndarray,
    za_norm2: np.ndarray,
    *,
    use_gpu: bool = True,
    gpu_device: Optional[str] = None,
    gpu_batch_rows: int = 512,
    cpu_batch_rows: int = 512,
    show_progress: bool = True,
) -> np.ndarray:
    if use_gpu:
        dev = _resolve_torch_device(gpu_device)
        if dev is not None:
            try:
                return _nearest_dist_each_anchor_to_others_gpu(
                    za,
                    za_norm2,
                    dev,
                    batch_rows=gpu_batch_rows,
                    show_progress=show_progress,
                )
            except Exception as exc:
                warnings.warn(
                    f"GPU redundancy distances failed ({exc}); using CPU NumPy.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            warnings.warn(
                "use_gpu=True but no CUDA/MPS device; using CPU NumPy.",
                UserWarning,
                stacklevel=2,
            )
    return _nearest_dist_each_anchor_to_others_cpu(
        za, za_norm2, batch_rows=cpu_batch_rows, show_progress=show_progress
    )


def build_patch_indices_from_anchor(
    anchor_idx: int,
    coord_obs_norm: ArrayLike,
    k_patch: int = 64,
    top_l: int = 128,
    metric_weights: Optional[Sequence[float]] = None,
    beta: float = 0.3,
) -> np.ndarray:
    """Same pipeline as ``build_train_patch`` (top_l → diverse_topk), indices only.

    No trace access, no random query split.
    """
    coords = _as_coord_obs_norm(coord_obs_norm, "coord_obs_norm")
    n = coords.shape[0]
    if anchor_idx < 0 or anchor_idx >= n:
        raise ValueError(f"anchor_idx out of range [0, {n})")
    anchor_coord = coords[anchor_idx]
    candidate_idx, _ = top_l_neighbors(
        center_coord=anchor_coord,
        all_coords=coords,
        top_l=int(top_l),
        metric_weights=metric_weights,
        exclude_self=False,
    )
    patch_idx = diverse_topk(
        center_coord=anchor_coord,
        candidate_idx=candidate_idx,
        all_coords=coords,
        k=int(k_patch),
        metric_weights=metric_weights,
        beta=float(beta),
    )
    return patch_idx.astype(np.int64)


def compute_anchor_coverage_metrics(
    anchor_idx: ArrayLike,
    coord_obs_norm: ArrayLike,
    metric_weights: Optional[Sequence[float]] = None,
    use_gpu: bool = True,
    gpu_device: Optional[str] = None,
    gpu_batch_rows: int = 4096,
    cpu_batch_rows: int = 2048,
    show_progress: bool = True,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Per observation: distance to nearest anchor (weighted Euclidean in normalized coords)."""
    coords = _as_coord_obs_norm(coord_obs_norm, "coord_obs_norm")
    anchors = np.asarray(anchor_idx, dtype=np.int64).reshape(-1)
    if anchors.size == 0:
        raise ValueError("anchor_idx is empty")
    if np.any(anchors < 0) or np.any(anchors >= coords.shape[0]):
        raise ValueError("anchor_idx out of range")
    w = parse_metric_weights(metric_weights)
    z = _embedding_z(coords, w)
    za = z[anchors]
    z_norm2 = np.sum(z * z, axis=1)
    za_norm2 = np.sum(za * za, axis=1)
    r = _nearest_dist_obs_to_anchors(
        z,
        z_norm2,
        za,
        za_norm2,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
        gpu_batch_rows=gpu_batch_rows,
        cpu_batch_rows=cpu_batch_rows,
        show_progress=show_progress,
    )
    summary = {
        "mean_nearest_anchor_dist": float(np.mean(r)),
        "median_nearest_anchor_dist": float(np.median(r)),
        "p95_nearest_anchor_dist": float(np.percentile(r, 95)),
        "max_nearest_anchor_dist": float(np.max(r)),
    }
    return summary, r


def compute_anchor_redundancy_metrics(
    anchor_idx: ArrayLike,
    coord_obs_norm: ArrayLike,
    metric_weights: Optional[Sequence[float]] = None,
    use_gpu: bool = True,
    gpu_device: Optional[str] = None,
    gpu_batch_rows: int = 512,
    cpu_batch_rows: int = 512,
    show_progress: bool = True,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Per anchor: distance to nearest *other* anchor."""
    coords = _as_coord_obs_norm(coord_obs_norm, "coord_obs_norm")
    anchors = np.asarray(anchor_idx, dtype=np.int64).reshape(-1)
    if anchors.size <= 1:
        u = np.full(anchors.size, np.inf, dtype=np.float64)
        summary = {
            "mean_anchor_nn_dist": float("nan"),
            "median_anchor_nn_dist": float("nan"),
            "min_anchor_nn_dist": float("nan"),
            "p05_anchor_nn_dist": float("nan"),
        }
        return summary, u
    w = parse_metric_weights(metric_weights)
    z = _embedding_z(coords, w)
    za = z[anchors]
    za_norm2 = np.sum(za * za, axis=1)
    u = _nearest_dist_each_anchor_to_others(
        za,
        za_norm2,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
        gpu_batch_rows=gpu_batch_rows,
        cpu_batch_rows=cpu_batch_rows,
        show_progress=show_progress,
    )
    summary = {
        "mean_anchor_nn_dist": float(np.mean(u)),
        "median_anchor_nn_dist": float(np.median(u)),
        "min_anchor_nn_dist": float(np.min(u)),
        "p05_anchor_nn_dist": float(np.percentile(u, 5)),
    }
    return summary, u


def compute_patch_radius_metrics(
    anchor_idx: ArrayLike,
    coord_obs_norm: ArrayLike,
    patch_idx_2d: Optional[np.ndarray] = None,
    patch_indices_list: Optional[Sequence[np.ndarray]] = None,
    k_patch: int = 64,
    top_l: int = 128,
    metric_weights: Optional[Sequence[float]] = None,
    beta: float = 0.3,
    use_gpu: bool = True,
    gpu_device: Optional[str] = None,
    show_progress: bool = True,
) -> Tuple[Dict[str, float], np.ndarray, List[np.ndarray]]:
    """Per anchor: mean weighted distance (embedding metric) from anchor to patch points.

    若提供 ``patch_idx_2d``（``[A,K]``，无效位为 ``-1``）或 ``patch_indices_list``，
    则不再调用 ``build_patch_indices_from_anchor``。否则按 ``k_patch/top_l/...`` 现场构造。
    距离计算在 GPU 上按锚点批量完成（失败则回退 CPU）。
    """
    coords = _as_coord_obs_norm(coord_obs_norm, "coord_obs_norm")
    anchors = np.asarray(anchor_idx, dtype=np.int64).reshape(-1)
    w = parse_metric_weights(metric_weights)
    z = _embedding_z(coords, w)
    p2d, _ = _resolve_patch_idx_2d(
        anchors,
        coords,
        patch_idx_2d=patch_idx_2d,
        patch_indices_list=patch_indices_list,
        k_patch=k_patch,
        top_l=top_l,
        metric_weights=metric_weights,
        beta=beta,
        show_progress=show_progress,
    )
    radii = _patch_radius_from_2d(
        z, anchors, p2d, use_gpu=use_gpu, gpu_device=gpu_device
    )
    patch_list = [
        np.asarray(p2d[i][p2d[i] >= 0], dtype=np.int64).reshape(-1)
        for i in range(anchors.size)
    ]
    summary = {
        "mean_patch_radius": float(np.mean(radii)),
        "median_patch_radius": float(np.median(radii)),
        "p95_patch_radius": float(np.percentile(radii, 95)),
        "max_patch_radius": float(np.max(radii)),
    }
    return summary, radii, patch_list


def compute_patch_diversity_metrics(
    coord_obs_norm: ArrayLike,
    *,
    patch_indices_list: Optional[Sequence[np.ndarray]] = None,
    patch_idx_2d: Optional[np.ndarray] = None,
    metric_weights: Optional[Sequence[float]] = None,
    use_gpu: bool = True,
    gpu_device: Optional[str] = None,
    show_progress: bool = True,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Per patch: mean pairwise distance in embedding space（GPU 批量或 CPU）。"""
    _ = show_progress  # 保留 API；批量实现无逐 patch 进度条
    coords = _as_coord_obs_norm(coord_obs_norm, "coord_obs_norm")
    w = parse_metric_weights(metric_weights)
    z = _embedding_z(coords, w)
    if patch_idx_2d is not None:
        p2d = np.asarray(patch_idx_2d, dtype=np.int64)
    elif patch_indices_list is not None:
        p2d = _pad_patch_list_to_2d(patch_indices_list)
    else:
        raise ValueError("必须提供 patch_idx_2d 或 patch_indices_list")
    divs = _patch_diversity_from_2d(
        z, p2d, use_gpu=use_gpu, gpu_device=gpu_device
    )
    summary = {
        "mean_patch_pairwise_dist": float(np.mean(divs)),
        "median_patch_pairwise_dist": float(np.median(divs)),
        "p95_patch_pairwise_dist": float(np.percentile(divs, 95)),
    }
    return summary, divs


def compute_patch_overlap_metrics(
    *,
    patch_indices_list: Optional[Sequence[np.ndarray]] = None,
    patch_idx_2d: Optional[np.ndarray] = None,
    num_obs: Optional[int] = None,
    max_pairs: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    show_progress: bool = True,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Jaccard-style overlap，稀疏关联矩阵 ``M M^T`` 上取交集规模（SciPy）。"""
    if patch_idx_2d is not None:
        p2d = np.asarray(patch_idx_2d, dtype=np.int64)
    elif patch_indices_list is not None:
        p2d = _pad_patch_list_to_2d(patch_indices_list)
    else:
        raise ValueError("必须提供 patch_idx_2d 或 patch_indices_list")
    if num_obs is None:
        v = p2d[p2d >= 0]
        num_obs = int(np.max(v)) + 1 if v.size else 0
    return _patch_overlap_from_2d(
        p2d,
        int(num_obs),
        max_pairs=max_pairs,
        rng=rng,
        show_progress=show_progress,
    )


def evaluate_anchor_quality_level1(
    coord_obs: ArrayLike,
    anchor_idx: ArrayLike,
    patch_idx_2d: Optional[ArrayLike] = None,
    patch_indices_list: Optional[Sequence[np.ndarray]] = None,
    k_patch: int = 64,
    top_l: int = 128,
    metric_weights: Optional[Sequence[float]] = None,
    beta: float = 0.3,
    normalize: bool = True,
    return_debug: bool = False,
    use_gpu: bool = True,
    gpu_device: Optional[str] = None,
    gpu_batch_rows_obs: int = 4096,
    gpu_batch_rows_anchor: int = 512,
    cpu_batch_rows_obs: int = 2048,
    cpu_batch_rows_anchor: int = 512,
    overlap_max_pairs: Optional[int] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Run coverage, redundancy, patch radius/diversity, and overlap metrics.

    Coverage and anchor–anchor redundancy use **GPU batched GEMM** by default
    (``use_gpu=True``); set ``use_gpu=False`` for CPU-only NumPy.

    Patch 相关指标：若提供 ``patch_idx_2d``（``[A,K]``，``-1`` 填充）或 ``patch_indices_list``，
    则不再按 ``k_patch/top_l`` 重算 patch；否则与训练侧一致现场构造 patch。

    Args:
        patch_idx_2d: 预计算 patch 全局索引，与 ``anchor_idx`` 行对齐。
        gpu_batch_rows_obs: rows of observations per GPU matmul block (coverage).
        gpu_batch_rows_anchor: rows of anchors per GPU matmul block (redundancy).
        cpu_batch_rows_*: batch sizes when falling back to CPU.
        overlap_max_pairs: cap on unordered patch pairs for overlap (None = all if not too many,
            else subsample to min(total_pairs, 500_000)).
        show_progress: if True, show tqdm bars for batched distance loops and per-anchor/patch steps.
    """
    raw = _as_coord_obs_norm(coord_obs, "coord_obs")
    if normalize:
        coord_obs_norm, _ = normalize_coords(raw)
    else:
        coord_obs_norm = raw

    cov_sum, r_all = compute_anchor_coverage_metrics(
        anchor_idx,
        coord_obs_norm,
        metric_weights=metric_weights,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
        gpu_batch_rows=gpu_batch_rows_obs,
        cpu_batch_rows=cpu_batch_rows_obs,
        show_progress=show_progress,
    )
    print(cov_sum)
    red_sum, u_all = compute_anchor_redundancy_metrics(
        anchor_idx,
        coord_obs_norm,
        metric_weights=metric_weights,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
        gpu_batch_rows=gpu_batch_rows_anchor,
        cpu_batch_rows=cpu_batch_rows_anchor,
        show_progress=show_progress,
    )
    print(red_sum)
    p2d_arg = None if patch_idx_2d is None else np.asarray(patch_idx_2d, dtype=np.int64)
    rad_sum, radii, patch_list = compute_patch_radius_metrics(
        anchor_idx,
        coord_obs_norm,
        patch_idx_2d=p2d_arg,
        patch_indices_list=patch_indices_list,
        k_patch=k_patch,
        top_l=top_l,
        metric_weights=metric_weights,
        beta=beta,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
        show_progress=show_progress,
    )
    # 与 radius 使用同一套 patch：优先 ndarray，其次 list，否则用刚构造的 patch_list
    div_patch_arg: Optional[Sequence[np.ndarray]]
    if p2d_arg is not None:
        div_patch_arg = None
        div_p2d = p2d_arg
    elif patch_indices_list is not None:
        div_patch_arg = patch_indices_list
        div_p2d = None
    else:
        div_patch_arg = patch_list
        div_p2d = None
    div_sum, div_per = compute_patch_diversity_metrics(
        coord_obs_norm,
        patch_idx_2d=div_p2d,
        patch_indices_list=div_patch_arg,
        metric_weights=metric_weights,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
        show_progress=show_progress,
    )

    m = len(patch_list)
    total_pairs = m * (m - 1) // 2 if m >= 2 else 0
    if overlap_max_pairs is not None:
        overlap_cap = overlap_max_pairs
    elif total_pairs > 500_000:
        overlap_cap = 500_000
    else:
        overlap_cap = None
    if p2d_arg is not None:
        ovl_p2d = p2d_arg
        ovl_list = None
    elif patch_indices_list is not None:
        ovl_p2d = None
        ovl_list = patch_indices_list
    else:
        ovl_p2d = None
        ovl_list = patch_list
    ovl_sum, ovl_vals = compute_patch_overlap_metrics(
        patch_idx_2d=ovl_p2d,
        patch_indices_list=ovl_list,
        num_obs=int(coord_obs_norm.shape[0]),
        max_pairs=overlap_cap,
        show_progress=show_progress,
    )

    out: Dict[str, Any] = {
        "coord_obs_norm": coord_obs_norm.astype(np.float32),
        "coverage": cov_sum,
        "redundancy": red_sum,
        "patch_radius": rad_sum,
        "patch_diversity": div_sum,
        "patch_overlap": ovl_sum,
    }
    if return_debug:
        out["debug"] = {
            "per_point_nearest_anchor_dist": r_all.astype(np.float64),
            "per_anchor_nn_dist": u_all.astype(np.float64),
            "per_anchor_patch_radius": radii.astype(np.float64),
            "per_patch_pairwise_mean_dist": div_per.astype(np.float64),
            "overlap_values": ovl_vals.astype(np.float64),
            "patch_indices_list": patch_list,
        }
    return out


def _print_level1_report(report: Dict[str, Any]) -> None:
    """Pretty-print ``evaluate_anchor_quality_level1`` result (no debug arrays)."""
    for section in ("coverage", "redundancy", "patch_radius", "patch_diversity", "patch_overlap"):
        print(f"\n--- {section} ---")
        for k, v in report[section].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.8g}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    # 与 anchor_patch_debug / core anchor_patch 落盘文件名一致：
    #   anchor_train_patch_idx_2d.npz, anchor_train_anchor_idx.npy, anchor_train_anchor_coord.npy
    # 全量评估另需观测归一化坐标 [N_obs,4]，可用 --coord_obs_norm 或事先保存 patch_dir/coord_obs_norm.npy
    parser = argparse.ArgumentParser(
        description="Load patch_dir 锚点索引并运行第一层几何评估（evaluate_anchor_quality_level1）",
    )
    parser.add_argument(
        "--patch_dir",
        type=str,
        default="/NAS/czt/mount/seis_flow_data12V2/h5/dongfang/patch",
        help="含 anchor_train_*.npy/npz 的目录（与 notebook 落盘一致）",
    )
    parser.add_argument(
        "--coord_obs_norm",
        type=str,
        default=None,
        help="观测归一化坐标 [N,4] float32 .npy；默认尝试 patch_dir/coord_obs_norm.npy",
    )
    parser.add_argument("--k_patch", type=int, default=64)
    parser.add_argument("--top_l", type=int, default=128)
    parser.add_argument(
        "--metric_weights",
        type=str,
        default="1,1,0.5,0.5",
        help="逗号分隔 4 个权重",
    )
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--no_gpu", action="store_true", help="强制 CPU（默认尝试 GPU）")
    parser.add_argument("--gpu_device", type=str, default=None)
    parser.add_argument("--overlap_max_pairs", type=int, default=None)
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument(
        "--save_report_json",
        type=str,
        default=None,
        help="将 summary 指标写入该路径（JSON，不含大数组）",
    )
    parser.add_argument(
        "--rebuild_patch",
        action="store_true",
        help="忽略 anchor_train_patch_idx_2d.npz，按 k_patch/top_l 现场重算 patch",
    )
    args = parser.parse_args()
    use_gpu = not args.no_gpu

    patch_dir = os.path.abspath(args.patch_dir)
    path_anchor = os.path.join(patch_dir, "anchor_train_anchor_idx.npy")
    path_patch2d = os.path.join(patch_dir, "anchor_train_patch_idx_2d.npz")
    path_anchor_coord = os.path.join(patch_dir, "anchor_train_anchor_coord.npy")

    if not os.path.isfile(path_anchor):
        raise FileNotFoundError(f"未找到锚点索引: {path_anchor}")

    anchor_idx = np.load(path_anchor)
    print(f"loaded anchor_train_anchor_idx.npy  shape={anchor_idx.shape} dtype={anchor_idx.dtype}")

    patch_2d_file = None
    if os.path.isfile(path_patch2d):
        patch_2d_file = np.load(path_patch2d, allow_pickle=True)["0"]
        print(
            f"loaded anchor_train_patch_idx_2d.npz  shape={patch_2d_file.shape} dtype={patch_2d_file.dtype}"
        )
    else:
        print(f"warning: 未找到 {path_patch2d}，patch 指标将按 k_patch/top_l 现场构造")

    patch_idx_for_eval: Optional[np.ndarray] = None
    if patch_2d_file is not None and not args.rebuild_patch:
        patch_idx_for_eval = np.asarray(patch_2d_file, dtype=np.int64)
        if patch_idx_for_eval.shape[0] != anchor_idx.size:
            raise ValueError(
                f"patch_idx_2d 行数 {patch_idx_for_eval.shape[0]} 与 anchor 数 {anchor_idx.size} 不一致"
            )
        print("使用磁盘上的 patch_idx_2d 作为 patch 指标输入（不重算 top_l/diverse_topk）")
    elif args.rebuild_patch:
        print("已指定 --rebuild_patch：忽略磁盘 patch 文件，按 k_patch/top_l 重算")

    if os.path.isfile(path_anchor_coord):
        ac = np.load(path_anchor_coord)
        print(f"loaded anchor_train_anchor_coord.npy  shape={ac.shape}")

    coord_path = args.coord_obs_norm or os.path.join(patch_dir, "coord_obs_norm.npy")
    if not os.path.isfile(coord_path):
        raise FileNotFoundError(
            f"需要全量观测归一化坐标以计算覆盖等指标。请指定 --coord_obs_norm，或在生成 patch 后执行一次:\n"
            f"  np.save(r'{os.path.join(patch_dir, 'coord_obs_norm.npy')}', coord_obs_norm.astype(np.float32))\n"
            f"当前尝试路径: {coord_path}"
        )
    coord_obs_norm = np.load(coord_path)
    print(f"loaded coord_obs_norm  shape={coord_obs_norm.shape} from {coord_path}")

    n_obs = int(coord_obs_norm.shape[0])
    if np.any(anchor_idx < 0) or np.any(anchor_idx >= n_obs):
        raise ValueError("anchor_idx 与 coord_obs_norm 行数不一致或索引越界")

    mw = [float(x) for x in args.metric_weights.split(",")]
    if len(mw) != 4:
        raise ValueError("--metric_weights 需要 4 个逗号分隔数值")

    report = evaluate_anchor_quality_level1(
        coord_obs_norm,
        anchor_idx,
        patch_idx_2d=patch_idx_for_eval,
        k_patch=args.k_patch,
        top_l=args.top_l,
        metric_weights=mw,
        beta=args.beta,
        normalize=False,
        return_debug=False,
        use_gpu=use_gpu,
        gpu_device=args.gpu_device,
        overlap_max_pairs=args.overlap_max_pairs,
        show_progress=not args.no_progress,
    )

    _print_level1_report(report)

    if args.save_report_json:
        serializable = {
            "patch_dir": patch_dir,
            "coord_obs_norm_path": coord_path,
            "k_patch": args.k_patch,
            "top_l": args.top_l,
            "metric_weights": mw,
            "beta": args.beta,
            **{k: report[k] for k in ("coverage", "redundancy", "patch_radius", "patch_diversity", "patch_overlap")},
        }
        with open(args.save_report_json, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"\nWrote JSON report to {args.save_report_json}")