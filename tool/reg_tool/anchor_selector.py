"""Coordinate-only anchor selection (NumPy only).

统一提供：
- 坐标归一化 ``normalize_coords``
- 加权欧氏距离工具 ``parse_metric_weights`` / ``weighted_sqdist_to_one``
- 锚点选择：
  - ``farthest_point_sampling``（FPS，子集候选）
  - ``facility_location_anchor_sampling``（Facility Location 贪心）
  - ``value_based_anchor_sampling``（价值分数 + 贪心 + 邻域抑制）

不包含 patch 构造、KDTree、学习特征等。

可选：若安装 PyTorch 且启用 ``use_gpu``，``_per_point_kth_neighbor_dist`` 可在 GPU 上分块计算 kNN 距离（CUDA 推荐）。
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tqdm

ArrayLike = Union[np.ndarray, Any]


def _to_numpy(array: ArrayLike, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Convert ndarray / torch.Tensor-like object to NumPy array."""
    if isinstance(array, np.ndarray):
        out = array
    elif hasattr(array, "detach") and hasattr(array, "cpu") and hasattr(array, "numpy"):
        out = array.detach().cpu().numpy()
    else:
        out = np.asarray(array)
    if dtype is not None:
        out = out.astype(dtype, copy=False)
    return out


def _check_coord_2d(coords: np.ndarray, name: str) -> np.ndarray:
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"{name} must have shape [N, 4], got {coords.shape}")
    return coords


def _as_float2d(x: ArrayLike, name: str) -> np.ndarray:
    arr = _to_numpy(x, np.float32)
    return _check_coord_2d(arr, name)


def parse_metric_weights(metric_weights: Optional[Sequence[float]]) -> np.ndarray:
    """Parse metric weights, default [1, 1, 0.5, 0.5]. Shape [4]."""
    if metric_weights is None:
        metric_weights = [1.0, 1.0, 0.5, 0.5]
    w = np.asarray(metric_weights, dtype=np.float64).reshape(-1)
    if w.shape[0] != 4:
        raise ValueError(f"metric_weights must have length 4, got {w.shape[0]}")
    if np.any(w < 0):
        raise ValueError("metric_weights must be non-negative")
    return w


def _normalize_minmax_neg1_to_1(coords: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    min_v = np.min(coords, axis=0)
    max_v = np.max(coords, axis=0)
    range_v = max_v - min_v
    safe_range = np.where(range_v > 0.0, range_v, 1.0)
    coords_norm = 2.0 * (coords - min_v) / safe_range - 1.0
    stats = {
        "min": min_v.astype(np.float32),
        "max": max_v.astype(np.float32),
        "mean": np.mean(coords, axis=0).astype(np.float32),
        "std": np.std(coords, axis=0).astype(np.float32),
    }
    return coords_norm.astype(np.float32), stats


def normalize_coords(
    coord_obs: ArrayLike,
    coord_grid: Optional[ArrayLike] = None,
):
    """Min-max normalize coordinates to [-1, 1] per dimension.

    Args:
        coord_obs: observed coordinates, shape [N_obs, 4]
        coord_grid: optional grid coordinates, shape [N_grid, 4]

    Returns:
        If coord_grid is None: ``coord_obs_norm, stats``
        Else: ``coord_obs_norm, coord_grid_norm, stats``
        ``stats`` has keys ``obs`` / ``grid`` with min/max/mean/std.
    """
    coord_obs_np = _check_coord_2d(_to_numpy(coord_obs, np.float32), "coord_obs")
    coord_obs_norm, obs_stats = _normalize_minmax_neg1_to_1(coord_obs_np)
    stats: Dict[str, Dict[str, np.ndarray]] = {"obs": obs_stats}

    if coord_grid is None:
        return coord_obs_norm, stats

    coord_grid_np = _check_coord_2d(_to_numpy(coord_grid, np.float32), "coord_grid")
    coord_grid_norm, grid_stats = _normalize_minmax_neg1_to_1(coord_grid_np)
    stats["grid"] = grid_stats
    return coord_obs_norm, coord_grid_norm, stats


def weighted_sqdist_to_one(
    center_coord: ArrayLike,
    all_coords: ArrayLike,
    metric_weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Weighted squared distance from one center to all points.

    Args:
        center_coord: shape [4]
        all_coords: shape [N, 4]

    Returns:
        d2: shape [N]
    """
    center = _to_numpy(center_coord, np.float32).reshape(-1)
    if center.shape[0] != 4:
        raise ValueError(f"center_coord must have shape [4], got {center.shape}")
    coords = _as_float2d(all_coords, "all_coords")
    w = parse_metric_weights(metric_weights)
    diff = coords - center[None, :]
    d2 = np.sum((diff * diff) * w[None, :], axis=1)
    return d2.astype(np.float64)


def farthest_point_sampling(
    coords: ArrayLike,
    candidate_idx: ArrayLike,
    num_anchors: int,
    metric_weights: Optional[Sequence[float]] = None,
    seed: int = 0,
) -> np.ndarray:
    """FPS on a candidate subset; returns global indices into ``coords``.

    Args:
        coords: all coordinates, shape [N, 4]
        candidate_idx: candidate global indices, shape [N_cand]
        num_anchors: number of anchors
        metric_weights: weighted Euclidean, length 4
        seed: RNG seed for first anchor

    Returns:
        anchor_idx: shape [M], M <= num_anchors
    """
    coords_np = _check_coord_2d(_to_numpy(coords, np.float32), "coords")
    cand = _to_numpy(candidate_idx, np.int64).reshape(-1)
    if cand.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if np.any(cand < 0) or np.any(cand >= coords_np.shape[0]):
        raise ValueError("candidate_idx contains out-of-range index")

    m = min(int(num_anchors), int(cand.size))
    if m <= 0:
        return np.zeros((0,), dtype=np.int64)

    w = parse_metric_weights(metric_weights)
    rng = np.random.default_rng(seed)

    cand_coords = coords_np[cand]
    w_sqrt = np.sqrt(w, dtype=np.float64).astype(np.float32)
    cand_w = cand_coords * w_sqrt.reshape(1, 4)
    cand_w_norm2 = np.sum(cand_w * cand_w, axis=1)

    first_local = int(rng.integers(0, cand.size))
    selected_locals: List[int] = [first_local]

    first_dot = cand_w @ cand_w[first_local]
    min_dist2 = cand_w_norm2 + cand_w_norm2[first_local] - 2.0 * first_dot
    min_dist2 = np.maximum(min_dist2, 0.0)
    min_dist2[first_local] = -1.0
    for _ in tqdm.trange(1, m,desc='sampling'):
        next_local = int(np.argmax(min_dist2))
        if min_dist2[next_local] < 0:
            break
        selected_locals.append(next_local)

        dot = cand_w @ cand_w[next_local]
        d2 = cand_w_norm2 + cand_w_norm2[next_local] - 2.0 * dot
        d2 = np.maximum(d2, 0.0)
        min_dist2 = np.minimum(min_dist2, d2)
        min_dist2[selected_locals] = -1.0

    return cand[np.asarray(selected_locals, dtype=np.int64)]


def _weighted_embed(coords: np.ndarray, metric_weights: np.ndarray) -> np.ndarray:
    w_sqrt = np.sqrt(metric_weights, dtype=np.float64).astype(np.float32)
    return coords.astype(np.float32) * w_sqrt[None, :]


def _resolve_torch_device(gpu_device: Optional[str]) -> Optional[Any]:
    """Return a torch.device for GPU kNN, or None if unavailable / should use NumPy."""
    try:
        import torch
    except ImportError:
        return None
    if gpu_device is not None:
        return torch.device(gpu_device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return None


def _per_point_kth_neighbor_dist_torch(
    z64: np.ndarray,
    k_nn: int,
    chunk_rows: int,
    full_matrix_max_n: int,
    device: Any,
) -> np.ndarray:
    """GPU/Accelerator implementation via PyTorch (float64). Falls back not handled here."""
    import torch

    z = torch.as_tensor(z64, dtype=torch.float64, device=device)
    n = z.shape[0]
    z_norm2 = (z * z).sum(dim=1)
    k_eff = max(1, min(int(k_nn), n - 1))

    # Dense: one N×N Gram on device
    if full_matrix_max_n > 0 and n <= int(full_matrix_max_n):
        gram = z @ z.T
        d2 = z_norm2.unsqueeze(1) + z_norm2.unsqueeze(0) - 2.0 * gram
        del gram
        d2 = torch.clamp(d2, min=0.0)
        d2.fill_diagonal_(float("inf"))
        vals, _ = torch.kthvalue(d2, k_eff, dim=1)
        return torch.sqrt(torch.clamp(vals, min=0.0)).cpu().numpy()

    cr = max(1, int(chunk_rows))
    kth_d2 = torch.empty(n, dtype=torch.float64, device="cpu")
    for start in tqdm.trange(0, n, cr, desc='chunked kNN'):
        end = min(start + cr, n)
        block = z[start:end]
        bsz = end - start
        sumsq = (block * block).sum(dim=1, keepdim=True)
        d2 = sumsq + z_norm2.unsqueeze(0) - 2.0 * (block @ z.T)
        d2 = torch.clamp(d2, min=0.0)
        rows = torch.arange(bsz, device=device)
        cols = start + torch.arange(bsz, device=device)
        d2[rows, cols] = float("inf")
        vals, _ = torch.kthvalue(d2, k_eff, dim=1)
        kth_d2[start:end] = vals.detach().cpu()
    return torch.sqrt(torch.clamp(kth_d2, min=0.0)).numpy()


def _value_based_suppression_loop_torch(
    z64_t: Any,
    v_t: Any,
    eligible_t: Any,
    sigma2: float,
    suppression: str,
    lam: float,
    L_eff: Optional[int],
    max_select: int,
    tol: float,
    desc: str,
) -> Tuple[List[int], List[float], Any]:
    """Greedy argmax + Gaussian suppression; all state on one PyTorch device.

    ``z64_t``: weighted embedding [N, 4] float64; matches ``weighted_sqdist_to_one`` geometry.
    """
    import torch

    n = z64_t.shape[0]
    z_norm2 = (z64_t * z64_t).sum(dim=1)
    neg_inf = torch.tensor(float("-inf"), device=z64_t.device, dtype=torch.float64)
    chosen = torch.zeros(n, dtype=torch.bool, device=z64_t.device)
    selected: List[int] = []
    selected_scores: List[float] = []

    for _ in tqdm.trange(max_select, desc=desc):
        cand_mask = ~chosen
        if not torch.any(cand_mask & eligible_t):
            break
        masked_v = torch.where(cand_mask & eligible_t, v_t, neg_inf)
        pick = int(torch.argmax(masked_v).item())
        pick_val = float(v_t[pick].item())
        if pick_val < tol:
            break
        selected.append(pick)
        selected_scores.append(pick_val)
        chosen[pick] = True

        zp = z64_t[pick]
        d2 = z_norm2 + z_norm2[pick] - 2.0 * (z64_t @ zp)
        d2 = torch.clamp(d2, min=0.0)
        kernel = torch.exp(-d2 / sigma2)

        if L_eff is None:
            if suppression == "subtractive":
                v_t = v_t - lam * kernel
                v_t = torch.clamp(v_t, min=0.0)
            else:
                v_t = v_t * torch.clamp(1.0 - lam * kernel, min=0.0)
        else:
            le = min(int(L_eff), n)
            _, idx = torch.topk(d2, k=le, largest=False, sorted=False)
            if suppression == "subtractive":
                v_t[idx] = v_t[idx] - lam * kernel[idx]
                v_t = torch.clamp(v_t, min=0.0)
            else:
                v_t[idx] = v_t[idx] * torch.clamp(1.0 - lam * kernel[idx], min=0.0)

    return selected, selected_scores, v_t


def _per_point_kth_neighbor_dist(
    z64: np.ndarray,
    k_nn: int,
    chunk_size: int = 2048,
    full_matrix_max_n: int = 4096,
    use_gpu: bool = False,
    gpu_batch_rows: Optional[int] = None,
    gpu_device: Optional[str] = None,
) -> np.ndarray:
    """Per-point distance to the k-th nearest neighbor in embedding space.

    ``z64`` rows are weighted-coordinate embeddings (same as ``estimate_sigma_from_knn``).

    When ``n <= full_matrix_max_n`` (and ``full_matrix_max_n > 0``), uses a **single**
    ``z64 @ z64.T`` Gram matrix plus broadcasted norms (one large GEMM, usually faster than
    many row blocks). For larger ``n``, falls back to chunked GEMM to limit peak memory O(n²).

    Args:
        full_matrix_max_n: max ``n`` for the dense pairwise path; set ``<= 0`` to always chunk.
        use_gpu: if True, use PyTorch on GPU/MPS when available (see ``gpu_device``).
        gpu_batch_rows: row tile size on GPU for the chunked path; default uses ``chunk_size``.
        gpu_device: e.g. ``\"cuda:1\"``; ``None`` picks ``cuda:0`` or MPS if available.

    Returns:
        dist: shape ``[N]``, Euclidean distance to k-th NN (not squared).
    """
    n = z64.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    if n == 1:
        return np.zeros((1,), dtype=np.float64)

    if use_gpu:
        dev = _resolve_torch_device(gpu_device)
        if dev is None:
            warnings.warn(
                "use_gpu=True but PyTorch CUDA/MPS is not available; using CPU NumPy.",
                UserWarning,
                stacklevel=2,
            )
        else:
            try:
                gr = int(gpu_batch_rows) if gpu_batch_rows is not None else int(chunk_size)
                return _per_point_kth_neighbor_dist_torch(
                    z64, k_nn, gr, full_matrix_max_n, dev
                )
            except Exception as exc:
                warnings.warn(
                    f"GPU kNN failed ({exc}); falling back to CPU NumPy.",
                    UserWarning,
                    stacklevel=2,
                )

    z_norm2 = np.sum(z64 * z64, axis=1, dtype=np.float64)
    k_eff = max(1, min(int(k_nn), n - 1))

    # Dense path: one N×N Gram matrix (fast BLAS), memory ~ 8 * n² bytes for float64
    if full_matrix_max_n > 0 and n <= int(full_matrix_max_n):
        gram = z64 @ z64.T
        d2 = z_norm2[:, None] + z_norm2[None, :] - 2.0 * gram
        del gram
        np.maximum(d2, 0.0, out=d2)
        np.fill_diagonal(d2, np.inf)
        part = np.partition(d2, kth=k_eff - 1, axis=1)
        kth_d2 = part[:, k_eff - 1]
        return np.sqrt(np.maximum(kth_d2, 0.0))

    kth_d2 = np.empty(n, dtype=np.float64)
    cs = max(1, int(chunk_size))
    for start in tqdm.trange(0, n, cs, desc='chunked kNN'):
        end = min(start + cs, n)
        block = z64[start:end]
        bsz = end - start
        sumsq = np.sum(block * block, axis=1, dtype=np.float64)[:, None]
        d2 = sumsq + z_norm2[None, :] - 2.0 * (block @ z64.T)
        np.maximum(d2, 0.0, out=d2)
        d2[np.arange(bsz), np.arange(start, end)] = np.inf
        part = np.partition(d2, kth=k_eff - 1, axis=1)
        kth_d2[start:end] = part[:, k_eff - 1]
    return np.sqrt(np.maximum(kth_d2, 0.0))


def estimate_sigma_from_knn(
    coords_norm: ArrayLike,
    k_nn: int = 8,
    metric_weights: Optional[Sequence[float]] = None,
    chunk_size: int = 2048,
    subsample_max: Optional[int] = 8192,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Median of k-th NN distances (weighted Euclidean).

    For large ``n``, full pairwise distance is O(n^2). When ``n > subsample_max``,
    randomly subsample ``subsample_max`` points and estimate sigma on that subset
    (kNN within the subset). Set ``subsample_max=None`` to disable subsampling.

    Args:
        coords_norm: normalized coordinates, shape [N, 4]
        k_nn: k-th nearest neighbor order
        metric_weights: length-4 weights
        chunk_size: row block size for GEMM (larger often faster on CPU)
        subsample_max: max points for sigma estimation; None = use all N
        rng: RNG for subsampling subsample_max < n; default seed 0

    Returns:
        sigma (float, > 0)
    """
    coords = _as_float2d(coords_norm, "coords_norm")
    n = coords.shape[0]
    if n == 0 or n == 1:
        return 1.0

    w = parse_metric_weights(metric_weights)
    z = _weighted_embed(coords, w)

    if subsample_max is not None and n > int(subsample_max):
        if rng is None:
            rng = np.random.default_rng(0)
        sub_idx = rng.choice(n, size=int(subsample_max), replace=False)
        z = z[sub_idx]
        n = z.shape[0]

    z64 = np.asarray(z, dtype=np.float64, order="C")
    kth_dist = _per_point_kth_neighbor_dist(z64, k_nn, chunk_size=chunk_size)

    sigma = float(np.median(kth_dist))
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = 1e-6
    return sigma


def _column_gains_local_from_d2(
    d2: np.ndarray,
    current_best: np.ndarray,
    sigma2: float,
    L_eff: int,
) -> np.ndarray:
    """Marginal facility-location gain per candidate using **only** the L_eff nearest points.

    For each column ``j`` (candidate anchor), take observation indices with smallest
    weighted squared distance in ``d2[:, j]``, evaluate RBF similarity and
    ``max(0, sim - current_best)`` **only** on those rows — no full ``[N, B]`` sim matrix.

    ``d2`` shape ``[N, B]`` (already clipped nonnegative). Returns gains ``[B]``.
    """
    n, B = d2.shape
    inv_s2 = 1.0 / sigma2
    if L_eff >= n:
        sim = np.exp(-d2 * inv_s2)
        delta = sim - current_best[:, None]
        np.maximum(delta, 0.0, out=delta)
        return np.sum(delta, axis=0)
    gains = np.empty(B, dtype=np.float64)
    cb = current_best
    for j in range(B):
        col = d2[:, j]
        idx = np.argpartition(col, L_eff - 1)[:L_eff]
        sim_sub = np.exp(-col[idx] * inv_s2)
        gains[j] = float(np.sum(np.maximum(0.0, sim_sub - cb[idx])))
    return gains


def facility_location_anchor_sampling(
    coord_obs: ArrayLike,
    num_anchors: int,
    metric_weights: Optional[Sequence[float]] = None,
    sigma: Optional[float] = None,
    k_nn_for_sigma: int = 8,
    sigma_subsample_max: Optional[int] = 8192,
    sigma_seed: int = 0,
    gain_tol: float = 0.0,
    candidate_batch_size: int = 512,
    nearest_l: Optional[int] = None,
    return_debug: bool = False,
):
    """Greedy Facility Location on normalized coordinates.

    Internally calls ``normalize_coords(coord_obs)`` first.

    Args:
        sigma_subsample_max: passed to ``estimate_sigma_from_knn`` when ``sigma`` is None;
            default 8192 speeds up large N. Use ``None`` for exact full-data sigma.
        sigma_seed: RNG seed for sigma subsampling.
        candidate_batch_size: number of candidate anchors per GEMM batch (memory ~ N * batch * 8 bytes).
        nearest_l: if ``None``, marginal gain sums over all ``N`` points (global). If a positive
            integer ``L``, gain is computed **only within** the ``L`` observation points nearest to
            each candidate (weighted embedding): RBF and ``max(0, sim - current_best)`` are evaluated
            on that set only — no full ``N``×candidate similarity matrix.
    """
    coords = _as_float2d(coord_obs, "coord_obs")
    n = coords.shape[0]
    m = int(num_anchors)
    cand_bs = max(1, int(candidate_batch_size))

    if n == 0 or m <= 0:
        empty_idx = np.zeros((0,), dtype=np.int64)
        if return_debug:
            debug = {
                "coord_obs_norm": np.zeros((0, 4), dtype=np.float32),
                "sigma": float(1.0),
                "selected_gains": np.zeros((0,), dtype=np.float64),
                "final_current_best": np.zeros((0,), dtype=np.float64),
                "coverage_score_history": np.zeros((0,), dtype=np.float64),
            }
            return empty_idx, debug
        return empty_idx

    coord_obs_norm, _ = normalize_coords(coords)
    w = parse_metric_weights(metric_weights)
    z = _weighted_embed(coord_obs_norm, w)
    # float64 once: batched (N,B) similarity dominates; avoids per-candidate cast
    z64 = np.asarray(z, dtype=np.float64, order="C")
    z_norm2 = np.sum(z64 * z64, axis=1)

    if sigma is None:
        sigma_val = estimate_sigma_from_knn(
            coords_norm=coord_obs_norm,
            k_nn=k_nn_for_sigma,
            metric_weights=w,
            subsample_max=sigma_subsample_max,
            rng=np.random.default_rng(sigma_seed),
        )
    else:
        sigma_val = float(sigma)

    if (not np.isfinite(sigma_val)) or sigma_val <= 0.0:
        sigma_val = 1e-6
    sigma2 = max(sigma_val * sigma_val, 1e-12)

    if nearest_l is not None:
        L_eff = int(nearest_l)
        if L_eff < 1:
            raise ValueError("nearest_l must be >= 1 when set")
        L_eff = min(L_eff, n)
    else:
        L_eff = None

    max_select = min(m, n)
    selected: List[int] = []
    selected_mask = np.zeros(n, dtype=bool)
    current_best = np.zeros(n, dtype=np.float64)
    selected_gains: List[float] = []
    coverage_hist: List[float] = []

    for _ in tqdm.trange(max_select, desc="facility location sampling"):
        candidates = np.flatnonzero(~selected_mask)
        if candidates.size == 0:
            break

        best_gain = -np.inf
        best_idx = -1

        # Batched gains: global mode builds full sim [N,B]; local (nearest_l) only uses d2 then L rows/col.
        for batch_start in tqdm.trange(0, candidates.size, cand_bs):
            batch_end = min(batch_start + cand_bs, candidates.size)
            cand_chunk = candidates[batch_start:batch_end]
            zc = z64[cand_chunk]
            # d2[i,j] = ||z_i - z_{cand_j}||^2_w  -> shape [N, B]
            d2 = z_norm2[:, None] + z_norm2[cand_chunk][None, :] - 2.0 * (z64 @ zc.T)
            np.maximum(d2, 0.0, out=d2)
            if L_eff is None:
                sim = np.exp(-d2 / sigma2)
                delta = sim - current_best[:, None]
                np.maximum(delta, 0.0, out=delta)
                gains = np.sum(delta, axis=0)
            else:
                gains = _column_gains_local_from_d2(d2, current_best, sigma2, L_eff)
            j = int(np.argmax(gains))
            g = float(gains[j])
            if g > best_gain:
                best_gain = g
                best_idx = int(cand_chunk[j])

        if best_idx < 0:
            break
        if best_gain < max(float(gain_tol), 1e-6):
            break

        # Single column for the winner (reuse for current_best update)
        dot = z64 @ z64[best_idx]
        d2_w = z_norm2 + z_norm2[best_idx] - 2.0 * dot
        np.maximum(d2_w, 0.0, out=d2_w)
        best_sim = np.exp(-d2_w / sigma2)

        selected.append(best_idx)
        selected_mask[best_idx] = True
        current_best = np.maximum(current_best, best_sim)
        selected_gains.append(best_gain)
        coverage_hist.append(float(np.sum(current_best)))

    anchor_idx = np.asarray(selected, dtype=np.int64)

    if return_debug:
        debug = {
            "coord_obs_norm": coord_obs_norm.astype(np.float32),
            "sigma": float(sigma_val),
            "selected_gains": np.asarray(selected_gains, dtype=np.float64),
            "final_current_best": current_best.astype(np.float64),
            "coverage_score_history": np.asarray(coverage_hist, dtype=np.float64),
        }
        return anchor_idx, debug
    return anchor_idx


def value_based_anchor_sampling(
    coord_obs: ArrayLike,
    num_anchors: int,
    metric_weights: Optional[Sequence[float]] = None,
    sigma: Optional[float] = None,
    k_nn_for_sigma: int = 8,
    sigma_subsample_max: Optional[int] = 4096,
    sigma_seed: int = 0,
    local_top_l: Optional[int] = None,
    suppression: str = "subtractive",
    suppression_lambda: float = 1.0,
    score_tol: float = 0.0,
    knn_chunk_size: int = 2048,
    knn_full_matrix_max_n: int = 4096,
    knn_use_gpu: bool = False,
    knn_gpu_batch_rows: Optional[int] = None,
    knn_gpu_device: Optional[str] = None,
    suppression_use_gpu: Optional[bool] = None,
    candidate_idx: Optional[ArrayLike] = None,
    return_debug: bool = False,
):
    """Greedy value-based anchor selection (coordinates only).

    1) Each point gets a **value score** = its k-th NN distance in weighted embedding space
       (same geometry as ``estimate_sigma_from_knn``): larger means more locally isolated /
       stronger need for local coverage.
    2) Repeatedly pick the remaining point with largest current value.
    3) After each pick, **suppress** nearby points via a Gaussian kernel ``exp(-d^2/\\sigma^2)``
       in the same weighted metric (via ``weighted_sqdist_to_one``).

    Internally calls ``normalize_coords(coord_obs)`` first.

    Args:
        sigma: bandwidth for suppression kernel; if ``None``, use ``estimate_sigma_from_knn``.
        k_nn_for_sigma: used both for estimating ``sigma`` (when ``sigma`` is None) and for
            the k-th NN distance that defines the initial value scores.
        sigma_subsample_max: passed to ``estimate_sigma_from_knn`` when ``sigma`` is None.
        sigma_seed: RNG seed for sigma subsampling.
        local_top_l: if set, suppression updates only the ``L`` observation points nearest to
            the chosen anchor (plus still using the same kernel weights on those indices).
            If ``None``, suppression is applied to **all** points (full O(N) per step).
        suppression: ``\"subtractive\"`` — ``v -= lambda * kernel``; ``\"multiplicative\"`` —
            ``v *= max(0, 1 - lambda * kernel)``.
        suppression_lambda: strength of suppression in ``[0, +inf)``.
        score_tol: stop when max remaining value falls below ``max(score_tol, 1e-12)``.
        knn_chunk_size: CPU chunked kNN row block size; also default GPU tile if
            ``knn_gpu_batch_rows`` is unset.
        knn_full_matrix_max_n: when ``n`` is at most this, use one dense pairwise matrix
            (CPU or GPU per ``knn_use_gpu``).
        knn_use_gpu: if True, compute k-th NN distances with PyTorch on GPU (CUDA/MPS) when
            available; requires PyTorch. Falls back to NumPy with a warning on failure.
        knn_gpu_batch_rows: GPU row batch size for chunked pairwise blocks; larger uses more VRAM.
        knn_gpu_device: e.g. ``\"cuda:1\"``; ``None`` auto-selects ``cuda:0`` or MPS.
        suppression_use_gpu: if True, run the greedy suppression loop on GPU (PyTorch), using
            embedding GEMV for ``d2`` instead of per-step NumPy ``weighted_sqdist_to_one``.
            If ``None``, defaults to ``knn_use_gpu`` (same device as ``knn_gpu_device``).
        candidate_idx: if set, anchors may only be chosen from these global indices (e.g.
            trusted traces). kNN value scores still use the full ``coord_obs`` geometry.

    **Performance:** Initial k-th NN distances use **all** ``N`` points. That requires
    pairwise distances in embedding space — **O(N²)** time (and the dense path below
    ``full_matrix_max_n`` uses **O(N²)** memory). In contrast, ``estimate_sigma_from_knn``
    often runs on a **subsample** (``sigma_subsample_max``), so sigma can be fast while
    this step still dominates when ``N`` is large. Enabling ``knn_use_gpu`` speeds up the
    GEMM / ``kthvalue`` work when VRAM fits the batch tiles (still **O(N²)** overall).
    When ``suppression_use_gpu`` is True (or equal to ``knn_use_gpu`` by default), each
    anchor step uses **O(N)** GPU work for distances/kernel updates instead of NumPy.
    """
    coords = _as_float2d(coord_obs, "coord_obs")
    n = coords.shape[0]
    m = int(num_anchors)

    if n == 0 or m <= 0:
        empty_idx = np.zeros((0,), dtype=np.int64)
        if return_debug:
            debug = {
                "coord_obs_norm": np.zeros((0, 4), dtype=np.float32),
                "sigma": float(1.0),
                "initial_value_scores": np.zeros((0,), dtype=np.float64),
                "final_value_scores": np.zeros((0,), dtype=np.float64),
                "selected_scores_at_pick": np.zeros((0,), dtype=np.float64),
            }
            return empty_idx, debug
        return empty_idx

    coord_obs_norm, _ = normalize_coords(coords)
    w = parse_metric_weights(metric_weights)
    z = _weighted_embed(coord_obs_norm, w)
    z64 = np.asarray(z, dtype=np.float64, order="C")
    if sigma is None:
        sigma_val = estimate_sigma_from_knn(
            coords_norm=coord_obs_norm,
            k_nn=k_nn_for_sigma,
            metric_weights=w,
            subsample_max=sigma_subsample_max,
            rng=np.random.default_rng(sigma_seed),
        )
    else:
        sigma_val = float(sigma)

    if (not np.isfinite(sigma_val)) or sigma_val <= 0.0:
        sigma_val = 1e-6
    sigma2 = max(sigma_val * sigma_val, 1e-12)
    # Representative value: k-th NN distance (embedding), full data — reuses _per_point_kth_neighbor_dist
    kth_dist = _per_point_kth_neighbor_dist(
        z64,
        k_nn=k_nn_for_sigma,
        chunk_size=knn_chunk_size,
        full_matrix_max_n=knn_full_matrix_max_n,
        use_gpu=knn_use_gpu,
        gpu_batch_rows=knn_gpu_batch_rows,
        gpu_device=knn_gpu_device,
    )
    base_score = np.maximum(kth_dist.astype(np.float64), 0.0)
    v = base_score.copy()
    if np.all(v <= 0.0):
        v[:] = 1.0

    eligible = np.ones(n, dtype=bool)
    if candidate_idx is not None:
        cand = _to_numpy(candidate_idx, np.int64).reshape(-1)
        if cand.size == 0:
            raise ValueError("candidate_idx cannot be empty when provided")
        if np.any(cand < 0) or np.any(cand >= n):
            raise ValueError("candidate_idx contains out-of-range index")
        eligible[:] = False
        eligible[cand] = True

    sup = str(suppression).lower().strip()
    if sup not in ("subtractive", "multiplicative"):
        raise ValueError(
            'suppression must be "subtractive" or "multiplicative", '
            f"got {suppression!r}"
        )
    lam = float(suppression_lambda)
    if not np.isfinite(lam) or lam < 0.0:
        raise ValueError("suppression_lambda must be a non-negative finite float")

    L_eff: Optional[int] = None
    if local_top_l is not None:
        L_eff = max(1, min(int(local_top_l), n))

    max_select = min(m, n)
    selected: List[int] = []
    selected_scores: List[float] = []
    tol = max(float(score_tol), 1e-12)

    run_supp_gpu = knn_use_gpu if suppression_use_gpu is None else bool(suppression_use_gpu)
    used_gpu_loop = False
    if run_supp_gpu:
        dev = _resolve_torch_device(knn_gpu_device)
        if dev is not None:
            try:
                import torch

                z64_t = torch.as_tensor(z64, dtype=torch.float64, device=dev)
                v_t = torch.as_tensor(v, dtype=torch.float64, device=dev)
                eligible_t = torch.as_tensor(eligible, dtype=torch.bool, device=dev)
                selected, selected_scores, v_t = _value_based_suppression_loop_torch(
                    z64_t,
                    v_t,
                    eligible_t,
                    float(sigma2),
                    sup,
                    lam,
                    L_eff,
                    max_select,
                    tol,
                    desc="value-based anchor sampling (GPU)",
                )
                v = v_t.detach().cpu().numpy()
                used_gpu_loop = True
            except Exception as exc:
                warnings.warn(
                    f"GPU suppression loop failed ({exc}); using CPU NumPy.",
                    UserWarning,
                    stacklevel=2,
                )

    if not used_gpu_loop:
        for _ in tqdm.trange(max_select, desc="value-based anchor sampling"):
            cand_mask = np.ones(n, dtype=bool)
            if selected:
                cand_mask[np.asarray(selected, dtype=np.int64)] = False
            if not np.any(cand_mask & eligible):
                break
            sub_v = np.where(cand_mask & eligible, v, -np.inf)
            pick = int(np.argmax(sub_v))
            pick_val = float(v[pick])
            if pick_val < tol:
                break

            selected.append(pick)
            selected_scores.append(pick_val)

            d2 = weighted_sqdist_to_one(coord_obs_norm[pick], coord_obs_norm, w)
            kernel = np.exp(-d2 / sigma2)

            if L_eff is None:
                idx = np.arange(n, dtype=np.int64)
            else:
                idx = np.argpartition(d2, min(L_eff, n) - 1)[: min(L_eff, n)]

            if sup == "subtractive":
                v[idx] = v[idx] - lam * kernel[idx]
                np.maximum(v, 0.0, out=v)
            else:
                v[idx] = v[idx] * np.maximum(0.0, 1.0 - lam * kernel[idx])

    anchor_idx = np.asarray(selected, dtype=np.int64)

    if return_debug:
        debug = {
            "coord_obs_norm": coord_obs_norm.astype(np.float32),
            "sigma": float(sigma_val),
            "initial_value_scores": base_score.astype(np.float64),
            "final_value_scores": v.astype(np.float64),
            "selected_scores_at_pick": np.asarray(selected_scores, dtype=np.float64),
        }
        return anchor_idx, debug
    return anchor_idx


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n_obs = 2000
    coord_obs_demo = rng.uniform(
        low=[0.0, 0.0, 0.0, 0.0],
        high=[1000.0, 1000.0, 1000.0, 1000.0],
        size=(n_obs, 4),
    ).astype(np.float32)

    anchors, debug_info = facility_location_anchor_sampling(
        coord_obs=coord_obs_demo,
        num_anchors=32,
        metric_weights=[1.0, 1.0, 0.5, 0.5],
        sigma=None,
        k_nn_for_sigma=8,
        gain_tol=0.0,
        return_debug=True,
    )

    print("selected anchor count:", anchors.size)
    print("anchor_idx (first 20):", anchors[:20])
    print("sigma:", debug_info["sigma"])
    print("selected_gains:", debug_info["selected_gains"])
