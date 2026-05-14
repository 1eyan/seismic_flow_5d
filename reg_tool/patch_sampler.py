"""Coordinate-only anchor patch sampler for training/inference.

This module implements:
1) Coordinate normalization / anchor selection (delegated to ``anchor_selector``)
2) Anchor selection by FPS on trusted points (via ``anchor_selector.farthest_point_sampling``)
3) Local top-L neighbor search
4) Diversity-aware top-K selection
5) Train patch builder
6) Grid block builder (with overlap and boundary coverage)
7) Inference patch builder
8) Block prediction accumulation/finalization
9) Fallback inference for uncovered grid points

All core logic is implemented with NumPy only.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from scipy.spatial import cKDTree

try:
    import tqdm as _tqdm_mod
except ImportError:
    _tqdm_mod = None

try:
    from .anchor_selector import (
        farthest_point_sampling,
        normalize_coords,
        parse_metric_weights,
        weighted_sqdist_to_one,
        facility_location_anchor_sampling,
        value_based_anchor_sampling,
    )
except ImportError:
    from anchor_selector import (
        farthest_point_sampling,
        normalize_coords,
        parse_metric_weights,
        weighted_sqdist_to_one,
        facility_location_anchor_sampling,
        value_based_anchor_sampling,
    )

ArrayLike = Union[np.ndarray, "Any"]


def _progress(iterable: Any, *args: Any, **kwargs: Any) -> Any:
    """Return a tqdm iterator when available, otherwise the raw iterable."""
    if _tqdm_mod is None:
        return iterable
    return _tqdm_mod.tqdm(iterable, *args, **kwargs)


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


def top_l_neighbors(
    center_coord: ArrayLike,
    all_coords: ArrayLike,
    top_l: int,
    metric_weights: Optional[Sequence[float]] = None,
    exclude_self: bool = False,
    self_index: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find top-L nearest neighbors from all observed points.

    Args:
        center_coord: center coordinate, shape [4]
        all_coords: all observed coordinates, shape [N, 4]
        top_l: number of nearest points
        metric_weights: weighted Euclidean metric weights
        exclude_self: if True, exclude self_index
        self_index: global index of center point in all_coords (needed when exclude_self=True)

    Returns:
        neighbor_idx: global indices of nearest neighbors, shape [L']
        neighbor_dist: Euclidean distances (not squared), shape [L']
    """
    all_coords_np = _check_coord_2d(_to_numpy(all_coords, np.float32), "all_coords")
    center = _to_numpy(center_coord, np.float32).reshape(-1)
    if center.shape[0] != 4:
        raise ValueError(f"center_coord must have shape [4], got {center.shape}")
    l = int(top_l)
    if l <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    d2 = weighted_sqdist_to_one(center, all_coords_np, metric_weights)

    valid = np.ones(all_coords_np.shape[0], dtype=bool)
    if exclude_self:
        if self_index is None:
            raise ValueError("self_index must be provided when exclude_self=True")
        if self_index < 0 or self_index >= all_coords_np.shape[0]:
            raise ValueError("self_index out of range")
        valid[self_index] = False

    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    l_eff = min(l, valid_idx.size)
    valid_d2 = d2[valid_idx]
    partial = np.argpartition(valid_d2, kth=l_eff - 1)[:l_eff]
    order = np.argsort(valid_d2[partial])
    chosen_local = partial[order]
    neighbor_idx = valid_idx[chosen_local].astype(np.int64)
    neighbor_dist = np.sqrt(valid_d2[chosen_local]).astype(np.float32)
    return neighbor_idx, neighbor_dist


def diverse_topk(
    center_coord: ArrayLike,
    candidate_idx: ArrayLike,
    all_coords: ArrayLike,
    k: int,
    metric_weights: Optional[Sequence[float]] = None,
    beta: float = 0.3,
) -> np.ndarray:
    """Greedy diverse top-k selection from candidate points.

    Score rule:
        score(i) = -d(center, i)^2 + beta * min_{j in selected} d(i, j)^2

    First selected point is nearest to center.

    Args:
        center_coord: center coordinate, shape [4]
        candidate_idx: global candidate indices, shape [L]
        all_coords: full coordinates, shape [N, 4]
        k: number of points to select
        metric_weights: weighted Euclidean metric weights
        beta: diversity weight

    Returns:
        selected_idx: selected global indices, shape [K']
    """
    coords_np = _check_coord_2d(_to_numpy(all_coords, np.float32), "all_coords")
    cand = _to_numpy(candidate_idx, np.int64).reshape(-1)
    if cand.size == 0 or int(k) <= 0:
        return np.zeros((0,), dtype=np.int64)
    if np.any(cand < 0) or np.any(cand >= coords_np.shape[0]):
        raise ValueError("candidate_idx contains out-of-range index")

    center = _to_numpy(center_coord, np.float32).reshape(-1)
    if center.shape[0] != 4:
        raise ValueError(f"center_coord must have shape [4], got {center.shape}")

    k_eff = min(int(k), int(cand.size))
    w = parse_metric_weights(metric_weights)
    cand_coords = coords_np[cand]

    center_d2 = np.sum(((cand_coords - center.reshape(1, 4)) ** 2) * w.reshape(1, 4), axis=1)
    first = int(np.argmin(center_d2))
    selected_locals: List[int] = [first]
    selected_mask = np.zeros(cand.size, dtype=bool)
    selected_mask[first] = True

    min_d2_to_selected = np.sum(
        ((cand_coords - cand_coords[first:first + 1]) ** 2) * w.reshape(1, 4),
        axis=1,
    )

    for _ in range(1, k_eff):
        scores = -center_d2 + float(beta) * min_d2_to_selected
        scores[selected_mask] = -np.inf
        next_local = int(np.argmax(scores))
        if not np.isfinite(scores[next_local]):
            break
        selected_locals.append(next_local)
        selected_mask[next_local] = True

        d2_new = np.sum(
            ((cand_coords - cand_coords[next_local:next_local + 1]) ** 2) * w.reshape(1, 4),
            axis=1,
        )
        min_d2_to_selected = np.minimum(min_d2_to_selected, d2_new)

    return cand[np.asarray(selected_locals, dtype=np.int64)]


def _gather_valid_obs_candidates(
    center: np.ndarray,
    obs_coords: np.ndarray,
    top_l: int,
    metric_weights: Optional[Sequence[float]],
    obs_valid_mask: np.ndarray,
    k_patch: int,
) -> np.ndarray:
    """Expand nearest-neighbor search under an optional observed-candidate filter.

    Here ``obs_valid_mask`` only means "allowed to be selected as context candidate".
    It does not define the final query/context input missingness semantics.
    """
    n_obs = int(obs_coords.shape[0])
    ovm = obs_valid_mask.reshape(-1).astype(bool)
    if ovm.shape[0] != n_obs:
        raise ValueError("obs_valid_mask must have shape [N_obs]")
    if not np.any(ovm):
        raise ValueError("obs_valid_mask has no True entries")
    L = max(1, int(top_l))
    valid = np.zeros((0,), dtype=np.int64)
    for _ in range(40):
        cand_raw, _ = top_l_neighbors(
            center_coord=center,
            all_coords=obs_coords,
            top_l=min(L, n_obs),
            metric_weights=metric_weights,
            exclude_self=False,
        )
        valid = cand_raw[ovm[cand_raw]]
        if valid.size >= k_patch:
            break
        if L >= n_obs:
            break
        L = min(L * 2, n_obs)
    if valid.size == 0:
        raise RuntimeError(
            "build_infer_patch: obs_valid_mask 下无可用观测道（扩大邻域后仍为空），请检查 mask 与坐标"
        )
    return valid


def _select_context_idx_for_anchor(
    anchor_coord: np.ndarray,
    obs_coords: np.ndarray,
    k_patch: int,
    top_l: int,
    metric_weights: Optional[Sequence[float]],
    beta: float,
    obs_valid_mask: Optional[np.ndarray],
) -> np.ndarray:
    """Select one anchor's observed context indices.

    If ``obs_valid_mask`` is provided, it is only used to restrict which observed
    traces may enter the candidate/context set.
    """
    if obs_valid_mask is not None:
        candidate_idx = _gather_valid_obs_candidates(
            center=anchor_coord,
            obs_coords=obs_coords,
            top_l=top_l,
            metric_weights=metric_weights,
            obs_valid_mask=obs_valid_mask,
            k_patch=k_patch,
        )
    else:
        candidate_idx, _ = top_l_neighbors(
            center_coord=anchor_coord,
            all_coords=obs_coords,
            top_l=top_l,
            metric_weights=metric_weights,
            exclude_self=False,
        )
    context_idx = diverse_topk(
        center_coord=anchor_coord,
        candidate_idx=candidate_idx,
        all_coords=obs_coords,
        k=k_patch,
        metric_weights=metric_weights,
        beta=beta,
    )
    if context_idx.size == 0:
        raise RuntimeError("build_infer_patch failed: empty context_idx")
    return context_idx.astype(np.int64, copy=False)


def _stable_unique_index_list(index_list: Sequence[np.ndarray]) -> np.ndarray:
    """Deduplicate indices while preserving first appearance order."""
    out: List[int] = []
    seen = set()
    for idx in index_list:
        arr = np.asarray(idx, dtype=np.int64).reshape(-1)
        for v in arr.tolist():
            iv = int(v)
            if iv not in seen:
                seen.add(iv)
                out.append(iv)
    return np.asarray(out, dtype=np.int64)


def _normalize_optional_bool_mask(
    mask: Optional[ArrayLike],
    size: int,
    name: str,
) -> Optional[np.ndarray]:
    """Normalize an optional mask to shape [size] bool."""
    if mask is None:
        return None
    out = _to_numpy(mask).reshape(-1).astype(bool)
    if out.shape[0] != int(size):
        raise ValueError(f"{name} must have shape [{int(size)}]")
    return out


def _sort_grid_indices_by_coord(
    grid_idx: ArrayLike,
    grid_coords: np.ndarray,
) -> np.ndarray:
    """Sort global grid indices lexicographically by their 4D coordinates."""
    idx = _to_numpy(grid_idx, np.int64).reshape(-1)
    if idx.size <= 1:
        return idx.astype(np.int64, copy=False)
    order = np.lexsort([grid_coords[idx, i] for i in (3, 2, 1, 0)])
    return idx[order].astype(np.int64, copy=False)


def _filter_grid_query_indices(
    block_grid_idx: ArrayLike,
    grid_query_mask: Optional[np.ndarray],
) -> np.ndarray:
    """Filter one block's grid indices by an optional global query mask."""
    idx = _to_numpy(block_grid_idx, np.int64).reshape(-1)
    if grid_query_mask is None or idx.size == 0:
        return idx.astype(np.int64, copy=False)
    return idx[grid_query_mask[idx]].astype(np.int64, copy=False)


def _resolve_block_center_coord(
    center_grid_idx: int,
    block_grid_idx: np.ndarray,
    grid_coords: np.ndarray,
) -> np.ndarray:
    """Resolve one block center coordinate, falling back to block mean when needed."""
    if 0 <= int(center_grid_idx) < int(grid_coords.shape[0]):
        return grid_coords[int(center_grid_idx)].astype(np.float32, copy=False)
    return np.mean(grid_coords[block_grid_idx], axis=0).astype(np.float32, copy=False)


def _build_query_self_context_payload(
    grid_query_idx: ArrayLike,
    grid_coords: np.ndarray,
    select_contexts: Callable[[np.ndarray], List[np.ndarray]],
) -> Optional[Dict[str, Any]]:
    """Build query-self anchor context payload for one inference sample."""
    q_idx = _to_numpy(grid_query_idx, np.int64).reshape(-1)
    if q_idx.size == 0:
        return None
    anchor_coords = grid_coords[q_idx].astype(np.float32, copy=False)
    context_idx_per_anchor = select_contexts(anchor_coords)
    if len(context_idx_per_anchor) != q_idx.size:
        raise RuntimeError(
            "context selector returned inconsistent number of anchor contexts: "
            f"expected {q_idx.size}, got {len(context_idx_per_anchor)}"
        )
    context_idx = _stable_unique_index_list(context_idx_per_anchor)
    if context_idx.size == 0:
        return None
    return {
        "grid_query_idx": q_idx.astype(np.int64, copy=False),
        "anchor_grid_idx": q_idx.astype(np.int64, copy=False),
        "context_idx_per_anchor": context_idx_per_anchor,
        "context_idx": context_idx.astype(np.int64, copy=False),
    }


def _make_empty_infer_patch(
    center: np.ndarray,
    trace_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Return an empty inference patch with stable schema."""
    out: Dict[str, Any] = {
        "anchor_grid_idx": np.zeros((0,), dtype=np.int64),
        "anchor_coord": np.zeros((0, 4), dtype=np.float32),
        "block_center_coord": center.astype(np.float32),
        "context_idx_per_anchor": [],
        "context_idx": np.zeros((0,), dtype=np.int64),
        "grid_query_idx": np.zeros((0,), dtype=np.int64),
        "query_first_count": np.asarray(0, dtype=np.int64),
        "context_count": np.asarray(0, dtype=np.int64),
    }
    if trace_length is not None:
        out.update(
            {
                "coord_context": np.zeros((0, 4), dtype=np.float32),
                "trace_context": np.zeros((0, trace_length), dtype=np.float32),
                "coord_query": np.zeros((0, 4), dtype=np.float32),
                "trace_query_input": np.zeros((0, trace_length), dtype=np.float32),
                "network_input_trace": np.zeros((0, trace_length), dtype=np.float32),
                "network_input_coord": np.zeros((0, 4), dtype=np.float32),
                "network_input_is_query": np.zeros((0,), dtype=bool),
            }
        )
    return out


def _resolve_torch_device(gpu_device: Optional[str] = None) -> Optional[Any]:
    """Return an accelerator device for batched query-context search, or None."""
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


def _make_context_selector(
    obs_coords: np.ndarray,
    k_patch: int,
    top_l: int,
    metric_weights: Optional[Sequence[float]],
    beta: float,
    obs_valid_mask: Optional[np.ndarray],
    use_gpu: bool = False,
    gpu_device: Optional[str] = None,
    gpu_query_chunk_size: int = 64,
) -> Callable[[np.ndarray], List[np.ndarray]]:
    """Build a reusable selector for many anchor coords.

    The heavy all-anchor x all-observation distance stage can run in batches on GPU.
    On CPU, when SciPy is available, a weighted ``cKDTree`` is used to avoid
    rescanning all observations for every anchor. The final diverse-topk is still
    performed per anchor on CPU, but only on the much smaller candidate set.
    """
    obs_np = _check_coord_2d(np.asarray(obs_coords, dtype=np.float32), "obs_coords")
    n_obs = int(obs_np.shape[0])
    ovm = _normalize_optional_bool_mask(obs_valid_mask, n_obs, "obs_valid_mask")
    n_valid = n_obs if ovm is None else int(ovm.sum())
    if ovm is not None and n_valid == 0:
        raise ValueError("obs_valid_mask has no True entries")
    w = parse_metric_weights(metric_weights).astype(np.float32, copy=False)
    w_sqrt = np.sqrt(w).astype(np.float32, copy=False)
    obs_w_np = obs_np * w_sqrt[None, :]

    tree_all = cKDTree(obs_w_np)
    valid_obs_idx = None
    tree_valid = None
    if ovm is not None:
        valid_obs_idx = np.flatnonzero(ovm).astype(np.int64, copy=False)
        tree_valid = cKDTree(obs_w_np[valid_obs_idx])

    def _select_cpu(anchor_coords: np.ndarray) -> List[np.ndarray]:
        anchors_np = _check_coord_2d(np.asarray(anchor_coords, dtype=np.float32), "anchor_coords")
        if anchors_np.shape[0] == 0:
            return []

        def _tree_query(tree: Any, query_points: np.ndarray, k: int) -> np.ndarray:
            if k <= 0 or query_points.shape[0] == 0:
                return np.zeros((query_points.shape[0], 0), dtype=np.int64)
            _, idx = tree.query(query_points, k=k, workers=-1)
            idx_np = np.asarray(idx, dtype=np.int64)
            if idx_np.ndim == 1:
                idx_np = idx_np.reshape(-1, 1)
            return idx_np

        out = []
        anchors_w_np = anchors_np * w_sqrt[None, :]
        if ovm is None:
            k_search = min(int(top_l), n_obs)
            cand_idx_batch = _tree_query(tree_all, anchors_w_np, k_search)
            for row_idx in range(cand_idx_batch.shape[0]):
                candidate_idx = cand_idx_batch[row_idx].astype(np.int64, copy=False)
                out.append(
                    diverse_topk(
                        center_coord=anchors_np[row_idx],
                        candidate_idx=candidate_idx,
                        all_coords=obs_np,
                        k=k_patch,
                        metric_weights=metric_weights,
                        beta=beta,
                    ).astype(np.int64, copy=False)
                )
            return out

        assert tree_valid is not None and valid_obs_idx is not None
        k_search = min(max(int(top_l), int(k_patch)), n_valid)
        cand_local_batch = _tree_query(tree_valid, anchors_w_np, k_search)
        for row_idx in range(cand_local_batch.shape[0]):
            candidate_idx = valid_obs_idx[cand_local_batch[row_idx]]
            out.append(
                diverse_topk(
                    center_coord=anchors_np[row_idx],
                    candidate_idx=candidate_idx,
                    all_coords=obs_np,
                    k=k_patch,
                    metric_weights=metric_weights,
                    beta=beta,
                ).astype(np.int64, copy=False)
            )
        return out

    if not use_gpu:
        return _select_cpu

    device = _resolve_torch_device(gpu_device)
    if device is None:
        return _select_cpu

    try:
        import torch
    except ImportError:
        return _select_cpu

    obs_w_t = torch.as_tensor(obs_np * w_sqrt[None, :], dtype=torch.float32, device=device)
    obs_norm2_t = (obs_w_t * obs_w_t).sum(dim=1)
    def _select_gpu(anchor_coords: np.ndarray) -> List[np.ndarray]:
        anchors_np = _check_coord_2d(np.asarray(anchor_coords, dtype=np.float32), "anchor_coords")
        if anchors_np.shape[0] == 0:
            return []
        out: List[np.ndarray] = []
        chunk_q = max(1, int(gpu_query_chunk_size))
        k_search = min(max(int(top_l), int(k_patch)), n_obs)
        if k_search <= 0:
            return [np.zeros((0,), dtype=np.int64) for _ in range(anchors_np.shape[0])]
        need_valid = min(int(k_patch), n_valid)

        for start in range(0, anchors_np.shape[0], chunk_q):
            end = min(start + chunk_q, anchors_np.shape[0])
            anc_chunk = anchors_np[start:end]
            anc_w_t = torch.as_tensor(anc_chunk * w_sqrt[None, :], dtype=torch.float32, device=device)
            anc_norm2_t = (anc_w_t * anc_w_t).sum(dim=1, keepdim=True)
            d2_t = anc_norm2_t + obs_norm2_t.unsqueeze(0) - 2.0 * (anc_w_t @ obs_w_t.T)
            d2_t = torch.clamp(d2_t, min=0.0)
            filtered_rows: Optional[List[np.ndarray]] = None
            k_cur = k_search
            while True:
                vals_t, idx_t = torch.topk(d2_t, k=k_cur, dim=1, largest=False, sorted=True)
                cand_idx_np = idx_t.detach().cpu().numpy()
                cand_val_np = vals_t.detach().cpu().numpy()
                if ovm is None:
                    filtered_rows = None
                    break

                filtered_rows = []
                enough = True
                for row_idx in range(cand_idx_np.shape[0]):
                    finite = np.isfinite(cand_val_np[row_idx])
                    raw_idx = cand_idx_np[row_idx, finite].astype(np.int64, copy=False)
                    valid_idx = raw_idx[ovm[raw_idx]]
                    filtered_rows.append(valid_idx)
                    if valid_idx.size < need_valid and k_cur < n_obs:
                        enough = False
                if enough or k_cur >= n_obs:
                    break
                k_cur = min(k_cur * 2, n_obs)

            for row_idx in range(cand_idx_np.shape[0]):
                if filtered_rows is None:
                    finite = np.isfinite(cand_val_np[row_idx])
                    candidate_idx = cand_idx_np[row_idx, finite].astype(np.int64, copy=False)
                else:
                    candidate_idx = filtered_rows[row_idx]
                context_idx = diverse_topk(
                    center_coord=anc_chunk[row_idx],
                    candidate_idx=candidate_idx,
                    all_coords=obs_np,
                    k=k_patch,
                    metric_weights=metric_weights,
                    beta=beta,
                )
                if context_idx.size == 0:
                    raise RuntimeError("build_infer_patch failed: empty context_idx")
                out.append(context_idx.astype(np.int64, copy=False))
        return out

    return _select_gpu


def build_train_patch(
    anchor_idx: int,
    coord_obs_norm: ArrayLike,
    trace_obs: ArrayLike,
    k_patch: int = 64,
    top_l: int = 128,
    num_query: Optional[int] = None,
    metric_weights: Optional[Sequence[float]] = None,
    beta: float = 0.3,
    seed: Optional[int] = None,
    return_features: bool = False,
    pool_size: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Build one train pool from one observed anchor.

    Workflow:
        1) top_l_neighbors around anchor
        2) diverse_topk from local candidates
        3) keep the resulting observed pool; online query/context sampling is handled
           by ``build_train_sample_from_pool``

    Args:
        anchor_idx: global anchor index in observed set
        coord_obs_norm: normalized observed coords, shape [N_obs, 4]
        trace_obs: observed traces, shape [N_obs, T]
        k_patch: target context size used later during training/inference matching
        top_l: local candidate size
        num_query: optional hint used to derive a default pool size
        metric_weights: weighted Euclidean metric weights
        beta: diversity weight in diverse_topk
        seed: kept for API compatibility
        return_features: if True, also return pool coord/trace arrays
        pool_size: optional observed pool size. Default is ``max(2*k_patch, k_patch+num_query)``.

    Returns:
        dict with pool semantics:
            anchor_idx, patch_idx, context_idx, query_idx
        if return_features=True, also includes:
            anchor_coord, coord_context, trace_context, coord_query, trace_query_gt
    """
    coords = _check_coord_2d(_to_numpy(coord_obs_norm, np.float32), "coord_obs_norm")
    traces = _to_numpy(trace_obs, np.float32)
    if traces.ndim != 2:
        raise ValueError(f"trace_obs must have shape [N_obs, T], got {traces.shape}")
    if traces.shape[0] != coords.shape[0]:
        raise ValueError("trace_obs and coord_obs_norm must share same N_obs")
    if anchor_idx < 0 or anchor_idx >= coords.shape[0]:
        raise ValueError("anchor_idx out of range")
    _ = seed

    anchor_coord = coords[anchor_idx]
    q_hint = max(1, 1 if num_query is None else int(num_query))
    k_pool = int(pool_size) if pool_size is not None else max(2 * int(k_patch), int(k_patch) + q_hint)
    if k_pool < 1:
        raise ValueError("pool_size must be >= 1")
    local_candidate_idx, _ = top_l_neighbors(
        center_coord=anchor_coord,
        all_coords=coords,
        top_l=top_l,
        metric_weights=metric_weights,
        exclude_self=False,
    )

    patch_idx = diverse_topk(
        center_coord=anchor_coord,
        candidate_idx=local_candidate_idx,
        all_coords=coords,
        k=k_pool,
        metric_weights=metric_weights,
        beta=beta,
    )
    if patch_idx.size == 0:
        raise RuntimeError("build_train_patch failed: empty patch_idx")
    context_idx = patch_idx.copy()
    query_idx = np.zeros((0,), dtype=np.int64)

    out = {
        "anchor_idx": np.asarray(anchor_idx, dtype=np.int64),
        "patch_idx": patch_idx.astype(np.int64),
        "context_idx": context_idx.astype(np.int64),
        "query_idx": query_idx.astype(np.int64),
    }
    if return_features:
        out.update(
            {
                "anchor_coord": anchor_coord.astype(np.float32),
                "coord_context": coords[context_idx].astype(np.float32),
                "trace_context": traces[context_idx].astype(np.float32),
                "coord_query": np.zeros((0, 4), dtype=np.float32),
                "trace_query_gt": np.zeros((0, traces.shape[1]), dtype=np.float32),
            }
        )
    return out


def build_train_sample_from_pool(
    pool_idx: ArrayLike,
    coord_obs_norm: ArrayLike,
    trace_obs: ArrayLike,
    num_query: int = 8,
    k_context: int = 64,
    metric_weights: Optional[Sequence[float]] = None,
    beta: float = 0.3,
    seed: Optional[int] = None,
    anchor_idx: Optional[int] = None,
    force_anchor_query: bool = False,
    return_features: bool = False,
) -> Dict[str, Any]:
    """Build one query-first training sample from a precomputed observed pool."""
    coords = _check_coord_2d(_to_numpy(coord_obs_norm, np.float32), "coord_obs_norm")
    traces = _to_numpy(trace_obs, np.float32)
    if traces.ndim != 2 or traces.shape[0] != coords.shape[0]:
        raise ValueError("trace_obs must have shape [N_obs, T] and align with coord_obs_norm")

    pool = _to_numpy(pool_idx, np.int64).reshape(-1)
    if pool.size == 0:
        raise ValueError("pool_idx cannot be empty")
    if np.any(pool < 0) or np.any(pool >= coords.shape[0]):
        raise ValueError("pool_idx contains out-of-range index")

    q_eff = int(num_query)
    if q_eff < 1:
        raise ValueError("num_query must be >= 1")
    q_eff = min(q_eff, int(pool.size))
    k_ctx = int(k_context)
    if k_ctx < 1:
        raise ValueError("k_context must be >= 1")

    rng = np.random.default_rng(seed)
    if force_anchor_query and anchor_idx is not None and int(anchor_idx) in set(pool.tolist()):
        rest = pool[pool != int(anchor_idx)]
        n_extra = min(q_eff - 1, int(rest.size))
        extra = rest[rng.permutation(rest.size)[:n_extra]] if rest.size > 0 else np.zeros((0,), dtype=np.int64)
        query_idx = np.concatenate(
            [np.asarray([int(anchor_idx)], dtype=np.int64), np.asarray(extra, dtype=np.int64)],
            axis=0,
        )
    else:
        query_idx = pool[rng.permutation(pool.size)[:q_eff]].astype(np.int64, copy=False)
    if query_idx.size == 0:
        raise RuntimeError("build_train_sample_from_pool failed: empty query_idx")

    candidate_pool = pool[~np.isin(pool, query_idx)]
    if candidate_pool.size == 0:
        raise RuntimeError("build_train_sample_from_pool failed: empty candidate_pool")

    context_idx_per_anchor: List[np.ndarray] = []
    for qidx in query_idx:
        context_idx_per_anchor.append(
            diverse_topk(
                center_coord=coords[int(qidx)],
                candidate_idx=candidate_pool,
                all_coords=coords,
                k=min(k_ctx, int(candidate_pool.size)),
                metric_weights=metric_weights,
                beta=beta,
            ).astype(np.int64, copy=False)
        )
    context_idx = _stable_unique_index_list(context_idx_per_anchor)
    if context_idx.size == 0:
        raise RuntimeError("build_train_sample_from_pool failed: empty context_idx")
    patch_idx = np.concatenate([query_idx, context_idx], axis=0).astype(np.int64, copy=False)

    out: Dict[str, Any] = {
        "anchor_idx": np.asarray(-1 if anchor_idx is None else int(anchor_idx), dtype=np.int64),
        "pool_idx": pool.astype(np.int64, copy=False),
        "patch_idx": patch_idx,
        "context_idx": context_idx.astype(np.int64, copy=False),
        "query_idx": query_idx.astype(np.int64, copy=False),
        "context_idx_per_anchor": context_idx_per_anchor,
        "query_first_count": np.asarray(query_idx.size, dtype=np.int64),
        "context_count": np.asarray(context_idx.size, dtype=np.int64),
    }
    if return_features:
        coord_query = coords[query_idx].astype(np.float32)
        coord_context = coords[context_idx].astype(np.float32)
        trace_query_gt = traces[query_idx].astype(np.float32)
        trace_context = traces[context_idx].astype(np.float32)
        trace_query_input = np.zeros((query_idx.size, traces.shape[1]), dtype=np.float32)
        out.update(
            {
                "anchor_coord": np.zeros((4,), dtype=np.float32)
                if anchor_idx is None
                else coords[int(anchor_idx)].astype(np.float32),
                "coord_context": coord_context,
                "trace_context": trace_context,
                "coord_query": coord_query,
                "trace_query_gt": trace_query_gt,
                "trace_query_input": trace_query_input,
                "network_input_trace": np.concatenate([trace_query_input, trace_context], axis=0),
                "network_input_coord": np.concatenate([coord_query, coord_context], axis=0),
                "network_input_is_query": np.concatenate(
                    [
                        np.ones((query_idx.size,), dtype=bool),
                        np.zeros((context_idx.size,), dtype=bool),
                    ],
                    axis=0,
                ),
            }
        )
    return out


def _parse_2tuple(value: Union[int, Sequence[int]], name: str) -> Tuple[int, int]:
    if isinstance(value, (int, np.integer)):
        v = int(value)
        return v, v
    seq = tuple(int(v) for v in value)
    if len(seq) != 2:
        raise ValueError(f"{name} must be int or length-2 sequence")
    return seq[0], seq[1]


def _parse_4tuple(value: Union[int, Sequence[int]], name: str) -> Tuple[int, int, int, int]:
    if isinstance(value, (int, np.integer)):
        v = int(value)
        return v, v, v, v
    seq = tuple(int(v) for v in value)
    if len(seq) != 4:
        raise ValueError(f"{name} must be int or length-4 sequence")
    return seq[0], seq[1], seq[2], seq[3]


def ravel_grid_index_4d(
    isx: np.ndarray,
    isy: np.ndarray,
    irx: np.ndarray,
    iry: np.ndarray,
    dims: Tuple[int, int, int, int],
) -> np.ndarray:
    """C-order 展平：``coord_grid_norm`` 第 ``k`` 行对应 ``(nsx,nsy,nrx,nry)`` 四维下标。

    约定与 ``numpy.ravel(order='C')`` 一致：最末维 ``iry`` 变化最快，``isx`` 最慢。
    列语义建议：``[:,0:2]`` 炮点平面 ``(isx, isy)``，``[:,2:4]`` 检波点平面 ``(irx, iry)``。
    """
    return np.ravel_multi_index(
        (isx, isy, irx, iry),
        dims,
        order="C",
    ).astype(np.int64)


def _compute_starts(n: int, block_n: int, stride_n: int) -> List[int]:
    if n <= 0:
        raise ValueError("grid dimension must be positive")
    if block_n <= 0 or stride_n <= 0:
        raise ValueError("block size and stride must be positive")
    if stride_n > block_n:
        raise ValueError("stride must be <= block size")

    if n <= block_n:
        return [0]

    starts = list(range(0, n - block_n + 1, stride_n))
    last = n - block_n
    if starts[-1] != last:
        starts.append(last)
    return starts


def make_grid_blocks_from_shape(
    nx: int,
    ny: int,
    block_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]],
) -> List[Dict[str, Any]]:
    """Build overlapped blocks for flattened regular grid.

    Grid flatten order: row-major, global_idx = y * nx + x.

    Args:
        nx: number of columns
        ny: number of rows
        block_size: (bx, by) or scalar
        stride: (sx, sy) or scalar, must satisfy sx<=bx and sy<=by

    Returns:
        list of block dict:
            block_id, grid_point_indices, block_center_grid_index, block_center_coord(None)
    """
    bx, by = _parse_2tuple(block_size, "block_size")
    sx, sy = _parse_2tuple(stride, "stride")
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive")

    grid_indices = np.arange(nx * ny, dtype=np.int64).reshape(ny, nx)
    x_starts = _compute_starts(nx, bx, sx)
    y_starts = _compute_starts(ny, by, sy)

    blocks: List[Dict[str, Any]] = []
    block_id = 0
    for y0 in y_starts:
        for x0 in x_starts:
            sub = grid_indices[y0:y0 + by, x0:x0 + bx]
            point_idx = sub.reshape(-1)
            cy = y0 + min(by - 1, sub.shape[0] // 2)
            cx = x0 + min(bx - 1, sub.shape[1] // 2)
            center_idx = int(grid_indices[cy, cx])
            blocks.append(
                {
                    "block_id": block_id,
                    "grid_point_indices": point_idx.astype(np.int64),
                    "block_center_grid_index": np.asarray(center_idx, dtype=np.int64),
                    "block_center_coord": None,
                }
            )
            block_id += 1
    return blocks


def make_grid_blocks_from_shape_4d(
    nsx: int,
    nsy: int,
    nrx: int,
    nry: int,
    block_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]],
) -> List[Dict[str, Any]]:
    """四维规则网格上的轴对齐滑窗分块（等价于「炮点 2D × 检波点 2D」笛卡尔积块）。

    与 ``coord_grid_norm.reshape(nsx, nsy, nrx, nry, 4)`` 的 C 序展平一致：
    ``global_idx = ravel_multi_index((isx,isy,irx,iry), (nsx,nsy,nrx,nry), order='C')``。

    Args:
        nsx, nsy: 炮点方向（对应 coord 前两维）网格尺寸
        nrx, nry: 检波点方向（对应 coord 后两维）网格尺寸
        block_size: ``(bsx, bsy, brx, bry)`` 或标量（四轴同尺寸）
        stride: ``(ssx, ssy, srx, sry)`` 或标量

    Returns:
        与 ``make_grid_blocks_from_shape`` 相同结构的 block 列表。
    """
    bsx, bsy, brx, bry = _parse_4tuple(block_size, "block_size")
    ssx, ssy, srx, sry = _parse_4tuple(stride, "stride")
    if min(nsx, nsy, nrx, nry) <= 0:
        raise ValueError("nsx, nsy, nrx, nry must be positive")

    x_starts = _compute_starts(nsx, bsx, ssx)
    y_starts = _compute_starts(nsy, bsy, ssy)
    rx_starts = _compute_starts(nrx, brx, srx)
    ry_starts = _compute_starts(nry, bry, sry)

    dims = (nsx, nsy, nrx, nry)
    blocks: List[Dict[str, Any]] = []
    block_id = 0

    for isx0 in x_starts:
        for isy0 in y_starts:
            for irx0 in rx_starts:
                for iry0 in ry_starts:
                    isx = np.arange(isx0, min(isx0 + bsx, nsx), dtype=np.int64)
                    isy = np.arange(isy0, min(isy0 + bsy, nsy), dtype=np.int64)
                    irx = np.arange(irx0, min(irx0 + brx, nrx), dtype=np.int64)
                    iry = np.arange(iry0, min(iry0 + bry, nry), dtype=np.int64)
                    if isx.size == 0 or isy.size == 0 or irx.size == 0 or iry.size == 0:
                        continue

                    II, JJ, KK, LL = np.meshgrid(isx, isy, irx, iry, indexing="ij")
                    point_idx = ravel_grid_index_4d(
                        II.ravel(), JJ.ravel(), KK.ravel(), LL.ravel(), dims
                    )

                    lenx, leny, lenrx, lenry = (
                        int(isx.size),
                        int(isy.size),
                        int(irx.size),
                        int(iry.size),
                    )
                    csx = int(isx0 + min(bsx - 1, lenx // 2))
                    csy = int(isy0 + min(bsy - 1, leny // 2))
                    crx = int(irx0 + min(brx - 1, lenrx // 2))
                    cry = int(iry0 + min(bry - 1, lenry // 2))
                    center_idx = int(
                        np.ravel_multi_index((csx, csy, crx, cry), dims, order="C")
                    )

                    blocks.append(
                        {
                            "block_id": block_id,
                            "grid_point_indices": point_idx,
                            "block_center_grid_index": np.asarray(center_idx, dtype=np.int64),
                            "block_center_coord": None,
                        }
                    )
                    block_id += 1
    return blocks


def make_grid_blocks_from_index_map_4d(
    index_map_4d: ArrayLike,
    block_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]],
) -> List[Dict[str, Any]]:
    """四维逻辑格上的滑窗：``index_map_4d[isx,isy,irx,iry]`` 为 ``coord_grid_norm`` 行号或 ``-1``（无点）。

    用于炮/检阵列**非完整长方体**、但逻辑上仍为等间距四维格、仅部分格点有有效坐标的情形。
    ``len(coord_grid_norm)`` 只需等于**有效点个数**（与 ``index_map>=0`` 的取值范围一致），
    不必等于 ``nsx*nsy*nrx*nry``。

    Returns:
        与 ``make_grid_blocks_from_shape_4d`` 相同字段；若窗口中心格为 ``-1``，
        ``block_center_grid_index`` 为 ``-1``，调用方需用块内坐标均值等作为 ``build_infer_patch`` 的中心。
    """
    m = np.asarray(index_map_4d, dtype=np.int64)
    if m.ndim != 4:
        raise ValueError("index_map_4d must have shape [nsx, nsy, nrx, nry]")
    nsx, nsy, nrx, nry = (int(m.shape[i]) for i in range(4))
    bsx, bsy, brx, bry = _parse_4tuple(block_size, "block_size")
    ssx, ssy, srx, sry = _parse_4tuple(stride, "stride")
    if min(nsx, nsy, nrx, nry) <= 0:
        raise ValueError("all index map dimensions must be positive")

    x_starts = _compute_starts(nsx, bsx, ssx)
    y_starts = _compute_starts(nsy, bsy, ssy)
    rx_starts = _compute_starts(nrx, brx, srx)
    ry_starts = _compute_starts(nry, bry, sry)

    blocks: List[Dict[str, Any]] = []
    block_id = 0

    for isx0 in x_starts:
        for isy0 in y_starts:
            for irx0 in rx_starts:
                for iry0 in ry_starts:
                    isx_hi = min(isx0 + bsx, nsx)
                    isy_hi = min(isy0 + bsy, nsy)
                    irx_hi = min(irx0 + brx, nrx)
                    iry_hi = min(iry0 + bry, nry)
                    sub = m[isx0:isx_hi, isy0:isy_hi, irx0:irx_hi, iry0:iry_hi]
                    flat = sub.ravel()
                    valid = flat[flat >= 0]
                    if valid.size == 0:
                        continue
                    point_idx = np.unique(valid.astype(np.int64, copy=False))

                    lenx = isx_hi - isx0
                    leny = isy_hi - isy0
                    lenrx = irx_hi - irx0
                    lenry = iry_hi - iry0
                    csx = int(isx0 + min(bsx - 1, lenx // 2))
                    csy = int(isy0 + min(bsy - 1, leny // 2))
                    crx = int(irx0 + min(brx - 1, lenrx // 2))
                    cry = int(iry0 + min(bry - 1, lenry // 2))
                    center_val = int(m[csx, csy, crx, cry])

                    blocks.append(
                        {
                            "block_id": block_id,
                            "grid_point_indices": point_idx,
                            "block_center_grid_index": np.asarray(center_val, dtype=np.int64),
                            "block_center_coord": None,
                        }
                    )
                    block_id += 1
    return blocks


def _make_4d_blocks(
    coord_grid_norm: ArrayLike,
    block_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]],
    grid_shape_4d: Optional[Tuple[int, int, int, int]] = None,
    grid_index_map_4d: Optional[ArrayLike] = None,
) -> List[Dict[str, Any]]:
    """Build 4D logical-grid blocks from either a dense shape or sparse index map."""
    grid_coords = _check_coord_2d(_to_numpy(coord_grid_norm, np.float32), "coord_grid_norm")
    n_g = int(grid_coords.shape[0])

    if grid_index_map_4d is not None:
        map_arr = np.asarray(grid_index_map_4d, dtype=np.int64)
        if map_arr.ndim != 4:
            raise ValueError("grid_index_map_4d must have shape [nsx, nsy, nrx, nry]")
        if np.any((map_arr != -1) & (map_arr < 0)):
            raise ValueError("grid_index_map_4d: only -1 (empty) or non-negative row indices are allowed")
        valid = map_arr[map_arr >= 0]
        if valid.size:
            mx = int(np.max(valid))
            if mx >= n_g:
                raise ValueError(
                    f"grid_index_map_4d values must be < len(coord_grid_norm)={n_g}, got max {mx}"
                )
        return make_grid_blocks_from_index_map_4d(
            map_arr, block_size=block_size, stride=stride
        )

    if grid_shape_4d is None:
        raise ValueError(
            "Provide grid_shape_4d (dense C-order grid) or grid_index_map_4d (logical 4D map, -1 empty)."
        )
    nsx, nsy, nrx, nry = (int(x) for x in grid_shape_4d)
    n_expect = nsx * nsy * nrx * nry
    if n_g != n_expect:
        raise ValueError(
            f"coord_grid_norm 行数 {n_g} 必须等于 grid_shape_4d 四轴乘积 "
            f"{nsx}×{nsy}×{nrx}×{nry}={n_expect}。若阵列非完整长方体，请改用 "
            "grid_index_map_4d（-1 表示空位）。"
        )
    return make_grid_blocks_from_shape_4d(
        nsx=nsx,
        nsy=nsy,
        nrx=nrx,
        nry=nry,
        block_size=block_size,
        stride=stride,
    )


def make_grid_blocks(
    grid_shape_or_indices: Union[Tuple[int, int], Sequence[int], np.ndarray],
    block_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]],
) -> List[Dict[str, Any]]:
    """General interface for regular-grid block generation.

    Args:
        grid_shape_or_indices:
            - (nx, ny), or
            - 2D index map [ny, nx], each value is global grid index
        block_size: (bx, by) or scalar
        stride: (sx, sy) or scalar

    Returns:
        list of block dicts with overlap and boundary completion.
    """
    if isinstance(grid_shape_or_indices, tuple) and len(grid_shape_or_indices) == 2:
        nx, ny = int(grid_shape_or_indices[0]), int(grid_shape_or_indices[1])
        return make_grid_blocks_from_shape(nx=nx, ny=ny, block_size=block_size, stride=stride)

    arr = _to_numpy(grid_shape_or_indices, np.int64)
    if arr.ndim != 2:
        raise ValueError(
            "grid_shape_or_indices must be (nx, ny) or 2D index map [ny, nx]"
        )

    ny, nx = arr.shape
    bx, by = _parse_2tuple(block_size, "block_size")
    sx, sy = _parse_2tuple(stride, "stride")
    x_starts = _compute_starts(nx, bx, sx)
    y_starts = _compute_starts(ny, by, sy)

    blocks: List[Dict[str, Any]] = []
    block_id = 0
    for y0 in y_starts:
        for x0 in x_starts:
            sub = arr[y0:y0 + by, x0:x0 + bx]
            point_idx = sub.reshape(-1).astype(np.int64)
            cy = y0 + min(by - 1, sub.shape[0] // 2)
            cx = x0 + min(bx - 1, sub.shape[1] // 2)
            center_idx = int(arr[cy, cx])
            blocks.append(
                {
                    "block_id": block_id,
                    "grid_point_indices": point_idx,
                    "block_center_grid_index": np.asarray(center_idx, dtype=np.int64),
                    "block_center_coord": None,
                }
            )
            block_id += 1
    return blocks


def build_infer_patch(
    block_center_coord: ArrayLike,
    block_grid_indices: ArrayLike,
    coord_obs_norm: ArrayLike,
    coord_grid_norm: ArrayLike,
    trace_obs: Optional[ArrayLike] = None,
    k_patch: int = 64,
    top_l: int = 128,
    metric_weights: Optional[Sequence[float]] = None,
    beta: float = 0.3,
    grid_query_mask: Optional[ArrayLike] = None,
    obs_valid_mask: Optional[ArrayLike] = None,
    return_features: bool = False,
) -> Dict[str, Any]:
    """Build one inference sample from one query set.

    Args:
        block_center_coord: kept for API compatibility. In the current inference semantics,
            each query point uses its own coordinate as anchor to search context, so this
            argument is metadata only.
        block_grid_indices: global grid indices of query traces in this sample, shape [Q]
        coord_obs_norm: observed coords, shape [N_obs, 4]
        coord_grid_norm: grid coords, shape [N_grid, 4]
        trace_obs: observed traces, shape [N_obs, T]. 仅当 ``return_features=True`` 时
            必须提供；否则可省略（仅算索引时只依赖坐标）。
        k_patch: observed context size
        top_l: local candidate size for context selection
        metric_weights: weighted Euclidean metric weights
        beta: diversity weight in diverse_topk
        grid_query_mask: optional bool mask [N_grid], True means this grid point is query target
        obs_valid_mask: optional bool mask [N_obs]. When provided, it only restricts
            which observed traces are allowed to enter the candidate/context pool.
            对 ``block_grid_indices`` 中的每个 query，都会以该 query 的坐标为锚点独立执行
            ``top_l + diverse_topk`` 选 context；若不传该 mask，则所有观测道都可作为 context。
        return_features: if True, also return query-first network payload arrays

    Returns:
        dict with primary semantics:
            - ``grid_query_idx``: [Q]，当前样本中的 query 规则格全局下标
            - ``anchor_grid_idx``: [Q]，每个 query 都把自己作为 context 搜索锚点
            - ``context_idx_per_anchor``: length-Q list，每个 query 各自选出的观测 context 下标
            - ``context_idx``: [K']，将上式按出现顺序去重后的 context 观测下标

        兼容旧接口：
            - ``context_idx`` 仍表示 context 观测集合
            - ``grid_query_idx`` 仍表示 query 规则格集合

        if return_features=True, also includes:
            - ``coord_query`` / ``coord_context``
            - ``network_input_trace``: [Q+K', T]，query 在前且为 0，context 在后
            - ``network_input_coord``: [Q+K', 4]，与上式顺序一致
            - ``network_input_is_query``: [Q+K'] bool，前 Q 条为 True
    """
    obs_coords = _check_coord_2d(_to_numpy(coord_obs_norm, np.float32), "coord_obs_norm")
    grid_coords = _check_coord_2d(_to_numpy(coord_grid_norm, np.float32), "coord_grid_norm")
    traces: Optional[np.ndarray] = None
    if return_features:
        if trace_obs is None:
            raise ValueError("trace_obs is required when return_features=True")
        traces = _to_numpy(trace_obs, np.float32)
        if traces.ndim != 2 or traces.shape[0] != obs_coords.shape[0]:
            raise ValueError("trace_obs must have shape [N_obs, T] and align with coord_obs_norm")
    elif trace_obs is not None:
        # 可选：调用方仍传入 trace 时仅做行数对齐检查
        t = _to_numpy(trace_obs, np.float32)
        if t.ndim != 2 or t.shape[0] != obs_coords.shape[0]:
            raise ValueError("trace_obs must align with coord_obs_norm when provided")

    center = _to_numpy(block_center_coord, np.float32).reshape(-1)
    if center.shape[0] != 4:
        raise ValueError("block_center_coord must have shape [4]")

    grid_idx = _to_numpy(block_grid_indices, np.int64).reshape(-1)
    if grid_idx.size == 0:
        raise ValueError("block_grid_indices cannot be empty")
    n_grid = int(grid_coords.shape[0])
    if np.any(grid_idx < 0) or np.any(grid_idx >= n_grid):
        raise ValueError(
            "block_grid_indices 越界：须在 [0, N_grid) 内，"
            f"N_grid={n_grid}，当前 min={int(np.min(grid_idx))}, max={int(np.max(grid_idx))}。"
            "常见原因：coord_grid_norm 行数与 grid_shape 四维乘积不一致，"
            "或网格展平顺序与 make_grid_blocks_from_shape_4d 使用的 C 序 (isx,isy,irx,iry) 不一致。"
        )
    qmask = _normalize_optional_bool_mask(
        grid_query_mask, grid_coords.shape[0], "grid_query_mask"
    )
    grid_idx = _filter_grid_query_indices(grid_idx, qmask)
    if grid_idx.size == 0:
        return _make_empty_infer_patch(
            center=center,
            trace_length=None if traces is None else traces.shape[1],
        )

    ovm = _normalize_optional_bool_mask(
        obs_valid_mask, obs_coords.shape[0], "obs_valid_mask"
    )

    anchor_grid_idx = grid_idx.astype(np.int64, copy=False)
    anchor_coords = grid_coords[anchor_grid_idx].astype(np.float32, copy=False)
    context_idx_per_anchor = [
        _select_context_idx_for_anchor(
            anchor_coord=anchor_coord,
            obs_coords=obs_coords,
            k_patch=k_patch,
            top_l=top_l,
            metric_weights=metric_weights,
            beta=beta,
            obs_valid_mask=ovm,
        )
        for anchor_coord in anchor_coords
    ]
    context_idx = _stable_unique_index_list(context_idx_per_anchor)

    out = {
        "anchor_grid_idx": anchor_grid_idx.astype(np.int64),
        "anchor_coord": anchor_coords.astype(np.float32),
        "block_center_coord": center.astype(np.float32),
        "context_idx_per_anchor": context_idx_per_anchor,
        "context_idx": context_idx.astype(np.int64),
        "grid_query_idx": grid_idx.astype(np.int64),
        "query_first_count": np.asarray(grid_idx.size, dtype=np.int64),
        "context_count": np.asarray(context_idx.size, dtype=np.int64),
    }
    if return_features:
        assert traces is not None
        coord_query = grid_coords[grid_idx].astype(np.float32)
        coord_context = obs_coords[context_idx].astype(np.float32)
        trace_context = traces[context_idx].astype(np.float32)
        trace_query_input = np.zeros((grid_idx.size, traces.shape[1]), dtype=np.float32)
        out.update(
            {
                "coord_context": coord_context,
                "trace_context": trace_context,
                "coord_query": coord_query,
                "trace_query_input": trace_query_input,
                "network_input_trace": np.concatenate([trace_query_input, trace_context], axis=0),
                "network_input_coord": np.concatenate([coord_query, coord_context], axis=0),
                "network_input_is_query": np.concatenate(
                    [
                        np.ones((grid_idx.size,), dtype=bool),
                        np.zeros((context_idx.size,), dtype=bool),
                    ],
                    axis=0,
                ),
            }
        )
    return out


def accumulate_block_predictions(
    pred_sum: ArrayLike,
    pred_cnt: ArrayLike,
    block_pred: ArrayLike,
    block_grid_indices: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """Accumulate block predictions to global sum/count buffers.

    Args:
        pred_sum: [N_grid, T]
        pred_cnt: [N_grid]
        block_pred: [B, T]
        block_grid_indices: [B], global grid indices for this block

    Returns:
        Updated pred_sum, pred_cnt (same arrays, also returned for chaining)
    """
    sum_np = _to_numpy(pred_sum, np.float32)
    cnt_np = _to_numpy(pred_cnt, np.float32)
    blk_pred = _to_numpy(block_pred, np.float32)
    blk_idx = _to_numpy(block_grid_indices, np.int64).reshape(-1)

    if sum_np.ndim != 2:
        raise ValueError(f"pred_sum must be [N_grid, T], got {sum_np.shape}")
    if cnt_np.ndim != 1 or cnt_np.shape[0] != sum_np.shape[0]:
        raise ValueError("pred_cnt must be [N_grid] and align with pred_sum")
    if blk_pred.ndim != 2:
        raise ValueError("block_pred must be [B, T]")
    if blk_pred.shape[0] != blk_idx.size:
        raise ValueError("block_pred first dim must match block_grid_indices length")
    if blk_pred.shape[1] != sum_np.shape[1]:
        raise ValueError("block_pred second dim (T) must match pred_sum")
    if np.any(blk_idx < 0) or np.any(blk_idx >= sum_np.shape[0]):
        raise ValueError("block_grid_indices contains out-of-range index")

    sum_np[blk_idx] += blk_pred
    cnt_np[blk_idx] += 1.0
    return sum_np, cnt_np


def find_uncovered_points(pred_cnt: ArrayLike) -> np.ndarray:
    """Return global grid indices where prediction count is zero."""
    cnt_np = _to_numpy(pred_cnt, np.float32).reshape(-1)
    return np.flatnonzero(cnt_np <= 0.0).astype(np.int64)


def finalize_predictions(
    pred_sum: ArrayLike,
    pred_cnt: ArrayLike,
    trusted_grid_mask: Optional[ArrayLike] = None,
    trusted_grid_trace: Optional[ArrayLike] = None,
) -> Dict[str, np.ndarray]:
    """Finalize fused predictions by averaging overlaps and optional hard replacement.

    Args:
        pred_sum: [N_grid, T]
        pred_cnt: [N_grid]
        trusted_grid_mask: optional bool mask [N_grid]
        trusted_grid_trace: optional trace array [N_grid, T], used when mask=True

    Returns:
        dict with:
            pred: final prediction [N_grid, T]
            covered_mask: bool [N_grid]
            uncovered_idx: int64 [N_uncovered]
            pred_cnt: float [N_grid]
    """
    sum_np = _to_numpy(pred_sum, np.float32)
    cnt_np = _to_numpy(pred_cnt, np.float32).reshape(-1)
    if sum_np.ndim != 2:
        raise ValueError("pred_sum must be [N_grid, T]")
    if cnt_np.shape[0] != sum_np.shape[0]:
        raise ValueError("pred_cnt must align with pred_sum")

    covered = cnt_np > 0.0
    pred = np.zeros_like(sum_np, dtype=np.float32)
    if np.any(covered):
        pred[covered] = sum_np[covered] / cnt_np[covered, None]

    if trusted_grid_mask is not None and trusted_grid_trace is not None:
        tmask = _to_numpy(trusted_grid_mask).reshape(-1).astype(bool)
        ttrace = _to_numpy(trusted_grid_trace, np.float32)
        if tmask.shape[0] != pred.shape[0]:
            raise ValueError("trusted_grid_mask must align with N_grid")
        if ttrace.shape != pred.shape:
            raise ValueError("trusted_grid_trace must have shape [N_grid, T]")
        pred[tmask] = ttrace[tmask]

    uncovered_idx = np.flatnonzero(~covered).astype(np.int64)
    return {
        "pred": pred,
        "covered_mask": covered,
        "uncovered_idx": uncovered_idx,
        "pred_cnt": cnt_np.astype(np.float32),
    }


def fallback_infer_for_uncovered(
    pred_sum: ArrayLike,
    pred_cnt: ArrayLike,
    coord_obs_norm: ArrayLike,
    coord_grid_norm: ArrayLike,
    trace_obs: ArrayLike,
    model_predict_fn: Callable[[Dict[str, np.ndarray]], np.ndarray],
    k_patch: int = 64,
    top_l: int = 128,
    metric_weights: Optional[Sequence[float]] = None,
    beta: float = 0.3,
    max_points: Optional[int] = None,
    obs_valid_mask: Optional[ArrayLike] = None,
) -> Dict[str, np.ndarray]:
    """Fallback inference for uncovered grid points.

    Simple implementation: for each uncovered grid point, build a single-point
    infer patch and call model_predict_fn once.

    Args:
        pred_sum: [N_grid, T]
        pred_cnt: [N_grid]
        coord_obs_norm: [N_obs, 4]
        coord_grid_norm: [N_grid, 4]
        trace_obs: [N_obs, T]
        model_predict_fn: callable(patch_dict) -> np.ndarray [B, T]
        k_patch, top_l, metric_weights, beta: same as build_infer_patch
        max_points: optional cap on number of uncovered points to process
        obs_valid_mask: optional [N_obs] bool, same as build_infer_patch. It only
            restricts context candidates, not the final input missingness semantics.

    Returns:
        dict with updated pred_sum/pred_cnt and processed indices.
    """
    sum_np = _to_numpy(pred_sum, np.float32)
    cnt_np = _to_numpy(pred_cnt, np.float32).reshape(-1)
    grid_coords = _check_coord_2d(_to_numpy(coord_grid_norm, np.float32), "coord_grid_norm")
    if sum_np.shape[0] != grid_coords.shape[0]:
        raise ValueError("pred_sum and coord_grid_norm must share same N_grid")

    uncovered = find_uncovered_points(cnt_np)
    if max_points is not None:
        uncovered = uncovered[: max(0, int(max_points))]

    ovm = None
    if obs_valid_mask is not None:
        ovm = _to_numpy(obs_valid_mask).reshape(-1).astype(bool)
        n_obs_chk = _check_coord_2d(_to_numpy(coord_obs_norm, np.float32), "coord_obs_norm").shape[0]
        if ovm.shape[0] != n_obs_chk:
            raise ValueError("obs_valid_mask must have shape [N_obs]")

    processed: List[int] = []
    for gidx in uncovered:
        center = grid_coords[gidx]
        patch = build_infer_patch(
            block_center_coord=center,
            block_grid_indices=np.asarray([gidx], dtype=np.int64),
            coord_obs_norm=coord_obs_norm,
            coord_grid_norm=coord_grid_norm,
            trace_obs=trace_obs,
            k_patch=k_patch,
            top_l=top_l,
            metric_weights=metric_weights,
            beta=beta,
            obs_valid_mask=ovm,
            return_features=True,
        )
        block_pred = _to_numpy(model_predict_fn(patch), np.float32)
        if block_pred.ndim != 2 or block_pred.shape[0] != 1:
            raise ValueError("fallback model_predict_fn must return [1, T] for single-point patch")
        sum_np, cnt_np = accumulate_block_predictions(
            pred_sum=sum_np,
            pred_cnt=cnt_np,
            block_pred=block_pred,
            block_grid_indices=np.asarray([gidx], dtype=np.int64),
        )
        processed.append(int(gidx))

    return {
        "pred_sum": sum_np,
        "pred_cnt": cnt_np.astype(np.float32),
        "processed_idx": np.asarray(processed, dtype=np.int64),
    }


def pad_index_list_to_2d(
    index_list: Sequence[np.ndarray],
    pad_value: int = -1,
    dtype: np.dtype = np.int64,
) -> np.ndarray:
    """Pad a list of 1D index arrays to a fixed-width 2D array.

    Args:
        index_list: list of [Li] index arrays
        pad_value: padding value for shorter rows
        dtype: output dtype

    Returns:
        padded: shape [N, Lmax]
    """
    n = len(index_list)
    if n == 0:
        return np.zeros((0, 0), dtype=dtype)
    max_len = max(int(np.asarray(x).reshape(-1).size) for x in index_list)
    out = np.full((n, max_len), fill_value=pad_value, dtype=dtype)
    for i, idx in enumerate(index_list):
        arr = np.asarray(idx, dtype=dtype).reshape(-1)
        out[i, : arr.size] = arr
    return out


def _coords_from_index_2d(
    index_2d: np.ndarray,
    coord_norm: np.ndarray,
    coord_name: str,
) -> np.ndarray:
    """Gather 4D coordinates by a padded 2D index array."""
    coords = _check_coord_2d(_to_numpy(coord_norm, np.float32), coord_name)
    idx_2d = np.asarray(index_2d, dtype=np.int64)
    if idx_2d.ndim != 2:
        raise ValueError("index_2d must be 2D [B, K]")
    if idx_2d.shape[0] == 0:
        return np.zeros((0, 0, 4), dtype=np.float32)
    out = np.full((idx_2d.shape[0], idx_2d.shape[1], 4), np.nan, dtype=np.float32)
    valid = idx_2d >= 0
    if not np.any(valid):
        return out
    flat_idx = idx_2d[valid]
    if np.any(flat_idx >= coords.shape[0]):
        raise ValueError(
            f"index_2d 含越界下标 (max={int(np.max(flat_idx))}, N={coords.shape[0]})"
        )
    out[valid] = coords[flat_idx]
    return out


def _obs_coords_from_patch_idx_2d(
    patch_idx_2d: np.ndarray,
    coord_obs_norm: np.ndarray,
) -> np.ndarray:
    """由 ``patch_idx_2d`` 取观测道四维坐标，与 ``patch_idx_2d`` 行列对齐。"""
    return _coords_from_index_2d(
        index_2d=patch_idx_2d,
        coord_norm=coord_obs_norm,
        coord_name="coord_obs_norm",
    )


def _summarize_patch_sizes(
    context_list: Sequence[np.ndarray],
    grid_query_list: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Summarize query/context counts for a precomputed infer patch list."""
    n_patch = len(grid_query_list)
    if n_patch == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )
    patch_query_count = np.fromiter(
        (len(x) for x in grid_query_list), dtype=np.int64, count=n_patch
    )
    patch_context_count = np.fromiter(
        (len(x) for x in context_list), dtype=np.int64, count=n_patch
    )
    patch_input_count = patch_query_count + patch_context_count
    patch_query_ratio = (
        patch_query_count.astype(np.float64)
        / np.maximum(patch_input_count.astype(np.float64), 1.0)
    ).astype(np.float32)
    return patch_query_count, patch_context_count, patch_input_count, patch_query_ratio


def _ensure_full_query_coverage(
    grid_query_list: Sequence[np.ndarray],
    grid_query_mask: Optional[np.ndarray],
) -> None:
    """Raise when a full-query-coverage request is violated."""
    if grid_query_mask is None:
        return
    target_idx = np.flatnonzero(grid_query_mask)
    if len(grid_query_list) == 0:
        covered_idx = np.zeros((0,), dtype=np.int64)
    else:
        covered_idx = np.unique(np.concatenate(grid_query_list, axis=0))
    uncovered = np.setdiff1d(target_idx, covered_idx, assume_unique=False)
    if uncovered.size > 0:
        raise ValueError(
            f"query coverage check failed: {uncovered.size} target points are not covered"
        )


def _report_infer_precompute_stats(
    prefix: str,
    grid_query_list: Sequence[np.ndarray],
    grid_query_mask: Optional[np.ndarray],
    patch_query_count: np.ndarray,
    patch_context_count: np.ndarray,
    patch_query_ratio: np.ndarray,
) -> None:
    """Print compact infer-precompute stats."""
    if not grid_query_list:
        return
    all_q = np.concatenate(grid_query_list, axis=0)
    n_slot = int(all_q.size)
    n_uniq = int(np.unique(all_q).size)
    dup_rate = float((n_slot - n_uniq) / n_slot) if n_slot > 0 else 0.0
    line = (
        f"[{prefix}] patches={len(grid_query_list)} "
        f"query_slots={n_slot} unique_query_indices={n_uniq} duplicate_rate={dup_rate:.6f}"
    )
    if grid_query_mask is not None:
        n_miss = int(np.count_nonzero(grid_query_mask))
        miss_idx = np.flatnonzero(grid_query_mask)
        cov = int(np.intersect1d(np.unique(all_q), miss_idx, assume_unique=False).size)
        line += f" missing_grid={n_miss} covered_unique={cov}/{n_miss}"
    print(line)
    if patch_query_count.size == 0:
        return
    qc = patch_query_count
    kc = patch_context_count
    qr = patch_query_ratio
    print(
        f"[{prefix}] per-patch Q: min={int(qc.min())} max={int(qc.max())} "
        f"mean={float(qc.mean()):.4f} | Kctx: min={int(kc.min())} max={int(kc.max())} "
        f"mean={float(kc.mean()):.4f} | Q/(Q+Kctx): min={float(qr.min()):.6f} "
        f"max={float(qr.max()):.6f} mean={float(qr.mean()):.6f} "
        f"p50={float(np.percentile(qr, 50)):.6f} p5={float(np.percentile(qr, 5)):.6f} "
        f"p95={float(np.percentile(qr, 95)):.6f}"
    )


def _build_infer_precompute_output(
    block_ids: Sequence[int],
    center_grid_ids: Sequence[int],
    context_list: Sequence[np.ndarray],
    grid_query_list: Sequence[np.ndarray],
    anchor_grid_list: Sequence[np.ndarray],
    context_idx_per_anchor_list: Sequence[List[np.ndarray]],
    obs_coords: np.ndarray,
    grid_coords: np.ndarray,
    obs_valid_mask: Optional[np.ndarray],
    include_padded: bool = True,
) -> Dict[str, Any]:
    """Pack infer-precompute outputs with list-first semantics.

    ``patch_idx`` in inference is the same thing as ``context_idx``. To make the
    returned schema easier to read, the dict keeps both names as aliases.
    ``include_padded=False`` avoids constructing the legacy ``*_2d`` arrays.
    """
    patch_query_count, patch_context_count, patch_input_count, patch_query_ratio = _summarize_patch_sizes(
        context_list=context_list,
        grid_query_list=grid_query_list,
    )
    out = {
        "num_samples": int(len(grid_query_list)),
        "block_id": np.asarray(block_ids, dtype=np.int64),
        "block_center_grid_idx": np.asarray(center_grid_ids, dtype=np.int64),
        "patch_idx_list": [np.asarray(x, dtype=np.int64).reshape(-1) for x in context_list],
        "context_idx_list": [np.asarray(x, dtype=np.int64).reshape(-1) for x in context_list],
        "grid_query_idx_list": [np.asarray(x, dtype=np.int64).reshape(-1) for x in grid_query_list],
        "anchor_grid_idx_list": [np.asarray(x, dtype=np.int64).reshape(-1) for x in anchor_grid_list],
        "context_idx_per_anchor_list": [
            [np.asarray(y, dtype=np.int64).reshape(-1) for y in xs]
            for xs in context_idx_per_anchor_list
        ],
        "patch_query_count": patch_query_count,
        "patch_context_count": patch_context_count,
        "patch_input_count": patch_input_count,
        "patch_query_ratio": patch_query_ratio,
    }
    if not include_padded:
        return out

    patch_idx_2d = pad_index_list_to_2d(context_list, pad_value=-1, dtype=np.int64)
    if obs_valid_mask is not None:
        patch_mask_2d = np.zeros(patch_idx_2d.shape, dtype=np.float32)
        valid = patch_idx_2d >= 0
        patch_mask_2d[valid] = obs_valid_mask[patch_idx_2d[valid]].astype(np.float32)
    else:
        patch_mask_2d = (patch_idx_2d >= 0).astype(np.float32)
    grid_query_idx_2d = pad_index_list_to_2d(grid_query_list, pad_value=-1, dtype=np.int64)
    anchor_grid_idx_2d = pad_index_list_to_2d(anchor_grid_list, pad_value=-1, dtype=np.int64)
    out.update(
        {
            "patch_idx_2d": patch_idx_2d,
            "context_idx_2d": patch_idx_2d,
            "patch_mask_2d": patch_mask_2d,
            "patch_coords_2d": _coords_from_index_2d(patch_idx_2d, obs_coords, "coord_obs_norm"),
            "context_coords_2d": _coords_from_index_2d(
                patch_idx_2d, obs_coords, "coord_obs_norm"
            ),
            "grid_query_idx_2d": grid_query_idx_2d,
            "grid_query_coords_2d": _coords_from_index_2d(
                grid_query_idx_2d, grid_coords, "coord_grid_norm"
            ),
            "anchor_grid_idx_2d": anchor_grid_idx_2d,
        }
    )
    return out


def precompute_train_patches_2d(
    coord_obs_norm: ArrayLike,
    trace_obs: ArrayLike,
    trusted_idx: ArrayLike,
    num_anchors: int,
    k_patch: int = 64,
    top_l: int = 128,
    metric_weights: Optional[Sequence[float]] = None,
    beta: float = 0.3,
    anchor_selector: str = "farthest_point_sampling",
    facility_nearest_l: Optional[int] = None,
    value_local_top_l: Optional[int] = None,
    value_suppression: str = "subtractive",
    value_suppression_lambda: float = 1.0,
    value_score_tol: float = 0.0,
    value_knn_use_gpu: bool = False,
    value_knn_gpu_batch_rows: Optional[int] = None,
    value_knn_gpu_device: Optional[str] = None,
    value_knn_full_matrix_max_n: int = 4096,
    value_suppression_use_gpu: Optional[bool] = None,
    num_query: int = 8,
    seed: int = 1551,
    pool_size: Optional[int] = None,
  ) -> Dict[str, np.ndarray]:
    """Precompute train patch indices and store them as 2D arrays.

    This stage only precomputes anchor-centered observed pools. Random training
    queries and query-first context assembly should be generated online from the pool
    via ``build_train_sample_from_pool``.

    Args:
        coord_obs_norm: observed normalized coords, [N_obs, 4]
        trace_obs: observed traces, [N_obs, T]
        trusted_idx: trusted observed indices, [N_trusted]
        num_anchors: number of anchors to sample
        k_patch, top_l, num_query, metric_weights, beta: patch params
        seed: random seed for anchor FPS
        pool_size: optional observed pool size; default follows ``build_train_patch``
        anchor_selector: ``farthest_point_sampling``, ``facility_location_anchor_sampling``, or
            ``value_based_anchor_sampling``.
        facility_nearest_l: when using ``facility_location_anchor_sampling``, optional ``nearest_l``
            passed to that routine (local gain over L nearest points; ``None`` = global over all N).
        value_local_top_l, value_suppression, value_suppression_lambda, value_score_tol: passed to
            ``value_based_anchor_sampling`` when that selector is chosen.
        value_knn_use_gpu, value_knn_gpu_batch_rows, value_knn_gpu_device, value_knn_full_matrix_max_n:
            GPU kNN options for ``value_based_anchor_sampling`` (see that function).
        value_suppression_use_gpu: optional override for the greedy suppression loop on GPU
            (``None`` = follow ``value_knn_use_gpu``).
    Returns:
        dict:
            - anchor_idx: [A]
            - patch_idx_2d: [A, Kpool]
            - pool_idx_2d: [A, Kpool]
            - context_idx_2d: [A, Kpool]  alias of pool for backward compatibility
            - query_idx_2d: [A, 0]
            - anchor_coord: [A, 4]
    """
    coords = _check_coord_2d(_to_numpy(coord_obs_norm, np.float32), "coord_obs_norm")
    traces = _to_numpy(trace_obs, np.float32)
    if traces.ndim != 2 or traces.shape[0] != coords.shape[0]:
        raise ValueError("trace_obs must have shape [N_obs, T] and align with coord_obs_norm")

    trusted = _to_numpy(trusted_idx, np.int64).reshape(-1)
    if trusted.size == 0:
        raise ValueError("trusted_idx cannot be empty")
    if np.any(trusted < 0) or np.any(trusted >= coords.shape[0]):
        raise ValueError("trusted_idx contains out-of-range index")
    print("building train patches")
    if anchor_selector == 'farthest_point_sampling':
        anchor_idx = farthest_point_sampling(
                coords=coords,
                candidate_idx=trusted,
                num_anchors=num_anchors,
                metric_weights=metric_weights,
                seed=seed
            )
    elif anchor_selector == "facility_location_anchor_sampling":
        anchor_idx = facility_location_anchor_sampling(
            coord_obs=coords,
            num_anchors=num_anchors,
            metric_weights=metric_weights,
            gain_tol=1e-3,
            nearest_l=facility_nearest_l,
        )
    elif anchor_selector == "value_based_anchor_sampling":
        anchor_idx = value_based_anchor_sampling(
            coord_obs=coords,
            num_anchors=num_anchors,
            metric_weights=metric_weights,
            local_top_l=value_local_top_l,
            suppression=value_suppression,
            suppression_lambda=value_suppression_lambda,
            score_tol=value_score_tol,
            knn_use_gpu=value_knn_use_gpu,
            knn_gpu_batch_rows=value_knn_gpu_batch_rows,
            knn_gpu_device=value_knn_gpu_device,
            knn_full_matrix_max_n=value_knn_full_matrix_max_n,
            suppression_use_gpu=value_suppression_use_gpu,
            candidate_idx=trusted,
        )
    else:
        raise ValueError(f"Invalid anchor selector: {anchor_selector}")

    patch_list: List[np.ndarray] = []
    context_list: List[np.ndarray] = []
    query_list: List[np.ndarray] = []
    anchor_coord = coords[anchor_idx].astype(np.float32)
    for aidx in _progress(anchor_idx):
        patch = build_train_patch(
            anchor_idx=int(aidx),
            coord_obs_norm=coords,
            trace_obs=traces,
            k_patch=k_patch,
            top_l=top_l,
            num_query=num_query,
            metric_weights=metric_weights,
            beta=beta,
            seed=seed,
            pool_size=pool_size,
        )
        patch_list.append(patch["patch_idx"])
        context_list.append(patch["context_idx"])
        query_list.append(patch["query_idx"])

    return {
        "anchor_idx": anchor_idx.astype(np.int64),
        "anchor_coord": anchor_coord,
        "patch_idx_2d": pad_index_list_to_2d(patch_list, pad_value=-1, dtype=np.int64),
        "pool_idx_2d": pad_index_list_to_2d(patch_list, pad_value=-1, dtype=np.int64),
        "context_idx_2d": pad_index_list_to_2d(context_list, pad_value=-1, dtype=np.int64),
        "query_idx_2d": pad_index_list_to_2d(query_list, pad_value=-1, dtype=np.int64),
    }


def precompute_infer_patches_2d(
    coord_obs_norm: ArrayLike,
    coord_grid_norm: ArrayLike,
    grid_shape_or_indices: Union[Tuple[int, int], Sequence[int], np.ndarray],
    block_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]],
    k_patch: int = 64,
    top_l: int = 128,
    metric_weights: Optional[Sequence[float]] = None,
    beta: float = 0.3,
    grid_query_mask: Optional[ArrayLike] = None,
    require_full_query_coverage: bool = False,
    obs_valid_mask: Optional[ArrayLike] = None,
) -> Dict[str, np.ndarray]:
    """Precompute inference patch indices and save as 2D arrays.

    Args:
        coord_obs_norm: [N_obs, 4]
        coord_grid_norm: [N_grid, 4]
        grid_shape_or_indices: grid shape tuple or 2D index map
        block_size: (bx, by) or int
        stride: (sx, sy) or int
        k_patch, top_l, metric_weights, beta: infer patch params
        grid_query_mask: optional bool mask [N_grid], only True positions are query targets
        require_full_query_coverage: if True, assert all target query points are covered
        obs_valid_mask: optional bool [N_obs], optional filter on which observed traces
            may be selected as context (see build_infer_patch)

    Returns:
        dict:
            - block_id: [B]
            - block_center_grid_idx: [B]
            - patch_idx_2d: [B, Kp]  观测道全局下标，padding 为 ``-1``
            - patch_mask_2d: [B, Kp]  ``float32``。这是对返回的 ``patch_idx_2d`` 槽位的标记：
              若传入 ``obs_valid_mask``，则 ``1``=该 returned context 满足候选过滤条件，
              ``0``=该槽位为 padding 或被过滤；未传入时 ``1``=非 padding，``0``=padding。
              它不表示最终 ``query + context`` 混合输入里的缺失率语义。
            - patch_coords_2d: [B, Kp, 4]  与 ``patch_idx_2d`` 对齐的 ``coord_obs_norm`` 四维坐标；padding 为 ``nan``
    """
    obs_coords = _check_coord_2d(_to_numpy(coord_obs_norm, np.float32), "coord_obs_norm")
    grid_coords = _check_coord_2d(_to_numpy(coord_grid_norm, np.float32), "coord_grid_norm")
    ovm = _normalize_optional_bool_mask(
        obs_valid_mask, obs_coords.shape[0], "obs_valid_mask"
    )
    qmask = _normalize_optional_bool_mask(
        grid_query_mask, grid_coords.shape[0], "grid_query_mask"
    )

    blocks = make_grid_blocks(
        grid_shape_or_indices=grid_shape_or_indices,
        block_size=block_size,
        stride=stride,
    )
    select_contexts = _make_context_selector(
        obs_coords=obs_coords,
        k_patch=k_patch,
        top_l=top_l,
        metric_weights=metric_weights,
        beta=beta,
        obs_valid_mask=ovm,
        use_gpu=False,
    )

    patch_list: List[np.ndarray] = []
    grid_query_list: List[np.ndarray] = []
    anchor_grid_list: List[np.ndarray] = []
    context_idx_per_anchor_list: List[List[np.ndarray]] = []
    block_ids: List[int] = []
    center_grid_ids: List[int] = []

    for blk in blocks:
        block_grid_idx = np.asarray(blk["grid_point_indices"], dtype=np.int64).reshape(-1)
        if block_grid_idx.size == 0:
            continue
        center_grid_idx = int(blk["block_center_grid_index"])
        if center_grid_idx < 0 or center_grid_idx >= grid_coords.shape[0]:
            raise ValueError("block center index out of range in generated blocks")

        q_idx = _filter_grid_query_indices(block_grid_idx, qmask)
        payload = _build_query_self_context_payload(q_idx, grid_coords, select_contexts)
        if payload is None:
            continue
        patch_list.append(payload["context_idx"])
        grid_query_list.append(payload["grid_query_idx"])
        anchor_grid_list.append(payload["anchor_grid_idx"])
        context_idx_per_anchor_list.append(payload["context_idx_per_anchor"])
        block_ids.append(int(blk["block_id"]))
        center_grid_ids.append(center_grid_idx)

    if require_full_query_coverage:
        _ensure_full_query_coverage(grid_query_list, qmask)

    return _build_infer_precompute_output(
        block_ids=block_ids,
        center_grid_ids=center_grid_ids,
        context_list=patch_list,
        grid_query_list=grid_query_list,
        anchor_grid_list=anchor_grid_list,
        context_idx_per_anchor_list=context_idx_per_anchor_list,
        obs_coords=obs_coords,
        grid_coords=grid_coords,
        obs_valid_mask=ovm,
    )


def precompute_infer_patches_4d_block_center(
    coord_obs_norm: ArrayLike,
    coord_grid_norm: ArrayLike,
    grid_shape_4d: Optional[Tuple[int, int, int, int]] = None,
    block_size: Optional[Union[int, Sequence[int]]] = None,
    stride: Optional[Union[int, Sequence[int]]] = None,
    k_patch: int = 64,
    top_l: int = 128,
    metric_weights: Optional[Sequence[float]] = None,
    beta: float = 0.3,
    grid_query_mask: Optional[ArrayLike] = None,
    require_full_query_coverage: bool = False,
    grid_index_map_4d: Optional[ArrayLike] = None,
    obs_valid_mask: Optional[ArrayLike] = None,
) -> Dict[str, np.ndarray]:
    """备份：每个滑窗块用几何中心（或块内点均值）作锚点，整块格点为 query。"""
    if block_size is None or stride is None:
        raise ValueError("block_size and stride are required")

    obs_coords = _check_coord_2d(_to_numpy(coord_obs_norm, np.float32), "coord_obs_norm")
    grid_coords = _check_coord_2d(_to_numpy(coord_grid_norm, np.float32), "coord_grid_norm")
    ovm = _normalize_optional_bool_mask(
        obs_valid_mask, obs_coords.shape[0], "obs_valid_mask"
    )
    qmask = _normalize_optional_bool_mask(
        grid_query_mask, grid_coords.shape[0], "grid_query_mask"
    )
    n_g = int(grid_coords.shape[0])
    blocks = _make_4d_blocks(
        coord_grid_norm=grid_coords,
        block_size=block_size,
        stride=stride,
        grid_shape_4d=grid_shape_4d,
        grid_index_map_4d=grid_index_map_4d,
    )

    patch_list: List[np.ndarray] = []
    grid_query_list: List[np.ndarray] = []
    anchor_grid_list: List[np.ndarray] = []
    context_idx_per_anchor_list: List[List[np.ndarray]] = []
    block_ids: List[int] = []
    center_grid_ids: List[int] = []

    for blk in _progress(blocks):
        block_grid_idx = np.asarray(blk["grid_point_indices"], dtype=np.int64).reshape(-1)
        if block_grid_idx.size == 0:
            continue
        center_grid_idx = int(blk["block_center_grid_index"])
        if center_grid_idx >= n_g:
            raise ValueError("block center index out of range in generated blocks")
        q_idx = _filter_grid_query_indices(block_grid_idx, qmask)
        if q_idx.size == 0:
            continue
        center_coord = _resolve_block_center_coord(center_grid_idx, block_grid_idx, grid_coords)
        context_idx = _select_context_idx_for_anchor(
            anchor_coord=center_coord,
            obs_coords=obs_coords,
            k_patch=k_patch,
            top_l=top_l,
            metric_weights=metric_weights,
            beta=beta,
            obs_valid_mask=ovm,
        )
        patch_list.append(context_idx)
        grid_query_list.append(q_idx.astype(np.int64, copy=False))
        anchor_grid_list.append(
            np.asarray(
                [center_grid_idx if 0 <= center_grid_idx < n_g else -1],
                dtype=np.int64,
            )
        )
        context_idx_per_anchor_list.append([context_idx.astype(np.int64, copy=False)])
        block_ids.append(int(blk["block_id"]))
        center_grid_ids.append(center_grid_idx)

    if require_full_query_coverage:
        _ensure_full_query_coverage(grid_query_list, qmask)

    return _build_infer_precompute_output(
        block_ids=block_ids,
        center_grid_ids=center_grid_ids,
        context_list=patch_list,
        grid_query_list=grid_query_list,
        anchor_grid_list=anchor_grid_list,
        context_idx_per_anchor_list=context_idx_per_anchor_list,
        obs_coords=obs_coords,
        grid_coords=grid_coords,
        obs_valid_mask=ovm,
    )


def precompute_infer_patches_4d(
    coord_obs_norm: ArrayLike,
    coord_grid_norm: ArrayLike,
    grid_shape_4d: Optional[Tuple[int, int, int, int]] = None,
    block_size: Optional[Union[int, Sequence[int]]] = None,
    stride: Optional[Union[int, Sequence[int]]] = None,
    k_patch: int = 64,
    top_l: int = 128,
    metric_weights: Optional[Sequence[float]] = None,
    beta: float = 0.3,
    grid_query_mask: Optional[ArrayLike] = None,
    require_full_query_coverage: bool = False,
    grid_index_map_4d: Optional[ArrayLike] = None,
    queries_per_patch: int = 128,
    max_query_per_patch: Optional[int] = None,
    greedy_fill_uncovered: bool = True,
    report_stats: bool = True,
    obs_valid_mask: Optional[ArrayLike] = None,
    use_gpu: bool = False,
    gpu_device: Optional[str] = None,
    gpu_query_chunk_size: int = 64,
) -> Dict[str, Any]:
    """在四维逻辑格上预计算推理 patch（轴对齐 4D 滑窗）。

    **稠密张量模式**（与训练一致）：``grid_shape_4d = (nsx, nsy, nrx, nry)``，
    ``coord_grid_norm`` 与 C 序展平一致且 ``N_grid == nsx*nsy*nrx*nry``。

    **稀疏逻辑格模式**（非完整长方体、仅等间距格点）：传入 ``grid_index_map_4d``，
    形状 ``(nsx, nsy, nrx, nry)``，每格为 ``coord_grid_norm`` 行号或 ``-1``（无点）。
    此时 ``len(coord_grid_norm)`` 只需等于有效点数，**不必**等于四轴乘积。
    若某块几何中心格为 ``-1``，块中心坐标取该块内有效网格点坐标均值。

    当前 4D 主推理路径采用更简单的 chunk-center 语义：
    1) 先按 4D 滑窗得到 block；
    2) 块内取待推理 query（``grid_query_mask=True``，未传则整块）；
    3) 每段至多 ``queries_per_patch`` 条 query 组成一个样本；
    4) 对该样本整段 query 取几何中心坐标，只查一次固定大小 ``k_patch`` 的 context；
    5) 形成近似固定 ``Q + K`` 结构的 query-first 语义；若块内 query 数不能整除 ``queries_per_patch``，
       会做均衡切分，尽量避免落出特别小的尾 patch。

    返回默认只保留无 padding 的 list 字段，避免大规模 2D padded 数组占内存。

    ``max_query_per_patch`` 为兼容旧接口：若传入则覆盖 ``queries_per_patch``。

    ``obs_valid_mask``：与 ``build_infer_patch`` 相同，仅作为 context 候选过滤器使用；
    不传时所有观测道都可被选为 context。

    ``use_gpu=True`` 时，会将一个样本内多条 query 到全部观测道的距离计算批量放到
    GPU / MPS 上执行；``diverse_topk`` 仍在 CPU 上对缩小后的候选集逐 query 运行。
    常见场景下这已覆盖主要热点。``gpu_query_chunk_size`` 控制每次并行处理多少个 query。

    主要返回键：
    - ``context_idx_list`` / ``patch_idx_list``：length-``N_patch`` list，每个样本固定大小 context 观测下标；
    - ``grid_query_idx_list``：length-``N_patch`` list，每个样本的 query 规则格下标；
    - ``anchor_grid_idx_list``：length-``N_patch`` list，每个样本用于代表 query chunk 中心的单个规则格下标；
    - ``context_idx_per_anchor_list``：length-``N_patch`` list；当前每个样本只含一个中心锚点，因此每项是单元素 list；
    - ``patch_query_count`` / ``patch_context_count`` / ``patch_input_count``：每个样本的真实长度统计。

    推理时：第 ``i`` 个样本先取 ``grid_query_idx_list[i]``（query）和 ``patch_idx_list[i]``（context），
    按固定顺序 ``query + context`` 组输入；模型输出后只把前 ``patch_query_count[i]`` 条写回 query 索引。

    ``report_stats=True`` 时除原有统计外，会打印 ``patch_query_count`` / ``patch_context_count`` /
    ``patch_query_ratio`` 的 min/max/mean 与分位数。
    """
    if block_size is None or stride is None:
        raise ValueError("block_size and stride are required")
    qp = int(queries_per_patch if max_query_per_patch is None else max_query_per_patch)
    if qp < 1:
        raise ValueError("queries_per_patch / max_query_per_patch must be >= 1")

    obs_coords = _check_coord_2d(_to_numpy(coord_obs_norm, np.float32), "coord_obs_norm")
    grid_coords = _check_coord_2d(_to_numpy(coord_grid_norm, np.float32), "coord_grid_norm")
    ovm = _normalize_optional_bool_mask(
        obs_valid_mask, obs_coords.shape[0], "obs_valid_mask"
    )
    n_g = int(grid_coords.shape[0])
    qmask_full = _normalize_optional_bool_mask(
        grid_query_mask, grid_coords.shape[0], "grid_query_mask"
    )
    if report_stats:
        print("[precompute_infer_patches_4d] build context selector")
    select_contexts = _make_context_selector(
        obs_coords=obs_coords,
        k_patch=k_patch,
        top_l=top_l,
        metric_weights=metric_weights,
        beta=beta,
        obs_valid_mask=ovm,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
        gpu_query_chunk_size=gpu_query_chunk_size,
    )
    if report_stats:
        print("[precompute_infer_patches_4d] build 4D blocks")
    blocks = _make_4d_blocks(
        coord_grid_norm=grid_coords,
        block_size=block_size,
        stride=stride,
        grid_shape_4d=grid_shape_4d,
        grid_index_map_4d=grid_index_map_4d,
    )

    context_list: List[np.ndarray] = []
    grid_query_list: List[np.ndarray] = []
    anchor_grid_list: List[np.ndarray] = []
    context_idx_per_anchor_list: List[List[np.ndarray]] = []
    block_ids: List[int] = []
    center_grid_ids: List[int] = []
    blocks_by_id: Dict[int, Dict[str, Any]] = {}
    prepared_blocks: List[Tuple[int, int, np.ndarray]] = []
    need_sort_query_idx = grid_index_map_4d is not None
    w = parse_metric_weights(metric_weights).astype(np.float32, copy=False)

    def _try_add_patch(g: np.ndarray, bid: int, block_center_idx: int) -> bool:
        g = np.asarray(g, dtype=np.int64).reshape(-1)
        if g.size == 0:
            return False
        query_coords = grid_coords[g].astype(np.float32, copy=False)
        anchor_coord = np.mean(query_coords, axis=0).astype(np.float32, copy=False)
        d2 = np.sum(((query_coords - anchor_coord[None, :]) ** 2) * w[None, :], axis=1)
        anchor_grid_idx = int(g[int(np.argmin(d2))])
        context_batch = select_contexts(anchor_coord[None, :])
        if len(context_batch) != 1:
            raise RuntimeError(
                "context selector returned inconsistent center-context batch size in precompute_infer_patches_4d"
            )
        context_idx = np.asarray(context_batch[0], dtype=np.int64).reshape(-1)
        if context_idx.size == 0:
            return False
        context_list.append(context_idx)
        grid_query_list.append(g.astype(np.int64, copy=False))
        anchor_grid_list.append(np.asarray([anchor_grid_idx], dtype=np.int64))
        context_idx_per_anchor_list.append([context_idx])
        block_ids.append(bid)
        center_grid_ids.append(block_center_idx)
        return True

    for blk in _progress(blocks, desc="prepare 4d blocks"):
        blocks_by_id[int(blk["block_id"])] = blk
        block_grid_idx = np.asarray(blk["grid_point_indices"], dtype=np.int64).reshape(-1)
        if block_grid_idx.size == 0:
            continue
        center_grid_idx = int(blk["block_center_grid_index"])
        if center_grid_idx < 0 or center_grid_idx >= n_g:
            center_grid_idx = int(block_grid_idx[len(block_grid_idx) // 2])
        q_idx = _filter_grid_query_indices(block_grid_idx, qmask_full)
        if q_idx.size == 0:
            continue
        if need_sort_query_idx:
            q_idx = _sort_grid_indices_by_coord(q_idx, grid_coords)
        prepared_blocks.append((int(blk["block_id"]), center_grid_idx, q_idx))
    if report_stats:
        total_query_slots = int(sum(q_idx.size for _, _, q_idx in prepared_blocks))
        est_patch_count = int(sum((q_idx.size + qp - 1) // qp for _, _, q_idx in prepared_blocks))
        print(
            "[precompute_infer_patches_4d] prepared query chunks: "
            f"blocks={len(prepared_blocks)} query_slots={total_query_slots} "
            f"estimated_patches={est_patch_count} queries_per_patch={qp} context_size={int(k_patch)}"
        )

    for bid, center_grid_idx, q_idx in _progress(
        prepared_blocks, desc="assemble infer patches"
    ):
        n_chunk = max(1, (q_idx.size + qp - 1) // qp)
        for g in np.array_split(q_idx, n_chunk):
            _try_add_patch(g, bid, center_grid_idx)

    if greedy_fill_uncovered and qmask_full is not None:
        missing_set = set(np.flatnonzero(qmask_full).tolist())
        if grid_query_list:
            covered = set(np.unique(np.concatenate(grid_query_list, axis=0)).tolist())
        else:
            covered = set()
        uncovered = missing_set - covered
        while uncovered:
            best_bid = -1
            best_hit: Optional[np.ndarray] = None
            best_n = 0
            for blk in blocks:
                bg = np.asarray(blk["grid_point_indices"], dtype=np.int64).reshape(-1)
                hit = np.sort([i for i in bg if i in uncovered])
                if hit.size > best_n:
                    best_n = int(hit.size)
                    best_hit = hit
                    best_bid = int(blk["block_id"])
            if best_hit is None or best_n == 0:
                break
            if need_sort_query_idx:
                best_hit = _sort_grid_indices_by_coord(best_hit, grid_coords)
            n_chunk = max(1, (best_hit.size + qp - 1) // qp)
            for g in np.array_split(best_hit, n_chunk):
                inter = np.asarray([i for i in g if i in uncovered], dtype=np.int64)
                if inter.size == 0:
                    continue
                best_blk = blocks_by_id.get(best_bid)
                if best_blk is None:
                    raise RuntimeError("greedy_fill_uncovered: failed to locate best block metadata")
                best_center_idx = int(best_blk["block_center_grid_index"])
                if best_center_idx < 0:
                    best_center_idx = int(inter[len(inter) // 2])
                if not _try_add_patch(inter, best_bid, best_center_idx):
                    raise RuntimeError(
                        "greedy_fill_uncovered: build_infer_patch 失败，无法覆盖剩余缺失格点；"
                        "请检查坐标/观测或减小 queries_per_patch。"
                    )
                for i in inter.tolist():
                    uncovered.discard(i)

    patch_query_count, patch_context_count, patch_input_count, patch_query_ratio = _summarize_patch_sizes(
        context_list=context_list,
        grid_query_list=grid_query_list,
    )

    if report_stats:
        _report_infer_precompute_stats(
            prefix="precompute_infer_patches_4d",
            grid_query_list=grid_query_list,
            grid_query_mask=qmask_full,
            patch_query_count=patch_query_count,
            patch_context_count=patch_context_count,
            patch_query_ratio=patch_query_ratio,
        )

    if require_full_query_coverage:
        _ensure_full_query_coverage(grid_query_list, qmask_full)

    ret = _build_infer_precompute_output(
        block_ids=block_ids,
        center_grid_ids=center_grid_ids,
        context_list=context_list,
        grid_query_list=grid_query_list,
        anchor_grid_list=anchor_grid_list,
        context_idx_per_anchor_list=context_idx_per_anchor_list,
        obs_coords=obs_coords,
        grid_coords=grid_coords,
        obs_valid_mask=ovm,
        include_padded=False,
    )
    return ret


def _demo_dummy_model_predict(patch: Dict[str, Any]) -> np.ndarray:
    """Tiny query-first demo model: predict each query by context mean."""
    net_trace = _to_numpy(patch["network_input_trace"], np.float32)
    if net_trace.ndim != 2:
        raise ValueError("network_input_trace must be [N, T]")
    q = int(np.asarray(patch.get("query_first_count", 0), dtype=np.int64).reshape(()))
    t = int(net_trace.shape[1])
    if q <= 0:
        return np.zeros((0, t), dtype=np.float32)
    context = _to_numpy(patch.get("trace_context", net_trace[q:]), np.float32)
    if context.ndim != 2:
        raise ValueError("trace_context must be [K, T]")
    if context.shape[0] == 0:
        return np.zeros((q, t), dtype=np.float32)
    pred = np.mean(context, axis=0, keepdims=True).astype(np.float32)
    return np.repeat(pred, q, axis=0)


def demo_patch_sampler() -> None:
    """Run a small end-to-end synthetic demo for the patch sampler."""
    rng = np.random.default_rng(123)
    n_obs, t = 500, 96
    nx, ny = 24, 18
    n_grid = nx * ny

    coord_obs = rng.uniform(
        low=[0, 0, 0, 0],
        high=[1000, 1000, 1000, 1000],
        size=(n_obs, 4),
    ).astype(np.float32)
    trace_obs = rng.normal(size=(n_obs, t)).astype(np.float32)
    trusted_mask = rng.random(n_obs) < 0.7
    trusted_idx = np.flatnonzero(trusted_mask).astype(np.int64)
    coord_grid = rng.uniform(
        low=[0, 0, 0, 0],
        high=[1000, 1000, 1000, 1000],
        size=(n_grid, 4),
    ).astype(np.float32)

    coord_obs_norm, coord_grid_norm, norm_stats = normalize_coords(coord_obs, coord_grid)
    print("norm_stats keys:", list(norm_stats.keys()))

    anchor_idx = farthest_point_sampling(
        coords=coord_obs_norm,
        candidate_idx=trusted_idx,
        num_anchors=16,
        metric_weights=[1.0, 1.0, 0.5, 0.5],
        seed=42,
    )
    print("num_anchors:", anchor_idx.size)

    train_patch = build_train_patch(
        anchor_idx=int(anchor_idx[0]),
        coord_obs_norm=coord_obs_norm,
        trace_obs=trace_obs,
        k_patch=64,
        top_l=128,
        num_query=8,
        metric_weights=[1.0, 1.0, 0.5, 0.5],
        beta=0.3,
        seed=7,
    )
    print(
        "train_patch:",
        "context=", train_patch["context_idx"].shape[0],
        "query=", train_patch["query_idx"].shape[0],
    )

    blocks = make_grid_blocks_from_shape(nx=nx, ny=ny, block_size=(6, 5), stride=(4, 3))
    print("num_blocks:", len(blocks))

    b0 = blocks[0]
    b0_center_idx = int(b0["block_center_grid_index"])
    infer_patch0 = build_infer_patch(
        block_center_coord=coord_grid_norm[b0_center_idx],
        block_grid_indices=b0["grid_point_indices"],
        coord_obs_norm=coord_obs_norm,
        coord_grid_norm=coord_grid_norm,
        trace_obs=trace_obs,
        k_patch=64,
        top_l=128,
        metric_weights=[1.0, 1.0, 0.5, 0.5],
        beta=0.3,
        return_features=True,
    )
    print(
        "infer_patch0:",
        "context=", infer_patch0["context_idx"].shape[0],
        "query=", infer_patch0["grid_query_idx"].shape[0],
    )

    pred_sum = np.zeros((n_grid, t), dtype=np.float32)
    pred_cnt = np.zeros((n_grid,), dtype=np.float32)
    for blk in blocks:
        center_idx = int(blk["block_center_grid_index"])
        patch = build_infer_patch(
            block_center_coord=coord_grid_norm[center_idx],
            block_grid_indices=blk["grid_point_indices"],
            coord_obs_norm=coord_obs_norm,
            coord_grid_norm=coord_grid_norm,
            trace_obs=trace_obs,
            k_patch=64,
            top_l=128,
            metric_weights=[1.0, 1.0, 0.5, 0.5],
            beta=0.3,
            return_features=True,
        )
        blk_pred = _demo_dummy_model_predict(patch)
        pred_sum, pred_cnt = accumulate_block_predictions(
            pred_sum=pred_sum,
            pred_cnt=pred_cnt,
            block_pred=blk_pred,
            block_grid_indices=patch["grid_query_idx"],
        )

    uncovered_before = find_uncovered_points(pred_cnt)
    print("uncovered_before_fallback:", uncovered_before.size)

    fallback_ret = fallback_infer_for_uncovered(
        pred_sum=pred_sum,
        pred_cnt=pred_cnt,
        coord_obs_norm=coord_obs_norm,
        coord_grid_norm=coord_grid_norm,
        trace_obs=trace_obs,
        model_predict_fn=_demo_dummy_model_predict,
        k_patch=64,
        top_l=128,
        metric_weights=[1.0, 1.0, 0.5, 0.5],
        beta=0.3,
    )
    final = finalize_predictions(
        pred_sum=fallback_ret["pred_sum"],
        pred_cnt=fallback_ret["pred_cnt"],
    )
    print("final_pred_shape:", final["pred"].shape)
    print("uncovered_after_fallback:", final["uncovered_idx"].size)


PATCH_SAMPLER_API: Dict[str, Any] = {
    "normalize_coords": normalize_coords,
    "farthest_point_sampling": farthest_point_sampling,
    "top_l_neighbors": top_l_neighbors,
    "diverse_topk": diverse_topk,
    "build_train_patch": build_train_patch,
    "build_train_sample_from_pool": build_train_sample_from_pool,
    "build_infer_patch": build_infer_patch,
    "make_grid_blocks": make_grid_blocks,
    "make_grid_blocks_from_shape": make_grid_blocks_from_shape,
    "make_grid_blocks_from_shape_4d": make_grid_blocks_from_shape_4d,
    "make_grid_blocks_from_index_map_4d": make_grid_blocks_from_index_map_4d,
    "precompute_train_patches_2d": precompute_train_patches_2d,
    "precompute_infer_patches_2d": precompute_infer_patches_2d,
    "precompute_infer_patches_4d_block_center": precompute_infer_patches_4d_block_center,
    "precompute_infer_patches_4d": precompute_infer_patches_4d,
    "accumulate_block_predictions": accumulate_block_predictions,
    "find_uncovered_points": find_uncovered_points,
    "finalize_predictions": finalize_predictions,
    "fallback_infer_for_uncovered": fallback_infer_for_uncovered,
    "demo_patch_sampler": demo_patch_sampler,
}


def get_patch_sampler_api() -> Dict[str, Any]:
    """Return a shallow copy of the public patch sampler API table."""
    return dict(PATCH_SAMPLER_API)


if __name__ == "__main__":
    demo_patch_sampler()
