from math import inf
from h5py import File
import numpy as np
import math
from scipy.spatial import cKDTree
from tqdm import tqdm
from collections import defaultdict
import os 
import sys
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

try:
    from .patch_sampler import (
        normalize_coords,
        precompute_train_patches_2d,
        precompute_infer_patches_4d,
    )
except ImportError:
    # Allow direct script usage from reg_tool directory.
    from patch_sampler import (
        normalize_coords,
        precompute_train_patches_2d,
        precompute_infer_patches_4d,
    )


# 生成 pos_dict
def generate_pos(info_f):
    return {
        "sx": info_f["sx"],
        "sy": info_f["sy"],
        "rx": info_f["rx"],
        "ry": info_f["ry"],
        "shot_line": info_f["shot_line"],
        "recv_line": info_f["recv_line"],
        "shot_stake": info_f["shot_stake"],
        "recv_stake": info_f["recv_stake"],
        "cmp": info_f["cmp"],
        "cmp_line": info_f["cmp_line"],
        "offset": info_f["offset"],
        'trace_idx': info_f["trace_idx"],
    }

def _read_array(info_f, key):
    value = info_f[key]
    try:
        return value[:]
    except (TypeError, ValueError, KeyError):
        return np.asarray(value)



def _rows_as_struct(arr2d):
    """Convert 2D numeric array rows to structured scalars for vectorized set ops."""
    arr2d = np.ascontiguousarray(arr2d)
    dt = np.dtype([(f"f{i}", arr2d.dtype) for i in range(arr2d.shape[1])])
    return arr2d.view(dt).reshape(-1)


def _aggregate_raw_by_keys_mean(trace_obs, coord_obs, raw_keys):
    """Aggregate raw traces/coords by raw_keys with mean reduction."""
    raw_keys_s = _rows_as_struct(raw_keys)
    _, first_idx, inverse, counts = np.unique(
        raw_keys_s, return_index=True, return_inverse=True, return_counts=True
    )
    n_unique = int(first_idx.shape[0])
    t = int(trace_obs.shape[1])

    trace_sum = np.zeros((n_unique, t), dtype=np.float64)
    coord_sum = np.zeros((n_unique, 4), dtype=np.float64)
    np.add.at(trace_sum, inverse, trace_obs)
    np.add.at(coord_sum, inverse, coord_obs)

    trace_agg = (trace_sum / counts[:, None]).astype(np.float32)
    coord_agg = (coord_sum / counts[:, None]).astype(np.float32)
    keys_agg = raw_keys[first_idx].astype(np.int64)
    return trace_agg, coord_agg, keys_agg, counts.astype(np.int64)

def generate_binning_keys(info_f):
    shot_line = _read_array(info_f, 'shot_line').astype(np.int32)
    shot_stake = _read_array(info_f, 'shot_stake').astype(np.int32)
    recv_line = _read_array(info_f, 'recv_line').astype(np.int32)
    recv_stake = _read_array(info_f, 'recv_stake').astype(np.int32)
    return np.column_stack((shot_line, shot_stake, recv_line, recv_stake))


def raw_obs_valid_mask_from_regular_trusted_mask(
    raw_binning_keys: np.ndarray,
    reg_binning_keys: np.ndarray,
    regular_trusted_mask: np.ndarray,
) -> np.ndarray:
    """
    与 raw 观测行序 1:1 对齐的 bool 掩码，供 ``obs_valid_mask`` / 训练筛选使用。

    - ``raw_binning_keys[i]``：raw 第 ``i`` 道与 ``data`` / ``sx``… 同行的四维整型键
      ``(shot_line, shot_stake, recv_line, recv_stake)``。
    - ``reg_binning_keys[j]``：规则网格第 ``j`` 道同行序键。
    - ``regular_trusted_mask[j]``：该规则格是否属于「有道」集合（例如 H5 ``mask`` 为 1）。

    返回 ``raw_obs_valid`` 形状 ``(N_raw,)``：
    ``raw_obs_valid[i]`` 为 True 当且仅当 ``raw_binning_keys[i]`` 落在集合
    ``{ reg_binning_keys[j] | regular_trusted_mask[j] }`` 中。

    若多条 raw 道共享同一键且该键被信任，则这些行均为 True。不提供「下标列表」；
    需要下标时使用 ``np.flatnonzero(raw_obs_valid)``（与 raw 全局行号一致）。
    """
    raw_k = np.asarray(raw_binning_keys, dtype=np.int64)
    reg_k = np.asarray(reg_binning_keys, dtype=np.int64)
    m = np.asarray(regular_trusted_mask, dtype=bool).reshape(-1)
    if reg_k.shape[0] != m.size:
        raise ValueError(
            f"reg_binning_keys 行数 {reg_k.shape[0]} 与 regular_trusted_mask 长度 {m.size} 不一致"
        )
    if raw_k.ndim != 2 or raw_k.shape[1] != 4:
        raise ValueError("raw_binning_keys 须为 [N_raw, 4]")
    if reg_k.ndim != 2 or reg_k.shape[1] != 4:
        raise ValueError("reg_binning_keys 须为 [N_reg, 4]")
    trusted_reg = reg_k[m]
    raw_s = _rows_as_struct(raw_k)
    trusted_s = _rows_as_struct(trusted_reg)
    return np.isin(raw_s, trusted_s)


def binning(raw_info, regular_info, raw_data=None):
    """
    按 (shot_line, shot_stake, recv_line, recv_stake) 将不规则 raw 对齐到规则网格。

    - regular_info：键仍须唯一；与 regular_info['data'] 一一对应。
    - raw_info：允许同一键对应多条道；规则网格命中该键时，对 raw_data 中这些道在样本维上求平均后写入。

    返回:
        regularized_target: 与 regular_info['data'] 同形状
        mask: 与规则 trace 数一致，命中为 1
        report: dict，含多道合并统计（见分箱结束时的打印说明）
    """
    raw_keys = generate_binning_keys(raw_info)
    regular_keys = generate_binning_keys(regular_info)

    if raw_data is None:
        raw_data = _read_array(raw_info, 'data')
    else:
        raw_data = np.asarray(raw_data)
    regular_data = _read_array(regular_info, 'data')

    if len(raw_keys) != len(raw_data):
        raise ValueError("raw_info 的键数量与 raw_data 条数不一致")
    if len(regular_keys) != len(regular_data):
        raise ValueError("regular_info 的键数量与 regular_info['data'] 条数不一致")

    regularized_target = np.zeros_like(regular_data)
    mask = np.zeros(len(regular_keys), dtype=np.uint8)

    raw_key_to_indices = defaultdict(list)
    for idx, key in enumerate(map(tuple, raw_keys)):
        raw_key_to_indices[key].append(idx)

    regular_key_to_idx = {}
    for idx, key in enumerate(map(tuple, regular_keys)):
        if key in regular_key_to_idx:
            raise ValueError(f"regular_info 存在重复键: {key}")
        regular_key_to_idx[key] = idx

    matched_count = 0
    n_multi_trace_matches = 0
    n_raw_traces_in_multi_matches = 0
    for key, regular_idx in regular_key_to_idx.items():
        raw_list = raw_key_to_indices.get(key)
        if not raw_list:
            continue
        n_raw = len(raw_list)
        if n_raw > 1:
            n_multi_trace_matches += 1
            n_raw_traces_in_multi_matches += n_raw
        sel = np.asarray(raw_list, dtype=np.intp)
        stack = raw_data[sel]
        avg = np.mean(stack, axis=0)
        regularized_target[regular_idx] = np.asarray(avg, dtype=regular_data.dtype)
        mask[regular_idx] = 1
        matched_count += 1

    if int(mask.sum()) != matched_count:
        raise ValueError("mask 命中数量与实际匹配数量不一致")
    if regularized_target.shape != regular_data.shape:
        raise ValueError("regularized_target 与 regular_info['data'] 形状不一致")

    report = {
        'n_matched_keys': matched_count,
        'n_multi_trace_matches': n_multi_trace_matches,
        'n_raw_traces_in_multi_matches': n_raw_traces_in_multi_matches,
    }
    if n_multi_trace_matches > 0:
        print(
            f"[binning] 多道合并: {n_multi_trace_matches} 个规则键在 irregular 中对应多条道，"
            f"共 {n_raw_traces_in_multi_matches} 条原始道参与平均（已按键合并为规则格点）"
        )
    else:
        print('[binning] 所有匹配键在 irregular 侧均为单道，无多道平均。')

    return regularized_target, mask, report

def saveh5(target,info_f,info_h5,key):
    header = generate_pos(info_f)
    os.makedirs(os.path.dirname(info_h5), exist_ok=True)
    with File(info_h5, 'w') as h5f:
        g = h5f.create_group(key)
        g.create_dataset('data', data=target, compression='gzip')
        for key in header.keys():
            g.create_dataset(key, data=header[key], compression='gzip')
        if "mask" in info_f:
            g.create_dataset("mask", data=_read_array(info_f, "mask"), compression='gzip')
    return None

def gather(info_f, mode):
    """
    按共炮点道集(CSG)或共检波点道集(CRG)划分 trace 下标。

    参数:
        info_f: h5py Group 或含 shot_line/shot_stake/recv_line/recv_stake 的可切片对象
        mode: 'csg' —— 键为 (shot_line, shot_stake); 'crg' —— 键为 (recv_line, recv_stake)

    返回:
        dict[tuple, np.ndarray]: 键为二元组 (int)，值为该道集内 trace 全局下标 (int64)，
        道集内按互补键 lexsort：CSG 内按 (recv_line, recv_stake)，CRG 内按 (shot_line, shot_stake)。
    """
    shot_line = _read_array(info_f, 'shot_line').astype(np.int64)
    shot_stake = _read_array(info_f, 'shot_stake').astype(np.int64)
    recv_line = _read_array(info_f, 'recv_line').astype(np.int64)
    recv_stake = _read_array(info_f, 'recv_stake').astype(np.int64)

    n = len(shot_line)
    if not (len(shot_stake) == len(recv_line) == len(recv_stake) == n):
        raise ValueError('shot_line/shot_stake/recv_line/recv_stake 长度不一致')

    if mode == 'csg':
        buckets = defaultdict(list)
        for i in range(n):
            buckets[(int(shot_line[i]), int(shot_stake[i]))].append(i)
        out = {}
        for k, idx_list in buckets.items():
            idx = np.asarray(idx_list, dtype=np.int64)
            order = np.lexsort((recv_stake[idx], recv_line[idx]))
            out[k] = idx[order]
        return out

    if mode == 'crg':
        buckets = defaultdict(list)
        for i in range(n):
            buckets[(int(recv_line[i]), int(recv_stake[i]))].append(i)
        out = {}
        for k, idx_list in buckets.items():
            idx = np.asarray(idx_list, dtype=np.int64)
            order = np.lexsort((shot_stake[idx], shot_line[idx]))
            out[k] = idx[order]
        return out

    raise ValueError(f"未知的 gather 模式: {mode!r}，应为 'csg' 或 'crg'")


def kdtree(
    info_f,
    task='denoise',
    train_ratio=0.99,
    search_size=250,
    traces_limit=2_000_000,
    batch_limit=300_000,
    k=270,
    max_candidates=200,
    knn_batch=10000,
):
    """
    一站式 KDTree 划分流程（与 generate_dataset.split_5d_kdtree_dataset 行为一致）：

    1. 从 info_f 读取 sx, sy, rx, ry，拼成点集 points (N, 4)，float32；
    2. 按 task 划分验证集索引 val_idx 与剩余 remain_idx；
    3. 对 remain_idx（或全量）做贪心集合覆盖：大数据量走分 batch 的 KDTree+kNN，否则单次 greedy；
    4. 将覆盖块随机划分为 train / test，拼接邻域 trace 索引返回。

    参数:
        task: 'denoise' | 'interp' | 其他（如 'recon'，与主脚本一致时 train_ratio 会收紧）
        train_ratio: 训练块占比（对应主流程里 1 - test_ratio）
        search_size: 每点邻域数量 k（传给贪心覆盖）
        traces_limit: 超过此道数则启用 batch 贪心
        batch_limit: batch 贪心时每批 indices 长度
        k, max_candidates, knn_batch: 贪心单批内部参数

    返回:
        train_neighbors, test_neighbors, val_idx
        前两者为 np.ndarray（与主文件 split_5d_kdtree_dataset 相同结构）
    """

    def _greedy_single(points, indices, kk, mc, kb):
        tree = cKDTree(points[indices])
        nloc = len(indices)
        uncovered = np.ones(nloc, dtype=bool)
        neighbors = np.empty((nloc, kk), dtype=np.int32)
        for i in range(0, nloc, kb):
            batch_idx = np.arange(i, min(i + kb, nloc))
            d, rel = tree.query(points[indices[batch_idx]], k=min(kk, nloc))
            neighbors[batch_idx, : rel.shape[1]] = rel
            if rel.shape[1] < kk:
                neighbors[batch_idx, rel.shape[1] :] = rel[:, -1:]
        selected = []
        while uncovered.any():
            uncovered_idx = np.flatnonzero(uncovered)
            if len(uncovered_idx) > mc:
                candidates = np.random.choice(uncovered_idx, mc, replace=False)
            else:
                candidates = uncovered_idx
            best = -1
            best_cov = -1
            for c in candidates:
                cov = np.count_nonzero(uncovered[neighbors[c]])
                if cov > best_cov:
                    best_cov = cov
                    best = c
            if best < 0:
                break
            selected.append(indices[neighbors[best]])
            uncovered[neighbors[best]] = False
        return np.array(selected)

    def _greedy_batch(points, indices, kk, mc, bl):
        print(f"点数 {len(indices)}，启用 batch greedy covering")
        all_selected = []
        for start in range(0, len(indices), bl):
            end = min(start + bl, len(indices))
            batch_indices = indices[start:end]
            print(f"处理 batch [{start}:{end}]")
            sel = _greedy_single(points, batch_indices, kk, mc, knn_batch)
            all_selected.append(sel)
        print(f"完成 {len(all_selected)} 个 batch")
        return all_selected

    rx = _read_array(info_f, 'rx').astype(np.float32)
    ry = _read_array(info_f, 'ry').astype(np.float32)
    sx = _read_array(info_f, 'sx').astype(np.float32)
    sy = _read_array(info_f, 'sy').astype(np.float32)
    n = len(rx)
    if not (len(ry) == len(sx) == len(sy) == n):
        raise ValueError('sx/sy/rx/ry 长度不一致')
    points = np.column_stack((sx, sy, rx, ry)).astype(np.float32)

    train_ratio = 1.0 ##fix (recommand when self-supervised) 
    val_idx = np.array([])
    remain_idx = np.arange(len(points))

    if len(points) > traces_limit:
        all_neighbors = _greedy_batch(points, remain_idx, search_size, max_candidates, batch_limit)
        indices = np.arange(len(all_neighbors))
        np.random.shuffle(indices)
        train_size = int(train_ratio * len(all_neighbors))
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        train_neighbors = np.concatenate([all_neighbors[i] for i in train_indices], axis=0)
        test_neighbors = np.concatenate([all_neighbors[i] for i in test_indices], axis=0)
    else:
        indices = remain_idx.copy()
        np.random.shuffle(indices)
        train_size = int(train_ratio * len(points))
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        train_neighbors = _greedy_single(points, train_indices, search_size, max_candidates, knn_batch)
        test_neighbors = _greedy_single(points, test_indices, search_size, max_candidates, knn_batch)
    return train_neighbors, test_neighbors, val_idx



## factory function
def split(info_f, mode, **kwargs):
    """
    按 mode 选择本模块已实现的划分方式（单份 info_f 输入）。

    - csg: 共炮点道集，等价 gather(info_f, 'csg')，返回 dict[(shot_line, shot_stake), ndarray]
    - crg: 共检波点道集，等价 gather(info_f, 'crg')，返回 dict[(recv_line, recv_stake), ndarray]
    - kdtree: 坐标域 KDTree 贪心覆盖，等价 kdtree(info_f, **kwargs)，返回 (train_neighbors, test_neighbors, val_idx)
    - binning: 需要 raw_info + regular_info 两套数据，请直接调用 binning(...)，返回 (target, mask, report)，不经由此工厂

    未实现: 5d_windows, 5d_cosine
    """
    if mode == 'csg':
        return gather(info_f, 'csg')
    if mode == 'crg':
        return gather(info_f, 'crg')
    if mode == 'kdtree':
        return kdtree(info_f, **kwargs)
    if mode == '5d_windows':
        raise NotImplementedError("5d_windows 尚未在 generate_dataset_bak 中实现")
    if mode == '5d_cosine':
        raise NotImplementedError("5d_cosine 尚未在 generate_dataset_bak 中实现")
    raise ValueError(f"未知的划分方法: {mode}")


# ── precompute utilities ────────────────────────────────────────────


def read_coord4(group: Any) -> np.ndarray:
    """Read (sx, sy, rx, ry) from H5 group as [N, 4] float32."""
    missing = [k for k in ("sx", "sy", "rx", "ry") if k not in group]
    if missing:
        raise KeyError(f"H5 group is missing coordinate datasets: {missing}")
    return np.column_stack([group[k][:] for k in ("sx", "sy", "rx", "ry")]).astype(np.float32)


def read_trace_data(group: Any) -> np.ndarray:
    """Read 'data' from H5 group as [N, T] float32."""
    if "data" not in group:
        raise KeyError("H5 group is missing dataset 'data'")
    data = group["data"][:]
    if data.ndim != 2:
        raise ValueError(f"expected trace data [N, T], got shape {data.shape}")
    return data.astype(np.float32, copy=False)


def read_regular_mask(group: Any, mask_key: str, n_grid: int,
                      target_h5: str = None, group_key: str = None) -> np.ndarray:
    """Read and validate a regular-grid boolean mask.

    If *mask_key* is not present in *group*, falls back to computing the mask
    from *target_h5* data: traces that are all-near-zero → ``False``, otherwise
    ``True``.
    """
    if mask_key in group:
        mask = np.asarray(group[mask_key][:]).reshape(-1).astype(bool)
    elif target_h5 and group_key:
        print(f"regular H5 has no mask key {mask_key!r}, computing from target_h5 data")
        with File(target_h5, "r") as f_target:
            data = f_target[group_key]["data"][:]
        mask = ~np.all(np.abs(data) < 1e-10, axis=1)
        print(f"target_h5 mask: true={int(mask.sum())} false={int((~mask).sum())}")
    else:
        raise KeyError(f"regular H5 group has no mask dataset {mask_key!r}")
    if mask.size != n_grid:
        raise ValueError(
            f"regular mask length {mask.size} does not match regular trace count {n_grid}"
        )
    return mask


def resolve_block_tuple(
    explicit: Optional[Sequence[int]],
    divisors: Sequence[int],
    dims: Sequence[int],
    name: str,
) -> Tuple[int, int, int, int]:
    """Resolve 4D block size or stride from explicit values or dims/divisors."""
    if explicit is not None:
        values = tuple(int(x) for x in explicit)
        if len(values) != 4:
            raise ValueError(f"{name} must have four integers")
        if min(values) <= 0:
            raise ValueError(f"{name} values must be positive")
        return values  # type: ignore[return-value]
    div = tuple(int(x) for x in divisors)
    if len(div) != 4 or min(div) <= 0:
        raise ValueError(f"{name} divisors must be four positive integers")
    return tuple(max(1, int(d) // int(v)) for d, v in zip(dims, div))  # type: ignore[return-value]


def build_grid_index_map_4d_from_coord_grid(
    coord_grid: np.ndarray,
    *,
    on_collision: str = "raise",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build sparse 4D logical map from grid coordinates.

    ``index_map[isx, isy, irx, iry]`` stores the row id in ``coord_grid``.
    Empty logical cells are ``-1``.
    """
    grid = np.asarray(coord_grid, dtype=np.float64)
    if grid.ndim != 2 or grid.shape[1] != 4:
        raise ValueError("coord_grid must have shape [N, 4] with columns [sx, sy, rx, ry]")

    n = int(grid.shape[0])
    sx_levels, inv_sx = np.unique(grid[:, 0], return_inverse=True)
    sy_levels, inv_sy = np.unique(grid[:, 1], return_inverse=True)
    rx_levels, inv_rx = np.unique(grid[:, 2], return_inverse=True)
    ry_levels, inv_ry = np.unique(grid[:, 3], return_inverse=True)
    dims = (len(sx_levels), len(sy_levels), len(rx_levels), len(ry_levels))

    linear = np.ravel_multi_index((inv_sx, inv_sy, inv_rx, inv_ry), dims)
    index_map = np.full(dims, -1, dtype=np.int64)
    if np.unique(linear).size != n:
        if on_collision == "raise":
            raise ValueError(
                "multiple coord_grid rows map to the same 4D logical cell; "
                "use --on-grid-collision last to keep the last row"
            )
        if on_collision != "last":
            raise ValueError("on_collision must be 'raise' or 'last'")
        for i in range(n):
            index_map[inv_sx[i], inv_sy[i], inv_rx[i], inv_ry[i]] = np.int64(i)
    else:
        index_map[inv_sx, inv_sy, inv_rx, inv_ry] = np.arange(n, dtype=np.int64)

    info = {
        "sx_levels": sx_levels,
        "sy_levels": sy_levels,
        "rx_levels": rx_levels,
        "ry_levels": ry_levels,
        "nsx": dims[0],
        "nsy": dims[1],
        "nrx": dims[2],
        "nry": dims[3],
    }
    return index_map, info


def make_query_mask(
    *,
    mode: str,
    regular_mask: Optional[np.ndarray],
    n_grid: int,
) -> Optional[np.ndarray]:
    """Build a query mask for inference from a regular-grid mask."""
    if mode == "none":
        return None
    if mode == "all":
        return np.ones((n_grid,), dtype=bool)
    if regular_mask is None:
        raise ValueError(f"query mask mode {mode!r} requires a regular mask")
    if mode == "regular_true":
        return regular_mask.astype(bool, copy=False)
    if mode == "regular_false":
        return ~regular_mask.astype(bool, copy=False)
    raise ValueError(f"unknown query mask mode: {mode}")


def save_norm_stats(path: Path, stats: Dict[str, Dict[str, np.ndarray]]) -> None:
    flat = {
        "obs_min": stats["obs"]["min"],
        "obs_max": stats["obs"]["max"],
        "obs_mean": stats["obs"]["mean"],
        "obs_std": stats["obs"]["std"],
        "grid_min": stats["grid"]["min"],
        "grid_max": stats["grid"]["max"],
        "grid_mean": stats["grid"]["mean"],
        "grid_std": stats["grid"]["std"],
    }
    np.savez(path, **flat)


def object_array(rows: Iterable[np.ndarray]) -> np.ndarray:
    """Pack a list of 1D arrays into an object-dtype ndarray."""
    rows_list = [np.asarray(row, dtype=np.int64).reshape(-1) for row in rows]
    out = np.empty((len(rows_list),), dtype=object)
    for i, row in enumerate(rows_list):
        out[i] = row
    return out


def index_row(storage: np.ndarray, idx: int) -> np.ndarray:
    """Extract valid (>=0) indices from an object-array row."""
    row = np.asarray(storage[idx], dtype=np.int64).reshape(-1)
    return row[row >= 0]


def summarize_query_context(
    query_list: Sequence[np.ndarray],
    context_list: Sequence[np.ndarray],
) -> np.ndarray:
    """Return per-sample input-missing ratios (1 - context/total)."""
    ratios = np.zeros((len(query_list),), dtype=np.float32)
    for i, (q_idx, c_idx) in enumerate(zip(query_list, context_list)):
        q = int(np.asarray(q_idx).size)
        c = int(np.asarray(c_idx).size)
        total = q + c
        ratios[i] = 0.0 if total <= 0 else 1.0 - (c / total)
    return ratios


def validate_outputs(
    *,
    patch_dir: Path,
    n_obs: int,
    n_grid: int,
    check_train: bool,
    check_infer: bool,
) -> Dict[str, Any]:
    """Lightweight validation of precomputed train/infer npz files."""
    summary: Dict[str, Any] = {}

    if check_train:
        train_path = patch_dir / "train_pool_idx_2d.npz"
        with np.load(train_path, allow_pickle=True) as train_npz:
            pool_idx_2d = np.asarray(train_npz["pool_idx_2d"], dtype=np.int64)
            anchor_idx = np.asarray(train_npz["anchor_idx"], dtype=np.int64)
        if pool_idx_2d.ndim != 2:
            raise ValueError(f"train pool_idx_2d must be 2D, got {pool_idx_2d.shape}")
        if anchor_idx.shape[0] != pool_idx_2d.shape[0]:
            raise ValueError("train anchor_idx length does not match pool rows")
        valid_pool = pool_idx_2d[pool_idx_2d >= 0]
        if valid_pool.size and (valid_pool.min() < 0 or valid_pool.max() >= n_obs):
            raise ValueError("train pool indices are out of raw observation range")
        sample = index_row(pool_idx_2d, 0)
        if sample.size < 2:
            raise ValueError("first train pool has fewer than two traces")
        summary["train_samples"] = int(pool_idx_2d.shape[0])
        summary["train_pool_width"] = int(pool_idx_2d.shape[1])

    if check_infer:
        infer_path = patch_dir / "infer_query_context.npz"
        with np.load(infer_path, allow_pickle=True) as infer_npz:
            q_list = infer_npz["grid_query_idx_list"]
            c_list = infer_npz["context_idx_list"]
            block_id = infer_npz["block_id"]
            center_idx = infer_npz["block_center_grid_idx"]
            has_anchor = "anchor_grid_idx_list" in infer_npz.files
            anchor_list = infer_npz["anchor_grid_idx_list"] if has_anchor else None

        if len(q_list) != len(c_list):
            raise ValueError("infer query/context list lengths differ")
        if len(q_list) == 0:
            raise ValueError("infer_query_context has no samples")
        if block_id.shape[0] != len(q_list) or center_idx.shape[0] != len(q_list):
            raise ValueError("infer block metadata length does not match sample count")
        if anchor_list is not None and len(anchor_list) != len(q_list):
            raise ValueError("infer anchor list length does not match sample count")

        check_ids = list(range(min(5, len(q_list))))
        if len(q_list) > 5:
            check_ids.append(len(q_list) - 1)
        for i in check_ids:
            q = index_row(q_list, i)
            c = index_row(c_list, i)
            if q.size == 0 or c.size == 0:
                raise ValueError(f"infer sample {i} has empty query or context")
            if q.min() < 0 or q.max() >= n_grid:
                raise ValueError(f"infer sample {i} query indices out of regular-grid range")
            if c.min() < 0 or c.max() >= n_obs:
                raise ValueError(f"infer sample {i} context indices out of raw-observation range")

        summary["infer_samples"] = int(len(q_list))
        summary["infer_checked_rows"] = [int(x) for x in check_ids]

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="reg_tool core utilities")
    parser.add_argument(
        "mode",
        nargs="?",
        default="anchor_patch",
        choices=["anchor_patch", "binning", "binning+csg", "binning+crg", "kdtree", "csg", "crg"],
        help="run mode (use binning+csg or binning+crg to chain binning then gather)",
    )
    # ---- Paths ----
    parser.add_argument("--base_dir", type=str, default="/NAS/czt/mount/seis_flow_data12V2/h5/dongfang/")
    parser.add_argument("--raw_h5", type=str, default=None)
    parser.add_argument("--regular_h5", type=str, default=None)
    parser.add_argument("--target_h5", type=str, default=None)
    parser.add_argument("--group_key", type=str, default="1551")
    parser.add_argument("--patch-dir", type=str, default=None, help="override patch output dir")
    parser.add_argument("--regular-mask-key", type=str, default="mask", help="key for regular mask in H5")

    # ---- Train hyperparams (defaults aligned with run_precompute.sh) ----
    parser.add_argument("--num_anchors", type=int, default=None,
                        help="auto as N_obs // anchor_stride if None")
    parser.add_argument("--anchor-stride", type=int, default=128)
    parser.add_argument("--k_patch", type=int, default=256)
    parser.add_argument("--top_l", type=int, default=None, help="auto as k_patch + 128 if None")
    parser.add_argument("--num_query", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metric_weights", type=str, default="1,1,0.5,0.5")

    # ---- Anchor selector ----
    parser.add_argument(
        "--train-anchor-selector",
        choices=["farthest_point_sampling", "facility_location_anchor_sampling", "value_based_anchor_sampling"],
        default="value_based_anchor_sampling",
    )
    parser.add_argument("--train-trusted-source", choices=["all", "regular_mask"], default="all")
    parser.add_argument("--trusted_mask_key", type=str, default=None,
                        help="raw H5 mask key for trusted obs (overrides trust derived from regular_mask)")

    # ---- Value-based anchor sampling params ----
    parser.add_argument("--value-local-top-l", type=int, default=None, help="auto as top_l if None")
    parser.add_argument("--value-suppression", choices=["subtractive", "multiplicative"], default="subtractive")
    parser.add_argument("--value-suppression-lambda", type=float, default=1.0)
    parser.add_argument("--value-score-tol", type=float, default=0.0)
    parser.add_argument("--value-knn-gpu-batch-rows", type=int, default=512)
    parser.add_argument("--value-knn-gpu-device", type=str, default="cuda:0")
    parser.add_argument("--value-knn-full-matrix-max-n", type=int, default=4096)
    parser.add_argument("--train-knn-use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-suppression-use-gpu", action=argparse.BooleanOptionalAction, default=True)

    # ---- Inference (4D block) ----
    parser.add_argument("--block-size", type=int, nargs=4, default=None)
    parser.add_argument("--stride", type=int, nargs=4, default=None)
    parser.add_argument("--block-divisors", type=int, nargs=4, default=(6, 21, 7, 5))
    parser.add_argument("--stride-divisors", type=int, nargs=4, default=(6, 21, 7, 5))
    parser.add_argument("--on-grid-collision", choices=["raise", "last"], default="raise")
    parser.add_argument("--query-mask-mode", choices=["regular_true", "regular_false", "all", "none"],
                        default="regular_true")
    parser.add_argument("--infer-obs-valid-source", choices=["none", "regular_mask"], default="none")
    parser.add_argument("--infer-top-l", type=int, default=None, help="auto as k_patch * 2 if None")
    parser.add_argument("--max-query-per-patch", type=int, default=128)
    parser.add_argument("--gpu-query-chunk-size", type=int, default=128)
    parser.add_argument("--infer-gpu-device", type=str, default="cuda:0")
    parser.add_argument("--infer-use-gpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-full-query-coverage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--greedy-fill-uncovered", action=argparse.BooleanOptionalAction, default=True)

    # ---- Misc ----
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-infer", action="store_true")
    parser.add_argument("--save-legacy-anchor-files", action="store_true")
    parser.add_argument("--save-grid-index-map", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--summary-json", type=str, default=None)
    parser.add_argument(
        "--raw_key_aggregate",
        type=str,
        default="mean",
        choices=["none", "mean"],
        help="anchor_patch模式下是否先按键聚合raw观测道",
    )
    parser.add_argument(
        "--enable-auto-params",
        action="store_true",
        help="根据观测系统自动计算num_anchors/k_patch/top_l/block_divisors等超参数",
    )
    parser.add_argument("--auto-params-anchor-stride", type=int, default=128)

    args = parser.parse_args()

    resolved_modes = [m.strip() for m in args.mode.split("+")]

    base_dir = args.base_dir
    info_h5_raw = args.raw_h5 or os.path.join(base_dir, "raw5d_data1104.h5")
    info_h5_regular = args.regular_h5 or os.path.join(base_dir, "reg5dbin_label1031.h5")

    # patch dir and target_h5 at the same level as input H5 files
    h5_dirs = set(os.path.dirname(os.path.abspath(p)) for p in (info_h5_raw, info_h5_regular))
    h5_root = sorted(h5_dirs)[0]
    info_h5_target = args.target_h5 or os.path.join(h5_root, "targeth5_binning.h5")
    actual_mode = resolved_modes[0]
    patch_dir = args.patch_dir if args.patch_dir else os.path.join(h5_root, f"patch_{actual_mode}")
    os.makedirs(patch_dir, exist_ok=True)

    metric_weights = [float(v) for v in args.metric_weights.split(",")]
    if len(metric_weights) != 4:
        raise ValueError("--metric_weights must contain 4 comma-separated values")

    def cover_dict2npy(gather_dict):
        return np.concatenate([np.array(gather_dict[key]) for key in gather_dict.keys()]), np.array(list(gather_dict.keys()))

    def _auto_compute_params(raw_group, reg_group):
        """Auto-compute anchor params from observation system."""
        sx_r = raw_group["sx"][:].astype(np.float32)
        sy_r = raw_group["sy"][:].astype(np.float32)
        rx_r = raw_group["rx"][:].astype(np.float32)
        ry_r = raw_group["ry"][:].astype(np.float32)
        sx_g = reg_group["sx"][:].astype(np.float32)
        sy_g = reg_group["sy"][:].astype(np.float32)
        rx_g = reg_group["rx"][:].astype(np.float32)
        ry_g = reg_group["ry"][:].astype(np.float32)

        n_obs = sx_r.shape[0]
        n_grid = sx_g.shape[0]

        num_anchors = max(1, n_obs // args.auto_params_anchor_stride)

        def _grid_step(arr):
            u = np.sort(np.unique(arr))
            if u.size < 2:
                return None, u
            d = np.diff(u)
            d = d[d > 1e-9]
            return float(np.median(d)) if d.size > 0 else None, u

        _, sx_u = _grid_step(sx_g)
        _, sy_u = _grid_step(sy_g)
        _, rx_u = _grid_step(rx_g)
        _, ry_u = _grid_step(ry_g)
        nsx, nsy, nrx, nry = len(sx_u), len(sy_u), len(rx_u), len(ry_u)
        dims_4d = (nsx, nsy, nrx, nry)

        avg_density = n_grid / max(1, nsx * nsy * nrx * nry)
        coverage = n_obs / n_grid if n_grid > 0 else 1.0

        k_patch = int(np.clip(32 + 192 * coverage, 32, 512))
        top_l = 2 * k_patch
        num_query = max(1, min(k_patch // 4, 32))

        target_cells = max(256, int(400 / max(avg_density, 0.01)))
        vol_per_dim = target_cells ** (1.0 / 4.0)
        block_divs = tuple(max(1, int(round(d / max(1.0, vol_per_dim)))) for d in dims_4d)
        stride_divs = block_divs  # 50% overlap
        block_sz = tuple(max(1, d // b) for d, b in zip(dims_4d, block_divs))

        # metric_weights from coordinate ranges
        r_sx = max(sx_u.max() - sx_u.min(), 1e-6)
        r_sy = max(sy_u.max() - sy_u.min(), 1e-6)
        r_rx = max(rx_u.max() - rx_u.min(), 1e-6)
        r_ry = max(ry_u.max() - ry_u.min(), 1e-6)
        w = [1.0, r_sx / r_sy, r_sx / r_rx * 0.5, r_sx / r_ry * 0.5]
        w_sum = sum(w)
        metric_w = tuple(round(v * 4.0 / w_sum, 3) for v in w)

        print("[auto_params] computed from observation system:")
        print(f"  grid dims: {nsx}x{nsy}x{nrx}x{nry}  N_obs={n_obs}  N_grid={n_grid}")
        print(f"  num_anchors={num_anchors} (anchor_stride={args.auto_params_anchor_stride})")
        print(f"  k_patch={k_patch}  top_l={top_l}  num_query={num_query}")
        print(f"  block_divisors={list(block_divs)}  block_size={list(block_sz)}")
        print(f"  stride_divisors={list(stride_divs)}")
        print(f"  metric_weights={metric_w}")

        return {
            "num_anchors": num_anchors,
            "k_patch": k_patch,
            "top_l": top_l,
            "num_query": num_query,
            "metric_weights": list(metric_w),
            "block_divisors": list(block_divs),
            "stride_divisors": list(stride_divs),
        }

    # ── Run binning if needed (always first in chain) ──
    ran_binning = False
    if "binning" in resolved_modes:
        with File(info_h5_raw, "r") as f_raw, File(info_h5_regular, "r+") as f_reg:
            info_f_raw = f_raw[args.group_key]
            info_f_regular = f_reg[args.group_key]
            print("raw keys:", list(info_f_raw.keys()))
            print("regular keys:", list(info_f_regular.keys()))
            target, mask, report = binning(info_f_raw, info_f_regular)
            info_f_regular["mask"] = mask
            print("缺失率：", (1 - mask.sum() / len(mask)))
            print("分箱报告:", report)
            saveh5(target, info_f_regular, info_h5_target, args.group_key)
            with File(info_h5_target, "r") as f_target:
                print("target keys:", list(f_target[args.group_key].keys()))
        ran_binning = True
        resolved_modes = [m for m in resolved_modes if m != "binning"]

    # ── Execute remaining mode(s) ──
    if not resolved_modes:
        sys.exit(0)

    #actual_mode = resolved_modes[0]
    if actual_mode not in ("anchor_patch", "kdtree", "csg", "crg"):
        raise ValueError(f"unknown post-binning mode: {actual_mode}")

    if ran_binning and actual_mode in ("csg", "crg"):
        _use_regular = info_h5_target
    else:
        _use_regular = info_h5_regular

    if actual_mode == "anchor_patch":
        with File(info_h5_raw, "r") as f_raw, File(_use_regular, "r+") as f_reg:
            info_f_raw = f_raw[args.group_key]
            info_f_regular = f_reg[args.group_key]
            print("raw keys:", list(info_f_raw.keys()))
            print("regular keys:", list(info_f_regular.keys()))

            trace_obs_raw = _read_array(info_f_raw, "data").astype(np.float32)
            coord_obs_raw = np.column_stack(
                (
                    _read_array(info_f_raw, "sx"),
                    _read_array(info_f_raw, "sy"),
                    _read_array(info_f_raw, "rx"),
                    _read_array(info_f_raw, "ry"),
                )
            ).astype(np.float32)
            raw_keys = generate_binning_keys(info_f_raw).astype(np.int64)
            if trace_obs_raw.shape[0] != coord_obs_raw.shape[0] or trace_obs_raw.shape[0] != raw_keys.shape[0]:
                raise ValueError("raw data/coord/keys length mismatch")

            # ── optional key-mean aggregation ──
            if args.raw_key_aggregate == "mean":
                trace_obs, coord_obs, obs_keys, key_counts = _aggregate_raw_by_keys_mean(
                    trace_obs=trace_obs_raw,
                    coord_obs=coord_obs_raw,
                    raw_keys=raw_keys,
                )
                np.save(os.path.join(patch_dir, "raw_key_counts.npy"), key_counts)
                print(
                    "raw key-mean aggregation:",
                    f"before={trace_obs_raw.shape[0]} after={trace_obs.shape[0]}",
                )
            else:
                trace_obs = trace_obs_raw
                coord_obs = coord_obs_raw
                obs_keys = raw_keys

            coord_grid = np.column_stack(
                (
                    _read_array(info_f_regular, "sx"),
                    _read_array(info_f_regular, "sy"),
                    _read_array(info_f_regular, "rx"),
                    _read_array(info_f_regular, "ry"),
                )
            ).astype(np.float32)
            reg_keys = generate_binning_keys(info_f_regular).astype(np.int64)

            if reg_keys.shape[0] != coord_grid.shape[0]:
                raise ValueError("regular coordinates and regular binning keys must have the same length")

            # ── regular mask ──
            regular_mask = read_regular_mask(
                info_f_regular, args.regular_mask_key, coord_grid.shape[0],
                target_h5=info_h5_target, group_key=args.group_key,
            )
            print("regular mask true count:", int(regular_mask.sum()))

            # ── raw_obs_valid from regular mask ──
            raw_obs_valid = raw_obs_valid_mask_from_regular_trusted_mask(
                raw_binning_keys=obs_keys,
                reg_binning_keys=reg_keys,
                regular_trusted_mask=regular_mask,
            )
            trusted_from_mask = np.flatnonzero(raw_obs_valid).astype(np.int64)
            np.save(os.path.join(patch_dir, "raw_obs_valid_mask.npy"), raw_obs_valid)
            
            # ── coordinate normalization ──
            coord_obs_norm, coord_grid_norm, norm_stats = normalize_coords(coord_obs, coord_grid)
            save_norm_stats(Path(patch_dir) / "coord_norm_stats.npz", norm_stats)
            np.save(os.path.join(patch_dir, "coord_obs_norm.npy"), coord_obs_norm)
            np.save(os.path.join(patch_dir, "coord_grid_norm.npy"), coord_grid_norm)
            print("coord_obs_norm range:", float(coord_obs_norm.min()), float(coord_obs_norm.max()))
            print("coord_grid_norm range:", float(coord_grid_norm.min()), float(coord_grid_norm.max()))

            # ── 4D grid index map ──
            grid_index_map_4d, grid_info = build_grid_index_map_4d_from_coord_grid(
                coord_grid_norm,
                on_collision=args.on_grid_collision,
            )
            dims_4d = (grid_info["nsx"], grid_info["nsy"], grid_info["nrx"], grid_info["nry"])
            valid_grid_cells = int(np.count_nonzero(grid_index_map_4d >= 0))
            if valid_grid_cells != coord_grid.shape[0] and args.on_grid_collision == "raise":
                raise ValueError("grid index map valid cell count does not match coord_grid rows")
            if args.save_grid_index_map:
                np.save(os.path.join(patch_dir, "grid_index_map_4d.npy"), grid_index_map_4d)
                np.savez(
                    os.path.join(patch_dir, "grid_index_map_4d_levels.npz"),
                    sx_levels=grid_info["sx_levels"],
                    sy_levels=grid_info["sy_levels"],
                    rx_levels=grid_info["rx_levels"],
                    ry_levels=grid_info["ry_levels"],
                )
            print("grid_index_map_4d shape:", grid_index_map_4d.shape)
            print("grid_index_map_4d valid cells:", valid_grid_cells)

            # ── resolve hyperparameters ──
            mw = metric_weights
            k_patch = int(args.k_patch)
            top_l = int(args.top_l) if args.top_l is not None else k_patch + 128
            infer_top_l = int(args.infer_top_l) if args.infer_top_l is not None else k_patch * 2
            num_anchors = (
                int(args.num_anchors)
                if args.num_anchors is not None
                else max(1, int(trace_obs.shape[0]) // int(args.anchor_stride))
            )
            num_query = args.num_query
            value_local_top_l = args.value_local_top_l
            if value_local_top_l is None:
                value_local_top_l = top_l

            if args.enable_auto_params:
                ap = _auto_compute_params(info_f_raw, info_f_regular)
                num_anchors = ap["num_anchors"]
                k_patch = ap["k_patch"]
                top_l = ap["top_l"]
                num_query = ap["num_query"]
                mw = ap["metric_weights"]
                infer_top_l = k_patch * 2
                value_local_top_l = top_l
                args.block_divisors = tuple(ap["block_divisors"])
                args.stride_divisors = tuple(ap["stride_divisors"])

            # ── summary dict ──
            summary: Dict[str, Any] = {
                "base_dir": str(base_dir),
                "raw_h5": str(info_h5_raw),
                "regular_h5": str(info_h5_regular),
                "patch_dir": str(patch_dir),
                "group_key": args.group_key,
                "n_obs": int(coord_obs.shape[0]),
                "n_grid": int(coord_grid.shape[0]),
                "grid_shape_4d": [int(x) for x in dims_4d],
                "k_patch": k_patch,
                "top_l": top_l,
                "infer_top_l": infer_top_l,
                "num_anchors": num_anchors,
                "metric_weights": mw,
            }

        # ── Training ──
        if not args.skip_train:
            if args.trusted_mask_key is not None and args.trusted_mask_key in info_f_raw:
                trusted_mask_arr = _read_array(info_f_raw, args.trusted_mask_key).astype(bool)
                trusted_idx = np.flatnonzero(trusted_mask_arr).astype(np.int64)
                print(
                    f"trusted_idx from raw mask key={args.trusted_mask_key}, "
                    f"count={trusted_idx.size}"
                )
            elif args.train_trusted_source == "all":
                trusted_idx = np.arange(coord_obs.shape[0], dtype=np.int64)
                print(f"train trusted_source=all, count={trusted_idx.size}")
            else:
                pass
            if trusted_idx.size == 0:
                raise ValueError("no trusted training observations are available")

            print("building train patches")
            print("train anchor selector:", args.train_anchor_selector)
            print("train num_anchors:", num_anchors, "k_patch:", k_patch, "top_l:", top_l)
            train_pack = precompute_train_patches_2d(
                coord_obs_norm=coord_obs_norm,
                trace_obs=trace_obs,
                trusted_idx=trusted_idx,
                num_anchors=num_anchors,
                k_patch=k_patch,
                top_l=top_l,
                metric_weights=mw,
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
                num_query=num_query,
                seed=args.seed,
                pool_size=args.pool_size,
            )
            np.savez(
                os.path.join(patch_dir, "train_pool_idx_2d.npz"),
                pool_idx_2d=train_pack["patch_idx_2d"],
                anchor_idx=train_pack["anchor_idx"],
            )
            if args.save_legacy_anchor_files:
                np.savez(os.path.join(patch_dir, "anchor_train_patch_idx_2d.npz"), **{"0": train_pack["patch_idx_2d"]})
                np.savez(os.path.join(patch_dir, "anchor_train_context_idx_2d.npz"), **{"0": train_pack["context_idx_2d"]})
                np.savez(os.path.join(patch_dir, "anchor_train_query_idx_2d.npz"), **{"0": train_pack["query_idx_2d"]})
                np.save(os.path.join(patch_dir, "anchor_train_anchor_idx.npy"), train_pack["anchor_idx"])
                np.save(os.path.join(patch_dir, "anchor_train_anchor_coord.npy"), train_pack["anchor_coord"])
            print("train_pool_idx_2d:", train_pack["patch_idx_2d"].shape)
            summary["train_pool_shape"] = [int(x) for x in train_pack["patch_idx_2d"].shape]

        # ── Inference ──
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

            print("building infer patches (4D)")
            print("block_size:", block_size, "stride:", stride)
            print("query_mask_mode:", args.query_mask_mode)
            if query_mask is not None:
                print("query targets:", int(query_mask.sum()))
            print("infer k_patch:", k_patch, "infer_top_l:", infer_top_l)
            infer_pack = precompute_infer_patches_4d(
                coord_obs_norm=coord_obs_norm,
                coord_grid_norm=coord_grid_norm,
                grid_shape_4d=None,
                block_size=block_size,
                stride=stride,
                k_patch=k_patch,
                top_l=infer_top_l,
                metric_weights=mw,
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
                os.path.join(patch_dir, "infer_query_context.npz"),
                grid_query_idx_list=object_array(query_list),
                context_idx_list=object_array(context_list),
                block_id=infer_pack["block_id"],
                block_center_grid_idx=infer_pack["block_center_grid_idx"],
                anchor_grid_idx_list=object_array(infer_pack["anchor_grid_idx_list"]),
            )
            np.savez(
                os.path.join(patch_dir, "infer_query_context_stats.npz"),
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

        # ── validation + summary ──
        validation = validate_outputs(
            patch_dir=Path(patch_dir),
            n_obs=coord_obs.shape[0],
            n_grid=coord_grid.shape[0],
            check_train=not args.skip_train,
            check_infer=not args.skip_infer,
        )
        summary["validation"] = validation
        print("validation:", validation)


    elif actual_mode in ("kdtree", "csg", "crg"):
        with File(info_h5_raw, "r") as f_raw, File(_use_regular, "r") as f_reg:
            info_f_raw = f_raw[args.group_key]
            info_f_regular = f_reg[args.group_key]
            print("raw keys:", list(info_f_raw.keys()))
            print("regular keys:", list(info_f_regular.keys()))

            if actual_mode == "kdtree":
                train_neighbors, test_neighbors, val_idx = kdtree(info_f_regular)
                print("kdtree train/test/val:", train_neighbors.shape, test_neighbors.shape, val_idx.shape)
                np.savez(os.path.join(patch_dir, "train_pool_idx_2d.npz"),
                         pool_idx_2d=train_neighbors)

            elif actual_mode == "csg":
                csg_reg = gather(info_f_regular, "csg")
                csg_raw = gather(info_f_raw, "csg")
                print("shot_num:", len(csg_reg.keys()))
                # Save gathers as object arrays (one per gather)
                reg_gathers = np.empty(len(csg_reg), dtype=object)
                reg_keys = np.empty(len(csg_reg), dtype=object)
                for i, (k, v) in enumerate(sorted(csg_reg.items())):
                    reg_gathers[i] = np.asarray(v, dtype=np.int64)
                    reg_keys[i] = np.asarray(k, dtype=np.int64)
                np.savez(os.path.join(patch_dir, "csg_train_pool.npz"),
                         pool_idx=reg_gathers, pool_key=reg_keys)
                raw_gathers = np.empty(len(csg_raw), dtype=object)
                for i, (k, v) in enumerate(sorted(csg_raw.items())):
                    raw_gathers[i] = np.asarray(v, dtype=np.int64)
                np.savez(os.path.join(patch_dir, "csg_raw_pool.npz"), pool_idx=raw_gathers)
                print(f"saved csg_train_pool.npz: {len(csg_reg)} gathers")

            elif actual_mode == "crg":
                crg_reg = gather(info_f_regular, "crg")
                crg_raw = gather(info_f_raw, "crg")
                print("recv_num:", len(crg_reg.keys()))
                reg_gathers = np.empty(len(crg_reg), dtype=object)
                reg_keys = np.empty(len(crg_reg), dtype=object)
                for i, (k, v) in enumerate(sorted(crg_reg.items())):
                    reg_gathers[i] = np.asarray(v, dtype=np.int64)
                    reg_keys[i] = np.asarray(k, dtype=np.int64)
                np.savez(os.path.join(patch_dir, "crg_train_pool.npz"),
                         pool_idx=reg_gathers, pool_key=reg_keys)
                raw_gathers = np.empty(len(crg_raw), dtype=object)
                for i, (k, v) in enumerate(sorted(crg_raw.items())):
                    raw_gathers[i] = np.asarray(v, dtype=np.int64)
                np.savez(os.path.join(patch_dir, "crg_raw_pool.npz"), pool_idx=raw_gathers)
                print(f"saved crg_train_pool.npz: {len(crg_reg)} gathers")

            # Build infer_query_context: mask computed from target_h5 data
            with File(info_h5_target, "r") as f_tgt:
                data_arr = f_tgt[args.group_key]["data"][:].astype(np.float32)
            mask_arr = ~np.all(np.abs(data_arr) < 1e-10, axis=1)
            print(f"target_h5 mask: true={int(mask_arr.sum())} false={int((~mask_arr).sum())}")
            if mask_arr.size > 0:
                if actual_mode == "csg":
                    query_list = []
                    context_list = []
                    skipped = 0
                    max_q = int(args.num_query) if args.num_query else 48
                    for k in sorted(csg_reg.keys()):
                        reg_idxs = csg_reg[k]
                        raw_idxs = csg_raw[k]
                        mask_g = mask_arr[reg_idxs]
                        missing = reg_idxs[~mask_g]
                        observed = raw_idxs[mask_g]
                        if missing.size == 0 or observed.size == 0:
                            skipped += 1
                            continue
                        # chunk query into groups of max_q;
                        # all chunks share the same observed raw-H5 positions
                        for start in range(0, missing.size, max_q):
                            chunk = missing[start:start + max_q]
                            query_list.append(chunk.astype(np.int64))
                            context_list.append(observed.astype(np.int64))
                    n_infer = len(query_list)
                    if n_infer == 0:
                        raise RuntimeError(
                            "csg: no gather with both missing and observed traces; "
                            f"all {skipped} gathers are fully observed or empty"
                        )
                    n_gathers = len(csg_reg) - skipped
                    print(
                        f"csg infer: {n_infer} chunks from {n_gathers} gathers "
                        f"(skipped {skipped}), max_query_per_chunk={max_q}"
                    )
                    np.savez(
                        os.path.join(patch_dir, "infer_query_context.npz"),
                        grid_query_idx_list=object_array(query_list),
                        context_idx_list=object_array(context_list),
                        block_id=np.arange(n_infer, dtype=np.int64),
                        block_center_grid_idx=np.full(n_infer, -1, dtype=np.int64),
                        anchor_grid_idx_list=object_array(query_list),
                    )

                elif actual_mode == "crg":
                    query_list = []
                    context_list = []
                    skipped = 0
                    max_q = 48
                    for k in sorted(crg_reg.keys()):
                        reg_idxs = crg_reg[k]
                        raw_idxs = crg_raw[k]
                        mask_g = mask_arr[reg_idxs]
                        missing = reg_idxs[~mask_g]
                        observed = raw_idxs[mask_g]
                        if missing.size == 0 or observed.size == 0:
                            skipped += 1
                            continue
                        for start in range(0, missing.size, max_q):
                            chunk = missing[start:start + max_q]
                            query_list.append(chunk.astype(np.int64))
                            context_list.append(observed.astype(np.int64))
                    n_infer = len(query_list)
                    if n_infer == 0:
                        raise RuntimeError(
                            "crg: no gather with both missing and observed traces; "
                            f"all {skipped} gathers are fully observed or empty"
                        )
                    n_gathers = len(crg_reg) - skipped
                    print(
                        f"crg infer: {n_infer} chunks from {n_gathers} gathers "
                        f"(skipped {skipped}), max_query_per_chunk={max_q}"
                    )
                    np.savez(
                        os.path.join(patch_dir, "infer_query_context.npz"),
                        grid_query_idx_list=object_array(query_list),
                        context_idx_list=object_array(context_list),
                        block_id=np.arange(n_infer, dtype=np.int64),
                        block_center_grid_idx=np.full(n_infer, -1, dtype=np.int64),
                        anchor_grid_idx_list=object_array(query_list),
                    )

                else:  # kdtree
                    missing_idx = np.flatnonzero(~mask_arr).astype(np.int64)
                    obs_idx = np.flatnonzero(mask_arr).astype(np.int64)
                    print(f"kdtree infer: missing={missing_idx.size}  observed={obs_idx.size}")
                    np.savez(
                        os.path.join(patch_dir, "infer_query_context.npz"),
                        grid_query_idx_list=np.asarray([missing_idx], dtype=object),
                        context_idx_list=np.asarray([obs_idx], dtype=object),
                        block_id=np.array([0], dtype=np.int64),
                        block_center_grid_idx=np.array([-1], dtype=np.int64),
                        anchor_grid_idx_list=np.asarray([missing_idx], dtype=object),
                    )

                print("saved infer_query_context.npz")
