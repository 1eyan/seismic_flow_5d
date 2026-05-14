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
        farthest_point_sampling,
        top_l_neighbors,
        diverse_topk,
        build_train_patch,
        make_grid_blocks,
        make_grid_blocks_from_shape,
        make_grid_blocks_from_shape_4d,
        make_grid_blocks_from_index_map_4d,
        ravel_grid_index_4d,
        build_infer_patch,
        accumulate_block_predictions,
        finalize_predictions,
        find_uncovered_points,
        fallback_infer_for_uncovered,
        precompute_train_patches_2d,
        precompute_infer_patches_2d,
        precompute_infer_patches_4d,
        precompute_infer_patches_4d_block_center,
    )
except ImportError:
    # Allow direct script usage from reg_tool directory.
    from patch_sampler import (
        normalize_coords,
        farthest_point_sampling,
        top_l_neighbors,
        diverse_topk,
        build_train_patch,
        make_grid_blocks,
        make_grid_blocks_from_shape,
        make_grid_blocks_from_shape_4d,
        make_grid_blocks_from_index_map_4d,
        ravel_grid_index_4d,
        build_infer_patch,
        accumulate_block_predictions,
        finalize_predictions,
        find_uncovered_points,
        fallback_infer_for_uncovered,
        precompute_train_patches_2d,
        precompute_infer_patches_2d,
        precompute_infer_patches_4d,
        precompute_infer_patches_4d_block_center,
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


def read_regular_mask(group: Any, mask_key: str, n_grid: int) -> np.ndarray:
    """Read and validate a regular-grid boolean mask."""
    if mask_key not in group:
        raise KeyError(f"regular H5 group has no mask dataset {mask_key!r}")
    mask = np.asarray(group[mask_key][:]).reshape(-1).astype(bool)
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
        choices=["anchor_patch", "binning", "kdtree", "csg", "crg"],
        help="run mode",
    )
    parser.add_argument("--base_dir", type=str, default="/NAS/czt/mount/seis_flow_data12V2/h5/dongfang/")
    parser.add_argument("--raw_h5", type=str, default=None)
    parser.add_argument("--regular_h5", type=str, default=None)
    parser.add_argument("--target_h5", type=str, default=None)
    parser.add_argument("--group_key", type=str, default="1551")
    parser.add_argument("--trusted_mask_key", type=str, default=None)
    parser.add_argument("--num_anchors", type=int, default=2048)
    parser.add_argument("--k_patch", type=int, default=64)
    parser.add_argument("--top_l", type=int, default=128)
    parser.add_argument("--num_query", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--facility-nearest-l",
        type=int,
        default=None,
        help="facility_location 锚点：增益仅在距候选点最近的 L 个观测上累计；默认 None 表示对全体 N 点累计",
    )
    parser.add_argument("--grid_nx", type=int, default=0)
    parser.add_argument("--grid_ny", type=int, default=0)
    parser.add_argument("--block_bx", type=int, default=16)
    parser.add_argument("--block_by", type=int, default=16)
    parser.add_argument("--stride_sx", type=int, default=8)
    parser.add_argument("--stride_sy", type=int, default=8)
    parser.add_argument("--metric_weights", type=str, default="1,1,0.5,0.5")
    parser.add_argument(
        "--raw_key_aggregate",
        type=str,
        default="mean",
        choices=["none", "mean"],
        help="anchor_patch模式下是否先按键聚合raw观测道",
    )
    parser.add_argument(
        "--infer_query_from_missing_only",
        action="store_true",
        help="推理query仅使用regular中缺失道(mask==0)",
    )
    parser.add_argument(
        "--require_full_missing_coverage",
        action="store_true",
        help="当启用缺失query时，强制检查是否覆盖全部缺失道",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    patch_dir = os.path.join(base_dir, "patch")
    os.makedirs(patch_dir, exist_ok=True)
    info_h5_raw = args.raw_h5 or os.path.join(base_dir, "raw5d_data1104.h5")
    info_h5_regular = args.regular_h5 or os.path.join(base_dir, "reg5dbin_label1031.h5")
    info_h5_target = args.target_h5 or os.path.join(base_dir, "reg5dbin_label1031_binning.h5")

    metric_weights = [float(v) for v in args.metric_weights.split(",")]
    if len(metric_weights) != 4:
        raise ValueError("--metric_weights must contain 4 comma-separated values")

    def cover_dict2npy(gather_dict):
        return np.concatenate([np.array(gather_dict[key]) for key in gather_dict.keys()]), np.array(list(gather_dict.keys()))

    with File(info_h5_raw, "r") as f_raw, File(info_h5_regular, "r+") as f_reg:
        info_f_raw = f_raw[args.group_key]
        info_f_regular = f_reg[args.group_key]
        print("raw keys:", list(info_f_raw.keys()))
        print("regular keys:", list(info_f_regular.keys()))

        if args.mode == "anchor_patch":
            trace_obs_raw = _read_array(info_f_raw, "data").astype(np.float32)
            coord_obs_raw = np.column_stack(
                (
                    _read_array(info_f_raw, "sx"),
                    _read_array(info_f_raw, "sy"),
                    _read_array(info_f_raw, "rx"),
                    _read_array(info_f_raw, "ry"),
                )
            ).astype(np.float32)
            raw_keys = generate_binning_keys(info_f_raw)
            if trace_obs_raw.shape[0] != coord_obs_raw.shape[0] or trace_obs_raw.shape[0] != raw_keys.shape[0]:
                raise ValueError("raw data/coord/keys length mismatch")

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
            reg_keys = generate_binning_keys(info_f_regular)

            if args.trusted_mask_key is not None and args.trusted_mask_key in info_f_raw:
                trusted_mask = _read_array(info_f_raw, args.trusted_mask_key).astype(bool)
                trusted_idx = np.flatnonzero(trusted_mask).astype(np.int64)
                print(
                    f"trusted_idx from raw mask key={args.trusted_mask_key}, "
                    f"count={trusted_idx.size}"
                )
            elif "mask" in info_f_regular:
                # Preferred mapping: regular mask -> trusted regular keys -> observed keys.
                reg_mask = _read_array(info_f_regular, "mask").astype(bool).reshape(-1)
                if reg_mask.shape[0] != reg_keys.shape[0]:
                    raise ValueError("regular mask length mismatch with regular keys")
                trusted_reg_keys = reg_keys[reg_mask]
                trusted_idx = np.flatnonzero(
                    np.isin(_rows_as_struct(obs_keys), _rows_as_struct(trusted_reg_keys))
                ).astype(np.int64)
                print(
                    "trusted_idx from regular mask-key mapping, "
                    f"regular_mask_sum={int(reg_mask.sum())}, trusted_idx_count={trusted_idx.size}"
                )
            else:
                trusted_idx = np.arange(coord_obs.shape[0], dtype=np.int64)
                print(f"trusted_idx use all observed traces, count={trusted_idx.size}")

            coord_obs_norm, coord_grid_norm, norm_stats = normalize_coords(coord_obs, coord_grid)
            norm_stats_flat = {
                "obs_min": norm_stats["obs"]["min"],
                "obs_max": norm_stats["obs"]["max"],
                "obs_mean": norm_stats["obs"]["mean"],
                "obs_std": norm_stats["obs"]["std"],
                "grid_min": norm_stats["grid"]["min"],
                "grid_max": norm_stats["grid"]["max"],
                "grid_mean": norm_stats["grid"]["mean"],
                "grid_std": norm_stats["grid"]["std"],
            }
            np.savez(os.path.join(patch_dir, "coord_norm_stats.npz"), **norm_stats_flat)

            train_pack = precompute_train_patches_2d(
                coord_obs_norm=coord_obs_norm,
                trace_obs=trace_obs,
                trusted_idx=trusted_idx,
                num_anchors=args.num_anchors,
                k_patch=args.k_patch,
                top_l=args.top_l,
                metric_weights=metric_weights,
                beta=args.beta,
                facility_nearest_l=args.facility_nearest_l,
            )
            # Saved as 2D arrays (pad=-1), compatible with np.load(... )['0'] style.
            np.savez(os.path.join(patch_dir, "anchor_train_patch_idx_2d.npz"), **{"0": train_pack["patch_idx_2d"]})
            np.savez(os.path.join(patch_dir, "anchor_train_context_idx_2d.npz"), **{"0": train_pack["context_idx_2d"]})
            np.savez(os.path.join(patch_dir, "anchor_train_query_idx_2d.npz"), **{"0": train_pack["query_idx_2d"]})
            np.save(os.path.join(patch_dir, "anchor_train_anchor_idx.npy"), train_pack["anchor_idx"])
            np.save(os.path.join(patch_dir, "anchor_train_anchor_coord.npy"), train_pack["anchor_coord"])

            if args.grid_nx > 0 and args.grid_ny > 0:
                grid_shape_or_indices = (args.grid_nx, args.grid_ny)
            else:
                # Fallback: treat flattened grid as one-row 2D index map.
                grid_shape_or_indices = np.arange(coord_grid_norm.shape[0], dtype=np.int64).reshape(1, -1)
                print("warning: grid_nx/grid_ny not set, fallback to shape [1, N_grid].")

            infer_pack = precompute_infer_patches_2d(
                coord_obs_norm=coord_obs_norm,
                coord_grid_norm=coord_grid_norm,
                grid_shape_or_indices=grid_shape_or_indices,
                block_size=(args.block_bx, args.block_by),
                stride=(args.stride_sx, args.stride_sy),
                k_patch=args.k_patch,
                top_l=args.top_l,
                metric_weights=metric_weights,
                beta=args.beta,
                grid_query_mask=(
                    (_read_array(info_f_regular, "mask").reshape(-1) == 0)
                    if (args.infer_query_from_missing_only and "mask" in info_f_regular)
                    else None
                ),
                require_full_query_coverage=args.require_full_missing_coverage,
            )
            np.savez(os.path.join(patch_dir, "infer_patch_idx_2d.npz"), **{"0": infer_pack["patch_idx_2d"]})
            np.savez(os.path.join(patch_dir, "infer_patch_mask_2d.npz"), **{"0": infer_pack["patch_mask_2d"]})
            np.save(os.path.join(patch_dir, "infer_block_id.npy"), infer_pack["block_id"])
            np.save(os.path.join(patch_dir, "infer_block_center_grid_idx.npy"), infer_pack["block_center_grid_idx"])

            print("saved train/infer index arrays to:", patch_dir)
            print("train patch 2d:", train_pack["patch_idx_2d"].shape)
            print("train context 2d:", train_pack["context_idx_2d"].shape)
            print("train query 2d:", train_pack["query_idx_2d"].shape)
            print("infer patch 2d:", infer_pack["patch_idx_2d"].shape)
            print("infer patch mask 2d:", infer_pack["patch_mask_2d"].shape)

        elif args.mode == "binning":
            target, mask, report = binning(info_f_raw, info_f_regular)
            info_f_regular["mask"] = mask
            print("缺失率：", (1 - mask.sum() / len(mask)))
            print("分箱报告:", report)
            saveh5(target, info_f_regular, info_h5_target, args.group_key)

            with File(info_h5_target, "r") as f_target:
                print("target keys:", list(f_target[args.group_key].keys()))
        else:
            with File(info_h5_target, "r") as f_target:
                info_f_target = f_target[args.group_key]
                print("target keys:", list(info_f_target.keys()))
                if args.mode == "kdtree":
                    train_neighbors, test_neighbors, val_idx = kdtree(info_f_regular)
                    print("kdtree train/test/val:", train_neighbors.shape, test_neighbors.shape, val_idx.shape)
                elif args.mode == "csg":
                    csg_reg = gather(info_f_regular, "csg")
                    csg_raw = gather(info_f_raw, "csg")
                    csg_tgt = gather(info_f_target, "csg")
                    csg_np_reg, idx_2_key_reg = cover_dict2npy(csg_reg)
                    csg_np_raw, idx_2_key_raw = cover_dict2npy(csg_raw)
                    csg_np_tgt, idx_2_key_tgt = cover_dict2npy(csg_tgt)
                    print("shot_num:", len(csg_reg.keys()))
                    np.savez(os.path.join(patch_dir, "csg_np_reg.npz"), csg_np_reg)
                    np.savez(os.path.join(patch_dir, "csg_np_raw.npz"), csg_np_raw)
                    np.savez(os.path.join(patch_dir, "csg_np_tgt.npz"), csg_np_tgt)
                    np.save(os.path.join(patch_dir, "csg_idx_2_key_reg.npy"), idx_2_key_reg)
                    np.save(os.path.join(patch_dir, "csg_idx_2_key_raw.npy"), idx_2_key_raw)
                    np.save(os.path.join(patch_dir, "csg_idx_2_key_tgt.npy"), idx_2_key_tgt)
                elif args.mode == "crg":
                    crg_reg = gather(info_f_regular, "crg")
                    crg_raw = gather(info_f_raw, "crg")
                    crg_tgt = gather(info_f_target, "crg")
                    crg_np_reg, idx_2_key_reg = cover_dict2npy(crg_reg)
                    crg_np_raw, idx_2_key_raw = cover_dict2npy(crg_raw)
                    crg_np_tgt, idx_2_key_tgt = cover_dict2npy(crg_tgt)
                    print("recv_num:", len(crg_reg.keys()))
                    np.savez(os.path.join(patch_dir, "crg_np_reg.npz"), crg_np_reg)
                    np.savez(os.path.join(patch_dir, "crg_np_raw.npz"), crg_np_raw)
                    np.savez(os.path.join(patch_dir, "crg_np_tgt.npz"), crg_np_tgt)
                    np.save(os.path.join(patch_dir, "crg_idx_2_key_reg.npy"), idx_2_key_reg)
                    np.save(os.path.join(patch_dir, "crg_idx_2_key_raw.npy"), idx_2_key_raw)
                    np.save(os.path.join(patch_dir, "crg_idx_2_key_tgt.npy"), idx_2_key_tgt)
