"""DatasetH5Interp: query-context interpolation dataset on the regular grid.

This is the single supervised-interpolation dataset for all precomputation modes
(binning, csg, crg, kdtree). Precomputation produces standardized ``train_pool_idx_2d.npz``
(training) and ``infer_query_context.npz`` (inference); this dataset consumes them
uniformly.

Key distinction from ``DatasetH5_all_queryctx``:
- One H5 source for both query and context (the binned regular grid).
- ``mask`` in the H5 marks observed (1) vs missing (0) grid positions.
- Training: self-supervised query-context on observed positions.
- Inference: query = missing positions, context = observed positions.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from h5py import File

from utils.sampler_utils import diverse_topk
from config.segy_config import get_coord_col, get_trace_sort_keys

    

def _amplitude_metadata(thres: float, clip_percentile: float = 99.5) -> Dict[str, Any]:
    thres = float(max(float(thres), 1e-6))
    return {
        "amp_scale": np.float32(thres),
        "amp_clip": np.float32(thres),
        "amp_clip_percentile": np.float32(clip_percentile),
    }


class DatasetH5Interp:
    """Query-context interpolation dataset on the regular grid.

    Consumes precomputed index npz files produced by ``reg_tool/core.py``
    (binning/csg/crg/kdtree + anchor_patch modes). Compatible with the
    same training loop used by ``DatasetH5_all_queryctx``.

    Parameters
    ----------
    h5File : str
        Path to the binned/regular-grid H5 (contains ``data``, coords, ``mask``).
    dataset_neighbors : str or None
        Path to ``train_pool_idx_2d.npz`` (train) or ``infer_query_context.npz``
        (infer). Required.
    train : bool
        True for training, False for inference.
    train_num_query : int
        Number of query traces per training sample.
    patch_beta : float
        Diversity weight for diverse_topk.
    patch_metric_weights : sequence of 4 floats or None
        Weights for (sx, sy, rx, ry) in context selection.
    trace_ps : int
        Total traces per patch (query + context).
    time_ps : int
        Number of time samples per trace.
    """

    _COORD_COL = get_coord_col()  # from segy_config (active preset)

    def __init__(
        self,
        h5File: str = None,
        dataset_neighbors: Optional[str] = None,
        train: bool = False,
        train_num_query: int = 16,
        train_context_size: Optional[int] = None,
        patch_beta: float = 0.3,
        patch_metric_weights=None,
        force_anchor_query: bool = False,
        trace_sort_keys: Optional[Tuple[str, ...]] = None,
        time_ps: int = 1256,
        trace_ps: int = 128,
    ):
        super().__init__()
        print("Loading DatasetH5Interp...")

        self.h5File = h5File
        self.time_ps = time_ps
        self.trace_ps = trace_ps
        self.train = train
        self._rng = np.random.default_rng(123)
        self.std_val = None
        self.train_num_query = int(max(1, train_num_query))
        self.train_context_size = (
            None if train_context_size is None else int(max(1, train_context_size))
        )
        self.patch_beta = float(patch_beta)
        self.patch_metric_weights = patch_metric_weights
        self.force_anchor_query = bool(force_anchor_query)
        if trace_sort_keys is None:
            trace_sort_keys = get_trace_sort_keys()
        self.trace_sort_keys = tuple(trace_sort_keys)

        self.dt_ms = 4
        self.t0_ms = 0

        self.h5_data = self._load_h5_group(self.h5File)
        print(self.h5_data["data"].shape)

        self.coord_stats = self._compute_coord_stats()
        print("coord_stats computed")

        self.patch_meta = self._load_patch_metadata(dataset_neighbors)
        self.patch_mode = self.patch_meta["mode"]
        self.num_samples = int(self.patch_meta["num_samples"])
        print(f"patch_mode={self.patch_mode}  samples={self.num_samples}")

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def typical_grid_step(self, arr, eps=1e-9):
        u = np.sort(np.unique(arr))
        if u.size < 2:
            return None, u
        d = np.diff(u)
        d = d[d > eps]
        if d.size == 0:
            return None, u
        return float(np.median(d)), u

    def __len__(self):
        return self.num_samples

    @staticmethod
    def _load_h5_group(file_path):
        with File(file_path, "r") as f:
            for key in f:
                node = f[key]
                if hasattr(node, "keys") and "data" in node:
                    break
            return {k: node[k][()] for k in node.keys()}

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _load_patch_metadata(self, path: Optional[str]) -> Dict[str, Any]:
        if path is None:
            raise ValueError("dataset_neighbors is required")
        raw = np.load(path, allow_pickle=True)
        if hasattr(raw, "files"):
            arrays = {k: raw[k] for k in raw.files}
            raw.close()
        else:
            arrays = {"0": np.asarray(raw, dtype=object)}

        # infer_query_context format (all modes)
        if "grid_query_idx_list" in arrays and "context_idx_list" in arrays:
            return {
                "mode": "infer_query_context",
                "num_samples": int(len(arrays["grid_query_idx_list"])),
                "grid_query_idx_list": arrays["grid_query_idx_list"],
                "context_idx_list": arrays["context_idx_list"],
                "block_id": arrays.get("block_id"),
                "block_center_grid_idx": arrays.get("block_center_grid_idx"),
                "anchor_grid_idx_list": arrays.get("anchor_grid_idx_list"),
            }

        # csg/crg native format: pool_idx (object array) + pool_key
        if "pool_idx" in arrays and "pool_key" in arrays:
            pool_idx = np.asarray(arrays["pool_idx"], dtype=object)
            return {
                "mode": "train_pool",
                "num_samples": int(pool_idx.shape[0]),
                "pool_idx_2d": pool_idx,
                "pool_key": arrays["pool_key"],
            }

        # train_pool_idx_2d format (kdtree / precompute_anchor_patch_v2)
        if "pool_idx_2d" in arrays:
            pool = np.asarray(arrays["pool_idx_2d"], dtype=np.int64)
            return {
                "mode": "train_pool",
                "num_samples": int(pool.shape[0]),
                "pool_idx_2d": pool,
                "anchor_idx": arrays.get("anchor_idx"),
            }

        # legacy patch_idx_2d
        if "patch_idx_2d" in arrays:
            p2d = np.asarray(arrays["patch_idx_2d"], dtype=np.int64)
            return {
                "mode": "train_pool",
                "num_samples": int(p2d.shape[0]),
                "pool_idx_2d": p2d,
                "anchor_idx": arrays.get("anchor_idx"),
            }

        if "0" in arrays:
            p2d = np.asarray(arrays["0"], dtype=np.int64)
            return {
                "mode": "train_pool" if self.train else "legacy",
                "num_samples": int(p2d.shape[0]),
                "pool_idx_2d": p2d,
                "anchor_idx": arrays.get("anchor_idx"),
            }

        raise ValueError("Unsupported dataset_neighbors format")

    # ------------------------------------------------------------------
    # Index utilities
    # ------------------------------------------------------------------

    def _index_row(self, storage: np.ndarray, idx: int) -> np.ndarray:
        row = np.asarray(storage[idx], dtype=np.int64).reshape(-1)
        return row[row >= 0]

    def _take_rows(self, dataset, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            s = np.asarray(dataset[:1])
            if s.ndim == 1:
                return np.zeros((0,), dtype=s.dtype)
            return np.zeros((0, s.shape[1]), dtype=s.dtype)
        order = np.argsort(idx, kind="stable")
        sorted_idx = idx[order]
        out = np.asarray(dataset[sorted_idx])
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)
        return out[inv]

    # ------------------------------------------------------------------
    # Time / scale
    # ------------------------------------------------------------------

    def _crop_or_pad_time(self, traces: np.ndarray) -> np.ndarray:
        traces = np.asarray(traces)
        diff = traces.shape[1] - self.time_ps
        if diff > 0:
            return traces[:, diff:]
        if diff < 0:
            return np.pad(traces, ((0, 0), (-diff, 0)), constant_values=0)
        return traces

    def _time_axis_2d(self, n_trace: int) -> np.ndarray:
        t1d = np.arange(0, self.time_ps, dtype=np.int32)
        ta = self.t0_ms + t1d.astype(np.float32) * self.dt_ms
        return np.tile(ta[None, :], (int(n_trace), 1)).astype(np.float32)

    def _scale_pair(self, data_patch, masked_patch, is_query):
        obs = masked_patch[~is_query]
        obs = obs[np.isfinite(obs)]
        std_val = np.float32(np.std(obs)) if obs.size > 0 else np.float32(0.0)
        std_val = np.float32(max(std_val, 1e-2))
        ref = np.abs(obs) if obs.size > 0 else np.abs(masked_patch[np.isfinite(masked_patch)])
        thres = np.percentile(ref, 99.5) if ref.size > 0 else 1e-6
        thres = float(max(thres, 1e-6))
        mp = np.clip(masked_patch, -thres, thres) / thres
        dp = np.clip(data_patch, -thres, thres) / thres
        self.std_val = std_val
        return dp.astype(np.float32), mp.astype(np.float32), std_val, np.float32(thres)

    def _sample_rng(self, idx: int) -> np.random.Generator:
        seed = int(self._rng.integers(0, 2 ** 31 - 1)) ^ int(idx)
        return np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Trace sorting
    # ------------------------------------------------------------------

    def _sort_traces(self, data_patch, is_query, coords_patch):
        if not self.trace_sort_keys:
            order = np.arange(data_patch.shape[0])
            return data_patch, is_query, coords_patch, order
        cols = [coords_patch[:, self._COORD_COL[k]] for k in reversed(self.trace_sort_keys)]
        order = np.lexsort(cols)
        return data_patch[order], is_query[order], coords_patch[order], order

    # ------------------------------------------------------------------
    # Coordinate normalization
    # ------------------------------------------------------------------

    def _normalize_coords(self, sx, sy, rx, ry):
        stats = self.coord_stats
        sx_n = 2.0 * (sx - stats["sx_min"]) / (stats["sx_max"] - stats["sx_min"]) - 1.0
        sy_n = 2.0 * (sy - stats["sy_min"]) / (stats["sy_max"] - stats["sy_min"]) - 1.0
        rx_n = 2.0 * (rx - stats["rx_min"]) / (stats["rx_max"] - stats["rx_min"]) - 1.0
        ry_n = 2.0 * (ry - stats["ry_min"]) / (stats["ry_max"] - stats["ry_min"]) - 1.0
        return sx_n, sy_n, rx_n, ry_n

    def _compute_coord_stats(self):
        sx_all = self.h5_data["sx"]
        sy_all = self.h5_data["sy"]
        rx_all = self.h5_data["rx"]
        ry_all = self.h5_data["ry"]
        dsx, sx_u = self.typical_grid_step(sx_all)
        dsy, sy_u = self.typical_grid_step(sy_all)
        drx, rx_u = self.typical_grid_step(rx_all)
        dry, ry_u = self.typical_grid_step(ry_all)

        return {
            "sx_min": float(sx_u.min()), "sx_max": float(sx_u.max()),
            "sy_min": float(sy_u.min()), "sy_max": float(sy_u.max()),
            "rx_min": float(rx_u.min()), "rx_max": float(rx_u.max()),
            "ry_min": float(ry_u.min()), "ry_max": float(ry_u.max()),
            "grid_step_sx": dsx, "grid_step_sy": dsy,
            "grid_step_rx": drx, "grid_step_ry": dry,
            "Lx": 0.5 * max(sx_u.max() - sx_u.min(), rx_u.max() - rx_u.min()),
            "Ly": 0.5 * max(sy_u.max() - sy_u.min(), ry_u.max() - ry_u.min()),
        }

    # ------------------------------------------------------------------
    # Training sample
    # ------------------------------------------------------------------

    def _build_train_query_context_sample(self, idx: int) -> Dict[str, Any]:
        pool_idx = self._index_row(self.patch_meta["pool_idx_2d"], idx)
        if pool_idx.size < 2:
            raise RuntimeError("train pool must contain at least 2 traces")

        anchor_idx = None
        if self.patch_meta.get("anchor_idx") is not None:
            raw_a = np.asarray(self.patch_meta["anchor_idx"])
            if raw_a.ndim == 1 and idx < raw_a.shape[0]:
                anchor_idx = int(raw_a[idx])

        data_pool = self._crop_or_pad_time(
            self._take_rows(self.h5_data["data"], pool_idx)
        ).astype(np.float32)
        rx_pool = self._take_rows(self.h5_data["rx"], pool_idx).astype(np.float32)
        ry_pool = self._take_rows(self.h5_data["ry"], pool_idx).astype(np.float32)
        sx_pool = self._take_rows(self.h5_data["sx"], pool_idx).astype(np.float32)
        sy_pool = self._take_rows(self.h5_data["sy"], pool_idx).astype(np.float32)

        sx_n, sy_n, rx_n, ry_n = self._normalize_coords(sx_pool, sy_pool, rx_pool, ry_pool)
        coords_pool = np.stack([sx_n, sy_n, rx_n, ry_n], axis=1).astype(np.float32)

        rng = self._sample_rng(idx)
        q_eff = min(self.train_num_query, int(pool_idx.size) - 1)
        if q_eff < 1:
            raise RuntimeError("effective train query count must be >= 1")

        k_ctx = min(
            max(1, self.trace_ps - q_eff) if self.train_context_size is None else self.train_context_size,
            int(pool_idx.size) - q_eff,
        )
        if k_ctx < 1:
            raise RuntimeError("effective train context count must be >= 1")

        perm = rng.permutation(pool_idx.size)

        if self.force_anchor_query and anchor_idx is not None and np.any(pool_idx == anchor_idx):
            anchor_local = int(np.flatnonzero(pool_idx == anchor_idx)[0])
            rest = perm[perm != anchor_local]
            query_local = np.concatenate([
                np.asarray([anchor_local], dtype=np.int64),
                rest[: max(0, q_eff - 1)].astype(np.int64),
            ])
        else:
            query_local = perm[:q_eff].astype(np.int64, copy=False)

        candidate_local = np.asarray(
            [i for i in range(pool_idx.size) if i not in set(query_local.tolist())],
            dtype=np.int64,
        )
        center_coord = np.mean(coords_pool[query_local], axis=0).astype(np.float32, copy=False)
        context_local = diverse_topk(
            center_coord=center_coord,
            candidate_idx=candidate_local,
            all_coords=coords_pool,
            k=k_ctx,
            metric_weights=self.patch_metric_weights,
            beta=self.patch_beta,
        ).astype(np.int64, copy=False)
        if context_local.size == 0:
            raise RuntimeError("empty training context")

        patch_local = np.concatenate([query_local, context_local], axis=0)
        data_patch = data_pool[patch_local].astype(np.float32, copy=False)
        is_query_orig = np.zeros((patch_local.size,), dtype=bool)
        is_query_orig[: query_local.size] = True

        coords_patch = coords_pool[patch_local].astype(np.float32, copy=False)
        data_patch, is_query, coords_patch, _ = self._sort_traces(data_patch, is_query_orig, coords_patch)

        masked_patch = data_patch.copy()
        masked_patch[is_query] = 0.0
        data_patch, masked_patch, std_val, thres = self._scale_pair(data_patch, masked_patch, is_query)

        return {
            "data": data_patch,
            "masked_patch": masked_patch,
            "rx_patch": coords_patch[:, 2].astype(np.float32, copy=False),
            "ry_patch": coords_patch[:, 3].astype(np.float32, copy=False),
            "sx_patch": coords_patch[:, 0].astype(np.float32, copy=False),
            "sy_patch": coords_patch[:, 1].astype(np.float32, copy=False),
            "time_axis_2d": self._time_axis_2d(patch_local.size),
            "std_val": std_val,
            "is_query": is_query,
            "query_count": np.int64(query_local.size),
            "context_count": np.int64(context_local.size),
            "query_global_idx": pool_idx[query_local].astype(np.int64, copy=False),
            "context_global_idx": pool_idx[context_local].astype(np.int64, copy=False),
            "pool_global_idx": pool_idx.astype(np.int64, copy=False),
            "anchor_global_idx": np.int64(-1 if anchor_idx is None else anchor_idx),
            **_amplitude_metadata(thres),
        }

    # ------------------------------------------------------------------
    # Inference sample
    # ------------------------------------------------------------------

    def _build_infer_query_context_sample(self, idx: int) -> Dict[str, Any]:
        query_idx = self._index_row(self.patch_meta["grid_query_idx_list"], idx)
        context_idx = self._index_row(self.patch_meta["context_idx_list"], idx)
        if query_idx.size == 0 or context_idx.size == 0:
            raise RuntimeError("infer sample must contain non-empty query and context")

        query_data = self._crop_or_pad_time(
            self._take_rows(self.h5_data["data"], query_idx)
        ).astype(np.float32)
        context_data = self._crop_or_pad_time(
            self._take_rows(self.h5_data["data"], context_idx)
        ).astype(np.float32)
        data_patch = np.concatenate([query_data, context_data], axis=0).astype(np.float32, copy=False)
        is_query_orig = np.zeros((data_patch.shape[0],), dtype=bool)
        is_query_orig[: query_idx.size] = True

        rx_q = self._take_rows(self.h5_data["rx"], query_idx).astype(np.float32)
        ry_q = self._take_rows(self.h5_data["ry"], query_idx).astype(np.float32)
        sx_q = self._take_rows(self.h5_data["sx"], query_idx).astype(np.float32)
        sy_q = self._take_rows(self.h5_data["sy"], query_idx).astype(np.float32)
        rx_c = self._take_rows(self.h5_data["rx"], context_idx).astype(np.float32)
        ry_c = self._take_rows(self.h5_data["ry"], context_idx).astype(np.float32)
        sx_c = self._take_rows(self.h5_data["sx"], context_idx).astype(np.float32)
        sy_c = self._take_rows(self.h5_data["sy"], context_idx).astype(np.float32)
        sx_qn, sy_qn, rx_qn, ry_qn = self._normalize_coords(sx_q, sy_q, rx_q, ry_q)
        sx_cn, sy_cn, rx_cn, ry_cn = self._normalize_coords(sx_c, sy_c, rx_c, ry_c)

        coords_patch = np.stack([
            np.concatenate([sx_qn, sx_cn]),
            np.concatenate([sy_qn, sy_cn]),
            np.concatenate([rx_qn, rx_cn]),
            np.concatenate([ry_qn, ry_cn]),
        ], axis=1).astype(np.float32)
        data_patch, is_query, coords_patch, _order = self._sort_traces(data_patch, is_query_orig, coords_patch)

        masked_patch = data_patch.copy()
        masked_patch[is_query] = 0.0
        data_raw = data_patch.astype(np.float32, copy=True)
        masked_raw = masked_patch.astype(np.float32, copy=True)
        data_patch, masked_patch, std_val, thres = self._scale_pair(data_patch, masked_patch, is_query)

        sl_q = self._take_rows(self.h5_data["shot_line"], query_idx)
        ss_q = self._take_rows(self.h5_data["shot_stake"], query_idx)
        rl_q = self._take_rows(self.h5_data["recv_line"], query_idx)
        rs_q = self._take_rows(self.h5_data["recv_stake"], query_idx)
        sl_c = self._take_rows(self.h5_data["shot_line"], context_idx)
        ss_c = self._take_rows(self.h5_data["shot_stake"], context_idx)
        rl_c = self._take_rows(self.h5_data["recv_line"], context_idx)
        rs_c = self._take_rows(self.h5_data["recv_stake"], context_idx)
        patch_info = {
            "shot_line": np.concatenate([sl_q, sl_c])[_order],
            "shot_stake": np.concatenate([ss_q, ss_c])[_order],
            "recv_line": np.concatenate([rl_q, rl_c])[_order],
            "recv_stake": np.concatenate([rs_q, rs_c])[_order],
        }

        out = {
            "data": data_patch,
            "masked_patch": masked_patch,
            "rx_patch": coords_patch[:, 2].astype(np.float32, copy=False),
            "ry_patch": coords_patch[:, 3].astype(np.float32, copy=False),
            "sx_patch": coords_patch[:, 0].astype(np.float32, copy=False),
            "sy_patch": coords_patch[:, 1].astype(np.float32, copy=False),
            "time_axis_2d": self._time_axis_2d(data_patch.shape[0]),
            "std_val": std_val,
            "is_query": is_query,
            "query_count": np.int64(query_idx.size),
            "context_count": np.int64(context_idx.size),
            "grid_query_idx": query_idx.astype(np.int64, copy=False),
            "context_idx": context_idx.astype(np.int64, copy=False),
            "patch_info": patch_info,
            "data_raw": data_raw,
            "masked_patch_raw": masked_raw,
            **_amplitude_metadata(thres),
        }
        if self.patch_meta.get("block_id") is not None:
            out["block_id"] = np.int64(np.asarray(self.patch_meta["block_id"])[idx])
        if self.patch_meta.get("block_center_grid_idx") is not None:
            out["block_center_grid_idx"] = np.int64(np.asarray(self.patch_meta["block_center_grid_idx"])[idx])
        if self.patch_meta.get("anchor_grid_idx_list") is not None:
            out["anchor_grid_idx"] = self._index_row(self.patch_meta["anchor_grid_idx_list"], idx)
        return out

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        if self.patch_mode == "train_pool":
            return self._build_train_query_context_sample(idx)
        if (not self.train) and self.patch_mode == "infer_query_context":
            return self._build_infer_query_context_sample(idx)
        raise NotImplementedError(
            f"DatasetH5Interp: unsupported train={self.train!r}, patch_mode={self.patch_mode!r}"
        )
