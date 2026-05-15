"""DatasetH5_all_queryctx: query-context dataset for seismic interpolation.

Provides a self-contained PyTorch Dataset that:
- Training mode (train_pool): randomly selects query traces from a per-sample pool,
  then selects context traces via diverse_topk. Query traces are masked to zero.
- Inference mode (infer_query_context): uses precomputed grid-query and context
  indices from separate H5 files (regular for queries, irregular for context).

Dependencies:
    numpy, h5py, torch
    queryctx_module.utils.sampler_utils (diverse_topk, parse_metric_weights, weighted_sqdist_to_one)
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from h5py import File

try:
    from ..utils.sampler_utils import diverse_topk
except ImportError:
    from queryctx_module.utils.sampler_utils import diverse_topk

try:
    from ..config.data_config import object_args as _default_args
except ImportError:
    from queryctx_module.config.data_config import object_args as _default_args

try:
    from ..config.segy_config import COORD_COL as _COORD_COL, TRACE_SORT_KEYS
except ImportError:
    from config.segy_config import COORD_COL as _COORD_COL, TRACE_SORT_KEYS


def amplitude_metadata(thres: float, clip_percentile: float = 99.5) -> Dict[str, Any]:
    thres = float(max(float(thres), 1e-6))
    return {
        "amp_scale": np.float32(thres),
        "amp_clip": np.float32(thres),
        "amp_clip_percentile": np.float32(clip_percentile),
    }


class DatasetH5_all_queryctx:
    """Query-context dataset for seismic 5D interpolation.

    Supports two metadata modes:
    - ``train_pool``: per-sample pool of trace indices; online query/context selection
    - ``infer_query_context``: precomputed grid-query + context index pairs

    Attributes:
        h5File: path to irregular (observed) H5 file
        h5File_regular: path to regular-grid H5 file
        time_ps: number of time samples per trace
        trace_ps: total traces per patch (query + context)
        coord_stats: per-axis min/max and grid-step statistics
        patch_mode: "train_pool" or "infer_query_context"
    """

    _COORD_COL = _COORD_COL  # from segy_config

    def __init__(
        self,
        h5File=None,
        h5File_regular=None,
        h5File_tgt=None,
        dataset_neighbors=None,
        train=None,
        train_num_query: int = 16,
        train_context_size: Optional[int] = None,
        patch_beta: float = 0.3,
        patch_metric_weights=None,
        force_anchor_query: bool = False,
        trace_sort_keys: Tuple[str, ...] = TRACE_SORT_KEYS,
        use_p_scale: bool = False,
        time_ps: int = 1256,
        trace_ps: int = 128,
    ):
        super().__init__()
        print("Loading dataset...")

        self.h5File = h5File
        self.h5File_regular = h5File_regular
        self.h5File_tgt = h5File_tgt
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
        self.trace_sort_keys = tuple(trace_sort_keys)
        self.use_p_scale = use_p_scale

        self.dt_ms = 4
        self.t0_ms = 0
        self.scale = None

        self.h5_data = self._load_h5_group(self.h5File)
        self.h5_data_regular = self._load_h5_group(self.h5File_regular)
        self.h5_data_tgt = {}

        print(self.h5_data_regular["data"].shape)
        print(self.h5_data["data"].shape)
        print("loading data")

        self.coord_stats = self.compute_coord_stats()
        print("coord_stats computed")
        self.patch_meta = self._load_patch_metadata(dataset_neighbors)
        self.patch_mode = self.patch_meta["mode"]
        print(self.patch_mode)
        self.num_samples = int(self.patch_meta["num_samples"])
        print(f"patch metadata mode: {self.patch_mode}, samples: {self.num_samples}")

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

    # ------------------------------------------------------------------
    # Internal: H5 loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_h5_group(file_path):
        with File(file_path, "r") as f:
            for key in f:
                node = f[key]
                if hasattr(node, "keys") and "data" in node:
                    break
            return {k: node[k][:] for k in node.keys()}

    # ------------------------------------------------------------------
    # Internal: metadata
    # ------------------------------------------------------------------

    def _load_patch_metadata(self, path: Optional[str]) -> Dict[str, Any]:
        if path is None:
            raise ValueError("dataset_neighbors is required")
        raw = np.load(path, allow_pickle=True)
        if hasattr(raw, "files"):
            arrays = {k: raw[k] for k in raw.files}
            raw.close()
        else:
            arrays = {"0": raw}

        if "grid_query_idx_list" in arrays and (
            "context_idx_list" in arrays or "patch_idx_list" in arrays
        ):
            num_samples = len(arrays["grid_query_idx_list"])
            return {
                "mode": "infer_query_context",
                "num_samples": int(num_samples),
                "grid_query_idx_list": arrays["grid_query_idx_list"],
                "context_idx_list": arrays.get(
                    "context_idx_list", arrays.get("patch_idx_list")
                ),
                "block_id": arrays.get("block_id"),
                "block_center_grid_idx": arrays.get("block_center_grid_idx"),
                "anchor_grid_idx_list": arrays.get("anchor_grid_idx_list"),
            }

        if "pool_idx_2d" in arrays:
            pool_idx_2d = np.asarray(arrays["pool_idx_2d"], dtype=np.int64)
            return {
                "mode": "train_pool",
                "num_samples": int(pool_idx_2d.shape[0]),
                "pool_idx_2d": pool_idx_2d,
                "anchor_idx": arrays.get("anchor_idx"),
            }

        if "patch_idx_2d" in arrays and self.train:
            patch_idx_2d = np.asarray(arrays["patch_idx_2d"], dtype=np.int64)
            return {
                "mode": "train_pool",
                "num_samples": int(patch_idx_2d.shape[0]),
                "pool_idx_2d": patch_idx_2d,
                "anchor_idx": arrays.get("anchor_idx"),
            }

        if "patch_idx_2d" in arrays and (not self.train):
            patch_idx_2d = np.asarray(arrays["patch_idx_2d"], dtype=np.int64)
            return {
                "mode": "legacy",
                "num_samples": int(patch_idx_2d.shape[0]),
                "patch_idx_2d": patch_idx_2d,
            }

        if "0" in arrays:
            patch_idx_2d = np.asarray(arrays["0"], dtype=np.int64)
            return {
                "mode": "legacy",
                "num_samples": int(patch_idx_2d.shape[0]),
                "patch_idx_2d": patch_idx_2d,
            }

        raise ValueError(
            "Unsupported dataset_neighbors format. Expected legacy ['0'], "
            "train pool keys, or infer query/context keys."
        )

    # ------------------------------------------------------------------
    # Internal: index utilities
    # ------------------------------------------------------------------

    def _index_row(self, storage: np.ndarray, idx: int) -> np.ndarray:
        row = np.asarray(storage[idx], dtype=np.int64).reshape(-1)
        return row[row >= 0]

    def _take_rows(self, dataset, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            sample = np.asarray(dataset[:1])
            if sample.ndim == 1:
                return np.zeros((0,), dtype=sample.dtype)
            return np.zeros((0, sample.shape[1]), dtype=sample.dtype)
        order = np.argsort(idx, kind="stable")
        sorted_idx = idx[order]
        out = np.asarray(dataset[sorted_idx])
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)
        return out[inv]

    # ------------------------------------------------------------------
    # Time / scale utilities
    # ------------------------------------------------------------------

    def _crop_or_pad_time(self, traces: np.ndarray) -> np.ndarray:
        traces = np.asarray(traces)
        if traces.ndim != 2:
            raise ValueError(f"traces must be 2D [N, T], got {traces.shape}")
        diff = traces.shape[1] - self.time_ps
        if diff > 0:
            return traces[:, diff:]
        if diff < 0:
            return np.pad(traces, ((0, 0), (-diff, 0)), "constant", constant_values=0)
        return traces

    def _time_axis_2d(self, n_trace: int) -> np.ndarray:
        time_idx_1d = np.arange(0, self.time_ps, dtype=np.int32)
        time_axis_1d = self.t0_ms + time_idx_1d.astype(np.float32) * self.dt_ms
        return np.tile(time_axis_1d[None, :], (int(n_trace), 1)).astype(np.float32)

    def _scale_pair(
        self,
        data_patch: np.ndarray,
        masked_patch: np.ndarray,
        is_query: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.float32, np.float32]:
        obs = masked_patch[~is_query]
        obs = obs[np.isfinite(obs)]
        std_val = np.float32(np.std(obs)) if obs.size > 0 else np.float32(0.0)
        std_val = np.float32(max(std_val, 1e-2))
        ref = np.abs(obs) if obs.size > 0 else np.abs(masked_patch[np.isfinite(masked_patch)])
        thres = np.percentile(ref, 99.5) if ref.size > 0 else 1e-6
        thres = float(max(thres, 1e-6))
        masked_patch = np.clip(masked_patch, -thres, thres) / thres
        data_patch = np.clip(data_patch, -thres, thres) / thres
        self.std_val = std_val
        return (
            data_patch.astype(np.float32),
            masked_patch.astype(np.float32),
            std_val,
            np.float32(thres),
        )

    def _sample_rng(self, idx: int) -> np.random.Generator:
        seed = int(self._rng.integers(0, 2**31 - 1)) ^ int(idx)
        return np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Trace sorting
    # ------------------------------------------------------------------

    def _sort_traces(
        self,
        data_patch: np.ndarray,
        is_query: np.ndarray,
        coords_patch: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.trace_sort_keys:
            order = np.arange(data_patch.shape[0])
            return data_patch, is_query, coords_patch, order
        cols = [coords_patch[:, self._COORD_COL[k]] for k in reversed(self.trace_sort_keys)]
        order = np.lexsort(cols)
        return data_patch[order], is_query[order], coords_patch[order], order

    # ------------------------------------------------------------------
    # Coordinate normalization
    # ------------------------------------------------------------------

    def _normalize_coords(self, sx, sy, rx, ry) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        stats = self.coord_stats
        sx_n = 2 * (sx - stats["sx_min"]) / (stats["sx_max"] - stats["sx_min"]) - 1
        sy_n = 2 * (sy - stats["sy_min"]) / (stats["sy_max"] - stats["sy_min"]) - 1
        rx_n = 2 * (rx - stats["rx_min"]) / (stats["rx_max"] - stats["rx_min"]) - 1
        ry_n = 2 * (ry - stats["ry_min"]) / (stats["ry_max"] - stats["ry_min"]) - 1
        return sx_n, sy_n, rx_n, ry_n

    def compute_coord_stats(self):
        sx_all = self.h5_data_regular["sx"]
        sy_all = self.h5_data_regular["sy"]
        rx_all = self.h5_data_regular["rx"]
        ry_all = self.h5_data_regular["ry"]
        dsx, sx_u = self.typical_grid_step(sx_all)
        dsy, sy_u = self.typical_grid_step(sy_all)
        drx, rx_u = self.typical_grid_step(rx_all)
        dry, ry_u = self.typical_grid_step(ry_all)

        sx_min, sx_max = float(sx_u.min()), float(sx_u.max())
        sy_min, sy_max = float(sy_u.min()), float(sy_u.max())
        rx_min, rx_max = float(rx_u.min()), float(rx_u.max())
        ry_min, ry_max = float(ry_u.min()), float(ry_u.max())

        deltas = {}
        if dsx is not None and (sx_max - sx_min) > 0:
            deltas["sx"] = float((sx_max - sx_min) / (2 * dsx))
        if dsy is not None and (sy_max - sy_min) > 0:
            deltas["sy"] = float((sy_max - sy_min) / (2 * dsy))
        if drx is not None and (rx_max - rx_min) > 0:
            deltas["rx"] = float((rx_max - rx_min) / (2 * drx))
        if dry is not None and (ry_max - ry_min) > 0:
            deltas["ry"] = float((ry_max - ry_min) / (2 * dry))
        self.scale = deltas

        stats = {
            "sx_min": sx_all.min(),
            "sx_max": sx_all.max(),
            "sy_min": sy_all.min(),
            "sy_max": sy_all.max(),
            "rx_min": rx_all.min(),
            "rx_max": rx_all.max(),
            "ry_min": ry_all.min(),
            "ry_max": ry_all.max(),
            "grid_step_sx": dsx,
            "grid_step_sy": dsy,
            "grid_step_rx": drx,
            "grid_step_ry": dry,
        }

        if self.use_p_scale and self.scale:
            for name in ("sx", "sy", "rx", "ry"):
                s = self.scale.get(name)
                if s is not None:
                    stats[f"{name}_min"] *= s
                    stats[f"{name}_max"] *= s
            print(f"[DatasetH5_all_queryctx] p_scale applied to coord_stats: {self.scale}")

        stats["Lx"] = 0.5 * max(
            stats["sx_max"] - stats["sx_min"], stats["rx_max"] - stats["rx_min"]
        )
        stats["Ly"] = 0.5 * max(
            stats["sy_max"] - stats["sy_min"], stats["ry_max"] - stats["ry_min"]
        )
        return stats

    # ------------------------------------------------------------------
    # Training sample builder
    # ------------------------------------------------------------------

    def _build_train_query_context_sample(self, idx: int) -> Dict[str, Any]:
        pool_idx = self._index_row(self.patch_meta["pool_idx_2d"], idx)
        if pool_idx.size < 2:
            raise RuntimeError("train pool must contain at least 2 traces")
        anchor_idx = None
        if self.patch_meta.get("anchor_idx") is not None:
            anchor_idx = int(np.asarray(self.patch_meta["anchor_idx"])[idx])

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
        k_ctx_target = (
            max(1, self.trace_ps - q_eff)
            if self.train_context_size is None
            else self.train_context_size
        )
        k_ctx = min(int(k_ctx_target), int(pool_idx.size) - q_eff)
        if k_ctx < 1:
            raise RuntimeError("effective train context count must be >= 1")

        perm = rng.permutation(pool_idx.size)
        if (
            self.force_anchor_query
            and anchor_idx is not None
            and np.any(pool_idx == anchor_idx)
        ):
            anchor_local = int(np.flatnonzero(pool_idx == anchor_idx)[0])
            rest = perm[perm != anchor_local]
            extra = rest[: max(0, q_eff - 1)]
            query_local = np.concatenate(
                [np.asarray([anchor_local], dtype=np.int64), extra.astype(np.int64)],
                axis=0,
            )
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
            raise RuntimeError("failed to build non-empty training context from pool")

        patch_local = np.concatenate([query_local, context_local], axis=0)
        data_patch = data_pool[patch_local].astype(np.float32, copy=False)
        is_query_orig = np.zeros((patch_local.size,), dtype=bool)
        is_query_orig[: query_local.size] = True

        coords_patch = coords_pool[patch_local].astype(np.float32, copy=False)
        data_patch, is_query, coords_patch, _ = self._sort_traces(
            data_patch, is_query_orig, coords_patch
        )

        masked_patch = data_patch.copy()
        masked_patch[is_query] = 0.0
        data_patch, masked_patch, std_val, thres = self._scale_pair(
            data_patch, masked_patch, is_query
        )
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
            **amplitude_metadata(thres),
        }

    # ------------------------------------------------------------------
    # Inference sample builder
    # ------------------------------------------------------------------

    def _build_infer_query_context_sample(self, idx: int) -> Dict[str, Any]:
        query_idx = self._index_row(self.patch_meta["grid_query_idx_list"], idx)
        context_idx = self._index_row(self.patch_meta["context_idx_list"], idx)
        if query_idx.size == 0 or context_idx.size == 0:
            raise RuntimeError("infer sample must contain non-empty query and context")

        query_data = self._crop_or_pad_time(
            self._take_rows(self.h5_data_regular["data"], query_idx)
        ).astype(np.float32)
        context_data = self._crop_or_pad_time(
            self._take_rows(self.h5_data["data"], context_idx)
        ).astype(np.float32)
        data_patch = np.concatenate([query_data, context_data], axis=0).astype(
            np.float32, copy=False
        )
        is_query_orig = np.zeros((data_patch.shape[0],), dtype=bool)
        is_query_orig[: query_idx.size] = True

        rx_q = self._take_rows(self.h5_data_regular["rx"], query_idx).astype(np.float32)
        ry_q = self._take_rows(self.h5_data_regular["ry"], query_idx).astype(np.float32)
        sx_q = self._take_rows(self.h5_data_regular["sx"], query_idx).astype(np.float32)
        sy_q = self._take_rows(self.h5_data_regular["sy"], query_idx).astype(np.float32)
        rx_c = self._take_rows(self.h5_data["rx"], context_idx).astype(np.float32)
        ry_c = self._take_rows(self.h5_data["ry"], context_idx).astype(np.float32)
        sx_c = self._take_rows(self.h5_data["sx"], context_idx).astype(np.float32)
        sy_c = self._take_rows(self.h5_data["sy"], context_idx).astype(np.float32)
        sx_qn, sy_qn, rx_qn, ry_qn = self._normalize_coords(sx_q, sy_q, rx_q, ry_q)
        sx_cn, sy_cn, rx_cn, ry_cn = self._normalize_coords(sx_c, sy_c, rx_c, ry_c)

        coords_patch = np.stack(
            [
                np.concatenate([sx_qn, sx_cn]),
                np.concatenate([sy_qn, sy_cn]),
                np.concatenate([rx_qn, rx_cn]),
                np.concatenate([ry_qn, ry_cn]),
            ],
            axis=1,
        ).astype(np.float32)
        data_patch, is_query, coords_patch, _order = self._sort_traces(
            data_patch, is_query_orig, coords_patch
        )
        masked_patch = data_patch.copy()
        masked_patch[is_query] = 0.0
        data_raw = data_patch.astype(np.float32, copy=True)
        masked_raw = masked_patch.astype(np.float32, copy=True)
        data_patch, masked_patch, std_val, thres = self._scale_pair(
            data_patch, masked_patch, is_query
        )

        sl_q = self._take_rows(self.h5_data_regular["shot_line"], query_idx)
        ss_q = self._take_rows(self.h5_data_regular["shot_stake"], query_idx)
        rl_q = self._take_rows(self.h5_data_regular["recv_line"], query_idx)
        rs_q = self._take_rows(self.h5_data_regular["recv_stake"], query_idx)
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
            **amplitude_metadata(thres),
        }
        if self.patch_meta.get("block_id") is not None:
            out["block_id"] = np.int64(np.asarray(self.patch_meta["block_id"])[idx])
        if self.patch_meta.get("block_center_grid_idx") is not None:
            out["block_center_grid_idx"] = np.int64(
                np.asarray(self.patch_meta["block_center_grid_idx"])[idx]
            )
        if self.patch_meta.get("anchor_grid_idx_list") is not None:
            out["anchor_grid_idx"] = self._index_row(
                self.patch_meta["anchor_grid_idx_list"], idx
            )
        return out

    # ------------------------------------------------------------------
    # __getitem__ dispatch
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        if self.patch_mode == "train_pool":
            return self._build_train_query_context_sample(idx)

        if (not self.train) and self.patch_mode == "infer_query_context":
            return self._build_infer_query_context_sample(idx)

        raise NotImplementedError(
            f"DatasetH5_all_queryctx: unsupported train={self.train!r}, "
            f"patch_mode={self.patch_mode!r}"
        )


# ------------------------------------------------------------------
# Convenience: batch parsing helper
# ------------------------------------------------------------------

def batch_to_xy(batch):
    """Unify batch dict format to (data, data_mask, rx, ry, sx, sy).

    Supports queryctx dict keys (data, masked_patch, rx_patch, ry_patch, sx_patch, sy_patch)
    and SegySSL dict keys (x_gt, x_obs, gx, gy, sx, sy).
    """
    if not isinstance(batch, dict):
        return None
    if "x_gt" in batch:
        return (
            batch["x_gt"],
            batch["x_obs"],
            batch["gx"],
            batch["gy"],
            batch["sx"],
            batch["sy"],
        )
    if "data" in batch:
        return (
            batch["data"],
            batch["masked_patch"],
            batch["rx_patch"],
            batch["ry_patch"],
            batch["sx_patch"],
            batch["sy_patch"],
        )
    return None
