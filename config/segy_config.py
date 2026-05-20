"""
SEG-Y configuration loader (YAML-based).

Loads byte positions AND key-column definitions from ``segy_config.yaml``.
All modules that need per-dataset configuration should import getter functions
from here instead of hardcoding field names.

Usage::

    from config import segy_config
    segy_config.load_config("field1031")   # select preset (or set env SEGY_CONFIG)
    bp = segy_config.get_byte_pos()        # {"shot_line": 17, "shot_x": 73, ...}
    kc = segy_config.get_key_columns()     # ("shot_line", "shot_stake", ...)

    # Backward-compatible module-level constants (updated by load_config):
    from config.segy_config import KEY_COLUMNS, SORT_KEYS, TRACE_SORT_KEYS

Presets are defined in ``segy_config.yaml`` alongside this file.
Each preset is a dict with shared keywords::

    byte_pos:       1-based byte positions in SEGY trace header
    key_columns:    Tuple forming the composite trace identity key
    sort_keys:      Sort order for output SEG-Y files
    trace_sort_keys: Default sort order for traces within a patch
    coord_col:      Mapping from coordinate name to axis index
    metric_weights: Default weights for spatial distance
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_YAML_PATH = Path(__file__).resolve().parent / "segy_config.yaml"


# ---------------------------------------------------------------------------
# Load YAML
# ---------------------------------------------------------------------------

def _load_yaml() -> Dict[str, dict]:
    with open(_YAML_PATH, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{_YAML_PATH} must contain a top-level mapping")
    return data


_PRESETS: Dict[str, dict] = _load_yaml()


# ---------------------------------------------------------------------------
# Active config state
# ---------------------------------------------------------------------------

_active_name: str = ""
_active_byte_pos: Dict[str, int] = {}
_active_key_columns: Tuple[str, ...] = ()
_active_sort_keys: List[str] = []
_active_trace_sort_keys: Tuple[str, ...] = ()
_active_coord_col: Dict[str, int] = {}
_active_metric_weights: List[float] = []


# ---------------------------------------------------------------------------
# Module-level defaults (these are UPDATED by load_config / auto-init)
# ---------------------------------------------------------------------------

KEY_COLUMNS: Tuple[str, ...] = ("shot_line", "shot_stake", "recv_line", "recv_stake")
SORT_KEYS: List[str] = ["recv_line", "recv_stake", "shot_line", "shot_stake"]
TRACE_SORT_KEYS: Tuple[str, ...] = ("rx", "ry", "sx", "sy")
COORD_COL: Dict[str, int] = {"sx": 0, "sy": 1, "rx": 2, "ry": 3}
METRIC_WEIGHTS: List[float] = [1.0, 1.0, 0.5, 0.5]

# Convenience: the coordinate-mode byte positions (used in self_computed mode)
BYTE_POS_SELF_COMPUTED = {
    "shot_x": 73,
    "shot_y": 77,
    "rec_x": 81,
    "rec_y": 85,
}

# Number of coordinate dimensions
N_COORD_DIMS = 4

# H5 dataset keys written by convert_tool
DATASET_KEYS_FIXED = [
    "data", "sx", "sy", "rx", "ry",
    "delta", "t0",
    "shot_line", "shot_no", "recv_line", "recv_no",
    "shot_stake", "recv_stake", "cmp", "cmp_line", "offset",
    "trace_idx",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_preset_fields(preset: dict) -> dict:
    """Extract structured fields from a preset dict.

    Handles both new format (``byte_pos`` key present) and old flat format
    (entire dict is byte positions).
    """
    if "byte_pos" in preset:
        return {
            "byte_pos": dict(preset["byte_pos"]),
            "key_columns": tuple(preset.get("key_columns", KEY_COLUMNS)),
            "sort_keys": list(preset.get("sort_keys", SORT_KEYS)),
            "trace_sort_keys": tuple(preset.get("trace_sort_keys", TRACE_SORT_KEYS)),
            "coord_col": dict(preset.get("coord_col", COORD_COL)),
            "metric_weights": list(preset.get("metric_weights", METRIC_WEIGHTS)),
        }
    # Old flat format: entire dict is byte_pos
    return {
        "byte_pos": {k: int(v) for k, v in preset.items() if isinstance(v, (int, float))},
        "key_columns": KEY_COLUMNS,
        "sort_keys": list(SORT_KEYS),
        "trace_sort_keys": TRACE_SORT_KEYS,
        "coord_col": dict(COORD_COL),
        "metric_weights": list(METRIC_WEIGHTS),
    }


def _set_active(name: str, fields: dict) -> None:
    """Set all active state from extracted preset fields."""
    global _active_name, _active_byte_pos, _active_key_columns
    global _active_sort_keys, _active_trace_sort_keys, _active_coord_col
    global _active_metric_weights
    global KEY_COLUMNS, SORT_KEYS, TRACE_SORT_KEYS, COORD_COL, METRIC_WEIGHTS

    _active_name = name
    _active_byte_pos = fields["byte_pos"]
    _active_key_columns = fields["key_columns"]
    _active_sort_keys = fields["sort_keys"]
    _active_trace_sort_keys = fields["trace_sort_keys"]
    _active_coord_col = fields["coord_col"]
    _active_metric_weights = fields["metric_weights"]

    # Also update module-level constants for backward compat
    KEY_COLUMNS = _active_key_columns
    SORT_KEYS = list(_active_sort_keys)
    TRACE_SORT_KEYS = _active_trace_sort_keys
    COORD_COL = dict(_active_coord_col)
    METRIC_WEIGHTS = list(_active_metric_weights)


# Auto-load from env or default to first preset
_DEFAULT_PRESET = os.environ.get("SEGY_CONFIG", "field1031")
if _DEFAULT_PRESET in _PRESETS:
    _set_active(_DEFAULT_PRESET, _extract_preset_fields(_PRESETS[_DEFAULT_PRESET]))
else:
    first = next(iter(_PRESETS.keys()))
    _set_active(first, _extract_preset_fields(_PRESETS[first]))


# ---------------------------------------------------------------------------
# Public API — preset management
# ---------------------------------------------------------------------------

def load_config(name: str) -> dict:
    """Load a named preset and set it as active.

    Returns the full extracted fields dict for the selected preset.
    """
    preset = _PRESETS.get(name)
    if preset is None:
        available = ", ".join(_PRESETS.keys())
        raise ValueError(f"Unknown SEG-Y config {name!r}. Available: {available}")
    fields = _extract_preset_fields(preset)
    _set_active(name, fields)
    return fields


def get_active_name() -> str:
    """Return the name of the currently active preset."""
    return _active_name


def list_presets() -> list:
    """Return list of available preset names."""
    return list(_PRESETS.keys())


def reload_yaml() -> None:
    """Re-read segy_config.yaml (useful after editing)."""
    global _PRESETS
    _PRESETS = _load_yaml()
    if _active_name in _PRESETS:
        _set_active(_active_name, _extract_preset_fields(_PRESETS[_active_name]))


# ---------------------------------------------------------------------------
# Public API — getter functions (always returns current active values)
# ---------------------------------------------------------------------------

def get_byte_pos() -> Dict[str, int]:
    """Return a copy of the currently active byte position dict."""
    return dict(_active_byte_pos)


def get_key_columns() -> Tuple[str, ...]:
    """Return the active key-columns tuple (trace identity fields)."""
    return _active_key_columns


def get_sort_keys() -> List[str]:
    """Return the active SEG-Y output sort keys."""
    return list(_active_sort_keys)


def get_trace_sort_keys() -> Tuple[str, ...]:
    """Return the active trace sort keys for patch construction."""
    return _active_trace_sort_keys


def get_coord_col() -> Dict[str, int]:
    """Return the active coordinate column mapping."""
    return dict(_active_coord_col)


def get_metric_weights() -> List[float]:
    """Return the active metric weights for spatial distance."""
    return list(_active_metric_weights)


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------

def get_config_summary() -> str:
    """Return a compact one-line summary of the active config."""
    return (
        f"[segy_config] preset={_active_name!r} "
        f"key_columns={list(_active_key_columns)} "
        f"byte_keys={sorted(_active_byte_pos.keys())}"
    )


def print_info() -> None:
    """Print active config state for debugging."""
    print(f"[segy_config] active preset: {_active_name!r}")
    print(f"[segy_config] available presets: {list_presets()}")
    print(f"[segy_config] key_columns:      {list(_active_key_columns)}")
    print(f"[segy_config] sort_keys:        {_active_sort_keys}")
    print(f"[segy_config] trace_sort_keys:  {list(_active_trace_sort_keys)}")
    print(f"[segy_config] coord_col:        {_active_coord_col}")
    print(f"[segy_config] metric_weights:   {_active_metric_weights}")
    print(f"[segy_config] byte positions:   {_active_byte_pos}")
