"""
SEG-Y configuration loader (YAML-based).

Loads byte position presets from ``segy_config.yaml`` and exposes them as a
module-level active config. All modules that need SEG-Y header byte positions
should import from here instead of hardcoding them.

Usage::

    import segy_config
    segy_config.load_config("field1031")   # select preset (or set env SEGY_CONFIG)
    bp = segy_config.get_byte_pos()       # {"shot_line": 17, "shot_x": 73, ...}

    # Backward-compatible module-level constants:
    from segy_config import SEGY_BYTE_POS, KEY_COLUMNS, BYTE_POS_SELF_COMPUTED

Presets are defined in ``segy_config.yaml`` alongside this file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

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


def _set_active(name: str, byte_pos: dict) -> None:
    global _active_name, _active_byte_pos
    _active_name = name
    _active_byte_pos = dict(byte_pos)


# Auto-load from env or default to first preset
_DEFAULT = os.environ.get("SEGY_CONFIG", "field1031")
if _DEFAULT in _PRESETS:
    _set_active(_DEFAULT, _PRESETS[_DEFAULT])
else:
    # Fall back to first preset in file
    first = next(iter(_PRESETS.keys()))
    _set_active(first, _PRESETS[first])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(name: str) -> dict:
    """Load a named byte-position preset and set it as active.

    Returns the byte_pos dict for the selected preset.
    """
    preset = _PRESETS.get(name)
    if preset is None:
        available = ", ".join(_PRESETS.keys())
        raise ValueError(f"Unknown SEG-Y config {name!r}. Available: {available}")
    _set_active(name, preset)
    return dict(preset)


def get_byte_pos() -> Dict[str, int]:
    """Return a copy of the currently active byte position dict."""
    return dict(_active_byte_pos)


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
        _set_active(_active_name, _PRESETS[_active_name])


# ---------------------------------------------------------------------------
# Module-level constants (backward-compatible with old hardcoded values)
# ---------------------------------------------------------------------------

# Convenience: compute the self_computed / segc3 byte pos subset
BYTE_POS_SELF_COMPUTED = {
    "shot_x": 73,
    "shot_y": 77,
    "rec_x": 81,
    "rec_y": 85,
}

# Tuple of (shot_line, shot_stake, recv_line, recv_stake) — used as
# composite trace key by infer_cli, segy_utils, etc.
KEY_COLUMNS: Tuple[str, ...] = ("shot_line", "shot_stake", "recv_line", "recv_stake")

# Sort order for traces in convert_tool
SORT_KEYS = ["recv_line", "recv_stake", "shot_line", "shot_stake"]

# Coordinate column order used by dataset: {field_name: axis_index}
COORD_COL = {"sx": 0, "sy": 1, "rx": 2, "ry": 3}

# Default trace sort order for patches (dataset.py)
TRACE_SORT_KEYS: Tuple[str, ...] = ("offset", "azimuth")


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

# Default metric weights for 4-axis spatial distance (sampler_utils.py)
METRIC_WEIGHTS = [1.0, 1.0, 0.5, 0.5]


def print_info() -> None:
    """Debug helper: print active config state."""
    print(f"[segy_config] active preset: {_active_name!r}")
    print(f"[segy_config] byte positions: {_active_byte_pos}")
    print(f"[segy_config] available presets: {list_presets()}")
