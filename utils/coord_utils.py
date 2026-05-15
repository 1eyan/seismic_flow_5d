"""
Coordinate normalization utilities (standalone).

Central module for:
- Inferring coordinate normalization state from a dataset
- Computing RoPE omega based on physical wavelength
- Saving/loading coord_stats.json and rope_frequency_config.json
- Checking training/inference coordinate consistency

Usage:
    from queryctx_module.utils.coord_utils import build_coord_config, save_coord_config, load_coord_config

    # After creating DatasetH5_all_queryctx (which has .coord_stats attribute):
    coord_config = build_coord_config(dataset)

    # Save to disk:
    save_coord_config(coord_config, "/path/to/output_dir")
"""

import json
import os
import math
import warnings
from typing import Dict, Any, Optional, List

import numpy as np


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer_coord_normalization_from_dataset(dataset) -> Dict[str, Any]:
    """
    Inspect dataset.coord_stats and sample one item to determine the coordinate
    normalization state.

    Returns a 'coord_state' dict with keys:
        coord_norm_mode, has_phys_coords, has_norm_coords,
        coord_keys, coord_stats, coord_range, rope_input_coord_unit, warnings
    """
    state: Dict[str, Any] = {
        "task_mode": "unknown",
        "coord_range_source": "unknown",
        "coord_norm_mode": "unknown",
        "has_phys_coords": False,
        "has_norm_coords": False,
        "coord_keys": [],
        "coord_stats": {},
        "coord_range": None,
        "rope_input_coord_unit": "unknown",
        "warnings": [],
    }

    # 0. Infer task_mode and coord_range_source from dataset attributes
    state["task_mode"] = getattr(dataset, "task_mode", None)
    state["coord_range_source"] = getattr(dataset, "coord_range_source", None)

    if state["task_mode"] is None:
        cls_name = type(dataset).__name__
        if "interp" in cls_name.lower():
            state["task_mode"] = "interpolation"
        elif "reg" in cls_name.lower():
            state["task_mode"] = "regularization"
        else:
            state["task_mode"] = "unknown"

    if state["coord_range_source"] is None:
        has_regular = hasattr(dataset, "h5_data_regular") and dataset.h5_data_regular is not None
        has_irregular = hasattr(dataset, "h5_data") and dataset.h5_data is not None
        if has_regular and not has_irregular:
            state["coord_range_source"] = "regular_only"
        elif has_regular and has_irregular:
            state["coord_range_source"] = "regular_only"
        elif has_irregular:
            state["coord_range_source"] = "irregular_only"
        else:
            state["coord_range_source"] = "unknown"

    # 1. Try to get coord_stats from dataset
    if hasattr(dataset, "coord_stats") and dataset.coord_stats:
        state["coord_stats"] = _serialize_stats(dataset.coord_stats)
        state["has_norm_coords"] = True
        state["coord_norm_mode"] = "global_minmax"
        state["rope_input_coord_unit"] = "normalized"
    else:
        state["warnings"].append(
            "dataset has no coord_stats attribute; cannot determine normalization"
        )
        return state

    # 2. Inspect the normalization formula by checking a sample
    try:
        sample = dataset[0]
    except Exception as e:
        state["warnings"].append(f"Could not sample dataset[0]: {e}")
        return state

    for key in ("coords", "coords_norm", "coords_phys",
                "rx_patch", "ry_patch", "sx_patch", "sy_patch",
                "coord_stats", "coord_norm_mode"):
        if key in sample:
            state["coord_keys"].append(key)

    # 3. Determine coord range by sampling the normalized coords
    if "coords" in sample:
        coords = np.asarray(sample["coords"], dtype=np.float64)
    elif "rx_patch" in sample:
        rx = np.asarray(sample["rx_patch"], dtype=np.float64)
        ry = np.asarray(sample["ry_patch"], dtype=np.float64)
        sx = np.asarray(sample["sx_patch"], dtype=np.float64)
        sy = np.asarray(sample["sy_patch"], dtype=np.float64)
        coords = np.stack([rx, ry, sx, sy], axis=-1)
    else:
        state["warnings"].append("sample has no coord fields; cannot inspect range")
        return state

    c_min, c_max = float(coords.min()), float(coords.max())

    if -1.2 < c_min < -0.8 and 0.8 < c_max < 1.2:
        state["coord_range"] = [-1.0, 1.0]
    elif -0.2 < c_min < 0.2 and 0.0 < c_max < 1.2 and c_max > 0.5:
        state["coord_range"] = [0.0, 1.0]
    elif c_min < -100 or c_max > 100:
        state["coord_range"] = [c_min, c_max]
        if state["rope_input_coord_unit"] == "normalized":
            state["warnings"].append(
                f"coord_stats says normalized but sample range [{c_min:.1f}, {c_max:.1f}] "
                "looks like physical coordinates -- possible inconsistency"
            )
    else:
        state["coord_range"] = [c_min, c_max]

    state["has_phys_coords"] = "coords_phys" in sample

    if "coord_norm_mode" in sample:
        state["coord_norm_mode"] = str(sample["coord_norm_mode"])

    if state["coord_norm_mode"] == "patch_minmax":
        state["rope_input_coord_unit"] = "patch_normalized"
        state["warnings"].append(
            "patch min-max normalization detected: physical frequency "
            "interpretation is disabled. Different patches have different "
            "physical scales. Legacy mode only."
        )
    elif state["coord_norm_mode"] in ("global_minmax", "global_centered"):
        state["rope_input_coord_unit"] = "normalized"
    elif state["coord_norm_mode"] == "physical":
        state["rope_input_coord_unit"] = "physical"

    return state


# ---------------------------------------------------------------------------
# Lambda_phys inference
# ---------------------------------------------------------------------------

def infer_lambda_phys_from_coord_stats(coord_stats: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Auto-compute physical wavelengths from dataset grid steps using Nyquist sampling theorem.

    lambda_phys represents the minimum resolvable spatial wavelength.
    By Nyquist: lambda_min = 2 * grid_step.
    We take the more conservative (smaller) of source and receiver grid steps per axis.

    Args:
        coord_stats: dict with optional grid_step_sx, grid_step_sy, grid_step_rx, grid_step_ry

    Returns:
        dict with keys lambda_phys_x, lambda_phys_y, and any warnings
    """
    result: Dict[str, Any] = {
        "lambda_phys_x": None,
        "lambda_phys_y": None,
        "grid_step_sx": coord_stats.get("grid_step_sx"),
        "grid_step_sy": coord_stats.get("grid_step_sy"),
        "grid_step_rx": coord_stats.get("grid_step_rx"),
        "grid_step_ry": coord_stats.get("grid_step_ry"),
        "warnings": [],
    }

    gs_sx = result["grid_step_sx"]
    gs_sy = result["grid_step_sy"]
    gs_rx = result["grid_step_rx"]
    gs_ry = result["grid_step_ry"]

    valid_x = [v for v in (gs_sx, gs_rx) if v is not None and v > 0]
    if valid_x:
        result["lambda_phys_x"] = 2.0 * min(valid_x)
    else:
        result["warnings"].append(
            "No valid grid_step_sx or grid_step_rx in coord_stats; "
            "cannot auto-compute lambda_phys_x."
        )

    valid_y = [v for v in (gs_sy, gs_ry) if v is not None and v > 0]
    if valid_y:
        result["lambda_phys_y"] = 2.0 * min(valid_y)
    else:
        result["warnings"].append(
            "No valid grid_step_sy or grid_step_ry in coord_stats; "
            "cannot auto-compute lambda_phys_y."
        )

    return result


# ---------------------------------------------------------------------------
# Omega computation
# ---------------------------------------------------------------------------

def compute_rope_omega(
    coord_stats: Dict[str, float],
    coord_norm_mode: str,
    lambda_phys_x: float,
    lambda_phys_y: float,
) -> Dict[str, Any]:
    """
    Compute RoPE angular frequencies based on coordinate normalization mode.

    Args:
        coord_stats: per-axis min/max dict
        coord_norm_mode: "global_minmax" | "physical" | "patch_minmax" | ...
        lambda_phys_x: physical wavelength in meters for x-axis
        lambda_phys_y: physical wavelength in meters for y-axis

    Returns dict with omega_x, omega_y, mode, and any warnings.

    Case A (normalized coords -> RoPE input):
        omega = pi * L / lambda_phys
        where L = coord_max_phys - coord_min_phys

    Case B (physical coords -> RoPE input):
        omega = 2*pi / lambda_phys

    Case C (patch_minmax):
        WARNING: physical frequency interpretation is disabled.
    """
    result: Dict[str, Any] = {
        "omega_x": None,
        "omega_y": None,
        "mode": coord_norm_mode,
        "warnings": [],
    }

    if coord_norm_mode == "patch_minmax":
        result["omega_x"] = float("nan")
        result["omega_y"] = float("nan")
        result["warnings"].append(
            "Patch min-max normalization: physical frequency interpretation "
            "is DISABLED. Omega set to NaN."
        )
        return result

    Lx = max(
        coord_stats.get("sx_max", 0) - coord_stats.get("sx_min", 0),
        coord_stats.get("rx_max", 0) - coord_stats.get("rx_min", 0),
    )
    Ly = max(
        coord_stats.get("sy_max", 0) - coord_stats.get("sy_min", 0),
        coord_stats.get("ry_max", 0) - coord_stats.get("ry_min", 0),
    )

    if Lx <= 0 or Ly <= 0:
        Lx = coord_stats.get("Lx", Lx)
        Ly = coord_stats.get("Ly", Ly)

    if Lx <= 0:
        Lx = 1.0
        result["warnings"].append("Lx <= 0, using fallback Lx=1.0")
    if Ly <= 0:
        Ly = 1.0
        result["warnings"].append("Ly <= 0, using fallback Ly=1.0")

    if coord_norm_mode in ("global_minmax", "global_centered"):
        result["omega_x"] = math.pi * Lx / lambda_phys_x
        result["omega_y"] = math.pi * Ly / lambda_phys_y
        result["_formula"] = "omega = pi * L / lambda_phys"
        result["_L"] = {"x": Lx, "y": Ly}
        result["_lambda_phys"] = {"x": lambda_phys_x, "y": lambda_phys_y}
    elif coord_norm_mode == "physical":
        result["omega_x"] = 2.0 * math.pi / lambda_phys_x
        result["omega_y"] = 2.0 * math.pi / lambda_phys_y
        result["_formula"] = "omega = 2*pi / lambda_phys"
        result["_lambda_phys"] = {"x": lambda_phys_x, "y": lambda_phys_y}
    else:
        result["omega_x"] = float("nan")
        result["omega_y"] = float("nan")
        result["warnings"].append(
            f"Unknown coord_norm_mode={coord_norm_mode!r}; "
            "cannot compute physical omega."
        )

    return result


# ---------------------------------------------------------------------------
# Unified config builder
# ---------------------------------------------------------------------------

def build_coord_config(
    dataset,
    lambda_phys_x: Optional[float] = None,
    lambda_phys_y: Optional[float] = None,
) -> Dict[str, Any]:
    """
    One-call entry point: infer coord state from dataset, compute omega,
    and return a unified coord_config dict.

    Args:
        dataset: torch Dataset instance with coord_stats attribute
        lambda_phys_x: physical wavelength in meters for x-axis.
            If None, auto-computed from grid_step_sx/grid_step_rx via Nyquist.
        lambda_phys_y: physical wavelength in meters for y-axis.
            If None, auto-computed from grid_step_sy/grid_step_ry via Nyquist.

    Returns:
        coord_config dict ready for saving and passing to model
    """
    state = infer_coord_normalization_from_dataset(dataset)

    lambda_info = infer_lambda_phys_from_coord_stats(state["coord_stats"])

    if lambda_phys_x is None:
        lambda_phys_x = lambda_info["lambda_phys_x"]
        if lambda_phys_x is None:
            lambda_phys_x = 200.0
            state["warnings"].append(
                "lambda_phys_x not provided and cannot auto-compute from grid steps; "
                "falling back to default 200.0 m"
            )
    if lambda_phys_y is None:
        lambda_phys_y = lambda_info["lambda_phys_y"]
        if lambda_phys_y is None:
            lambda_phys_y = 200.0
            state["warnings"].append(
                "lambda_phys_y not provided and cannot auto-compute from grid steps; "
                "falling back to default 200.0 m"
            )

    omega = compute_rope_omega(
        state["coord_stats"],
        state["coord_norm_mode"],
        lambda_phys_x,
        lambda_phys_y,
    )

    config = {
        "task_mode": state["task_mode"],
        "coord_range_source": state["coord_range_source"],
        "coord_norm_mode": state["coord_norm_mode"],
        "rope_input_coord_unit": state["rope_input_coord_unit"],
        "coord_range": state["coord_range"],
        "has_phys_coords": state["has_phys_coords"],
        "coord_stats": state["coord_stats"],
        "lambda_phys": {"x": lambda_phys_x, "y": lambda_phys_y},
        "L": omega.get("_L", {"x": None, "y": None}),
        "omega_mode": "physical" if omega["mode"] != "patch_minmax" else "disabled",
        "omega": {"x": omega["omega_x"], "y": omega["omega_y"]},
        "omega_formula": omega.get("_formula", "unknown"),
        "warnings": state["warnings"] + omega["warnings"] + lambda_info.get("warnings", []),
    }

    config["_summary"] = _build_summary(config)
    return config


# ---------------------------------------------------------------------------
# Config I/O
# ---------------------------------------------------------------------------

def save_coord_config(config: Dict[str, Any], output_dir: str) -> None:
    """Save coord_stats.json and rope_frequency_config.json to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    stats = config.get("coord_stats", {})
    stats_path = os.path.join(output_dir, "coord_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=_json_default)
    print(f"[coord_utils] Saved coord_stats.json to {stats_path}")

    freq_config = {
        "task_mode": config["task_mode"],
        "coord_range_source": config["coord_range_source"],
        "coord_norm_mode": config["coord_norm_mode"],
        "rope_input_coord_unit": config["rope_input_coord_unit"],
        "coord_range": config["coord_range"],
        "omega_mode": config["omega_mode"],
        "omega": config["omega"],
        "omega_formula": config["omega_formula"],
        "lambda_phys": config["lambda_phys"],
        "L": config["L"],
        "warnings": config.get("warnings", []),
    }
    freq_path = os.path.join(output_dir, "rope_frequency_config.json")
    with open(freq_path, "w", encoding="utf-8") as f:
        json.dump(freq_config, f, indent=2, ensure_ascii=False, default=_json_default)
    print(f"[coord_utils] Saved rope_frequency_config.json to {freq_path}")


def load_coord_config(dir_path: str) -> Optional[Dict[str, Any]]:
    """
    Load coord configuration from a directory containing
    coord_stats.json and rope_frequency_config.json.

    Returns None if files are not found.
    """
    stats_path = os.path.join(dir_path, "coord_stats.json")
    freq_path = os.path.join(dir_path, "rope_frequency_config.json")

    if not os.path.exists(stats_path) and not os.path.exists(freq_path):
        return None

    config = {}
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            config["coord_stats"] = json.load(f)
    if os.path.exists(freq_path):
        with open(freq_path, "r", encoding="utf-8") as f:
            freq = json.load(f)
            for k, v in freq.items():
                config[k] = v

    return config if config else None


# ---------------------------------------------------------------------------
# Consistency check
# ---------------------------------------------------------------------------

def check_coord_consistency(
    train_config: Dict[str, Any],
    infer_state: Dict[str, Any],
) -> List[str]:
    """
    Compare training coord config with inference dataset state.
    Returns list of warning strings (empty = all good).
    """
    warnings_list: List[str] = []

    train_task = train_config.get("task_mode", "unknown")
    infer_task = infer_state.get("task_mode", "unknown")
    if train_task != infer_task:
        warnings_list.append(
            f"Train task_mode={train_task!r} != Infer task_mode={infer_task!r}"
        )

    train_src = train_config.get("coord_range_source", "unknown")
    infer_src = infer_state.get("coord_range_source", "unknown")
    if train_src != infer_src:
        warnings_list.append(
            f"Train coord_range_source={train_src!r} != "
            f"Infer coord_range_source={infer_src!r}"
        )

    train_mode = train_config.get("coord_norm_mode", "unknown")
    infer_mode = infer_state.get("coord_norm_mode", "unknown")

    if train_mode != infer_mode:
        warnings_list.append(
            f"Train coord_norm_mode={train_mode!r} != "
            f"Infer coord_norm_mode={infer_mode!r}"
        )

    train_unit = train_config.get("rope_input_coord_unit", "unknown")
    infer_unit = infer_state.get("rope_input_coord_unit", "unknown")

    if train_unit != infer_unit:
        warnings_list.append(
            f"Train rope_input_coord_unit={train_unit!r} != "
            f"Infer rope_input_coord_unit={infer_unit!r}"
        )

    train_stats = train_config.get("coord_stats", {})
    infer_stats = infer_state.get("coord_stats", {})
    for key in ("sx_min", "sx_max", "sy_min", "sy_max",
                "rx_min", "rx_max", "ry_min", "ry_max"):
        tv = train_stats.get(key)
        iv = infer_stats.get(key)
        if tv is not None and iv is not None:
            if abs(float(tv) - float(iv)) > max(abs(float(tv)), abs(float(iv)), 1.0) * 0.1:
                warnings_list.append(
                    f"coord_stats.{key} differs: train={tv:.1f}, infer={iv:.1f}"
                )

    return warnings_list


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_stats(stats: Dict[str, Any]) -> Dict[str, Optional[float]]:
    out = {}
    for k, v in stats.items():
        if v is None:
            out[k] = None
        elif hasattr(v, "item"):
            out[k] = float(v.item())
        else:
            out[k] = float(v)
    return out


def _json_default(obj):
    if obj is None:
        return None
    if hasattr(obj, "item"):
        val = obj.item()
        if isinstance(val, float) and math.isnan(val):
            return None
        return val
    if isinstance(obj, float) and math.isnan(obj):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _build_summary(config: Dict[str, Any]) -> str:
    lines = [
        "=" * 60,
        "Coordinate Normalization Summary",
        "=" * 60,
        f"  task_mode:             {config['task_mode']}",
        f"  coord_range_source:    {config['coord_range_source']}",
        f"  coord_norm_mode:       {config['coord_norm_mode']}",
        f"  rope_input_coord_unit: {config['rope_input_coord_unit']}",
        f"  coord_range:           {config['coord_range']}",
        f"  has_phys_coords:       {config['has_phys_coords']}",
        f"  omega_mode:            {config['omega_mode']}",
    ]

    stats = config.get("coord_stats", {})
    has_grid_steps = any(stats.get(k) is not None for k in
                         ("grid_step_sx", "grid_step_sy", "grid_step_rx", "grid_step_ry"))
    if has_grid_steps:
        lines.append("  --- Grid Steps ---")
        for k in ("grid_step_sx", "grid_step_sy", "grid_step_rx", "grid_step_ry"):
            v = stats.get(k)
            if v is not None:
                lines.append(f"  {k}: {v:.3f} m")

    if config["omega"]["x"] is not None and not (
        isinstance(config["omega"]["x"], float) and math.isnan(config["omega"]["x"])
    ):
        lines.append("  --- RoPE Omega ---")
        lines.append(f"  lambda_phys_x:         {config['lambda_phys']['x']:.3f} m")
        lines.append(f"  lambda_phys_y:         {config['lambda_phys']['y']:.3f} m")
        lines.append(f"  Lx (phys range):       {config['L'].get('x', 'N/A')}")
        lines.append(f"  Ly (phys range):       {config['L'].get('y', 'N/A')}")
        lines.append(f"  omega_x:               {config['omega']['x']:.6f}")
        lines.append(f"  omega_y:               {config['omega']['y']:.6f}")
        lines.append(f"  omega_formula:         {config['omega_formula']}")
    if config.get("warnings"):
        lines.append("  --- WARNINGS ---")
        for w in config["warnings"]:
            lines.append(f"  [WARN] {w}")
    lines.append("=" * 60)
    return "\n".join(lines)
