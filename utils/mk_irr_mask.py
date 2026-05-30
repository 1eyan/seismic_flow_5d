"""
Generate irregular/mask/label data pairs from a single SEG-Y file for 5D training.
Randomly drops receiver grid cells to simulate irregular sampling.

Byte positions come from the active segy_config preset (not a standalone JSON).
Select preset via env:  SEGY_CONFIG=a002  (or export before running).

Outputs (per input file):
  - {name}_new.sgy/h5          — label (full data, sorted)
  - {name}_irr_{ratio}.sgy/h5  — irregular (kept traces only)
  - {name}_mask_{ratio}.sgy/h5 — mask  (missing traces zeroed, headers kept)
  - {name}_bool_mask_arr.npy   — boolean mask array
  - {name}_shot_xy_cut.dat / {name}_rcvs_xy_cut.dat  — coordinate QC
"""

import os
import json
import tempfile
import logging
import sys
import numpy as np
import pandas as pd
import h5py as h5
import seisio as sio

# Ensure project root is on sys.path (so "from config import segy_config" works)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import segy_config

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', force=True)
log = logging.getLogger("mk_irr_mask")

# ---------------------------------------------------------------------------
# Configuration — set via env vars
# ---------------------------------------------------------------------------
INPUT_SGY = os.environ.get("INPUT_SGY", "")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "")
MISSING_RATIO = float(os.environ.get("MISSING_RATIO", "0.3"))
PRESET = os.environ.get("SEGY_CONFIG", "a002")  # must exist in segy_config.yaml
MASK_DOMAIN = os.environ.get("MASK_DOMAIN", "receiver")  # receiver | shot | both | 4d

# Coordinate columns that need scalar correction
COORD_COLS = ["shot_x", "shot_y", "recv_x", "recv_y"]


# ---------------------------------------------------------------------------
# Build seisio thdef from segy_config
# ---------------------------------------------------------------------------

def _build_thdef() -> dict:
    """Build a seisio-compatible header definition dict from the active config."""
    byte_pos = segy_config.get_byte_pos()
    types = segy_config.get_thdef_types()
    thdef = {}
    for field, byte in byte_pos.items():
        thdef[field] = {"byte": byte, "type": types.get(field, "i")}
    return thdef


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _ensure_native_endian(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all big-endian numeric columns and index to native byte order."""
    for col in df.select_dtypes(include="number").columns:
        if df[col].dtype.byteorder == ">":
            df[col] = df[col].astype(df[col].dtype.newbyteorder("="))
    idx = df.index
    if hasattr(idx, "dtype") and idx.dtype.byteorder == ">":
        df.index = idx.astype(idx.dtype.newbyteorder("="))
    elif hasattr(idx, "levels"):
        new_levels = []
        for level in idx.levels:
            if level.dtype.byteorder == ">":
                new_levels.append(level.astype(level.dtype.newbyteorder("=")))
            else:
                new_levels.append(level)
        df.index = pd.MultiIndex(levels=new_levels, codes=idx.codes)
    return df


def _write_h5(h5_path: str, data: np.ndarray, headers_df: pd.DataFrame, group: str = "1551"):
    """Write trace data + headers into an H5 file."""
    with h5.File(h5_path, "w", locking=False) as h5f:
        g = h5f.create_group(group)
        g.create_dataset("data", data=data, dtype="f", compression="gzip")
        for col in headers_df.columns:
            g.create_dataset(col, data=headers_df[col].to_numpy(), dtype="f", compression="gzip")
        g.create_dataset("trace_idx", data=headers_df.index.to_numpy(), dtype="q", compression="gzip")


def _write_segy(sgy_path: str, traces: np.ndarray, ns: int, vsi: float, thdef_path: str):
    """Write a structured array of traces to a SEG-Y file."""
    sout = sio.output(sgy_path, ns=ns, vsi=vsi, endian=">", txtenc="ebcdic", thdef=thdef_path)
    sout.init()
    nwritten = sout.write_traces(traces=traces)
    log.info("Wrote %s — %d traces", sgy_path, nwritten)
    sout.finalize()


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------

def _grid_mask(headall: np.ndarray, line_key: str, stake_key: str,
               ratio: float) -> np.ndarray:
    """Build a 1D trace mask from a 2D (line, stake) grid."""
    uniq_lines = np.unique(headall[line_key])
    uniq_stakes = np.unique(headall[stake_key])
    grid = np.random.random((len(uniq_lines), len(uniq_stakes))) < ratio
    li = np.searchsorted(uniq_lines, headall[line_key])
    si = np.searchsorted(uniq_stakes, headall[stake_key])
    trace_mask = grid[li, si]
    actual = 1.0 - trace_mask.mean()
    log.info("  grid (%s, %s): %d lines × %d stakes, miss=%.4f",
             line_key, stake_key, len(uniq_lines), len(uniq_stakes), actual)
    return trace_mask


def _build_trace_mask(headall: np.ndarray, ratio: float, domain: str) -> np.ndarray:
    """Build 1D trace mask according to domain strategy.

    Parameters
    ----------
    domain : str
        ``"receiver"`` — 2D grid on (recv_line, recv_stake)
        ``"shot"``     — 2D grid on (shot_line, shot_stake)
        ``"both"``     — independent 2D grids on receiver & shot, union
        ``"4d"``       — 4D grid (shot_line, shot_stake, recv_line, recv_stake)
    """
    log.info("Mask domain=%s  target_ratio=%.2f", domain, ratio)
    if domain == "receiver":
        return _grid_mask(headall, "recv_line", "recv_stake", ratio)
    elif domain == "shot":
        return _grid_mask(headall, "shot_line", "shot_stake", ratio)
    elif domain == "both":
        m_recv = _grid_mask(headall, "recv_line", "recv_stake", ratio)
        m_shot = _grid_mask(headall, "shot_line", "shot_stake", ratio)
        trace_mask = m_recv | m_shot
        log.info("  union: miss=%.4f (recv alone=%.4f  shot alone=%.4f)",
                 1.0 - trace_mask.mean(),
                 1.0 - m_recv.mean(), 1.0 - m_shot.mean())
        return trace_mask
    elif domain == "4d":
        # Sample on unique (shot_line, shot_stake, recv_line, recv_stake) combos
        keys_4d = np.column_stack([
            headall["shot_line"], headall["shot_stake"],
            headall["recv_line"], headall["recv_stake"],
        ])
        uniq_keys, inv_idx = np.unique(keys_4d, axis=0, return_inverse=True)
        mask_4d = np.random.random(len(uniq_keys)) < ratio
        trace_mask = mask_4d[inv_idx]
        log.info("  4d grid: %d unique combos, miss=%.4f",
                 len(uniq_keys), 1.0 - trace_mask.mean())
        return trace_mask
    else:
        raise ValueError(f"Unknown MASK_DOMAIN={domain!r}; choose receiver|shot|both|4d")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not INPUT_SGY:
        log.error("INPUT_SGY is not set. Usage: INPUT_SGY=/path/to/data.sgy [SEGY_CONFIG=a002] bash run_mk_irr_mask.sh")
        return

    # ---- Load SEG-Y config preset ----
    try:
        segy_config.load_config(PRESET)
    except ValueError as e:
        log.error(e)
        return
    log.info("Active preset: %s", segy_config.get_active_name())
    log.info("Config summary: %s", segy_config.get_config_summary())

    sort_keys = segy_config.get_sort_keys()
    log.info("Sort keys: %s", sort_keys)

    # ---- Build thdef and write temp JSON for seisio ----
    thdef = _build_thdef()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(thdef, f, indent=2)
        tmp_thdef = f.name
    log.info("Generated temp thdef: %s (%d fields)", tmp_thdef, len(thdef))

    try:
        _run(thdef, tmp_thdef, sort_keys)
    finally:
        os.unlink(tmp_thdef)
        log.info("Cleaned up temp thdef: %s", tmp_thdef)


def _run(thdef: dict, tmp_thdef: str, sort_keys: list):
    """Core logic — wrapped in try/finally by main() for temp thdef cleanup."""
    # ---- Output paths ----
    stem = os.path.splitext(os.path.basename(INPUT_SGY))[0]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    h5_dir = os.path.join(OUTPUT_DIR, "h5")
    os.makedirs(h5_dir, exist_ok=True)

    segy_prefix = os.path.join(OUTPUT_DIR, f"{stem}-")
    h5_prefix = os.path.join(h5_dir, f"{stem}-")
    log.info("Output prefix: segy=%s  h5=%s", segy_prefix, h5_prefix)

    # ---- 1. Read SEG-Y & headers ----
    sio_obj = sio.input(INPUT_SGY, filetype="SGY", thdef=tmp_thdef)

    ntraces = sio_obj.nt
    nsamples = sio_obj.ns
    sampling_interval = sio_obj.vsi

    headall = sio_obj.read_all_headers()
    log.info("Traces=%d  Samples=%d  dt=%s", ntraces, nsamples, sampling_interval)

    # ---- 2. Build header DataFrame (field names = thdef keys) ----
    df = pd.DataFrame({col: headall[col] for col in thdef})
    df.index.name = "trace_idx"
    df = _ensure_native_endian(df)

    # Apply scalar correction to coordinates that need it
    scalar = df["scalar"].replace(0, 1).abs()
    for col in COORD_COLS:
        if col in df.columns:
            df[col] = df[col].div(scalar, axis=0)

    # ---- 3. Sort into regular 5D grid order ----
    df.sort_values(by=sort_keys, ascending=[True] * len(sort_keys),
                   na_position="last", inplace=True)
    sort_idx = df.index.to_numpy(dtype=np.intp)

    # ---- 4. Create missing trace mask ----
    trace_mask = _build_trace_mask(headall, MISSING_RATIO, MASK_DOMAIN)

    # ---- 5. Load all trace data in sorted order ----
    all_data = sio_obj.read_all_traces()[sort_idx]

    # ---- 6. Export label (complete data) ----
    log.info("Exporting label (full data) …")
    _write_h5(f"{h5_prefix}new.h5", all_data["data"], df)
    _write_segy(f"{segy_prefix}new.sgy", all_data, nsamples, sampling_interval, tmp_thdef)

    # ---- 7. Build irregular & mask variants ----
    irr_data = all_data[trace_mask]
    mask_data = all_data.copy()
    mask_data["data"][~trace_mask] = 0.0

    irr_df = df[trace_mask]
    suffix = f"{1 - MISSING_RATIO}"

    # ---- 8. Export irregular data ----
    log.info("Exporting irregular …")
    _write_h5(f"{h5_prefix}irr_{suffix}.h5", irr_data["data"], irr_df)
    _write_segy(f"{segy_prefix}irr_{suffix}.sgy", irr_data, nsamples, sampling_interval, tmp_thdef)

    # ---- 9. Export mask data (zeroed missing) ----
    log.info("Exporting mask …")
    _write_h5(f"{h5_prefix}mask_{suffix}.h5", mask_data["data"], df)
    _write_segy(f"{segy_prefix}mask_{suffix}.sgy", mask_data, nsamples, sampling_interval, tmp_thdef)

    # ---- 10. Save mask array ----
    np.save(f"{segy_prefix}bool_mask_arr", trace_mask)

    # ---- 11. Coordinate QC (grid coordinates: shot_x_grid / recv_x_grid) ----
    if "shot_x_grid" in thdef:
        sxy = np.unique(irr_data[["shot_x_grid", "shot_y_grid"]].astype(np.int32))
        sxy = sxy.view(np.int32).reshape(-1, 2)
        gxy = np.unique(irr_data[["recv_x_grid", "recv_y_grid"]].astype(np.int32))
        gxy = gxy.view(np.int32).reshape(-1, 2)
        np.savetxt(f"{segy_prefix}shot_xy_cut.dat", sxy, delimiter="\t", fmt="%d")
        np.savetxt(f"{segy_prefix}rcvs_xy_cut.dat", gxy, delimiter="\t", fmt="%d")
    else:
        log.warning("Skipping coordinate QC: shot_x_grid / recv_x_grid not in preset")

    # ---- 12. Binary QC dump ----
    irr_data["data"][:10000].astype("float32").tofile(f"{segy_prefix}tmp_5dgather_irr_{nsamples}.bin")
    mask_data["data"][:10000].astype("float32").tofile(f"{segy_prefix}tmp_5dgather_mask_{nsamples}.bin")

    log.info("All done.")


if __name__ == "__main__":
    main()
