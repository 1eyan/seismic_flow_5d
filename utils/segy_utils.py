"""SEG-Y I/O utilities (self-contained).

Provides:
    read_segy_headers()   – scan trace headers, return list of dicts
    read_segy_data()      – read trace data via segyio
    write_segy_data()     – write trace data into a copy of a template SEGY
    build_lookup()        – build (shot_line, shot_stake, recv_line, recv_stake) -> [trace_idx]
"""

from __future__ import annotations

import os
import shutil
import struct
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import segyio

# ---- Byte positions for SEG-Y headers ----
# Loaded from segy_config (YAML-based). To switch datasets:
#     segy_config.load_config("field1031")   or   segy_config.load_config("segc3")
try:
    from ..config.segy_config import get_byte_pos, KEY_COLUMNS
except ImportError:
    from config.segy_config import get_byte_pos, KEY_COLUMNS

SEGY_BYTE_POS = get_byte_pos()


def i32be(buf: bytes, pos_1b: int) -> int:
    return struct.unpack(">i", buf[pos_1b - 1 : pos_1b + 3])[0]


def bytes_per_sample(fmt: int) -> int:
    if fmt in (1, 2, 5):
        return 4
    if fmt == 3:
        return 2
    return 1


def scale_coord(v: int, scalar_raw: int) -> int:
    if scalar_raw == 0:
        return int(round(v))
    if scalar_raw > 0:
        return int(round(float(v) * float(scalar_raw)))
    return int(round(float(v) / float(-scalar_raw)))


def read_segy_headers(path: str, mode: str = "fixed") -> List[dict]:
    """Read all trace headers from a SEG-Y file.

    Args:
        path: SEG-Y file path.
        mode: "fixed" -> use shot_line/shot_no/recv_line/recv_no as key.
              "self_computed" -> use scaled (sy, sx, ry, rx) as key.

    Returns:
        List of header dicts with keys: trace_idx, key, coords, ns.
    """
    headers = []
    with open(path, "rb") as f:
        f.seek(3200)
        bin_header = f.read(400)
        ns_bin = struct.unpack(">H", bin_header[20:22])[0]
        bps = bytes_per_sample(struct.unpack(">H", bin_header[24:26])[0])
        f.seek(3600)
        trace_idx = 0
        while True:
            hdr = f.read(240)
            if len(hdr) < 240:
                break
            sx = i32be(hdr, SEGY_BYTE_POS["shot_x"])
            sy = i32be(hdr, SEGY_BYTE_POS["shot_y"])
            rx = i32be(hdr, SEGY_BYTE_POS["rec_x"])
            ry = i32be(hdr, SEGY_BYTE_POS["rec_y"])
            if mode == "fixed":
                key = (
                    i32be(hdr, SEGY_BYTE_POS["shot_line"]),
                    i32be(hdr, SEGY_BYTE_POS["shot_no"]),
                    i32be(hdr, SEGY_BYTE_POS["recv_line"]),
                    i32be(hdr, SEGY_BYTE_POS["recv_no"]),
                )
            else:
                scalar_raw = struct.unpack(">h", hdr[119:121])[0]
                key = (
                    scale_coord(sy, scalar_raw),
                    scale_coord(sx, scalar_raw),
                    scale_coord(ry, scalar_raw),
                    scale_coord(rx, scalar_raw),
                )
            ns_trace = struct.unpack(">H", hdr[114:116])[0] or ns_bin
            headers.append(
                {
                    "trace_idx": trace_idx,
                    "key": tuple(int(x) for x in key),
                    "coords": (sx, sy, rx, ry),
                    "ns": int(ns_trace),
                }
            )
            f.seek(int(ns_trace) * bps, os.SEEK_CUR)
            trace_idx += 1
    return headers


def read_segy_data(path: str) -> np.ndarray:
    """Read all trace data from a SEG-Y file. Returns (N, T) float32 array."""
    with segyio.open(path, "r", strict=False, ignore_geometry=True) as f:
        return f.trace.raw[:].astype(np.float32)


def write_segy_data(template_path: str, output_path: str, data: np.ndarray) -> None:
    """Copy template SEG-Y and overwrite its trace data.

    Args:
        template_path: source SEG-Y (headers copied).
        output_path: destination path.
        data: (N, T) float32 trace data.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template_path, output_path)
    with segyio.open(output_path, "r+", strict=False, ignore_geometry=True) as f:
        if f.tracecount != data.shape[0]:
            raise ValueError(
                f"SEGY tracecount={f.tracecount}, data traces={data.shape[0]}"
            )
        for i in range(f.tracecount):
            f.trace[i] = data[i].astype(np.float32)


def build_lookup(
    headers: Iterable[dict],
) -> Dict[Tuple[int, int, int, int], List[int]]:
    """Build (key -> [trace_idx, ...]) lookup from header list."""
    lookup: Dict[Tuple[int, int, int, int], List[int]] = defaultdict(list)
    for h in headers:
        lookup[h["key"]].append(int(h["trace_idx"]))
    return dict(lookup)
