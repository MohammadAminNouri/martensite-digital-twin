"""
martwin.io.ctf_reader
======================
Native reader for Oxford Instruments / HKL Channel 5 .ctf files.

CTF format summary
------------------
The file is plain ASCII with two sections:

1. HEADER  — key-value lines, terminated by the line "Phases" followed by
             phase blocks, terminated by the column-header line:
             "Phase\tX\tY\tBands\tError\tEuler1\tEuler2\tEuler3\tMAD\tBC\tBS"

2. DATA    — one tab-delimited row per pixel, columns as above.

Key header tokens (case-sensitive):
    Channel Text File      — magic identifier on line 1
    Prj                    — project name
    Author                 — operator name
    JobMode                — Grid / HexGrid
    XCells / YCells        — map dimensions (integer)
    XStep  / YStep         — step size in µm (float)
    AcqE1 / AcqE2 / AcqE3 — acquisition Euler angles (degrees)
    Phases                 — number of phases (integer)

Per-phase block (one per phase, between "Phases N" and the column header):
    Line 1: "lattice-a;lattice-b;lattice-c;alpha;beta;gamma\tPhaseName\t..."
    Line 2: Laue group index (integer)
    Line 3: Space group number

Data columns (tab-delimited):
    Phase  X  Y  Bands  Error  Euler1  Euler2  Euler3  MAD  BC  BS
    Euler angles are in DEGREES in CTF — converted to radians here.
    X/Y are in µm.

Coordinate frame
----------------
Oxford HKL defines x along the scan fast axis (→) and y along the slow axis (↓).
We store exactly as-is; the user can apply reference frame transformations later.

References
----------
DREAM3D-NX ReadCtfDataFilter documentation (2024)
OOF2 HKLreader specification (NIST)
DefDAP ctf reader source (github.com/MechMicroMan/DefDAP)
"""

from __future__ import annotations

import pathlib
import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ebsd_data import EBSDData, Phase

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_CTF_MAGIC = "Channel Text File"

# Column indices in the data section (0-based)
_COL_PHASE  = 0
_COL_X      = 1
_COL_Y      = 2
_COL_BANDS  = 3
_COL_ERROR  = 4
_COL_EULER1 = 5
_COL_EULER2 = 6
_COL_EULER3 = 7
_COL_MAD    = 8
_COL_BC     = 9
_COL_BS     = 10

_DATA_NCOLS_MIN = 11   # minimum columns expected


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

def _parse_ctf_header(lines: List[str]) -> Tuple[Dict, List[Phase], int]:
    """
    Parse the CTF header section.

    Parameters
    ----------
    lines : list of str
        All lines of the file (including the data section).

    Returns
    -------
    meta : dict
        All scalar header key–value pairs.
    phases : list of Phase
        One Phase object per phase block.
    data_start_line : int
        Index of the first data line (after the column header).
    """
    meta: Dict[str, str] = {}
    phases: List[Phase] = []
    data_start_line: int = 0

    # Validate magic
    if not lines[0].strip().startswith(_CTF_MAGIC):
        warnings.warn(
            f"First line does not start with '{_CTF_MAGIC}'. "
            f"File may not be a valid CTF. Proceeding anyway.",
            UserWarning,
        )

    i = 0
    n_phases_expected = 0

    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # Detect column header line — marks end of header
        if line.startswith("Phase") and "Euler1" in line:
            data_start_line = i + 1
            break

        # Key–value lines: "Key\tValue" or "Key Value"
        # Split on first tab or first run of spaces
        parts = re.split(r"\t|(?<=\S) {2,}", line, maxsplit=1)
        key = parts[0].strip()
        value = parts[1].strip() if len(parts) > 1 else ""

        if key == "Phases":
            # Next block: n_phases phase descriptions
            n_phases_expected = int(value) if value.isdigit() else 0
            i += 1
            for phase_idx in range(1, n_phases_expected + 1):
                phase, i = _parse_phase_block(lines, i, phase_idx)
                phases.append(phase)
            continue
        else:
            meta[key] = value

        i += 1

    return meta, phases, data_start_line


def _parse_phase_block(
    lines: List[str], start: int, phase_id: int
) -> Tuple[Phase, int]:
    """
    Parse one phase block starting at `start`.

    CTF phase block format (3 lines):
        Line 0:  "a;b;c;alpha;beta;gamma\\tPhaseName\\tinfo..."
        Line 1:  Laue group index (integer)
        Line 2:  Space group number (integer)

    Returns
    -------
    Phase, next_line_index
    """
    i = start

    # --- Line 0: lattice parameters and name ---
    line0 = lines[i].strip() if i < len(lines) else ""
    i += 1

    name = "Unknown"
    lattice_params = None

    if line0:
        tab_parts = line0.split("\t")
        if tab_parts:
            # Lattice params in first tab-part: "a;b;c;alpha;beta;gamma"
            lp_str = tab_parts[0].strip()
            lp_parts = re.split(r"[;,\s]+", lp_str)
            if len(lp_parts) >= 6:
                try:
                    lattice_params = np.array(
                        [float(v) for v in lp_parts[:6]], dtype=np.float64
                    )
                except ValueError:
                    lattice_params = None
            if len(tab_parts) > 1:
                name = tab_parts[1].strip()

    # --- Line 1: Laue group ---
    laue_raw = lines[i].strip() if i < len(lines) else ""
    i += 1
    laue_map = {
        "-1": "-1", "1": "-1", "2": "2/m", "3": "mmm", "4": "4/m",
        "5": "4/mmm", "6": "-3", "7": "-3m", "8": "6/m",
        "9": "6/mmm", "10": "m-3", "11": "m-3m",
    }
    laue_group = laue_map.get(laue_raw.strip(), laue_raw.strip())

    # --- Line 2: Space group ---
    sg_raw = lines[i].strip() if i < len(lines) else ""
    i += 1
    try:
        space_group = int(sg_raw) if sg_raw.isdigit() else None
    except ValueError:
        space_group = None

    phase = Phase(
        phase_id=phase_id,
        name=name,
        space_group=space_group,
        lattice_params=lattice_params,
        laue_group=laue_group,
        color=None,
    )
    return phase, i


# ---------------------------------------------------------------------------
# Data section parsing
# ---------------------------------------------------------------------------

def _parse_ctf_data(
    lines: List[str],
    data_start: int,
    n_expected: int,
) -> np.ndarray:
    """
    Parse the tab-delimited data block into a (N, 11+) float array.

    Parameters
    ----------
    lines : list of str
    data_start : int
        Index of first data line.
    n_expected : int
        Expected number of pixels (XCells * YCells); used for pre-allocation.

    Returns
    -------
    (N, ncols) float64 ndarray
    """
    data_lines = [
        l for l in lines[data_start:] if l.strip() and not l.startswith(";")
    ]

    if not data_lines:
        raise ValueError("CTF file contains no data rows.")

    # Detect actual column count from first row
    first_row = data_lines[0].split("\t")
    ncols = len(first_row)
    if ncols < _DATA_NCOLS_MIN:
        raise ValueError(
            f"CTF data has only {ncols} columns; expected at least {_DATA_NCOLS_MIN}. "
            f"First row: {data_lines[0]!r}"
        )

    # Use numpy's loadtxt for speed — write to a temp buffer
    try:
        arr = np.array([list(map(float, l.split("\t"))) for l in data_lines], dtype=np.float64)
    except ValueError as exc:
        # Fall back to line-by-line parsing (handles ragged rows)
        warnings.warn(
            f"Fast numpy parse failed ({exc}). Falling back to row-by-row parse.",
            UserWarning,
        )
        arr = _parse_ctf_data_slow(data_lines, ncols)

    if arr.ndim == 1:
        arr = arr[np.newaxis, :]   # single-row file

    return arr


def _parse_ctf_data_slow(data_lines: List[str], ncols: int) -> np.ndarray:
    """Robust row-by-row parser for malformed CTF data sections."""
    rows = []
    for ln, line in enumerate(data_lines, 1):
        parts = line.strip().split("\t")
        try:
            row = [float(p) for p in parts[:ncols]]
            # Pad with zeros if short row
            while len(row) < ncols:
                row.append(0.0)
            rows.append(row)
        except ValueError:
            warnings.warn(f"Skipping malformed CTF data row {ln}: {line!r}", UserWarning)
    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Public reader
# ---------------------------------------------------------------------------

def load_ctf(
    path: str | pathlib.Path,
    validate: bool = True,
    max_mad_deg: Optional[float] = None,
    degrees: bool = True,
) -> EBSDData:
    """
    Load an Oxford Instruments / HKL Channel 5 .ctf file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the .ctf file.
    validate : bool
        If True, run sanity checks on the parsed data and warn on anomalies.
    max_mad_deg : float or None
        If given, pixels with MAD > max_mad_deg are marked as non-indexed
        (phase_id set to 0) after loading.
    degrees : bool
        CTF Euler angles are always in degrees; they are always converted to
        radians internally.  This flag is kept for API consistency.

    Returns
    -------
    EBSDData
        Fully populated data container.

    Raises
    ------
    FileNotFoundError : if the file does not exist.
    ValueError        : if the file cannot be parsed.

    Examples
    --------
    >>> from martwin.io import load_ctf
    >>> data = load_ctf("my_scan.ctf")
    >>> print(data)
    >>> R = data.as_rotation_matrices()   # (N, 3, 3) rotation matrices
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CTF file not found: {path}")

    # Read all lines with UTF-8; fall back to latin-1 for legacy files
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")

    lines = text.splitlines()
    if not lines:
        raise ValueError(f"CTF file is empty: {path}")

    # ---- Parse header ----
    meta, phases, data_start = _parse_ctf_header(lines)

    # Extract geometry scalars
    def _get(key: str, default, cast=str):
        v = meta.get(key, "")
        try:
            return cast(v) if v else default
        except (ValueError, TypeError):
            return default

    n_cols   = _get("XCells", 0, int)
    n_rows   = _get("YCells", 0, int)
    x_step   = _get("XStep",  0.0, float)
    y_step   = _get("YStep",  0.0, float)
    job_mode = _get("JobMode", "Grid", str).lower()
    grid_type = "hexagonal" if "hex" in job_mode else "square"

    n_expected = n_cols * n_rows

    # ---- Parse data ----
    arr = _parse_ctf_data(lines, data_start, n_expected)
    N = arr.shape[0]

    # Warn if pixel count doesn't match header
    if n_expected > 0 and N != n_expected:
        warnings.warn(
            f"CTF pixel count mismatch: header says {n_expected} "
            f"({n_cols}×{n_rows}), data has {N} rows.",
            UserWarning,
        )
        # Infer n_cols / n_rows from data if header was wrong
        if n_cols == 0 or n_rows == 0:
            n_cols = len(np.unique(arr[:, _COL_X]))
            n_rows = len(np.unique(arr[:, _COL_Y]))

    # ---- Assemble EBSDData ----
    # Euler angles: CTF stores in degrees — convert to radians
    euler1 = np.radians(arr[:, _COL_EULER1])
    euler2 = np.radians(arr[:, _COL_EULER2])
    euler3 = np.radians(arr[:, _COL_EULER3])

    # Clamp Euler angles to valid Bunge ranges
    euler1 = euler1 % (2 * np.pi)
    euler2 = np.clip(euler2, 0.0, np.pi)
    euler3 = euler3 % (2 * np.pi)

    phase_id = arr[:, _COL_PHASE].astype(np.int32)

    data = EBSDData(
        source_file=str(path),
        file_format="CTF",
        x_step=x_step,
        y_step=y_step,
        n_cols=n_cols,
        n_rows=n_rows,
        grid_type=grid_type,
        x=arr[:, _COL_X],
        y=arr[:, _COL_Y],
        euler1=euler1,
        euler2=euler2,
        euler3=euler3,
        phase_id=phase_id,
        bands=arr[:, _COL_BANDS].astype(np.int32),
        error=arr[:, _COL_ERROR].astype(np.int32),
        mad=arr[:, _COL_MAD].astype(np.float32),
        bc=arr[:, _COL_BC].astype(np.int32),
        bs=arr[:, _COL_BS].astype(np.float32),
        detector_intensity=None,
        phases={p.phase_id: p for p in phases},
        header_raw=meta,
    )

    # ---- Optional MAD filter ----
    if max_mad_deg is not None:
        data = data.filter_by_mad(max_mad_deg)

    # ---- Validation ----
    if validate:
        _validate_ctf(data)

    return data


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_ctf(data: EBSDData) -> None:
    """Emit warnings for common CTF data quality issues."""
    N = data.n_pixels

    if N == 0:
        warnings.warn("CTF: zero pixels loaded.", UserWarning)
        return

    # Indexed fraction
    frac = np.sum(data.is_indexed) / N
    if frac < 0.1:
        warnings.warn(
            f"CTF: only {frac:.1%} of pixels are indexed. "
            "Check for phase/error column mismatch.",
            UserWarning,
        )

    # Euler range sanity (after radian conversion)
    bad_euler = (
        (data.euler1 < 0) | (data.euler1 > 2 * np.pi + 0.01) |
        (data.euler2 < 0) | (data.euler2 > np.pi + 0.01) |
        (data.euler3 < 0) | (data.euler3 > 2 * np.pi + 0.01)
    )
    if np.any(bad_euler & data.is_indexed):
        n_bad = int(np.sum(bad_euler & data.is_indexed))
        warnings.warn(
            f"CTF: {n_bad} indexed pixels have Euler angles outside valid Bunge range.",
            UserWarning,
        )

    # MAD outliers
    indexed_mad = data.mad[data.is_indexed]
    if len(indexed_mad) > 0 and float(np.max(indexed_mad)) > 3.0:
        n_high = int(np.sum(indexed_mad > 3.0))
        warnings.warn(
            f"CTF: {n_high} pixels have MAD > 3.0°. Consider filtering with max_mad_deg.",
            UserWarning,
        )

    # Step size sanity
    if data.x_step <= 0 or data.y_step <= 0:
        warnings.warn(
            f"CTF: step size x={data.x_step} y={data.y_step} µm looks wrong.",
            UserWarning,
        )
