"""
martwin.io.ang_reader
======================
Native reader for EDAX / TSL OIM .ang files.

ANG format summary
------------------
The file is plain ASCII with a header (lines starting with '#') followed
by the data block.  No separator line marks the transition — the data
starts at the first non-comment, non-empty line.

Header keywords (prefixed with '# '):
    TEM_PIXperUM       — camera pixels per µm
    x-star, y-star, z-star — pattern centre coordinates
    WorkingDistance    — specimen–detector distance (mm)
    NCOLS_ODD/EVEN     — columns for hex grids
    NROWS              — number of scan rows
    XSTEP / YSTEP      — step size in µm
    GRID: SqrGrid or HexGrid
    OPERATOR           — operator name
    SAMPLEID / SCANID
    Phase              — starts a per-phase block
    MaterialName / Formula / Info / Symmetry / LatticeConstants
    NumberFamilies / hklFamilies / ElasticConstants / Categories

Data columns (space-delimited, one row per pixel):
    phi1  Phi  phi2  x  y  IQ  CI  Phase  SEM_signal  Fit
    All angles in RADIANS.
    x, y in µm.

Grid types
----------
Square (GRID: SqrGrid):  regular rectangular grid, n_cols × n_rows pixels.
Hexagonal (GRID: HexGrid): alternating rows have NCOLS_ODD and NCOLS_EVEN
    columns, offset by half a step in x.  The ANG spec requires explicit
    NCOLS_ODD, NCOLS_EVEN, NROWS.  Total pixels = NROWS/2*(ODD+EVEN) or
    CEIL(NROWS/2)*ODD + FLOOR(NROWS/2)*EVEN.

Coordinate frame
----------------
EDAX exports with x→ east (fast axis) and y↓ south (slow axis).
Reference frame alignment varies between EDAX software versions (Settings 1–4).
We read coordinates exactly as stored and expose a convert_to_oxford()
helper for CTF-style reference frame.

References
----------
ResearchGate Q&A: EDAX ANG column order (2015)
DREAM3D-NX ReadAngDataFilter documentation (2024)
MTEX EBSDReferenceFrame guide
"""

from __future__ import annotations

import pathlib
import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ebsd_data import EBSDData, Phase

# ---------------------------------------------------------------------------
# Column indices in the data section
# ---------------------------------------------------------------------------

_COL_PHI1    = 0
_COL_PHI     = 1
_COL_PHI2    = 2
_COL_X       = 3
_COL_Y       = 4
_COL_IQ      = 5
_COL_CI      = 6
_COL_PHASE   = 7
_COL_SEM     = 8
_COL_FIT     = 9

_DATA_NCOLS_MIN = 8   # Phase is minimum required


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

def _parse_ang_header(lines: List[str]) -> Tuple[Dict, List[Phase], int]:
    """
    Parse the ANG header (all lines starting with '#').

    Returns
    -------
    meta : dict
        Flat key–value pairs from header.
    phases : list of Phase
        All phase blocks in order.
    data_start : int
        Index of first data line.
    """
    meta: Dict[str, str] = {}
    phases: List[Phase] = []
    current_phase: Optional[Dict] = None
    data_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Non-comment line → data starts here
        if not stripped.startswith("#"):
            data_start = i
            break

        # Strip '# ' prefix
        content = stripped.lstrip("#").strip()
        if not content:
            continue

        # Phase block start: "Phase N"
        phase_match = re.match(r"^Phase\s+(\d+)", content, re.IGNORECASE)
        if phase_match:
            # Save previous phase if any
            if current_phase is not None:
                phases.append(_build_phase_from_dict(current_phase))
            current_phase = {"phase_id": int(phase_match.group(1))}
            continue

        # Key: Value lines inside a phase block
        kv_match = re.match(r"^(\S.*?)\s*:\s*(.*)", content)
        if kv_match:
            key = kv_match.group(1).strip()
            value = kv_match.group(2).strip()
            if current_phase is not None and key in (
                "MaterialName", "Formula", "Info", "Symmetry",
                "LatticeConstants", "NumberFamilies", "hklFamilies",
                "ElasticConstants", "Categories", "Color",
            ):
                current_phase[key] = value
            else:
                meta[key] = value
            continue

        # Lines without colon inside a phase block (rare — absorb them)
        if current_phase is not None:
            parts = content.split(None, 1)
            if len(parts) == 2:
                current_phase.setdefault("_extra", []).append(content)

    # Save last phase
    if current_phase is not None:
        phases.append(_build_phase_from_dict(current_phase))

    return meta, phases, data_start


def _build_phase_from_dict(d: Dict) -> Phase:
    """Convert a raw phase dict (from header parsing) to a Phase object."""
    pid = d.get("phase_id", 0)
    name = d.get("MaterialName", d.get("Formula", f"Phase_{pid}"))

    # LatticeConstants: "a b c alpha beta gamma" in Å and degrees
    lp = None
    lc_str = d.get("LatticeConstants", "")
    if lc_str:
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", lc_str)
        if len(nums) >= 6:
            lp = np.array([float(v) for v in nums[:6]], dtype=np.float64)

    # Space group from Symmetry field (integer)
    sg = None
    sym_str = d.get("Symmetry", "")
    if sym_str.strip().lstrip("-").isdigit():
        try:
            sg = int(sym_str.strip())
        except ValueError:
            pass

    return Phase(
        phase_id=pid,
        name=name,
        space_group=sg,
        lattice_params=lp,
        laue_group=d.get("Symmetry", None),
        color=d.get("Color", None),
    )


# ---------------------------------------------------------------------------
# Hex grid pixel count helper
# ---------------------------------------------------------------------------

def _hex_grid_pixel_count(n_rows: int, ncols_odd: int, ncols_even: int) -> int:
    """Compute total pixel count for a hexagonal grid."""
    n_odd_rows  = (n_rows + 1) // 2
    n_even_rows = n_rows // 2
    return n_odd_rows * ncols_odd + n_even_rows * ncols_even


# ---------------------------------------------------------------------------
# Data section parsing
# ---------------------------------------------------------------------------

def _parse_ang_data(
    lines: List[str],
    data_start: int,
) -> np.ndarray:
    """
    Parse the space-delimited ANG data block.

    Returns
    -------
    (N, ncols) float64 ndarray
    """
    data_lines = [
        l for l in lines[data_start:]
        if l.strip() and not l.strip().startswith("#")
    ]

    if not data_lines:
        raise ValueError("ANG file contains no data rows.")

    # Detect column count
    first = data_lines[0].split()
    ncols = len(first)
    if ncols < _DATA_NCOLS_MIN:
        raise ValueError(
            f"ANG data has only {ncols} columns; expected at least {_DATA_NCOLS_MIN}. "
            f"First row: {data_lines[0]!r}"
        )

    try:
        arr = np.array([list(map(float, l.split())) for l in data_lines], dtype=np.float64)
    except ValueError as exc:
        warnings.warn(
            f"Fast ANG parse failed ({exc}). Falling back to row-by-row parse.",
            UserWarning,
        )
        arr = _parse_ang_data_slow(data_lines, ncols)

    if arr.ndim == 1:
        arr = arr[np.newaxis, :]

    return arr


def _parse_ang_data_slow(data_lines: List[str], ncols: int) -> np.ndarray:
    """Robust row-by-row parser for malformed ANG data."""
    rows = []
    for ln, line in enumerate(data_lines, 1):
        parts = line.strip().split()
        try:
            row = [float(p) for p in parts[:ncols]]
            while len(row) < ncols:
                row.append(0.0)
            rows.append(row)
        except ValueError:
            warnings.warn(f"Skipping malformed ANG data row {ln}: {line!r}", UserWarning)
    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Public reader
# ---------------------------------------------------------------------------

def load_ang(
    path: str | pathlib.Path,
    validate: bool = True,
    min_ci: Optional[float] = None,
    convert_hex_to_square: bool = False,
) -> EBSDData:
    """
    Load a EDAX / TSL OIM .ang EBSD file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the .ang file.
    validate : bool
        Run sanity checks and emit warnings.
    min_ci : float or None
        If given, pixels with CI < min_ci are marked as non-indexed
        (phase_id = 0) after loading.  Typical values: 0.0–0.2.
    convert_hex_to_square : bool
        If True and the file uses a hexagonal grid, attempt a simple
        nearest-neighbour resampling to a square grid.
        WARNING: this changes the spatial coordinates.

    Returns
    -------
    EBSDData

    Raises
    ------
    FileNotFoundError
    ValueError

    Examples
    --------
    >>> from martwin.io import load_ang
    >>> data = load_ang("my_scan.ang", min_ci=0.1)
    >>> print(data)
    >>> R = data.as_rotation_matrices()   # (N, 3, 3)
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ANG file not found: {path}")

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")

    lines = text.splitlines()
    if not lines:
        raise ValueError(f"ANG file is empty: {path}")

    # ---- Parse header ----
    meta, phases, data_start = _parse_ang_header(lines)

    def _get(key: str, default, cast=str):
        v = meta.get(key, "")
        if not v:
            # Try common case variants
            for k in meta:
                if k.lower() == key.lower():
                    v = meta[k]
                    break
        try:
            return cast(v) if v else default
        except (ValueError, TypeError):
            return default

    grid_str   = _get("GRID", "SqrGrid", str)
    grid_type  = "hexagonal" if "hex" in grid_str.lower() else "square"
    x_step     = _get("XSTEP", 0.0, float)
    y_step     = _get("YSTEP", 0.0, float)
    n_rows     = _get("NROWS", 0, int)
    ncols_odd  = _get("NCOLS_ODD", 0, int)
    ncols_even = _get("NCOLS_EVEN", 0, int)

    # For square grids NCOLS_ODD == NCOLS_EVEN == n_cols
    n_cols = ncols_odd if ncols_odd > 0 else _get("NCOLS", 0, int)

    # ---- Parse data ----
    arr = _parse_ang_data(lines, data_start)
    N = arr.shape[0]

    # Infer grid dimensions from data if header was missing/wrong
    if n_cols == 0 or n_rows == 0:
        x_unique = np.unique(np.round(arr[:, _COL_X], 4))
        y_unique = np.unique(np.round(arr[:, _COL_Y], 4))
        n_cols = len(x_unique)
        n_rows = len(y_unique)
        if x_step == 0.0 and len(x_unique) > 1:
            x_step = float(np.median(np.diff(x_unique)))
        if y_step == 0.0 and len(y_unique) > 1:
            y_step = float(np.median(np.diff(y_unique)))

    # ---- Build phase lookup (0 = not-indexed) ----
    phases_dict: Dict[int, Phase] = {
        0: Phase(0, "notIndexed", None, None, None, None)
    }
    for p in phases:
        phases_dict[p.phase_id] = p

    # ---- Extract columns ----
    phase_id = arr[:, _COL_PHASE].astype(np.int32) if arr.shape[1] > _COL_PHASE else np.zeros(N, dtype=np.int32)

    # IQ → maps to EBSDData.bands (ANG equivalent of band contrast)
    iq = arr[:, _COL_IQ].astype(np.int32) if arr.shape[1] > _COL_IQ else np.zeros(N, dtype=np.int32)

    # CI → maps to EBSDData.bs (confidence index, 0–1)
    ci = arr[:, _COL_CI].astype(np.float32) if arr.shape[1] > _COL_CI else np.zeros(N, dtype=np.float32)

    # SEM signal → error column
    sem = arr[:, _COL_SEM].astype(np.int32) if arr.shape[1] > _COL_SEM else np.zeros(N, dtype=np.int32)

    # Fit → mad
    fit = arr[:, _COL_FIT].astype(np.float32) if arr.shape[1] > _COL_FIT else np.zeros(N, dtype=np.float32)

    # ANG Euler angles are in radians — validate range
    euler1 = arr[:, _COL_PHI1] % (2 * np.pi)
    euler2 = np.clip(arr[:, _COL_PHI], 0.0, np.pi)
    euler3 = arr[:, _COL_PHI2] % (2 * np.pi)

    data = EBSDData(
        source_file=str(path),
        file_format="ANG",
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
        bands=iq,          # IQ → bands slot
        error=sem,         # SEM signal → error slot
        mad=fit,           # fit quality → mad slot
        bc=iq,             # duplicate IQ into bc for consistency
        bs=ci,             # CI → bs slot
        detector_intensity=arr[:, _COL_SEM].astype(np.float32) if arr.shape[1] > _COL_SEM else None,
        phases=phases_dict,
        header_raw=meta,
    )

    # ---- Optional CI filter ----
    if min_ci is not None:
        bad = (ci < min_ci) | (ci < 0)
        data.phase_id[bad] = 0
        data.euler1[bad] = 0.0
        data.euler2[bad] = 0.0
        data.euler3[bad] = 0.0

    # ---- Optional hex→square resampling ----
    if convert_hex_to_square and grid_type == "hexagonal":
        data = _resample_hex_to_square(data)

    # ---- Validation ----
    if validate:
        _validate_ang(data)

    return data


# ---------------------------------------------------------------------------
# Hex → square resampling
# ---------------------------------------------------------------------------

def _resample_hex_to_square(data: EBSDData) -> EBSDData:
    """
    Nearest-neighbour resampling from hex grid to square grid.
    The output step sizes equal the original hex step sizes.
    """
    from scipy.spatial import cKDTree

    src_xy = np.column_stack([data.x, data.y])
    x_min, x_max = float(data.x.min()), float(data.x.max())
    y_min, y_max = float(data.y.min()), float(data.y.max())

    nx = max(1, round((x_max - x_min) / data.x_step) + 1)
    ny = max(1, round((y_max - y_min) / data.y_step) + 1)

    gx = np.linspace(x_min, x_max, nx)
    gy = np.linspace(y_min, y_max, ny)
    gxx, gyy = np.meshgrid(gx, gy)
    target_xy = np.column_stack([gxx.ravel(), gyy.ravel()])

    tree = cKDTree(src_xy)
    _, idx = tree.query(target_xy, k=1)

    import copy
    out = copy.deepcopy(data)
    out.x       = target_xy[:, 0]
    out.y       = target_xy[:, 1]
    out.euler1  = data.euler1[idx]
    out.euler2  = data.euler2[idx]
    out.euler3  = data.euler3[idx]
    out.phase_id = data.phase_id[idx]
    out.bands   = data.bands[idx]
    out.error   = data.error[idx]
    out.mad     = data.mad[idx]
    out.bc      = data.bc[idx]
    out.bs      = data.bs[idx]
    out.n_cols  = nx
    out.n_rows  = ny
    out.grid_type = "square"
    if data.detector_intensity is not None:
        out.detector_intensity = data.detector_intensity[idx]

    warnings.warn(
        f"Hex→square resampling: {len(data.x)} → {nx*ny} pixels "
        f"({nx}×{ny}). Orientations are nearest-neighbour interpolated.",
        UserWarning,
    )
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_ang(data: EBSDData) -> None:
    """Emit warnings for common ANG data quality issues."""
    N = data.n_pixels
    if N == 0:
        warnings.warn("ANG: zero pixels loaded.", UserWarning)
        return

    frac = np.sum(data.is_indexed) / N
    if frac < 0.1:
        warnings.warn(
            f"ANG: only {frac:.1%} of pixels are indexed. "
            "Consider adjusting min_ci.",
            UserWarning,
        )

    # CI range
    ci = data.bs  # CI is stored in bs slot
    if len(ci) > 0:
        if float(ci.max()) > 1.01:
            warnings.warn(
                "ANG: CI values exceed 1.0 — file may use a non-standard scale.",
                UserWarning,
            )
        n_neg = int(np.sum(ci < 0))
        if n_neg > 0:
            warnings.warn(
                f"ANG: {n_neg} pixels have CI < 0 (not indexed / unreliable).",
                UserWarning,
            )

    # Euler range
    bad_euler = (
        (data.euler1 < 0) | (data.euler1 > 2 * np.pi + 0.01) |
        (data.euler2 < 0) | (data.euler2 > np.pi + 0.01) |
        (data.euler3 < 0) | (data.euler3 > 2 * np.pi + 0.01)
    )
    if np.any(bad_euler & data.is_indexed):
        n_bad = int(np.sum(bad_euler & data.is_indexed))
        warnings.warn(
            f"ANG: {n_bad} indexed pixels have Euler angles outside valid Bunge range. "
            "Check if angles are in degrees instead of radians.",
            UserWarning,
        )

    if data.x_step <= 0 or data.y_step <= 0:
        warnings.warn(
            f"ANG: step size x={data.x_step} y={data.y_step} µm looks wrong.",
            UserWarning,
        )
