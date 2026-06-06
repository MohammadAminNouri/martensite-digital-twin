"""
martwin.io
==========
EBSD file readers for the martensite-digital-twin project.

Supported formats
-----------------
.ctf    Oxford Instruments / HKL Channel 5 (ASCII, degrees)
.ang    EDAX / TSL OIM (ASCII, radians)

Quick start
-----------
>>> from martwin.io import load
>>> data = load("scan.ctf")       # auto-detects format by extension
>>> data = load("scan.ang")

>>> from martwin.io import load_ctf, load_ang
>>> ctf = load_ctf("scan.ctf", max_mad_deg=1.5)
>>> ang = load_ang("scan.ang", min_ci=0.1)

>>> print(data)
>>> R = data.as_rotation_matrices()      # (N, 3, 3) rotation matrices
>>> euler_deg = data.euler_angles_deg    # (N, 3) degrees
>>> ipf_map = data.map_array(data.bc)   # reshape (N,) → (rows, cols)

Reference frame utilities
-------------------------
>>> from martwin.io import oxford_to_edax, edax_to_oxford
>>> data_edax_frame = oxford_to_edax(ctf_data)
"""

from __future__ import annotations

import pathlib
import warnings
from typing import Optional

from .ebsd_data import EBSDData, Phase
from .ctf_reader import load_ctf
from .ang_reader import load_ang


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

def load(
    path: str | pathlib.Path,
    validate: bool = True,
    **kwargs,
) -> EBSDData:
    """
    Load an EBSD file, auto-detecting format from the file extension.

    Parameters
    ----------
    path : str or pathlib.Path
    validate : bool
        Run post-load sanity checks.
    **kwargs :
        Passed to the format-specific loader:
        - CTF: max_mad_deg (float)
        - ANG: min_ci (float), convert_hex_to_square (bool)

    Returns
    -------
    EBSDData

    Raises
    ------
    ValueError : if the extension is not recognised.
    FileNotFoundError : if the file does not exist.
    """
    p = pathlib.Path(path)
    ext = p.suffix.lower()

    if ext == ".ctf":
        return load_ctf(p, validate=validate, **kwargs)
    elif ext == ".ang":
        return load_ang(p, validate=validate, **kwargs)
    else:
        raise ValueError(
            f"Unrecognised EBSD file extension '{ext}'. "
            f"Supported: .ctf, .ang"
        )


# ---------------------------------------------------------------------------
# Reference frame conversions
# ---------------------------------------------------------------------------

def oxford_to_edax(data: EBSDData) -> EBSDData:
    """
    Convert CTF (Oxford HKL) reference frame to EDAX/TSL convention.

    Oxford: x→ (scan direction), y↓ (normal), Euler ZXZ w.r.t. crystal
    EDAX:   x→, y↑, Euler ZXZ rotated 180° about RD (x-axis)

    The conversion applies a 180° rotation about the x-axis to all
    Euler angles and flips the y spatial coordinate.

    Parameters
    ----------
    data : EBSDData  (CTF)

    Returns
    -------
    EBSDData  (ANG convention)
    """
    import copy
    import numpy as np

    out = copy.deepcopy(data)

    # Flip y coordinate
    if data.n_rows > 0 and data.y_step > 0:
        y_max = float(data.y.max())
        out.y = y_max - data.y

    # Rotate Euler angles by 180° about x-axis:
    # (φ₁, Φ, φ₂) → (φ₁, π - Φ, π - φ₂) for indexed pixels only
    mask = data.is_indexed
    out.euler2[mask] = np.pi - data.euler2[mask]
    out.euler3[mask] = (np.pi - data.euler3[mask]) % (2 * np.pi)

    out.file_format = "ANG"
    return out


def edax_to_oxford(data: EBSDData) -> EBSDData:
    """
    Convert EDAX/TSL ANG reference frame to Oxford CTF convention.
    Inverse of oxford_to_edax().
    """
    # The transformation is its own inverse (180° rotation)
    return oxford_to_edax(data)


# ---------------------------------------------------------------------------
# Convenience: combine CTF + ANG data from the same scan
# ---------------------------------------------------------------------------

def merge_ctf_ang(ctf: EBSDData, ang: EBSDData) -> EBSDData:
    """
    Merge a CTF and an ANG file from the same scan into one EBSDData.

    Uses CTF orientation data (more reliable for steels) and ANG quality
    metrics (IQ, CI) where CTF has none.  Spatial coordinates are taken
    from CTF.

    Parameters
    ----------
    ctf : EBSDData loaded from .ctf
    ang : EBSDData loaded from .ang  (same scan, same pixel count)

    Returns
    -------
    EBSDData (merged)
    """
    import copy
    if ctf.n_pixels != ang.n_pixels:
        warnings.warn(
            f"CTF ({ctf.n_pixels}) and ANG ({ang.n_pixels}) have different "
            "pixel counts. Proceeding with CTF pixel count — ANG quality "
            "metrics may be misaligned.",
            UserWarning,
        )
    out = copy.deepcopy(ctf)
    n = min(ctf.n_pixels, ang.n_pixels)
    # Supplement with ANG CI where CTF lacks it
    out.bs[:n] = ang.bs[:n]          # CI → bs slot
    out.detector_intensity = ang.detector_intensity[:n] if ang.detector_intensity is not None else None
    return out


# ---------------------------------------------------------------------------
# Public symbols
# ---------------------------------------------------------------------------

__all__ = [
    "load",
    "load_ctf",
    "load_ang",
    "EBSDData",
    "Phase",
    "oxford_to_edax",
    "edax_to_oxford",
    "merge_ctf_ang",
]
