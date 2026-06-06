"""
martwin.io.ebsd_data
====================
Shared data container returned by every EBSD file reader.

EBSDData holds everything parsed from a .ctf or .ang file in one place:
scan metadata, per-phase crystallographic info, and the full pixel-level
data arrays.  All angular quantities are stored internally in RADIANS.
Spatial coordinates are stored in micrometres.

Usage
-----
>>> from martwin.io import load_ctf, load_ang
>>> data = load_ctf("scan.ctf")
>>> data = load_ang("scan.ang")
>>> print(data)
>>> rotations = data.as_rotation_matrices()   # (N, 3, 3) ndarray
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Phase descriptor
# ---------------------------------------------------------------------------

@dataclass
class Phase:
    """Crystallographic phase information parsed from the file header."""
    phase_id: int                    # integer index (0 = not-indexed)
    name: str                        # e.g. "Iron bcc", "Iron fcc", "NiTi"
    space_group: Optional[int]       # International Tables number (1–230)
    lattice_params: Optional[np.ndarray]  # (6,) [a, b, c, alpha, beta, gamma] Å / °
    laue_group: Optional[str]        # e.g. "m-3m", "6/mmm"
    color: Optional[str]             # display color hint from header

    def __repr__(self) -> str:
        lp = (
            f"a={self.lattice_params[0]:.4f} b={self.lattice_params[1]:.4f} "
            f"c={self.lattice_params[2]:.4f} Å"
            if self.lattice_params is not None
            else "unknown"
        )
        return f"Phase({self.phase_id}: '{self.name}', {lp})"


# ---------------------------------------------------------------------------
# Main data container
# ---------------------------------------------------------------------------

@dataclass
class EBSDData:
    """
    Unified EBSD dataset container, format-agnostic.

    Attributes
    ----------
    source_file : str
        Path of the file that was read.
    file_format : str
        'CTF' or 'ANG'.

    Scan geometry
    -------------
    x_step, y_step : float
        Step size in µm (from header).
    n_cols, n_rows : int
        Number of columns / rows in the map grid.
    grid_type : str
        'square' or 'hexagonal'.

    Per-pixel arrays  (all length N = n_cols * n_rows, row-major order)
    -------------------------------------------------------------------
    x, y : (N,) float64
        Pixel centre coordinates in µm.
    euler1, euler2, euler3 : (N,) float64
        Bunge Euler angles φ₁, Φ, φ₂ in **radians**.
        CTF files are converted from degrees automatically.
        ANG files are stored natively in radians.
    phase_id : (N,) int32
        Phase index for each pixel.  0 = not indexed.
    bands / n_bands : (N,) int32
        Number of detected bands (CTF) or IQ (ANG).
    error : (N,) int32
        Error flag (CTF: 0=indexed, 1=not indexed, 3=discarded).
        For ANG this holds SEM signal.
    mad / fit : (N,) float32
        Mean angular deviation (CTF) or fit quality (ANG) in degrees.
    bc / iq : (N,) int32
        Band contrast (CTF) or image quality (ANG).
    bs / ci : (N,) float32
        Band slope (CTF) or confidence index (ANG, range 0–1).
    detector_intensity : (N,) float32
        ANG only; detector intensity. None for CTF.

    Derived / optional
    ------------------
    is_indexed : (N,) bool
        True where the pixel has a valid orientation.
    phases : dict of int → Phase
        Phase table keyed by phase_id integer.
    header_raw : dict
        All key–value pairs from the file header, unparsed strings.
    """

    # provenance
    source_file: str = ""
    file_format: str = ""

    # scan geometry
    x_step: float = 0.0
    y_step: float = 0.0
    n_cols: int = 0
    n_rows: int = 0
    grid_type: str = "square"

    # per-pixel arrays
    x: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    y: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    euler1: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    euler2: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    euler3: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    phase_id: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    bands: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    error: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    mad: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    bc: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    bs: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    detector_intensity: Optional[np.ndarray] = None

    # metadata
    phases: Dict[int, Phase] = field(default_factory=dict)
    header_raw: Dict[str, str] = field(default_factory=dict)

    # ---------------------------------------------------------------------------
    # Computed properties
    # ---------------------------------------------------------------------------

    @property
    def n_pixels(self) -> int:
        return len(self.x)

    @property
    def is_indexed(self) -> np.ndarray:
        """Boolean mask: True where phase_id > 0 (valid orientation)."""
        return self.phase_id > 0

    @property
    def euler_angles_deg(self) -> np.ndarray:
        """Return (N, 3) Euler angles in degrees."""
        return np.degrees(np.column_stack([self.euler1, self.euler2, self.euler3]))

    @property
    def euler_angles_rad(self) -> np.ndarray:
        """Return (N, 3) Euler angles in radians."""
        return np.column_stack([self.euler1, self.euler2, self.euler3])

    def as_rotation_matrices(self) -> np.ndarray:
        """
        Convert Bunge Euler angles (φ₁, Φ, φ₂) to (N, 3, 3) rotation matrices.

        Convention: passive rotation, ZXZ sequence (Bunge).
        R = Rz(φ₁) · Rx(Φ) · Rz(φ₂)

        Returns
        -------
        (N, 3, 3) float64 ndarray
            Rotation matrix for each pixel.  Pixels with phase_id == 0
            (not indexed) are set to np.nan.
        """
        N = self.n_pixels
        R = np.full((N, 3, 3), np.nan)

        mask = self.is_indexed
        ph1 = self.euler1[mask]
        Ph  = self.euler2[mask]
        ph2 = self.euler3[mask]

        c1, s1 = np.cos(ph1), np.sin(ph1)
        cP, sP = np.cos(Ph),  np.sin(Ph)
        c2, s2 = np.cos(ph2), np.sin(ph2)

        # Bunge ZXZ passive convention
        R[mask, 0, 0] =  c1*c2 - s1*s2*cP
        R[mask, 0, 1] = -c1*s2 - s1*c2*cP
        R[mask, 0, 2] =  s1*sP
        R[mask, 1, 0] =  s1*c2 + c1*s2*cP
        R[mask, 1, 1] = -s1*s2 + c1*c2*cP
        R[mask, 1, 2] = -c1*sP
        R[mask, 2, 0] =  s2*sP
        R[mask, 2, 1] =  c2*sP
        R[mask, 2, 2] =  cP
        return R

    def map_array(self, arr: np.ndarray) -> np.ndarray:
        """
        Reshape a flat per-pixel array (N,) into the 2-D map (n_rows, n_cols).
        """
        if len(arr) != self.n_pixels:
            raise ValueError(
                f"Array length {len(arr)} != n_pixels {self.n_pixels}"
            )
        return arr.reshape(self.n_rows, self.n_cols)

    def phase_mask(self, phase_id: int) -> np.ndarray:
        """Boolean mask selecting pixels belonging to a given phase."""
        return self.phase_id == phase_id

    def filter_by_mad(self, max_mad_deg: float) -> "EBSDData":
        """
        Return a copy with non-indexed or high-MAD pixels zeroed out.
        Useful for cleaning the dataset before reconstruction.
        """
        import copy
        out = copy.deepcopy(self)
        bad = (self.mad > max_mad_deg) | (~self.is_indexed)
        out.phase_id[bad] = 0
        out.euler1[bad] = 0.0
        out.euler2[bad] = 0.0
        out.euler3[bad] = 0.0
        return out

    def __repr__(self) -> str:
        indexed = int(np.sum(self.is_indexed))
        pct = 100.0 * indexed / max(self.n_pixels, 1)
        phase_str = ", ".join(
            f"{pid}:{p.name}" for pid, p in self.phases.items()
        )
        return (
            f"EBSDData [{self.file_format}] {self.source_file}\n"
            f"  Grid      : {self.n_cols} × {self.n_rows} ({self.grid_type})\n"
            f"  Step      : x={self.x_step:.3f} µm  y={self.y_step:.3f} µm\n"
            f"  Pixels    : {self.n_pixels:,}  indexed={indexed:,} ({pct:.1f}%)\n"
            f"  Phases    : {phase_str if phase_str else 'none'}\n"
        )
