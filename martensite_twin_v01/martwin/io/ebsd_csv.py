from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from martwin.core.rotations import bunge_euler_to_matrix

REQUIRED_EULER_COLUMNS = ("phi1", "Phi", "phi2")


def read_ebsd_csv(path: str | Path, degrees: bool = True) -> pd.DataFrame:
    """Read a simple EBSD CSV export.

    Expected columns: x, y, phi1, Phi, phi2, optionally phase, grain_id, ci/iq.
    Vendor-native formats (.ctf, .ang, .h5) should be converted or handled by
    future kikuchipy/orix importers.
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_EULER_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing Euler columns: {missing}. Expected {REQUIRED_EULER_COLUMNS}")
    mats = [bunge_euler_to_matrix(r.phi1, r.Phi, r.phi2, degrees=degrees) for r in df.itertuples()]
    df = df.copy()
    df["orientation_matrix"] = mats
    return df


def write_synthetic_ebsd_csv(path: str | Path, orientations: list[np.ndarray], grid_shape: tuple[int, int] | None = None) -> None:
    """Write synthetic orientations to a CSV with matrix entries, not Euler inversion.

    For demos we store r00..r22 directly because robust Euler inversion is convention-sensitive.
    """
    n = len(orientations)
    if grid_shape is None:
        grid_shape = (n, 1)
    rows = []
    for idx, R in enumerate(orientations):
        y, x = divmod(idx, grid_shape[1])
        row = {"x": x, "y": y, "phase": "child", "point_id": idx}
        for i in range(3):
            for j in range(3):
                row[f"r{i}{j}"] = float(R[i, j])
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def read_orientation_matrix_csv(path: str | Path) -> list[np.ndarray]:
    df = pd.read_csv(path)
    cols = [f"r{i}{j}" for i in range(3) for j in range(3)]
    if not set(cols).issubset(df.columns):
        raise ValueError("Matrix CSV requires r00..r22 columns")
    return [np.array([[r.r00, r.r01, r.r02], [r.r10, r.r11, r.r12], [r.r20, r.r21, r.r22]], dtype=float) for r in df.itertuples()]
