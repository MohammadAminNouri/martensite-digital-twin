from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from martwin.core.rotations import axis_angle, project_to_rotation
from martwin.crystallography.variants import Variant, predict_child_orientations


@dataclass
class SyntheticMap:
    dataframe: pd.DataFrame
    child_orientations: list[np.ndarray]
    parent_orientations: list[np.ndarray]
    grid_shape: tuple[int, int]


def random_orientation(rng: np.random.Generator) -> np.ndarray:
    """Uniform random proper rotation using normalized quaternion."""
    q = rng.normal(size=4)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=float)


def small_random_rotation(rng: np.random.Generator, sigma_deg: float) -> np.ndarray:
    if sigma_deg <= 0:
        return np.eye(3)
    axis = rng.normal(size=3)
    angle = np.deg2rad(abs(rng.normal(loc=0.0, scale=sigma_deg)))
    return axis_angle(axis, angle)


def generate_synthetic_child_map(
    variants: list[Variant],
    grid_shape: tuple[int, int] = (60, 60),
    n_parent_grains: int = 4,
    active_variant_fraction: float = 0.55,
    orientation_noise_deg: float = 0.75,
    seed: int = 7,
) -> SyntheticMap:
    """Generate a toy EBSD-like martensite orientation map.

    The map is not meant to simulate real lath morphology. It creates contiguous
    rectangular parent regions and assigns a subset of variants inside each parent
    grain. This gives a useful file for testing variant assignment and UI workflow.
    """
    rng = np.random.default_rng(seed)
    h, w = grid_shape
    n_parent_grains = max(1, int(n_parent_grains))
    parent_oris = [random_orientation(rng) for _ in range(n_parent_grains)]

    # Create simple parent blocks by splitting grid columns.
    block_width = max(1, int(np.ceil(w / n_parent_grains)))
    rows = []
    child_oris = []
    for y in range(h):
        for x in range(w):
            parent_id = min(n_parent_grains - 1, x // block_width)
            parent = parent_oris[parent_id]
            all_child = predict_child_orientations(parent, variants)
            k = max(1, int(round(active_variant_fraction * len(all_child))))
            active_ids = rng.choice(len(all_child), size=k, replace=False)
            chosen_idx = int(rng.choice(active_ids))
            R = project_to_rotation(small_random_rotation(rng, orientation_noise_deg) @ all_child[chosen_idx])
            child_oris.append(R)
            row = {
                "x": x,
                "y": y,
                "point_id": y * w + x,
                "parent_region_id": parent_id + 1,
                "true_variant_id": variants[chosen_idx].id,
                "phase": "martensite",
            }
            for i in range(3):
                for j in range(3):
                    row[f"r{i}{j}"] = float(R[i, j])
            rows.append(row)
    return SyntheticMap(pd.DataFrame(rows), child_oris, parent_oris, grid_shape)
