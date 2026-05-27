from __future__ import annotations

import numpy as np


def transformation_strain_from_distortion(F: np.ndarray) -> np.ndarray:
    """Small-strain approximation from deformation gradient/distortion F."""
    I = np.eye(3)
    return 0.5 * ((F - I) + (F - I).T)


def green_lagrange_strain(F: np.ndarray) -> np.ndarray:
    return 0.5 * (F.T @ F - np.eye(3))


def schmid_like_variant_score(stress: np.ndarray, transformation_strain: np.ndarray) -> float:
    """Mechanical work proxy: stress : strain."""
    return float(np.tensordot(stress, transformation_strain, axes=2))
