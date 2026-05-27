from __future__ import annotations

import itertools
import numpy as np


def cubic_proper_rotations() -> list[np.ndarray]:
    """Return 24 proper rotation matrices of the cubic point group 432.

    These are signed permutation matrices with determinant +1.
    """
    ops = []
    for perm in itertools.permutations(range(3)):
        P = np.zeros((3, 3))
        for i, j in enumerate(perm):
            P[i, j] = 1.0
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            S = np.diag(signs) @ P
            if round(np.linalg.det(S)) == 1:
                ops.append(S)
    # Deduplicate numerically
    unique = []
    for op in ops:
        if not any(np.allclose(op, u) for u in unique):
            unique.append(op)
    return unique


def monoclinic_2_unique_axis_b() -> list[np.ndarray]:
    """Minimal proper rotations for monoclinic 2/m with unique axis b.

    For orientation-variant enumeration this is a simplified proper subgroup: identity
    and 180° rotation about b. Mirror/inversion operations are not proper rotations.
    """
    return [np.eye(3), np.diag([-1.0, 1.0, -1.0])]


def unique_rotations(rotations: list[np.ndarray], tol_deg: float = 0.1) -> list[np.ndarray]:
    from .rotations import misorientation_angle

    unique: list[np.ndarray] = []
    for R in rotations:
        if not any(misorientation_angle(R, U) < tol_deg for U in unique):
            unique.append(R)
    return unique
