from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

Array = np.ndarray


def normalize(v: Iterable[float], *, tol: float = 1e-12) -> Array:
    arr = np.asarray(v, dtype=float)
    n = np.linalg.norm(arr)
    if n < tol:
        raise ValueError("Cannot normalize a near-zero vector")
    return arr / n


def axis_angle(axis: Iterable[float], angle_rad: float) -> Array:
    """Proper rotation matrix from axis-angle using Rodrigues' formula."""
    x, y, z = normalize(axis)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    C = 1.0 - c
    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C],
    ], dtype=float)


def bunge_euler_to_matrix(phi1: float, Phi: float, phi2: float, degrees: bool = True) -> Array:
    """Convert Bunge Euler angles (phi1, Phi, phi2) to rotation matrix.

    Convention: R = Rz(phi1) Rx(Phi) Rz(phi2), common in texture/EBSD work.
    Always document your vendor convention before comparing absolute maps.
    """
    if degrees:
        phi1, Phi, phi2 = map(math.radians, (phi1, Phi, phi2))
    c1, s1 = math.cos(phi1), math.sin(phi1)
    c, s = math.cos(Phi), math.sin(Phi)
    c2, s2 = math.cos(phi2), math.sin(phi2)
    return np.array([
        [c1*c2 - s1*s2*c,  s1*c2 + c1*s2*c,  s2*s],
        [-c1*s2 - s1*c2*c, -s1*s2 + c1*c2*c, c2*s],
        [s1*s,             -c1*s,             c],
    ], dtype=float)


def matrix_to_quaternion(R: Array) -> Array:
    """Return quaternion [w, x, y, z] from a rotation matrix."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        return np.array([0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S])
    idx = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    if idx == 0:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return np.array([(R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S])
    if idx == 1:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return np.array([(R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S])
    S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
    return np.array([(R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S])


def rotation_angle(R: Array, degrees: bool = True) -> float:
    """Return rotation angle of matrix R."""
    R = np.asarray(R, dtype=float)
    x = (np.trace(R) - 1.0) / 2.0
    x = max(-1.0, min(1.0, float(x)))
    ang = math.acos(x)
    return math.degrees(ang) if degrees else ang


def misorientation_angle(Ra: Array, Rb: Array, sym_ops: list[Array] | None = None, degrees: bool = True) -> float:
    """Minimum misorientation angle between two orientations under optional crystal symmetry.

    If sym_ops is supplied, each S is applied as Ra @ S @ Rb.T.
    This is a simple proper-rotation formulation suitable for v0.1.
    """
    if sym_ops is None:
        return rotation_angle(Ra @ Rb.T, degrees=degrees)
    return min(rotation_angle(Ra @ S @ Rb.T, degrees=degrees) for S in sym_ops)


def project_to_rotation(M: Array) -> Array:
    """Project a near-rotation matrix to SO(3) with SVD."""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


@dataclass(frozen=True)
class Orientation:
    matrix: Array

    @classmethod
    def from_euler(cls, phi1: float, Phi: float, phi2: float, degrees: bool = True) -> "Orientation":
        return cls(bunge_euler_to_matrix(phi1, Phi, phi2, degrees=degrees))

    def misorientation_to(self, other: "Orientation", sym_ops: list[Array] | None = None) -> float:
        return misorientation_angle(self.matrix, other.matrix, sym_ops=sym_ops)
