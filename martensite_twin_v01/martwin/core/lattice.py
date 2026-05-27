from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .rotations import normalize


@dataclass(frozen=True)
class Lattice:
    """Direct lattice for converting crystal directions and plane normals.

    Supports triclinic metrics; used here mainly for cubic and monoclinic B19′.
    Angles are in degrees.
    """

    a: float
    b: float
    c: float
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0

    def basis(self) -> np.ndarray:
        alpha, beta, gamma = map(math.radians, (self.alpha, self.beta, self.gamma))
        ax = np.array([self.a, 0.0, 0.0])
        bx = self.b * math.cos(gamma)
        by = self.b * math.sin(gamma)
        bvec = np.array([bx, by, 0.0])
        cx = self.c * math.cos(beta)
        cy = self.c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / math.sin(gamma)
        cz_sq = self.c**2 - cx**2 - cy**2
        cz = math.sqrt(max(cz_sq, 0.0))
        cvec = np.array([cx, cy, cz])
        return np.column_stack([ax, bvec, cvec])

    def reciprocal_basis(self) -> np.ndarray:
        B = self.basis()
        return np.linalg.inv(B).T

    def direction_cart(self, uvw: Iterable[float]) -> np.ndarray:
        return self.basis() @ np.asarray(uvw, dtype=float)

    def plane_normal_cart(self, hkl: Iterable[float]) -> np.ndarray:
        return self.reciprocal_basis() @ np.asarray(hkl, dtype=float)

    def unit_direction(self, uvw: Iterable[float]) -> np.ndarray:
        return normalize(self.direction_cart(uvw))

    def unit_plane_normal(self, hkl: Iterable[float]) -> np.ndarray:
        return normalize(self.plane_normal_cart(hkl))


CUBIC = Lattice(1.0, 1.0, 1.0)
