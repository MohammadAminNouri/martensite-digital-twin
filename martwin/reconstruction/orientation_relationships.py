from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.spatial.transform import Rotation


def _normalise(v: np.ndarray) -> np.ndarray:
    """Return a unit vector."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Cannot normalise near-zero vector")
    return v / n


def _rotation_from_parallel_pairs(
    parent_dir: np.ndarray,
    child_dir: np.ndarray,
    parent_plane_normal: np.ndarray,
    child_plane_normal: np.ndarray,
) -> np.ndarray:
    """
    Build a rotation matrix R such that:

        R @ parent_dir          ≈ child_dir
        R @ parent_plane_normal ≈ child_plane_normal

    Used to define prototype OR variants.
    """

    p_x = _normalise(parent_dir)
    p_z = _normalise(parent_plane_normal)
    p_y = _normalise(np.cross(p_z, p_x))
    p_x = _normalise(np.cross(p_y, p_z))
    Bp = np.column_stack([p_x, p_y, p_z])

    c_x = _normalise(child_dir)
    c_z = _normalise(child_plane_normal)
    c_y = _normalise(np.cross(c_z, c_x))
    c_x = _normalise(np.cross(c_y, c_z))
    Bc = np.column_stack([c_x, c_y, c_z])

    R = Bc @ Bp.T
    return _project_to_rotation(R)


def _project_to_rotation(R: np.ndarray) -> np.ndarray:
    """Project a near-rotation matrix onto SO(3)."""
    U, _, Vt = np.linalg.svd(R)
    Q = U @ Vt
    if np.linalg.det(Q) < 0:
        U[:, -1] *= -1
        Q = U @ Vt
    return Q


def _rotation_key(R: np.ndarray, decimals: int = 6) -> tuple:
    """Hashable key for approximate duplicate removal."""
    return tuple(np.round(R.reshape(-1), decimals=decimals))


def _cubic_symmetry_operators() -> List[np.ndarray]:
    """
    Generate the 24 proper cubic symmetry rotations.

    These are all signed permutation matrices with determinant +1.
    """
    ops: List[np.ndarray] = []
    axes = np.eye(3)

    import itertools

    for perm in itertools.permutations(range(3)):
        P = axes[:, perm]
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            S = P @ np.diag(signs)
            if np.linalg.det(S) > 0.5:
                ops.append(S)

    unique: List[np.ndarray] = []
    seen = set()
    for op in ops:
        key = _rotation_key(op)
        if key not in seen:
            seen.add(key)
            unique.append(op)

    return unique


_CUBIC_SYM = _cubic_symmetry_operators()


def _deduplicate_rotations(rotations: List[np.ndarray], tol_deg: float = 0.25) -> List[np.ndarray]:
    """Remove nearly identical rotations."""
    unique: List[np.ndarray] = []

    for R in rotations:
        R = _project_to_rotation(R)
        duplicate = False

        for U in unique:
            M = R @ U.T
            c = np.clip((np.trace(M) - 1.0) / 2.0, -1.0, 1.0)
            angle = np.degrees(np.arccos(c))
            if angle < tol_deg:
                duplicate = True
                break

        if not duplicate:
            unique.append(R)

    return unique


def _apply_cubic_symmetry(prototype: np.ndarray, target_count: int | None = None) -> List[np.ndarray]:
    """
    Generate crystallographic variants by applying cubic symmetry.

    The returned matrices are proper rotations.
    """
    candidates: List[np.ndarray] = []

    for S_child in _CUBIC_SYM:
        for S_parent in _CUBIC_SYM:
            R = S_child @ prototype @ S_parent.T
            candidates.append(_project_to_rotation(R))

    variants = _deduplicate_rotations(candidates)

    if target_count is not None:
        variants = variants[:target_count]

    return variants


@dataclass(frozen=True)
class OrientationRelationship:
    """
    Container for an orientation relationship.

    Convention:
        R_child = R_variant @ R_parent

    Therefore:
        R_parent = R_variant.T @ R_child
    """

    name: str
    variants: List[np.ndarray]
    parent_phase: str = "parent"
    child_phase: str = "child"
    description: str = ""
    source_note: str = ""

    @property
    def n_variants(self) -> int:
        return len(self.variants)

    @property
    def matrix_child_to_parent(self) -> np.ndarray:
        if not self.variants:
            return np.eye(3)
        return self.variants[0].T

    def parent_from_child(self, R_child: np.ndarray, variant_id: int) -> np.ndarray:
        Rv = self.variants[int(variant_id)]
        return _project_to_rotation(Rv.T @ R_child)

    def child_from_parent(self, R_parent: np.ndarray, variant_id: int) -> np.ndarray:
        Rv = self.variants[int(variant_id)]
        return _project_to_rotation(Rv @ R_parent)


def _make_ks() -> OrientationRelationship:
    """
    Kurdjumov-Sachs OR.

    Approximate prototype:
        (111)_fcc || (011)_bcc
        [1 -1 0]_fcc || [1 -1 1]_bcc
    """

    prototype = _rotation_from_parallel_pairs(
        parent_dir=np.array([1.0, -1.0, 0.0]),
        child_dir=np.array([1.0, -1.0, 1.0]),
        parent_plane_normal=np.array([1.0, 1.0, 1.0]),
        child_plane_normal=np.array([0.0, 1.0, 1.0]),
    )

    variants = _apply_cubic_symmetry(prototype, target_count=24)

    return OrientationRelationship(
        name="KS",
        variants=variants,
        parent_phase="fcc/austenite",
        child_phase="bcc/bct/martensite",
        description="Kurdjumov-Sachs fcc→bcc/bct orientation relationship.",
        source_note="Prototype OR generated from parallel plane/direction pair and cubic symmetry.",
    )


def _make_nw() -> OrientationRelationship:
    """
    Nishiyama-Wassermann OR.

    Approximate prototype:
        (111)_fcc || (011)_bcc
        [1 -1 0]_fcc || [0 0 1]_bcc
    """

    prototype = _rotation_from_parallel_pairs(
        parent_dir=np.array([1.0, -1.0, 0.0]),
        child_dir=np.array([0.0, 0.0, 1.0]),
        parent_plane_normal=np.array([1.0, 1.0, 1.0]),
        child_plane_normal=np.array([0.0, 1.0, 1.0]),
    )

    variants = _apply_cubic_symmetry(prototype, target_count=12)

    return OrientationRelationship(
        name="NW",
        variants=variants,
        parent_phase="fcc/austenite",
        child_phase="bcc/bct/martensite",
        description="Nishiyama-Wassermann fcc→bcc/bct orientation relationship.",
        source_note="Prototype OR generated from parallel plane/direction pair and cubic symmetry.",
    )


def _make_pitsch() -> OrientationRelationship:
    """
    Pitsch OR.

    Approximate prototype variant.
    """

    prototype = Rotation.from_euler("zxz", [45.0, 10.5, 45.0], degrees=True).as_matrix()
    variants = _apply_cubic_symmetry(prototype, target_count=24)

    return OrientationRelationship(
        name="Pitsch",
        variants=variants,
        parent_phase="fcc/austenite",
        child_phase="bcc/bct/martensite",
        description="Approximate Pitsch fcc→bcc/bct orientation relationship.",
        source_note="Simplified prototype used for reconstruction testing.",
    )


def _make_gt() -> OrientationRelationship:
    """
    Greninger-Troiano-like OR.

    Approximate prototype between KS and NW.
    """

    ks = _make_ks().variants[0]
    nw = _make_nw().variants[0]

    r_ks = Rotation.from_matrix(ks)
    r_nw = Rotation.from_matrix(nw)

    # Simple halfway interpolation in rotation-vector space
    rv = 0.5 * (r_ks.as_rotvec() + r_nw.as_rotvec())
    prototype = Rotation.from_rotvec(rv).as_matrix()

    variants = _apply_cubic_symmetry(prototype, target_count=24)

    return OrientationRelationship(
        name="GT",
        variants=variants,
        parent_phase="fcc/austenite",
        child_phase="bcc/bct/martensite",
        description="Approximate Greninger-Troiano-like orientation relationship.",
        source_note="Simplified interpolation between KS and NW prototypes.",
    )


def _make_cayron_niti() -> OrientationRelationship:
    """
    Cayron-inspired NiTi B2→B19′ OR prototype.

    This is a simplified educational OR for the digital twin prototype.
    """

    prototype = Rotation.from_euler("zxz", [0.0, 6.8, 45.0], degrees=True).as_matrix()
    variants = _apply_cubic_symmetry(prototype, target_count=12)

    return OrientationRelationship(
        name="Cayron_NiTi",
        variants=variants,
        parent_phase="B2 austenite",
        child_phase="B19′ martensite",
        description="Cayron-inspired simplified NiTi B2→B19′ orientation relationship.",
        source_note="Prototype for educational reconstruction; verify conventions before publication.",
    )


OR_REGISTRY: Dict[str, OrientationRelationship] = {
    "KS": _make_ks(),
    "NW": _make_nw(),
    "Pitsch": _make_pitsch(),
    "GT": _make_gt(),
    "Cayron_NiTi": _make_cayron_niti(),
}


def get_OR(name: str) -> OrientationRelationship:
    """Return an orientation relationship by name."""
    if name not in OR_REGISTRY:
        available = ", ".join(OR_REGISTRY.keys())
        raise KeyError(f"Unknown OR '{name}'. Available ORs: {available}")
    return OR_REGISTRY[name]
    