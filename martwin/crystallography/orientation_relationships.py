from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from martwin.core.lattice import CUBIC, Lattice
from martwin.core.rotations import normalize, project_to_rotation


def _basis_from_dir_plane(direction: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """Build orthonormal basis from one direction and one plane normal.

    The direction is projected into the plane to reduce inconsistency from non-exact
    literature indices or metric simplifications.
    """
    n = normalize(plane_normal)
    d = np.asarray(direction, dtype=float)
    d_proj = d - np.dot(d, n) * n
    if np.linalg.norm(d_proj) < 1e-8:
        # If direction is accidentally normal to plane, fall back to a perpendicular vector.
        trial = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(trial, n)) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        d_proj = trial - np.dot(trial, n) * n
    x = normalize(d_proj)
    z = n
    y = normalize(np.cross(z, x))
    x = normalize(np.cross(y, z))
    return np.column_stack([x, y, z])


def or_from_parallel_direction_plane(
    parent_lattice: Lattice,
    child_lattice: Lattice,
    parent_direction: Iterable[float],
    child_direction: Iterable[float],
    parent_plane: Iterable[float],
    child_plane: Iterable[float],
) -> np.ndarray:
    """Construct child-to-parent OR rotation from parallel direction and plane.

    Returns R_cp such that a vector in child crystal coordinates maps to parent
    crystal coordinates by v_parent = R_cp @ v_child in an orthonormal Cartesian basis.
    """
    p_dir = parent_lattice.direction_cart(parent_direction)
    c_dir = child_lattice.direction_cart(child_direction)
    p_n = parent_lattice.plane_normal_cart(parent_plane)
    c_n = child_lattice.plane_normal_cart(child_plane)

    Bp = _basis_from_dir_plane(p_dir, p_n)
    Bc = _basis_from_dir_plane(c_dir, c_n)
    return project_to_rotation(Bp @ Bc.T)


@dataclass(frozen=True)
class OrientationRelationship:
    name: str
    parent_phase: str
    child_phase: str
    matrix_child_to_parent: np.ndarray
    parent_plane: tuple[float, float, float] | None = None
    child_plane: tuple[float, float, float] | None = None
    parent_direction: tuple[float, float, float] | None = None
    child_direction: tuple[float, float, float] | None = None
   source_note: str = ""

@property
def description(self) -> str:
    """Human-readable explanation used by the Streamlit app/reporting layer."""
    return self.source_note or self.name

def as_parent_to_child(self) -> np.ndarray:
    return self.matrix_child_to_parent.T


def steel_ks_or() -> OrientationRelationship:
    # Kurdjumov-Sachs: {111}_gamma // {110}_alpha, <110>_gamma // <111>_alpha
    R = or_from_parallel_direction_plane(
        CUBIC, CUBIC,
        parent_direction=(1, -1, 0), child_direction=(1, 1, -1),
        parent_plane=(1, 1, 1), child_plane=(1, 1, 0),
    )
    return OrientationRelationship(
        name="Kurdjumov-Sachs approximation",
        parent_phase="fcc_austenite",
        child_phase="bcc_martensite",
        matrix_child_to_parent=R,
        parent_plane=(1, 1, 1), child_plane=(1, 1, 0),
        parent_direction=(1, -1, 0), child_direction=(1, 1, -1),
        source_note="Classical fcc→bcc/bct OR used as initial rational OR; real lath martensite may be irrational/refined from EBSD.",
    )


def steel_nw_or() -> OrientationRelationship:
    # Nishiyama-Wassermann: {111}_gamma // {110}_alpha, <112>_gamma // <110>_alpha
    R = or_from_parallel_direction_plane(
        CUBIC, CUBIC,
        parent_direction=(1, 1, -2), child_direction=(1, -1, 0),
        parent_plane=(1, 1, 1), child_plane=(1, 1, 0),
    )
    return OrientationRelationship(
        name="Nishiyama-Wassermann approximation",
        parent_phase="fcc_austenite",
        child_phase="bcc_martensite",
        matrix_child_to_parent=R,
        parent_plane=(1, 1, 1), child_plane=(1, 1, 0),
        parent_direction=(1, 1, -2), child_direction=(1, -1, 0),
        source_note="Classical fcc→bcc OR; useful comparator for steel martensite.",
    )


def steel_pitsch_or() -> OrientationRelationship:
    # One common Pitsch representation: {001}_gamma // {011}_alpha, <110>_gamma // <111>_alpha
    R = or_from_parallel_direction_plane(
        CUBIC, CUBIC,
        parent_direction=(1, 1, 0), child_direction=(1, 1, 1),
        parent_plane=(0, 0, 1), child_plane=(0, 1, 1),
    )
    return OrientationRelationship(
        name="Pitsch approximation",
        parent_phase="fcc_austenite",
        child_phase="bcc_martensite",
        matrix_child_to_parent=R,
        parent_plane=(0, 0, 1), child_plane=(0, 1, 1),
        parent_direction=(1, 1, 0), child_direction=(1, 1, 1),
        source_note="Classical fcc→bcc OR comparator; exact convention varies in literature.",
    )


def cayron_niti_natural_or(beta_deg: float = 96.8) -> OrientationRelationship:
    """Prototype Cayron natural OR for NiTi B2→B19′.

    Uses the dense plane/direction parallelism reported by Cayron:
    (010)B19′ // (110)B2 and [101]B19′ // [111]B2.

    Default B19′ lattice parameters are representative approximate values. Users
    should replace them with composition/temperature-specific XRD/DFT values.
    """
    b2 = Lattice(1.0, 1.0, 1.0)
    b19 = Lattice(a=2.889, b=4.120, c=4.622, alpha=90.0, beta=beta_deg, gamma=90.0)
    R = or_from_parallel_direction_plane(
        parent_lattice=b2,
        child_lattice=b19,
        parent_direction=(1, 1, 1), child_direction=(1, 0, 1),
        parent_plane=(1, 1, 0), child_plane=(0, 1, 0),
    )
    return OrientationRelationship(
        name="Cayron NiTi natural OR prototype",
        parent_phase="B2_austenite",
        child_phase="B19prime_martensite",
        matrix_child_to_parent=R,
        parent_plane=(1, 1, 0), child_plane=(0, 1, 0),
        parent_direction=(1, 1, 1), child_direction=(1, 0, 1),
        source_note="Cayron natural OR: (010)B19′//(110)B2 and [101]B19′//[111]B2. Replace lattice constants with measured sample values for exact work.",
    )


def default_or_library() -> dict[str, OrientationRelationship]:
    ors = [steel_ks_or(), steel_nw_or(), steel_pitsch_or(), cayron_niti_natural_or()]
    return {orx.name: orx for orx in ors}
