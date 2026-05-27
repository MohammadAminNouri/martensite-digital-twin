import numpy as np

from martwin.core.rotations import axis_angle, rotation_angle
from martwin.core.symmetry import cubic_proper_rotations
from martwin.crystallography.orientation_relationships import steel_ks_or, cayron_niti_natural_or
from martwin.crystallography.variants import generate_variants
from martwin.core.symmetry import monoclinic_2_unique_axis_b


def test_axis_angle_identity():
    R = axis_angle([1, 0, 0], 0.0)
    assert np.allclose(R, np.eye(3))
    assert rotation_angle(R) < 1e-8


def test_cubic_ops_count():
    assert len(cubic_proper_rotations()) == 24


def test_generate_steel_variants_nonzero():
    variants = generate_variants(steel_ks_or(), cubic_proper_rotations(), cubic_proper_rotations())
    assert len(variants) > 0


def test_generate_niti_variants_nonzero():
    variants = generate_variants(cayron_niti_natural_or(), cubic_proper_rotations(), monoclinic_2_unique_axis_b())
    assert len(variants) > 0
