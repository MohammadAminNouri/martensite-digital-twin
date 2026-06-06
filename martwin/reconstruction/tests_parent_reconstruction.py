"""
tests/reconstruction/test_parent_reconstruction.py
====================================================
Unit and integration tests for the martwin parent phase reconstruction module.

Run with:
    pytest tests/reconstruction/ -v

Dependencies: numpy, scipy, pytest
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Imports from martwin reconstruction module
# ---------------------------------------------------------------------------
import sys
import pathlib
# Allow running from the repo root
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from martwin.reconstruction.orientation_relationships import (
    OR_REGISTRY,
    get_OR,
    _apply_cubic_symmetry,
    _normalise,
)
from martwin.reconstruction.grain_graph import (
    GrainData,
    build_variant_graph,
    markov_cluster,
    _misorientation_angle,
    _OH_SYM,
    _edge_probability,
    _candidate_parent_orientations,
)
from martwin.reconstruction.parent_reconstructor import (
    ParentReconstructor,
    ORRefiner,
    detect_OR,
    _vote_parent_orientation,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def ks_or():
    return get_OR("KS")


@pytest.fixture(scope="module")
def nw_or():
    return get_OR("NW")


def _make_synthetic_grain_data(
    n_parent_grains: int = 5,
    variants_per_parent: int = 8,
    or_name: str = "KS",
    noise_deg: float = 1.0,
    seed: int = 42,
) -> tuple:
    """
    Build a synthetic EBSD dataset where ground-truth parent orientations
    are known.

    For each parent grain, `variants_per_parent` child grains are created
    by applying distinct OR variants (with Gaussian noise) to the parent
    orientation.

    Returns
    -------
    grain_data : GrainData
    ground_truth_parents : (N,) array of parent grain index per child grain
    ground_truth_variants : (N,) array of variant index per child grain
    """
    or_obj = get_OR(or_name)
    rng = np.random.default_rng(seed)

    parent_orientations = Rotation.random(n_parent_grains, random_state=rng).as_matrix()

    child_orientations = []
    gt_parents = []
    gt_variants = []

    nv = or_obj.n_variants
    for pid, Rp in enumerate(parent_orientations):
        # Pick `variants_per_parent` distinct variant indices
        chosen = rng.choice(nv, size=min(variants_per_parent, nv), replace=False)
        for vid in chosen:
            # child = variant @ parent + small noise
            noise_angle = rng.normal(0, noise_deg)
            noise_axis = _normalise(rng.normal(0, 1, 3))
            noise_R = Rotation.from_rotvec(
                np.radians(noise_angle) * noise_axis
            ).as_matrix()
            Rc = noise_R @ or_obj.variants[vid] @ Rp
            child_orientations.append(Rc)
            gt_parents.append(pid)
            gt_variants.append(int(vid))

    N = len(child_orientations)
    orientations = np.stack(child_orientations)

    # Build a simple chain-and-block adjacency:
    # grains from the same parent are adjacent to each other
    # (and to one grain from each neighbouring parent for realism)
    adjacency = [[] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j and gt_parents[i] == gt_parents[j]:
                adjacency[i].append(j)
    # Add inter-parent edges (first grain of consecutive parents)
    for pid in range(n_parent_grains - 1):
        # find first grains of each parent
        ia = next(k for k in range(N) if gt_parents[k] == pid)
        ib = next(k for k in range(N) if gt_parents[k] == pid + 1)
        adjacency[ia].append(ib)
        adjacency[ib].append(ia)

    gd = GrainData.from_arrays(orientations, adjacency)
    return gd, np.array(gt_parents), np.array(gt_variants)


# ===========================================================================
# Tests: Orientation Relationships
# ===========================================================================

class TestOrientationRelationships:

    def test_all_ors_exist(self):
        for name in ["KS", "NW", "Pitsch", "GT", "Cayron_NiTi"]:
            or_obj = get_OR(name)
            assert or_obj is not None

    def test_unknown_or_raises(self):
        with pytest.raises(KeyError):
            get_OR("NonexistentOR")

    def test_ks_has_24_variants(self, ks_or):
        assert ks_or.n_variants == 24, f"KS should have 24 variants, got {ks_or.n_variants}"

    def test_nw_has_12_variants(self, nw_or):
        assert nw_or.n_variants == 12, f"NW should have 12 variants, got {nw_or.n_variants}"

    def test_cayron_niti_has_12_variants(self):
        or_obj = get_OR("Cayron_NiTi")
        assert or_obj.n_variants == 12

    @pytest.mark.parametrize("or_name", ["KS", "NW", "Pitsch"])
    def test_variants_are_proper_rotations(self, or_name):
        or_obj = get_OR(or_name)
        for i, R in enumerate(or_obj.variants):
            det = np.linalg.det(R)
            RtR = R.T @ R
            assert abs(det - 1.0) < 1e-6, f"{or_name} variant {i}: det={det:.6f}"
            assert np.allclose(RtR, np.eye(3), atol=1e-6), (
                f"{or_name} variant {i}: R^T R not identity"
            )

    @pytest.mark.parametrize("or_name", ["KS", "NW", "Pitsch"])
    def test_variants_are_distinct(self, or_name):
        or_obj = get_OR(or_name)
        nv = or_obj.n_variants
        for i in range(nv):
            for j in range(i + 1, nv):
                M = or_obj.variants[i] @ or_obj.variants[j].T
                trace = np.clip((np.trace(M) - 1.0) / 2.0, -1.0, 1.0)
                angle = np.degrees(np.arccos(trace))
                assert angle > 0.5, (
                    f"{or_name} variants {i} and {j} are nearly identical (Δ={angle:.3f}°)"
                )

    def test_parent_from_child_roundtrip(self, ks_or):
        """
        Applying a variant to a parent and then inverting should recover the parent.
        """
        rng = np.random.default_rng(0)
        R_parent = Rotation.random(random_state=rng).as_matrix()
        for vid, R_var in enumerate(ks_or.variants):
            R_child = R_var @ R_parent
            R_parent_recovered = ks_or.parent_from_child(R_child, vid)
            angle = _misorientation_angle(R_parent, R_parent_recovered, [np.eye(3)])
            assert angle < 1e-4, (
                f"Round-trip failed for variant {vid}: Δ={angle:.6f}°"
            )

    def test_ks_parallel_planes_and_dirs(self, ks_or):
        """
        Verify KS prototype variant: (111)_fcc ‖ (011)_bcc, [1-10]_fcc ‖ [1-11]_bcc.
        """
        R0 = ks_or.variants[0]
        plane_fcc = _normalise(np.array([1., 1., 1.]))
        plane_bcc = _normalise(np.array([0., 1., 1.]))
        dir_fcc = _normalise(np.array([1., -1., 0.]))
        dir_bcc = _normalise(np.array([1., -1., 1.]))

        # After rotating by R0, parent planes/dirs should align with child
        rotated_plane = R0 @ plane_fcc
        rotated_dir = R0 @ dir_fcc
        angle_plane = np.degrees(np.arccos(np.clip(np.dot(rotated_plane, plane_bcc), -1, 1)))
        angle_dir = np.degrees(np.arccos(np.clip(np.dot(rotated_dir, dir_bcc), -1, 1)))
        assert angle_plane < 2.0, f"KS plane parallelism violated: {angle_plane:.2f}°"
        assert angle_dir < 5.0, f"KS direction parallelism violated: {angle_dir:.2f}°"


# ===========================================================================
# Tests: Misorientation and Graph utilities
# ===========================================================================

class TestMisorientationAndGraph:

    def test_identity_misorientation_is_zero(self):
        R = Rotation.random(random_state=1).as_matrix()
        angle = _misorientation_angle(R, R, _OH_SYM)
        assert angle < 1e-6

    def test_edge_probability_at_threshold_is_half(self):
        p = _edge_probability(5.0, threshold_deg=5.0, tolerance_deg=2.5)
        assert abs(p - 0.5) < 1e-4

    def test_edge_probability_below_threshold_exceeds_half(self):
        p = _edge_probability(2.0, threshold_deg=5.0, tolerance_deg=2.5)
        assert p > 0.5

    def test_edge_probability_above_threshold_below_half(self):
        p = _edge_probability(10.0, threshold_deg=5.0, tolerance_deg=2.5)
        assert p < 0.5

    def test_candidate_parents_shape(self, ks_or):
        rng = np.random.default_rng(5)
        R_child = Rotation.random(random_state=rng).as_matrix()
        parents = _candidate_parent_orientations(R_child, ks_or)
        assert parents.shape == (ks_or.n_variants, 3, 3)

    def test_candidate_parents_are_rotations(self, ks_or):
        rng = np.random.default_rng(6)
        R_child = Rotation.random(random_state=rng).as_matrix()
        parents = _candidate_parent_orientations(R_child, ks_or)
        for i, Rp in enumerate(parents):
            det = np.linalg.det(Rp)
            assert abs(det - 1.0) < 1e-5, f"Parent {i} det={det:.6f}"

    def test_grain_data_construction(self):
        N = 20
        rng = np.random.default_rng(7)
        orients = Rotation.random(N, random_state=rng).as_matrix()
        adj = [[j for j in range(max(0, i-2), min(N, i+3)) if j != i]
               for i in range(N)]
        gd = GrainData.from_arrays(orients, adj)
        assert gd.n_grains == N
        assert gd.orientations.shape == (N, 3, 3)


# ===========================================================================
# Tests: Variant Graph
# ===========================================================================

class TestVariantGraph:

    def test_variant_graph_builds(self, ks_or):
        N = 15
        rng = np.random.default_rng(10)
        orients = Rotation.random(N, random_state=rng).as_matrix()
        adj = [[j for j in range(max(0, i-2), min(N, i+3)) if j != i]
               for i in range(N)]
        gd = GrainData.from_arrays(orients, adj)
        vg = build_variant_graph(gd, ks_or, threshold_deg=5.0, tolerance_deg=3.0)
        assert vg.n_nodes == N * ks_or.n_variants
        assert vg.P.shape == (vg.n_nodes, vg.n_nodes)

    def test_variant_graph_diagonal_is_one(self, ks_or):
        N = 10
        rng = np.random.default_rng(11)
        orients = Rotation.random(N, random_state=rng).as_matrix()
        adj = [[j for j in range(max(0, i-2), min(N, i+3)) if j != i]
               for i in range(N)]
        gd = GrainData.from_arrays(orients, adj)
        vg = build_variant_graph(gd, ks_or)
        diag = vg.P.diagonal()
        assert np.all(np.abs(diag - 1.0) < 1e-5), "Diagonal should be all 1.0"

    def test_variant_graph_is_symmetric(self, ks_or):
        N = 10
        rng = np.random.default_rng(12)
        orients = Rotation.random(N, random_state=rng).as_matrix()
        adj = [[j for j in range(max(0, i-2), min(N, i+3)) if j != i]
               for i in range(N)]
        gd = GrainData.from_arrays(orients, adj)
        vg = build_variant_graph(gd, ks_or)
        diff = (vg.P - vg.P.T).data
        assert np.all(np.abs(diff) < 1e-5), "P matrix should be symmetric"

    def test_mcl_returns_labels_for_all_nodes(self, ks_or):
        N = 15
        rng = np.random.default_rng(13)
        orients = Rotation.random(N, random_state=rng).as_matrix()
        adj = [[j for j in range(max(0, i-2), min(N, i+3)) if j != i]
               for i in range(N)]
        gd = GrainData.from_arrays(orients, adj)
        vg = build_variant_graph(gd, ks_or)
        labels = markov_cluster(vg, n_iterations=20, verbose=False)
        assert len(labels) == vg.n_nodes


# ===========================================================================
# Tests: Full Reconstruction Pipeline (synthetic data)
# ===========================================================================

class TestFullReconstruction:

    def test_reconstruction_runs_on_synthetic_data(self):
        gd, gt_parents, gt_variants = _make_synthetic_grain_data(
            n_parent_grains=3, variants_per_parent=6, noise_deg=0.5
        )
        rec = ParentReconstructor(
            gd, or_name="KS", refine_or=False,
            threshold_deg=5.0, tolerance_deg=3.0,
            mcl_iterations=30
        )
        result = rec.run(verbose=False)
        assert result.n_parent_grains > 0
        assert result.parent_orientations.shape[0] == gd.n_grains

    def test_reconstruction_fraction_on_clean_data(self):
        """
        With very low noise and clear parent boundaries, reconstruction
        fraction should be reasonably high.
        """
        gd, gt_parents, gt_variants = _make_synthetic_grain_data(
            n_parent_grains=4, variants_per_parent=8,
            or_name="KS", noise_deg=0.3, seed=0
        )
        rec = ParentReconstructor(
            gd, or_name="KS", refine_or=False,
            threshold_deg=5.0, tolerance_deg=3.0,
            mcl_iterations=50
        )
        result = rec.run(verbose=False)
        assert result.reconstruction_fraction >= 0.5, (
            f"Expected ≥50% reconstruction on clean data, "
            f"got {result.reconstruction_fraction:.1%}"
        )

    def test_fit_values_are_finite_or_nan(self):
        gd, _, _ = _make_synthetic_grain_data(n_parent_grains=2, variants_per_parent=5)
        rec = ParentReconstructor(gd, or_name="KS", refine_or=False, mcl_iterations=20)
        result = rec.run(verbose=False)
        valid_fit = result.fit[~np.isnan(result.fit)]
        assert np.all(valid_fit >= 0), "Fit values should be non-negative"
        assert np.all(valid_fit <= 180), "Fit values should be ≤180°"

    def test_parent_orientations_are_proper_rotations(self):
        gd, _, _ = _make_synthetic_grain_data(n_parent_grains=2, variants_per_parent=5)
        rec = ParentReconstructor(gd, or_name="KS", refine_or=False, mcl_iterations=20)
        result = rec.run(verbose=False)
        for i, Rp in enumerate(result.parent_orientations):
            if np.any(np.isnan(Rp)):
                continue
            det = np.linalg.det(Rp)
            assert abs(det - 1.0) < 1e-4, f"Parent orientation {i}: det={det:.6f}"

    def test_result_summary_produces_string(self):
        gd, _, _ = _make_synthetic_grain_data(n_parent_grains=2, variants_per_parent=5)
        rec = ParentReconstructor(gd, or_name="KS", refine_or=False, mcl_iterations=10)
        result = rec.run(verbose=False)
        summary = result.summary()
        assert isinstance(summary, str)
        assert "OR used" in summary
        assert "Reconstructed grains" in summary


# ===========================================================================
# Tests: OR Refinement
# ===========================================================================

class TestORRefinement:

    def test_refiner_runs_and_returns_or(self):
        gd, _, _ = _make_synthetic_grain_data(
            n_parent_grains=3, variants_per_parent=6, noise_deg=1.0
        )
        or_obj = get_OR("KS")
        refiner = ORRefiner(gd, or_obj)
        refined = refiner.refine(max_pairs=50, verbose=False)
        assert refined.n_variants == or_obj.n_variants
        assert "refined" in refined.name

    def test_refined_or_variants_are_rotations(self):
        gd, _, _ = _make_synthetic_grain_data(n_parent_grains=2, variants_per_parent=5)
        or_obj = get_OR("KS")
        refiner = ORRefiner(gd, or_obj)
        refined = refiner.refine(max_pairs=30, verbose=False)
        for i, R in enumerate(refined.variants):
            det = np.linalg.det(R)
            assert abs(det - 1.0) < 1e-5, f"Refined variant {i}: det={det:.6f}"


# ===========================================================================
# Tests: Voting
# ===========================================================================

class TestVoting:

    def test_vote_recovers_parent_from_perfect_variants(self):
        """
        With zero noise, the voted parent should match the ground truth exactly.
        """
        or_obj = get_OR("KS")
        rng = np.random.default_rng(99)
        R_parent = Rotation.random(random_state=rng).as_matrix()
        nv = or_obj.n_variants
        chosen_variants = [0, 3, 7, 11, 15, 20]

        child_orients = np.stack([or_obj.variants[v] @ R_parent for v in chosen_variants])
        variant_ids = np.array(chosen_variants)

        mean_parent, mean_fit = _vote_parent_orientation(
            child_orients, variant_ids, or_obj
        )
        angle = _misorientation_angle(mean_parent, R_parent, _OH_SYM)
        assert angle < 1.0, f"Voted parent deviates {angle:.3f}° from ground truth"
        assert mean_fit < 1.0, f"Mean fit should be near 0°, got {mean_fit:.3f}°"

    def test_vote_stable_under_small_noise(self):
        """
        With 1° noise, voted parent should still be within 2° of ground truth.
        """
        or_obj = get_OR("KS")
        rng = np.random.default_rng(100)
        R_parent = Rotation.random(random_state=rng).as_matrix()
        chosen_variants = list(range(12))

        child_orients = []
        for v in chosen_variants:
            noise = np.radians(rng.normal(0, 1.0)) * _normalise(rng.normal(0, 1, 3))
            R_noise = Rotation.from_rotvec(noise).as_matrix()
            child_orients.append(R_noise @ or_obj.variants[v] @ R_parent)
        child_orients = np.stack(child_orients)

        mean_parent, _ = _vote_parent_orientation(
            child_orients, np.array(chosen_variants), or_obj
        )
        angle = _misorientation_angle(mean_parent, R_parent, _OH_SYM)
        assert angle < 2.0, f"Noisy voting deviated {angle:.3f}° from ground truth"


# ===========================================================================
# Integration test: detect + reconstruct
# ===========================================================================

def test_detect_and_reconstruct_end_to_end():
    """
    Full pipeline test: build synthetic data for KS, auto-detect the OR,
    and run reconstruction.
    """
    gd, gt_parents, _ = _make_synthetic_grain_data(
        n_parent_grains=3, variants_per_parent=8, or_name="KS", noise_deg=0.5
    )
    rec = ParentReconstructor(
        gd, or_name=None,   # force auto-detect
        refine_or=False,
        threshold_deg=5.0,
        tolerance_deg=3.0,
        mcl_iterations=30,
    )
    result = rec.run(verbose=False)
    assert result.n_parent_grains >= 1
    assert result.reconstruction_fraction > 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
