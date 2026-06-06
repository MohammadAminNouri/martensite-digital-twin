"""
tests/io/test_ebsd_readers.py
==============================
Unit and integration tests for the martwin CTF and ANG EBSD readers.

Generates synthetic .ctf and .ang files in a temp directory so the
tests run with zero external dependencies.

Run with:
    pytest tests/io/test_ebsd_readers.py -v
"""

from __future__ import annotations

import pathlib
import tempfile
import textwrap

import numpy as np
import pytest

# ---- Allow running from repo root ----
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from martwin.io import load, load_ctf, load_ang, oxford_to_edax, edax_to_oxford
from martwin.io.ebsd_data import EBSDData, Phase


# ===========================================================================
# Synthetic file generators
# ===========================================================================

def _make_ctf_content(n_cols=5, n_rows=4, n_phases=2, hex_grid=False) -> str:
    """Generate a minimal but valid CTF file as a string."""
    grid = "HexGrid" if hex_grid else "Grid"
    lines = [
        "Channel Text File",
        "Prj\ttest_project",
        "Author\tMartwin Test",
        f"JobMode\t{grid}",
        f"XCells\t{n_cols}",
        f"YCells\t{n_rows}",
        "XStep\t0.500",
        "YStep\t0.500",
        "AcqE1\t0.000",
        "AcqE2\t0.000",
        "AcqE3\t0.000",
        f"Phases\t{n_phases}",
    ]

    # Phase blocks
    lines += [
        "2.870;2.870;2.870;90.000;90.000;90.000\tIron bcc\tBCC",
        "11",    # Laue group → m-3m
        "229",   # Space group Im-3m
    ]
    if n_phases >= 2:
        lines += [
            "3.600;3.600;3.600;90.000;90.000;90.000\tIron fcc\tFCC",
            "11",
            "225",
        ]

    # Column header
    lines.append("Phase\tX\tY\tBands\tError\tEuler1\tEuler2\tEuler3\tMAD\tBC\tBS")

    # Data rows — deterministic orientations
    rng = np.random.default_rng(42)
    N = n_cols * n_rows
    for k in range(N):
        row_idx = k // n_cols
        col_idx = k % n_cols
        phase = 1 if rng.random() > 0.15 else 0    # 85% indexed
        x = col_idx * 0.5
        y = row_idx * 0.5
        bands = rng.integers(5, 9) if phase else 0
        error = 0 if phase else 1
        e1 = rng.uniform(0, 360) if phase else 0.0
        e2 = rng.uniform(0, 90)  if phase else 0.0
        e3 = rng.uniform(0, 360) if phase else 0.0
        mad = rng.uniform(0.1, 0.8) if phase else 0.0
        bc = rng.integers(100, 250) if phase else 0
        bs = rng.integers(200, 255) if phase else 0
        lines.append(
            f"{phase}\t{x:.3f}\t{y:.3f}\t{bands}\t{error}"
            f"\t{e1:.3f}\t{e2:.3f}\t{e3:.3f}\t{mad:.1f}\t{bc}\t{bs}"
        )

    return "\n".join(lines) + "\n"


def _make_ang_content(n_cols=5, n_rows=4, hex_grid=False) -> str:
    """Generate a minimal but valid ANG file as a string."""
    grid_tag = "HexGrid" if hex_grid else "SqrGrid"
    header = textwrap.dedent(f"""\
        # TEM_PIXperUM          1.000000
        # x-star                0.413492
        # y-star                0.892391
        # z-star                0.613825
        # WorkingDistance       14.999908
        #
        # Phase 1
        # MaterialName  Iron bcc
        # Formula       Fe
        # Info
        # Symmetry              43
        # LatticeConstants      2.870 2.870 2.870  90.000  90.000  90.000
        # NumberFamilies        4
        # hklFamilies      1  1  0 1 0.000000 1
        # ElasticConstants  -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 -1.000000
        # Categories 0 0 0 0 0
        #
        # Phase 2
        # MaterialName  Iron fcc
        # Formula       Fe
        # Info
        # Symmetry              43
        # LatticeConstants      3.600 3.600 3.600  90.000  90.000  90.000
        # NumberFamilies        4
        # hklFamilies      1  1  1 1 0.000000 1
        # ElasticConstants  -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 -1.000000
        # Categories 0 0 0 0 0
        #
        # GRID: {grid_tag}
        # XSTEP: 0.500000
        # YSTEP: 0.500000
        # NCOLS_ODD: {n_cols}
        # NCOLS_EVEN: {n_cols}
        # NROWS: {n_rows}
        #
        # OPERATOR:   MartwinTest
        #
        # SAMPLEID:
        #
        # SCANID:
        #
    """)

    rng = np.random.default_rng(123)
    N = n_cols * n_rows
    data_rows = []
    for k in range(N):
        row_idx = k // n_cols
        col_idx = k % n_cols
        phase = 1 if rng.random() > 0.15 else 0
        x = col_idx * 0.5
        y = row_idx * 0.5
        # ANG: Euler angles in radians
        e1 = rng.uniform(0, 2 * np.pi) if phase else 0.0
        e2 = rng.uniform(0, np.pi)     if phase else 0.0
        e3 = rng.uniform(0, 2 * np.pi) if phase else 0.0
        iq  = rng.integers(100, 300)
        ci  = round(rng.uniform(0.05, 0.99), 4) if phase else -1.0
        sem = rng.integers(0, 255)
        fit = round(rng.uniform(0.1, 2.0), 3) if phase else 0.0
        data_rows.append(
            f"  {e1:.5f}  {e2:.5f}  {e3:.5f}  {x:.5f}  {y:.5f}"
            f"  {iq:.1f}  {ci:.4f}  {phase}  {sem}  {fit:.3f}"
        )

    return header + "\n".join(data_rows) + "\n"


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield pathlib.Path(d)


@pytest.fixture(scope="module")
def ctf_file(tmp_dir):
    p = tmp_dir / "test_scan.ctf"
    p.write_text(_make_ctf_content(n_cols=5, n_rows=4, n_phases=2))
    return p


@pytest.fixture(scope="module")
def ang_file(tmp_dir):
    p = tmp_dir / "test_scan.ang"
    p.write_text(_make_ang_content(n_cols=5, n_rows=4))
    return p


@pytest.fixture(scope="module")
def ctf_single_phase(tmp_dir):
    p = tmp_dir / "single_phase.ctf"
    p.write_text(_make_ctf_content(n_cols=3, n_rows=3, n_phases=1))
    return p


# ===========================================================================
# Tests: CTF reader
# ===========================================================================

class TestCTFReader:

    def test_load_returns_ebsd_data(self, ctf_file):
        data = load_ctf(ctf_file)
        assert isinstance(data, EBSDData)

    def test_format_tag(self, ctf_file):
        data = load_ctf(ctf_file)
        assert data.file_format == "CTF"

    def test_pixel_count(self, ctf_file):
        data = load_ctf(ctf_file)
        assert data.n_pixels == 5 * 4

    def test_grid_dimensions(self, ctf_file):
        data = load_ctf(ctf_file)
        assert data.n_cols == 5
        assert data.n_rows == 4

    def test_step_size(self, ctf_file):
        data = load_ctf(ctf_file)
        assert abs(data.x_step - 0.5) < 1e-6
        assert abs(data.y_step - 0.5) < 1e-6

    def test_phase_count(self, ctf_file):
        data = load_ctf(ctf_file)
        assert len(data.phases) == 2
        assert 1 in data.phases
        assert 2 in data.phases

    def test_phase_names(self, ctf_file):
        data = load_ctf(ctf_file)
        assert "bcc" in data.phases[1].name.lower() or "iron" in data.phases[1].name.lower()

    def test_phase_lattice_params(self, ctf_file):
        data = load_ctf(ctf_file)
        lp = data.phases[1].lattice_params
        assert lp is not None
        assert len(lp) == 6
        assert abs(lp[0] - 2.870) < 0.01   # a parameter of BCC Fe

    def test_euler_angles_are_radians(self, ctf_file):
        """CTF Euler angles are stored in degrees and must be converted."""
        data = load_ctf(ctf_file)
        indexed = data.is_indexed
        # After conversion: phi1, phi2 in [0, 2π], Phi in [0, π]
        assert np.all(data.euler1[indexed] >= 0)
        assert np.all(data.euler1[indexed] <= 2 * np.pi + 1e-6)
        assert np.all(data.euler2[indexed] >= 0)
        assert np.all(data.euler2[indexed] <= np.pi + 1e-6)

    def test_euler_angles_not_in_degrees(self, ctf_file):
        """Verify that phi1 values are not still in degree scale (0–360)."""
        data = load_ctf(ctf_file)
        indexed = data.is_indexed
        if np.sum(indexed) > 0:
            assert float(data.euler1[indexed].max()) <= 2 * np.pi + 1e-4, (
                "euler1 max exceeds 2π — angles may still be in degrees"
            )

    def test_indexed_fraction_positive(self, ctf_file):
        data = load_ctf(ctf_file)
        assert np.sum(data.is_indexed) > 0

    def test_phase_id_range(self, ctf_file):
        data = load_ctf(ctf_file)
        assert int(data.phase_id.min()) >= 0
        assert int(data.phase_id.max()) <= 2

    def test_x_y_coordinates(self, ctf_file):
        data = load_ctf(ctf_file)
        assert float(data.x.min()) >= 0.0
        assert float(data.y.min()) >= 0.0

    def test_mad_values(self, ctf_file):
        data = load_ctf(ctf_file)
        indexed_mad = data.mad[data.is_indexed]
        assert np.all(indexed_mad >= 0)
        assert np.all(indexed_mad < 10.0)

    def test_bc_values(self, ctf_file):
        data = load_ctf(ctf_file)
        assert np.all(data.bc >= 0)

    def test_max_mad_filter(self, ctf_file):
        data_raw  = load_ctf(ctf_file, max_mad_deg=None)
        data_filt = load_ctf(ctf_file, max_mad_deg=0.5)
        assert np.sum(data_filt.is_indexed) <= np.sum(data_raw.is_indexed)

    def test_single_phase_file(self, ctf_single_phase):
        data = load_ctf(ctf_single_phase)
        assert data.n_pixels == 9
        assert 1 in data.phases

    def test_repr_contains_format(self, ctf_file):
        data = load_ctf(ctf_file)
        assert "CTF" in repr(data)

    def test_file_not_found_raises(self, tmp_dir):
        with pytest.raises(FileNotFoundError):
            load_ctf(tmp_dir / "nonexistent.ctf")


# ===========================================================================
# Tests: ANG reader
# ===========================================================================

class TestANGReader:

    def test_load_returns_ebsd_data(self, ang_file):
        data = load_ang(ang_file)
        assert isinstance(data, EBSDData)

    def test_format_tag(self, ang_file):
        data = load_ang(ang_file)
        assert data.file_format == "ANG"

    def test_pixel_count(self, ang_file):
        data = load_ang(ang_file)
        assert data.n_pixels == 5 * 4

    def test_grid_type(self, ang_file):
        data = load_ang(ang_file)
        assert data.grid_type == "square"

    def test_step_size(self, ang_file):
        data = load_ang(ang_file)
        assert abs(data.x_step - 0.5) < 0.01
        assert abs(data.y_step - 0.5) < 0.01

    def test_phase_count(self, ang_file):
        data = load_ang(ang_file)
        assert len(data.phases) >= 2

    def test_phase_names(self, ang_file):
        data = load_ang(ang_file)
        # Non-zero phases should have non-empty names
        non_zero = [p for pid, p in data.phases.items() if pid > 0]
        assert len(non_zero) > 0
        assert all(p.name != "" for p in non_zero)

    def test_phase_lattice_params(self, ang_file):
        data = load_ang(ang_file)
        # At least one non-zero phase should have lattice params or it should not crash
        for pid, phase in data.phases.items():
            if pid == 0:
                continue
            if phase.lattice_params is not None:
                assert len(phase.lattice_params) == 6

    def test_euler_angles_in_radians(self, ang_file):
        """ANG files store Euler angles natively in radians."""
        data = load_ang(ang_file)
        indexed = data.is_indexed
        if np.sum(indexed) > 0:
            assert np.all(data.euler1[indexed] >= 0)
            assert np.all(data.euler1[indexed] <= 2 * np.pi + 1e-4)
            assert np.all(data.euler2[indexed] >= 0)
            assert np.all(data.euler2[indexed] <= np.pi + 1e-4)

    def test_indexed_fraction_positive(self, ang_file):
        data = load_ang(ang_file)
        assert np.sum(data.is_indexed) > 0

    def test_ci_values(self, ang_file):
        data = load_ang(ang_file)
        ci = data.bs
        # Not-indexed pixels have CI = -1
        indexed_ci = ci[data.is_indexed]
        assert np.all(indexed_ci >= 0)
        assert np.all(indexed_ci <= 1.01)

    def test_min_ci_filter(self, ang_file):
        data_raw  = load_ang(ang_file, min_ci=None)
        data_filt = load_ang(ang_file, min_ci=0.5)
        assert np.sum(data_filt.is_indexed) <= np.sum(data_raw.is_indexed)

    def test_file_not_found_raises(self, tmp_dir):
        with pytest.raises(FileNotFoundError):
            load_ang(tmp_dir / "nonexistent.ang")


# ===========================================================================
# Tests: Unified load() dispatcher
# ===========================================================================

class TestUnifiedLoader:

    def test_load_ctf_by_extension(self, ctf_file):
        data = load(ctf_file)
        assert data.file_format == "CTF"

    def test_load_ang_by_extension(self, ang_file):
        data = load(ang_file)
        assert data.file_format == "ANG"

    def test_unknown_extension_raises(self, tmp_dir):
        p = tmp_dir / "scan.h5"
        p.write_text("dummy")
        with pytest.raises(ValueError, match="Unrecognised"):
            load(p)


# ===========================================================================
# Tests: EBSDData methods
# ===========================================================================

class TestEBSDDataMethods:

    def test_rotation_matrices_shape(self, ctf_file):
        data = load_ctf(ctf_file)
        R = data.as_rotation_matrices()
        assert R.shape == (data.n_pixels, 3, 3)

    def test_rotation_matrices_are_proper(self, ctf_file):
        data = load_ctf(ctf_file)
        R = data.as_rotation_matrices()
        indexed = data.is_indexed
        for i in np.where(indexed)[0][:10]:  # check first 10
            Ri = R[i]
            det = np.linalg.det(Ri)
            assert abs(det - 1.0) < 1e-6, f"Pixel {i}: det={det:.6f}"
            RtR = Ri.T @ Ri
            assert np.allclose(RtR, np.eye(3), atol=1e-6)

    def test_rotation_matrices_nan_for_unindexed(self, ctf_file):
        data = load_ctf(ctf_file)
        R = data.as_rotation_matrices()
        not_indexed = ~data.is_indexed
        assert np.all(np.isnan(R[not_indexed]))

    def test_euler_angles_deg_shape(self, ctf_file):
        data = load_ctf(ctf_file)
        e = data.euler_angles_deg
        assert e.shape == (data.n_pixels, 3)

    def test_euler_angles_deg_range(self, ctf_file):
        data = load_ctf(ctf_file)
        e = data.euler_angles_deg[data.is_indexed]
        assert np.all(e[:, 0] >= 0) and np.all(e[:, 0] <= 360.01)
        assert np.all(e[:, 1] >= 0) and np.all(e[:, 1] <= 180.01)

    def test_map_array_shape(self, ctf_file):
        data = load_ctf(ctf_file)
        m = data.map_array(data.bc)
        assert m.shape == (data.n_rows, data.n_cols)

    def test_phase_mask(self, ctf_file):
        data = load_ctf(ctf_file)
        mask1 = data.phase_mask(1)
        mask0 = data.phase_mask(0)
        assert (mask1 | mask0).sum() <= data.n_pixels
        assert np.sum(mask1) + np.sum(mask0) == data.n_pixels  # all pixels accounted for (2-phase test)

    def test_filter_by_mad(self, ctf_file):
        data = load_ctf(ctf_file)
        cleaned = data.filter_by_mad(0.3)
        assert np.sum(cleaned.is_indexed) <= np.sum(data.is_indexed)
        high_mad = data.mad[data.is_indexed] > 0.3
        if high_mad.any():
            assert np.sum(cleaned.is_indexed) < np.sum(data.is_indexed)

    def test_is_indexed_consistent_with_phase_id(self, ctf_file):
        data = load_ctf(ctf_file)
        assert np.all(data.is_indexed == (data.phase_id > 0))


# ===========================================================================
# Tests: Reference frame conversions
# ===========================================================================

class TestReferenceFrameConversions:

    def test_oxford_to_edax_y_flip(self, ctf_file):
        data = load_ctf(ctf_file)
        converted = oxford_to_edax(data)
        y_max = float(data.y.max())
        expected_y = y_max - data.y
        assert np.allclose(converted.y, expected_y, atol=1e-9)

    def test_oxford_to_edax_euler2_complement(self, ctf_file):
        data = load_ctf(ctf_file)
        converted = oxford_to_edax(data)
        indexed = data.is_indexed
        expected_e2 = np.pi - data.euler2[indexed]
        assert np.allclose(converted.euler2[indexed], expected_e2, atol=1e-9)

    def test_double_conversion_is_identity(self, ctf_file):
        """Applying oxford_to_edax twice should recover the original."""
        data = load_ctf(ctf_file)
        roundtrip = edax_to_oxford(oxford_to_edax(data))
        indexed = data.is_indexed
        assert np.allclose(data.euler1[indexed], roundtrip.euler1[indexed], atol=1e-9)
        assert np.allclose(data.euler2[indexed], roundtrip.euler2[indexed], atol=1e-9)
        assert np.allclose(data.euler3[indexed], roundtrip.euler3[indexed], atol=1e-9)


# ===========================================================================
# Tests: Hex grid
# ===========================================================================

class TestHexGrid:

    def test_ctf_hex_grid_loads(self, tmp_dir):
        p = tmp_dir / "hex.ctf"
        p.write_text(_make_ctf_content(n_cols=4, n_rows=3, n_phases=1, hex_grid=True))
        data = load_ctf(p)
        assert data.grid_type == "hexagonal"
        assert data.n_pixels == 4 * 3

    def test_ang_hex_grid_loads(self, tmp_dir):
        p = tmp_dir / "hex.ang"
        p.write_text(_make_ang_content(n_cols=4, n_rows=3, hex_grid=True))
        data = load_ang(p)
        assert data.grid_type == "hexagonal"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
