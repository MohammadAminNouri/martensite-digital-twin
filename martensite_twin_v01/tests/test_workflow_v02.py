import numpy as np

from martwin.workflows.digital_twin import TwinConfiguration, build_twin_model, run_known_parent_analysis
from martwin.simulation.synthetic import generate_synthetic_child_map
from martwin.workflows.digital_twin import variant_library_tables
from martwin.io.ebsd_csv import dataframe_to_orientation_matrices


def test_build_model_and_variant_tables():
    model = build_twin_model(TwinConfiguration(material_system="NiTi B2→B19′"))
    tables = variant_library_tables(model)
    assert len(model.variants) > 0
    assert len(tables["variants"]) == len(model.variants)
    assert tables["misorientation_matrix_deg"].shape == (len(model.variants), len(model.variants))


def test_synthetic_workflow_runs():
    model = build_twin_model(TwinConfiguration(material_system="Steel fcc→bcc/bct", steel_or="KS"))
    synth = generate_synthetic_child_map(model.variants, grid_shape=(8, 10), n_parent_grains=2, seed=3)
    assert len(synth.dataframe) == 80
    mats = dataframe_to_orientation_matrices(synth.dataframe)
    assert len(mats) == 80
    assignment, recon, _ = run_known_parent_analysis(model, mats, parent_orientation=synth.parent_orientations[0])
    assert not assignment.assignments.empty
    assert len(recon.labels) == 80
    assert np.isfinite(assignment.mean_error_deg)
