from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


import numpy as np

from martwin.core.rotations import axis_angle
from martwin.core.symmetry import cubic_proper_rotations, monoclinic_2_unique_axis_b
from martwin.crystallography.orientation_relationships import cayron_niti_natural_or
from martwin.crystallography.variants import generate_variants, identify_variant_for_known_parent, predict_child_orientations
from martwin.crystallography.parent_reconstruction import reconstruct_parent_orientations_greedy
from martwin.io.ebsd_csv import write_synthetic_ebsd_csv, read_orientation_matrix_csv
from martwin.visualization.simple_plots import plot_variant_ids
from martwin.calibration.gap_analysis import assess_data_gaps

OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# Parent B2 grain orientation in sample frame: synthetic, known ground truth.
parent_orientation = axis_angle([0.2, 0.5, 0.8], np.deg2rad(23.0))

orx = cayron_niti_natural_or(beta_deg=96.8)
variants = generate_variants(orx, cubic_proper_rotations(), monoclinic_2_unique_axis_b(), tol_deg=0.2)
child_predictions = predict_child_orientations(parent_orientation, variants)

# Build a synthetic 12x12 map using first 12 variants, with small deterministic noise-free blocks.
grid_shape = (12, 12)
chosen = []
variant_ids = []
for y in range(grid_shape[0]):
    for x in range(grid_shape[1]):
        v = variants[(x // 3 + 4 * (y // 3)) % min(12, len(variants))]
        chosen.append(parent_orientation @ v.matrix_child_to_parent)
        variant_ids.append(v.id)

csv_path = OUT / "synthetic_niti_b19prime_orientations.csv"
write_synthetic_ebsd_csv(csv_path, chosen, grid_shape=grid_shape)
loaded = read_orientation_matrix_csv(csv_path)

identified = [identify_variant_for_known_parent(Gc, parent_orientation, variants, child_sym_ops=monoclinic_2_unique_axis_b()) for Gc in loaded]
identified_ids = [r["variant_id"] for r in identified]
plot_variant_ids(identified_ids, grid_shape, OUT / "synthetic_niti_variant_map.png")

recon = reconstruct_parent_orientations_greedy(loaded, variants, threshold_deg=7.0)

gaps = assess_data_gaps("NiTi LPBF", {
    "composition": False,
    "heat_treatment": False,
    "ebsd_or_tkd": True,
    "DSC": False,
    "XRD_lattice": False,
    "stress_strain": False,
    "oxygen_carbon": False,
    "thermal_history": False,
    "laser_parameters": False,
    "scan_strategy": False,
    "powder_chemistry": False,
    "melt_pool_or_thermal_model": False,
    "porosity": False,
    "residual_stress": False,
})

summary = {
    "demo": "NiTi Cayron natural OR prototype",
    "orientation_relationship": orx.name,
    "source_note": orx.source_note,
    "number_of_unique_variants": len(variants),
    "synthetic_points": len(loaded),
    "mean_known_parent_variant_error_deg": float(np.mean([r["angular_error_deg"] for r in identified])),
    "prototype_reconstructed_parent_clusters": len(recon.parent_orientations),
    "gap_confidence_score_without_real_data": gaps.confidence_score,
    "missing_for_full_twin": gaps.missing,
    "recommended_next_experiments": gaps.recommended_next_experiments,
    "outputs": {
        "csv": str(csv_path.relative_to(ROOT)),
        "variant_map": "data/processed/synthetic_niti_variant_map.png",
    },
}

(OUT / "niti_demo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
