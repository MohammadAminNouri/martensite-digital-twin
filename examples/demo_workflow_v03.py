from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from martwin.workflows.digital_twin import TwinConfiguration, build_twin_model
from martwin.simulation.synthetic import generate_synthetic_child_map
from martwin.analysis.variant_analysis import assign_variants_known_parent_regions
from martwin.calibration.gap_analysis import assess_data_gaps
from martwin.explain import workflow_dataframe, data_requirement_table

OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

config = TwinConfiguration(material_system="NiTi B2→B19′", angular_tolerance_deg=5.0)
model = build_twin_model(config)
synth = generate_synthetic_child_map(model.variants, grid_shape=(20, 30), n_parent_grains=3, seed=33)
assignment = assign_variants_known_parent_regions(
    synth.child_orientations,
    synth.parent_orientations,
    synth.dataframe["parent_region_id"].tolist(),
    model.variants,
    child_sym_ops=model.child_sym_ops,
    tolerance_deg=config.angular_tolerance_deg,
)
gaps = assess_data_gaps("NiTi", {"composition": False, "heat_treatment": False, "ebsd_or_tkd": True, "DSC": False})

payload = {
    "version": "0.3",
    "n_variants": len(model.variants),
    "n_points": len(synth.dataframe),
    "mean_error_deg": assignment.mean_error_deg,
    "confidence_score": assignment.confidence_score,
    "missing_data": gaps.missing,
    "workflow_steps": len(workflow_dataframe()),
    "required_data_rows": len(data_requirement_table("NiTi", lpbf=False)),
}
(OUT / "v03_demo_summary.json").write_text(json.dumps(payload, indent=2))
print(json.dumps(payload, indent=2))
