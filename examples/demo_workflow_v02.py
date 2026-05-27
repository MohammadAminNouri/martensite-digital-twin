from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from martwin.workflows.digital_twin import TwinConfiguration, build_twin_model, run_known_parent_analysis
from martwin.simulation.synthetic import generate_synthetic_child_map
from martwin.reporting.report import build_markdown_report, build_json_report
from martwin.calibration.gap_analysis import assess_data_gaps
from martwin.analysis.variant_analysis import assign_variants_known_parent_regions


def main() -> None:
    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)

    config = TwinConfiguration(material_system="NiTi B2→B19′", beta_deg=96.8, angular_tolerance_deg=5.0)
    model = build_twin_model(config)
    synth = generate_synthetic_child_map(model.variants, grid_shape=(25, 40), n_parent_grains=3, seed=11)
    synth.dataframe.to_csv(out / "v02_synthetic_niti_dataset.csv", index=False)

    assignment = assign_variants_known_parent_regions(synth.child_orientations, parent_orientations=synth.parent_orientations, parent_region_ids=synth.dataframe["parent_region_id"].tolist(), variants=model.variants, child_sym_ops=model.child_sym_ops, tolerance_deg=model.config.angular_tolerance_deg)
    _, recon, _ = run_known_parent_analysis(model, synth.child_orientations, parent_orientation=synth.parent_orientations[0])
    gap_report = assess_data_gaps("NiTi", {"composition": False, "heat_treatment": False, "ebsd_or_tkd": True})
    metrics = {
        "n_points": len(synth.dataframe),
        "n_variants": len(model.variants),
        "mean_variant_error_deg": assignment.mean_error_deg,
        "assignment_confidence": assignment.confidence_score,
        "n_parent_clusters": len(set(recon.labels)),
        "data_confidence_score": gap_report.confidence_score,
    }
    (out / "v02_report.md").write_text(build_markdown_report(model, assignment.summary, metrics, gap_report), encoding="utf-8")
    (out / "v02_report.json").write_text(build_json_report(model, metrics, gap_report), encoding="utf-8")
    print("Wrote v0.2 demo outputs to", out)
    print(metrics)


if __name__ == "__main__":
    main()
