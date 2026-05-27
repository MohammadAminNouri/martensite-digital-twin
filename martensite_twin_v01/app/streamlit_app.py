from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from martwin.workflows.digital_twin import TwinConfiguration, build_twin_model, run_known_parent_analysis, variant_library_tables
from martwin.simulation.synthetic import generate_synthetic_child_map
from martwin.io.ebsd_csv import read_ebsd_csv, dataframe_to_orientation_matrices
from martwin.analysis.variant_analysis import assign_variants_known_parent_regions
from martwin.io.manifest import read_open_data_manifest
from martwin.kinetics.km import km_curve
from martwin.kinetics.niti_transform import NiTiTransformationTemperatures, linear_cooling_fraction, linear_heating_fraction_austenite
from martwin.calibration.gap_analysis import assess_data_gaps
from martwin.reporting.report import build_json_report, build_markdown_report
from martwin.visualization.maps import plot_variant_map

st.set_page_config(page_title="OpenMartensiteTwin v0.2", layout="wide")

st.title("OpenMartensiteTwin v0.2")
st.caption("Workflow version: configure → generate/import data → analyze variants → reconstruct parent clusters → assess gaps → export report.")

with st.sidebar:
    st.header("1. Configure twin")
    material_system = st.selectbox("Material system", ["NiTi B2→B19′", "Steel fcc→bcc/bct"])
    if material_system.startswith("NiTi"):
        beta = st.number_input("B19′ beta angle (°)", value=96.8, min_value=90.0, max_value=110.0, step=0.1)
        steel_or = "KS"
    else:
        beta = 96.8
        steel_or = st.selectbox("Steel OR", ["KS", "NW", "Pitsch"])
    tol = st.slider("Variant fit tolerance (°)", min_value=1.0, max_value=15.0, value=5.0, step=0.5)
    recon_thr = st.slider("Parent reconstruction threshold (°)", min_value=1.0, max_value=15.0, value=5.0, step=0.5)
    notes = st.text_area("Analyst notes", placeholder="Sample ID, heat treatment, LPBF parameters, data source...")

config = TwinConfiguration(
    material_system=material_system,
    beta_deg=beta,
    steel_or=steel_or,
    angular_tolerance_deg=tol,
    reconstruction_threshold_deg=recon_thr,
    notes=notes,
)
model = build_twin_model(config)

# Session state for data and analysis.
if "ebsd_df" not in st.session_state:
    st.session_state.ebsd_df = None
if "child_oris" not in st.session_state:
    st.session_state.child_oris = None
if "synthetic_parents" not in st.session_state:
    st.session_state.synthetic_parents = None
if "assignment_result" not in st.session_state:
    st.session_state.assignment_result = None
if "recon_result" not in st.session_state:
    st.session_state.recon_result = None
if "gap_report" not in st.session_state:
    st.session_state.gap_report = None

status_cols = st.columns(5)
status_cols[0].metric("Material", "NiTi" if material_system.startswith("NiTi") else "Steel")
status_cols[1].metric("OR", model.orientation_relationship.name.split()[0])
status_cols[2].metric("Variants", len(model.variants))
status_cols[3].metric("Tolerance", f"{tol:.1f}°")
status_cols[4].metric("Dataset", "loaded" if st.session_state.ebsd_df is not None else "none")

workflow_tabs = st.tabs([
    "A. Model",
    "B. Data",
    "C. Variant analysis",
    "D. Kinetics",
    "E. Data gaps",
    "F. Open data/tools",
    "G. Report export",
])

with workflow_tabs[0]:
    st.subheader("Crystallographic model")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.write("**Orientation relationship**")
        st.write(model.orientation_relationship.name)
        st.write(model.orientation_relationship.source_note)
        st.write("**OR matrix: child crystal → parent crystal**")
        st.dataframe(pd.DataFrame(model.orientation_relationship.matrix_child_to_parent, columns=["x", "y", "z"]), use_container_width=True)
    with c2:
        tables = variant_library_tables(model)
        st.write("**Variant library**")
        st.dataframe(tables["variants"].head(50), use_container_width=True)
        st.download_button(
            "Download variant library CSV",
            tables["variants"].to_csv(index=False),
            file_name="variant_library.csv",
            mime="text/csv",
        )
    st.write("**Pairwise theoretical variant misorientation matrix (deg)**")
    st.dataframe(tables["misorientation_matrix_deg"], use_container_width=True)

with workflow_tabs[1]:
    st.subheader("Data: import EBSD/TKD CSV or generate synthetic map")
    st.info("Supported CSV formats: x, y plus either matrix columns r00..r22 or Bunge Euler columns phi1, Phi, phi2. Vendor .ctf/.ang/.h5 support is planned through kikuchipy/orix.")
    data_mode = st.radio("Choose data source", ["Generate synthetic demo", "Upload EBSD/TKD CSV"], horizontal=True)

    if data_mode == "Generate synthetic demo":
        gc1, gc2, gc3, gc4 = st.columns(4)
        h = gc1.number_input("Rows", value=50, min_value=5, max_value=200, step=5)
        w = gc2.number_input("Columns", value=70, min_value=5, max_value=250, step=5)
        n_parents = gc3.number_input("Parent regions", value=4, min_value=1, max_value=12)
        noise = gc4.number_input("Orientation noise (°)", value=0.75, min_value=0.0, max_value=10.0, step=0.25)
        active_fraction = st.slider("Active variant fraction inside each parent region", 0.05, 1.0, 0.55, 0.05)
        seed = st.number_input("Random seed", value=7, min_value=0, max_value=99999)
        if st.button("Generate synthetic EBSD-like dataset", type="primary"):
            synth = generate_synthetic_child_map(
                model.variants,
                grid_shape=(int(h), int(w)),
                n_parent_grains=int(n_parents),
                active_variant_fraction=float(active_fraction),
                orientation_noise_deg=float(noise),
                seed=int(seed),
            )
            st.session_state.ebsd_df = synth.dataframe
            st.session_state.child_oris = synth.child_orientations
            st.session_state.synthetic_parents = synth.parent_orientations
            st.session_state.assignment_result = None
            st.session_state.recon_result = None
            st.success(f"Generated {len(synth.dataframe)} synthetic points.")
    else:
        uploaded = st.file_uploader("Upload EBSD/TKD CSV", type=["csv"])
        if uploaded is not None:
            try:
                df = read_ebsd_csv(uploaded)
                st.session_state.ebsd_df = df
                st.session_state.child_oris = list(df["orientation_matrix"])
                st.session_state.synthetic_parents = None
                st.session_state.assignment_result = None
                st.session_state.recon_result = None
                st.success(f"Loaded {len(df)} orientation points.")
            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")

    if st.session_state.ebsd_df is not None:
        df = st.session_state.ebsd_df
        st.write("**Dataset preview**")
        st.dataframe(df.drop(columns=["orientation_matrix"], errors="ignore").head(200), use_container_width=True)
        st.download_button("Download current dataset CSV", df.drop(columns=["orientation_matrix"], errors="ignore").to_csv(index=False), "current_ebsd_dataset.csv", "text/csv")
        if "true_variant_id" in df.columns:
            fig = plot_variant_map(df.rename(columns={"true_variant_id": "variant_id"}), value_col="variant_id", title="True synthetic variant map")
            st.pyplot(fig)

with workflow_tabs[2]:
    st.subheader("Variant assignment and parent reconstruction")
    if st.session_state.child_oris is None:
        st.warning("Load or generate a dataset first in tab B.")
    else:
        st.write("Analysis uses a known/assumed parent orientation. For synthetic data, you can use the first generated parent orientation. For real data, use identity first only as a test; serious work needs fitted/reconstructed parent orientation.")
        parent_mode = st.radio("Parent orientation", ["Identity / unknown prototype", "Use first synthetic parent orientation", "Use synthetic parent-region orientations"], horizontal=True)
        parent_orientation = np.eye(3)
        if parent_mode == "Use first synthetic parent orientation":
            if st.session_state.synthetic_parents:
                parent_orientation = st.session_state.synthetic_parents[0]
                st.success("Using first synthetic parent orientation.")
            else:
                st.warning("No synthetic parent stored. Falling back to identity.")
        available_data_for_analysis = {"composition": False, "heat_treatment": False, "ebsd_or_tkd": True}
        if st.button("Run variant + parent reconstruction analysis", type="primary"):
            if (
                parent_mode == "Use synthetic parent-region orientations"
                and st.session_state.synthetic_parents
                and "parent_region_id" in st.session_state.ebsd_df.columns
            ):
                # Best mode for validating the synthetic demo: each point uses its true parent region.
                assignment = assign_variants_known_parent_regions(
                    st.session_state.child_oris,
                    st.session_state.synthetic_parents,
                    st.session_state.ebsd_df["parent_region_id"].tolist(),
                    model.variants,
                    child_sym_ops=model.child_sym_ops,
                    tolerance_deg=model.config.angular_tolerance_deg,
                )
                _, recon, _ = run_known_parent_analysis(
                    model,
                    st.session_state.child_oris,
                    parent_orientation=np.eye(3),
                    available_data=available_data_for_analysis,
                )
            else:
                assignment, recon, _ = run_known_parent_analysis(
                    model,
                    st.session_state.child_oris,
                    parent_orientation=parent_orientation,
                    available_data=available_data_for_analysis,
                )
            st.session_state.assignment_result = assignment
            st.session_state.recon_result = recon
            result_df = st.session_state.ebsd_df.drop(columns=["orientation_matrix"], errors="ignore").copy()
            # Drop previous analysis columns before merging fresh results.
            result_df = result_df.drop(columns=["variant_id", "angular_error_deg", "fit_quality", "is_in_tolerance", "used_parent_region_id", "reconstructed_parent_cluster"], errors="ignore")
            result_df = result_df.merge(assignment.assignments, on="point_id", how="left")
            result_df["reconstructed_parent_cluster"] = recon.labels
            st.session_state.ebsd_df = result_df
            st.success("Analysis complete.")

        assignment = st.session_state.assignment_result
        recon = st.session_state.recon_result
        if assignment is not None:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Mean error", f"{assignment.mean_error_deg:.2f}°")
            m2.metric("Max error", f"{assignment.max_error_deg:.2f}°")
            m3.metric("Fit confidence", f"{assignment.confidence_score:.2f}")
            m4.metric("Parent clusters", len(set(recon.labels)) if recon else "—")
            st.write("**Variant population summary**")
            st.dataframe(assignment.summary, use_container_width=True)
            st.write("**Point-level assignments**")
            st.dataframe(st.session_state.ebsd_df.head(500), use_container_width=True)
            if {"x", "y", "variant_id"}.issubset(st.session_state.ebsd_df.columns):
                fig = plot_variant_map(st.session_state.ebsd_df, value_col="variant_id", title="Assigned variant map")
                st.pyplot(fig)
            if recon is not None and {"x", "y", "reconstructed_parent_cluster"}.issubset(st.session_state.ebsd_df.columns):
                fig2 = plot_variant_map(st.session_state.ebsd_df, value_col="reconstructed_parent_cluster", title="Prototype reconstructed parent clusters")
                st.pyplot(fig2)
            st.download_button("Download assignments CSV", st.session_state.ebsd_df.to_csv(index=False), "variant_assignments.csv", "text/csv")
            with st.expander("Reconstruction notes"):
                st.write(recon.notes if recon else "No reconstruction yet.")

with workflow_tabs[3]:
    st.subheader("Transformation kinetics quick model")
    if material_system.startswith("Steel"):
        kc1, kc2, kc3 = st.columns(3)
        Ms = kc1.number_input("Ms (°C)", value=350.0, step=10.0)
        alpha = kc2.number_input("KM alpha", value=0.011, step=0.001, format="%.4f")
        Tmin = kc3.number_input("Minimum temperature (°C)", value=20.0, step=10.0)
        temps = np.linspace(Ms + 100, Tmin, 180)
        frac = km_curve(list(temps), Ms_C=Ms, alpha=alpha)
        chart_df = pd.DataFrame({"Temperature_C": temps, "martensite_fraction": frac}).set_index("Temperature_C")
        st.line_chart(chart_df)
        st.caption("Koistinen–Marburger is a first-order model. It requires alloy-specific calibration and is not enough for bainite/tempering/property prediction.")
    else:
        nc1, nc2, nc3, nc4 = st.columns(4)
        Ms = nc1.number_input("Ms (°C)", value=30.0, step=5.0)
        Mf = nc2.number_input("Mf (°C)", value=-10.0, step=5.0)
        As = nc3.number_input("As (°C)", value=15.0, step=5.0)
        Af = nc4.number_input("Af (°C)", value=55.0, step=5.0)
        temps = np.linspace(min(Mf, As) - 30, max(Ms, Af) + 30, 200)
        tr = NiTiTransformationTemperatures(Ms_C=Ms, Mf_C=Mf, As_C=As, Af_C=Af)
        cooling = [linear_cooling_fraction(float(T), tr) for T in temps]
        heating = [linear_heating_fraction_austenite(float(T), tr) for T in temps]
        chart_df = pd.DataFrame({"Temperature_C": temps, "B19prime_fraction_cooling": cooling, "B2_fraction_heating": heating}).set_index("Temperature_C")
        st.line_chart(chart_df)
        st.caption("This is a DSC-calibrated placeholder curve. A real NiTi twin needs measured hysteresis, stress dependence, and composition/precipitation state.")

with workflow_tabs[4]:
    st.subheader("Data-gap and reliability assessment")
    st.write("Check only data that you truly have. The twin should never pretend that missing data is known.")
    keys = [
        "composition", "heat_treatment", "ebsd_or_tkd", "DSC", "XRD_lattice", "stress_strain", "oxygen_carbon", "thermal_history",
        "cooling_curve", "prior_austenite_reference", "hardness", "retained_austenite_XRD",
        "laser_parameters", "scan_strategy", "powder_chemistry", "melt_pool_or_thermal_model", "porosity", "residual_stress",
    ]
    cols = st.columns(3)
    available = {}
    for i, key in enumerate(keys):
        default = bool(key == "ebsd_or_tkd" and st.session_state.ebsd_df is not None)
        available[key] = cols[i % 3].checkbox(key, value=default)
    material_for_gaps = model.material_key
    if st.checkbox("This is LPBF/additive manufacturing data"):
        material_for_gaps += " lpbf"
    gap_report = assess_data_gaps(material_for_gaps, available)
    st.session_state.gap_report = gap_report
    g1, g2 = st.columns([0.4, 0.6])
    with g1:
        st.metric("Twin confidence", f"{gap_report.confidence_score:.2f}")
        st.write("**Missing data**")
        st.write(gap_report.missing)
    with g2:
        st.write("**Recommended next experiments/actions**")
        for item in gap_report.recommended_next_experiments:
            st.write(f"- {item}")

with workflow_tabs[5]:
    st.subheader("Open data/tool manifest")
    manifest_path = ROOT / "data" / "open_data_manifest" / "open_data_sources.csv"
    if manifest_path.exists():
        dfm = read_open_data_manifest(manifest_path)
        priority = st.multiselect("Filter priority", sorted(dfm["priority"].dropna().unique()), default=list(sorted(dfm["priority"].dropna().unique()))) if "priority" in dfm.columns else []
        if priority and "priority" in dfm.columns:
            dfm = dfm[dfm["priority"].isin(priority)]
        st.dataframe(dfm, use_container_width=True)
        st.download_button("Download manifest CSV", dfm.to_csv(index=False), "open_data_sources.csv", "text/csv")
    else:
        st.warning("Manifest not found.")

with workflow_tabs[6]:
    st.subheader("Export report")
    assignment = st.session_state.assignment_result
    gap_report = st.session_state.gap_report
    metrics = {
        "material_key": model.material_key,
        "orientation_relationship": model.orientation_relationship.name,
        "n_variants": len(model.variants),
        "dataset_points": int(len(st.session_state.ebsd_df)) if st.session_state.ebsd_df is not None else 0,
    }
    if assignment is not None:
        metrics.update({
            "mean_variant_error_deg": float(assignment.mean_error_deg),
            "max_variant_error_deg": float(assignment.max_error_deg),
            "assignment_confidence": float(assignment.confidence_score),
        })
    if gap_report is not None:
        metrics["data_confidence_score"] = float(gap_report.confidence_score)
    md = build_markdown_report(model, assignment.summary if assignment else None, metrics, gap_report, notes=notes)
    js = build_json_report(model, metrics, gap_report)
    st.download_button("Download Markdown report", md, "martensite_twin_report.md", "text/markdown")
    st.download_button("Download JSON report", js, "martensite_twin_report.json", "application/json")
    st.markdown(md)
