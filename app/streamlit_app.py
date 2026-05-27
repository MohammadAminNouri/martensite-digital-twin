from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from martwin.workflows.digital_twin import TwinConfiguration, build_twin_model, run_known_parent_analysis, variant_library_tables
from martwin.simulation.synthetic import generate_synthetic_child_map
from martwin.io.ebsd_csv import read_ebsd_csv
from martwin.analysis.variant_analysis import assign_variants_known_parent_regions
from martwin.io.manifest import read_open_data_manifest
from martwin.kinetics.km import km_curve
from martwin.kinetics.niti_transform import NiTiTransformationTemperatures, linear_cooling_fraction, linear_heating_fraction_austenite
from martwin.calibration.gap_analysis import assess_data_gaps
from martwin.reporting.report import build_json_report, build_markdown_report
from martwin.visualization.maps import plot_variant_map
from martwin.explain import (
    CONCEPTS,
    TABLE_EXPLANATIONS,
    MATURITY_LEVELS,
    workflow_dataframe,
    data_requirement_table,
    explain_columns,
)

st.set_page_config(page_title="OpenMartensiteTwin v0.3", layout="wide", initial_sidebar_state="expanded")


def css() -> None:
    st.markdown(
        """
        <style>
        .small-muted { color: #9ca3af; font-size: 0.92rem; }
        .card { border: 1px solid rgba(255,255,255,0.14); border-radius: 14px; padding: 1rem; background: rgba(255,255,255,0.035); margin-bottom: 0.7rem; }
        .good { color: #22c55e; font-weight: 700; }
        .warn { color: #f59e0b; font-weight: 700; }
        .bad { color: #ef4444; font-weight: 700; }
        .step { font-weight: 700; color: #e5e7eb; }
        </style>
        """,
        unsafe_allow_html=True,
    )


css()


def card(title: str, body: str, footer: str | None = None) -> None:
    foot = f"<div class='small-muted'>{footer}</div>" if footer else ""
    st.markdown(f"<div class='card'><b>{title}</b><br>{body}{foot}</div>", unsafe_allow_html=True)


def explain_table(key: str, columns: list[str] | None = None) -> None:
    meta = TABLE_EXPLANATIONS[key]
    with st.expander(f"How to read this: {meta['title']}", expanded=True):
        st.markdown(f"**What it is:** {meta['what']}")
        st.markdown(f"**Why it matters:** {meta['why']}")
        st.markdown(f"**How to read it:** {meta['how_to_read']}")
        st.markdown(f"**Where it is used later:** {meta['used_downstream']}")
        if columns:
            st.markdown("**Column meanings**")
            st.dataframe(explain_columns(columns), hide_index=True, use_container_width=True)


def maturity_from_gap_score(score: float, has_dataset: bool, has_kinetics_data: bool, lpbf: bool) -> tuple[str, str]:
    if not has_dataset:
        return "L0", "Theoretical crystallography only: no EBSD/TKD map is loaded."
    if score < 0.35:
        return "L1", "EBSD/TKD interpretation prototype: orientation map exists, but calibration data are mostly missing."
    if score < 0.65:
        return "L2", "Partly calibrated transformation twin: some measurements exist, but process/property links are incomplete."
    if score < 0.85 or not lpbf:
        return "L3", "Process-aware research twin: enough evidence for meaningful comparisons, but still needs independent validation."
    return "L4", "Predictive engineering twin candidate: high data coverage; validate on independent samples before decisions."


def make_niti_kinetics_plot(temps: np.ndarray, cooling: list[float], heating: list[float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(temps, cooling, label="B19′ martensite fraction during cooling")
    ax.plot(temps, heating, label="B2 austenite fraction during heating")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Phase fraction (0 to 1)")
    ax.set_title("Simplified NiTi transformation hysteresis")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    return fig


def make_steel_kinetics_plot(temps: np.ndarray, frac: list[float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(temps, frac, label="Martensite fraction during cooling")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Martensite fraction (0 to 1)")
    ax.set_title("Koistinen–Marburger first-order steel martensite model")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    return fig


def init_state() -> None:
    defaults: dict[str, Any] = {
        "ebsd_df": None,
        "child_oris": None,
        "synthetic_parents": None,
        "assignment_result": None,
        "recon_result": None,
        "gap_report": None,
        "last_kinetics_df": None,
        "available_data": {},
        "sample_record": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# Sidebar configuration
with st.sidebar:
    st.header("Build the twin")
    material_system = st.selectbox("Material system", ["NiTi B2→B19′", "Steel fcc→bcc/bct"], help="Choose the parent→martensite transformation family.")
    is_niti = material_system.startswith("NiTi")
    if is_niti:
        beta = st.number_input("B19′ monoclinic beta angle (°)", value=96.8, min_value=90.0, max_value=110.0, step=0.1, help="Sample-specific B19′ lattice parameter. Default is a prototype value; replace with XRD/refined data for real work.")
        steel_or = "KS"
    else:
        beta = 96.8
        steel_or = st.selectbox("Steel orientation relationship", ["KS", "NW", "Pitsch"], help="Comparator OR used to generate fcc→bcc/bct martensite variants.")
    tol = st.slider("Variant fit tolerance (°)", min_value=1.0, max_value=15.0, value=5.0, step=0.5, help="Maximum angular mismatch accepted when assigning a measured point to a theoretical variant.")
    recon_thr = st.slider("Parent reconstruction threshold (°)", min_value=1.0, max_value=15.0, value=5.0, step=0.5, help="Prototype clustering threshold for parent reconstruction.")
    lpbf = st.checkbox("LPBF/additive manufacturing route", value=False, help="Adds AM-specific data requirements such as powder chemistry, scan strategy, thermal history, porosity, and residual stress.")
    education_mode = st.checkbox("Show teaching explanations", value=True)

    st.divider()
    st.subheader("Sample/process record")
    sample_id = st.text_input("Sample ID", value="demo_sample_001")
    process_route = st.selectbox("Process route", ["unknown", "cast/wrought", "heat treated", "LPBF", "cold worked + annealed", "literature dataset"], index=0)
    composition_note = st.text_area("Composition / chemistry", placeholder="Example: Ni 50.8 at.%, Ti balance, O < 500 ppm; or Fe-0.2C-1.5Mn...")
    heat_treatment_note = st.text_area("Heat treatment / thermal cycle", placeholder="Example: solution 850°C 30 min, water quench, age 500°C 30 min...")
    analyst_notes = st.text_area("Analyst notes", placeholder="EBSD settings, data source, uncertainty, literature DOI, lab notes...")

config = TwinConfiguration(
    material_system=material_system,
    beta_deg=beta,
    steel_or=steel_or,
    angular_tolerance_deg=tol,
    reconstruction_threshold_deg=recon_thr,
    notes=analyst_notes,
)
model = build_twin_model(config)

st.session_state.sample_record = {
    "sample_id": sample_id,
    "material_system": material_system,
    "process_route": process_route,
    "composition_note": composition_note,
    "heat_treatment_note": heat_treatment_note,
    "lpbf": lpbf,
    "analyst_notes": analyst_notes,
}

# Header
st.title("OpenMartensiteTwin v0.3")
st.caption("Guided martensitic-transformation digital twin: data record → crystallography → EBSD/TKD variants → parent reconstruction → kinetics → reliability → report.")

# Data availability from state
has_dataset = st.session_state.ebsd_df is not None
material_key_for_gaps = model.material_key + (" lpbf" if lpbf else "")
if not st.session_state.available_data:
    st.session_state.available_data = {}
base_available = dict(st.session_state.available_data)
base_available["ebsd_or_tkd"] = bool(has_dataset or base_available.get("ebsd_or_tkd", False))
base_available["composition"] = bool(composition_note.strip()) or bool(base_available.get("composition", False))
base_available["heat_treatment"] = bool(heat_treatment_note.strip()) or bool(base_available.get("heat_treatment", False))
current_gap_report = assess_data_gaps(material_key_for_gaps, base_available)
level, level_note = maturity_from_gap_score(current_gap_report.confidence_score, has_dataset, bool(base_available.get("DSC") or base_available.get("cooling_curve")), lpbf)
st.session_state.gap_report = current_gap_report

mcols = st.columns(6)
mcols[0].metric("Twin level", level)
mcols[1].metric("Material", "NiTi" if is_niti else "Steel")
mcols[2].metric("OR", model.orientation_relationship.name.split()[0])
mcols[3].metric("Variants", len(model.variants))
mcols[4].metric("Dataset points", len(st.session_state.ebsd_df) if has_dataset else 0)
mcols[5].metric("Data confidence", f"{current_gap_report.confidence_score:.2f}")
st.info(level_note)

if education_mode:
    with st.expander("What makes this a digital twin, and what is still missing?", expanded=True):
        st.write(CONCEPTS["digital_twin"]["plain"])
        st.write(CONCEPTS["digital_twin"]["used_for"])
        st.dataframe(pd.DataFrame(MATURITY_LEVELS), hide_index=True, use_container_width=True)
        st.warning("v0.3 is still a research prototype. It becomes a real calibrated twin only when measured EBSD/TKD, DSC/dilatometry, XRD, composition, processing history, and mechanical validation are supplied.")

workflow_tabs = st.tabs([
    "0. Workflow",
    "1. Data record",
    "2. Crystallography",
    "3. EBSD/TKD data",
    "4. Variant analysis",
    "5. Kinetics",
    "6. Reliability",
    "7. Open data/tools",
    "8. Report",
])

with workflow_tabs[0]:
    st.header("How the twin works")
    st.write("This tab explains the logic before showing tables and graphs. Every table in the app is either an **input**, a **model assumption**, an **analysis result**, or a **reliability check**.")
    st.dataframe(workflow_dataframe(), hide_index=True, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        card("Input layer", "Composition, heat treatment, process route, thermal history, and experimental files.", "Without these, the twin runs in theoretical/demo mode.")
    with c2:
        card("Physics layer", "Orientation relationships, symmetry operations, variants, misorientation, and transformation kinetics.", "This is the engine, not the final truth.")
    with c3:
        card("Feedback layer", "EBSD/TKD/XRD/DSC/mechanical data are used to compare, calibrate, and score reliability.", "This is what turns a model into a twin.")

    st.subheader("One-click demo")
    st.write("This creates a synthetic EBSD-like map, assigns variants using the known synthetic parent regions, and produces maps/statistics. It proves the workflow works; it is not real experimental validation.")
    if st.button("Run complete synthetic twin demo", type="primary"):
        synth = generate_synthetic_child_map(
            model.variants,
            grid_shape=(50, 70),
            n_parent_grains=4,
            active_variant_fraction=0.55,
            orientation_noise_deg=0.75,
            seed=7,
        )
        st.session_state.ebsd_df = synth.dataframe
        st.session_state.child_oris = synth.child_orientations
        st.session_state.synthetic_parents = synth.parent_orientations
        assignment = assign_variants_known_parent_regions(
            st.session_state.child_oris,
            st.session_state.synthetic_parents,
            st.session_state.ebsd_df["parent_region_id"].tolist(),
            model.variants,
            child_sym_ops=model.child_sym_ops,
            tolerance_deg=model.config.angular_tolerance_deg,
        )
        _, recon, _ = run_known_parent_analysis(model, st.session_state.child_oris, parent_orientation=np.eye(3), available_data={"ebsd_or_tkd": True})
        result_df = st.session_state.ebsd_df.drop(columns=["orientation_matrix"], errors="ignore").copy()
        result_df = result_df.merge(assignment.assignments, on="point_id", how="left")
        result_df["reconstructed_parent_cluster"] = recon.labels
        st.session_state.ebsd_df = result_df
        st.session_state.assignment_result = assignment
        st.session_state.recon_result = recon
        st.session_state.available_data["ebsd_or_tkd"] = True
        st.success("Synthetic demo completed. Go to tabs 3–4 to inspect the dataset, variants, and maps.")

with workflow_tabs[1]:
    st.header("Material, process, and evidence record")
    st.write("A digital twin needs a record of what is measured, what is assumed, and what is unknown. This tab is the evidence ledger.")
    rec = pd.DataFrame([
        {"field": "sample_id", "value": sample_id, "why it matters": "Connects all files/results to one physical or synthetic sample."},
        {"field": "material_system", "value": material_system, "why it matters": "Selects the transformation model and required data."},
        {"field": "process_route", "value": process_route, "why it matters": "Processing controls grain structure, texture, stresses, phases, and transformation path."},
        {"field": "composition_note", "value": composition_note or "missing", "why it matters": "Chemistry strongly affects lattice parameters, driving force, and transformation temperatures."},
        {"field": "heat_treatment_note", "value": heat_treatment_note or "missing", "why it matters": "Heat treatment controls precipitation, residual stress, phase state, and transformation response."},
        {"field": "LPBF/AM", "value": str(lpbf), "why it matters": "If true, process thermal history and residual stress become essential."},
    ])
    st.dataframe(rec, hide_index=True, use_container_width=True)

    st.subheader("Data requirements for this material system")
    req = data_requirement_table("NiTi" if is_niti else "Steel", lpbf=lpbf)
    st.dataframe(req, hide_index=True, use_container_width=True)
    st.caption("These are not decorative fields. Each missing item reduces what the twin can honestly predict.")

with workflow_tabs[2]:
    st.header("Crystallographic model")
    if education_mode:
        st.write(CONCEPTS["orientation_relationship"]["plain"])
        st.write(CONCEPTS["variant"]["plain"])

    tables = variant_library_tables(model)
    or_df = pd.DataFrame(model.orientation_relationship.matrix_child_to_parent, index=["x_parent", "y_parent", "z_parent"], columns=["x_child", "y_child", "z_child"])
    vdf = tables["variants"].copy()
    mdf = tables["misorientation_matrix_deg"].copy().round(3)

    c1, c2 = st.columns([1.0, 1.1])
    with c1:
        st.subheader("Orientation relationship")
        st.write(f"**Model:** {model.orientation_relationship.name}")
        st.write(f"**Parent phase:** `{model.orientation_relationship.parent_phase}`")
        st.write(f"**Martensite/child phase:** `{model.orientation_relationship.child_phase}`")
        st.write(model.orientation_relationship.source_note)
        st.dataframe(or_df.round(4), use_container_width=True)
        if education_mode:
            explain_table("or_matrix", list(or_df.columns))
    with c2:
        st.subheader("Variant library")
        st.dataframe(vdf.head(100).round(4), use_container_width=True)
        st.download_button("Download variant library CSV", vdf.to_csv(index=False), file_name="variant_library.csv", mime="text/csv")
        if education_mode:
            explain_table("variant_library", list(vdf.columns))

    st.subheader("Theoretical variant-pair misorientation")
    st.dataframe(mdf, use_container_width=True)
    if education_mode:
        explain_table("misorientation_matrix", list(map(str, mdf.columns)))
        st.markdown("**Why many values repeat:** symmetry collapses many different rotations into equivalent angles. That is normal for crystallographic variant families.")

with workflow_tabs[3]:
    st.header("EBSD/TKD data")
    st.write("This is where the twin receives experimental orientation data. For now v0.3 supports CSV matrices/Euler angles and synthetic maps. Vendor .ctf/.ang/.h5 import remains a v0.4 target.")
    data_mode = st.radio("Data source", ["Generate synthetic demo", "Upload EBSD/TKD CSV"], horizontal=True)

    if data_mode == "Generate synthetic demo":
        gc1, gc2, gc3, gc4 = st.columns(4)
        h = gc1.number_input("Rows", value=50, min_value=5, max_value=200, step=5, help="Number of grid rows in the synthetic orientation map.")
        w = gc2.number_input("Columns", value=70, min_value=5, max_value=250, step=5, help="Number of grid columns in the synthetic orientation map.")
        n_parents = gc3.number_input("Parent regions", value=4, min_value=1, max_value=12, help="Number of synthetic parent grains/regions.")
        noise = gc4.number_input("Orientation noise (°)", value=0.75, min_value=0.0, max_value=10.0, step=0.25, help="Random angular noise added to orientations to mimic measurement scatter.")
        active_fraction = st.slider("Active variant fraction inside each parent region", 0.05, 1.0, 0.55, 0.05, help="Fraction of theoretical variants allowed to appear in each synthetic parent region.")
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
            st.session_state.available_data["ebsd_or_tkd"] = True
            st.success(f"Generated {len(synth.dataframe)} synthetic points.")
    else:
        st.info("CSV can contain x,y and either r00..r22 rotation-matrix columns or Bunge Euler columns phi1,Phi,phi2. Optional columns: point_id, phase, grain_id, ci, iq.")
        uploaded = st.file_uploader("Upload EBSD/TKD CSV", type=["csv"])
        if uploaded is not None:
            try:
                df = read_ebsd_csv(uploaded)
                st.session_state.ebsd_df = df
                st.session_state.child_oris = list(df["orientation_matrix"])
                st.session_state.synthetic_parents = None
                st.session_state.assignment_result = None
                st.session_state.recon_result = None
                st.session_state.available_data["ebsd_or_tkd"] = True
                st.success(f"Loaded {len(df)} orientation points.")
            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")

    if st.session_state.ebsd_df is None:
        st.warning("No dataset loaded yet. Generate a synthetic demo or upload a CSV.")
    else:
        df = st.session_state.ebsd_df
        preview = df.drop(columns=["orientation_matrix"], errors="ignore")
        st.subheader("Dataset preview")
        st.dataframe(preview.head(300), use_container_width=True)
        if education_mode:
            explain_table("dataset_preview", list(preview.columns))
        st.download_button("Download current dataset CSV", preview.to_csv(index=False), "current_ebsd_dataset.csv", "text/csv")
        if {"x", "y", "true_variant_id"}.issubset(df.columns):
            st.subheader("Synthetic ground-truth variant map")
            fig = plot_variant_map(df.rename(columns={"true_variant_id": "variant_id"}), value_col="variant_id", title="Synthetic ground-truth variant ID at each point")
            st.pyplot(fig)
            st.caption("This map exists only for synthetic data. Real EBSD/TKD does not come with a true_variant_id; the twin must infer variant_id from the measured orientation.")

with workflow_tabs[4]:
    st.header("Variant assignment and parent reconstruction")
    st.write("This is the first real analysis step: compare every measured/synthetic orientation with the theoretical variant library.")
    if st.session_state.child_oris is None:
        st.warning("Load or generate data in tab 3 first.")
    else:
        parent_mode = st.radio(
            "Parent-orientation assumption",
            ["Identity / unknown prototype", "Use first synthetic parent orientation", "Use synthetic parent-region orientations"],
            horizontal=True,
            help="For real data, identity is only a weak prototype. Publication-grade work needs OR fitting and graph-based parent reconstruction.",
        )
        parent_orientation = np.eye(3)
        if parent_mode == "Use first synthetic parent orientation":
            if st.session_state.synthetic_parents:
                parent_orientation = st.session_state.synthetic_parents[0]
                st.success("Using first synthetic parent orientation.")
            else:
                st.warning("No synthetic parent stored. Falling back to identity.")

        if st.button("Run variant assignment + prototype parent reconstruction", type="primary"):
            if (
                parent_mode == "Use synthetic parent-region orientations"
                and st.session_state.synthetic_parents
                and "parent_region_id" in st.session_state.ebsd_df.columns
            ):
                assignment = assign_variants_known_parent_regions(
                    st.session_state.child_oris,
                    st.session_state.synthetic_parents,
                    st.session_state.ebsd_df["parent_region_id"].tolist(),
                    model.variants,
                    child_sym_ops=model.child_sym_ops,
                    tolerance_deg=model.config.angular_tolerance_deg,
                )
                _, recon, _ = run_known_parent_analysis(model, st.session_state.child_oris, parent_orientation=np.eye(3), available_data={"ebsd_or_tkd": True})
            else:
                assignment, recon, _ = run_known_parent_analysis(model, st.session_state.child_oris, parent_orientation=parent_orientation, available_data={"ebsd_or_tkd": True})

            result_df = st.session_state.ebsd_df.drop(columns=["orientation_matrix"], errors="ignore").copy()
            result_df = result_df.drop(columns=["variant_id", "angular_error_deg", "fit_quality", "is_in_tolerance", "used_parent_region_id", "reconstructed_parent_cluster"], errors="ignore")
            result_df = result_df.merge(assignment.assignments, on="point_id", how="left")
            result_df["reconstructed_parent_cluster"] = recon.labels
            st.session_state.ebsd_df = result_df
            st.session_state.assignment_result = assignment
            st.session_state.recon_result = recon
            st.success("Analysis complete.")

        assignment = st.session_state.assignment_result
        recon = st.session_state.recon_result
        if assignment is None:
            st.info("Run the analysis to create variant maps, error metrics, and parent clusters.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Mean angular error", f"{assignment.mean_error_deg:.2f}°")
            m2.metric("Max angular error", f"{assignment.max_error_deg:.2f}°")
            m3.metric("Variant fit confidence", f"{assignment.confidence_score:.2f}")
            m4.metric("Prototype parent clusters", len(set(recon.labels)) if recon else "—")
            if education_mode:
                st.markdown("- **Mean angular error**: average mismatch between measured/synthetic orientations and the nearest theoretical variant. Lower is better.")
                st.markdown("- **Fit confidence**: simple normalized quality score. It is not a validated uncertainty model yet.")
                st.markdown("- **Parent clusters**: prototype grouping of points that may share a parent orientation. v0.3 uses a simple method; graph-based reconstruction is next.")

            st.subheader("Variant population summary")
            summary = assignment.summary.copy()
            st.dataframe(summary, use_container_width=True)
            if education_mode:
                explain_table("variant_summary", list(summary.columns))

            st.subheader("Point-level assignments")
            point_df = st.session_state.ebsd_df.copy()
            st.dataframe(point_df.head(500), use_container_width=True)
            if education_mode:
                explain_table("point_assignments", list(point_df.columns))

            if {"x", "y", "variant_id"}.issubset(st.session_state.ebsd_df.columns):
                st.subheader("Assigned variant map")
                fig = plot_variant_map(st.session_state.ebsd_df, value_col="variant_id", title="Assigned martensite variant at each EBSD/TKD point")
                st.pyplot(fig)
                st.caption("Each color/number is the nearest theoretical variant ID. Spatial regions of the same or related variants can indicate variant selection, packets, twins, or transformation accommodation. Validate with real EBSD quality and OR fitting before publication.")
            if recon is not None and {"x", "y", "reconstructed_parent_cluster"}.issubset(st.session_state.ebsd_df.columns):
                st.subheader("Prototype reconstructed parent-cluster map")
                fig2 = plot_variant_map(st.session_state.ebsd_df, value_col="reconstructed_parent_cluster", title="Prototype parent-cluster labels")
                st.pyplot(fig2)
                st.caption("This is not yet final parent-grain reconstruction. It is a clustering prototype showing which points may share a parent orientation. v0.4 should replace this with graph-based PGR and OR refinement.")
            st.download_button("Download assignments CSV", st.session_state.ebsd_df.to_csv(index=False), "variant_assignments.csv", "text/csv")
            with st.expander("Algorithm notes and current limitations", expanded=education_mode):
                st.write(recon.notes if recon else "No reconstruction notes available.")
                st.warning("If a real dataset gives high angular error, possible causes include wrong OR, wrong phase symmetry, wrong Euler convention, wrong parent orientation, pseudosymmetry, poor EBSD indexing, or actual transformation physics not represented by the current model.")

with workflow_tabs[5]:
    st.header("Transformation kinetics")
    st.write("Kinetics converts thermal path into phase fraction. This is necessary for a real twin because variant crystallography alone does not say how much martensite forms.")
    if not is_niti:
        kc1, kc2, kc3 = st.columns(3)
        Ms = kc1.number_input("Ms (°C)", value=350.0, step=10.0, help="Martensite start temperature. Needs alloy/process calibration.")
        alpha = kc2.number_input("Koistinen–Marburger alpha", value=0.011, step=0.001, format="%.4f", help="Empirical rate parameter. Common first approximation, not universal.")
        Tmin = kc3.number_input("Minimum temperature (°C)", value=20.0, step=10.0, help="Lowest temperature reached during cooling.")
        temps = np.linspace(Ms + 100, Tmin, 180)
        frac = km_curve(list(temps), Ms_C=Ms, alpha=alpha)
        kdf = pd.DataFrame({"Temperature_C": temps, "martensite_fraction": frac})
        st.session_state.last_kinetics_df = kdf
        st.pyplot(make_steel_kinetics_plot(temps, frac))
        st.dataframe(kdf.head(20), hide_index=True, use_container_width=True)
        with st.expander("How to read this graph", expanded=education_mode):
            st.markdown("**x-axis:** temperature during cooling. **y-axis:** predicted martensite fraction from 0 to 1.")
            st.markdown("Above Ms, the model predicts no fresh martensite. Below Ms, martensite fraction increases according to the KM equation. This does not model bainite, carbon partitioning, tempering, retained austenite stabilization, or mechanical properties.")
            st.dataframe(explain_columns(list(kdf.columns)), hide_index=True, use_container_width=True)
    else:
        nc1, nc2, nc3, nc4 = st.columns(4)
        Ms = nc1.number_input("Ms (°C)", value=30.0, step=5.0, help="Martensite start during cooling.")
        Mf = nc2.number_input("Mf (°C)", value=-10.0, step=5.0, help="Martensite finish during cooling.")
        As = nc3.number_input("As (°C)", value=15.0, step=5.0, help="Austenite/B2 start during heating.")
        Af = nc4.number_input("Af (°C)", value=55.0, step=5.0, help="Austenite/B2 finish during heating.")
        temps = np.linspace(min(Mf, As) - 30, max(Ms, Af) + 30, 200)
        tr = NiTiTransformationTemperatures(Ms_C=Ms, Mf_C=Mf, As_C=As, Af_C=Af)
        cooling = [linear_cooling_fraction(float(T), tr) for T in temps]
        heating = [linear_heating_fraction_austenite(float(T), tr) for T in temps]
        kdf = pd.DataFrame({"Temperature_C": temps, "B19prime_fraction_cooling": cooling, "B2_fraction_heating": heating})
        st.session_state.last_kinetics_df = kdf
        st.pyplot(make_niti_kinetics_plot(temps, cooling, heating))
        st.dataframe(kdf.head(20), hide_index=True, use_container_width=True)
        with st.expander("How to read this graph", expanded=education_mode):
            st.markdown("**Cooling curve:** B19′ martensite fraction becomes high between Ms and Mf. **Heating curve:** B2 austenite fraction becomes high between As and Af.")
            st.markdown("This graph is a placeholder calibrated by user-entered transformation temperatures. A real NiTi twin needs DSC hysteresis, stress dependence, exact composition, ageing/precipitation state, and ideally in-situ diffraction or EBSD/TKD validation.")
            st.dataframe(explain_columns(list(kdf.columns)), hide_index=True, use_container_width=True)
    st.download_button("Download kinetics CSV", st.session_state.last_kinetics_df.to_csv(index=False) if st.session_state.last_kinetics_df is not None else "", "kinetics_curve.csv", "text/csv")

with workflow_tabs[6]:
    st.header("Reliability, data gaps, and next experiments")
    st.write("A serious digital twin must say what it does not know. Check only data you truly have; do not check boxes just to increase confidence.")
    req = data_requirement_table("NiTi" if is_niti else "Steel", lpbf=lpbf)
    keys = req["key"].tolist()
    cols = st.columns(3)
    available = {}
    for i, key in enumerate(keys):
        default = bool(base_available.get(key, False))
        available[key] = cols[i % 3].checkbox(key, value=default, key=f"gap_{key}")
    st.session_state.available_data.update(available)
    gap_report = assess_data_gaps(material_key_for_gaps, st.session_state.available_data)
    st.session_state.gap_report = gap_report
    level, level_note = maturity_from_gap_score(gap_report.confidence_score, st.session_state.ebsd_df is not None, bool(available.get("DSC") or available.get("cooling_curve")), lpbf)
    g1, g2 = st.columns([0.35, 0.65])
    with g1:
        st.metric("Twin maturity", level)
        st.metric("Data confidence score", f"{gap_report.confidence_score:.2f}")
        st.write(level_note)
    with g2:
        gap_df = req.copy()
        gap_df["available"] = gap_df["key"].map(lambda k: bool(available.get(k, False)))
        st.dataframe(gap_df, hide_index=True, use_container_width=True)
    st.subheader("Missing data")
    if gap_report.missing:
        st.write(", ".join(gap_report.missing))
    else:
        st.success("No required data marked missing for the selected level. Still validate on an independent sample.")
    st.subheader("Recommended next experiments/actions")
    for item in gap_report.recommended_next_experiments:
        st.write(f"- {item}")

with workflow_tabs[7]:
    st.header("Open data and tool manifest")
    st.write("These are the public data/tool sources currently registered in the project. They are not automatically downloaded by the app yet; they guide validation and future ingestion scripts.")
    manifest_path = ROOT / "data" / "open_data_manifest" / "open_data_sources.csv"
    if manifest_path.exists():
        dfm = read_open_data_manifest(manifest_path)
        c1, c2 = st.columns([0.35, 0.65])
        with c1:
            if "priority" in dfm.columns:
                priority = st.multiselect("Filter by priority", sorted(dfm["priority"].dropna().unique()), default=list(sorted(dfm["priority"].dropna().unique())))
            else:
                priority = []
            if priority and "priority" in dfm.columns:
                dfm = dfm[dfm["priority"].isin(priority)]
        st.dataframe(dfm, use_container_width=True)
        if education_mode:
            explain_table("open_manifest", list(dfm.columns))
        st.download_button("Download manifest CSV", dfm.to_csv(index=False), "open_data_sources.csv", "text/csv")
    else:
        st.warning("Manifest not found.")

with workflow_tabs[8]:
    st.header("Export report")
    assignment = st.session_state.assignment_result
    gap_report = st.session_state.gap_report
    metrics = {
        "twin_version": "0.3",
        "twin_maturity_level": level,
        "material_key": model.material_key,
        "orientation_relationship": model.orientation_relationship.name,
        "n_variants": len(model.variants),
        "dataset_points": int(len(st.session_state.ebsd_df)) if st.session_state.ebsd_df is not None else 0,
        "sample_id": sample_id,
        "process_route": process_route,
        "lpbf": lpbf,
    }
    if assignment is not None:
        metrics.update({
            "mean_variant_error_deg": float(assignment.mean_error_deg),
            "max_variant_error_deg": float(assignment.max_error_deg),
            "assignment_confidence": float(assignment.confidence_score),
        })
    if gap_report is not None:
        metrics["data_confidence_score"] = float(gap_report.confidence_score)
        metrics["missing_data"] = gap_report.missing
    if st.session_state.last_kinetics_df is not None:
        metrics["kinetics_points"] = int(len(st.session_state.last_kinetics_df))

    md = build_markdown_report(model, assignment.summary if assignment else None, metrics, gap_report, notes=analyst_notes)
    js = build_json_report(model, metrics, gap_report)
    st.download_button("Download Markdown report", md, "martensite_twin_report.md", "text/markdown")
    st.download_button("Download JSON report", js, "martensite_twin_report.json", "application/json")
    st.markdown(md)
