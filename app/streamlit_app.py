from __future__ import annotations

import json
import inspect
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
from martwin.io.ebsd_csv import read_ebsd_csv, dataframe_to_orientation_matrices
from martwin.kinetics.km import km_curve
from martwin.kinetics.niti_transform import NiTiTransformationTemperatures, linear_cooling_fraction, linear_heating_fraction_austenite
from martwin.calibration.gap_analysis import assess_data_gaps
from martwin.reporting.report import build_markdown_report
from martwin.visualization.maps import plot_variant_map
from martwin.digital_twin.evidence import (
    PARAMETER_GUIDE,
    TWIN_LAYER_MATRIX,
    OPEN_SOURCE_DATASETS,
    DEFENSIBILITY_REQUIREMENTS,
    FIDELITY_LEVELS,
    TwinEvidence,
    dataframe,
    maturity_level,
    evidence_table,
    CHARACTERIZATION_MODULES,
    ARTICLE_EVIDENCE_MAP,
)

st.set_page_config(page_title="OpenMartensiteTwin v0.5.6 no-crash report hotfix", layout="wide", initial_sidebar_state="expanded")


def css() -> None:
    st.markdown(
        """
        <style>
        .small-muted { color: #9ca3af; font-size: 0.92rem; }
        .tiny-muted { color: #9ca3af; font-size: 0.78rem; }
        .card { border: 1px solid rgba(255,255,255,0.14); border-radius: 14px; padding: 1rem; background: rgba(255,255,255,0.035); margin-bottom: 0.75rem; }
        .claim { border-left: 4px solid #ef4444; padding: .75rem 1rem; background: rgba(239,68,68,.08); border-radius: 8px; }
        .ok { color: #22c55e; font-weight: 700; }
        .warn { color: #f59e0b; font-weight: 700; }
        .bad { color: #ef4444; font-weight: 700; }
        .pill { display: inline-block; padding: .15rem .45rem; border-radius: 999px; background: rgba(255,255,255,.08); margin-right: .35rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def card(title: str, body: str, tone: str = "") -> None:
    cls = "card" if not tone else f"card {tone}"
    st.markdown(f"<div class='{cls}'><b>{title}</b><br>{body}</div>", unsafe_allow_html=True)


def section_help(title: str, what: str, use: str, warning: str | None = None) -> None:
    with st.expander(f"What this section means — {title}", expanded=False):
        st.markdown(f"**What it is:** {what}")
        st.markdown(f"**Where it is used:** {use}")
        if warning:
            st.warning(warning)


def df_download(label: str, df: pd.DataFrame, filename: str) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv")


def json_safe(value: Any) -> Any:
    """Convert NumPy/Pandas/dataclass objects to JSON-safe objects."""
    try:
        import numpy as _np
        if isinstance(value, _np.generic):
            return value.item()
        if isinstance(value, _np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes)):
        try:
            return json_safe(vars(value))
        except Exception:
            pass
    return value


def df_markdown_safe(df: pd.DataFrame) -> str:
    """Return a Markdown table without crashing if tabulate is absent."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```text\n" + df.to_string(index=False) + "\n```"


def build_markdown_report_safe(model, assignment_summary, metrics, gap_report, notes: str) -> str:
    """Call package report builder if compatible; otherwise build a local fallback."""
    try:
        return build_markdown_report(model, assignment_summary, metrics, gap_report, notes=notes)
    except TypeError:
        try:
            return build_markdown_report(model, metrics, gap_report)
        except Exception:
            pass
    except Exception:
        pass
    lines = [
        "# OpenMartensiteTwin Report",
        "",
        "## Configuration",
        f"- Material system: {getattr(getattr(model, 'config', None), 'material_system', 'unknown')}",
        f"- Orientation relationship: {getattr(getattr(model, 'orientation_relationship', None), 'name', 'unknown')}",
        f"- Number of variants: {len(getattr(model, 'variants', []))}",
        "",
        "## Metrics",
    ]
    for k, v in (metrics or {}).items():
        lines.append(f"- `{k}`: {v}")
    if assignment_summary is not None and hasattr(assignment_summary, "empty") and not assignment_summary.empty:
        lines += ["", "## Variant population summary", "", df_markdown_safe(assignment_summary)]
    if gap_report is not None:
        lines += ["", "## Gap assessment", f"- Confidence score: {getattr(gap_report, 'confidence_score', 'unknown')}"]
        missing = getattr(gap_report, "missing", [])
        lines.append("- Missing: " + (", ".join(map(str, missing)) if missing else "none"))
    if notes:
        lines += ["", "## Notes", notes]
    return "\n".join(lines)


def build_local_json_report(model, assignment_summary, metrics, gap_report, notes: str) -> dict[str, Any]:
    """Build the JSON report locally so Streamlit cannot crash from report.py API drift."""
    orx = getattr(model, "orientation_relationship", None)
    cfg = getattr(model, "config", None)
    gap_payload = None
    if gap_report is not None:
        gap_payload = {
            "confidence_score": getattr(gap_report, "confidence_score", None),
            "missing": list(getattr(gap_report, "missing", []) or []),
            "recommended_next_experiments": list(getattr(gap_report, "recommended_next_experiments", []) or []),
        }
    payload = {
        "app_version": "v0.5.6-no-crash-report-hotfix",
        "configuration": json_safe(cfg),
        "orientation_relationship": {
            "name": getattr(orx, "name", "unknown"),
            "parent_phase": getattr(orx, "parent_phase", "unknown"),
            "child_phase": getattr(orx, "child_phase", "unknown"),
            "source_note": getattr(orx, "source_note", ""),
            "description": getattr(orx, "description", getattr(orx, "source_note", getattr(orx, "name", ""))),
            "matrix_child_to_parent": json_safe(getattr(orx, "matrix_child_to_parent", None)),
        },
        "n_variants": len(getattr(model, "variants", [])),
        "variant_population_summary": json_safe(assignment_summary) if assignment_summary is not None else None,
        "metrics": json_safe(metrics or {}),
        "gap_report": json_safe(gap_payload),
        "notes": notes,
    }
    return json_safe(payload)


def make_niti_temperatures(Ms: float, Mf: float, As: float, Af: float) -> NiTiTransformationTemperatures:
    """Create a NiTiTransformationTemperatures object across v0.5 API variants.

    Some package versions use Ms_C/Mf_C/As_C/Af_C while earlier app code used
    Ms/Mf/As_/Af. Streamlit Cloud can keep stale package files after uploads, so
    this function tries the supported constructor names instead of crashing.
    """
    candidates = [
        {"Ms_C": Ms, "Mf_C": Mf, "As_C": As, "Af_C": Af},
        {"Ms": Ms, "Mf": Mf, "As_": As, "Af": Af},
        {"Ms": Ms, "Mf": Mf, "As": As, "Af": Af},
    ]
    try:
        params = set(inspect.signature(NiTiTransformationTemperatures).parameters)
        for kwargs in candidates:
            if set(kwargs).issubset(params):
                return NiTiTransformationTemperatures(**kwargs)
    except Exception:
        pass
    for kwargs in candidates:
        try:
            return NiTiTransformationTemperatures(**kwargs)
        except TypeError:
            continue
    return NiTiTransformationTemperatures(Ms, Mf, As, Af)


def make_niti_kinetics_plot(temps: np.ndarray, cooling: list[float], heating: list[float], Ms: float, Mf: float, As: float, Af: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(temps, cooling, label="B19′ martensite fraction during cooling")
    ax.plot(temps, heating, label="B2 austenite fraction during heating")
    for val, name in [(Ms, "Ms"), (Mf, "Mf"), (As, "As"), (Af, "Af")]:
        ax.axvline(val, linestyle="--", alpha=0.5)
        ax.text(val, 1.02, name, rotation=90, va="bottom", ha="center", fontsize=8)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Phase fraction (0 = none, 1 = all)")
    ax.set_title("NiTi simplified hysteresis: DSC-style placeholder")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    return fig


def make_steel_kinetics_plot(temps: np.ndarray, frac: list[float], Ms: float, alpha: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(temps, frac, label=f"Martensite fraction, KM alpha={alpha:.4f}")
    ax.axvline(Ms, linestyle="--", alpha=0.5)
    ax.text(Ms, 1.02, "Ms", rotation=90, va="bottom", ha="center", fontsize=8)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Martensite fraction (0 = none, 1 = all)")
    ax.set_title("Steel Koistinen–Marburger cooling model")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    return fig


def plot_xrd_pattern(xrd_df: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    """Simple XRD preview. This is not a Rietveld refinement."""
    cols = {c.lower().strip(): c for c in xrd_df.columns}
    x_col = cols.get("2theta") or cols.get("two_theta") or cols.get("theta") or cols.get("q") or xrd_df.columns[0]
    y_col = cols.get("intensity") or cols.get("counts") or xrd_df.columns[1]
    x = pd.to_numeric(xrd_df[x_col], errors="coerce")
    y = pd.to_numeric(xrd_df[y_col], errors="coerce")
    clean = pd.DataFrame({"x": x, "intensity": y}).dropna()
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(clean["x"], clean["intensity"], lw=1.2)
    ax.set_xlabel(str(x_col))
    ax.set_ylabel(str(y_col))
    ax.set_title("XRD / diffraction preview: raw pattern, not refinement")
    ax.grid(True, alpha=0.25)
    peaks = pd.DataFrame()
    try:
        from scipy.signal import find_peaks
        yy = clean["intensity"].to_numpy(dtype=float)
        prominence = max(float(np.nanstd(yy)) * 1.5, 1e-9)
        idx, props = find_peaks(yy, prominence=prominence)
        if len(idx):
            peaks = clean.iloc[idx].copy().rename(columns={"x": "peak_position", "intensity": "peak_intensity"})
            peaks["relative_intensity"] = peaks["peak_intensity"] / max(float(clean["intensity"].max()), 1e-9)
            ax.scatter(peaks["peak_position"], peaks["peak_intensity"], s=18)
    except Exception:
        pass
    return fig, peaks


def normalize_eds_table(eds_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a simple EDS table with element and wt/at columns."""
    df = eds_df.copy()
    lower = {c.lower().strip(): c for c in df.columns}
    elem_col = lower.get("element") or lower.get("el") or lower.get("symbol") or df.columns[0]
    at_col = lower.get("at%") or lower.get("at.%") or lower.get("atomic%") or lower.get("atomic_percent")
    wt_col = lower.get("wt%") or lower.get("wt.%") or lower.get("weight%") or lower.get("weight_percent")
    out = pd.DataFrame({"element": df[elem_col].astype(str)})
    if at_col:
        out["at_percent"] = pd.to_numeric(df[at_col], errors="coerce")
    if wt_col:
        out["wt_percent"] = pd.to_numeric(df[wt_col], errors="coerce")
    return out


def init_state() -> None:
    defaults: dict[str, Any] = {
        "ebsd_df": None,
        "child_oris": None,
        "synthetic_parents": None,
        "assignment_result": None,
        "recon_result": None,
        "last_kinetics_df": None,
        "last_report": None,
        "dataset_origin": "none",
        "xrd_df": None,
        "xrd_peaks": None,
        "eds_df": None,
        "sem_images": [],
        "tem_images": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


css()
init_state()

# Sidebar controls with explanations
with st.sidebar:
    st.title("Twin controls")
    st.caption("These controls do not magically make the twin real. They define the model assumptions and evidence state for this run.")

    with st.expander("1. Material model", expanded=True):
        material_system = st.selectbox(
            "Material system",
            ["NiTi B2→B19′", "Steel fcc→bcc/bct"],
            help="Chooses the transformation family, symmetry, orientation relationships, variant library and missing-data checklist.",
        )
        is_niti = material_system.startswith("NiTi")
        if is_niti:
            beta = st.number_input(
                "B19′ beta angle (°)", value=96.8, min_value=90.0, max_value=110.0, step=0.1,
                help="Monoclinic angle of B19′ martensite. This affects the prototype Cayron-style NiTi orientation matrix and variant library. Replace with XRD/refined sample data for real work.",
            )
            steel_or = "KS"
        else:
            beta = 96.8
            steel_or = st.selectbox(
                "Steel OR", ["KS", "NW", "Pitsch"],
                help="The parent austenite→martensite orientation relationship used to create theoretical variants. Real EBSD work should refine this OR.",
            )

    with st.expander("2. Fitting / reconstruction settings", expanded=True):
        tol = st.slider(
            "Variant fit tolerance (°)", 1.0, 15.0, 5.0, 0.5,
            help="Measured point is accepted as a variant if its angular error is below this. Higher = more assignments but more false positives; lower = stricter but may reject noisy EBSD.",
        )
        recon_thr = st.slider(
            "Parent reconstruction threshold (°)", 1.0, 15.0, 5.0, 0.5,
            help="Prototype threshold for grouping candidate parent orientations. Higher merges parent clusters; lower splits them. v0.5 is not yet full MTEX/ARPGE graph reconstruction.",
        )

    with st.expander("3. Synthetic data controls", expanded=False):
        st.caption("Synthetic data tests the workflow. It is not evidence for any real alloy.")
        grid_n = st.slider("Synthetic map size", 20, 100, 50, 10, help="Creates grid_n × grid_n synthetic EBSD-like orientation points.")
        n_parents = st.slider("Synthetic parent grains", 1, 8, 4, 1, help="Number of rectangular parent regions used in the synthetic test map.")
        active_fraction = st.slider("Active variant fraction", 0.1, 1.0, 0.55, 0.05, help="Fraction of theoretical variants allowed inside each synthetic parent grain.")
        noise_deg = st.slider("Orientation noise (°)", 0.0, 5.0, 0.8, 0.1, help="Random angular perturbation added to synthetic orientations. Higher noise increases angular fit error.")

    with st.expander("4. Evidence available for this sample", expanded=True):
        st.caption("Check only what you really have. These boxes change confidence/maturity, not the raw orientation math.")
        composition_known = st.checkbox("composition known", value=False)
        heat_known = st.checkbox("heat treatment / thermal cycle known", value=False)
        dsc_known = st.checkbox("DSC / transformation temperatures available", value=False)
        xrd_known = st.checkbox("XRD refined lattice/phase fractions available", value=False, help="Means you have refined lattice parameters or phase fractions, not only a picture of a pattern.")
        xrd_pattern_known = st.checkbox("raw XRD/synchrotron pattern uploaded/available", value=False, help="Raw 2θ/q-intensity pattern. Used for phase/lattice validation after peak fitting/refinement.")
        sem_known = st.checkbox("SEM/optical micrographs available", value=False, help="Morphology, porosity, cracks, melt-pool tracks, etched grain structure, or correlative microstructure context.")
        eds_known = st.checkbox("EDS/WDS chemistry data available", value=False, help="Element composition, Ni/Ti ratio, impurities, alloying additions, local segregation.")
        tem_known = st.checkbox("TEM/STEM images available", value=False, help="Nanoscale twins, precipitates, dislocations, habit interfaces, local defects.")
        tem_diff_known = st.checkbox("TEM/SAED/4D-STEM diffraction available", value=False, help="Local diffraction/orientation/strain evidence at nanoscale.")
        mech_known = st.checkbox("stress-strain / mechanical data available", value=False)
        oxy_known = st.checkbox("oxygen/carbon/impurities known", value=False)
        cooling_known = st.checkbox("cooling curve / dilatometry available", value=False)
        parent_ref_known = st.checkbox("known parent reference map/grain size available", value=False)
        hardness_known = st.checkbox("hardness data available", value=False)
        retained_known = st.checkbox("retained austenite XRD available", value=False)
        lpbf = st.checkbox("LPBF/additive manufacturing route", value=False)
        laser_known = st.checkbox("LPBF laser parameters known", value=False, disabled=not lpbf)
        scan_known = st.checkbox("LPBF scan strategy known", value=False, disabled=not lpbf)
        powder_known = st.checkbox("LPBF powder chemistry known", value=False, disabled=not lpbf)
        thermal_known = st.checkbox("thermal history / melt-pool model available", value=False)
        porosity_known = st.checkbox("porosity data available", value=False, disabled=not lpbf)
        residual_known = st.checkbox("residual stress data available", value=False)

    with st.expander("5. Sample notes", expanded=False):
        sample_id = st.text_input("Sample ID", value="sample_001")
        process_route = st.selectbox("Process route", ["unknown", "literature dataset", "cast/wrought", "solution treated", "aged", "quenched", "LPBF", "cold worked + annealed"])
        composition_note = st.text_area("Composition note", placeholder="Example: Ni 50.8 at.% Ti bal.; Fe-0.2C-1.5Mn...", height=80)
        heat_note = st.text_area("Heat treatment / process note", placeholder="Example: 500°C 30 min ageing; quench rate; LPBF P-v-h-t...", height=80)
        analyst_notes = st.text_area("Analyst notes", placeholder="Data source, DOI, EBSD settings, caveats...", height=80)

config = TwinConfiguration(
    material_system=material_system,
    beta_deg=beta,
    steel_or=steel_or,
    angular_tolerance_deg=tol,
    reconstruction_threshold_deg=recon_thr,
    notes=analyst_notes,
)
model = build_twin_model(config)

has_dataset = st.session_state.ebsd_df is not None
if has_dataset:
    dataset_available = True
else:
    dataset_available = False

evidence = TwinEvidence(
    composition=composition_known or bool(composition_note.strip()),
    heat_treatment=heat_known or bool(heat_note.strip()),
    ebsd_or_tkd=dataset_available,
    dsc=dsc_known,
    xrd_lattice=xrd_known,
    xrd_pattern=xrd_pattern_known or (st.session_state.xrd_df is not None),
    sem_images=sem_known or bool(st.session_state.sem_images),
    eds_maps=eds_known or (st.session_state.eds_df is not None),
    tem_stem=tem_known or bool(st.session_state.tem_images),
    tem_diffraction=tem_diff_known,
    stress_strain=mech_known,
    oxygen_carbon=oxy_known,
    thermal_history=thermal_known,
    cooling_curve=cooling_known,
    prior_austenite_reference=parent_ref_known,
    hardness=hardness_known,
    retained_austenite_xrd=retained_known,
    laser_parameters=laser_known,
    scan_strategy=scan_known,
    powder_chemistry=powder_known,
    melt_pool_or_thermal_model=thermal_known,
    porosity=porosity_known,
    residual_stress=residual_known,
)

material_key_for_gaps = model.material_key + (" lpbf" if lpbf else "")
gap_report = assess_data_gaps(material_key_for_gaps, evidence.as_gap_dict())
has_calibration = dsc_known or xrd_known or xrd_pattern_known or eds_known or cooling_known or mech_known or hardness_known
has_process = heat_known or bool(heat_note.strip()) or thermal_known or cooling_known or laser_known
has_validation = parent_ref_known or retained_known or mech_known or hardness_known or sem_known or tem_known
level, level_note = maturity_level(gap_report.confidence_score, has_dataset, has_calibration, has_process, has_validation)

# Header
st.title("OpenMartensiteTwin v0.5.4 hotfix")
st.caption("A guided, evidence-aware martensitic-transformation twin. v0.5 is designed to be honest: it separates calculation, evidence, assumptions and missing validation.")

cols = st.columns(7)
cols[0].metric("Twin maturity", level.split(" — ")[0])
cols[1].metric("Material", "NiTi" if is_niti else "Steel")
cols[2].metric("OR", model.orientation_relationship.name.split()[0])
cols[3].metric("Variants", len(model.variants))
cols[4].metric("Dataset points", len(st.session_state.ebsd_df) if has_dataset else 0)
cols[5].metric("Data confidence", f"{gap_report.confidence_score:.2f}")
cols[6].metric("Dataset", st.session_state.dataset_origin)

st.warning(level_note)
st.markdown(
    "<span class='pill'>calculation ≠ validation</span>"
    "<span class='pill'>synthetic data ≠ experiment</span>"
    "<span class='pill'>fit tolerance changes conclusions</span>",
    unsafe_allow_html=True,
)

TABS = st.tabs([
    "0. Twin map",
    "1. Controls explained",
    "2. Evidence/state vector",
    "3. Crystallography",
    "4. EBSD/TKD workspace",
    "5. Variant & parent analysis",
    "6. Kinetics",
    "7. XRD/EDS/SEM/TEM",
    "8. Article-derived gap map",
    "9. Open data/tools",
    "10. Defensibility gaps",
    "11. Report/export",
])

with TABS[0]:
    st.header("What this app is actually doing")
    card(
        "Digital-twin loop",
        "<b>State</b> = material + process + measured data. <b>Model</b> = crystallography + kinetics + future thermodynamics/mechanics. <b>Update</b> = compare predictions to EBSD/TKD/DSC/XRD/mechanical evidence. <b>Decision</b> = identify missing data and recommend the next experiment."
    )
    st.subheader("Current implementation vs complete target")
    layer_df = dataframe(TWIN_LAYER_MATRIX)
    st.dataframe(layer_df, hide_index=True, use_container_width=True)
    st.markdown("### Fidelity ladder")
    st.dataframe(dataframe(FIDELITY_LEVELS), hide_index=True, use_container_width=True)
    st.info("The roadmap you gave is correct: the twin must combine Cayron-style crystallography, EBSD/TKD reconstruction, XRD/EDS/SEM/TEM/DSC evidence, thermodynamics, kinetics, phase-field/mechanics, LPBF/heat-treatment data, uncertainty and reporting. v0.5 makes those layers visible instead of hiding them.")

with TABS[1]:
    st.header("Every left-panel control: what it means and what it changes")
    guide_df = dataframe(PARAMETER_GUIDE)
    st.dataframe(guide_df, hide_index=True, use_container_width=True)
    st.markdown("### Quick sensitivity explanation")
    st.write("Changing **fit tolerance** or **reconstruction threshold** can change the number of assigned points and parent clusters. Changing **beta angle** or **steel OR** changes the theoretical variant library itself. Evidence checkboxes change the maturity/confidence score and gap register, but they do not magically create measured data.")

with TABS[2]:
    st.header("Evidence and state vector")
    section_help(
        "state vector",
        "The state vector is the current description of the sample: material, OR, process, available measurements, assumptions and data source.",
        "Every downstream table and plot should be traceable to this state vector, so reports do not mix real data with assumptions.",
        "A defensible twin must store the raw data/provenance, not only checkboxes.",
    )
    state = {
        "sample_id": sample_id,
        "material_system": material_system,
        "orientation_relationship": model.orientation_relationship.name,
        "beta_deg": beta if is_niti else None,
        "steel_or": steel_or if not is_niti else None,
        "variant_fit_tolerance_deg": tol,
        "parent_reconstruction_threshold_deg": recon_thr,
        "process_route": process_route,
        "lpbf_route": lpbf,
        "dataset_origin": st.session_state.dataset_origin,
        "dataset_points": len(st.session_state.ebsd_df) if has_dataset else 0,
        "xrd_pattern_uploaded": st.session_state.xrd_df is not None,
        "eds_table_uploaded": st.session_state.eds_df is not None,
        "sem_image_count": len(st.session_state.sem_images),
        "tem_image_count": len(st.session_state.tem_images),
        "composition_note": composition_note,
        "heat_treatment_note": heat_note,
        "analyst_notes": analyst_notes,
        "maturity_level": level,
        "confidence_score": gap_report.confidence_score,
    }
    st.json(state)
    st.subheader("Evidence table")
    e_df = evidence_table(evidence)
    st.dataframe(e_df, hide_index=True, use_container_width=True)
    df_download("Download evidence table", e_df, "open_martensite_twin_evidence.csv")

with TABS[3]:
    st.header("Crystallographic model")
    section_help(
        "orientation relationship and variants",
        "The OR matrix maps a child/martensite crystal frame to the parent crystal frame. Symmetry operations generate all allowed theoretical variants.",
        "Variant assignment compares measured EBSD/TKD orientations against these theoretical variant orientations. The misorientation matrix shows expected angular separations between variants.",
        "The current NiTi model is a Cayron-inspired prototype; exact work requires sample-specific lattice parameters, convention checks and benchmark validation.",
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Orientation relationship")
        st.write(
            getattr(
                model.orientation_relationship,
                "description",
                getattr(
                    model.orientation_relationship,
                    "source_note",
                    model.orientation_relationship.name,
                ),
            )
        )
        st.caption("Matrix convention here: child crystal frame → parent crystal frame. If your EBSD software uses a different convention, convert before comparing.")
        or_df = pd.DataFrame(model.orientation_relationship.matrix_child_to_parent, columns=["parent x", "parent y", "parent z"], index=["child x", "child y", "child z"])
        st.dataframe(or_df.style.format("{:.4f}"), use_container_width=True)
    with c2:
        st.subheader("Variant library")
        tables = variant_library_tables(model)
        vdf = tables["variants"].copy()
        vdf.insert(1, "meaning", "candidate child orientation from parent symmetry + OR + child symmetry")
        st.dataframe(vdf.head(12).style.format({c: "{:.4f}" for c in vdf.columns if c.startswith("r")}), use_container_width=True)
        st.caption("r00…r22 are the 3×3 rotation-matrix entries. They are used internally for angular comparisons; most users should focus on variant_id and fit error, not memorize matrix entries.")
    st.subheader("Pairwise theoretical variant misorientation matrix")
    st.caption("Cell (i,j) = angular separation between theoretical variants i and j after child-phase symmetry is considered. Repeated angles are expected because variants form symmetry families.")
    st.dataframe(tables["misorientation_matrix_deg"].style.format("{:.1f}"), use_container_width=True)

with TABS[4]:
    st.header("Data workspace")
    section_help(
        "EBSD/TKD data",
        "The real value starts when a measured orientation map is loaded. v0.5 supports CSV with either r00…r22 rotation-matrix columns or Bunge Euler angles phi1/Phi/phi2.",
        "The orientation map feeds variant assignment, parent reconstruction, fit-error statistics and report export.",
        "Direct .ctf/.ang/.h5 import is listed as a required v0.5 gap. Use vendor/MTEX/orix export to CSV for now.",
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Generate synthetic test map")
        if st.button("Generate synthetic EBSD-like map", type="primary"):
            synthetic = generate_synthetic_child_map(
                model.variants,
                grid_shape=(grid_n, grid_n),
                n_parent_grains=n_parents,
                active_variant_fraction=active_fraction,
                orientation_noise_deg=noise_deg,
                seed=7,
            )
            st.session_state.ebsd_df = synthetic.dataframe
            st.session_state.child_oris = synthetic.child_orientations
            st.session_state.synthetic_parents = synthetic.parent_orientations
            st.session_state.dataset_origin = "synthetic"
            st.success(f"Generated {len(synthetic.dataframe)} synthetic points. This tests code only; it is not experimental evidence.")
    with col_b:
        st.subheader("Upload CSV orientation map")
        uploaded = st.file_uploader("CSV with x,y and r00…r22 OR phi1,Phi,phi2", type=["csv"])
        if uploaded is not None:
            try:
                df = read_ebsd_csv(uploaded)
                st.session_state.ebsd_df = df
                st.session_state.child_oris = dataframe_to_orientation_matrices(df)
                st.session_state.synthetic_parents = None
                st.session_state.dataset_origin = "uploaded_csv"
                st.success(f"Loaded {len(df)} orientation points.")
            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")
    if has_dataset:
        df = st.session_state.ebsd_df
        st.subheader("Dataset preview")
        st.dataframe(df.drop(columns=["orientation_matrix"], errors="ignore").head(50), use_container_width=True)
        st.caption("Important columns: x,y are map coordinates; r00…r22 or phi1/Phi/phi2 define orientation; parent_region_id and true_variant_id appear only in synthetic validation data.")
        df_download("Download current dataset CSV", df.drop(columns=["orientation_matrix"], errors="ignore"), "current_orientation_dataset.csv")
    else:
        st.info("No orientation dataset loaded yet. Generate a synthetic test map or upload a CSV.")

with TABS[5]:
    st.header("Variant assignment and parent reconstruction")
    section_help(
        "variant assignment",
        "Each measured child orientation is compared with every theoretical variant. The closest variant gets assigned and the angular error is reported.",
        "Low angular error and high in-tolerance fraction support the chosen OR/model; high error suggests wrong OR, wrong convention, noisy data or wrong phase labels.",
        "The parent reconstruction in v0.5 is exploratory. Full defensible reconstruction needs graph-based topology and OR-probability methods.",
    )
    if not has_dataset:
        st.info("Load or generate data first.")
    else:
        run = st.button("Run variant + parent analysis", type="primary")
        if run or st.session_state.assignment_result is not None:
            if run:
                if st.session_state.synthetic_parents is not None and "parent_region_id" in st.session_state.ebsd_df.columns:
                    # known synthetic parent regions create more meaningful assignments
                    from martwin.analysis.variant_analysis import assign_variants_known_parent_regions
                    assignment = assign_variants_known_parent_regions(
                        st.session_state.child_oris,
                        st.session_state.synthetic_parents,
                        st.session_state.ebsd_df["parent_region_id"],
                        model.variants,
                        model.child_sym_ops,
                        tolerance_deg=tol,
                    )
                    _, recon, _ = run_known_parent_analysis(model, st.session_state.child_oris)
                else:
                    assignment, recon, _ = run_known_parent_analysis(model, st.session_state.child_oris)
                st.session_state.assignment_result = assignment
                st.session_state.recon_result = recon
            assignment = st.session_state.assignment_result
            recon = st.session_state.recon_result
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Mean angular error", f"{assignment.mean_error_deg:.2f}°")
            m2.metric("Max angular error", f"{assignment.max_error_deg:.2f}°")
            m3.metric("In tolerance", f"{assignment.assignments['is_in_tolerance'].mean()*100:.1f}%")
            m4.metric("Parent clusters", len(set(recon.labels)) if recon else 0)
            st.subheader("Variant population summary")
            st.caption("count = number of map points assigned to a variant. fraction = count divided by all points. mean_error_deg = average angular mismatch for that variant. If one/few variants dominate, it may indicate texture, variant selection, synthetic setup, or biased sampling.")
            st.dataframe(assignment.summary, hide_index=True, use_container_width=True)
            st.subheader("Point-level assignments")
            st.caption("point_id links back to the data map. angular_error_deg is the mismatch between measured orientation and closest theoretical variant. fit_quality is 1 at zero error and decreases toward 0 at the selected tolerance.")
            st.dataframe(assignment.assignments.head(200), hide_index=True, use_container_width=True)
            plot_df = st.session_state.ebsd_df.copy()
            plot_df["variant_id"] = assignment.assignments["variant_id"].values
            plot_df["angular_error_deg"] = assignment.assignments["angular_error_deg"].values
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Variant map")
                st.caption("Each pixel/color is the closest theoretical martensite variant. This is where crystallography becomes spatial microstructure interpretation.")
                st.pyplot(plot_variant_map(plot_df, value_col="variant_id", title="Assigned variant ID map"))
            with c2:
                st.subheader("Angular-error map")
                st.caption("Bright/high-error areas are where the selected OR/model does not fit well. Use this to find bad data, wrong OR, convention errors, or real local distortions.")
                st.pyplot(plot_variant_map(plot_df, value_col="angular_error_deg", title="Angular fit error (deg)"))
            if recon:
                plot_df["parent_cluster"] = recon.labels
                st.subheader("Prototype parent-cluster map")
                st.caption("This is not yet a publication-grade PAG/B2 reconstruction. It is a provisional clustering of candidate parent orientations used to show the intended workflow.")
                st.pyplot(plot_variant_map(plot_df, value_col="parent_cluster", title="Prototype reconstructed parent clusters"))

with TABS[6]:
    st.header("Transformation kinetics")
    section_help(
        "kinetics curve",
        "Kinetics connects temperature/cooling/heating history to phase fraction. This is necessary for a true process-aware twin.",
        "The graph is used to compare DSC/dilatometry/XRD phase-fraction data and to estimate phase fraction at a temperature.",
        "v0.5 kinetics is educational unless you supply measured DSC/dilatometry data and fit parameters.",
    )
    if is_niti:
        Ms = st.number_input("Ms: martensite start during cooling (°C)", value=30.0)
        Mf = st.number_input("Mf: martensite finish during cooling (°C)", value=-10.0)
        As = st.number_input("As: austenite start during heating (°C)", value=15.0)
        Af = st.number_input("Af: austenite finish during heating (°C)", value=55.0)
        tmin = min(Mf, As, Ms, Af) - 30
        tmax = max(Mf, As, Ms, Af) + 30
        temps = np.linspace(tmin, tmax, 220)
        trans = make_niti_temperatures(Ms=Ms, Mf=Mf, As=As, Af=Af)
        cooling = [linear_cooling_fraction(float(T), trans) for T in temps]
        heating = [linear_heating_fraction_austenite(float(T), trans) for T in temps]
        kinetics_df = pd.DataFrame({"temperature_C": temps, "B19prime_fraction_cooling": cooling, "B2_fraction_heating": heating})
        st.pyplot(make_niti_kinetics_plot(temps, cooling, heating, Ms, Mf, As, Af))
        st.caption("Legend: B19′ fraction during cooling rises between Ms and Mf. B2 fraction during heating rises between As and Af. Real NiTi needs DSC-measured hysteresis, composition and precipitation state.")
    else:
        Ms = st.number_input("Ms: martensite start temperature (°C)", value=420.0)
        alpha = st.number_input("KM alpha", value=0.011, min_value=0.0001, max_value=0.1, step=0.001, format="%.4f")
        temps = np.linspace(Ms + 80, Ms - 300, 220)
        frac = [km_curve(float(T), Ms=Ms, alpha=alpha) for T in temps]
        kinetics_df = pd.DataFrame({"temperature_C": temps, "martensite_fraction": frac})
        st.pyplot(make_steel_kinetics_plot(temps, frac, Ms, alpha))
        st.caption("Legend: martensite fraction stays near zero above Ms and increases below Ms. Real steel work needs composition-dependent Ms, dilatometry/cooling curve and retained-austenite validation.")
    st.session_state.last_kinetics_df = kinetics_df
    st.dataframe(kinetics_df.head(40), use_container_width=True)
    df_download("Download kinetics curve", kinetics_df, "kinetics_curve.csv")

with TABS[7]:
    st.header("XRD, EDS, SEM and TEM characterization evidence")
    section_help(
        "characterization evidence",
        "A defensible martensitic-transformation twin cannot rely only on orientation matrices. XRD validates phases/lattice parameters, EDS validates chemistry, SEM validates morphology/defects, and TEM/STEM validates nanoscale twins, precipitates and interfaces.",
        "These measurements update the sample state vector and reduce uncertainty. In v0.5, XRD/EDS have simple upload previews and SEM/TEM are evidence/metadata workspaces; full scientific refinement is planned through open-source connectors.",
        "Do not call a plotted XRD curve a phase refinement. Use GSAS-II/MAUD-style Rietveld refinement or a validated connector for publication-grade phase fractions.",
    )
    st.subheader("What each characterization module contributes")
    char_df = dataframe(CHARACTERIZATION_MODULES)
    st.dataframe(char_df, hide_index=True, use_container_width=True)
    st.divider()

    xcol, ecol = st.columns(2)
    with xcol:
        st.subheader("XRD / diffraction CSV preview")
        st.caption("Upload columns like `2theta,intensity` or `q,intensity`. This finds visible peaks only; it is not Rietveld refinement.")
        xrd_upload = st.file_uploader("Upload XRD CSV", type=["csv"], key="xrd_upload")
        if xrd_upload is not None:
            try:
                xdf = pd.read_csv(xrd_upload)
                fig, peaks = plot_xrd_pattern(xdf)
                st.session_state.xrd_df = xdf
                st.session_state.xrd_peaks = peaks
                st.pyplot(fig)
                st.write("Detected peak candidates")
                st.dataframe(peaks.head(30), hide_index=True, use_container_width=True)
            except Exception as exc:
                st.error(f"Could not read/plot XRD CSV: {exc}")
        elif st.session_state.xrd_df is not None:
            fig, peaks = plot_xrd_pattern(st.session_state.xrd_df)
            st.pyplot(fig)
            st.dataframe(peaks.head(30), hide_index=True, use_container_width=True)
        st.info("How XRD affects the twin: it supplies B2/B19′/R or retained-austenite phase evidence, refined lattice parameters, peak shifts and phase fractions. These values should replace defaults such as the NiTi B19′ beta angle.")

    with ecol:
        st.subheader("EDS/WDS chemistry table")
        st.caption("Upload columns like `element,at%` or `element,wt%`. For NiTi, the Ni/Ti atomic ratio is critical.")
        eds_upload = st.file_uploader("Upload EDS/WDS composition CSV", type=["csv"], key="eds_upload")
        if eds_upload is not None:
            try:
                edf = normalize_eds_table(pd.read_csv(eds_upload))
                st.session_state.eds_df = edf
            except Exception as exc:
                st.error(f"Could not read EDS/WDS CSV: {exc}")
        if st.session_state.eds_df is not None:
            edf = st.session_state.eds_df
            st.dataframe(edf, hide_index=True, use_container_width=True)
            if "at_percent" in edf.columns:
                vals = {r.element.strip().lower(): r.at_percent for r in edf.itertuples() if pd.notna(r.at_percent)}
                ni = vals.get("ni")
                ti = vals.get("ti")
                if ni is not None and ti is not None and ti != 0:
                    st.metric("Ni/Ti atomic ratio", f"{ni/ti:.4f}")
                    st.caption("In NiTi, small composition differences can shift transformation temperatures. Treat this as a quality-control signal, not a full thermodynamic calculation.")
        st.info("How EDS affects the twin: chemistry determines CALPHAD inputs, NiTi transformation-temperature risk, powder/as-built differences, impurity risk and local segregation.")

    scol, tcol = st.columns(2)
    with scol:
        st.subheader("SEM / optical morphology evidence")
        sem_files = st.file_uploader("Upload SEM/optical images", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True, key="sem_upload")
        if sem_files:
            st.session_state.sem_images = [f.name for f in sem_files]
            for f in sem_files[:3]:
                st.image(f, caption=f"SEM/optical: {f.name}", use_container_width=True)
        st.caption("SEM/optical images are used for morphology, porosity/cracks, melt-pool tracks, etched PAG comparison and sanity-checking EBSD-derived maps. Automated segmentation is planned.")
    with tcol:
        st.subheader("TEM / STEM / SAED / 4D-STEM evidence")
        tem_files = st.file_uploader("Upload TEM/STEM images or diffraction previews", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True, key="tem_upload")
        if tem_files:
            st.session_state.tem_images = [f.name for f in tem_files]
            for f in tem_files[:3]:
                st.image(f, caption=f"TEM/STEM: {f.name}", use_container_width=True)
        st.caption("TEM/STEM validates nanoscale twins, precipitates, habit/interface planes, diffuse scattering and local strain that EBSD may miss. Raw 4D-STEM/SAED connectors are planned.")

with TABS[8]:
    st.header("Article-derived missing-data map")
    section_help(
        "article-derived gap map",
        "This table translates the literature/open-source landscape into concrete missing pieces for the twin.",
        "Use it as a development roadmap and as the app's honesty layer: what can be filled from articles/open data now, and what must be generated by our own experiments later.",
        "Article values can set defaults or validate algorithms, but they cannot replace same-sample evidence for a calibrated twin.",
    )
    article_df = dataframe(ARTICLE_EVIDENCE_MAP)
    st.dataframe(article_df, hide_index=True, use_container_width=True)
    df_download("Download article-derived gap map", article_df, "article_derived_gap_map_v05.csv")
    st.markdown(
        "**Realistic rule:** literature can fill *model form, expected ranges, ORs, benchmark algorithms,* and sometimes open benchmark data. "
        "It cannot defensibly fill *your sample's exact composition, heat treatment, EBSD/TKD map, DSC curve, XRD lattice parameters, SEM/EDS/TEM evidence,* or mechanical response. Those must be uploaded later by us."
    )

with TABS[9]:
    st.header("Open data and open tools to fill the gaps")
    section_help(
        "open data/tools",
        "This table is the ingestion roadmap. It identifies public datasets/tools that can validate individual layers.",
        "Use these datasets to turn the app from a demo into a benchmarked research tool.",
        "Large datasets are not bundled in the repo; use the URLs/scripts and respect each dataset license.",
    )
    sources_df = dataframe(OPEN_SOURCE_DATASETS)
    st.dataframe(sources_df, hide_index=True, use_container_width=True)
    df_download("Download open-data manifest", sources_df, "open_data_manifest_v05.csv")
    st.subheader("How these sources are used")
    st.markdown(
        "- **Steel first validation:** use Zenodo high-temperature EBSD/CTF + dilatometry to benchmark parent reconstruction.\n"
        "- **NiTi first originality:** use Cayron's open article as crystallographic reference, but collect/upload raw NiTi EBSD/TKD/DSC/XRD/SEM/EDS/TEM to make it calibrated.\n"
        "- **Python ecosystem:** use orix/kikuchipy for robust EBSD/orientation handling; pycalphad for thermodynamics; DAMASK/OpenPhase for future mechanics/phase-field coupling."
    )

with TABS[10]:
    st.header("What is missing before we can defensibly call it the most comprehensive twin?")
    st.markdown("This is the gap register. It prevents us from pretending the app is more mature than it is.")
    gap_df = dataframe(DEFENSIBILITY_REQUIREMENTS)
    st.dataframe(gap_df, hide_index=True, use_container_width=True)
    st.subheader("Current missing data for this run")
    missing_df = pd.DataFrame({"missing_item": gap_report.missing}) if gap_report.missing else pd.DataFrame({"missing_item": ["none from current checklist"]})
    st.dataframe(missing_df, hide_index=True, use_container_width=True)
    st.subheader("Recommended next experiments / data actions")
    for item in gap_report.recommended_next_experiments:
        st.write(f"- {item}")
    st.error("For a defensible 'most comprehensive' claim, the next code milestone is graph-based parent reconstruction + real open steel dataset ingestion + raw NiTi dataset collection. The next data milestone is measured same-sample EBSD/TKD + DSC + XRD + SEM/EDS/TEM + mechanical data.")

with TABS[11]:
    st.header("Report/export")
    assignment = st.session_state.assignment_result
    metrics = {
        "maturity_level": level,
        "confidence_score": gap_report.confidence_score,
        "material_system": material_system,
        "orientation_relationship": model.orientation_relationship.name,
        "n_variants": len(model.variants),
        "dataset_origin": st.session_state.dataset_origin,
        "dataset_points": len(st.session_state.ebsd_df) if has_dataset else 0,
        "xrd_pattern_uploaded": st.session_state.xrd_df is not None,
        "eds_table_uploaded": st.session_state.eds_df is not None,
        "sem_image_count": len(st.session_state.sem_images),
        "tem_image_count": len(st.session_state.tem_images),
        "variant_fit_tolerance_deg": tol,
        "parent_reconstruction_threshold_deg": recon_thr,
        "lpbf": lpbf,
    }
    if assignment is not None:
        metrics.update({
            "mean_variant_error_deg": float(assignment.mean_error_deg),
            "max_variant_error_deg": float(assignment.max_error_deg),
            "in_tolerance_fraction": float(assignment.assignments["is_in_tolerance"].mean()),
        })
    notes = "\n".join([f"Sample ID: {sample_id}", f"Process route: {process_route}", composition_note, heat_note, analyst_notes])
    assignment_summary = assignment.summary if assignment else None
    md = build_markdown_report_safe(model, assignment_summary, metrics, gap_report, notes=notes)
    md += "\n\n## Evidence table\n\n" + df_markdown_safe(evidence_table(evidence))
    md += "\n\n## Defensibility gap register\n\n" + df_markdown_safe(gap_df)
    st.text_area("Markdown report preview", value=md, height=500)
    st.download_button("Download Markdown report", md.encode("utf-8"), "open_martensite_twin_v056_report.md", "text/markdown")

    json_report = build_local_json_report(model, assignment_summary, metrics, gap_report, notes=notes)
    json_report["state_vector"] = json_safe(state)
    json_report["evidence"] = json_safe(evidence_table(evidence))
    json_report["defensibility_gaps"] = json_safe(gap_df)
    st.download_button("Download JSON report", json.dumps(json_report, indent=2).encode("utf-8"), "open_martensite_twin_v056_report.json", "application/json")
