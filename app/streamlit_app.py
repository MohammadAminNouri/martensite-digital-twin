# Optional advanced modules
EBSD_IO_AVAILABLE = False
RECONSTRUCTION_AVAILABLE = False

try:
    from martwin.io import load as load_ebsd
    from martwin.io import oxford_to_edax, edax_to_oxford
    EBSD_IO_AVAILABLE = True
except Exception as e:
    EBSD_IO_ERROR = str(e)

try:
    from martwin.reconstruction.parent_reconstructor import ParentReconstructor
    RECONSTRUCTION_AVAILABLE = True
except Exception as e:
    RECONSTRUCTION_ERROR = str(e)
"""
OpenMartensiteTwin  v0.6.0
==========================
Streamlit app — full rewrite integrating:
  • martwin.io        : native .ctf / .ang EBSD readers  (EBSDData)
  • martwin.reconstruction : graph-based parent reconstruction
    (ParentReconstructor, OR_REGISTRY, GrainData, detect_OR, ORRefiner)

All original twin-model, kinetics, evidence, reporting and XRD/EDS/SEM/TEM
functionality is preserved and extended to consume EBSDData directly.
"""

from __future__ import annotations

import json
import inspect
import sys
import tempfile
import pathlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ── repo root on sys.path ──────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── original twin-model imports (unchanged) ────────────────────────────────
from martwin.workflows.digital_twin import (
    TwinConfiguration,
    build_twin_model,
    run_known_parent_analysis,
    variant_library_tables,
)
from martwin.simulation.synthetic import generate_synthetic_child_map
from martwin.io.ebsd_csv import read_ebsd_csv, dataframe_to_orientation_matrices
from martwin.kinetics.km import km_curve
from martwin.kinetics.niti_transform import (
    NiTiTransformationTemperatures,
    linear_cooling_fraction,
    linear_heating_fraction_austenite,
)
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

# ── NEW: native EBSD file readers ─────────────────────────────────────────
from martwin.io import load_ctf, load_ang, load, oxford_to_edax, merge_ctf_ang
from martwin.io.ebsd_data import EBSDData, Phase

# ── NEW: graph-based parent reconstruction ─────────────────────────────────
from martwin.reconstruction import (
    OR_REGISTRY,
    get_OR,
    GrainData,
    ParentReconstructor,
    ParentReconstructionResult,
    ORRefiner,
    detect_OR,
)

# ===========================================================================
# Page config
# ===========================================================================

st.set_page_config(
    page_title="OpenMartensiteTwin v0.6.0",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===========================================================================
# CSS / card helpers  (unchanged from v0.5)
# ===========================================================================

def css() -> None:
    st.markdown(
        """
        <style>
        .small-muted { color: #9ca3af; font-size: 0.92rem; }
        .tiny-muted  { color: #9ca3af; font-size: 0.78rem; }
        .card { border: 1px solid rgba(255,255,255,0.14); border-radius: 14px;
                padding: 1rem; background: rgba(255,255,255,0.035); margin-bottom: 0.75rem; }
        .claim { border-left: 4px solid #ef4444; padding: .75rem 1rem;
                 background: rgba(239,68,68,.08); border-radius: 8px; }
        .ok   { color: #22c55e; font-weight: 700; }
        .warn { color: #f59e0b; font-weight: 700; }
        .bad  { color: #ef4444; font-weight: 700; }
        .pill { display: inline-block; padding: .15rem .45rem; border-radius: 999px;
                background: rgba(255,255,255,.08); margin-right: .35rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def card(title: str, body: str, tone: str = "") -> None:
    cls = "card" if not tone else f"card {tone}"
    st.markdown(
        f"<div class='{cls}'><b>{title}</b><br>{body}</div>",
        unsafe_allow_html=True,
    )


def section_help(title: str, what: str, use: str, warning: str | None = None) -> None:
    with st.expander(f"What this section means — {title}", expanded=False):
        st.markdown(f"**What it is:** {what}")
        st.markdown(f"**Where it is used:** {use}")
        if warning:
            st.warning(warning)


def df_download(label: str, df: pd.DataFrame, filename: str) -> None:
    st.download_button(
        label,
        df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


# ===========================================================================
# JSON / report helpers  (unchanged from v0.5)
# ===========================================================================

def json_safe(value: Any) -> Any:
    try:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
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
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```text\n" + df.to_string(index=False) + "\n```"


def build_markdown_report_safe(model, assignment_summary, metrics, gap_report, notes: str) -> str:
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
        lines += [
            "",
            "## Gap assessment",
            f"- Confidence score: {getattr(gap_report, 'confidence_score', 'unknown')}",
        ]
        missing = getattr(gap_report, "missing", [])
        lines.append("- Missing: " + (", ".join(map(str, missing)) if missing else "none"))
    if notes:
        lines += ["", "## Notes", notes]
    return "\n".join(lines)


def build_local_json_report(model, assignment_summary, metrics, gap_report, notes: str) -> dict[str, Any]:
    orx = getattr(model, "orientation_relationship", None)
    cfg = getattr(model, "config", None)
    gap_payload = None
    if gap_report is not None:
        gap_payload = {
            "confidence_score": getattr(gap_report, "confidence_score", None),
            "missing": list(getattr(gap_report, "missing", []) or []),
            "recommended_next_experiments": list(
                getattr(gap_report, "recommended_next_experiments", []) or []
            ),
        }
    payload = {
        "app_version": "v0.6.0",
        "configuration": json_safe(cfg),
        "orientation_relationship": {
            "name": getattr(orx, "name", "unknown"),
            "parent_phase": getattr(orx, "parent_phase", "unknown"),
            "child_phase": getattr(orx, "child_phase", "unknown"),
            "source_note": getattr(orx, "source_note", ""),
            "description": getattr(
                orx,
                "description",
                getattr(orx, "source_note", getattr(orx, "name", "")),
            ),
            "matrix_child_to_parent": json_safe(getattr(orx, "matrix_child_to_parent", None)),
        },
        "n_variants": len(getattr(model, "variants", [])),
        "variant_population_summary": json_safe(assignment_summary) if assignment_summary is not None else None,
        "metrics": json_safe(metrics or {}),
        "gap_report": json_safe(gap_payload),
        "notes": notes,
    }
    return json_safe(payload)


# ===========================================================================
# NiTi kinetics helpers  (unchanged from v0.5)
# ===========================================================================

def make_niti_temperatures(Ms, Mf, As, Af) -> NiTiTransformationTemperatures:
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


def make_niti_kinetics_plot(temps, cooling, heating, Ms, Mf, As, Af) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(temps, cooling, label="B19′ martensite fraction during cooling")
    ax.plot(temps, heating, label="B2 austenite fraction during heating")
    for val, name in [(Ms, "Ms"), (Mf, "Mf"), (As, "As"), (Af, "Af")]:
        ax.axvline(val, linestyle="--", alpha=0.5)
        ax.text(val, 1.02, name, rotation=90, va="bottom", ha="center", fontsize=8)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Phase fraction")
    ax.set_title("NiTi simplified hysteresis: DSC-style placeholder")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    return fig


def make_steel_kinetics_plot(temps, frac, Ms, alpha) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(temps, frac, label=f"Martensite fraction, KM alpha={alpha:.4f}")
    ax.axvline(Ms, linestyle="--", alpha=0.5)
    ax.text(Ms, 1.02, "Ms", rotation=90, va="bottom", ha="center", fontsize=8)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Martensite fraction")
    ax.set_title("Steel Koistinen–Marburger cooling model")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    return fig


# ===========================================================================
# XRD / EDS helpers  (unchanged from v0.5)
# ===========================================================================

def plot_xrd_pattern(xrd_df: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    cols = {c.lower().strip(): c for c in xrd_df.columns}
    x_col = (
        cols.get("2theta") or cols.get("two_theta") or
        cols.get("theta") or cols.get("q") or xrd_df.columns[0]
    )
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
        idx, _ = find_peaks(yy, prominence=prominence)
        if len(idx):
            peaks = clean.iloc[idx].copy().rename(
                columns={"x": "peak_position", "intensity": "peak_intensity"}
            )
            peaks["relative_intensity"] = peaks["peak_intensity"] / max(
                float(clean["intensity"].max()), 1e-9
            )
            ax.scatter(peaks["peak_position"], peaks["peak_intensity"], s=18)
    except Exception:
        pass
    return fig, peaks


def normalize_eds_table(eds_df: pd.DataFrame) -> pd.DataFrame:
    df = eds_df.copy()
    lower = {c.lower().strip(): c for c in df.columns}
    elem_col = (
        lower.get("element") or lower.get("el") or
        lower.get("symbol") or df.columns[0]
    )
    at_col = (
        lower.get("at%") or lower.get("at.%") or
        lower.get("atomic%") or lower.get("atomic_percent")
    )
    wt_col = (
        lower.get("wt%") or lower.get("wt.%") or
        lower.get("weight%") or lower.get("weight_percent")
    )
    out = pd.DataFrame({"element": df[elem_col].astype(str)})
    if at_col:
        out["at_percent"] = pd.to_numeric(df[at_col], errors="coerce")
    if wt_col:
        out["wt_percent"] = pd.to_numeric(df[wt_col], errors="coerce")
    return out


# ===========================================================================
# NEW: EBSDData → GrainData bridge
# ===========================================================================

def ebsd_data_to_grain_data(ebsd: EBSDData) -> GrainData | None:
    """
    Convert an EBSDData (from load_ctf / load_ang) into a GrainData object
    suitable for graph-based parent reconstruction.

    Strategy
    --------
    1. Filter to indexed pixels only.
    2. Use pixel-level orientation matrices as grain orientations.
       (For a real use-case the caller would first segment grains; here we
       treat each indexed pixel as a grain for demo / small-map use.)
    3. Build a square-grid pixel adjacency (4-connected neighbourhood).

    Returns None if the map has no indexed pixels.
    """
    if ebsd is None or int(np.sum(ebsd.is_indexed)) == 0:
        return None

    R_all = ebsd.as_rotation_matrices()          # (N, 3, 3), NaN for unindexed
    indexed_idx = np.where(ebsd.is_indexed)[0]   # flat pixel indices
    N = len(indexed_idx)
    if N == 0:
        return None

    orientations = R_all[indexed_idx]            # (N, 3, 3)
    phase_ids = ebsd.phase_id[indexed_idx].astype(int)

    # Build pixel adjacency on the 2-D grid (4-connected)
    n_cols = max(ebsd.n_cols, 1)
    idx_map = {int(flat): local for local, flat in enumerate(indexed_idx)}

    adjacency: list[list[int]] = [[] for _ in range(N)]
    for local, flat in enumerate(indexed_idx):
        row, col = divmod(int(flat), n_cols)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < ebsd.n_rows and 0 <= nc < n_cols:
                neighbour_flat = nr * n_cols + nc
                if neighbour_flat in idx_map:
                    adjacency[local].append(idx_map[neighbour_flat])

    grain_sizes = np.ones(N, dtype=int)
    return GrainData.from_arrays(
        orientations=orientations,
        adjacency=adjacency,
        phase=phase_ids,
        grain_sizes=grain_sizes,
    )


def ebsd_data_summary_df(ebsd: EBSDData) -> pd.DataFrame:
    """Compact per-phase summary of an EBSDData object."""
    rows = []
    total = ebsd.n_pixels
    for pid, ph in ebsd.phases.items():
        count = int(np.sum(ebsd.phase_id == pid))
        rows.append({
            "phase_id": pid,
            "name": ph.name,
            "count": count,
            "fraction_%": round(100.0 * count / max(total, 1), 2),
            "space_group": ph.space_group,
            "laue_group": ph.laue_group,
            "a_Å": round(float(ph.lattice_params[0]), 4) if ph.lattice_params is not None else None,
            "b_Å": round(float(ph.lattice_params[1]), 4) if ph.lattice_params is not None else None,
            "c_Å": round(float(ph.lattice_params[2]), 4) if ph.lattice_params is not None else None,
        })
    return pd.DataFrame(rows)


def ebsd_data_to_variant_csv_df(ebsd: EBSDData) -> pd.DataFrame:
    """Export a flat CSV-friendly DataFrame from an EBSDData for downstream use."""
    df = pd.DataFrame({
        "x_um": ebsd.x,
        "y_um": ebsd.y,
        "phi1_deg": np.degrees(ebsd.euler1),
        "Phi_deg":  np.degrees(ebsd.euler2),
        "phi2_deg": np.degrees(ebsd.euler3),
        "phase_id": ebsd.phase_id,
        "mad_deg":  ebsd.mad,
        "bc":       ebsd.bc,
        "bs_ci":    ebsd.bs,
        "is_indexed": ebsd.is_indexed.astype(int),
    })
    if ebsd.detector_intensity is not None:
        df["detector_intensity"] = ebsd.detector_intensity
    return df


# ===========================================================================
# Session state initialisation
# ===========================================================================

def init_state() -> None:
    defaults: dict[str, Any] = {
        # legacy CSV-based orientation data
        "ebsd_df": None,
        "child_oris": None,
        "synthetic_parents": None,
        "assignment_result": None,
        "recon_result": None,
        "last_kinetics_df": None,
        "last_report": None,
        "dataset_origin": "none",
        # XRD / EDS / imaging
        "xrd_df": None,
        "xrd_peaks": None,
        "eds_df": None,
        "sem_images": [],
        "tem_images": [],
        # NEW: native EBSD data
        "ebsd_native": None,           # EBSDData from .ctf or .ang
        "ebsd_native_origin": "none",  # "ctf" | "ang" | "none"
        "native_grain_data": None,     # GrainData built from EBSDData
        # NEW: graph-based reconstruction result
        "graph_recon_result": None,    # ParentReconstructionResult
        "graph_or_used": "KS",         # OR name chosen for graph recon
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ===========================================================================
# App bootstrap
# ===========================================================================

css()
init_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Twin controls")
    st.caption(
        "These controls define the model assumptions and evidence state for this run. "
        "They do not create experimental data."
    )

    with st.expander("1. Material model", expanded=True):
        material_system = st.selectbox(
            "Material system",
            ["NiTi B2→B19′", "Steel fcc→bcc/bct"],
        )
        is_niti = material_system.startswith("NiTi")
        if is_niti:
            beta = st.number_input("B19′ beta angle (°)", value=96.8, min_value=90.0, max_value=110.0, step=0.1)
            steel_or = "KS"
        else:
            beta = 96.8
            steel_or = st.selectbox("Steel OR", ["KS", "NW", "Pitsch"])

    with st.expander("2. Fitting / reconstruction settings", expanded=True):
        tol = st.slider("Variant fit tolerance (°)", 1.0, 15.0, 5.0, 0.5)
        recon_thr = st.slider("Parent reconstruction threshold (°)", 1.0, 15.0, 5.0, 0.5)

    with st.expander("3. Synthetic data controls", expanded=False):
        grid_n = st.slider("Synthetic map size", 20, 100, 50, 10)
        n_parents = st.slider("Synthetic parent grains", 1, 8, 4, 1)
        active_fraction = st.slider("Active variant fraction", 0.1, 1.0, 0.55, 0.05)
        noise_deg = st.slider("Orientation noise (°)", 0.0, 5.0, 0.8, 0.1)

    with st.expander("4. Graph reconstruction settings (NEW)", expanded=True):
        st.caption("Controls for the new graph-based parent phase reconstruction (martwin.reconstruction).")
        graph_or_choice = st.selectbox(
            "Orientation relationship for graph reconstruction",
            ["auto-detect"] + list(OR_REGISTRY.keys()),
            help="'auto-detect' tests KS, NW, Pitsch, GT and picks the best fit from the EBSD data.",
        )
        refine_or_flag = st.checkbox(
            "Refine OR from data before reconstruction",
            value=True,
            help="Runs scipy optimisation to find the OR that minimises mean parent misorientation error.",
        )
        mcl_inflation = st.slider(
            "MCL inflation exponent", 1.2, 3.0, 2.0, 0.1,
            help="Higher → smaller, more distinct parent clusters. Lower → broader merging.",
        )
        mcl_iterations = st.slider("MCL max iterations", 10, 200, 60, 10)
        max_mad_filter = st.number_input(
            "Max MAD filter for .ctf (°)", value=1.5, min_value=0.0, max_value=5.0, step=0.1,
            help="Pixels with MAD above this are marked non-indexed before reconstruction.",
        )
        min_ci_filter = st.number_input(
            "Min CI filter for .ang (0–1)", value=0.1, min_value=0.0, max_value=1.0, step=0.05,
            help="Pixels with CI below this are marked non-indexed before reconstruction.",
        )
        ref_frame_convert = st.checkbox(
            "Convert .ctf to EDAX reference frame",
            value=False,
            help="Applies 180° rotation about RD to Oxford CTF Euler angles before use.",
        )

    with st.expander("5. Evidence available", expanded=True):
        composition_known = st.checkbox("composition known", value=False)
        heat_known = st.checkbox("heat treatment known", value=False)
        dsc_known = st.checkbox("DSC / transformation temperatures", value=False)
        xrd_known = st.checkbox("XRD refined lattice/phase fractions", value=False)
        xrd_pattern_known = st.checkbox("raw XRD pattern uploaded", value=False)
        sem_known = st.checkbox("SEM/optical micrographs", value=False)
        eds_known = st.checkbox("EDS/WDS chemistry data", value=False)
        tem_known = st.checkbox("TEM/STEM images", value=False)
        tem_diff_known = st.checkbox("TEM/SAED/4D-STEM diffraction", value=False)
        mech_known = st.checkbox("stress-strain / mechanical data", value=False)
        oxy_known = st.checkbox("oxygen/carbon/impurities known", value=False)
        cooling_known = st.checkbox("cooling curve / dilatometry", value=False)
        parent_ref_known = st.checkbox("known parent reference map", value=False)
        hardness_known = st.checkbox("hardness data", value=False)
        retained_known = st.checkbox("retained austenite XRD", value=False)
        lpbf = st.checkbox("LPBF/additive manufacturing route", value=False)
        laser_known = st.checkbox("LPBF laser parameters", value=False, disabled=not lpbf)
        scan_known = st.checkbox("LPBF scan strategy", value=False, disabled=not lpbf)
        powder_known = st.checkbox("LPBF powder chemistry", value=False, disabled=not lpbf)
        thermal_known = st.checkbox("thermal history / melt-pool model", value=False)
        porosity_known = st.checkbox("porosity data", value=False, disabled=not lpbf)
        residual_known = st.checkbox("residual stress data", value=False)

    with st.expander("6. Sample notes", expanded=False):
        sample_id = st.text_input("Sample ID", value="sample_001")
        process_route = st.selectbox(
            "Process route",
            ["unknown", "literature dataset", "cast/wrought", "solution treated",
             "aged", "quenched", "LPBF", "cold worked + annealed"],
        )
        composition_note = st.text_area("Composition note", height=80)
        heat_note = st.text_area("Heat treatment / process note", height=80)
        analyst_notes = st.text_area("Analyst notes", height=80)

# ---------------------------------------------------------------------------
# Build twin model
# ---------------------------------------------------------------------------

config = TwinConfiguration(
    material_system=material_system,
    beta_deg=beta,
    steel_or=steel_or,
    angular_tolerance_deg=tol,
    reconstruction_threshold_deg=recon_thr,
    notes=analyst_notes,
)
model = build_twin_model(config)

has_legacy_dataset = st.session_state.ebsd_df is not None
has_native_ebsd = st.session_state.ebsd_native is not None
has_any_dataset = has_legacy_dataset or has_native_ebsd

evidence = TwinEvidence(
    composition=composition_known or bool(composition_note.strip()),
    heat_treatment=heat_known or bool(heat_note.strip()),
    ebsd_or_tkd=has_any_dataset,
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
has_calibration = any([dsc_known, xrd_known, xrd_pattern_known, eds_known, cooling_known, mech_known, hardness_known])
has_process = heat_known or bool(heat_note.strip()) or thermal_known or cooling_known or laser_known
has_validation = any([parent_ref_known, retained_known, mech_known, hardness_known, sem_known, tem_known])
level, level_note = maturity_level(gap_report.confidence_score, has_any_dataset, has_calibration, has_process, has_validation)

# ---------------------------------------------------------------------------
# Header metrics bar
# ---------------------------------------------------------------------------

st.title("OpenMartensiteTwin v0.6.0")
st.caption(
    "Evidence-aware martensitic-transformation twin. "
    "v0.6 adds native .ctf/.ang EBSD readers and graph-based parent reconstruction."
)

n_dataset_pts = 0
if has_native_ebsd:
    n_dataset_pts = st.session_state.ebsd_native.n_pixels
elif has_legacy_dataset:
    n_dataset_pts = len(st.session_state.ebsd_df)

cols = st.columns(8)
cols[0].metric("Maturity", level.split(" — ")[0])
cols[1].metric("Material", "NiTi" if is_niti else "Steel")
cols[2].metric("OR", model.orientation_relationship.name.split()[0])
cols[3].metric("Variants", len(model.variants))
cols[4].metric("EBSD pixels", n_dataset_pts)
cols[5].metric("Confidence", f"{gap_report.confidence_score:.2f}")
cols[6].metric("Data source", st.session_state.dataset_origin)
cols[7].metric(
    "Graph recon",
    "done" if st.session_state.graph_recon_result is not None else "—",
)

st.warning(level_note)
st.markdown(
    "<span class='pill'>calculation ≠ validation</span>"
    "<span class='pill'>synthetic ≠ experiment</span>"
    "<span class='pill'>fit tolerance changes conclusions</span>"
    "<span class='pill'>native readers: v0.6 NEW</span>",
    unsafe_allow_html=True,
)

# ===========================================================================
# Tabs
# ===========================================================================

TABS = st.tabs([
    "0. Twin map",
    "1. Controls explained",
    "2. Evidence/state",
    "3. Crystallography",
    "4. EBSD workspace",        # extended: now has .ctf/.ang sub-section
    "5. Native EBSD reader",    # NEW tab
    "6. Graph reconstruction",  # NEW tab
    "7. Variant & parent",
    "8. Kinetics",
    "9. XRD/EDS/SEM/TEM",
    "10. Article gap map",
    "11. Open data/tools",
    "12. Defensibility gaps",
    "13. Report/export",
])

# ---------------------------------------------------------------------------
# TAB 0 — Twin map
# ---------------------------------------------------------------------------
with TABS[0]:
    st.header("What this app is actually doing")
    card(
        "Digital-twin loop",
        "<b>State</b> = material + process + measured data. "
        "<b>Model</b> = crystallography + kinetics + thermodynamics/mechanics. "
        "<b>Update</b> = compare predictions to EBSD/TKD/DSC/XRD/mechanical evidence. "
        "<b>Decision</b> = identify missing data and recommend the next experiment.",
    )
    st.subheader("v0.6 additions vs v0.5")
    card(
        "New in v0.6",
        "• <b>martwin.io</b>: native .ctf (Oxford HKL) and .ang (EDAX/TSL) readers. "
        "Returns <code>EBSDData</code> with Euler angles (radians), phase map, MAD/BC/BS/CI, "
        "multi-phase header, hex-grid support, reference-frame converters.<br>"
        "• <b>martwin.reconstruction</b>: graph-based parent phase reconstruction "
        "(Hielscher et al. 2022 variant graph + Markov clustering). "
        "Supports KS, NW, Pitsch, GT, Cayron-NiTi ORs; OR auto-detection; "
        "OR refinement via scipy; weighted quaternion parent voting; pixel-level back-projection.",
    )
    st.subheader("Current implementation vs complete target")
    st.dataframe(dataframe(TWIN_LAYER_MATRIX), hide_index=True, use_container_width=True)
    st.markdown("### Fidelity ladder")
    st.dataframe(dataframe(FIDELITY_LEVELS), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 1 — Controls explained
# ---------------------------------------------------------------------------
with TABS[1]:
    st.header("Every control: what it means and what it changes")
    st.dataframe(dataframe(PARAMETER_GUIDE), hide_index=True, use_container_width=True)
    st.markdown("### New controls in v0.6")
    new_controls = pd.DataFrame([
        {
            "control": "Graph OR choice",
            "changes": "OR used for the variant-graph edge probabilities",
            "caveat": "'auto-detect' compares KS/NW/Pitsch/GT and picks the best fit from measured data",
        },
        {
            "control": "Refine OR from data",
            "changes": "Applies scipy Nelder-Mead to minimise mean parent misorientation before MCL clustering",
            "caveat": "Refinement is iterative; may not converge on very noisy maps",
        },
        {
            "control": "MCL inflation",
            "changes": "Controls cluster granularity in Markov clustering of the variant graph",
            "caveat": "Higher values produce smaller clusters but may over-split; tune with synthetic data first",
        },
        {
            "control": "Max MAD filter",
            "changes": "Masks low-quality pixels in .ctf before building the grain graph",
            "caveat": "Too aggressive masking removes real variants near grain boundaries",
        },
        {
            "control": "Min CI filter",
            "changes": "Masks low-confidence pixels in .ang before reconstruction",
            "caveat": "CI < 0.1 usually means non-indexed or unreliable; CI > 0.2 is typically safe",
        },
        {
            "control": "Reference frame conversion",
            "changes": "Applies 180° rotation about RD to Oxford CTF Euler angles → EDAX frame",
            "caveat": "Only needed when comparing CTF and ANG from different vendors on the same scan",
        },
    ])
    st.dataframe(new_controls, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2 — Evidence / state vector
# ---------------------------------------------------------------------------
with TABS[2]:
    st.header("Evidence and state vector")
    section_help(
        "state vector",
        "The state vector describes the sample: material, OR, process, available measurements, assumptions and data source.",
        "Every downstream table and plot should be traceable to this state vector.",
        "A defensible twin must store raw data/provenance, not only checkboxes.",
    )
    native = st.session_state.ebsd_native
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
        "legacy_csv_points": len(st.session_state.ebsd_df) if has_legacy_dataset else 0,
        "native_ebsd_format": st.session_state.ebsd_native_origin,
        "native_ebsd_pixels": native.n_pixels if native else 0,
        "native_ebsd_indexed_%": round(100.0 * float(np.sum(native.is_indexed)) / max(native.n_pixels, 1), 2) if native else 0,
        "native_ebsd_grid": f"{native.n_cols}×{native.n_rows} {native.grid_type}" if native else "—",
        "xrd_pattern_uploaded": st.session_state.xrd_df is not None,
        "eds_table_uploaded": st.session_state.eds_df is not None,
        "sem_image_count": len(st.session_state.sem_images),
        "tem_image_count": len(st.session_state.tem_images),
        "graph_reconstruction_done": st.session_state.graph_recon_result is not None,
        "graph_or_used": st.session_state.graph_or_used,
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
    df_download("Download evidence table", e_df, "evidence_v06.csv")

# ---------------------------------------------------------------------------
# TAB 3 — Crystallography
# ---------------------------------------------------------------------------
with TABS[3]:
    st.header("Crystallographic model")
    section_help(
        "orientation relationship and variants",
        "The OR matrix maps a child/martensite crystal frame to the parent crystal frame.",
        "Variant assignment compares measured EBSD/TKD orientations against these theoretical variants.",
        "The current NiTi model is a Cayron-inspired prototype; exact work requires sample-specific lattice parameters and convention checks.",
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Orientation relationship")
        st.write(
            getattr(
                model.orientation_relationship,
                "description",
                getattr(model.orientation_relationship, "source_note", model.orientation_relationship.name),
            )
        )
        or_df = pd.DataFrame(
            model.orientation_relationship.matrix_child_to_parent,
            columns=["parent x", "parent y", "parent z"],
            index=["child x", "child y", "child z"],
        )
        st.dataframe(or_df.style.format("{:.4f}"), use_container_width=True)
    with c2:
        st.subheader("Variant library (first 12)")
        tables = variant_library_tables(model)
        vdf = tables["variants"].copy()
        st.dataframe(
            vdf.head(12).style.format({c: "{:.4f}" for c in vdf.columns if c.startswith("r")}),
            use_container_width=True,
        )
    st.subheader("Pairwise theoretical variant misorientation matrix (°)")
    st.dataframe(tables["misorientation_matrix_deg"].style.format("{:.1f}"), use_container_width=True)

    st.divider()
    st.subheader("OR_REGISTRY — built-in orientation relationships")
    st.caption(
        "These are the ORs available to the graph-based reconstruction module. "
        "Each is a list of rotation matrices generated by applying Oh cubic symmetry to the prototype variant."
    )
    or_rows = []
    for name, or_obj in OR_REGISTRY.items():
        or_rows.append({
            "name": name,
            "n_variants": or_obj.n_variants,
            "description": or_obj.description,
        })
    st.dataframe(pd.DataFrame(or_rows), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 4 — EBSD workspace  (legacy CSV + synthetic, unchanged)
# ---------------------------------------------------------------------------
with TABS[4]:
    st.header("EBSD data workspace (CSV / synthetic)")
    section_help(
        "CSV/synthetic EBSD data",
        "Legacy CSV-based orientation data (r00…r22 or Bunge Euler phi1/Phi/phi2) and synthetic test maps.",
        "Use Tab 5 for native .ctf / .ang file loading.",
        "Synthetic data tests the code path only; it is not experimental evidence.",
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
            st.success(
                f"Generated {len(synthetic.dataframe)} synthetic points. "
                "This tests code only; it is not experimental evidence."
            )
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

    if has_legacy_dataset:
        df = st.session_state.ebsd_df
        st.subheader("Dataset preview")
        st.dataframe(df.drop(columns=["orientation_matrix"], errors="ignore").head(50), use_container_width=True)
        df_download(
            "Download dataset CSV",
            df.drop(columns=["orientation_matrix"], errors="ignore"),
            "orientation_dataset.csv",
        )
    else:
        st.info("No CSV/synthetic dataset loaded yet.")

# ---------------------------------------------------------------------------
# TAB 5 — Native EBSD reader  (NEW)
# ---------------------------------------------------------------------------
with TABS[5]:
    st.header("Native EBSD file reader  ✦ NEW in v0.6")
    section_help(
        ".ctf and .ang file readers",
        "martwin.io provides native Python parsers for Oxford Instruments .ctf (HKL Channel 5) "
        "and EDAX/TSL .ang files. No MATLAB, orix or third-party format library required.",
        "The loaded EBSDData is used for variant assignment, graph-based parent reconstruction, "
        "phase statistics, MAD/CI quality maps, and export.",
        "Always check the indexed fraction, MAD distribution and phase proportions before running reconstruction.",
    )

    upload_col, info_col = st.columns([1, 1])

    with upload_col:
        st.subheader("Upload .ctf file (Oxford HKL)")
        ctf_upload = st.file_uploader("Upload .ctf", type=["ctf"], key="ctf_uploader")
        if ctf_upload is not None:
            with tempfile.NamedTemporaryFile(suffix=".ctf", delete=False) as tmp:
                tmp.write(ctf_upload.read())
                tmp_path = pathlib.Path(tmp.name)
            try:
                with st.spinner("Parsing .ctf file…"):
                    ebsd = load_ctf(tmp_path, validate=True, max_mad_deg=float(max_mad_filter))
                if ref_frame_convert:
                    ebsd = oxford_to_edax(ebsd)
                    st.info("Reference frame converted: Oxford → EDAX (180° about RD).")
                st.session_state.ebsd_native = ebsd
                st.session_state.ebsd_native_origin = "ctf"
                st.session_state.dataset_origin = f"ctf:{ctf_upload.name}"
                st.session_state.native_grain_data = None
                st.session_state.graph_recon_result = None
                st.success(
                    f"Loaded {ebsd.n_pixels:,} pixels "
                    f"({int(np.sum(ebsd.is_indexed)):,} indexed, "
                    f"{100*float(np.sum(ebsd.is_indexed))/max(ebsd.n_pixels,1):.1f}%)."
                )
            except Exception as exc:
                st.error(f"CTF parse error: {exc}")

        st.subheader("Upload .ang file (EDAX/TSL)")
        ang_upload = st.file_uploader("Upload .ang", type=["ang"], key="ang_uploader")
        if ang_upload is not None:
            with tempfile.NamedTemporaryFile(suffix=".ang", delete=False) as tmp:
                tmp.write(ang_upload.read())
                tmp_path = pathlib.Path(tmp.name)
            try:
                with st.spinner("Parsing .ang file…"):
                    ebsd = load_ang(
                        tmp_path, validate=True,
                        min_ci=float(min_ci_filter),
                        convert_hex_to_square=True,
                    )
                st.session_state.ebsd_native = ebsd
                st.session_state.ebsd_native_origin = "ang"
                st.session_state.dataset_origin = f"ang:{ang_upload.name}"
                st.session_state.native_grain_data = None
                st.session_state.graph_recon_result = None
                st.success(
                    f"Loaded {ebsd.n_pixels:,} pixels "
                    f"({int(np.sum(ebsd.is_indexed)):,} indexed, "
                    f"{100*float(np.sum(ebsd.is_indexed))/max(ebsd.n_pixels,1):.1f}%)."
                )
            except Exception as exc:
                st.error(f"ANG parse error: {exc}")

    with info_col:
        st.subheader("Format reference")
        fmt_df = pd.DataFrame([
            {
                "format": ".ctf",
                "vendor": "Oxford Instruments / HKL",
                "angles": "Euler degrees → stored as radians",
                "quality": "MAD, BC, BS, Bands, Error",
                "grid": "square + hex",
                "notes": "Multi-phase header; lattice params per phase",
            },
            {
                "format": ".ang",
                "vendor": "EDAX / TSL OIM",
                "angles": "Euler radians (native)",
                "quality": "IQ, CI, SEM signal, Fit",
                "grid": "square + hex (auto-resampled)",
                "notes": "Phase blocks with LatticeConstants; CI 0–1",
            },
        ])
        st.dataframe(fmt_df, hide_index=True, use_container_width=True)
        st.markdown(
            "**Column mapping to EBSDData:**\n"
            "- `euler1/2/3` — Bunge φ₁, Φ, φ₂ in **radians** (both formats)\n"
            "- `mad` — MAD (CTF) or Fit quality (ANG), in degrees\n"
            "- `bc` — Band contrast (CTF) or IQ (ANG)\n"
            "- `bs` — Band slope (CTF) or CI 0–1 (ANG)\n"
            "- `is_indexed` — True where phase_id > 0\n"
            "- `as_rotation_matrices()` — (N,3,3) Bunge ZXZ passive"
        )

    # ---- Show loaded EBSDData ----
    if st.session_state.ebsd_native is not None:
        ebsd: EBSDData = st.session_state.ebsd_native
        st.divider()
        st.subheader(f"Loaded: {st.session_state.ebsd_native_origin.upper()}  —  {ebsd.source_file.split('/')[-1]}")

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Total pixels", f"{ebsd.n_pixels:,}")
        m2.metric("Indexed pixels", f"{int(np.sum(ebsd.is_indexed)):,}")
        m3.metric("Indexed %", f"{100*float(np.sum(ebsd.is_indexed))/max(ebsd.n_pixels,1):.1f}%")
        m4.metric("Grid", f"{ebsd.n_cols}×{ebsd.n_rows}")
        m5.metric("Step (µm)", f"{ebsd.x_step:.3f}")
        m6.metric("Grid type", ebsd.grid_type)

        st.subheader("Per-phase statistics")
        ph_df = ebsd_data_summary_df(ebsd)
        st.dataframe(ph_df, hide_index=True, use_container_width=True)

        st.subheader("Quality metric distributions (indexed pixels only)")
        if int(np.sum(ebsd.is_indexed)) > 0:
            q_col1, q_col2, q_col3 = st.columns(3)
            with q_col1:
                fig_mad, ax = plt.subplots(figsize=(4, 3))
                indexed_mad = ebsd.mad[ebsd.is_indexed]
                ax.hist(indexed_mad, bins=40, color="#3b82f6", edgecolor="none")
                ax.set_xlabel("MAD / Fit quality (°)")
                ax.set_ylabel("Count")
                ax.set_title("MAD distribution")
                ax.grid(True, alpha=0.2)
                st.pyplot(fig_mad)
            with q_col2:
                fig_bc, ax = plt.subplots(figsize=(4, 3))
                ax.hist(ebsd.bc[ebsd.is_indexed], bins=40, color="#10b981", edgecolor="none")
                ax.set_xlabel("BC / IQ")
                ax.set_ylabel("Count")
                ax.set_title("Band contrast / IQ")
                ax.grid(True, alpha=0.2)
                st.pyplot(fig_bc)
            with q_col3:
                fig_bs, ax = plt.subplots(figsize=(4, 3))
                ax.hist(ebsd.bs[ebsd.is_indexed], bins=40, color="#f59e0b", edgecolor="none")
                ax.set_xlabel("BS / CI")
                ax.set_ylabel("Count")
                ax.set_title("Band slope / CI")
                ax.grid(True, alpha=0.2)
                st.pyplot(fig_bs)

        st.subheader("Orientation quality map (BC / IQ)")
        if ebsd.n_rows > 1 and ebsd.n_cols > 1:
            fig_map, ax = plt.subplots(figsize=(8, 4))
            bc_map = ebsd.map_array(ebsd.bc.astype(float))
            im = ax.imshow(bc_map, origin="upper", cmap="gray", aspect="auto")
            plt.colorbar(im, ax=ax, label="BC / IQ")
            ax.set_title("Band contrast / IQ map")
            ax.set_xlabel("x pixels")
            ax.set_ylabel("y pixels")
            st.pyplot(fig_map)
        else:
            st.caption("Map too small to display 2-D image.")

        st.subheader("Phase ID map")
        if ebsd.n_rows > 1 and ebsd.n_cols > 1:
            fig_phase, ax = plt.subplots(figsize=(8, 4))
            phase_map = ebsd.map_array(ebsd.phase_id.astype(float))
            im = ax.imshow(phase_map, origin="upper", cmap="tab10", aspect="auto")
            plt.colorbar(im, ax=ax, label="Phase ID")
            ax.set_title("Phase ID map (0 = not indexed)")
            ax.set_xlabel("x pixels")
            ax.set_ylabel("y pixels")
            st.pyplot(fig_phase)

        st.subheader("Export EBSDData as CSV")
        flat_df = ebsd_data_to_variant_csv_df(ebsd)
        df_download("Download EBSDData CSV", flat_df, "ebsd_native_export.csv")
        st.caption(
            "Columns: x_um, y_um, phi1_deg, Phi_deg, phi2_deg (Bunge, converted from internal radians), "
            "phase_id, mad_deg, bc, bs_ci, is_indexed, detector_intensity."
        )

        st.subheader("Header metadata")
        if ebsd.header_raw:
            hdr_df = pd.DataFrame(
                [{"key": k, "value": v} for k, v in ebsd.header_raw.items()]
            )
            st.dataframe(hdr_df, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 6 — Graph-based parent reconstruction  (NEW)
# ---------------------------------------------------------------------------
with TABS[6]:
    st.header("Graph-based parent phase reconstruction  ✦ NEW in v0.6")
    section_help(
        "variant graph + Markov clustering",
        "Implements the Hielscher–Nyyssönen–Niessen–Gazder (2022) variant graph algorithm. "
        "Each (grain, variant) pair is a node. Edges carry the probability that two adjacent grains "
        "share a common parent. Markov clustering groups nodes into parent grains.",
        "The reconstructed parent orientation per cluster is the weighted quaternion mean of all "
        "child orientations in that cluster (Markley 2007).",
        "v0.6 treats each indexed pixel as a grain (pixel-level graph). "
        "For publication-grade work, segment grains first and use grain-mean orientations as nodes.",
    )

    if not has_native_ebsd:
        st.info(
            "Load a .ctf or .ang file in **Tab 5** first. "
            "The graph reconstruction uses EBSDData from the native readers."
        )
    else:
        ebsd: EBSDData = st.session_state.ebsd_native

        # OR selection
        run_or_name = graph_or_choice if graph_or_choice != "auto-detect" else None

        st.subheader("Step 1 — Build grain graph from loaded EBSD data")
        st.caption(
            f"Indexed pixels: {int(np.sum(ebsd.is_indexed)):,}  |  "
            f"OR: {'auto-detect' if run_or_name is None else run_or_name}  |  "
            f"MCL inflation: {mcl_inflation}  |  refine OR: {refine_or_flag}"
        )

        run_graph = st.button("Run graph-based parent reconstruction", type="primary")

        if run_graph:
            with st.spinner("Building GrainData from EBSDData…"):
                grain_data = ebsd_data_to_grain_data(ebsd)

            if grain_data is None:
                st.error("No indexed pixels found in the loaded EBSD map. Check MAD/CI filter settings.")
            else:
                st.session_state.native_grain_data = grain_data
                n_grains = grain_data.n_grains

                # Clamp: for very large maps run on a sub-sample to avoid timeout
                MAX_GRAINS = 5000
                if n_grains > MAX_GRAINS:
                    st.warning(
                        f"Map has {n_grains:,} indexed pixels. "
                        f"Subsampling to {MAX_GRAINS:,} random pixels for reconstruction speed. "
                        "Segment grains first for full-map use."
                    )
                    rng = np.random.default_rng(42)
                    idx = rng.choice(n_grains, MAX_GRAINS, replace=False)
                    sub_orients = grain_data.orientations[idx]
                    sub_adj = []
                    idx_set = set(idx.tolist())
                    old_to_new = {int(old): new for new, old in enumerate(idx)}
                    for new, old in enumerate(idx):
                        sub_adj.append([
                            old_to_new[nb] for nb in grain_data.adjacency[old]
                            if nb in idx_set
                        ])
                    sub_phase = grain_data.phase[idx]
                    sub_sizes = grain_data.grain_sizes[idx]
                    grain_data = GrainData.from_arrays(sub_orients, sub_adj, sub_phase, sub_sizes)
                    n_grains = grain_data.n_grains

                # Auto-detect OR if requested
                actual_or_name = run_or_name
                if actual_or_name is None:
                    with st.spinner("Auto-detecting best OR…"):
                        detected, score = detect_OR(grain_data, verbose=False)
                    actual_or_name = detected
                    st.info(f"Auto-detected OR: **{detected}** (mean parent misfit {score:.2f}°)")

                st.session_state.graph_or_used = actual_or_name

                with st.spinner(f"Running ParentReconstructor (OR={actual_or_name}, MCL r={mcl_inflation})…"):
                    try:
                        rec = ParentReconstructor(
                            grain_data,
                            or_name=actual_or_name,
                            refine_or=refine_or_flag,
                            threshold_deg=recon_thr,
                            tolerance_deg=recon_thr,
                            mcl_inflation=mcl_inflation,
                            mcl_iterations=mcl_iterations,
                        )
                        result: ParentReconstructionResult = rec.run(verbose=False)
                        st.session_state.graph_recon_result = result
                        st.success("Reconstruction complete.")
                    except Exception as exc:
                        st.error(f"Reconstruction failed: {exc}")
                        st.session_state.graph_recon_result = None

        # ---- Display results ----
        result: ParentReconstructionResult | None = st.session_state.graph_recon_result

        if result is not None:
            st.divider()
            st.subheader("Reconstruction results")

            r1, r2, r3, r4, r5 = st.columns(5)
            r1.metric("OR used", result.or_used.name)
            r2.metric("Parent grains found", result.n_parent_grains)
            r3.metric("Reconstruction fraction", f"{result.reconstruction_fraction:.1%}")
            r4.metric("Mean fit (°)", f"{float(np.nanmean(result.fit)):.2f}")
            r5.metric("Variants in OR", result.or_used.n_variants)

            st.text(result.summary())

            # Fit distribution
            fit_valid = result.fit[~np.isnan(result.fit)]
            if len(fit_valid) > 0:
                fig_fit, ax = plt.subplots(figsize=(8, 3))
                ax.hist(fit_valid, bins=60, color="#6366f1", edgecolor="none")
                ax.axvline(float(np.mean(fit_valid)), color="red", linestyle="--", label=f"Mean {np.mean(fit_valid):.2f}°")
                ax.set_xlabel("Fit (°) — misorientation between child-implied parent and cluster mean parent")
                ax.set_ylabel("Count")
                ax.set_title("Parent orientation fit distribution")
                ax.legend()
                ax.grid(True, alpha=0.2)
                st.pyplot(fig_fit)

            # Cluster size distribution
            _, cluster_sizes = np.unique(result.parent_grain_id[result.parent_grain_id >= 0], return_counts=True)
            if len(cluster_sizes) > 0:
                fig_cs, ax = plt.subplots(figsize=(8, 3))
                ax.hist(cluster_sizes, bins=min(40, len(cluster_sizes)), color="#10b981", edgecolor="none")
                ax.set_xlabel("Pixels per reconstructed parent grain")
                ax.set_ylabel("Count")
                ax.set_title("Reconstructed parent grain size distribution")
                ax.grid(True, alpha=0.2)
                st.pyplot(fig_cs)

            # Variant assignment summary
            st.subheader("Variant assignment summary")
            vid_counts = pd.Series(result.variant_id[result.variant_id >= 0]).value_counts().sort_index()
            v_df = pd.DataFrame({
                "variant_id": vid_counts.index,
                "count": vid_counts.values,
                "fraction_%": (100.0 * vid_counts.values / max(len(result.variant_id), 1)).round(2),
            })
            st.dataframe(v_df, hide_index=True, use_container_width=True)

            # Export
            export_df = pd.DataFrame({
                "pixel_index": np.arange(len(result.parent_grain_id)),
                "parent_grain_id": result.parent_grain_id,
                "variant_id": result.variant_id,
                "fit_deg": result.fit,
            })
            df_download("Download reconstruction result CSV", export_df, "graph_reconstruction_result.csv")

            st.subheader("Parent orientation matrices (first 10 parent grains)")
            po = result.parent_orientations
            unique_pids = sorted(set(result.parent_grain_id[result.parent_grain_id >= 0].tolist()))[:10]
            po_rows = []
            for pid in unique_pids:
                members = np.where(result.parent_grain_id == pid)[0]
                if len(members) == 0:
                    continue
                rep = members[0]
                Rp = po[rep]
                if np.any(np.isnan(Rp)):
                    continue
                po_rows.append({
                    "parent_id": pid,
                    "n_child_pixels": len(members),
                    "mean_fit_deg": round(float(np.nanmean(result.fit[members])), 3),
                    "R00": round(float(Rp[0, 0]), 4), "R01": round(float(Rp[0, 1]), 4), "R02": round(float(Rp[0, 2]), 4),
                    "R10": round(float(Rp[1, 0]), 4), "R11": round(float(Rp[1, 1]), 4), "R12": round(float(Rp[1, 2]), 4),
                    "R20": round(float(Rp[2, 0]), 4), "R21": round(float(Rp[2, 1]), 4), "R22": round(float(Rp[2, 2]), 4),
                })
            if po_rows:
                st.dataframe(pd.DataFrame(po_rows), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 7 — Variant & parent analysis  (unchanged logic, now also accepts native EBSD)
# ---------------------------------------------------------------------------
with TABS[7]:
    st.header("Variant assignment and parent reconstruction (CSV/synthetic path)")
    section_help(
        "variant assignment",
        "Each measured child orientation is compared with every theoretical variant. "
        "The closest variant gets assigned and the angular error is reported.",
        "Low angular error and high in-tolerance fraction support the chosen OR/model.",
        "For .ctf/.ang files, the graph reconstruction in Tab 6 is preferred. "
        "This tab remains for CSV/synthetic data and cross-validation.",
    )
    if not has_legacy_dataset:
        st.info("Load CSV or generate synthetic data in Tab 4 first.")
    else:
        run = st.button("Run variant + parent analysis (CSV/synthetic)", type="primary")
        if run or st.session_state.assignment_result is not None:
            if run:
                if (
                    st.session_state.synthetic_parents is not None
                    and "parent_region_id" in st.session_state.ebsd_df.columns
                ):
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

            st.dataframe(assignment.summary, hide_index=True, use_container_width=True)
            st.dataframe(assignment.assignments.head(200), hide_index=True, use_container_width=True)

            plot_df = st.session_state.ebsd_df.copy()
            plot_df["variant_id"] = assignment.assignments["variant_id"].values
            plot_df["angular_error_deg"] = assignment.assignments["angular_error_deg"].values
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(plot_variant_map(plot_df, value_col="variant_id", title="Assigned variant ID map"))
            with c2:
                st.pyplot(plot_variant_map(plot_df, value_col="angular_error_deg", title="Angular fit error (°)"))

            if recon:
                plot_df["parent_cluster"] = recon.labels
                st.pyplot(plot_variant_map(plot_df, value_col="parent_cluster", title="Prototype parent clusters"))

# ---------------------------------------------------------------------------
# TAB 8 — Kinetics  (unchanged)
# ---------------------------------------------------------------------------
with TABS[8]:
    st.header("Transformation kinetics")
    section_help(
        "kinetics curve",
        "Kinetics connects temperature/cooling/heating history to phase fraction.",
        "Compare against DSC/dilatometry/XRD phase-fraction data and estimate phase fraction at a temperature.",
        "v0.6 kinetics is educational unless you supply measured DSC/dilatometry data and fit parameters.",
    )
    if is_niti:
        Ms = st.number_input("Ms (°C)", value=30.0)
        Mf = st.number_input("Mf (°C)", value=-10.0)
        As = st.number_input("As (°C)", value=15.0)
        Af = st.number_input("Af (°C)", value=55.0)
        tmin = min(Mf, As, Ms, Af) - 30
        tmax = max(Mf, As, Ms, Af) + 30
        temps = np.linspace(tmin, tmax, 220)
        trans = make_niti_temperatures(Ms=Ms, Mf=Mf, As=As, Af=Af)
        cooling = [linear_cooling_fraction(float(T), trans) for T in temps]
        heating = [linear_heating_fraction_austenite(float(T), trans) for T in temps]
        kinetics_df = pd.DataFrame({"temperature_C": temps, "B19prime_fraction_cooling": cooling, "B2_fraction_heating": heating})
        st.pyplot(make_niti_kinetics_plot(temps, cooling, heating, Ms, Mf, As, Af))
    else:
        Ms = st.number_input("Ms (°C)", value=420.0)
        alpha = st.number_input("KM alpha", value=0.011, min_value=0.0001, max_value=0.1, step=0.001, format="%.4f")
        temps = np.linspace(Ms + 80, Ms - 300, 220)
        frac = [km_curve(float(T), Ms=Ms, alpha=alpha) for T in temps]
        kinetics_df = pd.DataFrame({"temperature_C": temps, "martensite_fraction": frac})
        st.pyplot(make_steel_kinetics_plot(temps, frac, Ms, alpha))
    st.session_state.last_kinetics_df = kinetics_df
    st.dataframe(kinetics_df.head(40), use_container_width=True)
    df_download("Download kinetics curve", kinetics_df, "kinetics_curve.csv")

# ---------------------------------------------------------------------------
# TAB 9 — XRD / EDS / SEM / TEM  (unchanged)
# ---------------------------------------------------------------------------
with TABS[9]:
    st.header("XRD, EDS, SEM and TEM characterization evidence")
    section_help(
        "characterization evidence",
        "A defensible twin needs XRD (phases/lattice), EDS (chemistry), SEM (morphology), TEM (nanoscale structure).",
        "These measurements update the state vector and reduce uncertainty.",
        "XRD peak preview ≠ Rietveld refinement. Use GSAS-II/MAUD for publication-grade phase fractions.",
    )
    st.dataframe(dataframe(CHARACTERIZATION_MODULES), hide_index=True, use_container_width=True)
    st.divider()

    xcol, ecol = st.columns(2)
    with xcol:
        st.subheader("XRD / diffraction CSV preview")
        xrd_upload = st.file_uploader("Upload XRD CSV", type=["csv"], key="xrd_upload")
        if xrd_upload is not None:
            try:
                xdf = pd.read_csv(xrd_upload)
                fig, peaks = plot_xrd_pattern(xdf)
                st.session_state.xrd_df = xdf
                st.session_state.xrd_peaks = peaks
                st.pyplot(fig)
                st.dataframe(peaks.head(30), hide_index=True, use_container_width=True)
            except Exception as exc:
                st.error(f"Could not read/plot XRD CSV: {exc}")
        elif st.session_state.xrd_df is not None:
            fig, peaks = plot_xrd_pattern(st.session_state.xrd_df)
            st.pyplot(fig)
            st.dataframe(peaks.head(30), hide_index=True, use_container_width=True)

    with ecol:
        st.subheader("EDS/WDS chemistry table")
        eds_upload = st.file_uploader("Upload EDS/WDS CSV", type=["csv"], key="eds_upload")
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
                ni, ti = vals.get("ni"), vals.get("ti")
                if ni is not None and ti is not None and ti != 0:
                    st.metric("Ni/Ti atomic ratio", f"{ni/ti:.4f}")

    scol, tcol = st.columns(2)
    with scol:
        st.subheader("SEM / optical morphology")
        sem_files = st.file_uploader("Upload SEM/optical images", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True, key="sem_upload")
        if sem_files:
            st.session_state.sem_images = [f.name for f in sem_files]
            for f in sem_files[:3]:
                st.image(f, caption=f"SEM/optical: {f.name}", use_container_width=True)
    with tcol:
        st.subheader("TEM / STEM / SAED")
        tem_files = st.file_uploader("Upload TEM/STEM images", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True, key="tem_upload")
        if tem_files:
            st.session_state.tem_images = [f.name for f in tem_files]
            for f in tem_files[:3]:
                st.image(f, caption=f"TEM/STEM: {f.name}", use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 10 — Article gap map  (unchanged)
# ---------------------------------------------------------------------------
with TABS[10]:
    st.header("Article-derived missing-data map")
    article_df = dataframe(ARTICLE_EVIDENCE_MAP)
    st.dataframe(article_df, hide_index=True, use_container_width=True)
    df_download("Download article gap map", article_df, "article_derived_gap_map_v06.csv")

# ---------------------------------------------------------------------------
# TAB 11 — Open data/tools  (unchanged)
# ---------------------------------------------------------------------------
with TABS[11]:
    st.header("Open data and open tools")
    sources_df = dataframe(OPEN_SOURCE_DATASETS)
    st.dataframe(sources_df, hide_index=True, use_container_width=True)
    df_download("Download open-data manifest", sources_df, "open_data_manifest_v06.csv")

# ---------------------------------------------------------------------------
# TAB 12 — Defensibility gaps  (unchanged + graph recon gap)
# ---------------------------------------------------------------------------
with TABS[12]:
    st.header("What is missing before we can call this the most comprehensive twin?")
    gap_df = dataframe(DEFENSIBILITY_REQUIREMENTS)
    st.dataframe(gap_df, hide_index=True, use_container_width=True)
    missing_df = (
        pd.DataFrame({"missing_item": gap_report.missing})
        if gap_report.missing
        else pd.DataFrame({"missing_item": ["none from current checklist"]})
    )
    st.subheader("Missing data for this run")
    st.dataframe(missing_df, hide_index=True, use_container_width=True)
    st.subheader("Recommended next steps")
    for item in gap_report.recommended_next_experiments:
        st.write(f"- {item}")
    st.subheader("v0.6 remaining gaps in martwin.reconstruction")
    recon_gaps = pd.DataFrame([
        {"gap": "native .ctf/.ang grain segmentation", "status": "pixel-level only; grain-mean OR needed for large maps"},
        {"gap": "MTEX-style grain-boundary topology input", "status": "adjacency built from 4-connected pixel grid"},
        {"gap": "Σ3 twin boundary handling", "status": "not yet implemented; leads to over-splitting near twins"},
        {"gap": "OR refinement convergence on noisy maps", "status": "Nelder-Mead may not converge; add bounds"},
        {"gap": "pixel-level back-projection at scale", "status": "implemented but slow for >100k pixel maps"},
        {"gap": "independent benchmark validation", "status": "no published steel/NiTi dataset tested yet"},
        {"gap": "uncertainty propagation (SALib/Sobol)", "status": "planned; not yet in v0.6"},
    ])
    st.dataframe(recon_gaps, hide_index=True, use_container_width=True)
    st.error(
        "For a defensible 'most comprehensive' claim: "
        "graph reconstruction needs grain-level segmentation input, "
        "Σ3 twin boundary awareness, "
        "real open steel EBSD dataset validation, "
        "and raw same-sample NiTi EBSD/TKD/DSC/XRD/SEM/EDS/TEM evidence."
    )

# ---------------------------------------------------------------------------
# TAB 13 — Report / export  (extended with native EBSD and graph recon data)
# ---------------------------------------------------------------------------
with TABS[13]:
    st.header("Report / export")

    assignment = st.session_state.assignment_result
    graph_result: ParentReconstructionResult | None = st.session_state.graph_recon_result
    native: EBSDData | None = st.session_state.ebsd_native

    metrics: dict[str, Any] = {
        "maturity_level": level,
        "confidence_score": gap_report.confidence_score,
        "material_system": material_system,
        "orientation_relationship": model.orientation_relationship.name,
        "n_variants": len(model.variants),
        "dataset_origin": st.session_state.dataset_origin,
        "legacy_csv_points": len(st.session_state.ebsd_df) if has_legacy_dataset else 0,
        "native_ebsd_format": st.session_state.ebsd_native_origin,
        "native_ebsd_pixels": native.n_pixels if native else 0,
        "native_ebsd_indexed_%": round(
            100.0 * float(np.sum(native.is_indexed)) / max(native.n_pixels, 1), 2
        ) if native else 0,
        "graph_recon_done": graph_result is not None,
        "graph_or_used": st.session_state.graph_or_used,
        "graph_n_parent_grains": graph_result.n_parent_grains if graph_result else 0,
        "graph_reconstruction_fraction": graph_result.reconstruction_fraction if graph_result else 0,
        "graph_mean_fit_deg": float(np.nanmean(graph_result.fit)) if graph_result else None,
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

    notes = "\n".join([
        f"Sample ID: {sample_id}",
        f"Process route: {process_route}",
        composition_note,
        heat_note,
        analyst_notes,
    ])
    assignment_summary = assignment.summary if assignment else None
    gap_df = dataframe(DEFENSIBILITY_REQUIREMENTS)

    md = build_markdown_report_safe(model, assignment_summary, metrics, gap_report, notes=notes)
    md += "\n\n## Evidence table\n\n" + df_markdown_safe(evidence_table(evidence))
    md += "\n\n## Defensibility gap register\n\n" + df_markdown_safe(gap_df)

    if graph_result is not None:
        md += f"\n\n## Graph reconstruction summary\n\n{graph_result.summary()}"

    if native is not None:
        md += "\n\n## Native EBSD file statistics\n\n"
        md += df_markdown_safe(ebsd_data_summary_df(native))

    st.text_area("Markdown report preview", value=md, height=500)
    st.download_button(
        "Download Markdown report",
        md.encode("utf-8"),
        "open_martensite_twin_v060_report.md",
        "text/markdown",
    )

    json_report = build_local_json_report(model, assignment_summary, metrics, gap_report, notes=notes)
    json_report["state_vector"] = json_safe(state if "state" in dir() else {})
    json_report["evidence"] = json_safe(evidence_table(evidence))
    json_report["defensibility_gaps"] = json_safe(gap_df)
    if graph_result is not None:
        json_report["graph_reconstruction"] = {
            "or_used": graph_result.or_used.name,
            "n_parent_grains": graph_result.n_parent_grains,
            "reconstruction_fraction": graph_result.reconstruction_fraction,
            "mean_fit_deg": float(np.nanmean(graph_result.fit)),
        }
    if native is not None:
        json_report["native_ebsd"] = {
            "format": native.file_format,
            "source_file": native.source_file,
            "n_pixels": native.n_pixels,
            "n_indexed": int(np.sum(native.is_indexed)),
            "grid": f"{native.n_cols}x{native.n_rows}",
            "step_um": [native.x_step, native.y_step],
            "phases": {
                str(pid): {"name": ph.name, "space_group": ph.space_group}
                for pid, ph in native.phases.items()
            },
        }

    st.download_button(
        "Download JSON report",
        json.dumps(json_report, indent=2).encode("utf-8"),
        "open_martensite_twin_v060_report.json",
        "application/json",
    )
