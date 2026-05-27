from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from martwin.core.symmetry import cubic_proper_rotations, monoclinic_2_unique_axis_b
from martwin.crystallography.orientation_relationships import cayron_niti_natural_or, steel_ks_or, steel_nw_or, steel_pitsch_or
from martwin.crystallography.variants import generate_variants
from martwin.calibration.gap_analysis import assess_data_gaps
from martwin.io.manifest import read_open_data_manifest

st.set_page_config(page_title="Martensite Twin v0.1", layout="wide")
st.title("Martensite Twin v0.1")
st.caption("Crystallography-first scaffold for a martensitic-transformation digital twin.")

system = st.selectbox("Material system", ["NiTi B2→B19′", "Steel fcc→bcc/bct"])

if system.startswith("NiTi"):
    beta = st.number_input("B19′ monoclinic beta angle (deg)", value=96.8, min_value=90.0, max_value=110.0, step=0.1)
    orx = cayron_niti_natural_or(beta)
    variants = generate_variants(orx, cubic_proper_rotations(), monoclinic_2_unique_axis_b())
else:
    or_name = st.selectbox("Orientation relationship", ["KS", "NW", "Pitsch"])
    orx = {"KS": steel_ks_or, "NW": steel_nw_or, "Pitsch": steel_pitsch_or}[or_name]()
    variants = generate_variants(orx, cubic_proper_rotations(), cubic_proper_rotations())

col1, col2 = st.columns(2)
with col1:
    st.subheader("Orientation relationship")
    st.write(orx.name)
    st.write(orx.source_note)
    st.metric("Unique orientation variants in current convention", len(variants))
with col2:
    st.subheader("Data gaps")
    available = {
        "composition": st.checkbox("composition known"),
        "heat_treatment": st.checkbox("heat treatment known"),
        "ebsd_or_tkd": st.checkbox("EBSD/TKD available"),
        "DSC": st.checkbox("DSC available"),
        "XRD_lattice": st.checkbox("XRD/lattice available"),
        "stress_strain": st.checkbox("mechanical curves available"),
        "thermal_history": st.checkbox("thermal history available"),
    }
    gaps = assess_data_gaps("NiTi" if system.startswith("NiTi") else "steel", available)
    st.metric("Current twin confidence", f"{gaps.confidence_score:.2f}")
    st.write("Missing:", gaps.missing)
    st.write("Recommended next experiments:", gaps.recommended_next_experiments)

st.subheader("Open data/tool manifest")
manifest_path = ROOT / "data" / "open_data_manifest" / "open_data_sources.csv"
if manifest_path.exists():
    df = read_open_data_manifest(manifest_path)
    st.dataframe(df, use_container_width=True)
else:
    st.warning("Manifest not found.")
