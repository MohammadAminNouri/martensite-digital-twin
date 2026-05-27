# OpenMartensiteTwin v0.5

A Python-first, evidence-aware digital-twin framework for martensitic transformations in NiTi, steels, and future transformation materials.

This repository is **not a black-box AI model** and not yet an industrially validated twin. It is a modular scientific platform that separates:

- crystallographic calculation,
- experimental evidence,
- literature-derived assumptions,
- data gaps,
- validation status,
- and future user-supplied measurements.

## What v0.5 adds

v0.5 expands the app beyond EBSD/TKD orientation tables and adds a defensible characterization layer:

- XRD / synchrotron diffraction evidence workspace
- EDS/WDS chemistry evidence workspace
- SEM / optical morphology evidence workspace
- TEM / STEM / SAED / 4D-STEM evidence tracking
- article-derived missing-data map
- clearer distinction between literature values and same-sample evidence
- expanded defensibility gap register
- updated maturity scoring

The current scientific core still includes:

- NiTi B2→B19′ Cayron-style natural OR prototype
- steel fcc→bcc/bct KS/NW/Pitsch comparators
- variant generation and assignment
- synthetic EBSD-like datasets
- prototype parent clustering
- simple kinetics placeholders
- evidence/state-vector reporting

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
streamlit run app/streamlit_app.py
```

## Test

```bash
pytest -q
```

## Current honesty level

v0.5 is a much better research prototype, but it is not yet the final defensible digital twin. The remaining major gaps are:

1. native `.ctf`, `.ang`, `.h5` EBSD/TKD import;
2. graph-based parent reconstruction comparable to MTEX/ARPGE-style workflows;
3. OR refinement from measured maps;
4. validated XRD/Rietveld connector such as GSAS-II;
5. real DSC/dilatometry fitting;
6. quantitative EDS/TEM/STEM analysis connectors;
7. benchmark ingestion for open steel datasets;
8. raw NiTi EBSD/TKD/DSC/XRD/SEM/EDS/TEM datasets from the same sample;
9. CALPHAD/phase-field/crystal-plasticity coupling;
10. independent validation and uncertainty propagation.

## Deployment on Streamlit Cloud

Set the app path to:

```text
app/streamlit_app.py
```

Do not deploy older nested folders such as `martensite_twin_v01/app/streamlit_app.py`.
