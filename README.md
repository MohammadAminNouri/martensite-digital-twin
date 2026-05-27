# OpenMartensiteTwin v0.4

A Python-first, evidence-aware scaffold for a comprehensive martensitic-transformation digital twin.

**Important:** v0.4 is not yet a fully validated industrial twin. It is a clearer, more defensible research platform that separates:

- crystallographic calculation,
- measured evidence,
- synthetic/test data,
- assumptions,
- missing validation,
- exportable reports.

## What v0.4 adds

- A guided Streamlit workflow with explanations for every control, table and graph.
- Evidence/state-vector tracking for sample, process route, data availability and assumptions.
- Maturity levels L0–L4 so the app cannot pretend to be more validated than it is.
- Explicit defensibility gap register: what is implemented, what is missing, and how to close it.
- Open-source data/tool manifest for MTEX, orix, kikuchipy, pycalphad, OpenPhase, DAMASK and open Zenodo steel datasets.
- Clear distinction between synthetic data, uploaded data and real calibration evidence.
- Better variant maps, angular-error maps, kinetics explanations and report exports.

## Install locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .[app]
streamlit run app/streamlit_app.py
```

## Streamlit Cloud

Use this as the main file path:

```text
app/streamlit_app.py
```

Do not deploy the old `martensite_twin_v01/app/streamlit_app.py` path.

## Current scientific status

| Layer | Status in v0.4 |
|---|---|
| Material/process state vector | partial, usable |
| Cayron-style NiTi crystallography | prototype |
| Steel KS/NW/Pitsch crystallography | prototype |
| CSV EBSD-like import | working |
| Synthetic EBSD map generation | working, for testing only |
| Variant assignment | working for known/assumed parent mode |
| Parent reconstruction | prototype only, not full MTEX/ARPGE graph method |
| Kinetics | educational KM/NiTi hysteresis placeholders |
| Thermodynamics | connector stub |
| Mechanics / phase-field | connector roadmap |
| Validation/uncertainty | gap-aware reporting, not statistical propagation yet |

## What must be added for a defensible “most comprehensive” twin

1. Real `.ctf`, `.ang`, `.h5` EBSD/TKD import through orix/kikuchipy.
2. Graph-based parent reconstruction with grain adjacency and OR probability functions.
3. OR refinement and convention validation against MTEX/ARPGE/AZtec-style examples.
4. Ingestion scripts for open steel datasets from Zenodo.
5. A raw NiTi EBSD/TKD/DSC/XRD/mechanical dataset for calibration.
6. pycalphad thermodynamics with legal database files.
7. DSC/dilatometry fitting and uncertainty bands.
8. OpenPhase/DAMASK/MOOSE coupling for phase-field and mechanics.
9. Independent validation suite and versioned sample database.

## Repository structure

```text
app/                    Streamlit app
martwin/                Python scientific core
martwin/digital_twin/   evidence, maturity and gap logic
examples/               command-line demos
data/                   open-data manifests and generated outputs
docs/                   roadmap and user documentation
tests/                  basic tests
scripts/                helper scripts for future data ingestion
```
