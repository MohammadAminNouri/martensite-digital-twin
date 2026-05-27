# OpenMartensiteTwin v0.3

A Python-first, physics-based framework for building a **martensitic-transformation digital twin**.

v0.3 is focused on making the workflow understandable and usable: every table, graph, metric, and data field now has an explanation of what it is, why it matters, how to read it, and where it is used downstream.

## What the app does

```text
material/process record
→ data requirements and evidence ledger
→ crystallographic OR and variant library
→ EBSD/TKD CSV or synthetic map
→ variant assignment and fit-error analysis
→ prototype parent reconstruction
→ first-order transformation kinetics
→ reliability / data-gap scoring
→ Markdown + JSON report export
```

## Target systems

- **NiTi B2 → B19′** with a Cayron-inspired natural orientation-relationship prototype.
- **Steel fcc/austenite → bcc/bct martensite** with KS, NW, and Pitsch comparator ORs.

## What is improved in v0.3

- Guided workflow tab explaining the full digital-twin pipeline.
- Sample/process/evidence ledger.
- Data requirement table for NiTi, steel, and LPBF/AM cases.
- Explanation panels for:
  - orientation-relationship matrix;
  - variant library;
  - pairwise variant misorientation matrix;
  - EBSD/TKD dataset preview;
  - variant population summary;
  - point-level assignments;
  - kinetics graphs;
  - open data/tool manifest.
- One-click complete synthetic demo.
- Maturity level from L0 to L4.
- Reliability and missing-data logic that makes clear when the app is only a prototype rather than a calibrated twin.
- Safer Markdown report generation.

## Important reliability warning

This is still a research prototype, not a validated industrial twin. It becomes a real calibrated digital twin only when supplied with measured data such as:

- exact composition;
- heat treatment or process route;
- EBSD/TKD orientation maps;
- DSC or dilatometry;
- XRD lattice parameters and phase fractions;
- mechanical curves;
- LPBF thermal history, porosity, residual stress, and powder chemistry when relevant.

The app is intentionally honest about missing data. It does not fill scientific gaps with fake certainty.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[app]
streamlit run app/streamlit_app.py
```

Run tests:

```bash
pytest -q
```

## CSV input format

The app accepts CSV orientation data with either rotation-matrix columns:

```text
x,y,r00,r01,r02,r10,r11,r12,r20,r21,r22
```

or Bunge Euler columns:

```text
x,y,phi1,Phi,phi2
```

Optional columns include:

```text
point_id, phase, grain_id, ci, iq
```

Synthetic demo data also includes `true_variant_id` and `parent_region_id`, which are ground-truth validation labels and do not exist in normal experimental EBSD/TKD data.

## Current limitations

- Vendor `.ctf`, `.ang`, and `.h5` import is planned but not yet active.
- Parent reconstruction is still a prototype, not graph-based publication-grade PGR.
- OR fitting/refinement from EBSD data is not finished.
- Habit-plane trace overlay is planned.
- Thermodynamics and mechanics are still connectors/stubs, not full pycalphad/OpenPhase/DAMASK coupling.

## Roadmap

### v0.4

- `.ctf`, `.ang`, and `.h5` import through kikuchipy/orix.
- Graph-based parent reconstruction.
- OR fitting/refinement from measured EBSD/TKD data.
- Habit-plane trace overlays.
- Error maps and confidence maps.

### v0.5

- pycalphad integration with real thermodynamic databases.
- DSC/dilatometry fitting.
- XRD phase-fraction/lattice-parameter ingestion.
- Open dataset ingestion scripts.

### v1.0

- OpenPhase/DAMASK connectors.
- LPBF thermal-history import.
- Uncertainty propagation.
- Validated benchmark reports.
- FastAPI backend for deployment.

## Repository identity

**OpenMartensiteTwin** is an integration platform. Its purpose is to connect crystallography, EBSD/TKD analysis, thermodynamics, kinetics, mechanics, process history, and experimental feedback in one honest workflow.
