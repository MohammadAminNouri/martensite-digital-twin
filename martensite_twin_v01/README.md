# OpenMartensiteTwin v0.2

A Python-first, physics-based digital twin framework for martensitic phase transformations.

This version is no longer only a scaffold. It has a usable workflow:

```text
configure crystallographic model
→ generate/import EBSD-like orientation data
→ assign martensite variants
→ prototype parent-phase reconstruction
→ run first-order transformation kinetics
→ assess missing data and confidence
→ export CSV/Markdown/JSON reports
```

## Target systems

- **NiTi B2 → B19′** with a Cayron-inspired natural orientation-relationship prototype.
- **Steel fcc/austenite → bcc/bct martensite** using KS, NW, and Pitsch OR comparators.

## What v0.2 can do

- Generate unique orientation variants from parent/child symmetry.
- Show orientation relationship matrices and variant tables.
- Generate synthetic EBSD-like martensite maps for testing.
- Import CSV orientation data using either `r00..r22` matrix columns or Bunge Euler columns `phi1, Phi, phi2`.
- Assign measured/synthetic child orientations to theoretical variants.
- Produce variant population statistics and error metrics.
- Run a prototype greedy parent reconstruction.
- Plot assigned variant maps and parent-cluster maps in the Streamlit app.
- Run first-order kinetics models:
  - Koistinen–Marburger for steel;
  - simple DSC-calibrated linear hysteresis for NiTi.
- Assess data gaps and produce next-experiment recommendations.
- Export Markdown and JSON reports.
- Run CI tests on GitHub Actions.

## Important reliability warning

This is still a research prototype. It is not yet publication-grade EBSD reconstruction and not yet an industrial digital twin. Vendor EBSD conventions, phase symmetries, lattice parameters, parent/child orientation definitions, and sample-specific calibration data must be validated.

The goal of v0.2 is to provide a **working workflow**, not final scientific certainty.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[app]
streamlit run app/streamlit_app.py
```

Run examples and tests:

```bash
python examples/demo_niti_cayron.py
python examples/demo_steel_ks.py
python examples/demo_workflow_v02.py
pytest -q
```

## CSV input format

The app accepts either matrix columns:

```text
x,y,r00,r01,r02,r10,r11,r12,r20,r21,r22
```

or Bunge Euler columns:

```text
x,y,phi1,Phi,phi2
```

Optional columns include `phase`, `grain_id`, `point_id`, `ci`, and `iq`.

## Roadmap to v0.3/v1.0

### v0.3

- `.ctf`, `.ang`, and `.h5` import through kikuchipy/orix.
- Better graph-based parent reconstruction.
- OR refinement from measured EBSD data.
- Habit-plane trace overlay on maps.
- Ingestion scripts for open Zenodo steel EBSD/dilatometry datasets.

### v0.5

- pycalphad integration with real thermodynamic databases.
- DSC/dilatometry fitting.
- XRD phase-fraction/lattice-parameter ingestion.
- Better NiTi transformation-temperature model.

### v1.0

- OpenPhase/DAMASK connectors.
- LPBF thermal-history import and process metadata schema.
- Uncertainty propagation.
- Validation reports against known datasets.
- Web API via FastAPI.

## Repository identity

**OpenMartensiteTwin** is designed as an integration platform, not a black-box AI model. It combines crystallography, EBSD/TKD analysis, parent reconstruction, thermodynamics, kinetics, mechanics, and experimental feedback in one modular workflow.
