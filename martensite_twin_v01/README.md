# Martensite Twin v0.1

A Python-first scaffold for a comprehensive martensitic-transformation digital twin.

This repository is designed as a **physics-based integration platform**, not a black-box AI model. It starts with a working crystallographic/EBSD core and provides extension points for CALPHAD, phase-field, crystal plasticity, LPBF thermal histories, and experimental calibration.

## What works now

- Rotation/orientation math: Euler angles, quaternions, misorientation angles.
- Symmetry operators: cubic 24 proper rotations and a minimal monoclinic setting.
- Orientation relationship construction from parallel plane/direction pairs.
- Steel fcc → bcc/bct orientation relationships: KS, NW, Pitsch approximations.
- NiTi B2 → B19′ Cayron-style natural OR prototype.
- Variant generation from symmetry.
- Synthetic EBSD-like CSV generation and import.
- Variant identification for a known parent orientation.
- Simple parent-orientation candidate generation.
- Koistinen–Marburger martensite fraction model for steels.
- Transformation-temperature record and gap-aware confidence scoring.
- Data manifest of open/public datasets and tools to collect/validate against.
- Optional connector stubs for MTEX, ORTools, pycalphad, OpenPhase, DAMASK.

## What is deliberately not claimed yet

This is **not yet a fully validated industrial digital twin**. The hard part is experimental calibration: exact alloy chemistry, EBSD/TKD maps, DSC, XRD, stress–strain curves, residual stress, LPBF thermal history, and heat-treatment records.

The code marks missing data and lowers confidence instead of inventing numbers.

## Quick start

```bash
cd martensite_twin_v01
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
python examples/demo_niti_cayron.py
python examples/demo_steel_ks.py
```

Outputs are written to `data/processed/`.

## Repository structure

```text
martwin/
  core/                 matrix/orientation math
  materials/            material-system definitions and data schema
  crystallography/      ORs, variants, habit-plane placeholders, reconstruction
  io/                   EBSD CSV import/export and data manifests
  kinetics/             KM and transformation curve utilities
  thermodynamics/       pycalphad/TC-Python hooks and gap-aware stubs
  mechanics/            transformation-strain and variant-selection hooks
  calibration/          confidence and gap analysis
  visualization/        simple plotting utilities
  connectors/           MTEX, OpenPhase, DAMASK, pycalphad connector stubs
examples/               runnable examples
open_data_manifest/     public/open datasets and tool URLs
```

## Scientific identity

The intended final platform is:

```text
composition + process + heat treatment + EBSD/TKD + DSC/XRD/mechanics
→ CALPHAD/driving force
→ martensite kinetics
→ Cayron/OR crystallography
→ variant and parent reconstruction
→ habit-plane/strain/mechanics
→ validation and uncertainty
→ report/dashboard/API
```

## License

Starter scaffold: MIT. External tools/datasets have their own licenses; check each manifest row before reuse.
