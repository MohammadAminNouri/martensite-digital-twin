# Roadmap

## v0.1 — delivered scaffold
- Python crystallography engine.
- NiTi Cayron natural OR prototype.
- Steel KS/NW/Pitsch OR approximations.
- Variant generation and known-parent variant identification.
- Greedy parent reconstruction prototype.
- Data-source manifest and gap analysis.

## v0.2 — serious EBSD analysis
- Robust .ctf/.ang/.h5 import through kikuchipy/orix.
- Grain segmentation.
- Graph-based parent reconstruction.
- OR refinement from measured child-child boundaries.
- Variant-pair/operator statistics.
- Habit-plane trace overlays.

## v0.3 — validation
- Download and parse Zenodo steel datasets.
- Reproduce MTEX martensite example results within tolerances.
- Build NiTi validation from Cayron public paper data and any available raw maps.
- Unit tests against synthetic ground truth.

## v0.4 — thermodynamics/kinetics
- pycalphad connector with user-supplied TDB.
- DSC/dilatometry fitting.
- Ms/Mf/As/Af and transformation-fraction uncertainty.

## v0.5 — mechanics/process
- LPBF thermal-history connector.
- OpenPhase and DAMASK export/import.
- Transformation-strain and stress-assisted variant-selection models.

## v1.0 — full platform
- FastAPI backend, PostgreSQL database, object storage.
- Streamlit/React dashboard.
- Reproducible workflows with DVC/MLflow.
- Confidence scoring and next-best-experiment recommendation.
