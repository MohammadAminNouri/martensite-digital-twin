# Data strategy for the most complete martensitic-transformation digital twin

## Data we need for true-twin mode

### Common
- Exact alloy chemistry with units and uncertainty.
- Parent and child phase definitions.
- Processing route and full heat-treatment/thermal history.
- EBSD/TKD orientation map with phase map, CI/IQ, step size and vendor convention.
- XRD/synchrotron phase fractions and lattice parameters.
- DSC/dilatometry transformation curves.
- Mechanical curves and testing temperature.
- Micrographs and sample geometry.

### NiTi / LPBF
- Ni/Ti atomic ratio and O/C contamination.
- Powder chemistry and reuse history.
- Laser power, scan speed, hatch spacing, layer thickness, spot size, scan rotation, build orientation.
- Thermal history or calibrated thermal model.
- B2 grain texture, B19′ variants/twins, residual stress, porosity.
- Ms/Mf/As/Af from DSC after the exact post-processing route.

### Steel
- C/Mn/Cr/Ni/Mo/Si/etc. composition.
- Austenitization temperature/time, prior austenite grain size.
- Cooling curve/dilatometry and retained austenite fraction.
- Martensite/bainite/lath/packet/block statistics.
- Hardness/tensile/impact data after tempering if relevant.

## What to do when data is missing

The twin must never invent missing measurements. It should:

1. Mark the gap.
2. Fall back to literature/default values with a clear warning.
3. Propagate lower confidence.
4. Recommend the next experiment.
5. Store assumptions in the project database.

## Data fidelity levels

- **Level A: theoretical** — ORs, variants, habit candidates only.
- **Level B: experimental-analysis** — EBSD/TKD variant indexing and parent reconstruction.
- **Level C: calibrated digital twin** — process + thermodynamics + kinetics + mechanics + experimental calibration.
