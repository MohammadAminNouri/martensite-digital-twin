# OpenMartensiteTwin v0.3 User Guide

## What the main tables mean

### Orientation relationship matrix

A 3×3 rotation matrix connecting the child/martensite crystal frame to the parent crystal frame. It is a model assumption, not an EBSD measurement.

Used for: generating variants.

### Variant library

A catalogue of all theoretical variants generated from the OR and symmetry operations.

Important columns:

- `variant_id`: variant label.
- `parent_sym_index`: parent symmetry operation used.
- `child_sym_index`: child symmetry operation used.
- `r00...r22`: the 3×3 rotation matrix for the variant.

Used for: assigning EBSD/TKD points to variants.

### Pairwise misorientation matrix

A table comparing every theoretical variant against every other variant.

Rows and columns are variant IDs. Each cell is the misorientation angle in degrees. The diagonal is always zero.

Used for: interpreting variant boundaries, packets, blocks, and parent reconstruction.

### Variant population summary

Shows which variants dominate the measured or synthetic map.

- `count`: number of points assigned to a variant.
- `fraction`: count divided by total points.
- `mean_error_deg`: average mismatch to the theoretical variant.

Used for: variant-selection analysis and process comparison.

### Kinetics graph

For NiTi, the graph shows the simplified B19′ fraction during cooling and B2 fraction during heating. For steel, it shows martensite fraction during cooling.

This is not enough for final property prediction. It needs DSC/dilatometry calibration.

## Maturity levels

- L0: theoretical crystallography only.
- L1: EBSD/TKD interpretation prototype.
- L2: partly calibrated transformation twin.
- L3: process-aware research twin.
- L4: predictive engineering twin candidate.

v0.3 can demonstrate L0–L1 and structure the path to L2–L4. It does not yet replace validated experimental workflows.
