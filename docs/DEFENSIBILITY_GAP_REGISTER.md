# Defensibility gap register

OpenMartensiteTwin can become a serious digital twin only if each scientific claim has enough data and validation.

## Claim levels

- **Crystallographic possibility:** needs phase pair, OR and symmetry.
- **EBSD/TKD interpretation:** needs measured orientation maps and convention validation.
- **Parent reconstruction:** needs graph-based reconstruction, adjacency, OR refinement and benchmark data.
- **Kinetics prediction:** needs DSC/dilatometry/XRD phase fraction and fitted parameters.
- **Process prediction:** needs heat-treatment/LPBF thermal history and process metadata.
- **Property prediction:** needs stress-strain, residual stress, transformation strain and mechanics calibration.

## Highest priority gaps

1. Replace prototype parent reconstruction with graph-based reconstruction.
2. Add robust EBSD/TKD import for `.ctf`, `.ang`, `.h5`.
3. Ingest open steel benchmark datasets and compare results to known parent austenite.
4. Collect or locate raw NiTi EBSD/TKD/DSC/XRD/mechanical data.
5. Add uncertainty propagation and independent validation.

## Data honesty rule

If a value is not measured, the report must label it as assumed, literature-default, synthetic or missing.
