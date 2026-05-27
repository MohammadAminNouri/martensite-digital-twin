from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GapReport:
    missing: list[str]
    confidence_score: float
    recommended_next_experiments: list[str]


def assess_data_gaps(material_system: str, available: dict[str, bool]) -> GapReport:
    required = {
        "common": ["composition", "heat_treatment", "ebsd_or_tkd"],
        "niti": ["DSC", "XRD_lattice", "XRD_pattern", "EDS_maps", "SEM_images", "TEM_STEM", "stress_strain", "oxygen_carbon", "thermal_history"],
        "steel": ["cooling_curve", "prior_austenite_reference", "hardness", "retained_austenite_XRD", "SEM_images", "EDS_maps", "XRD_pattern"],
        "lpbf": ["laser_parameters", "scan_strategy", "powder_chemistry", "melt_pool_or_thermal_model", "porosity", "residual_stress"],
    }
    keys = list(required["common"])
    low = material_system.lower()
    if "niti" in low:
        keys += required["niti"]
    if "steel" in low:
        keys += required["steel"]
    if "lpbf" in low or "additive" in low:
        keys += required["lpbf"]

    missing = [k for k in keys if not available.get(k, False)]
    confidence = max(0.0, 1.0 - len(missing) / max(1, len(keys)))
    rec = []
    if "ebsd_or_tkd" in missing:
        rec.append("Collect EBSD/TKD orientation map for child martensite and, if possible, parent phase.")
    if "DSC" in missing and "niti" in low:
        rec.append("Run DSC to measure Ms/Mf/As/Af and hysteresis.")
    if "cooling_curve" in missing and "steel" in low:
        rec.append("Record quench/cooling curve or dilatometry trace.")
    if "XRD_lattice" in missing or "XRD_pattern" in missing:
        rec.append("Measure/upload XRD or synchrotron diffraction patterns and refine lattice parameters/phase fractions.")
    if "EDS_maps" in missing:
        rec.append("Add EDS/WDS chemistry data: Ni/Ti ratio, alloying additions, oxygen/carbon/impurity risk and segregation maps if possible.")
    if "SEM_images" in missing:
        rec.append("Add SEM/optical images for morphology, porosity, cracks, melt-pool tracks and correlative context with EBSD.")
    if "TEM_STEM" in missing:
        rec.append("Add TEM/STEM/SAED/4D-STEM evidence for nanoscale twins, precipitates, interfaces and local strain.")
    if "thermal_history" in missing or "melt_pool_or_thermal_model" in missing:
        rec.append("Build or measure thermal history; start with Rosenthal/FE thermal model if sensors are absent.")
    if not rec:
        rec.append("Run validation against independent EBSD/TKD/DSC/mechanical sample.")

    return GapReport(missing=missing, confidence_score=confidence, recommended_next_experiments=rec)
