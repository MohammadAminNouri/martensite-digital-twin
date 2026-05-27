from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import pandas as pd


FIDELITY_LEVELS = [
    {
        "level": "L0 — Theory demonstrator",
        "minimum_evidence": "Material system + orientation relationship + assumed lattice/OR parameters",
        "what_you_can_claim": "Possible variants, misorientation families, and qualitative crystallographic logic.",
        "what_you_cannot_claim": "Real sample prediction, validated parent reconstruction, properties, or process optimization.",
    },
    {
        "level": "L1 — EBSD/TKD interpretation tool",
        "minimum_evidence": "Measured child-orientation map and known/assumed OR convention.",
        "what_you_can_claim": "Variant assignment, fit error, simple parent-cluster hypothesis, data-quality warnings.",
        "what_you_cannot_claim": "True thermal/process response or validated phase fraction without DSC/XRD/process data.",
    },
    {
        "level": "L2 — Calibrated transformation analysis",
        "minimum_evidence": "EBSD/TKD + composition + heat treatment + DSC/dilatometry or XRD phase fraction.",
        "what_you_can_claim": "Measured transformation temperatures, calibrated kinetics, parent reconstruction with validation metrics.",
        "what_you_cannot_claim": "Reliable property prediction under new stress/process states without mechanics validation.",
    },
    {
        "level": "L3 — Process-aware research twin",
        "minimum_evidence": "L2 + thermal history/cooling curve or LPBF process record + mechanical response.",
        "what_you_can_claim": "Process-to-microstructure trends, uncertainty-aware next-experiment recommendations.",
        "what_you_cannot_claim": "Industrial qualification without independent validation and reproducibility studies.",
    },
    {
        "level": "L4 — Predictive engineering twin candidate",
        "minimum_evidence": "L3 + independent validation sets + uncertainty model + versioned databases + benchmark comparison.",
        "what_you_can_claim": "Defensible prediction within the calibrated domain, with explicit uncertainty bounds.",
        "what_you_cannot_claim": "Universal prediction outside alloy/process/data domain.",
    },
]


PARAMETER_GUIDE = [
    {
        "control": "Material system",
        "meaning": "Selects which parent→martensite phase transformation the model represents.",
        "changes_in_model": "Chooses phase symmetry, orientation relationship library, variant count, data requirements, and kinetics model.",
        "safe_default": "NiTi B2→B19′ for Cayron-focused work; Steel fcc→bcc/bct for prior-austenite reconstruction work.",
        "defensibility_warning": "Do not compare NiTi and steel results directly; their phases, kinetics, and data needs are different.",
    },
    {
        "control": "B19′ monoclinic beta angle",
        "meaning": "Monoclinic lattice angle of NiTi B19′ martensite, normally refined from XRD/diffraction for the specific composition.",
        "changes_in_model": "Updates the prototype Cayron-style B2→B19′ orientation matrix and therefore variant matrices and fit errors.",
        "safe_default": "96.8° as a placeholder prototype.",
        "defensibility_warning": "For real NiTi claims, replace this with measured sample-specific lattice parameters, not a default.",
    },
    {
        "control": "Steel orientation relationship",
        "meaning": "Selects the theoretical parent austenite→martensite relationship used for variant generation.",
        "changes_in_model": "Changes the variant library and the reference misorientation families used for assignment/reconstruction.",
        "safe_default": "KS is a common starting comparator; refine OR from EBSD when possible.",
        "defensibility_warning": "A theoretical KS/NW/Pitsch OR is not enough for publication-grade reconstruction; OR refinement is usually needed.",
    },
    {
        "control": "Variant fit tolerance",
        "meaning": "Maximum angular mismatch allowed between a measured orientation and a theoretical variant.",
        "changes_in_model": "Higher tolerance assigns more points but risks false matches; lower tolerance is stricter but may reject noisy/real data.",
        "safe_default": "3–7° depending on EBSD quality, phase, and OR convention.",
        "defensibility_warning": "Always report the tolerance; conclusions change when this value changes.",
    },
    {
        "control": "Parent reconstruction threshold",
        "meaning": "Angular threshold for grouping candidate parent orientations into prototype parent clusters.",
        "changes_in_model": "Higher threshold merges clusters; lower threshold splits clusters.",
        "safe_default": "5° for exploratory work only.",
        "defensibility_warning": "v0.4 uses prototype clustering, not full MTEX/ARPGE graph-based reconstruction.",
    },
    {
        "control": "Synthetic data: grid size",
        "meaning": "Controls the number of synthetic EBSD-like points generated for testing.",
        "changes_in_model": "More points make maps smoother but slower. This does not add real experimental evidence.",
        "safe_default": "40×40 or 60×60.",
        "defensibility_warning": "Synthetic data is for software testing and teaching only; it is not validation.",
    },
    {
        "control": "Synthetic data: orientation noise",
        "meaning": "Random angular perturbation applied to synthetic orientations.",
        "changes_in_model": "Higher noise increases angular fit error and lowers variant-confidence scores.",
        "safe_default": "0.5–1.5° for clean demonstration maps.",
        "defensibility_warning": "Real EBSD noise is spatially/phase dependent; this toy noise is not a microscope model.",
    },
    {
        "control": "Evidence checkboxes",
        "meaning": "Declare what real evidence exists for the sample: composition, heat treatment, EBSD/TKD, DSC, XRD, mechanical data, etc.",
        "changes_in_model": "Changes data-confidence score, maturity level, and recommended next experiments. It does not change raw physics calculations unless the relevant measured values are entered.",
        "safe_default": "Leave unchecked unless you actually have the evidence.",
        "defensibility_warning": "Checking a box without uploading/recording data only means ‘available in principle’; it is not verification.",
    },
]


TWIN_LAYER_MATRIX = [
    {
        "layer": "1. Material/process record",
        "implemented_in_v04": "Structured state vector, assumptions, evidence checkboxes, sample notes.",
        "needed_for_defensible_twin": "Versioned sample database with chemistry, heat treatment, process route, uncertainty and provenance.",
        "open_source_path": "PostgreSQL/SQLite + DVC/MLflow later; CSV/JSON now.",
        "status": "partial",
    },
    {
        "layer": "2. Crystallography / Cayron logic",
        "implemented_in_v04": "NiTi B2→B19′ prototype OR, steel KS/NW/Pitsch comparators, variant library, misorientation matrix.",
        "needed_for_defensible_twin": "Sample-specific lattice parameters, complete correspondence/distortion model, OR refinement and benchmark against Cayron/MTEX examples.",
        "open_source_path": "NumPy/orix; Cayron papers; MTEX benchmark comparison.",
        "status": "working prototype",
    },
    {
        "layer": "3. EBSD/TKD data",
        "implemented_in_v04": "CSV importer for rotation matrices or Bunge Euler angles; synthetic EBSD-like maps; data-source flags.",
        "needed_for_defensible_twin": ".ctf/.ang/.h5 import, grain adjacency graph, phase maps, quality filtering, real open datasets ingested.",
        "open_source_path": "orix, kikuchipy, MTEX example datasets, Zenodo steel datasets.",
        "status": "partial",
    },
    {
        "layer": "4. Parent reconstruction",
        "implemented_in_v04": "Prototype greedy parent-cluster map and known-parent variant assignment.",
        "needed_for_defensible_twin": "Graph-based variant reconstruction with OR probability functions, neighboring-grain topology, and retained-parent support.",
        "open_source_path": "MTEX parentGrainReconstructor logic as benchmark; ARPGE/GenOVa literature.",
        "status": "prototype only",
    },
    {
        "layer": "5. Thermodynamics",
        "implemented_in_v04": "Gap-aware placeholder/connector plan; no real CALPHAD database bundled.",
        "needed_for_defensible_twin": "pycalphad/Thermo-Calc connector with licensed/open databases, phase stability and driving force calculations.",
        "open_source_path": "pycalphad + open TDB files where legally available.",
        "status": "connector stub",
    },
    {
        "layer": "6. Kinetics",
        "implemented_in_v04": "Steel KM curve; NiTi linear DSC-style hysteresis placeholder with explicit warning.",
        "needed_for_defensible_twin": "Fitted DSC/dilatometry curves, cooling-rate dependence, stress-temperature coupling for NiTi.",
        "open_source_path": "SciPy fitting + public/own DSC datasets.",
        "status": "educational model",
    },
    {
        "layer": "7. Mechanics/properties",
        "implemented_in_v04": "Gap matrix and connector plan; no calibrated constitutive law yet.",
        "needed_for_defensible_twin": "Transformation strain, variant interaction work, residual stress, stress-strain calibration, crystal plasticity/phase-field coupling.",
        "open_source_path": "DAMASK/OpenPhase/MOOSE connectors; own mechanical tests.",
        "status": "missing",
    },
    {
        "layer": "8. Validation/uncertainty",
        "implemented_in_v04": "Maturity score, missing-data register, report export, assumption labels.",
        "needed_for_defensible_twin": "Benchmark suite against open steel datasets and NiTi lab/open datasets; uncertainty propagation and independent holdout samples.",
        "open_source_path": "Zenodo steel datasets, MTEX examples, future NiTi dataset collection.",
        "status": "partial",
    },
]


OPEN_SOURCE_DATASETS = [
    {
        "name": "In-situ Heating-Stage EBSD Validation of PAG Reconstruction",
        "system": "steel martensite/bainite",
        "what_it_contains": "High-temperature EBSD, martensitic/bainitic transformation frames, CTF files, thermocouple readouts, raw dilatometry.",
        "how_to_use_in_twin": "Benchmark parent reconstruction against measured high-temperature parent austenite and transformation sequence.",
        "url": "https://zenodo.org/records/8348372",
        "license_or_access": "Zenodo open dataset; check record license before redistribution.",
        "priority": "very high",
    },
    {
        "name": "Prior Austenite Grain Measurement supplementary material",
        "system": "steel",
        "what_it_contains": "EBSD files, SE images, grain-size lists, EDS, microhardness, Thermo-Calc predictions.",
        "how_to_use_in_twin": "Validate PAG size reconstruction and connect reconstruction to hardness/chemistry metadata.",
        "url": "https://zenodo.org/records/10469461",
        "license_or_access": "Zenodo open dataset; check record license before redistribution.",
        "priority": "very high",
    },
    {
        "name": "MTEX martensite parent grain reconstruction example",
        "system": "steel martensite",
        "what_it_contains": "Example dataset and reconstruction workflow inside MTEX documentation.",
        "how_to_use_in_twin": "Benchmark our reconstruction output and OR refinement logic against a respected reference workflow.",
        "url": "https://mtex-toolbox.github.io/MaParentGrainReconstruction.html",
        "license_or_access": "Open documentation/example; MTEX is open-source MATLAB toolbox.",
        "priority": "very high",
    },
    {
        "name": "orix-data",
        "system": "orientation mapping general",
        "what_it_contains": "Datasets used by the orix Python library for crystal orientation mapping data.",
        "how_to_use_in_twin": "Test orientation conventions, plotting, and import/export layers.",
        "url": "https://github.com/pyxem/orix-data",
        "license_or_access": "Check individual dataset licenses.",
        "priority": "medium",
    },
    {
        "name": "Cayron NiTi B2→B19′ EBSD/TKD paper",
        "system": "NiTi",
        "what_it_contains": "Natural OR, prior B2 reconstruction concept, TKD/EBSD interpretation, continuum orientations and habit-plane discussion.",
        "how_to_use_in_twin": "Scientific reference for NiTi OR/distortion/habit-plane modules. Raw EBSD/TKD files are still needed for calibration.",
        "url": "https://www.mdpi.com/2073-4352/10/7/562",
        "license_or_access": "Open-access article; raw data not bundled here.",
        "priority": "very high",
    },
    {
        "name": "ORTools4MTEX",
        "system": "martensitic/phase transformations",
        "what_it_contains": "OR discovery, OR analysis, habit-plane and plotting utilities for MTEX.",
        "how_to_use_in_twin": "Benchmark OR analysis and reporting outputs; use as conceptual comparator.",
        "url": "https://github.com/ORTools4MTEX/ORTools",
        "license_or_access": "Repository license shown on GitHub; verify before copying code.",
        "priority": "high",
    },
    {
        "name": "OpenPhase martensite/bainite framework paper",
        "system": "steel martensite/bainite",
        "what_it_contains": "OpenPhase-based phase-field framework combining phase evolution, chemical diffusion, temperature evolution, finite-strain elastoplasticity.",
        "how_to_use_in_twin": "Target physics connector for spatial phase-field simulation after EBSD and kinetics layers mature.",
        "url": "https://www.sciencedirect.com/science/article/pii/S0927025624002544",
        "license_or_access": "Open-access article; software/data access must be checked separately.",
        "priority": "high",
    },
    {
        "name": "DAMASK",
        "system": "mechanics/crystal plasticity",
        "what_it_contains": "Unified multiphysics crystal-plasticity simulation package.",
        "how_to_use_in_twin": "Connector for stress/strain/texture/property simulations once microstructure state is calibrated.",
        "url": "https://damask-multiphysics.org/",
        "license_or_access": "Open-source/project access; check license for redistribution.",
        "priority": "high",
    },
    {
        "name": "pycalphad",
        "system": "thermodynamics",
        "what_it_contains": "Python CALPHAD equilibrium/thermodynamic calculation library.",
        "how_to_use_in_twin": "Thermodynamic engine when legal thermodynamic database files are supplied.",
        "url": "https://pycalphad.org/",
        "license_or_access": "Open-source code; databases may be licensed separately.",
        "priority": "high",
    },
]


DEFENSIBILITY_REQUIREMENTS = [
    {
        "claim": "Variant assignment",
        "minimum_required": "Measured EBSD/TKD orientation map + explicit OR + tolerance + phase/symmetry convention.",
        "v04_status": "Works for CSV/synthetic maps; real EBSD vendor formats not fully supported yet.",
        "gap_closure": "Add orix/kikuchipy import, quality filters, phase labels, convention tests.",
    },
    {
        "claim": "Parent grain reconstruction",
        "minimum_required": "Child grain adjacency, OR probability graph, parent-orientation clustering, benchmark against known parent map.",
        "v04_status": "Prototype greedy clustering only.",
        "gap_closure": "Implement graph-based MTEX/ARPGE-style reconstruction and benchmark on Zenodo in-situ EBSD steel dataset.",
    },
    {
        "claim": "NiTi transformation temperatures",
        "minimum_required": "DSC or in-situ measurement for Ms/Mf/As/Af and composition/precipitation state.",
        "v04_status": "User-entered linear hysteresis placeholder.",
        "gap_closure": "Add DSC upload/fitting and calibrated NiTi models; collect/open raw NiTi data.",
    },
    {
        "claim": "Steel martensite fraction",
        "minimum_required": "Cooling curve/dilatometry, composition, Ms calibration; retained-austenite measurement for validation.",
        "v04_status": "Koistinen–Marburger educational model.",
        "gap_closure": "Fit KM parameters to dilatometry/XRD and composition-dependent Ms models.",
    },
    {
        "claim": "LPBF process prediction",
        "minimum_required": "Powder chemistry, laser/scan strategy, thermal model or sensor data, porosity/residual stress, post-processing.",
        "v04_status": "Evidence tracking only.",
        "gap_closure": "Add thermal-history module, AM metadata schema, and calibrated LPBF NiTi/steel datasets.",
    },
    {
        "claim": "Mechanical/property prediction",
        "minimum_required": "Stress-strain data, transformation strain/variant-selection law, residual stress or crystal-plasticity/phase-field calibration.",
        "v04_status": "Not implemented beyond data-gap tracking.",
        "gap_closure": "Connect to DAMASK/OpenPhase/MOOSE and validate against experiments.",
    },
]


@dataclass
class TwinEvidence:
    composition: bool = False
    heat_treatment: bool = False
    ebsd_or_tkd: bool = False
    dsc: bool = False
    xrd_lattice: bool = False
    stress_strain: bool = False
    oxygen_carbon: bool = False
    thermal_history: bool = False
    cooling_curve: bool = False
    prior_austenite_reference: bool = False
    hardness: bool = False
    retained_austenite_xrd: bool = False
    laser_parameters: bool = False
    scan_strategy: bool = False
    powder_chemistry: bool = False
    melt_pool_or_thermal_model: bool = False
    porosity: bool = False
    residual_stress: bool = False

    def as_gap_dict(self) -> dict[str, bool]:
        d = asdict(self)
        return {
            "composition": d["composition"],
            "heat_treatment": d["heat_treatment"],
            "ebsd_or_tkd": d["ebsd_or_tkd"],
            "DSC": d["dsc"],
            "XRD_lattice": d["xrd_lattice"],
            "stress_strain": d["stress_strain"],
            "oxygen_carbon": d["oxygen_carbon"],
            "thermal_history": d["thermal_history"],
            "cooling_curve": d["cooling_curve"],
            "prior_austenite_reference": d["prior_austenite_reference"],
            "hardness": d["hardness"],
            "retained_austenite_XRD": d["retained_austenite_xrd"],
            "laser_parameters": d["laser_parameters"],
            "scan_strategy": d["scan_strategy"],
            "powder_chemistry": d["powder_chemistry"],
            "melt_pool_or_thermal_model": d["melt_pool_or_thermal_model"],
            "porosity": d["porosity"],
            "residual_stress": d["residual_stress"],
        }


def dataframe(rows: Iterable[dict]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def maturity_level(confidence_score: float, has_dataset: bool, has_calibration: bool, has_process: bool, has_validation: bool) -> tuple[str, str]:
    if not has_dataset:
        return "L0 — Theory demonstrator", "No measured orientation map is loaded; results are crystallographic possibilities, not a real twin for a sample."
    if not has_calibration:
        return "L1 — EBSD/TKD interpretation tool", "Orientation data exist, but transformation/thermal/property calibration is missing."
    if confidence_score < 0.65 or not has_process:
        return "L2 — Calibrated transformation analysis", "Some calibration exists, but process-history or property links remain incomplete."
    if not has_validation:
        return "L3 — Process-aware research twin", "Process and calibration evidence exist; independent validation is still required."
    return "L4 — Predictive engineering twin candidate", "Evidence is broad enough for prediction inside the calibrated domain, with uncertainty reporting."


def evidence_table(evidence: TwinEvidence) -> pd.DataFrame:
    rows = []
    mapping = asdict(evidence)
    categories = {
        "composition": "material chemistry",
        "oxygen_carbon": "material chemistry",
        "powder_chemistry": "material chemistry / AM",
        "heat_treatment": "process history",
        "thermal_history": "process history",
        "cooling_curve": "process history",
        "laser_parameters": "LPBF process",
        "scan_strategy": "LPBF process",
        "melt_pool_or_thermal_model": "LPBF process",
        "ebsd_or_tkd": "microstructure measurement",
        "xrd_lattice": "phase/lattice measurement",
        "dsc": "thermal transformation measurement",
        "stress_strain": "mechanical validation",
        "hardness": "mechanical validation",
        "residual_stress": "mechanical validation",
        "porosity": "AM defect state",
        "prior_austenite_reference": "validation reference",
        "retained_austenite_xrd": "phase validation",
    }
    effects = {
        "composition": "Enables composition-aware thermodynamics and Ms/Ms-like estimates.",
        "oxygen_carbon": "Critical for NiTi and AM because impurities shift transformation temperatures and brittleness.",
        "powder_chemistry": "Needed for LPBF because powder differs from nominal alloy and may pick up oxygen.",
        "heat_treatment": "Defines precipitates, residual stress relief, parent grain state and transformation path.",
        "thermal_history": "Needed for process→microstructure prediction, especially AM.",
        "cooling_curve": "Needed for steel martensite/bainite kinetics and dilatometry comparison.",
        "laser_parameters": "Needed for LPBF thermal input and melt-pool prediction.",
        "scan_strategy": "Controls texture, melt-pool overlap, residual stress and anisotropy in LPBF.",
        "melt_pool_or_thermal_model": "Turns AM settings into actual local thermal history.",
        "ebsd_or_tkd": "Grounds the crystallographic twin in measured orientations.",
        "xrd_lattice": "Provides sample-specific lattice parameters and phase fractions.",
        "dsc": "Calibrates Ms/Mf/As/Af and hysteresis for NiTi or transformation temperatures generally.",
        "stress_strain": "Needed for shape-memory/superelasticity or strength predictions.",
        "hardness": "Cheap validation signal for steel microstructure/property trends.",
        "residual_stress": "Important for variant selection and transformation under constraint.",
        "porosity": "AM defects affect mechanical response and EBSD interpretation.",
        "prior_austenite_reference": "Gold-standard validation for parent reconstruction algorithms.",
        "retained_austenite_xrd": "Validates martensite/parent phase fraction and retained austenite.",
    }
    for k, v in mapping.items():
        rows.append({
            "evidence_item": k,
            "available": bool(v),
            "category": categories.get(k, "other"),
            "why_it_matters": effects.get(k, "Used to increase confidence and reduce assumptions."),
        })
    return pd.DataFrame(rows)
