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
        "defensibility_warning": "v0.5 uses prototype clustering, not full MTEX/ARPGE graph-based reconstruction.",
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
        "implemented_in_v05": "Structured state vector, assumptions, evidence checkboxes, sample notes.",
        "needed_for_defensible_twin": "Versioned sample database with chemistry, heat treatment, process route, uncertainty and provenance.",
        "open_source_path": "PostgreSQL/SQLite + DVC/MLflow later; CSV/JSON now.",
        "status": "partial",
    },
    {
        "layer": "2. Crystallography / Cayron logic",
        "implemented_in_v05": "NiTi B2→B19′ prototype OR, steel KS/NW/Pitsch comparators, variant library, misorientation matrix.",
        "needed_for_defensible_twin": "Sample-specific lattice parameters, complete correspondence/distortion model, OR refinement and benchmark against Cayron/MTEX examples.",
        "open_source_path": "NumPy/orix; Cayron papers; MTEX benchmark comparison.",
        "status": "working prototype",
    },
    {
        "layer": "3. EBSD/TKD data",
        "implemented_in_v05": "CSV importer for rotation matrices or Bunge Euler angles; synthetic EBSD-like maps; data-source flags.",
        "needed_for_defensible_twin": ".ctf/.ang/.h5 import, grain adjacency graph, phase maps, quality filtering, real open datasets ingested.",
        "open_source_path": "orix, kikuchipy, MTEX example datasets, Zenodo steel datasets.",
        "status": "partial",
    },
    {
        "layer": "4. Parent reconstruction",
        "implemented_in_v05": "Prototype greedy parent-cluster map and known-parent variant assignment.",
        "needed_for_defensible_twin": "Graph-based variant reconstruction with OR probability functions, neighboring-grain topology, and retained-parent support.",
        "open_source_path": "MTEX parentGrainReconstructor logic as benchmark; ARPGE/GenOVa literature.",
        "status": "prototype only",
    },
    {
        "layer": "5. Thermodynamics",
        "implemented_in_v05": "Gap-aware placeholder/connector plan; no real CALPHAD database bundled.",
        "needed_for_defensible_twin": "pycalphad/Thermo-Calc connector with licensed/open databases, phase stability and driving force calculations.",
        "open_source_path": "pycalphad + open TDB files where legally available.",
        "status": "connector stub",
    },
    {
        "layer": "6. Kinetics",
        "implemented_in_v05": "Steel KM curve; NiTi linear DSC-style hysteresis placeholder with explicit warning.",
        "needed_for_defensible_twin": "Fitted DSC/dilatometry curves, cooling-rate dependence, stress-temperature coupling for NiTi.",
        "open_source_path": "SciPy fitting + public/own DSC datasets.",
        "status": "educational model",
    },
    {
        "layer": "7. Mechanics/properties",
        "implemented_in_v05": "Gap matrix and connector plan; no calibrated constitutive law yet.",
        "needed_for_defensible_twin": "Transformation strain, variant interaction work, residual stress, stress-strain calibration, crystal plasticity/phase-field coupling.",
        "open_source_path": "DAMASK/OpenPhase/MOOSE connectors; own mechanical tests.",
        "status": "missing",
    },
    {
        "layer": "8. Validation/uncertainty",
        "implemented_in_v05": "Maturity score, missing-data register, report export, assumption labels.",
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
    {
        "name": "GSAS-II",
        "system": "XRD/neutron diffraction",
        "what_it_contains": "Open-source Python diffraction analysis and Rietveld refinement framework.",
        "how_to_use_in_twin": "Connector target for refined lattice parameters and phase fractions from XRD/synchrotron data.",
        "url": "https://advancedphotonsource.github.io/GSAS-II-tutorials/",
        "license_or_access": "Open-source; verify license for redistribution.",
        "priority": "high",
    },
    {
        "name": "HyperSpy / eXSpy",
        "system": "EDS/EELS/TEM/SEM spectra",
        "what_it_contains": "Open-source multidimensional data analysis with EDS/EELS extensions.",
        "how_to_use_in_twin": "Connector target for EDS/WDS chemistry and electron microscopy spectral data.",
        "url": "https://hyperspy.org/",
        "license_or_access": "Open-source; check package licenses.",
        "priority": "high",
    },
    {
        "name": "py4DSTEM",
        "system": "TEM/STEM/4D-STEM",
        "what_it_contains": "Open-source Python tools for 4D-STEM processing and analysis.",
        "how_to_use_in_twin": "Future connector for nanoscale orientation/strain/diffraction validation in NiTi martensite.",
        "url": "https://py4dstem.readthedocs.io/en/stable/",
        "license_or_access": "Open-source; check license.",
        "priority": "medium",
    },
    {
        "name": "pyxem",
        "system": "electron diffraction/TEM",
        "what_it_contains": "HyperSpy extension for multidimensional diffraction data and phase/orientation mapping.",
        "how_to_use_in_twin": "Future connector for TEM/electron diffraction evidence and nanoscale phase/orientation maps.",
        "url": "https://www.pyxem.org/",
        "license_or_access": "Open-source; check license.",
        "priority": "medium",
    },
]


DEFENSIBILITY_REQUIREMENTS = [
    {
        "claim": "Variant assignment",
        "minimum_required": "Measured EBSD/TKD orientation map + explicit OR + tolerance + phase/symmetry convention.",
        "v05_status": "Works for CSV/synthetic maps; real EBSD vendor formats not fully supported yet.",
        "gap_closure": "Add orix/kikuchipy import, quality filters, phase labels, convention tests.",
    },
    {
        "claim": "Parent grain reconstruction",
        "minimum_required": "Child grain adjacency, OR probability graph, parent-orientation clustering, benchmark against known parent map.",
        "v05_status": "Prototype greedy clustering only.",
        "gap_closure": "Implement graph-based MTEX/ARPGE-style reconstruction and benchmark on Zenodo in-situ EBSD steel dataset.",
    },
    {
        "claim": "NiTi transformation temperatures",
        "minimum_required": "DSC or in-situ measurement for Ms/Mf/As/Af and composition/precipitation state.",
        "v05_status": "User-entered linear hysteresis placeholder.",
        "gap_closure": "Add DSC upload/fitting and calibrated NiTi models; collect/open raw NiTi data.",
    },
    {
        "claim": "Steel martensite fraction",
        "minimum_required": "Cooling curve/dilatometry, composition, Ms calibration; retained-austenite measurement for validation.",
        "v05_status": "Koistinen–Marburger educational model.",
        "gap_closure": "Fit KM parameters to dilatometry/XRD and composition-dependent Ms models.",
    },
    {
        "claim": "LPBF process prediction",
        "minimum_required": "Powder chemistry, laser/scan strategy, thermal model or sensor data, porosity/residual stress, post-processing.",
        "v05_status": "Evidence tracking only.",
        "gap_closure": "Add thermal-history module, AM metadata schema, and calibrated LPBF NiTi/steel datasets.",
    },
    {
        "claim": "Mechanical/property prediction",
        "minimum_required": "Stress-strain data, transformation strain/variant-selection law, residual stress or crystal-plasticity/phase-field calibration.",
        "v05_status": "Not implemented beyond data-gap tracking.",
        "gap_closure": "Connect to DAMASK/OpenPhase/MOOSE and validate against experiments.",
    },
]


CHARACTERIZATION_MODULES = [
    {
        "module": "EBSD/TKD orientation mapping",
        "raw_or_minimum_input": ".ctf, .ang, .h5, or CSV containing orientation matrices/Euler angles, phase labels, x-y coordinates, and quality metrics",
        "what_it_extracts": "child/martensite orientations, grains, boundaries, variant IDs, angular error, parent reconstruction candidates",
        "used_by_twin": "crystallography, variant assignment, parent reconstruction, OR refinement, habit-trace validation",
        "open_source_route": "orix + kikuchipy + MTEX benchmark workflows",
        "current_v05_status": "CSV workflow working; native vendor import planned",
    },
    {
        "module": "XRD / synchrotron diffraction",
        "raw_or_minimum_input": "2θ-intensity or q-intensity CSV; optional CIF/phase library; temperature/time/load metadata",
        "what_it_extracts": "B2/B19′/R/retained austenite phase fractions, lattice parameters, peak shifts, microstrain, texture clues",
        "used_by_twin": "sample-specific lattice parameters, phase-fraction validation, DSC/kinetics calibration, retained-austenite validation",
        "open_source_route": "GSAS-II, pyFAI/pydidas, SciPy peak fitting, XERUS-style phase identification",
        "current_v05_status": "CSV preview + peak table; full Rietveld connector planned",
    },
    {
        "module": "DSC / dilatometry",
        "raw_or_minimum_input": "temperature vs heat-flow or length-change curve with heating/cooling rate and sample mass",
        "what_it_extracts": "Ms, Mf, As, Af, hysteresis width, transformation enthalpy, steel transformation strain/temperature events",
        "used_by_twin": "kinetics calibration, maturity score, process-to-phase-fraction comparison",
        "open_source_route": "NumPy/SciPy peak and baseline fitting; Zenodo steel dilatometry datasets",
        "current_v05_status": "placeholder kinetics + optional curve upload planned",
    },
    {
        "module": "EDS / WDS chemistry",
        "raw_or_minimum_input": "element wt.% / at.% table, point/line/map data, detector/correction metadata",
        "what_it_extracts": "Ni/Ti ratio, alloying additions, oxygen/carbon/impurity risk, segregation or composition gradients",
        "used_by_twin": "composition-aware thermodynamics, NiTi transformation-temperature risk, LPBF powder/as-built chemistry tracking",
        "open_source_route": "HyperSpy/eXSpy/RosettaSciIO for spectra; CSV table support now",
        "current_v05_status": "CSV composition table + Ni/Ti ratio check; spectral quantification planned",
    },
    {
        "module": "SEM / optical microscopy",
        "raw_or_minimum_input": "micrographs with scale, etching/preparation notes, optional masks/grain-size labels",
        "what_it_extracts": "morphology, porosity, cracks, melt-pool tracks, grain/packet size clues, quality control context",
        "used_by_twin": "microstructure sanity check, AM defect-state evidence, validation against reconstructed parent/martensite maps",
        "open_source_route": "scikit-image/OpenCV/ImageJ/Fiji; Zenodo steel SE-image datasets",
        "current_v05_status": "image upload/display + metadata notes; automated segmentation planned",
    },
    {
        "module": "TEM / STEM / 4D-STEM / SAED",
        "raw_or_minimum_input": "TEM images, SAED/diffraction patterns, 4D-STEM datasets, foil orientation and calibration metadata",
        "what_it_extracts": "nano-twins, precipitates, habit/interface planes, local orientation/strain, diffuse scattering, dislocation/defect structures",
        "used_by_twin": "NiTi B19′ variant/twin validation, compatibility/habit-plane validation, precipitate/residual-strain evidence",
        "open_source_route": "py4DSTEM, pyxem, HyperSpy/eXSpy",
        "current_v05_status": "metadata/evidence tracking; full analysis connector planned",
    },
    {
        "module": "Mechanical tests",
        "raw_or_minimum_input": "stress-strain, cyclic loading, superelastic/shape-memory recovery, hardness, test temperature",
        "what_it_extracts": "transformation stress, hysteresis, residual strain, modulus/strength/hardness, fatigue/cyclic degradation",
        "used_by_twin": "property validation, stress-assisted transformation, variant-selection mechanics, DAMASK/OpenPhase calibration",
        "open_source_route": "pandas/SciPy fitting; DAMASK/OpenPhase/MOOSE connector later",
        "current_v05_status": "evidence tracking; advanced mechanics planned",
    },
]

ARTICLE_EVIDENCE_MAP = [
    {
        "missing_or_weak_item": "Graph-based parent reconstruction",
        "what_to_extract_from_sources": "variant graph logic, grain adjacency, parent candidate probabilities, OR refinement workflow, benchmark metrics",
        "source_now": "MTEX parent reconstruction docs; Hielscher et al. variant graph paper; ARPGE/GenOVa literature",
        "data_can_fill_now": "Use MTEX examples and Zenodo in-situ steel EBSD/CTF data to benchmark against known parent austenite.",
        "future_our_data": "Upload paired parent/child EBSD maps from our own steels/NiTi to validate reconstruction on our alloys.",
        "status": "high-priority code gap",
    },
    {
        "missing_or_weak_item": "Native EBSD/TKD import",
        "what_to_extract_from_sources": "orientation conventions, symmetry metadata, grain IDs, CI/MAD/BC/IQ quality fields, phase labels",
        "source_now": "orix/kikuchipy docs, MTEX import workflows, Zenodo CTF steel datasets",
        "data_can_fill_now": "Import CSV exports now; use CTF from Zenodo after adding native parser/connector.",
        "future_our_data": "Require every uploaded EBSD/TKD file to include software, step size, phase definitions and map coordinate conventions.",
        "status": "high-priority IO gap",
    },
    {
        "missing_or_weak_item": "NiTi raw EBSD/TKD validation",
        "what_to_extract_from_sources": "Cayron natural OR, coexisting ORs, habit-plane trace logic, B2 parent reconstruction evidence",
        "source_now": "Cayron 2020 NiTi EBSD/TKD open article and related correspondence-theory paper",
        "data_can_fill_now": "Use article values as reference; no full raw NiTi EBSD/TKD benchmark found bundled in open form yet.",
        "future_our_data": "Create our own open NiTi benchmark: B19′ EBSD/TKD + DSC + XRD + SEM/TEM + chemistry from the same sample.",
        "status": "main data gap",
    },
    {
        "missing_or_weak_item": "XRD lattice/phase module",
        "what_to_extract_from_sources": "B2/B19′/R phase peaks, lattice parameters, phase fractions, retained austenite, temperature/load dependence",
        "source_now": "GSAS-II open-source refinement; NiTi XRD/DSC articles; Zenodo steel retained-austenite/EDS/hardness supplementary data",
        "data_can_fill_now": "Use 2θ-intensity CSV to preview peaks; use GSAS-II externally for defensible Rietveld refinement until connector is built.",
        "future_our_data": "Upload raw XRD/synchrotron patterns and refined phase/lattice reports for each sample.",
        "status": "new v0.5 module, full refinement pending",
    },
    {
        "missing_or_weak_item": "EDS chemistry and impurity module",
        "what_to_extract_from_sources": "Ni/Ti ratio, alloying additions, oxygen/carbon risk, segregation, powder vs as-built chemistry",
        "source_now": "HyperSpy/eXSpy EDS workflows; Zenodo PAG dataset includes EDS measurements",
        "data_can_fill_now": "Use simple CSV element tables now; spectral quantification remains external.",
        "future_our_data": "Upload EDS/WDS point, line and map data; record standards/corrections and uncertainty.",
        "status": "new v0.5 module, quantitative spectra pending",
    },
    {
        "missing_or_weak_item": "SEM morphology/defect evidence",
        "what_to_extract_from_sources": "grain/martensite morphology, packet/lath/plate morphology, porosity/cracks, melt-pool tracks, etched PAG comparison",
        "source_now": "Zenodo prior-austenite supplementary dataset contains SE images and grain-size lists; open image tools can segment images",
        "data_can_fill_now": "Upload images and notes now; automatic segmentation later.",
        "future_our_data": "Collect SEM/optical maps from same regions as EBSD/TKD and use correlative registration.",
        "status": "new v0.5 evidence layer",
    },
    {
        "missing_or_weak_item": "TEM/STEM nanoscale validation",
        "what_to_extract_from_sources": "NiTi nanoscale twins, precipitates, habit interfaces, diffuse scattering, local strain and orientation",
        "source_now": "Cayron TKD/TEM context, 4D-STEM NiTi transformation literature, py4DSTEM/pyxem/HyperSpy tools",
        "data_can_fill_now": "Track TEM/SAED/4D-STEM as evidence; full analysis requires raw calibrated datasets.",
        "future_our_data": "Upload raw/calibrated TEM/4D-STEM/SAED with sample orientation and transformation state.",
        "status": "new v0.5 evidence layer, advanced connector pending",
    },
    {
        "missing_or_weak_item": "Thermodynamics and mechanics coupling",
        "what_to_extract_from_sources": "CALPHAD phase stability/driving force, stress-assisted transformation laws, phase-field/crystal plasticity workflows",
        "source_now": "pycalphad, OpenPhase, DAMASK, MOOSE literature/tools",
        "data_can_fill_now": "Use connectors as stubs; thermodynamic databases and constitutive parameters are not bundled by default.",
        "future_our_data": "Fit models using our DSC/dilatometry/XRD/stress-strain datasets and document valid prediction domain.",
        "status": "major physics gap",
    },
]


@dataclass
class TwinEvidence:
    composition: bool = False
    heat_treatment: bool = False
    ebsd_or_tkd: bool = False
    dsc: bool = False
    xrd_lattice: bool = False
    xrd_pattern: bool = False
    sem_images: bool = False
    eds_maps: bool = False
    tem_stem: bool = False
    tem_diffraction: bool = False
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
            "XRD_pattern": d["xrd_pattern"],
            "SEM_images": d["sem_images"],
            "EDS_maps": d["eds_maps"],
            "TEM_STEM": d["tem_stem"],
            "TEM_diffraction": d["tem_diffraction"],
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
        "xrd_pattern": "phase/lattice measurement",
        "sem_images": "morphology/defects",
        "eds_maps": "chemistry measurement",
        "tem_stem": "nanoscale validation",
        "tem_diffraction": "nanoscale diffraction",
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
        "xrd_pattern": "Raw diffraction pattern can validate phases, peak shifts, phase fractions and retained parent/austenite.",
        "sem_images": "Shows morphology, porosity, cracks, melt-pool tracks and whether EBSD interpretation is microstructurally plausible.",
        "eds_maps": "Checks local chemistry/segregation and Ni/Ti ratio; essential for NiTi and AM powder/as-built comparisons.",
        "tem_stem": "Validates nanoscale twins, precipitates, dislocations and habit/interface structures not resolved by EBSD.",
        "tem_diffraction": "SAED/4D-STEM diffraction validates local phases, orientations and strain at nanoscale.",
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
