from __future__ import annotations

import pandas as pd

CONCEPTS = {
    "digital_twin": {
        "plain": "A digital twin is not just a plot. It is a structured link between material/process inputs, physics models, measured data, uncertainty, and decisions.",
        "used_for": "Used to explain current microstructure, estimate missing transformation behavior, and decide the next experiment or process change.",
    },
    "orientation_relationship": {
        "plain": "An orientation relationship (OR) tells how a crystal direction/plane in the parent phase lines up with a direction/plane in the martensite child phase.",
        "used_for": "Used to generate all possible martensite variants and compare measured EBSD/TKD orientations against theory.",
    },
    "variant": {
        "plain": "A variant is one crystallographically allowed way the parent phase can transform into martensite. One parent grain can create many child variants.",
        "used_for": "Used to label EBSD pixels, quantify variant selection, infer parent grains, and connect microstructure to transformation strain.",
    },
    "misorientation": {
        "plain": "Misorientation is the smallest rotation angle needed to bring one crystal orientation into another, considering symmetry.",
        "used_for": "Used to compare variants, classify boundaries, fit ORs, and reconstruct parent grains.",
    },
    "parent_reconstruction": {
        "plain": "Parent reconstruction estimates the original parent grains from the martensite orientations measured after transformation.",
        "used_for": "Used for prior-austenite reconstruction in steel and B2 parent reconstruction in NiTi.",
    },
    "habit_plane": {
        "plain": "The habit plane is the approximate interface plane between parent and martensite.",
        "used_for": "Used to compare predicted interface traces against EBSD/TKD maps and microscopy.",
    },
    "kinetics": {
        "plain": "Kinetics describes how much phase transforms as temperature, stress, or time changes.",
        "used_for": "Used to connect thermal cycle/DSC/dilatometry to martensite fraction and later property predictions.",
    },
}

TABLE_EXPLANATIONS = {
    "or_matrix": {
        "title": "Orientation relationship matrix",
        "what": "This 3×3 rotation matrix converts a child/martensite crystal orientation into the parent crystal reference frame used by the model.",
        "why": "It is the mathematical core of the OR. The app uses it to generate theoretical variants.",
        "how_to_read": "Rows and columns are x/y/z basis components. Numbers near 1 or -1 mean strong alignment; values near 0 mean little projection on that axis. This matrix is not a measurement table; it is the model assumption.",
        "used_downstream": "Variant library → EBSD variant assignment → parent reconstruction → habit/compatibility analysis.",
    },
    "variant_library": {
        "title": "Variant library",
        "what": "Each row is one theoretical martensite variant produced by combining the OR with parent/child crystal symmetry operations.",
        "why": "The app needs this library as a catalogue of all possible child orientations that could come from a parent grain.",
        "how_to_read": "variant_id is the label. parent_sym_index and child_sym_index are the symmetry operations used to create that variant. r00...r22 are the 3×3 rotation-matrix entries for that variant.",
        "used_downstream": "Measured EBSD/TKD orientations are compared with every row; the nearest row becomes the assigned variant.",
    },
    "misorientation_matrix": {
        "title": "Pairwise theoretical variant misorientation matrix",
        "what": "This table compares every theoretical variant against every other theoretical variant. The cell value is the misorientation angle in degrees.",
        "why": "Variant boundaries and packets often have characteristic misorientation angles. This table tells which variant pairs are crystallographically related.",
        "how_to_read": "Rows and columns are variant IDs. The diagonal is 0 because a variant compared to itself has no misorientation. A 90°, 120°, or 180° cell means those two variants differ by that rotation angle under the selected symmetry convention.",
        "used_downstream": "Boundary interpretation, variant-pair statistics, packet/block grouping, parent reconstruction quality checks.",
    },
    "dataset_preview": {
        "title": "EBSD/TKD dataset preview",
        "what": "Each row is one measured or synthetic orientation point, usually one EBSD/TKD pixel or grid point.",
        "why": "This is the experimental/synthetic evidence that the twin analyzes.",
        "how_to_read": "x and y are map coordinates. r00...r22 or phi1/Phi/phi2 define the orientation. Optional columns such as phase, grain_id, CI/IQ, true_variant_id, and parent_region_id give metadata or validation labels.",
        "used_downstream": "Variant assignment, map plotting, parent reconstruction, confidence scoring.",
    },
    "variant_summary": {
        "title": "Variant population summary",
        "what": "This table says which variants appear most often in the measured/synthetic map.",
        "why": "Dominant variants may indicate texture, stress-assisted transformation, self-accommodation, or process history effects.",
        "how_to_read": "count is the number of points assigned to that variant; fraction is count divided by all analyzed points; mean_error_deg is the average angular mismatch between measured orientations and the theoretical variant.",
        "used_downstream": "Variant selection analysis, self-accommodation checks, comparison between heat treatments or LPBF parameters.",
    },
    "point_assignments": {
        "title": "Point-level variant assignments",
        "what": "This table gives the detailed classification result for every map point.",
        "why": "It allows you to inspect whether individual pixels/points are well fitted or suspicious.",
        "how_to_read": "variant_id is the nearest theoretical variant; angular_error_deg is mismatch; fit_quality goes from 0 to 1; is_in_tolerance says whether the error is acceptable; reconstructed_parent_cluster is the prototype parent group assigned by the current algorithm.",
        "used_downstream": "Variant map, error map, parent cluster map, export/report.",
    },
    "open_manifest": {
        "title": "Open data/tool manifest",
        "what": "A registry of public datasets, tools, papers, and software that can support the twin.",
        "why": "A complete twin needs validation data and known reference tools. This table tells what can be used and what each source contributes.",
        "how_to_read": "material_system indicates relevance; data_types says what the source provides; license_or_access tells whether it is open data, documentation, code, or a paper; priority ranks usefulness for validation.",
        "used_downstream": "Dataset ingestion, validation, benchmarking, roadmap planning.",
    },
}

COLUMN_EXPLANATIONS = {
    "variant_id": "Unique theoretical martensite variant label. It is not a grain ID; one parent grain can contain many variants.",
    "parent_sym_index": "Index of the parent-phase symmetry operation used to create the variant.",
    "child_sym_index": "Index of the child/martensite symmetry operation used to remove duplicate equivalent variants.",
    "r00": "Rotation matrix entry row 0, column 0. Together r00...r22 define orientation of the variant.",
    "r01": "Rotation matrix entry row 0, column 1.",
    "r02": "Rotation matrix entry row 0, column 2.",
    "r10": "Rotation matrix entry row 1, column 0.",
    "r11": "Rotation matrix entry row 1, column 1.",
    "r12": "Rotation matrix entry row 1, column 2.",
    "r20": "Rotation matrix entry row 2, column 0.",
    "r21": "Rotation matrix entry row 2, column 1.",
    "r22": "Rotation matrix entry row 2, column 2.",
    "point_id": "Index of one EBSD/TKD map point or synthetic pixel.",
    "x": "Horizontal map coordinate or grid column.",
    "y": "Vertical map coordinate or grid row.",
    "parent_region_id": "Synthetic or known parent-region label. In real work this may come from reconstructed or measured parent grains.",
    "true_variant_id": "Only available in synthetic/demo data. It is the known ground-truth variant used to test the algorithm.",
    "angular_error_deg": "Angular mismatch in degrees between the measured/synthetic orientation and the nearest theoretical variant.",
    "fit_quality": "Simple quality score: 1 is excellent; 0 means the mismatch reaches or exceeds the chosen tolerance.",
    "is_in_tolerance": "True if angular_error_deg is below the chosen tolerance. False means the point may not fit the assumed OR/parent orientation.",
    "reconstructed_parent_cluster": "Prototype cluster label for points estimated to share a parent orientation. Current v0.3 clustering is not publication-grade yet.",
    "count": "Number of map points assigned to a variant.",
    "fraction": "count divided by total analyzed map points.",
    "mean_error_deg": "Average angular mismatch for points assigned to this variant.",
    "Temperature_C": "Temperature in degrees Celsius.",
    "martensite_fraction": "Predicted transformed martensite fraction during cooling for the simple steel model.",
    "B19prime_fraction_cooling": "Estimated fraction of B19′ martensite during cooling in the simplified NiTi hysteresis model.",
    "B2_fraction_heating": "Estimated fraction of B2 austenite during heating in the simplified NiTi hysteresis model.",
}

MATURITY_LEVELS = [
    {"level": "L0", "name": "Theoretical crystallography", "needed": "Material system + OR + lattice/symmetry assumptions", "trust": "Useful for learning and hypothesis generation only."},
    {"level": "L1", "name": "EBSD/TKD interpretation", "needed": "Orientation map + phase/symmetry + OR/variant library", "trust": "Useful for variant maps and preliminary parent reconstruction."},
    {"level": "L2", "name": "Calibrated transformation twin", "needed": "EBSD/TKD + DSC/dilatometry + XRD + composition + heat treatment", "trust": "Useful for explaining measured transformation behavior."},
    {"level": "L3", "name": "Process-aware twin", "needed": "L2 + cooling/thermal history or LPBF parameters + process metadata", "trust": "Useful for comparing routes and predicting process effects."},
    {"level": "L4", "name": "Predictive engineering twin", "needed": "L3 + mechanical tests + residual stress + validated model uncertainty", "trust": "Useful for design decisions after validation against independent samples."},
]


def workflow_dataframe() -> pd.DataFrame:
    return pd.DataFrame([
        {"step": "1. Material/process record", "input": "composition, heat treatment, LPBF route, thermal cycle", "model action": "stores assumptions and checks missing data", "output": "evidence ledger + confidence"},
        {"step": "2. Crystallographic model", "input": "parent phase, child phase, OR, symmetries", "model action": "generates variant library and misorientation operators", "output": "theoretical variants"},
        {"step": "3. EBSD/TKD data", "input": "orientation map or synthetic map", "model action": "reads orientations point-by-point", "output": "analysis-ready orientation dataset"},
        {"step": "4. Variant assignment", "input": "orientation dataset + variant library", "model action": "finds nearest theoretical variant for each point", "output": "variant map + fit errors"},
        {"step": "5. Parent reconstruction", "input": "variant assignments + orientation operators", "model action": "clusters child orientations into possible parent groups", "output": "prototype parent map"},
        {"step": "6. Kinetics", "input": "Ms/Mf/As/Af or Ms/KM alpha/cooling curve", "model action": "computes phase fraction vs temperature", "output": "transformation curve"},
        {"step": "7. Reliability", "input": "which data are measured vs assumed", "model action": "scores gaps and recommends experiments", "output": "twin maturity level + next actions"},
    ])


def data_requirement_table(material: str = "NiTi", lpbf: bool = False) -> pd.DataFrame:
    common = [
        ("composition", "Exact chemistry", "Composition affects lattice parameters, driving force, transformation temperatures.", "thermodynamics, kinetics, confidence"),
        ("heat_treatment", "Thermal processing route", "Solution/ageing/quench history changes phases, precipitates, stress, and grain structure.", "kinetics, interpretation"),
        ("ebsd_or_tkd", "Orientation map", "Needed to identify variants and reconstruct parent grains.", "variant analysis, parent reconstruction"),
    ]
    niti = [
        ("DSC", "Ms/Mf/As/Af + hysteresis", "NiTi transformation temperatures are highly composition/process sensitive.", "kinetics curve calibration"),
        ("XRD_lattice", "B2/B19′ lattice parameters", "Actual lattice parameters are needed for exact distortion/habit-plane work.", "crystallography, phase fractions"),
        ("stress_strain", "Mechanical response", "Needed for shape-memory/superelastic response and stress-assisted transformation.", "mechanics, validation"),
        ("oxygen_carbon", "O/C contamination", "Impurities shift NiTi transformation behavior and can create brittle phases.", "confidence, process quality"),
        ("thermal_history", "Cooling/heating path", "Transformation depends on the real thermal path, not only final temperature.", "kinetics, LPBF process link"),
    ]
    steel = [
        ("cooling_curve", "Quench/cooling curve", "Martensite/bainite fractions depend on cooling path.", "kinetics, dilatometry fitting"),
        ("prior_austenite_reference", "Reference parent grain map", "Needed to validate parent reconstruction algorithms.", "validation"),
        ("hardness", "Hardness/strength data", "Links microstructure to mechanical outcome.", "property prediction"),
        ("retained_austenite_XRD", "Retained austenite fraction", "Needed to know whether transformation completed.", "phase balance, validation"),
    ]
    am = [
        ("laser_parameters", "LPBF laser power/speed/hatch/layer", "Defines melt-pool thermal history and texture formation.", "process twin"),
        ("scan_strategy", "Scan rotation/stripe/chessboard", "Changes thermal gradients, texture, and residual stress.", "process twin"),
        ("powder_chemistry", "Powder composition/oxygen", "Powder chemistry often differs from nominal alloy chemistry.", "confidence, composition"),
        ("melt_pool_or_thermal_model", "Measured/simulated thermal history", "Needed to connect processing to transformation.", "thermal-process coupling"),
        ("porosity", "Porosity/defects", "Defects affect mechanical behavior and EBSD reliability.", "mechanics, quality"),
        ("residual_stress", "Residual stress", "Stress affects variant selection and transformation temperatures.", "mechanics, variant selection"),
    ]
    rows = common + (niti if material.lower().startswith("niti") else steel)
    if lpbf:
        rows += am
    return pd.DataFrame(rows, columns=["key", "data needed", "why it matters", "used in"])


def explain_columns(columns: list[str]) -> pd.DataFrame:
    rows = []
    for c in columns:
        rows.append({"column": c, "meaning": COLUMN_EXPLANATIONS.get(c, "No special explanation registered yet; inspect source data conventions.")})
    return pd.DataFrame(rows)
