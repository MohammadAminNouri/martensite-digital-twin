from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


import numpy as np

from martwin.core.rotations import axis_angle
from martwin.core.symmetry import cubic_proper_rotations
from martwin.crystallography.orientation_relationships import steel_ks_or, steel_nw_or, steel_pitsch_or
from martwin.crystallography.variants import generate_variants, identify_variant_for_known_parent
from martwin.kinetics.km import km_curve

OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

parent_orientation = axis_angle([0.4, 0.2, 0.7], np.deg2rad(17.0))
ors = [steel_ks_or(), steel_nw_or(), steel_pitsch_or()]
report = {}

for orx in ors:
    variants = generate_variants(orx, cubic_proper_rotations(), cubic_proper_rotations(), tol_deg=0.2)
    measured = parent_orientation @ variants[3].matrix_child_to_parent
    identified = identify_variant_for_known_parent(measured, parent_orientation, variants, child_sym_ops=cubic_proper_rotations())
    report[orx.name] = {
        "number_of_unique_variants": len(variants),
        "test_identified_variant": identified,
        "source_note": orx.source_note,
    }

temps = list(range(450, -51, -50))
fraction = km_curve(temps, Ms_C=350.0, alpha=0.011)
report["Koistinen-Marburger demo"] = {"temperatures_C": temps, "martensite_fraction": fraction, "Ms_C": 350.0}

(OUT / "steel_demo_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
