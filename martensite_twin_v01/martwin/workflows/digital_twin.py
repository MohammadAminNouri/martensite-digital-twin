from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np
import pandas as pd

from martwin.core.symmetry import cubic_proper_rotations, monoclinic_2_unique_axis_b
from martwin.crystallography.orientation_relationships import (
    OrientationRelationship,
    cayron_niti_natural_or,
    steel_ks_or,
    steel_nw_or,
    steel_pitsch_or,
)
from martwin.crystallography.variants import Variant, generate_variants
from martwin.analysis.variant_analysis import (
    VariantAssignmentResult,
    assign_variants_known_parent,
    variant_misorientation_matrix,
    variant_table,
)
from martwin.crystallography.parent_reconstruction import ReconstructionResult, reconstruct_parent_orientations_greedy
from martwin.calibration.gap_analysis import GapReport, assess_data_gaps

MaterialSystem = Literal["NiTi B2→B19′", "Steel fcc→bcc/bct"]


@dataclass
class TwinConfiguration:
    material_system: MaterialSystem
    orientation_relationship: str = "Cayron natural OR"
    beta_deg: float = 96.8
    steel_or: str = "KS"
    angular_tolerance_deg: float = 5.0
    reconstruction_threshold_deg: float = 5.0
    notes: str = ""


@dataclass
class TwinModel:
    config: TwinConfiguration
    orientation_relationship: OrientationRelationship
    variants: list[Variant]
    parent_sym_ops: list[np.ndarray]
    child_sym_ops: list[np.ndarray]

    @property
    def material_key(self) -> str:
        return "NiTi" if self.config.material_system.startswith("NiTi") else "steel"


def build_twin_model(config: TwinConfiguration) -> TwinModel:
    parent_sym = cubic_proper_rotations()
    if config.material_system.startswith("NiTi"):
        orx = cayron_niti_natural_or(config.beta_deg)
        child_sym = monoclinic_2_unique_axis_b()
    else:
        or_map = {"KS": steel_ks_or, "NW": steel_nw_or, "Pitsch": steel_pitsch_or}
        orx = or_map.get(config.steel_or, steel_ks_or)()
        child_sym = cubic_proper_rotations()
    variants = generate_variants(orx, parent_sym, child_sym)
    return TwinModel(config=config, orientation_relationship=orx, variants=variants, parent_sym_ops=parent_sym, child_sym_ops=child_sym)


@dataclass
class TwinAnalysisResult:
    config: dict
    material_key: str
    orientation_relationship_name: str
    n_variants: int
    assignment_summary: pd.DataFrame | None
    assignments: pd.DataFrame | None
    reconstruction_labels: list[int] | None
    reconstruction_confidence: list[float] | None
    data_gap_report: GapReport | None
    warnings: list[str]

    def metrics(self) -> dict:
        out = {
            "material_key": self.material_key,
            "orientation_relationship": self.orientation_relationship_name,
            "n_variants": self.n_variants,
        }
        if self.assignments is not None and not self.assignments.empty:
            out["n_points"] = int(len(self.assignments))
            out["mean_variant_error_deg"] = float(self.assignments["angular_error_deg"].mean())
            out["points_in_tolerance_fraction"] = float(self.assignments["is_in_tolerance"].mean())
        if self.reconstruction_labels is not None:
            out["n_reconstructed_parent_clusters"] = int(len(set(self.reconstruction_labels)))
        if self.data_gap_report is not None:
            out["data_confidence_score"] = float(self.data_gap_report.confidence_score)
        return out


def run_known_parent_analysis(
    model: TwinModel,
    child_orientations: list[np.ndarray],
    parent_orientation: np.ndarray | None = None,
    available_data: dict[str, bool] | None = None,
) -> tuple[VariantAssignmentResult, ReconstructionResult, GapReport | None]:
    if parent_orientation is None:
        parent_orientation = np.eye(3)
    assignments = assign_variants_known_parent(
        child_orientations,
        parent_orientation=parent_orientation,
        variants=model.variants,
        child_sym_ops=model.child_sym_ops,
        tolerance_deg=model.config.angular_tolerance_deg,
    )
    recon = reconstruct_parent_orientations_greedy(
        child_orientations,
        model.variants,
        threshold_deg=model.config.reconstruction_threshold_deg,
    )
    gaps = None
    if available_data is not None:
        gaps = assess_data_gaps(model.material_key, available_data)
    return assignments, recon, gaps


def variant_library_tables(model: TwinModel) -> dict[str, pd.DataFrame]:
    return {
        "variants": variant_table(model.variants),
        "misorientation_matrix_deg": variant_misorientation_matrix(model.variants, model.child_sym_ops),
    }
