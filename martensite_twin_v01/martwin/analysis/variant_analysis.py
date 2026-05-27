from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from martwin.core.rotations import misorientation_angle, project_to_rotation
from martwin.crystallography.variants import Variant, identify_variant_for_known_parent, predict_child_orientations


@dataclass
class VariantAssignmentResult:
    assignments: pd.DataFrame
    summary: pd.DataFrame
    mean_error_deg: float
    max_error_deg: float
    confidence_score: float


def _ensure_matrix_list(orientations: Iterable[np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for R in orientations:
        arr = np.asarray(R, dtype=float).reshape(3, 3)
        out.append(project_to_rotation(arr))
    return out


def assign_variants_known_parent(
    child_orientations: Iterable[np.ndarray],
    parent_orientation: np.ndarray,
    variants: list[Variant],
    child_sym_ops: list[np.ndarray] | None = None,
    tolerance_deg: float = 5.0,
) -> VariantAssignmentResult:
    """Assign each measured child orientation to the closest theoretical variant.

    This is the practical first analysis mode when a parent orientation is known,
    assumed, or reconstructed externally. Results are convention-sensitive and
    should be validated against vendor/MTEX/orix conventions for publication-grade
    EBSD analysis.
    """
    rows = []
    for idx, Gc in enumerate(_ensure_matrix_list(child_orientations)):
        hit = identify_variant_for_known_parent(Gc, parent_orientation, variants, child_sym_ops=child_sym_ops)
        err = float(hit["angular_error_deg"])
        rows.append({
            "point_id": idx,
            "variant_id": int(hit["variant_id"]),
            "angular_error_deg": err,
            "fit_quality": max(0.0, 1.0 - err / tolerance_deg),
            "is_in_tolerance": bool(err <= tolerance_deg),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        summary = pd.DataFrame(columns=["variant_id", "count", "fraction", "mean_error_deg"])
        return VariantAssignmentResult(df, summary, float("nan"), float("nan"), 0.0)
    summary = (
        df.groupby("variant_id", as_index=False)
        .agg(count=("variant_id", "size"), mean_error_deg=("angular_error_deg", "mean"))
        .sort_values(["count", "variant_id"], ascending=[False, True])
    )
    summary["fraction"] = summary["count"] / len(df)
    mean_error = float(df["angular_error_deg"].mean())
    max_error = float(df["angular_error_deg"].max())
    confidence = float(df["fit_quality"].clip(0, 1).mean())
    return VariantAssignmentResult(df, summary, mean_error, max_error, confidence)


def assign_variants_known_parent_regions(
    child_orientations: Iterable[np.ndarray],
    parent_orientations: list[np.ndarray],
    parent_region_ids: Iterable[int],
    variants: list[Variant],
    child_sym_ops: list[np.ndarray] | None = None,
    tolerance_deg: float = 5.0,
) -> VariantAssignmentResult:
    """Assign variants when each point has a known parent-region label.

    This is useful for synthetic validation or interrupted/in-situ experiments
    where a parent-grain map is available. Region ids may be 1-based or 0-based;
    both are handled when possible.
    """
    orientations = _ensure_matrix_list(child_orientations)
    labels = list(parent_region_ids)
    rows = []
    for idx, (Gc, label) in enumerate(zip(orientations, labels)):
        li = int(label)
        if 1 <= li <= len(parent_orientations):
            parent = parent_orientations[li - 1]
        elif 0 <= li < len(parent_orientations):
            parent = parent_orientations[li]
        else:
            parent = parent_orientations[0]
        hit = identify_variant_for_known_parent(Gc, parent, variants, child_sym_ops=child_sym_ops)
        err = float(hit["angular_error_deg"])
        rows.append({
            "point_id": idx,
            "variant_id": int(hit["variant_id"]),
            "angular_error_deg": err,
            "fit_quality": max(0.0, 1.0 - err / tolerance_deg),
            "is_in_tolerance": bool(err <= tolerance_deg),
            "used_parent_region_id": int(label),
        })
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("variant_id", as_index=False)
        .agg(count=("variant_id", "size"), mean_error_deg=("angular_error_deg", "mean"))
        .sort_values(["count", "variant_id"], ascending=[False, True])
    ) if not df.empty else pd.DataFrame(columns=["variant_id", "count", "mean_error_deg"])
    if not summary.empty:
        summary["fraction"] = summary["count"] / len(df)
    mean_error = float(df["angular_error_deg"].mean()) if not df.empty else float("nan")
    max_error = float(df["angular_error_deg"].max()) if not df.empty else float("nan")
    confidence = float(df["fit_quality"].clip(0, 1).mean()) if not df.empty else 0.0
    return VariantAssignmentResult(df, summary, mean_error, max_error, confidence)


def variant_misorientation_matrix(variants: list[Variant], child_sym_ops: list[np.ndarray] | None = None) -> pd.DataFrame:
    """Return a pairwise misorientation matrix between theoretical variants."""
    ids = [v.id for v in variants]
    mat = np.zeros((len(variants), len(variants)))
    for i, vi in enumerate(variants):
        for j, vj in enumerate(variants):
            mat[i, j] = misorientation_angle(vi.matrix_child_to_parent, vj.matrix_child_to_parent, sym_ops=child_sym_ops)
    return pd.DataFrame(mat, index=ids, columns=ids)


def variant_table(variants: list[Variant]) -> pd.DataFrame:
    """Compact table of generated variants and their matrices."""
    rows = []
    for v in variants:
        row = {"variant_id": v.id, "parent_sym_index": v.parent_sym_index, "child_sym_index": v.child_sym_index}
        for i in range(3):
            for j in range(3):
                row[f"r{i}{j}"] = float(v.matrix_child_to_parent[i, j])
        rows.append(row)
    return pd.DataFrame(rows)
