from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from martwin.core.rotations import misorientation_angle, project_to_rotation
from .variants import Variant, parent_candidates_from_child


@dataclass
class ReconstructionResult:
    parent_orientations: list[np.ndarray]
    labels: np.ndarray
    confidence: list[float]
    notes: list[str]


def _mean_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    M = np.mean(np.stack(rotations), axis=0)
    return project_to_rotation(M)


def reconstruct_parent_orientations_greedy(
    child_orientations: list[np.ndarray],
    variants: list[Variant],
    threshold_deg: float = 5.0,
) -> ReconstructionResult:
    """Very simple orientation-space parent reconstruction prototype.

    For each child orientation, all parent candidates are generated. Each child is
    assigned to the closest existing parent cluster if any candidate is within
    threshold; otherwise a new cluster is created from its first candidate.

    This is NOT a replacement for MTEX/ARPGE/graph-clustering. It is a working
    placeholder that establishes the data path and can be replaced by graph-based
    reconstruction later.
    """
    clusters: list[list[np.ndarray]] = []
    labels = np.full(len(child_orientations), -1, dtype=int)
    scores = np.zeros(len(child_orientations), dtype=float)

    for idx, Gc in enumerate(child_orientations):
        candidates = parent_candidates_from_child(Gc, variants)
        best_cluster = None
        best_angle = float("inf")
        best_candidate = candidates[0]

        for ci, members in enumerate(clusters):
            center = _mean_rotation(members)
            for cand in candidates:
                angle = misorientation_angle(cand, center)
                if angle < best_angle:
                    best_angle = angle
                    best_cluster = ci
                    best_candidate = cand

        if best_cluster is not None and best_angle <= threshold_deg:
            clusters[best_cluster].append(best_candidate)
            labels[idx] = best_cluster
            scores[idx] = max(0.0, 1.0 - best_angle / threshold_deg)
        else:
            clusters.append([candidates[0]])
            labels[idx] = len(clusters) - 1
            scores[idx] = 0.5

    parents = [_mean_rotation(c) for c in clusters]
    conf = [float(np.mean(scores[labels == i])) for i in range(len(clusters))]
    return ReconstructionResult(
        parent_orientations=parents,
        labels=labels,
        confidence=conf,
        notes=[
            "Greedy prototype only; validate with MTEX/ARPGE-style graph reconstruction for serious EBSD work.",
            f"Threshold used: {threshold_deg} degrees.",
        ],
    )
