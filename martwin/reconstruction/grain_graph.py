"""
martwin.reconstruction.grain_graph
===================================
Build and manipulate the variant grain graph used for parent phase
reconstruction.  Implements the algorithm described in:

    Hielscher, Nyyssönen, Niessen, Gazder (2022).
    "The variant graph approach to improved parent grain reconstruction."
    arXiv:2201.02103.

The graph has one *node per (grain, variant)* pair rather than one node per
grain.  This variant-level representation is the key innovation: it allows
transitivity to be maintained across multi-step grain chains while keeping
the implementation sparse and efficient.

Key objects
-----------
GrainData
    Lightweight container for the per-grain mean orientations and adjacency.

VariantGraph
    Sparse probability graph and its Markov-clustering solver.

Public API
----------
build_variant_graph(grain_data, or_obj, threshold_deg, tolerance_deg)
    → VariantGraph

reconstruct_parents(variant_graph, n_iterations, inflation)
    → ParentReconstructionResult
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import norm as sp_norm

from .orientation_relationships import OrientationRelationship, _apply_cubic_symmetry

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class GrainData:
    """
    Minimal per-grain information required for parent reconstruction.

    Attributes
    ----------
    n_grains : int
    orientations : (N, 3, 3) ndarray
        Mean orientation (rotation matrix) for each grain.
        Column j of orientations[i] is the j-th crystal axis in the
        sample reference frame.
    adjacency : list of lists
        adjacency[i] = sorted list of grain indices adjacent to grain i.
        Derived from the EBSD grain boundary map.
    phase : (N,) int ndarray
        0 = parent (retained austenite), 1 = child (martensite).
        All-child maps are handled; set phase = np.ones(N, dtype=int).
    grain_sizes : (N,) int ndarray, optional
        Number of pixels per grain (used for weighted averaging).
    """
    n_grains: int
    orientations: np.ndarray               # (N, 3, 3)
    adjacency: List[List[int]]
    phase: np.ndarray                      # (N,) int
    grain_sizes: Optional[np.ndarray] = None  # (N,) int

    def __post_init__(self):
        assert self.orientations.shape == (self.n_grains, 3, 3)
        assert self.phase.shape == (self.n_grains,)
        if self.grain_sizes is None:
            self.grain_sizes = np.ones(self.n_grains, dtype=int)

    @classmethod
    def from_arrays(
        cls,
        orientations: np.ndarray,
        adjacency: List[List[int]],
        phase: Optional[np.ndarray] = None,
        grain_sizes: Optional[np.ndarray] = None,
    ) -> "GrainData":
        N = orientations.shape[0]
        if phase is None:
            phase = np.ones(N, dtype=int)   # assume all child (fully transformed)
        return cls(
            n_grains=N,
            orientations=orientations,
            adjacency=adjacency,
            phase=phase.astype(int),
            grain_sizes=grain_sizes,
        )


# ---------------------------------------------------------------------------
# Misorientation utilities
# ---------------------------------------------------------------------------

# Pre-compute the 24 Oh symmetry operators once
_OH_SYM = _apply_cubic_symmetry(np.eye(3))


def _misorientation_angle(Ra: np.ndarray, Rb: np.ndarray,
                           sym_ops: List[np.ndarray]) -> float:
    """
    Minimum misorientation angle (degrees) between two rotation matrices,
    accounting for crystal symmetry.
    """
    M = Ra @ Rb.T
    best = 180.0
    for S in sym_ops:
        Ms = S @ M
        trace = np.clip((np.trace(Ms) - 1.0) / 2.0, -1.0, 1.0)
        angle = np.degrees(np.arccos(trace))
        if angle < best:
            best = angle
    return best


def _child_to_child_expected_angles(
    or_obj: OrientationRelationship,
) -> List[float]:
    """
    Compute all unique child–child misorientation angles that arise within
    a single parent grain for a given OR.

    These are the angles between Ri @ Rj^T for all distinct variant pairs
    (i, j), which the grain graph uses to compute edge probabilities.
    """
    angles = []
    n = or_obj.n_variants
    for i in range(n):
        for j in range(i + 1, n):
            M = or_obj.variants[i] @ or_obj.variants[j].T
            trace = np.clip((np.trace(M) - 1.0) / 2.0, -1.0, 1.0)
            angle = np.degrees(np.arccos(trace))
            # Take symmetry minimum
            min_angle = angle
            for S in _OH_SYM:
                Ms = S @ M
                tr = np.clip((np.trace(Ms) - 1.0) / 2.0, -1.0, 1.0)
                a = np.degrees(np.arccos(tr))
                if a < min_angle:
                    min_angle = a
            angles.append(min_angle)
    return sorted(set(round(a, 2) for a in angles))


def _edge_probability(
    measured_angle: float,
    threshold_deg: float,
    tolerance_deg: float,
) -> float:
    """
    Cumulative Gaussian probability that a grain pair shares a common parent.

    P(angle) = 0.5 * erfc((angle - threshold) / (sqrt(2) * tolerance))

    - At angle == threshold     → P = 0.50
    - At angle == 0             → P ≈ 1.0
    - At angle >> threshold + 3σ → P ≈ 0.0
    """
    from scipy.special import erfc
    z = (measured_angle - threshold_deg) / (np.sqrt(2) * tolerance_deg)
    return 0.5 * float(erfc(z))


# ---------------------------------------------------------------------------
# Variant-level graph construction
# ---------------------------------------------------------------------------

@dataclass
class VariantGraph:
    """
    Sparse probability graph where each node represents one (grain, variant).

    node_id(grain_i, variant_j) = grain_i * n_variants + variant_j

    Attributes
    ----------
    n_grains : int
    n_variants : int
    n_nodes : int  = n_grains * n_variants
    P : scipy.sparse.csr_matrix, shape (n_nodes, n_nodes)
        Probability matrix.  P[u, v] = probability that nodes u and v
        share a common parent orientation.
    or_obj : OrientationRelationship
    grain_data : GrainData
    """
    n_grains: int
    n_variants: int
    or_obj: OrientationRelationship
    grain_data: GrainData
    P: sparse.csr_matrix = field(repr=False, default=None)

    @property
    def n_nodes(self) -> int:
        return self.n_grains * self.n_variants

    def node_id(self, grain: int, variant: int) -> int:
        return grain * self.n_variants + variant

    def grain_variant(self, node: int) -> Tuple[int, int]:
        return divmod(node, self.n_variants)


def _candidate_parent_orientations(
    grain_orient: np.ndarray,   # (3,3)
    or_obj: OrientationRelationship,
) -> np.ndarray:
    """
    Compute all n_variants candidate parent orientations from a single
    child grain orientation.

    g_parent_j = R_variant_j^T @ g_child

    Returns
    -------
    (n_variants, 3, 3) ndarray
    """
    parents = np.stack(
        [v.T @ grain_orient for v in or_obj.variants], axis=0
    )
    return parents


def _common_parent_probability(
    orient_a: np.ndarray,   # (3,3) child grain a
    orient_b: np.ndarray,   # (3,3) child grain b
    or_obj: OrientationRelationship,
    threshold_deg: float,
    tolerance_deg: float,
) -> np.ndarray:
    """
    Compute the (n_variants_a, n_variants_b) matrix of probabilities that
    variant i of grain a and variant j of grain b have the same parent.

    Two variants share a parent iff the misorientation between the parent
    orientations they imply is below the clustering threshold.

    Returns
    -------
    P_ab : (n_variants, n_variants) ndarray  (values in [0, 1])
    """
    nv = or_obj.n_variants
    parents_a = _candidate_parent_orientations(orient_a, or_obj)  # (nv,3,3)
    parents_b = _candidate_parent_orientations(orient_b, or_obj)  # (nv,3,3)

    P_ab = np.zeros((nv, nv), dtype=np.float32)
    for i in range(nv):
        for j in range(nv):
            angle = _misorientation_angle(parents_a[i], parents_b[j], _OH_SYM)
            P_ab[i, j] = _edge_probability(angle, threshold_deg, tolerance_deg)
    return P_ab


def build_variant_graph(
    grain_data: GrainData,
    or_obj: OrientationRelationship,
    threshold_deg: float = 3.0,
    tolerance_deg: float = 3.0,
) -> VariantGraph:
    """
    Construct the variant-level probability graph for parent reconstruction.

    Parameters
    ----------
    grain_data : GrainData
    or_obj : OrientationRelationship
        The OR to use (e.g. OR_REGISTRY["KS"]).
    threshold_deg : float
        Misorientation angle at which edge probability = 0.5 (degrees).
    tolerance_deg : float
        Standard deviation of the Gaussian probability model (degrees).

    Returns
    -------
    VariantGraph
    """
    N = grain_data.n_grains
    nv = or_obj.n_variants
    n_nodes = N * nv

    rows, cols, data = [], [], []

    # Diagonal self-edges: every node has probability 1 with itself
    for node in range(n_nodes):
        rows.append(node)
        cols.append(node)
        data.append(1.0)

    # Off-diagonal edges: iterate over all adjacent grain pairs
    for grain_i in range(N):
        orient_i = grain_data.orientations[grain_i]
        for grain_j in grain_data.adjacency[grain_i]:
            if grain_j <= grain_i:
                continue   # process each pair once; P is symmetric
            orient_j = grain_data.orientations[grain_j]
            P_ij = _common_parent_probability(
                orient_i, orient_j, or_obj, threshold_deg, tolerance_deg
            )
            for vi in range(nv):
                for vj in range(nv):
                    p = float(P_ij[vi, vj])
                    if p > 1e-4:   # skip near-zero entries for sparsity
                        u = grain_i * nv + vi
                        v = grain_j * nv + vj
                        rows.append(u); cols.append(v); data.append(p)
                        rows.append(v); cols.append(u); data.append(p)

    P = sparse.csr_matrix(
        (data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32
    )
    return VariantGraph(
        n_grains=N,
        n_variants=nv,
        or_obj=or_obj,
        grain_data=grain_data,
        P=P,
    )


# ---------------------------------------------------------------------------
# Markov Clustering (MCL) on the variant graph
# ---------------------------------------------------------------------------

def _normalise_columns(M: sparse.csr_matrix) -> sparse.csr_matrix:
    """Column-normalise a sparse matrix so each column sums to 1."""
    col_sums = np.array(M.sum(axis=0)).flatten()
    col_sums[col_sums == 0] = 1.0          # avoid division by zero
    D_inv = sparse.diags(1.0 / col_sums)
    return M @ D_inv


def _inflate(M: sparse.csr_matrix, r: float) -> sparse.csr_matrix:
    """
    MCL inflation step: element-wise power followed by column normalisation.
    For sparse matrices we operate on the stored data array directly.
    """
    M = M.copy()
    M.data = np.power(M.data, r)
    return _normalise_columns(M)


def _expand(M: sparse.csr_matrix, e: int) -> sparse.csr_matrix:
    """MCL expansion step: matrix power (e=2 is standard)."""
    result = M
    for _ in range(e - 1):
        result = result @ M
    return result


def _prune(M: sparse.csr_matrix, threshold: float = 1e-4) -> sparse.csr_matrix:
    """Remove near-zero entries to maintain sparsity."""
    M = M.copy()
    M.data[M.data < threshold] = 0
    M.eliminate_zeros()
    return M


def markov_cluster(
    vg: VariantGraph,
    e: int = 2,
    r: float = 2.0,
    n_iterations: int = 100,
    convergence_tol: float = 1e-5,
    prune_threshold: float = 1e-4,
    verbose: bool = True,
) -> np.ndarray:
    """
    Run the Markov Clustering Algorithm (MCL) on the variant graph.

    Parameters
    ----------
    vg : VariantGraph
    e : int
        Expansion exponent (default 2).
    r : float
        Inflation exponent (default 2.0).  Higher r → smaller clusters.
    n_iterations : int
        Maximum number of expand–inflate iterations.
    convergence_tol : float
        Stop when the Frobenius norm of the change falls below this value.
    prune_threshold : float
        Entries below this value are set to zero for sparsity.
    verbose : bool

    Returns
    -------
    cluster_labels : (n_nodes,) int ndarray
        For each (grain, variant) node, the cluster ID.  Nodes in the same
        cluster share a common parent orientation variant.
    """
    M = _normalise_columns(vg.P.astype(np.float64))

    prev_M = None
    for iteration in range(n_iterations):
        M = _expand(M, e)
        M = _inflate(M, r)
        M = _prune(M, prune_threshold)

        if prev_M is not None:
            diff = M - prev_M
            change = sp_norm(diff, ord="fro")
            if verbose:
                print(f"  MCL iteration {iteration+1:3d}: Δ = {change:.2e}")
            if change < convergence_tol:
                if verbose:
                    print(f"  Converged at iteration {iteration+1}.")
                break
        prev_M = M.copy()

    # Extract clusters: two nodes belong to the same cluster iff they
    # have the same dominant column in the steady-state matrix M.
    # Implementation: argmax of each column → "attractor" node.
    M_csc = M.tocsc()
    n_nodes = M.shape[1]
    attractors = np.zeros(n_nodes, dtype=int)
    for col in range(n_nodes):
        col_data = M_csc.getcol(col)
        if col_data.nnz == 0:
            attractors[col] = col
        else:
            attractors[col] = col_data.indices[np.argmax(col_data.data)]

    # Map attractors to sequential cluster IDs
    unique_attractors = np.unique(attractors)
    attractor_to_id = {a: idx for idx, a in enumerate(unique_attractors)}
    cluster_labels = np.array([attractor_to_id[a] for a in attractors])
    return cluster_labels
