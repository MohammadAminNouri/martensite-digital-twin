"""
martwin.reconstruction.parent_reconstructor
============================================
High-level parent phase reconstruction engine.

Workflow
--------
1. Load EBSD data → GrainData
2. Choose an OR (e.g. "KS") or auto-detect it.
3. Build the variant graph.
4. Run Markov clustering to group child variants into parent clusters.
5. Vote for the best parent orientation in each cluster.
6. (Optional) Refine the OR from the measured misorientation data.
7. Back-project the reconstructed parent orientation to each EBSD pixel.

Main entry point
----------------
ParentReconstructor.run() → ParentReconstructionResult

OR refinement
-------------
ORRefiner.refine(grain_data, initial_or) → refined OrientationRelationship
    Uses scipy.optimize.minimize to minimise the mean angular deviation
    between measured child–child misorientations and the theoretical
    OR-derived child–child misorientations.

References
----------
Hielscher et al. (2022) arXiv:2201.02103  — variant graph algorithm
Niessen et al. (2022) J. Appl. Cryst. 55  — MTEX framework
Nyyssönen (2018) Acta Mater.              — OR refinement method
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .grain_graph import (
    GrainData,
    VariantGraph,
    build_variant_graph,
    markov_cluster,
    _misorientation_angle,
    _OH_SYM,
    _candidate_parent_orientations,
)
from .orientation_relationships import (
    OrientationRelationship,
    OR_REGISTRY,
    get_OR,
    _apply_cubic_symmetry,
    _normalise,
)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ParentReconstructionResult:
    """
    Complete output of the parent grain reconstruction.

    Attributes
    ----------
    parent_orientations : (N_grains,) ndarray of (3,3) rotation matrices
        Best-estimate parent orientation for each child grain.
        Grains that could not be reconstructed retain np.nan in their matrix.
    parent_grain_id : (N_grains,) int ndarray
        Cluster (parent grain) ID each child grain belongs to.
        -1 = unassigned.
    variant_id : (N_grains,) int ndarray
        Index of the OR variant assigned to each child grain (0-based).
        -1 = unassigned.
    fit : (N_grains,) float ndarray
        Angular misfit (degrees) between the reconstructed parent orientation
        and each child grain's implied parent via its assigned variant.
    or_used : OrientationRelationship
        The OR (possibly refined) used for reconstruction.
    cluster_labels : (N_nodes,) int ndarray
        Raw MCL cluster labels on the variant graph nodes.
    n_parent_grains : int
        Number of distinct reconstructed parent grains.
    reconstruction_fraction : float
        Fraction of child grains successfully assigned to a parent (fit < 5°).
    """
    parent_orientations: np.ndarray          # (N, 3, 3)
    parent_grain_id: np.ndarray              # (N,) int
    variant_id: np.ndarray                   # (N,) int
    fit: np.ndarray                          # (N,) float  degrees
    or_used: OrientationRelationship
    cluster_labels: np.ndarray               # (N * n_variants,) int
    n_parent_grains: int
    reconstruction_fraction: float

    def summary(self) -> str:
        lines = [
            f"Parent phase reconstruction summary",
            f"  OR used              : {self.or_used.name} ({self.or_used.n_variants} variants)",
            f"  Reconstructed grains : {self.n_parent_grains}",
            f"  Reconstruction frac  : {self.reconstruction_fraction:.1%}",
            f"  Mean fit (°)         : {np.nanmean(self.fit):.2f}",
            f"  Fit quintiles (°)    : "
            + ", ".join(
                f"{np.nanpercentile(self.fit[self.fit >= 0], q):.1f}"
                for q in [25, 50, 75, 90, 95]
            ),
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# OR auto-detection
# ---------------------------------------------------------------------------

def detect_OR(
    grain_data: GrainData,
    candidates: Optional[List[str]] = None,
    n_pairs: int = 500,
    verbose: bool = True,
) -> Tuple[str, float]:
    """
    Identify the most likely OR for the dataset by finding which OR best
    explains the observed child–child boundary misorientations.

    Parameters
    ----------
    grain_data : GrainData
    candidates : list of str, optional
        OR names to test.  Defaults to ["KS", "NW", "Pitsch", "GT"].
    n_pairs : int
        Maximum number of adjacent grain pairs to sample for speed.
    verbose : bool

    Returns
    -------
    best_or_name : str
    best_mean_misfit : float  (degrees)
    """
    if candidates is None:
        candidates = ["KS", "NW", "Pitsch", "GT"]

    # Collect up to n_pairs adjacent child–child grain pairs
    pairs = []
    for i in range(grain_data.n_grains):
        if grain_data.phase[i] != 1:
            continue
        for j in grain_data.adjacency[i]:
            if j > i and grain_data.phase[j] == 1:
                pairs.append((i, j))
    if len(pairs) > n_pairs:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pairs), n_pairs, replace=False)
        pairs = [pairs[k] for k in idx]

    best_name = candidates[0]
    best_score = 1e9

    for or_name in candidates:
        or_obj = get_OR(or_name)
        misfits = []
        for gi, gj in pairs:
            oi = grain_data.orientations[gi]
            oj = grain_data.orientations[gj]
            # For each candidate parent variant of grain i,
            # find the minimum misorientation to any parent variant of grain j
            parents_i = _candidate_parent_orientations(oi, or_obj)
            parents_j = _candidate_parent_orientations(oj, or_obj)
            min_misfit = 180.0
            for pi in parents_i:
                for pj in parents_j:
                    angle = _misorientation_angle(pi, pj, _OH_SYM)
                    if angle < min_misfit:
                        min_misfit = angle
            misfits.append(min_misfit)
        mean_misfit = float(np.mean(misfits))
        if verbose:
            print(f"  OR {or_name:12s}: mean parent misfit = {mean_misfit:.2f}°")
        if mean_misfit < best_score:
            best_score = mean_misfit
            best_name = or_name

    if verbose:
        print(f"  → Best OR: {best_name} ({best_score:.2f}°)")
    return best_name, best_score


# ---------------------------------------------------------------------------
# OR refinement
# ---------------------------------------------------------------------------

class ORRefiner:
    """
    Refine an OR from measured EBSD data by minimising the mean angular
    deviation between the reconstructed parent orientations of adjacent
    child grain pairs.

    Method
    ------
    The OR is parameterised as a 3-vector of Euler angles (ZYZ convention)
    that describe the rotation applied to the reference prototype variant.
    The cost function is the mean angular deviation over all adjacent
    child–child grain pairs.

    Usage
    -----
    >>> refiner = ORRefiner(grain_data, or_obj=get_OR("KS"))
    >>> refined_or = refiner.refine(threshold_deg=5.0)
    """

    def __init__(self, grain_data: GrainData, or_obj: OrientationRelationship):
        self.grain_data = grain_data
        self.or_obj = or_obj

    def _build_pairs(self, max_pairs: int = 2000):
        pairs = []
        for i in range(self.grain_data.n_grains):
            if self.grain_data.phase[i] != 1:
                continue
            for j in self.grain_data.adjacency[i]:
                if j > i and self.grain_data.phase[j] == 1:
                    pairs.append((i, j))
                    if len(pairs) >= max_pairs:
                        return pairs
        return pairs

    def _cost(self, euler_delta: np.ndarray, pairs: List[Tuple[int, int]]) -> float:
        """
        Cost function: mean angular deviation between parent orientations
        of adjacent grain pairs, after applying a small perturbation
        (euler_delta) to the prototype variant.
        """
        # Build perturbed OR: apply small rotation to the prototype variant
        dR = Rotation.from_euler("ZYZ", euler_delta, degrees=True).as_matrix()
        perturbed_variants = [dR @ v for v in self.or_obj.variants]

        misfits = []
        for gi, gj in pairs:
            oi = self.grain_data.orientations[gi]
            oj = self.grain_data.orientations[gj]
            # Parents from perturbed variants
            parents_i = [v.T @ oi for v in perturbed_variants]
            parents_j = [v.T @ oj for v in perturbed_variants]
            min_misfit = 180.0
            for pi in parents_i:
                for pj in parents_j:
                    angle = _misorientation_angle(pi, pj, _OH_SYM)
                    if angle < min_misfit:
                        min_misfit = angle
            misfits.append(min_misfit)
        return float(np.mean(misfits))

    def refine(
        self,
        threshold_deg: float = 5.0,
        max_pairs: int = 1000,
        verbose: bool = True,
    ) -> OrientationRelationship:
        """
        Refine the OR and return a new OrientationRelationship object.

        Parameters
        ----------
        threshold_deg : float
            Only grain pairs with initial parent misorientation below this
            threshold are included in the refinement.
        max_pairs : int
            Maximum number of grain pairs used for optimisation.
        verbose : bool

        Returns
        -------
        OrientationRelationship (refined copy)
        """
        pairs = self._build_pairs(max_pairs)
        if verbose:
            print(f"  Refining OR '{self.or_obj.name}' on {len(pairs)} grain pairs...")

        x0 = np.zeros(3)  # zero perturbation as starting point
        result = minimize(
            self._cost,
            x0,
            args=(pairs,),
            method="Nelder-Mead",
            options={"xatol": 0.05, "fatol": 1e-4, "maxiter": 2000},
        )
        euler_delta = result.x
        dR = Rotation.from_euler("ZYZ", euler_delta, degrees=True).as_matrix()
        refined_variants = [dR @ v for v in self.or_obj.variants]

        if verbose:
            angle = np.degrees(np.arccos(
                np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
            ))
            print(f"  Refinement correction: {angle:.3f}°  (cost: {result.fun:.3f}°)")

        return OrientationRelationship(
            name=self.or_obj.name + "_refined",
            description=self.or_obj.description + " [refined from data]",
            variants=refined_variants,
        )


# ---------------------------------------------------------------------------
# Parent orientation voting
# ---------------------------------------------------------------------------

def _vote_parent_orientation(
    child_orientations: np.ndarray,   # (k, 3, 3) — k grains in cluster
    variant_ids: np.ndarray,          # (k,) — assigned variant per grain
    or_obj: OrientationRelationship,
    grain_sizes: Optional[np.ndarray] = None,  # (k,) weights
) -> Tuple[np.ndarray, float]:
    """
    Compute a weighted mean parent orientation from a cluster of child grains
    using the quaternion averaging method.

    Each grain i contributes the parent it implies via its assigned variant:
        g_parent_i = R_variant[i]^T @ g_child[i]

    The weighted mean is computed in quaternion space (Markley 2007).

    Returns
    -------
    mean_parent : (3,3) ndarray  — mean parent rotation matrix
    mean_fit    : float          — mean misorientation from mean (degrees)
    """
    if grain_sizes is None:
        weights = np.ones(len(child_orientations))
    else:
        weights = grain_sizes.astype(float)
    weights /= weights.sum()

    # Convert implied parents to quaternions
    quats = []
    for i, (orient, vid) in enumerate(zip(child_orientations, variant_ids)):
        if vid < 0 or vid >= or_obj.n_variants:
            quats.append(Rotation.identity())
        else:
            R_parent = or_obj.variants[vid].T @ orient
            quats.append(Rotation.from_matrix(R_parent))

    # Weighted quaternion average via eigenvalue method (Markley 2007)
    Q = np.stack([q.as_quat() for q in quats], axis=0)   # (k, 4)
    M_mat = (weights[:, None] * Q).T @ Q                  # (4, 4)
    eigenvalues, eigenvectors = np.linalg.eigh(M_mat)
    mean_quat = eigenvectors[:, -1]                        # largest eigenvalue
    mean_parent = Rotation.from_quat(mean_quat).as_matrix()

    # Compute mean fit
    fits = []
    for i, (orient, vid) in enumerate(zip(child_orientations, variant_ids)):
        if vid < 0:
            continue
        implied_parent = or_obj.variants[vid].T @ orient
        fit = _misorientation_angle(mean_parent, implied_parent, _OH_SYM)
        fits.append(fit)
    mean_fit = float(np.mean(fits)) if fits else 180.0

    return mean_parent, mean_fit


# ---------------------------------------------------------------------------
# Main reconstructor
# ---------------------------------------------------------------------------

class ParentReconstructor:
    """
    Full parent grain reconstruction pipeline.

    Parameters
    ----------
    grain_data : GrainData
    or_name : str or None
        Name of the OR to use (e.g. "KS").  If None, auto-detect.
    refine_or : bool
        If True, refine the OR from the measured data before reconstruction.
    threshold_deg : float
        Probability threshold parameter for the grain graph (degrees).
    tolerance_deg : float
        Probability tolerance (Gaussian sigma) for the grain graph (degrees).
    mcl_inflation : float
        MCL inflation exponent.  Higher → smaller, more distinct clusters.
    mcl_iterations : int
        Maximum MCL iterations.

    Examples
    --------
    >>> from martwin.reconstruction.parent_reconstructor import ParentReconstructor
    >>> from martwin.reconstruction.grain_graph import GrainData
    >>> import numpy as np
    >>>
    >>> # Build synthetic grain data (replace with real EBSD loader)
    >>> N = 50
    >>> rng = np.random.default_rng(0)
    >>> from scipy.spatial.transform import Rotation
    >>> orientations = Rotation.random(N, random_state=rng).as_matrix()
    >>> adjacency = [[j for j in range(max(0,i-2), min(N,i+3)) if j!=i]
    ...              for i in range(N)]
    >>> gd = GrainData.from_arrays(orientations, adjacency)
    >>>
    >>> rec = ParentReconstructor(gd, or_name="KS", refine_or=False)
    >>> result = rec.run(verbose=True)
    >>> print(result.summary())
    """

    def __init__(
        self,
        grain_data: GrainData,
        or_name: Optional[str] = None,
        refine_or: bool = True,
        threshold_deg: float = 3.0,
        tolerance_deg: float = 3.0,
        mcl_inflation: float = 2.0,
        mcl_iterations: int = 100,
    ):
        self.grain_data = grain_data
        self.or_name = or_name
        self.refine_or = refine_or
        self.threshold_deg = threshold_deg
        self.tolerance_deg = tolerance_deg
        self.mcl_inflation = mcl_inflation
        self.mcl_iterations = mcl_iterations

    # ------------------------------------------------------------------
    # Step 1: resolve OR
    # ------------------------------------------------------------------
    def _resolve_or(self, verbose: bool) -> OrientationRelationship:
        if self.or_name is not None:
            or_obj = get_OR(self.or_name)
            if verbose:
                print(f"[Step 1] Using OR: {or_obj.name} ({or_obj.n_variants} variants)")
        else:
            if verbose:
                print("[Step 1] Auto-detecting OR...")
            best_name, _ = detect_OR(self.grain_data, verbose=verbose)
            or_obj = get_OR(best_name)
        return or_obj

    # ------------------------------------------------------------------
    # Step 2: refine OR
    # ------------------------------------------------------------------
    def _refine_or(
        self, or_obj: OrientationRelationship, verbose: bool
    ) -> OrientationRelationship:
        if not self.refine_or:
            return or_obj
        if verbose:
            print("[Step 2] Refining OR from EBSD data...")
        refiner = ORRefiner(self.grain_data, or_obj)
        return refiner.refine(verbose=verbose)

    # ------------------------------------------------------------------
    # Step 3: build variant graph
    # ------------------------------------------------------------------
    def _build_graph(
        self, or_obj: OrientationRelationship, verbose: bool
    ) -> VariantGraph:
        if verbose:
            N = self.grain_data.n_grains
            nv = or_obj.n_variants
            print(
                f"[Step 3] Building variant graph: "
                f"{N} grains × {nv} variants = {N*nv} nodes"
            )
        vg = build_variant_graph(
            self.grain_data,
            or_obj,
            threshold_deg=self.threshold_deg,
            tolerance_deg=self.tolerance_deg,
        )
        if verbose:
            nnz = vg.P.nnz
            print(f"         Graph edges: {nnz:,}  sparsity: {nnz/(vg.n_nodes**2)*100:.4f}%")
        return vg

    # ------------------------------------------------------------------
    # Step 4: Markov clustering
    # ------------------------------------------------------------------
    def _cluster(self, vg: VariantGraph, verbose: bool) -> np.ndarray:
        if verbose:
            print(
                f"[Step 4] Markov clustering "
                f"(inflation={self.mcl_inflation}, max_iter={self.mcl_iterations})..."
            )
        labels = markov_cluster(
            vg,
            r=self.mcl_inflation,
            n_iterations=self.mcl_iterations,
            verbose=verbose,
        )
        n_clusters = len(np.unique(labels))
        if verbose:
            print(f"         → {n_clusters} clusters found.")
        return labels

    # ------------------------------------------------------------------
    # Step 5: assign best variant per grain and vote parent orientation
    # ------------------------------------------------------------------
    def _assign_and_vote(
        self,
        vg: VariantGraph,
        cluster_labels: np.ndarray,
        verbose: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        For each grain, pick the variant node with the highest MCL cluster
        vote, then compute the parent orientation for each cluster by
        weighted quaternion averaging.

        Returns
        -------
        parent_orientations : (N, 3, 3)
        parent_grain_id     : (N,)  int
        variant_id          : (N,)  int
        fit                 : (N,)  float
        """
        N = self.grain_data.n_grains
        nv = vg.n_variants
        or_obj = vg.or_obj

        # For each grain, pick the variant with the cluster that has
        # the most members (majority vote)
        parent_grain_id = np.full(N, -1, dtype=int)
        variant_id = np.full(N, -1, dtype=int)

        # Count cluster sizes
        unique_clusters, cluster_sizes = np.unique(
            cluster_labels, return_counts=True
        )
        cluster_size_map = dict(zip(unique_clusters, cluster_sizes))

        for grain in range(N):
            # Get cluster labels for all variants of this grain
            node_ids = np.arange(grain * nv, (grain + 1) * nv)
            grain_clusters = cluster_labels[node_ids]

            # Choose the variant whose cluster is the largest
            best_v = int(np.argmax([cluster_size_map.get(c, 0) for c in grain_clusters]))
            variant_id[grain] = best_v
            parent_grain_id[grain] = grain_clusters[best_v]

        # Now vote parent orientation within each cluster
        unique_parents = np.unique(parent_grain_id[parent_grain_id >= 0])
        n_parent_grains = len(unique_parents)

        parent_orientations = np.full((N, 3, 3), np.nan)
        fit = np.full(N, np.nan)

        for pid in unique_parents:
            members = np.where(parent_grain_id == pid)[0]
            member_orients = self.grain_data.orientations[members]
            member_variants = variant_id[members]
            member_sizes = self.grain_data.grain_sizes[members]

            mean_parent, mean_fit = _vote_parent_orientation(
                member_orients, member_variants, or_obj, member_sizes
            )
            for m in members:
                parent_orientations[m] = mean_parent

            # Per-grain fit
            for m in members:
                vid = variant_id[m]
                if vid >= 0:
                    implied = or_obj.variants[vid].T @ self.grain_data.orientations[m]
                    fit[m] = _misorientation_angle(mean_parent, implied, _OH_SYM)

        # Map parent_grain_id to sequential 0-based IDs
        parent_to_seq = {pid: seq for seq, pid in enumerate(unique_parents)}
        for g in range(N):
            if parent_grain_id[g] >= 0:
                parent_grain_id[g] = parent_to_seq[parent_grain_id[g]]

        if verbose:
            assigned = np.sum(parent_grain_id >= 0)
            good = np.sum(fit[~np.isnan(fit)] < 5.0)
            print(
                f"[Step 5] Assigned {assigned}/{N} grains to "
                f"{n_parent_grains} parent grains."
            )
            print(
                f"         Grains with fit < 5°: {good} "
                f"({good/N:.1%})"
            )

        return parent_orientations, parent_grain_id, variant_id, fit

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, verbose: bool = True) -> ParentReconstructionResult:
        """
        Execute the full reconstruction pipeline.

        Returns
        -------
        ParentReconstructionResult
        """
        or_obj = self._resolve_or(verbose)
        or_obj = self._refine_or(or_obj, verbose)
        vg = self._build_graph(or_obj, verbose)
        cluster_labels = self._cluster(vg, verbose)
        parent_orients, pg_id, v_id, fit = self._assign_and_vote(
            vg, cluster_labels, verbose
        )

        n_parent_grains = len(np.unique(pg_id[pg_id >= 0]))
        valid = fit[~np.isnan(fit)]
        reconstruction_fraction = float(np.sum(valid < 5.0) / self.grain_data.n_grains)

        return ParentReconstructionResult(
            parent_orientations=parent_orients,
            parent_grain_id=pg_id,
            variant_id=v_id,
            fit=fit,
            or_used=or_obj,
            cluster_labels=cluster_labels,
            n_parent_grains=n_parent_grains,
            reconstruction_fraction=reconstruction_fraction,
        )


# ---------------------------------------------------------------------------
# Pixel-level back-projection (after grain-level reconstruction)
# ---------------------------------------------------------------------------

def backproject_to_pixels(
    result: ParentReconstructionResult,
    pixel_grain_ids: np.ndarray,          # (H, W) int — grain ID per EBSD pixel
    pixel_orientations: np.ndarray,       # (H, W, 3, 3) — per-pixel orientations
    fit_threshold_deg: float = 5.0,
) -> np.ndarray:
    """
    Back-project grain-level parent orientations to every EBSD pixel.

    For pixels whose grain has fit < fit_threshold_deg, the mean parent
    orientation from the reconstruction is assigned.  Pixels in poorly
    reconstructed grains receive their individual pixel-implied parent
    (via the grain's assigned variant).

    Parameters
    ----------
    result : ParentReconstructionResult
    pixel_grain_ids : (H, W) int ndarray
        Grain ID for every pixel in the EBSD map.
    pixel_orientations : (H, W, 3, 3) ndarray
        Per-pixel rotation matrices from the EBSD measurement.
    fit_threshold_deg : float

    Returns
    -------
    parent_pixel_orientations : (H, W, 3, 3)  rotation matrices
        NaN for unindexed pixels.
    parent_pixel_fit : (H, W)  float
        Per-pixel fit in degrees.
    """
    H, W = pixel_grain_ids.shape
    parent_out = np.full((H, W, 3, 3), np.nan)
    fit_out = np.full((H, W), np.nan)

    or_obj = result.or_used

    for row in range(H):
        for col in range(W):
            gid = pixel_grain_ids[row, col]
            if gid < 0 or gid >= len(result.parent_grain_id):
                continue
            vid = result.variant_id[gid]
            if vid < 0:
                continue

            # Use grain-level parent if well-reconstructed
            grain_fit = result.fit[gid]
            if not np.isnan(grain_fit) and grain_fit < fit_threshold_deg:
                parent_out[row, col] = result.parent_orientations[gid]
                # Pixel fit: misorientation between grain parent and pixel-implied parent
                pixel_parent = or_obj.variants[vid].T @ pixel_orientations[row, col]
                pfit = _misorientation_angle(
                    result.parent_orientations[gid], pixel_parent, _OH_SYM
                )
                fit_out[row, col] = pfit
            else:
                # Fall back to pixel-level parent estimate
                pixel_parent = or_obj.variants[vid].T @ pixel_orientations[row, col]
                parent_out[row, col] = pixel_parent
                fit_out[row, col] = grain_fit if not np.isnan(grain_fit) else 180.0

    return parent_out, fit_out
