"""
martwin.reconstruction
======================
Parent phase reconstruction from EBSD orientation maps of martensitic
microstructures.

Public API
----------
from martwin.reconstruction import (
    OR_REGISTRY,          # dict of all built-in ORs
    get_OR,               # fetch one OR by name
    GrainData,            # grain data container
    ParentReconstructor,  # main reconstruction class
    ParentReconstructionResult,
    ORRefiner,            # OR refinement from data
    detect_OR,            # auto-detect best OR
    backproject_to_pixels,
)

Quick start
-----------
>>> from martwin.reconstruction import ParentReconstructor, GrainData
>>> import numpy as np
>>> from scipy.spatial.transform import Rotation
>>>
>>> # Replace this synthetic data with your real EBSD loader output
>>> N = 100
>>> rng = np.random.default_rng(42)
>>> orientations = Rotation.random(N, random_state=rng).as_matrix()
>>> adjacency = [[j for j in range(max(0,i-3), min(N,i+4)) if j!=i]
...              for i in range(N)]
>>> gd = GrainData.from_arrays(orientations, adjacency)
>>>
>>> rec = ParentReconstructor(gd, or_name="KS", refine_or=False,
...                           threshold_deg=3.0, mcl_inflation=2.0)
>>> result = rec.run(verbose=True)
>>> print(result.summary())
"""

from .orientation_relationships import OR_REGISTRY, get_OR, OrientationRelationship
from .grain_graph import GrainData, VariantGraph, build_variant_graph, markov_cluster
from .parent_reconstructor import (
    ParentReconstructor,
    ParentReconstructionResult,
    ORRefiner,
    detect_OR,
    backproject_to_pixels,
)

__all__ = [
    "OR_REGISTRY",
    "get_OR",
    "OrientationRelationship",
    "GrainData",
    "VariantGraph",
    "build_variant_graph",
    "markov_cluster",
    "ParentReconstructor",
    "ParentReconstructionResult",
    "ORRefiner",
    "detect_OR",
    "backproject_to_pixels",
]
