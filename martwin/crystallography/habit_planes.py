from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from martwin.core.rotations import normalize


@dataclass(frozen=True)
class HabitPlaneCandidate:
    parent_hkl: tuple[float, float, float]
    child_hkl: tuple[float, float, float] | None
    source: str
    confidence: str


NITI_CAYRON_NATURAL_HABIT_PROTOTYPE = HabitPlaneCandidate(
    parent_hkl=(1, 2, 0),
    child_hkl=(1, 0, 0),
    source="Cayron 2020 natural-OR interpretation reports (12)B2 // (10)B19′ style habit-plane candidate; verify index convention/sample metric.",
    confidence="prototype",
)


def trace_direction_on_surface(plane_normal_sample: Iterable[float], surface_normal_sample: Iterable[float] = (0, 0, 1)) -> np.ndarray:
    """Return unit trace direction of a plane on a surface in sample coordinates."""
    n = normalize(plane_normal_sample)
    s = normalize(surface_normal_sample)
    t = np.cross(n, s)
    if np.linalg.norm(t) < 1e-9:
        return np.array([np.nan, np.nan, np.nan])
    return normalize(t)
