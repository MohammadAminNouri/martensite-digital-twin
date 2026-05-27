"""DAMASK connector placeholder for crystal-plasticity/mechanical response."""

from __future__ import annotations


def damask_status() -> dict:
    return {"connector": "DAMASK", "status": "stub", "needed": ["crystal plasticity law", "elastic constants", "slip/twin systems", "variant transformation strain", "mesh/grid"]}
