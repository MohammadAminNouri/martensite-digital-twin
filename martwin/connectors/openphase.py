"""OpenPhase connector placeholder.

Future implementation should write OpenPhase input files for phase-field simulations
and parse phase fraction/microstructure outputs back into the twin.
"""

from __future__ import annotations


def openphase_status() -> dict:
    return {"connector": "OpenPhase", "status": "stub", "needed": ["phase-field parameters", "interface energy", "mobility", "elastic constants", "thermal history"]}
