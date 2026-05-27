"""pycalphad connector placeholder."""

from __future__ import annotations

from martwin.thermodynamics.engine import pycalphad_available


def status() -> dict:
    return {"connector": "pycalphad", "available_in_environment": pycalphad_available(), "needed": ["TDB database", "components", "phases", "conditions"]}
