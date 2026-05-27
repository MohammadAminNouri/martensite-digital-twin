from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ThermodynamicResult:
    phases: dict[str, float]
    driving_forces: dict[str, float]
    temperature_C: float
    confidence: str
    notes: list[str]


def pycalphad_available() -> bool:
    try:
        import pycalphad  # noqa: F401
        return True
    except Exception:
        return False


def placeholder_thermodynamics(temperature_C: float, known_phases: list[str] | None = None) -> ThermodynamicResult:
    return ThermodynamicResult(
        phases={p: float("nan") for p in (known_phases or [])},
        driving_forces={},
        temperature_C=temperature_C,
        confidence="not_computed",
        notes=[
            "No thermodynamic database supplied. Connect pycalphad or Thermo-Calc TC-Python and provide a licensed/open TDB.",
            "Do not infer phase stability from this placeholder.",
        ],
    )


def run_pycalphad_stub(*args: Any, **kwargs: Any) -> ThermodynamicResult:
    if not pycalphad_available():
        return placeholder_thermodynamics(kwargs.get("temperature_C", 25.0), kwargs.get("phases"))
    raise NotImplementedError("Add database path, components, phases, conditions, and phase models for your alloy system.")
