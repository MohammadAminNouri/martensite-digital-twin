from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NiTiTransformationTemperatures:
    Ms_C: float | None = None
    Mf_C: float | None = None
    As_C: float | None = None
    Af_C: float | None = None

    def gaps(self) -> list[str]:
        return [k for k, v in {"Ms": self.Ms_C, "Mf": self.Mf_C, "As": self.As_C, "Af": self.Af_C}.items() if v is None]


def linear_cooling_fraction(T_C: float, temps: NiTiTransformationTemperatures) -> float | None:
    """Simple DSC-calibrated cooling fraction for B2→B19′.

    Returns None if Ms/Mf are unknown. This is a placeholder until fitted DSC or
    thermomechanical hysteresis models are supplied.
    """
    if temps.Ms_C is None or temps.Mf_C is None:
        return None
    if T_C >= temps.Ms_C:
        return 0.0
    if T_C <= temps.Mf_C:
        return 1.0
    return (temps.Ms_C - T_C) / (temps.Ms_C - temps.Mf_C)


def linear_heating_fraction_austenite(T_C: float, temps: NiTiTransformationTemperatures) -> float | None:
    """Simple heating fraction for B19′→B2."""
    if temps.As_C is None or temps.Af_C is None:
        return None
    if T_C <= temps.As_C:
        return 0.0
    if T_C >= temps.Af_C:
        return 1.0
    return (T_C - temps.As_C) / (temps.Af_C - temps.As_C)
