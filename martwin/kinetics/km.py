from __future__ import annotations

import math


def koistinen_marburger_fraction(T_C: float, Ms_C: float, alpha: float = 0.011) -> float:
    """Koistinen-Marburger martensite fraction for steel, below Ms.

    f = 1 - exp[-alpha*(Ms - T)] for T < Ms; otherwise 0.
    alpha often needs alloy-specific calibration.
    """
    if T_C >= Ms_C:
        return 0.0
    return max(0.0, min(1.0, 1.0 - math.exp(-alpha * (Ms_C - T_C))))


def km_curve(temperatures_C: list[float], Ms_C: float, alpha: float = 0.011) -> list[float]:
    return [koistinen_marburger_fraction(T, Ms_C, alpha) for T in temperatures_C]
