from __future__ import annotations

from dataclasses import dataclass, field

from martwin.core.lattice import Lattice


@dataclass(frozen=True)
class SteelMaterial:
    composition_wt: dict[str, float] = field(default_factory=dict)
    prior_austenite_grain_size_um: float | None = None
    austenitization_C: float | None = None
    cooling_rate_C_s: float | None = None
    fcc_lattice: Lattice = Lattice(3.59, 3.59, 3.59)
    bcc_lattice: Lattice = Lattice(2.87, 2.87, 2.87)

    def critical_gaps(self) -> list[str]:
        gaps = []
        if not self.composition_wt:
            gaps.append("full steel composition")
        if self.prior_austenite_grain_size_um is None:
            gaps.append("prior austenite grain size")
        if self.austenitization_C is None:
            gaps.append("austenitization temperature/time")
        if self.cooling_rate_C_s is None:
            gaps.append("cooling curve or cooling rate")
        return gaps
