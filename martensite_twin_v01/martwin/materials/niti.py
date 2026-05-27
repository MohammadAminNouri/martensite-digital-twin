from __future__ import annotations

from dataclasses import dataclass

from martwin.core.lattice import Lattice


@dataclass(frozen=True)
class NiTiMaterial:
    ni_at_percent: float | None = None
    ti_at_percent: float | None = None
    oxygen_wt_percent: float | None = None
    process_route: str = "unknown"
    b2_lattice: Lattice = Lattice(3.015, 3.015, 3.015)
    b19prime_lattice: Lattice = Lattice(a=2.889, b=4.120, c=4.622, beta=96.8)

    def critical_gaps(self) -> list[str]:
        gaps = []
        if self.ni_at_percent is None or self.ti_at_percent is None:
            gaps.append("exact Ni/Ti atomic ratio")
        if self.oxygen_wt_percent is None:
            gaps.append("oxygen/carbon contamination level")
        if self.process_route == "unknown":
            gaps.append("process route / thermal history")
        return gaps
