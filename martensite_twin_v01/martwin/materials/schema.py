from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExperimentRecord:
    sample_id: str
    material_system: str
    composition: dict[str, float] = field(default_factory=dict)
    composition_units: str = "unknown"
    process: dict[str, Any] = field(default_factory=dict)
    heat_treatment: dict[str, Any] = field(default_factory=dict)
    transformation_temperatures_C: dict[str, float | None] = field(default_factory=lambda: {"Ms": None, "Mf": None, "As": None, "Af": None})
    files: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def missing_critical_fields(self) -> list[str]:
        missing = []
        if not self.composition:
            missing.append("composition")
        for key in ["Ms", "Mf", "As", "Af"]:
            if self.transformation_temperatures_C.get(key) is None:
                missing.append(f"transformation_temperatures_C.{key}")
        if "ebsd" not in self.files and "tkd" not in self.files:
            missing.append("files.ebsd_or_tkd")
        return missing
