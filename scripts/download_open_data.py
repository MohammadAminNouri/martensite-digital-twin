"""Helper skeleton for future open-data ingestion.

This script intentionally does not auto-download large datasets by default.
Read each source license first, then add explicit download commands.
"""
from __future__ import annotations

from pathlib import Path
import csv

SOURCES = [
    {"name": "zenodo_8348372", "url": "https://zenodo.org/records/8348372", "purpose": "steel in-situ EBSD/PAG benchmark"},
    {"name": "zenodo_10469461", "url": "https://zenodo.org/records/10469461", "purpose": "steel PAG size/hardness/EDS/Thermo-Calc benchmark"},
    {"name": "orix_data", "url": "https://github.com/pyxem/orix-data", "purpose": "orientation mapping examples"},
]


def main() -> None:
    out = Path("data/open_data_manifest/download_targets.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "url", "purpose"])
        writer.writeheader()
        writer.writerows(SOURCES)
    print(f"Wrote {out}. Review licenses and dataset sizes before downloading.")


if __name__ == "__main__":
    main()
