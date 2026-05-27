"""MTEX connector notes.

The Python core can export CSV/Mat files for MTEX and import MTEX results later.
For serious parent-grain reconstruction, compare this package's prototype results
against MTEX parentGrainReconstructor and ORTools.
"""

from __future__ import annotations

from pathlib import Path


def write_mtex_todo(output_path: str | Path) -> None:
    Path(output_path).write_text(
        "% TODO: generated placeholder for MTEX validation.\n"
        "% Load EBSD, define crystal symmetries, refine OR, run parentGrainReconstructor.\n",
        encoding="utf-8",
    )
