from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_open_data_manifest(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def list_download_candidates(df: pd.DataFrame, material_system: str | None = None) -> pd.DataFrame:
    out = df.copy()
    if material_system:
        mask = out["material_system"].str.contains(material_system, case=False, na=False)
        out = out[mask]
    return out[["name", "material_system", "data_types", "url_or_doi", "license_or_access", "priority"]]
