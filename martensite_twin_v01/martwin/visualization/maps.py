from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_variant_map(df: pd.DataFrame, value_col: str = "variant_id", title: str = "Variant map"):
    """Create a matplotlib figure for x/y EBSD-like maps."""
    if not {"x", "y", value_col}.issubset(df.columns):
        raise ValueError(f"DataFrame must contain x, y, and {value_col}")
    x_vals = sorted(df["x"].unique())
    y_vals = sorted(df["y"].unique())
    x_index = {v: i for i, v in enumerate(x_vals)}
    y_index = {v: i for i, v in enumerate(y_vals)}
    arr = np.full((len(y_vals), len(x_vals)), np.nan)
    for row in df.itertuples(index=False):
        data = row._asdict()
        arr[y_index[data["y"]], x_index[data["x"]]] = data[value_col]
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    im = ax.imshow(arr, origin="lower", interpolation="nearest", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_col)
    fig.tight_layout()
    return fig
