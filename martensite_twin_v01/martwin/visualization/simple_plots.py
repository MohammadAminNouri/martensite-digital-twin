from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_variant_ids(variant_ids: list[int] | np.ndarray, grid_shape: tuple[int, int], output_path: str | Path) -> None:
    arr = np.asarray(variant_ids).reshape(grid_shape)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(arr, interpolation="nearest")
    plt.colorbar(im, label="Variant ID")
    plt.title("Synthetic/identified martensite variant map")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
