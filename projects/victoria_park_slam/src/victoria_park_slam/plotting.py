"""Small plotting helpers for Victoria Park diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_scan_xy(scan_xy_m: np.ndarray, title: str):
    fig, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(scan_xy_m[:, 0], scan_xy_m[:, 1], s=2)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    fig.tight_layout()
    return fig, axis

