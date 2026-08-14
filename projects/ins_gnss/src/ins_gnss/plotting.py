"""Small plotting helpers for INS/GNSS diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_covariance_diagonal(times_s: np.ndarray, covariance_diagonals: np.ndarray, title: str):
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(times_s, covariance_diagonals)
    axis.set_title(title)
    axis.set_xlabel("time [s]")
    axis.set_ylabel("diagonal covariance")
    fig.tight_layout()
    return fig, axis

