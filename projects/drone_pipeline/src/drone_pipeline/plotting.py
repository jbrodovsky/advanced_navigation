"""Lightweight plotting helpers for student diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_position_xyz(times_s: np.ndarray, estimated_xyz: np.ndarray, truth_xyz: np.ndarray, title: str):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    labels = ["x", "y", "z"]
    for axis_idx, axis in enumerate(axes):
        axis.plot(times_s, truth_xyz[axis_idx], label="truth")
        axis.plot(times_s, estimated_xyz[axis_idx], label="estimate")
        axis.set_ylabel(f"{labels[axis_idx]} [m]")
    axes[0].set_title(title)
    axes[-1].set_xlabel("time [s]")
    axes[0].legend()
    fig.tight_layout()
    return fig, axes


def plot_rpy(times_s: np.ndarray, estimated_rpy: np.ndarray, truth_rpy: np.ndarray, title: str):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    labels = ["roll", "pitch", "yaw"]
    for axis_idx, axis in enumerate(axes):
        axis.plot(times_s, truth_rpy[axis_idx], label="truth")
        axis.plot(times_s, estimated_rpy[axis_idx], label="estimate")
        axis.set_ylabel(f"{labels[axis_idx]} [rad]")
    axes[0].set_title(title)
    axes[-1].set_xlabel("time [s]")
    axes[0].legend()
    fig.tight_layout()
    return fig, axes

