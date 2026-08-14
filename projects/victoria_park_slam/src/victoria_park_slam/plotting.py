"""Small plotting helpers for Victoria Park diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .types import TreeObservation
from .utils import scan_to_xy


def plot_scan_xy(scan_xy_m: np.ndarray, title: str):
    """Plot scanner-frame XY points from one laser scan."""

    fig, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(scan_xy_m[:, 0], scan_xy_m[:, 1], s=2)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    fig.tight_layout()
    return fig, axis


def plot_scan_with_observations(
    scan_ranges_m: np.ndarray,
    observations: list[TreeObservation],
    *,
    title: str = "Laser Scan With Tree Observations",
):
    """Plot one scan and overlay extracted range-bearing tree observations."""

    scan_xy_m = scan_to_xy(scan_ranges_m)
    fig, axis = plot_scan_xy(scan_xy_m, title)

    if observations:
        ranges = np.array([obs.range_m for obs in observations], dtype=np.float64)
        bearings = np.array([obs.bearing_rad for obs in observations], dtype=np.float64)
        obs_xy = np.column_stack([ranges * np.cos(bearings), ranges * np.sin(bearings)])
        axis.scatter(obs_xy[:, 0], obs_xy[:, 1], s=35, marker="x", label="tree candidates")
        axis.legend()

    return fig, axis


def plot_trajectory_and_landmarks(
    trajectory_xy_m: np.ndarray,
    landmarks_xy_m: np.ndarray,
    *,
    gps_xy_m: np.ndarray | None = None,
    title: str = "Estimated Trajectory and Landmark Map",
):
    """Plot an estimated vehicle trajectory with landmark positions."""

    trajectory = np.asarray(trajectory_xy_m, dtype=np.float64)
    landmarks = np.asarray(landmarks_xy_m, dtype=np.float64)

    fig, axis = plt.subplots(figsize=(8, 8))
    if trajectory.size:
        axis.plot(trajectory[:, 0], trajectory[:, 1], label="estimated trajectory")
    if landmarks.size:
        axis.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker="*", label="landmarks")
    if gps_xy_m is not None:
        gps = np.asarray(gps_xy_m, dtype=np.float64)
        axis.plot(gps[:, 0], gps[:, 1], color="0.5", alpha=0.7, label="GPS reference")

    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title)
    axis.set_xlabel("east [m]")
    axis.set_ylabel("north [m]")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    return fig, axis


def plot_effective_sample_size(step_indexes: np.ndarray, effective_counts: np.ndarray):
    """Plot effective particle count over time."""

    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot(step_indexes, effective_counts)
    axis.set_title("Effective Sample Size")
    axis.set_xlabel("update step")
    axis.set_ylabel("N_eff")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, axis


def plot_landmark_counts(step_indexes: np.ndarray, landmark_counts: np.ndarray):
    """Plot landmark map size over time."""

    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot(step_indexes, landmark_counts)
    axis.set_title("Landmark Count")
    axis.set_xlabel("update step")
    axis.set_ylabel("landmarks")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, axis
