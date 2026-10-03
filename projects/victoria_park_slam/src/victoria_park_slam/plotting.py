"""Small plotting helpers for Victoria Park diagnostics.

Scan plots use the vehicle frame (x forward, y left, origin at the laser), so
"up" on the page is straight ahead and positive bearings appear on the left.
The functions return ``(figure, axis)`` and never call ``plt.show()``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .types import TreeObservation
from .utils import scan_to_xy


def plot_scan_xy(scan_xy_m: np.ndarray, title: str):
    """Plot vehicle-frame scan points ``[forward, left]`` with forward pointing up."""

    fig, axis = plt.subplots(figsize=(7, 6))
    # Plot left (y) on the horizontal axis, mirrored so the vehicle's left is
    # on the page's left, and forward (x) up the page.
    axis.scatter(-scan_xy_m[:, 1], scan_xy_m[:, 0], s=2, label="laser returns")
    axis.plot([0.0], [0.0], marker="^", color="black", ms=8, label="laser")
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title)
    axis.set_xlabel("right of the vehicle [m]  (= -y)")
    axis.set_ylabel("ahead of the vehicle [m]  (= x)")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, axis


def plot_scan_with_observations(
    scan_ranges_m: np.ndarray,
    observations: list[TreeObservation],
    *,
    title: str = "Laser Scan With Tree Observations",
):
    """Plot one scan and overlay extracted trunk observations (vehicle frame)."""

    scan_xy_m = scan_to_xy(scan_ranges_m, frame="vehicle")
    fig, axis = plot_scan_xy(scan_xy_m, title)

    if observations:
        ranges = np.array([obs.range_m for obs in observations], dtype=np.float64)
        bearings = np.array([obs.bearing_rad for obs in observations], dtype=np.float64)
        diameters = np.array([obs.diameter_m for obs in observations], dtype=np.float64)
        forward, left = ranges * np.cos(bearings), ranges * np.sin(bearings)
        axis.scatter(-left, forward, s=40, marker="x", color="C3", label="trunk centers")
        for x_page, y_page, diameter in zip(-left, forward, diameters):
            axis.add_patch(plt.Circle((x_page, y_page), diameter / 2.0, fill=False, color="C3", lw=1))
    axis.legend(loc="upper right")
    return fig, axis


def plot_trajectory_and_landmarks(
    trajectory_xy_m: np.ndarray,
    landmarks_xy_m: np.ndarray,
    *,
    gps_xy_m: np.ndarray | None = None,
    extra_tracks: dict[str, np.ndarray] | None = None,
    title: str = "Estimated Trajectory and Landmark Map",
):
    """Plot an estimated laser track with landmark positions (and optional GPS)."""

    trajectory = np.asarray(trajectory_xy_m, dtype=np.float64)
    landmarks = np.asarray(landmarks_xy_m, dtype=np.float64)

    fig, axis = plt.subplots(figsize=(8, 8))
    if gps_xy_m is not None:
        gps = np.asarray(gps_xy_m, dtype=np.float64)
        axis.plot(gps[:, 0], gps[:, 1], ".", ms=2, color="0.55", label="GPS (aligned frame)")
    for name, track in (extra_tracks or {}).items():
        track = np.asarray(track, dtype=np.float64)
        axis.plot(track[:, 0], track[:, 1], lw=1, alpha=0.8, label=name)
    if trajectory.size:
        axis.plot(trajectory[:, 0], trajectory[:, 1], lw=1.5, color="C0", label="estimate")
    if landmarks.size:
        axis.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker="*", color="C2", label="landmarks")

    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title)
    axis.set_xlabel("east [m]")
    axis.set_ylabel("north [m]")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize="small")
    fig.tight_layout()
    return fig, axis


def plot_effective_sample_size(step_indexes: np.ndarray, effective_counts: np.ndarray, num_particles: int | None = None):
    """Plot effective particle count over time."""

    fig, axis = plt.subplots(figsize=(8, 3.5))
    axis.plot(step_indexes, effective_counts, lw=1)
    if num_particles is not None:
        axis.axhline(num_particles, color="0.5", ls=":", lw=1, label="N")
        axis.legend(loc="upper right")
    axis.set_title("Effective Sample Size (before resampling)")
    axis.set_xlabel("scan index")
    axis.set_ylabel("N_eff")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, axis


def plot_landmark_counts(step_indexes: np.ndarray, landmark_counts: np.ndarray, confirmed_counts: np.ndarray | None = None):
    """Plot landmark map size over time (best particle)."""

    fig, axis = plt.subplots(figsize=(8, 3.5))
    axis.plot(step_indexes, landmark_counts, label="all (incl. tentative)")
    if confirmed_counts is not None:
        axis.plot(step_indexes, confirmed_counts, label="confirmed")
    axis.set_title("Landmark Count (highest-weight particle)")
    axis.set_xlabel("scan index")
    axis.set_ylabel("landmarks")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper left")
    fig.tight_layout()
    return fig, axis
