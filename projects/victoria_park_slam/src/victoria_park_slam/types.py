"""Typed records for the Victoria Park assignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class VictoriaParkDataset:
    root: Path
    laser_ranges_m: np.ndarray
    laser_time_s: np.ndarray
    speed_mps: np.ndarray
    steering_rad: np.ndarray
    odometry_time_s: np.ndarray
    gps_lat_m: np.ndarray | None = None
    gps_lon_m: np.ndarray | None = None
    gps_time_s: np.ndarray | None = None


@dataclass(slots=True)
class TreeObservation:
    range_m: float
    bearing_rad: float
    diameter_m: float


@dataclass(slots=True)
class LandmarkState:
    mean_xy_m: np.ndarray
    covariance: np.ndarray
    observation_count: int = 0


@dataclass(slots=True)
class FastSLAMParticleState:
    pose_xy_theta: np.ndarray
    weight: float
    landmarks: dict[int, LandmarkState] = field(default_factory=dict)

