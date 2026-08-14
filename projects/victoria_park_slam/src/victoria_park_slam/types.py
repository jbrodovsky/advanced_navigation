"""Typed records for the Victoria Park FastSLAM assignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class VictoriaParkDataset:
    """Loaded Victoria Park data with units normalized for the assignment."""

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
class MotionModelConfig:
    """Configuration for the planar bicycle prediction model.

    `control_noise_covariance` is the 2x2 covariance for additive noise on
    `[speed_mps, steering_rad]` before applying the deterministic transition.
    """

    wheelbase_m: float = 2.83
    control_noise_covariance: np.ndarray = field(
        default_factory=lambda: np.diag([0.25**2, np.deg2rad(2.0) ** 2])
    )


@dataclass(slots=True)
class MeasurementModelConfig:
    """Configuration for range-bearing tree observations."""

    range_std_m: float = 0.50
    bearing_std_rad: float = float(np.deg2rad(3.0))
    gate_threshold: float = 5.99
    max_range_m: float = 75.0

    @property
    def covariance(self) -> np.ndarray:
        return np.diag([self.range_std_m**2, self.bearing_std_rad**2])


class LandmarkTrackState(StrEnum):
    """Simple lifecycle labels for per-particle landmark tracks."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    INACTIVE = "inactive"


@dataclass(slots=True)
class TreeObservation:
    """Range-bearing observation of one tree candidate in the scanner frame."""

    range_m: float
    bearing_rad: float
    diameter_m: float
    num_points: int | None = None


@dataclass(slots=True)
class LandmarkState:
    """Estimated 2D landmark state conditioned on one particle path."""

    mean_xy_m: np.ndarray
    covariance: np.ndarray
    observation_count: int = 0
    track_state: LandmarkTrackState = LandmarkTrackState.TENTATIVE


@dataclass(slots=True)
class FastSLAMParticleState:
    """Serializable particle state for diagnostics and tests."""

    pose_xy_theta: np.ndarray
    log_weight: float
    landmarks: dict[int, LandmarkState] = field(default_factory=dict)


@dataclass(slots=True)
class ScanFeatureDiagnostics:
    """Per-scan summary for laser feature extraction diagnostics."""

    scan_index: int
    timestamp_s: float
    valid_returns: int
    candidate_count: int
    rejected_segments: int


@dataclass(slots=True)
class AssociationDiagnostics:
    """Per-scan summary of association decisions."""

    scan_index: int
    accepted: int
    new_landmarks: int
    rejected: int


@dataclass(slots=True)
class WeightDiagnostics:
    """Particle-weight summary for one FastSLAM update."""

    step_index: int
    effective_sample_size: float
    resampled: bool


@dataclass(slots=True)
class MapDiagnostics:
    """Map-size summary for one FastSLAM update."""

    step_index: int
    landmark_count: int
    confirmed_landmark_count: int
