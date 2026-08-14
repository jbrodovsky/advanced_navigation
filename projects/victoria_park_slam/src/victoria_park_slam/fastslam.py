"""Starter FastSLAM interfaces for the Victoria Park assignment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import LandmarkState


class LandmarkEKF:
    """Per-landmark EKF stored inside one FastSLAM particle."""

    def initialize(self, pose, measurement, measurement_covariance):
        raise NotImplementedError("students should implement landmark initialization")

    def update(self, pose, measurement, measurement_covariance):
        raise NotImplementedError("students should implement landmark updates")


@dataclass(slots=True)
class FastSLAMParticle:
    pose_xy_theta: np.ndarray
    weight: float
    landmarks: dict[int, LandmarkState] = field(default_factory=dict)
    trajectory: list[np.ndarray] = field(default_factory=list)


@dataclass(slots=True)
class FastSLAM:
    particles: list[FastSLAMParticle]
    process_noise: np.ndarray
    measurement_noise: np.ndarray

    def predict(self, speed_mps: float, steering_rad: float, dt: float) -> None:
        raise NotImplementedError("students should implement the FastSLAM prediction step")

    def update(self, observations) -> None:
        raise NotImplementedError("students should implement the FastSLAM update step")

    def resample(self) -> None:
        raise NotImplementedError("students should implement particle resampling")

