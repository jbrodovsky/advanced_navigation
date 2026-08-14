"""Starter FastSLAM interfaces for the Victoria Park assignment.

These classes intentionally define contracts rather than a full solution. The
assignment asks students to implement the landmark EKFs, particle prediction,
per-particle data association, likelihood weighting, and resampling policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import LandmarkState


class LandmarkEKF:
    """Per-landmark EKF stored inside one FastSLAM particle.

    Expected invariant:
    - `mean_xy_m` is a 2-vector in the world frame.
    - `covariance` is a symmetric 2x2 landmark covariance.
    - prediction uses range-bearing geometry conditioned on the particle pose.
    - bearing innovations are wrapped before gating or updating.
    """

    def initialize(self, pose, measurement, measurement_covariance):
        """Initialize a landmark from one `[range, bearing]` observation.

        Students should use the inverse observation model and propagate
        measurement covariance through its Jacobian.
        """
        raise NotImplementedError("students should implement landmark initialization")

    def update(self, pose, measurement, measurement_covariance):
        """Apply a range-bearing EKF update conditioned on one particle pose.

        Students should compute innovation, Jacobian, innovation covariance,
        Kalman gain using `np.linalg.solve`, and a stable covariance update.
        """
        raise NotImplementedError("students should implement landmark updates")


@dataclass(slots=True)
class FastSLAMParticle:
    """One FastSLAM pose hypothesis and its conditionally independent map."""

    pose_xy_theta: np.ndarray
    log_weight: float
    landmarks: dict[int, LandmarkState] = field(default_factory=dict)
    trajectory: list[np.ndarray] = field(default_factory=list)


@dataclass(slots=True)
class FastSLAM:
    """Student-facing FastSLAM backend contract.

    Required behavior:
    - sample each particle pose from the bicycle motion model;
    - associate observations inside each particle's own landmark map;
    - initialize, update, reject, or deactivate landmarks;
    - update log weights from measurement likelihoods;
    - resample only when the effective sample size crosses the chosen threshold.
    """

    particles: list[FastSLAMParticle]
    process_noise: np.ndarray
    measurement_noise: np.ndarray

    def predict(self, speed_mps: float, steering_rad: float, dt: float) -> None:
        """Propagate every particle pose through the motion model."""
        raise NotImplementedError("students should implement the FastSLAM prediction step")

    def update(self, observations) -> None:
        """Associate observations and update particle maps and weights."""
        raise NotImplementedError("students should implement the FastSLAM update step")

    def resample(self) -> None:
        """Replace particles using a low-variance/systematic resampling method."""
        raise NotImplementedError("students should implement particle resampling")
