"""Starter nonlinear-filter interfaces for the drone pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class EKFNavigator:
    initial_state: np.ndarray
    P0: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    history: list[dict[str, np.ndarray]] = field(default_factory=list)

    def predict(self, acc: np.ndarray, drpy: np.ndarray, dt: float) -> None:
        raise NotImplementedError("students should implement the EKF prediction step")

    def update(self, measurement: np.ndarray) -> None:
        raise NotImplementedError("students should implement the EKF update step")


@dataclass(slots=True)
class UKFNavigator:
    initial_state: np.ndarray
    P0: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0
    history: list[dict[str, np.ndarray]] = field(default_factory=list)

    def predict(self, acc: np.ndarray, drpy: np.ndarray, dt: float) -> None:
        raise NotImplementedError("students should implement the UKF prediction step")

    def update(self, measurement: np.ndarray) -> None:
        raise NotImplementedError("students should implement the UKF update step")


@dataclass(slots=True)
class ParticleFilterNavigator:
    initial_particles: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    weights: np.ndarray | None = None
    history: list[dict[str, np.ndarray]] = field(default_factory=list)

    def predict(self, acc: np.ndarray, drpy: np.ndarray, dt: float) -> None:
        raise NotImplementedError("students should implement the particle-filter prediction step")

    def update(self, measurement: np.ndarray) -> None:
        raise NotImplementedError("students should implement the particle-filter update step")

    def resample(self) -> None:
        raise NotImplementedError("students should implement low-variance or systematic resampling")

