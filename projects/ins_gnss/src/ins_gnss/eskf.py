"""Starter ESKF interface for the INS/GNSS assignment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .strapdown import StrapdownNavigator


@dataclass(slots=True)
class INSGNSSFilter:
    navigator: StrapdownNavigator
    initial_covariance: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    innovations: list[np.ndarray] = field(default_factory=list)

    def predict_with_imu(self, accel: np.ndarray, gyro: np.ndarray, dt: float) -> None:
        raise NotImplementedError("students should implement the ESKF prediction step")

    def update_with_gnss(self, gnss_position: np.ndarray, gnss_velocity: np.ndarray | None = None) -> None:
        raise NotImplementedError("students should implement the ESKF update step")

