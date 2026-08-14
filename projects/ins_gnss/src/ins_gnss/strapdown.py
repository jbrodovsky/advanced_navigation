"""Curved-earth strapdown starter interfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class IMUCalibration:
    accel_bias: np.ndarray
    gyro_bias: np.ndarray
    accel_scale: np.ndarray
    gyro_scale: np.ndarray


class StrapdownNavigator:
    """Starter interface for the curved-earth mechanization assignment."""

    def __init__(self, initial_state_geodetic: dict[str, object], imu_params: IMUCalibration):
        self.lat = float(initial_state_geodetic["latitude"])
        self.lon = float(initial_state_geodetic["longitude"])
        self.alt = float(initial_state_geodetic["altitude"])
        self.v_ned = np.asarray(initial_state_geodetic["velocity_ned"], dtype=np.float64)
        self.q = np.asarray(initial_state_geodetic["attitude"], dtype=np.float64)
        self.imu_params = imu_params

        self.a = 6378137.0
        self.e2 = 6.69437999014132e-3
        self.omega_ie = 7.2921150e-5

    def integrate_imu(self, accel_raw: np.ndarray, gyro_raw: np.ndarray, dt: float) -> None:
        raise NotImplementedError("students should implement curved-earth strapdown integration")

    def get_meridian_radius(self) -> float:
        denom = 1.0 - self.e2 * np.sin(self.lat) ** 2
        return self.a * (1.0 - self.e2) / denom ** 1.5

    def get_prime_vertical_radius(self) -> float:
        denom = 1.0 - self.e2 * np.sin(self.lat) ** 2
        return self.a / np.sqrt(denom)

    def compute_gravity(self) -> float:
        sin_lat = np.sin(self.lat)
        sin2_lat = sin_lat**2
        g_surface = 9.7803253359 * (1.0 + 0.0053024 * sin2_lat - 0.0000058 * sin2_lat**2)
        return g_surface - 3.087e-6 * self.alt

