"""Typed records for the INS/GNSS project (Project 3).

Conventions (book-wide, @sec-frames-conventions): NED navigation frame,
latitude/longitude in radians, ellipsoidal height in meters, Hamilton
scalar-first quaternion ``q_b^n`` with ``C(q_b^n) = C_b^n``, biases in the
body frame, errors are true minus estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class IMUSample:
    """One IMU sample: specific force and angular rate on the sensor axes.

    For the MEMS dataset the axes are the phone's (Android sensor frame: ``x``
    right, ``y`` toward the top of the screen, ``z`` out of the screen). Use
    :func:`~ins_gnss.datasets.estimate_mount_rotation` and
    :func:`~ins_gnss.datasets.rotate_imu_samples` to express them on
    vehicle forward-right-down axes.
    """

    time_s: float
    accel_mps2: np.ndarray
    gyro_rps: np.ndarray


@dataclass(slots=True)
class GNSSSample:
    """One GNSS fix as reported by the receiver (loosely coupled: a position solution).

    ``altitude_m`` is the receiver's height (treated as ellipsoidal). The
    accuracy fields are the receiver's own one-sigma-ish estimates (Android
    reports a 68% radius for ``horizontal_accuracy_m``); they are a starting
    point for ``R``, not gospel. ``bearing_rad`` is the course over ground,
    clockwise from north, and is meaningless when the speed is near zero.
    ``velocity_ned_mps`` is a full NED velocity when a source provides one
    (the conventional CSV layout); the MEMS data has speed and bearing only,
    see :meth:`velocity_ne_mps`.
    """

    time_s: float
    latitude_rad: float
    longitude_rad: float
    altitude_m: float
    velocity_ned_mps: np.ndarray | None = None
    speed_mps: float | None = None
    bearing_rad: float | None = None
    horizontal_accuracy_m: float | None = None
    vertical_accuracy_m: float | None = None
    speed_accuracy_mps: float | None = None
    bearing_accuracy_rad: float | None = None

    def velocity_ne_mps(self) -> np.ndarray | None:
        """Horizontal velocity ``[v_N, v_E]`` from speed and bearing (``None`` if unavailable)."""
        if self.velocity_ned_mps is not None:
            return np.asarray(self.velocity_ned_mps, dtype=np.float64)[:2]
        if self.speed_mps is None or self.bearing_rad is None:
            return None
        if not (np.isfinite(self.speed_mps) and np.isfinite(self.bearing_rad)):
            return None
        return self.speed_mps * np.array([np.cos(self.bearing_rad), np.sin(self.bearing_rad)])


@dataclass(slots=True)
class INSGNSSDataset:
    """A conventional ``imu.csv`` / ``gnss.csv`` dataset (see ``load_ins_gnss_data``)."""

    root: Path
    imu_samples: list[IMUSample]
    gnss_samples: list[GNSSSample]
    ground_truth: np.ndarray | None = None
    metadata: dict[str, object] | None = None


@dataclass(slots=True)
class MEMSReferenceState:
    """One row of the dataset's baseline solution (a GNSS-aided reference, not truth).

    ``attitude_rpy_rad`` is copied verbatim from the file. It is **not** in
    the book's convention: read as Z-Y-X angles of ``C_b^n`` with ``b`` the
    phone axes it puts gravity upside down. Do not use it as attitude truth
    (see the Project 3 handout, @sec-project3-data).
    """

    time_s: float
    latitude_rad: float
    longitude_rad: float
    altitude_m: float
    velocity_ned_mps: np.ndarray
    attitude_rpy_rad: np.ndarray
    accel_bias_mps2: np.ndarray | None = None
    gyro_bias_rps: np.ndarray | None = None
    covariance: dict[str, float] | None = None


@dataclass(slots=True)
class MEMSNavTrial:
    """One trial of the MEMS navigation dataset.

    All ``time_s`` values are seconds since one epoch, the first input
    timestamp (``metadata["epoch_utc"]``), including the reference states.
    """

    name: str
    root: Path
    input_path: Path
    baseline_path: Path
    imu_samples: list[IMUSample]
    gnss_samples: list[GNSSSample]
    reference_states: list[MEMSReferenceState]
    metadata: dict[str, object] | None = None
    auxiliary: dict[str, np.ndarray] | None = None
    """Per-row arrays of the input columns the filter does not use by default
    (``time_s``, ``pressure_hpa``, ``relative_altitude_m``, ``mag_ut`` (N, 3),
    ``gravity_mps2`` (N, 3), ``orientation_q`` (N, 4) scalar first), NaN where empty.
    Handy for the barometer and magnetometer extensions."""


@dataclass
class NavState:
    """Nominal navigation state of the ESKF (@sec-ins-gnss-error-state-formulation).

    ``[L, lambda, h, v^n, q_b^n, b_a, b_g]``: geodetic latitude and longitude
    [rad], ellipsoidal height [m], NED velocity [m/s], attitude quaternion
    ``q_b^n`` (scalar first), accelerometer bias [m/s^2] and gyro bias
    [rad/s] on body axes. Biases are *subtracted* from the raw measurements:
    ``f^b = f_meas - b_a``, ``omega_ib^b = omega_meas - b_g``.
    """

    latitude_rad: float
    longitude_rad: float
    height_m: float
    velocity_ned_mps: np.ndarray = field(default_factory=lambda: np.zeros(3))
    q_bn: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    accel_bias_mps2: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_bias_rps: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        self.latitude_rad = float(self.latitude_rad)
        self.longitude_rad = float(self.longitude_rad)
        self.height_m = float(self.height_m)
        self.velocity_ned_mps = np.array(self.velocity_ned_mps, dtype=np.float64).reshape(3)
        self.q_bn = np.array(self.q_bn, dtype=np.float64).reshape(4)
        self.accel_bias_mps2 = np.array(self.accel_bias_mps2, dtype=np.float64).reshape(3)
        self.gyro_bias_rps = np.array(self.gyro_bias_rps, dtype=np.float64).reshape(3)

    def copy(self) -> NavState:
        """Deep copy (the arrays are copied too)."""
        return NavState(
            self.latitude_rad,
            self.longitude_rad,
            self.height_m,
            self.velocity_ned_mps.copy(),
            self.q_bn.copy(),
            self.accel_bias_mps2.copy(),
            self.gyro_bias_rps.copy(),
        )

    @property
    def position(self) -> np.ndarray:
        """``[L, lambda, h]``."""
        return np.array([self.latitude_rad, self.longitude_rad, self.height_m])


@dataclass
class IMUNoise:
    """Continuous-time IMU noise model in SI units (@sec-ins-errors-from-imu-specs).

    * ``accel_vrw``: velocity random walk ``N_a`` [m/s/sqrt(s)] (= white
      accelerometer noise density in m/s^2/sqrt(Hz)).
    * ``gyro_arw``: angle random walk ``N_g`` [rad/sqrt(s)].
    * ``accel_bias_rw``: accelerometer bias random walk ``K_a`` [m/s^2/sqrt(s)].
    * ``gyro_bias_rw``: gyro bias random walk ``K_g`` [rad/s/sqrt(s)].

    Datasheets quote ARW in deg/sqrt(h) and VRW in m/s/sqrt(h): divide by 60
    (and convert degrees to radians) to get per-sqrt(s) values. Bias
    instability (deg/h, mg) is not a random walk; @sec-ins-errors-from-imu-specs
    shows how to turn it into one over the horizon you care about.
    The defaults equal the starter ``run_all.TUNING``: a deliberately rough,
    pessimistic guess for a 1 Hz smartphone IMU, not a tuned answer.
    """

    accel_vrw: float = 0.1
    gyro_arw: float = float(np.deg2rad(0.1))
    accel_bias_rw: float = 1.0e-3
    gyro_bias_rw: float = float(np.deg2rad(0.01))

    def psd(self) -> np.ndarray:
        """Diagonal of ``Q_c`` for the noise vector ``[w_a, w_g, w_ba, w_bg]`` (12 values)."""
        return np.repeat(
            np.array([self.accel_vrw, self.gyro_arw, self.accel_bias_rw, self.gyro_bias_rw]) ** 2,
            3,
        )
