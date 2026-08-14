"""Typed records for the INS/GNSS assignment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class IMUSample:
    time_s: float
    accel_mps2: np.ndarray
    gyro_rps: np.ndarray


@dataclass(slots=True)
class GNSSSample:
    time_s: float
    latitude_rad: float
    longitude_rad: float
    altitude_m: float
    velocity_ned_mps: np.ndarray | None = None


@dataclass(slots=True)
class INSGNSSDataset:
    root: Path
    imu_samples: list[IMUSample]
    gnss_samples: list[GNSSSample]
    ground_truth: np.ndarray | None = None
    metadata: dict[str, object] | None = None


@dataclass(slots=True)
class MEMSReferenceState:
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
    name: str
    root: Path
    input_path: Path
    baseline_path: Path
    imu_samples: list[IMUSample]
    gnss_samples: list[GNSSSample]
    reference_states: list[MEMSReferenceState]
    metadata: dict[str, object] | None = None
