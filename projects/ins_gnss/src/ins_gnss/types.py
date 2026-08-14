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

