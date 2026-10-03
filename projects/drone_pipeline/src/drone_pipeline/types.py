"""Typed records for the drone visual-inertial dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class DronePacket:
    """One onboard packet: AprilTag detections plus IMU readings.

    ``drpy`` holds the packet's rate signal, and ``rate_field`` records which raw
    field it came from. In ``studentdata1``-``7`` that is ``omg``, the body
    angular rate (it tracks the Vicon angular velocity closely). In
    ``studentdata0`` it is the file's ``drpy`` field, whose physical meaning is
    still unconfirmed: it may be Euler-angle rates or use another frame (GitHub
    issue #4). Check ``rate_field`` before treating ``drpy`` as a body rate.
    """

    id: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    p3: np.ndarray
    p4: np.ndarray
    t: float
    rpy: np.ndarray
    drpy: np.ndarray
    acc: np.ndarray
    p0: np.ndarray | None = None
    img: np.ndarray | None = None
    is_ready: bool | None = None
    rate_field: str = "omg"


@dataclass(slots=True)
class DroneDataset:
    path: Path
    vicon: np.ndarray
    time: np.ndarray
    packets: list[DronePacket]

    @property
    def packet_times(self) -> np.ndarray:
        return np.array([packet.t for packet in self.packets], dtype=np.float64)


@dataclass(slots=True)
class TruthSample:
    t: float
    position: np.ndarray
    rpy: np.ndarray
    velocity: np.ndarray
    angular_velocity: np.ndarray


@dataclass(slots=True)
class DroneParameters:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    translation_m: np.ndarray
    yaw_offset_rad: float
    tag_ids: np.ndarray
    T_camera_body: np.ndarray


@dataclass(slots=True)
class PoseObservation:
    t: float
    position: np.ndarray
    rpy: np.ndarray
    reprojection_error: float
    num_points: int

