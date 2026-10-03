"""Typed records for the drone visual-inertial dataset.

Frames used throughout the package (see the Project 1 handout and the PnP
chapter of the course notes):

* ``w``: the AprilTag-mat world frame. ``x`` runs down the mat, ``y`` to the
  right, ``z`` up out of the mat. The origin is the top-left corner of tag 0.
* ``b``: the drone body (IMU) frame, ``z`` up.
* ``c``: the optical camera frame (``x`` right, ``y`` down, ``z`` along the
  optical axis, which points down at the mat).

Attitudes are roll, pitch, yaw in the dataset's Z-X-Y sequence:
``C_world_body = Rz(yaw) @ Rx(roll) @ Ry(pitch)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class DronePacket:
    """One onboard packet: AprilTag detections plus IMU readings.

    Attributes
    ----------
    id:
        AprilTag IDs detected in this frame, shape ``(n_tags,)``. Empty when no
        tag is visible.
    p0, p1, p2, p3, p4:
        Pixel coordinates of the detections, each shape ``(2, n_tags)`` with
        row 0 = ``u`` (column) and row 1 = ``v`` (row). ``p1``..``p4`` are the
        bottom-left, bottom-right, top-right and top-left tag corners; ``p0``
        is the tag centre (the mean of the four corners). The corners are
        **already undistorted**: use zero distortion when you pass them to
        ``cv2.solvePnP``. ``p0`` is absent (``None``) in ``studentdata0``.
    t:
        Packet timestamp in seconds, on the same clock as the Vicon ``time``.
    rpy:
        The drone's *onboard* roll, pitch, yaw estimate (Z-X-Y). Roll and pitch
        agree with Vicon to about a degree; yaw has an arbitrary offset. Use it
        for sanity checks and for comparing initializations, never as a filter
        measurement.
    omg:
        Gyroscope body angular rate in rad/s, shape ``(3,)``. In
        ``studentdata0`` this slot holds that file's ``drpy`` field, whose
        physical meaning is unresolved (GitHub issue #4); check ``rate_field``.
    acc:
        Accelerometer specific force in the body frame, m/s^2, shape ``(3,)``.
        It reads about ``+9.8`` on ``z`` in hover.
    img:
        The 240 x 376 grayscale camera image (``uint8``), when present.
    is_ready:
        Onboard ready flag (``studentdata1``-``7`` only; always true there).
    rate_field:
        Name of the raw field ``omg`` was read from: ``"omg"`` or ``"drpy"``.
    """

    id: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    p3: np.ndarray
    p4: np.ndarray
    t: float
    rpy: np.ndarray
    omg: np.ndarray
    acc: np.ndarray
    p0: np.ndarray | None = None
    img: np.ndarray | None = None
    is_ready: bool | None = None
    rate_field: str = "omg"

    @property
    def drpy(self) -> np.ndarray:
        """Read-only alias for ``omg`` kept for backward compatibility."""
        return self.omg

    @property
    def num_tags(self) -> int:
        """Number of AprilTags detected in this packet."""
        return int(np.asarray(self.id).size)


@dataclass(slots=True)
class DroneDataset:
    """One ``studentdata<i>.mat`` file.

    ``vicon`` is ``12 x N``: rows 0:3 position (m, world frame), 3:6 roll,
    pitch, yaw (rad, Z-X-Y), 6:9 velocity (m/s, world frame), 9:12 body angular
    rate (rad/s). ``time`` holds the ``N`` Vicon timestamps.
    """

    path: Path
    vicon: np.ndarray
    time: np.ndarray
    packets: list[DronePacket]

    @property
    def packet_times(self) -> np.ndarray:
        """Packet timestamps as one float array."""
        return np.array([packet.t for packet in self.packets], dtype=np.float64)


@dataclass(slots=True)
class TruthSample:
    """Vicon truth interpolated onto one timestamp.

    ``valid`` is false when ``t`` lies outside the Vicon time span. The other
    fields are then NaN: drop those samples instead of comparing against them.
    """

    t: float
    position: np.ndarray
    rpy: np.ndarray
    velocity: np.ndarray
    angular_velocity: np.ndarray
    valid: bool = True


@dataclass(slots=True)
class DroneParameters:
    """Camera and rig parameters parsed from ``parameters.txt``.

    ``dist_coeffs`` are the lens coefficients ``[k1, k2, p1, p2, k3]`` exactly
    as the file states them. The packet corner detections are already
    undistorted, so pass ``np.zeros(5)`` to ``cv2.solvePnP`` rather than these.

    ``T_camera_body`` is the 4 x 4 transform that maps body (IMU) coordinates
    into camera coordinates, ``p_c = T_camera_body @ [p_b, 1]``. Its rotation is
    ``Rz(-yaw_offset_rad) @ Rx(pi)`` and its translation is ``translation_m``
    (the file's ``XYZ``), the position of the body origin in camera
    coordinates. The body pose in the world is
    ``T_world_body = inv(T_camera_world) @ T_camera_body``.
    """

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    translation_m: np.ndarray
    yaw_offset_rad: float
    tag_ids: np.ndarray
    T_camera_body: np.ndarray


@dataclass(slots=True)
class PoseObservation:
    """A body pose measured from one camera frame (the Part B output).

    ``position`` is the body origin in the world frame (m) and ``rpy`` the body
    attitude as Z-X-Y roll, pitch, yaw (rad). ``reprojection_error`` is the RMS
    pixel distance between detected and reprojected corners, and
    ``num_points`` the number of corner correspondences used.
    """

    t: float
    position: np.ndarray
    rpy: np.ndarray
    reprojection_error: float
    num_points: int

    @property
    def z(self) -> np.ndarray:
        """The 6-vector measurement ``[px, py, pz, roll, pitch, yaw]``."""
        return np.concatenate(
            [np.asarray(self.position, dtype=np.float64), np.asarray(self.rpy, dtype=np.float64)]
        )
