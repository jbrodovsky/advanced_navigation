"""Small geometry and interpolation helpers for the drone project.

Rotation matrices follow the course convention: ``C_world_body`` maps body
coordinates into world coordinates. Project 1 attitudes use the dataset's
Z-X-Y Euler sequence, ``C_world_body = Rz(yaw) @ Rx(roll) @ Ry(pitch)``, which
is the one documented exception to the book-wide Z-Y-X convention.
"""

from __future__ import annotations

import math

import numpy as np


def wrap_angle(angle_rad: float) -> float:
    """Wrap one angle into [-pi, pi)."""
    wrapped = (angle_rad + math.pi) % (2.0 * math.pi) - math.pi
    if wrapped >= math.pi:
        return wrapped - 2.0 * math.pi
    return wrapped


def wrap_angles(values: np.ndarray) -> np.ndarray:
    """Wrap every angle in an array into [-pi, pi).

    Vectorized with ``np.mod``. The final ``where`` catches the rare case where
    floating-point rounding makes ``np.mod`` return exactly ``2 pi``.
    """
    values = np.asarray(values, dtype=np.float64)
    wrapped = np.mod(values + np.pi, 2.0 * np.pi) - np.pi
    return np.where(wrapped >= np.pi, wrapped - 2.0 * np.pi, wrapped)


def Rx(angle_rad: float) -> np.ndarray:
    """Rotation matrix for a right-handed rotation about ``x``."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def Ry(angle_rad: float) -> np.ndarray:
    """Rotation matrix for a right-handed rotation about ``y``."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def Rz(angle_rad: float) -> np.ndarray:
    """Rotation matrix for a right-handed rotation about ``z``."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rpy_to_rotation_matrix(rpy: np.ndarray) -> np.ndarray:
    """Return ``C_world_body`` for Project 1 roll, pitch, yaw (Z-X-Y).

    ``C_world_body = Rz(yaw) @ Rx(roll) @ Ry(pitch)``. This matches the Vicon
    and onboard attitude in the dataset.
    """
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    return Rz(yaw) @ Rx(roll) @ Ry(pitch)


def rotation_matrix_to_rpy(C_world_body: np.ndarray) -> np.ndarray:
    """Invert :func:`rpy_to_rotation_matrix` (Z-X-Y roll, pitch, yaw).

    The matrix entries give ``roll = asin(C[2, 1])``,
    ``pitch = atan2(-C[2, 0], C[2, 2])`` and ``yaw = atan2(-C[0, 1], C[1, 1])``.
    At ``roll = +/- pi/2`` (gimbal lock) pitch and yaw are not separable; pitch
    is set to zero there. A drone flying over a tag mat never gets close.
    """
    C = np.asarray(C_world_body, dtype=np.float64)
    roll = math.asin(float(np.clip(C[2, 1], -1.0, 1.0)))
    if abs(math.cos(roll)) < 1e-9:
        return np.array([roll, 0.0, math.atan2(C[1, 0], C[0, 0])], dtype=np.float64)
    pitch = math.atan2(-C[2, 0], C[2, 2])
    yaw = math.atan2(-C[0, 1], C[1, 1])
    return np.array([roll, pitch, yaw], dtype=np.float64)


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Create the 4 x 4 homogeneous transform ``[[C, t], [0, 1]]``."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a rigid 4 x 4 transform using the rotation transpose."""
    transform = np.asarray(transform, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return make_transform(rotation.T, -rotation.T @ translation)


def interpolate_rows(
    sample_times: np.ndarray,
    row_major_values: np.ndarray,
    query_times: np.ndarray,
    fill_value: float | None = None,
) -> np.ndarray:
    """Linearly interpolate each row of a row-major array onto query times.

    With ``fill_value=None`` query times outside ``sample_times`` are clamped to
    the end values (``np.interp``'s default). Pass ``fill_value=np.nan`` to mark
    them instead, which is what truth interpolation should do.
    """
    sample_times = np.asarray(sample_times, dtype=np.float64)
    row_major_values = np.asarray(row_major_values, dtype=np.float64)
    query_times = np.asarray(query_times, dtype=np.float64)
    kwargs = {} if fill_value is None else {"left": fill_value, "right": fill_value}
    return np.vstack(
        [
            np.interp(query_times, sample_times, row_major_values[row_idx], **kwargs)
            for row_idx in range(row_major_values.shape[0])
        ]
    )
