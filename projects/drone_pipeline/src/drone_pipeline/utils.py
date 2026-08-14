"""Small geometry and interpolation helpers for the drone project."""

from __future__ import annotations

import math

import numpy as np


def wrap_angle(angle_rad: float) -> float:
    """Wrap one angle into [-pi, pi)."""
    wrapped = (angle_rad + math.pi) % (2.0 * math.pi) - math.pi
    if wrapped == math.pi:
        return -math.pi
    return wrapped


def wrap_angles(values: np.ndarray) -> np.ndarray:
    """Wrap every angle in an array into [-pi, pi)."""
    values = np.asarray(values, dtype=np.float64)
    return np.vectorize(wrap_angle, otypes=[np.float64])(values)


def Rx(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def Ry(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def Rz(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rpy_to_rotation_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    return Rz(yaw) @ Ry(pitch) @ Rx(roll)


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Create a 4x4 homogeneous transform."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def interpolate_rows(sample_times: np.ndarray, row_major_values: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    """Interpolate each row of a row-major array onto query times."""
    sample_times = np.asarray(sample_times, dtype=np.float64)
    row_major_values = np.asarray(row_major_values, dtype=np.float64)
    query_times = np.asarray(query_times, dtype=np.float64)
    return np.vstack(
        [np.interp(query_times, sample_times, row_major_values[row_idx]) for row_idx in range(row_major_values.shape[0])]
    )

