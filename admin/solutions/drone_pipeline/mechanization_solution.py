"""Reference solution for Project 1 Part A (instructor only).

Forward-Euler mechanization of the 15-state model, vectorized over a leading
particle axis so the same function serves the EKF (one state), the UKF (31
sigma points) and the particle filter (N particles).
"""

from __future__ import annotations

import numpy as np

from drone_pipeline.mechanization import PropagationParams
from drone_pipeline.utils import wrap_angles


def rotation_zxy(rpy: np.ndarray) -> np.ndarray:
    """C_world_body = Rz(yaw) Rx(roll) Ry(pitch) for (..., 3) angles -> (..., 3, 3)."""
    roll, pitch, yaw = np.moveaxis(np.asarray(rpy, dtype=np.float64), -1, 0)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rows = [
        [cy * cp - sr * sy * sp, -cr * sy, cy * sp + cp * sr * sy],
        [cp * sy + cy * sr * sp, cr * cy, sy * sp - cy * cp * sr],
        [-cr * sp, sr, cr * cp],
    ]
    return np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)


def euler_rate_matrix(rpy: np.ndarray) -> np.ndarray:
    """W(rpy) with omega_body = W @ [roll_dot, pitch_dot, yaw_dot] (Z-X-Y)."""
    roll, pitch, _ = np.moveaxis(np.asarray(rpy, dtype=np.float64), -1, 0)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    zero, one = np.zeros_like(roll), np.ones_like(roll)
    rows = [[cp, zero, -cr * sp], [zero, one, sr], [sp, zero, cr * cp]]
    return np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)


def euler_rate_matrix_inverse(rpy: np.ndarray) -> np.ndarray:
    """W^-1(rpy): [roll_dot, pitch_dot, yaw_dot] = W^-1 @ omega_body."""
    roll, pitch, _ = np.moveaxis(np.asarray(rpy, dtype=np.float64), -1, 0)
    cr, tr = np.cos(roll), np.tan(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    zero, one = np.zeros_like(roll), np.ones_like(roll)
    rows = [[cp, zero, sp], [sp * tr, one, -cp * tr], [-sp / cr, zero, cp / cr]]
    return np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)


def propagate_state(
    x: np.ndarray,
    acc: np.ndarray,
    omg: np.ndarray,
    dt: float,
    params: PropagationParams | None = None,
) -> np.ndarray:
    """One forward-Euler step; accepts (15,) or (N, 15)."""
    params = params or PropagationParams()
    x = np.asarray(x, dtype=np.float64)
    single = x.ndim == 1
    X = np.atleast_2d(x)
    position, rpy, velocity = X[:, 0:3], X[:, 3:6], X[:, 6:9]
    gyro_bias, accel_bias = X[:, 9:12], X[:, 12:15]

    body_rate = np.asarray(omg, dtype=np.float64) - gyro_bias
    specific_force = np.asarray(acc, dtype=np.float64) - accel_bias
    rpy_dot = np.einsum("nij,nj->ni", euler_rate_matrix_inverse(rpy), body_rate)
    accel_world = np.einsum("nij,nj->ni", rotation_zxy(rpy), specific_force) + params.gravity_world

    out = X.copy()
    out[:, 0:3] = position + velocity * dt
    out[:, 3:6] = wrap_angles(rpy + rpy_dot * dt)
    out[:, 6:9] = velocity + accel_world * dt
    return out[0] if single else out
