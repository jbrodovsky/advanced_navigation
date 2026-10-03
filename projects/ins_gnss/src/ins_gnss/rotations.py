"""Quaternion and rotation helpers in the book's conventions (provided, tested).

* Quaternions are Hamilton, scalar first, ``q = [w, x, y, z]``.
* ``q_b^n`` rotates body-frame vectors into the navigation frame:
  ``v^n = C(q_b^n) v^b`` with ``C(q_b^n) = C_b^n``.
* Products compose like matrices: ``C(p (x) q) = C(p) C(q)``.
* Rotation matrices are written ``C`` (never ``R``).
* ``skew(a) @ b == np.cross(a, b)``.
* Euler angles (Z-Y-X: yaw, then pitch, then roll) are for display and
  initialization only: ``C_b^n = R_z(yaw) R_y(pitch) R_x(roll)``.

See @sec-frames-conventions for the derivations.
"""

from __future__ import annotations

import numpy as np


def skew(vector: np.ndarray) -> np.ndarray:
    """Skew-symmetric (cross-product) matrix ``[a]_x`` with ``[a]_x b = a x b``."""
    a = np.asarray(vector, dtype=np.float64)
    return np.array(
        [
            [0.0, -a[2], a[1]],
            [a[2], 0.0, -a[0]],
            [-a[1], a[0], 0.0],
        ]
    )


def quat_multiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product ``p (x) q`` of two scalar-first quaternions."""
    pw, px, py, pz = np.asarray(p, dtype=np.float64)
    qw, qx, qy, qz = np.asarray(q, dtype=np.float64)
    return np.array(
        [
            pw * qw - px * qx - py * qy - pz * qz,
            pw * qx + px * qw + py * qz - pz * qy,
            pw * qy - px * qz + py * qw + pz * qx,
            pw * qz + px * qy - py * qx + pz * qw,
        ]
    )


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Unit quaternion with a non-negative scalar part (``q`` and ``-q`` are the same rotation)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    return -q if q[0] < 0.0 else q


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate (the inverse rotation for a unit quaternion)."""
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.array([w, -x, -y, -z])


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Rotation matrix ``C(q)``; for ``q = q_b^n`` this is ``C_b^n``."""
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def dcm_to_quat(dcm: np.ndarray) -> np.ndarray:
    """Unit quaternion of a rotation matrix (Shepperd's method, numerically safe)."""
    C = np.asarray(dcm, dtype=np.float64)
    trace = np.trace(C)
    candidates = np.array([trace, C[0, 0], C[1, 1], C[2, 2]])
    i = int(np.argmax(candidates))
    if i == 0:
        w = 0.5 * np.sqrt(1.0 + trace)
        q = [w, (C[2, 1] - C[1, 2]) / (4 * w), (C[0, 2] - C[2, 0]) / (4 * w), (C[1, 0] - C[0, 1]) / (4 * w)]
    elif i == 1:
        x = 0.5 * np.sqrt(1.0 + 2 * C[0, 0] - trace)
        q = [(C[2, 1] - C[1, 2]) / (4 * x), x, (C[0, 1] + C[1, 0]) / (4 * x), (C[0, 2] + C[2, 0]) / (4 * x)]
    elif i == 2:
        y = 0.5 * np.sqrt(1.0 + 2 * C[1, 1] - trace)
        q = [(C[0, 2] - C[2, 0]) / (4 * y), (C[0, 1] + C[1, 0]) / (4 * y), y, (C[1, 2] + C[2, 1]) / (4 * y)]
    else:
        z = 0.5 * np.sqrt(1.0 + 2 * C[2, 2] - trace)
        q = [(C[1, 0] - C[0, 1]) / (4 * z), (C[0, 2] + C[2, 0]) / (4 * z), (C[1, 2] + C[2, 1]) / (4 * z), z]
    return quat_normalize(np.array(q))


def rotvec_to_quat(rotation_vector: np.ndarray) -> np.ndarray:
    """Exact quaternion of a rotation vector ``phi`` (angle ``|phi|`` about ``phi/|phi|``).

    ``[cos(|phi|/2), sin(|phi|/2) phi/|phi|]``, with a Taylor series near zero.
    For the body-rate increment of one IMU step use ``phi = omega_ib^b * dt``.
    """
    phi = np.asarray(rotation_vector, dtype=np.float64)
    angle = float(np.linalg.norm(phi))
    if angle < 1e-8:
        # sin(a/2)/a ~ 1/2 - a^2/48; cos(a/2) ~ 1 - a^2/8
        return quat_normalize(np.concatenate([[1.0 - angle**2 / 8.0], (0.5 - angle**2 / 48.0) * phi]))
    return np.concatenate([[np.cos(0.5 * angle)], np.sin(0.5 * angle) * phi / angle])


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Rotation vector (axis times angle, angle in [0, pi]) of a quaternion."""
    q = quat_normalize(q)
    sin_half = float(np.linalg.norm(q[1:]))
    if sin_half < 1e-12:
        return 2.0 * q[1:]
    angle = 2.0 * np.arctan2(sin_half, q[0])
    return angle * q[1:] / sin_half


def dcm_to_rotvec(dcm: np.ndarray) -> np.ndarray:
    """Rotation vector of a rotation matrix (via the quaternion, so it is safe near pi)."""
    return quat_to_rotvec(dcm_to_quat(dcm))


def euler_to_dcm(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """``C_b^n = R_z(yaw) R_y(pitch) R_x(roll)`` (Z-Y-X sequence, display convention)."""
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def dcm_to_euler(dcm: np.ndarray) -> np.ndarray:
    """``[roll, pitch, yaw]`` [rad] of ``C_b^n`` (Z-Y-X). Ill-conditioned near pitch = +/-90 deg."""
    C = np.asarray(dcm, dtype=np.float64)
    roll = np.arctan2(C[2, 1], C[2, 2])
    pitch = -np.arcsin(np.clip(C[2, 0], -1.0, 1.0))
    yaw = np.arctan2(C[1, 0], C[0, 0])
    return np.array([roll, pitch, yaw])


def euler_to_quat(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """Quaternion ``q_b^n`` of Z-Y-X Euler angles."""
    return dcm_to_quat(euler_to_dcm(roll_rad, pitch_rad, yaw_rad))


def attitude_error(C_true: np.ndarray, C_est: np.ndarray) -> np.ndarray:
    """Navigation-frame attitude error ``dtheta`` [rad] with ``C_true = Exp(dtheta) C_est``.

    True minus estimate, the same convention as the filter's error state. Use
    this (or its norm) to compare attitudes; never subtract Euler angles, which
    are meaningless near pitch = +/-90 deg.
    """
    return dcm_to_rotvec(np.asarray(C_true, dtype=np.float64) @ np.asarray(C_est, dtype=np.float64).T)
