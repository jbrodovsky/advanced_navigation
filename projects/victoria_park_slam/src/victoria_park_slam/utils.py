"""Geometry and particle-filter utilities for the Victoria Park starter package."""

from __future__ import annotations

import math

import numpy as np


def wrap_angle(angle_rad: float | np.ndarray) -> float | np.ndarray:
    """Wrap angles into the interval [-pi, pi)."""

    values = np.asarray(angle_rad, dtype=np.float64)
    wrapped = (values + np.pi) % (2.0 * np.pi) - np.pi

    if np.isscalar(angle_rad):
        return float(wrapped)
    return wrapped


def scan_angles(count: int, angle_min: float = 0.0, angle_max: float = math.pi) -> np.ndarray:
    """Return scanner bearing angles for a scan with `count` range bins."""

    if count <= 0:
        raise ValueError("count must be positive")
    return np.linspace(angle_min, angle_max, count, dtype=np.float64)


def scan_to_xy(
    scan_ranges_m: np.ndarray,
    *,
    angle_min: float = 0.0,
    angle_max: float = math.pi,
    min_range_m: float = 0.5,
    max_range_m: float | None = 75.0,
) -> np.ndarray:
    """Convert valid polar scan returns into scanner-frame Cartesian points."""

    ranges = np.asarray(scan_ranges_m, dtype=np.float64).reshape(-1)
    angles = scan_angles(ranges.size, angle_min=angle_min, angle_max=angle_max)

    valid = np.isfinite(ranges) & (ranges >= min_range_m)
    if max_range_m is not None:
        valid &= ranges <= max_range_m

    valid_ranges = ranges[valid]
    valid_angles = angles[valid]
    return np.column_stack(
        [
            valid_ranges * np.cos(valid_angles),
            valid_ranges * np.sin(valid_angles),
        ]
    )


def predict_bicycle_pose(
    pose_xy_theta: np.ndarray,
    speed_mps: float,
    steering_rad: float,
    dt_s: float,
    wheelbase_m: float,
    *,
    control_noise_covariance: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Propagate `[x, y, theta]` using a planar bicycle model.

    Optional 2x2 control noise is sampled additively on `[speed, steering]`
    before propagation. This is a utility for students to test against, not a
    complete FastSLAM prediction implementation.
    """

    if dt_s < 0.0:
        raise ValueError("dt_s must be non-negative")
    if wheelbase_m <= 0.0:
        raise ValueError("wheelbase_m must be positive")

    pose = np.asarray(pose_xy_theta, dtype=np.float64).reshape(3)
    speed = float(speed_mps)
    steering = float(steering_rad)

    if control_noise_covariance is not None:
        if rng is None:
            rng = np.random.default_rng()
        noise = rng.multivariate_normal(np.zeros(2), np.asarray(control_noise_covariance, dtype=np.float64))
        speed += float(noise[0])
        steering += float(noise[1])

    x, y, theta = pose
    yaw_rate = speed / wheelbase_m * math.tan(steering)

    if abs(yaw_rate) < 1e-12:
        x_new = x + speed * math.cos(theta) * dt_s
        y_new = y + speed * math.sin(theta) * dt_s
    else:
        theta_next = theta + yaw_rate * dt_s
        radius = speed / yaw_rate
        x_new = x + radius * (math.sin(theta_next) - math.sin(theta))
        y_new = y - radius * (math.cos(theta_next) - math.cos(theta))

    theta_new = wrap_angle(theta + yaw_rate * dt_s)
    return np.array([x_new, y_new, theta_new], dtype=np.float64)


def range_bearing_prediction(pose_xy_theta: np.ndarray, landmark_xy: np.ndarray) -> np.ndarray:
    """Predict `[range, bearing]` from a pose to a 2D landmark."""

    pose = np.asarray(pose_xy_theta, dtype=np.float64).reshape(3)
    landmark = np.asarray(landmark_xy, dtype=np.float64).reshape(2)
    delta = landmark - pose[:2]
    range_m = float(np.hypot(delta[0], delta[1]))
    if range_m <= 0.0:
        raise ValueError("landmark must not coincide with the pose position")
    bearing = wrap_angle(math.atan2(delta[1], delta[0]) - pose[2])
    return np.array([range_m, bearing], dtype=np.float64)


def inverse_range_bearing(pose_xy_theta: np.ndarray, measurement: np.ndarray) -> np.ndarray:
    """Initialize a landmark position from pose and `[range, bearing]`."""

    pose = np.asarray(pose_xy_theta, dtype=np.float64).reshape(3)
    range_m, bearing = np.asarray(measurement, dtype=np.float64).reshape(2)
    if range_m <= 0.0:
        raise ValueError("range must be positive")
    global_bearing = pose[2] + bearing
    return np.array(
        [
            pose[0] + range_m * math.cos(global_bearing),
            pose[1] + range_m * math.sin(global_bearing),
        ],
        dtype=np.float64,
    )


def range_bearing_jacobian(pose_xy_theta: np.ndarray, landmark_xy: np.ndarray) -> np.ndarray:
    """Jacobian of range-bearing prediction with respect to landmark position."""

    pose = np.asarray(pose_xy_theta, dtype=np.float64).reshape(3)
    landmark = np.asarray(landmark_xy, dtype=np.float64).reshape(2)
    dx, dy = landmark - pose[:2]
    q = dx * dx + dy * dy
    if q <= 0.0:
        raise ValueError("landmark must not coincide with the pose position")
    r = math.sqrt(q)
    return np.array(
        [
            [dx / r, dy / r],
            [-dy / q, dx / q],
        ],
        dtype=np.float64,
    )


def inverse_range_bearing_jacobian(pose_xy_theta: np.ndarray, measurement: np.ndarray) -> np.ndarray:
    """Jacobian of inverse range-bearing initialization with respect to measurement."""

    pose = np.asarray(pose_xy_theta, dtype=np.float64).reshape(3)
    range_m, bearing = np.asarray(measurement, dtype=np.float64).reshape(2)
    angle = pose[2] + bearing
    return np.array(
        [
            [math.cos(angle), -range_m * math.sin(angle)],
            [math.sin(angle), range_m * math.cos(angle)],
        ],
        dtype=np.float64,
    )


def mahalanobis_squared(residual: np.ndarray, covariance: np.ndarray) -> float:
    """Return `residual.T @ inv(covariance) @ residual` without forming an inverse."""

    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    covariance = np.asarray(covariance, dtype=np.float64)
    solved = np.linalg.solve(covariance, residual)
    return float(residual @ solved)


def effective_sample_size(weights: np.ndarray) -> float:
    """Compute particle-filter effective sample size from normalized weights."""

    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights.size == 0:
        raise ValueError("weights must not be empty")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("weights must have positive total mass")
    normalized = weights / total
    return float(1.0 / np.sum(normalized * normalized))


def systematic_resample_indices(weights: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return source particle indexes selected by systematic resampling."""

    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights.size == 0:
        raise ValueError("weights must not be empty")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("weights must have positive total mass")

    if rng is None:
        rng = np.random.default_rng()

    normalized = weights / total
    n = normalized.size
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(normalized)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions)
