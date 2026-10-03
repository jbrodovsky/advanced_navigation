"""Geometry and particle-filter utilities for the Victoria Park starter package.

Everything in this module is provided and tested. The conventions are listed in
:mod:`victoria_park_slam.types`; in short, the filter carries the **laser pose**
``[x_L, y_L, phi]`` and laser bearings are ``beta = A - pi/2`` in the vehicle
frame (x forward, y left).

Most functions accept a single pose ``(3,)`` or a stack of poses ``(N, 3)``
(one per particle) and broadcast scalar or ``(N,)`` controls, so a particle
filter can propagate all particles in one call.
"""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import Literal

import numpy as np

from .types import VehicleGeometry

ScanFrame = Literal["vehicle", "scanner"]

DEFAULT_GEOMETRY = VehicleGeometry()


# ---------------------------------------------------------------------------
# Angles and laser scans
# ---------------------------------------------------------------------------


def wrap_angle(angle_rad: float | np.ndarray) -> float | np.ndarray:
    """Wrap angles into the interval [-pi, pi)."""

    values = np.asarray(angle_rad, dtype=np.float64)
    wrapped = (values + np.pi) % (2.0 * np.pi) - np.pi

    if np.isscalar(angle_rad) or np.ndim(angle_rad) == 0:
        return float(wrapped)
    return wrapped


def scan_angles(count: int = 361, *, frame: ScanFrame = "vehicle") -> np.ndarray:
    """Return the beam angles of a Victoria Park scan with ``count`` beams.

    ``frame="vehicle"`` (the default, used by all SLAM code) returns bearings
    ``beta = A - pi/2`` in ``[-pi/2, pi/2]``: 0 is straight ahead, positive is
    to the vehicle's left. ``frame="scanner"`` returns the scanner's own angle
    ``A`` in ``[0, pi]`` (``A = 0`` points to the vehicle's right, ``A = pi/2``
    straight ahead), as used by the original MATLAB tools.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    scanner = np.linspace(0.0, math.pi, count, dtype=np.float64)
    if frame == "scanner":
        return scanner
    if frame == "vehicle":
        return scanner - math.pi / 2.0
    raise ValueError(f"frame must be 'vehicle' or 'scanner', not {frame!r}")


def scan_to_xy(
    scan_ranges_m: np.ndarray,
    *,
    frame: ScanFrame = "vehicle",
    min_range_m: float = 0.5,
    max_range_m: float | None = 75.0,
) -> np.ndarray:
    """Convert the valid returns of one scan into Cartesian points about the laser.

    With ``frame="vehicle"`` the points are ``[forward, left]`` in meters in a
    vehicle-aligned frame centered on the laser. With ``frame="scanner"`` they
    are ``[right, forward]`` (``r cos A, r sin A``), the layout ``viewLsr.m``
    plots. ``NaN`` (no return), ranges below ``min_range_m`` and ranges above
    ``max_range_m`` are dropped.
    """

    ranges = np.asarray(scan_ranges_m, dtype=np.float64).reshape(-1)
    angles = scan_angles(ranges.size, frame=frame)

    with np.errstate(invalid="ignore"):
        valid = np.isfinite(ranges) & (ranges >= min_range_m)
        if max_range_m is not None:
            valid &= ranges <= max_range_m

    valid_ranges = ranges[valid]
    valid_angles = angles[valid]
    return np.column_stack([valid_ranges * np.cos(valid_angles), valid_ranges * np.sin(valid_angles)])


def scan_to_world(
    scan_ranges_m: np.ndarray,
    laser_pose: np.ndarray,
    *,
    min_range_m: float = 0.5,
    max_range_m: float | None = 75.0,
) -> np.ndarray:
    """Project the valid returns of one scan into the world from a laser pose."""

    points = scan_to_xy(scan_ranges_m, frame="vehicle", min_range_m=min_range_m, max_range_m=max_range_m)
    x, y, phi = np.asarray(laser_pose, dtype=np.float64).reshape(3)
    c, s = math.cos(phi), math.sin(phi)
    return np.column_stack([x + c * points[:, 0] - s * points[:, 1], y + s * points[:, 0] + c * points[:, 1]])


# ---------------------------------------------------------------------------
# Vehicle model
# ---------------------------------------------------------------------------


def encoder_to_centerline_speed(
    encoder_speed_mps: float | np.ndarray,
    steering_rad: float | np.ndarray,
    H: float = DEFAULT_GEOMETRY.encoder_offset_m,
    L: float = DEFAULT_GEOMETRY.wheelbase_m,
) -> float | np.ndarray:
    """Convert the rear-left wheel encoder speed to the rear-axle centerline speed.

    In a turn with steering ``delta`` the vehicle rotates about a point on the
    rear-axle line at distance ``R = L / tan(delta)`` from the axle center
    (positive ``delta`` = left turn, center on the left). The left wheel sits
    ``H`` closer to that center, so it rolls at ``v_e = v_c (1 - H tan(delta) / L)``:

        v_c = v_e / (1 - (H / L) tan(delta)).

    Ignoring this correction makes dead reckoning drift about twice as fast on
    this dataset.
    """

    v_e = np.asarray(encoder_speed_mps, dtype=np.float64)
    delta = np.asarray(steering_rad, dtype=np.float64)
    v_c = v_e / (1.0 - (H / L) * np.tan(delta))
    if v_c.ndim == 0:
        return float(v_c)
    return v_c


def _lever_arm_world(phi: np.ndarray, geometry: VehicleGeometry) -> tuple[np.ndarray, np.ndarray]:
    a, b = geometry.laser_forward_m, geometry.laser_left_m
    c, s = np.cos(phi), np.sin(phi)
    return a * c - b * s, a * s + b * c


def laser_pose_from_rear_axle(rear_pose: np.ndarray, geometry: VehicleGeometry = DEFAULT_GEOMETRY) -> np.ndarray:
    """Laser pose ``[x_L, y_L, phi]`` from the rear-axle pose ``[x, y, phi]``.

    ``p_L = p_rear + C(phi) [a, b]^T`` with ``a`` forward and ``b`` left.
    Accepts ``(3,)`` or ``(N, 3)``.
    """

    pose = np.array(rear_pose, dtype=np.float64, copy=True)
    dx, dy = _lever_arm_world(pose[..., 2], geometry)
    pose[..., 0] += dx
    pose[..., 1] += dy
    return pose


def rear_axle_pose_from_laser(laser_pose: np.ndarray, geometry: VehicleGeometry = DEFAULT_GEOMETRY) -> np.ndarray:
    """Inverse of :func:`laser_pose_from_rear_axle`."""

    pose = np.array(laser_pose, dtype=np.float64, copy=True)
    dx, dy = _lever_arm_world(pose[..., 2], geometry)
    pose[..., 0] -= dx
    pose[..., 1] -= dy
    return pose


def predict_bicycle_pose(
    pose_xy_theta: np.ndarray,
    speed_mps: float | np.ndarray,
    steering_rad: float | np.ndarray,
    dt_s: float,
    wheelbase_m: float = DEFAULT_GEOMETRY.wheelbase_m,
) -> np.ndarray:
    """Propagate the **rear-axle** pose ``[x, y, theta]`` with the bicycle model.

    ``speed_mps`` is the rear-axle centerline speed and the controls are held
    constant over ``dt_s``, so the vehicle follows an exact circular arc with
    yaw rate ``omega = v tan(delta) / L`` (a straight line when ``omega`` is
    tiny). Accepts ``(3,)`` or ``(N, 3)`` poses with scalar or ``(N,)``
    controls; returns the same shape as the pose.
    """

    if dt_s < 0.0:
        raise ValueError("dt_s must be non-negative")
    if wheelbase_m <= 0.0:
        raise ValueError("wheelbase_m must be positive")

    pose = np.asarray(pose_xy_theta, dtype=np.float64)
    single = pose.ndim == 1
    poses = np.atleast_2d(pose).reshape(-1, 3)
    speed = np.broadcast_to(np.asarray(speed_mps, dtype=np.float64), poses.shape[:1])
    steering = np.broadcast_to(np.asarray(steering_rad, dtype=np.float64), poses.shape[:1])

    x, y, theta = poses[:, 0], poses[:, 1], poses[:, 2]
    yaw_rate = speed / wheelbase_m * np.tan(steering)
    dtheta = yaw_rate * dt_s
    theta_next = theta + dtheta

    # Exact arc: v/omega [sin(th + dth) - sin(th)] = v dt * sinc-like factor.
    # Written with the half-angle form so it is well conditioned as omega -> 0.
    half = 0.5 * dtheta
    small = np.abs(half) < 1e-9
    factor = np.where(small, 1.0, np.sin(np.where(small, 1.0, half)) / np.where(small, 1.0, half))
    distance = speed * dt_s * factor
    mid_heading = theta + half
    out = np.column_stack(
        [x + distance * np.cos(mid_heading), y + distance * np.sin(mid_heading), wrap_angle(theta_next)]
    )
    return out[0] if single else out


def predict_laser_pose(
    laser_pose: np.ndarray,
    centerline_speed_mps: float | np.ndarray,
    steering_rad: float | np.ndarray,
    dt_s: float,
    geometry: VehicleGeometry = DEFAULT_GEOMETRY,
) -> np.ndarray:
    """Propagate the **laser** pose ``[x_L, y_L, phi]`` over ``dt_s``.

    Exact for controls held constant over the interval: move the pose to the
    rear axle, follow the bicycle arc, and move back out along the lever arm.
    To first order in ``dt_s`` this is Guivant and Nebot's laser-centered model
    (:func:`laser_pose_rates`). ``centerline_speed_mps`` is the rear-axle
    centerline speed (apply :func:`encoder_to_centerline_speed` to the raw
    encoder speed first). Accepts ``(3,)`` or ``(N, 3)`` poses.
    """

    rear = rear_axle_pose_from_laser(laser_pose, geometry)
    rear_next = predict_bicycle_pose(rear, centerline_speed_mps, steering_rad, dt_s, geometry.wheelbase_m)
    return laser_pose_from_rear_axle(rear_next, geometry)


def laser_pose_rates(
    laser_pose: np.ndarray,
    centerline_speed_mps: float | np.ndarray,
    steering_rad: float | np.ndarray,
    geometry: VehicleGeometry = DEFAULT_GEOMETRY,
) -> np.ndarray:
    """Guivant and Nebot's continuous-time laser-centered kinematics.

    With ``v_c`` the centerline speed, ``omega = (v_c / L) tan(delta)``, and the
    laser at ``(a, b)`` in the vehicle frame::

        x_dot   = v_c cos(phi) - omega (a sin(phi) + b cos(phi))
        y_dot   = v_c sin(phi) + omega (a cos(phi) - b sin(phi))
        phi_dot = omega

    Returns ``[x_dot, y_dot, phi_dot]`` with the pose's shape.
    """

    pose = np.asarray(laser_pose, dtype=np.float64)
    phi = pose[..., 2]
    v = np.asarray(centerline_speed_mps, dtype=np.float64)
    omega = v / geometry.wheelbase_m * np.tan(np.asarray(steering_rad, dtype=np.float64))
    a, b = geometry.laser_forward_m, geometry.laser_left_m
    c, s = np.cos(phi), np.sin(phi)
    rates = np.stack(
        np.broadcast_arrays(v * c - omega * (a * s + b * c), v * s + omega * (a * c - b * s), omega), axis=-1
    )
    return rates


# ---------------------------------------------------------------------------
# Range-bearing observation model (pose = laser pose)
# ---------------------------------------------------------------------------


def range_bearing_prediction(pose_xy_theta: np.ndarray, landmark_xy: np.ndarray) -> np.ndarray:
    """Predict ``[range, bearing]`` from a laser pose to a 2D landmark.

    ``h(x, l) = [ |l - p|, wrap(atan2(dy, dx) - phi) ]``.
    """

    pose = np.asarray(pose_xy_theta, dtype=np.float64).reshape(3)
    landmark = np.asarray(landmark_xy, dtype=np.float64).reshape(2)
    delta = landmark - pose[:2]
    range_m = float(np.hypot(delta[0], delta[1]))
    if range_m <= 0.0:
        raise ValueError("landmark must not coincide with the pose position")
    bearing = wrap_angle(math.atan2(delta[1], delta[0]) - pose[2])
    return np.array([range_m, bearing], dtype=np.float64)


def range_bearing_jacobian(pose_xy_theta: np.ndarray, landmark_xy: np.ndarray) -> np.ndarray:
    """Jacobian ``H_l = dh/dl`` of the range-bearing prediction (2x2)."""

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


def range_bearing_pose_jacobian(pose_xy_theta: np.ndarray, landmark_xy: np.ndarray) -> np.ndarray:
    """Jacobian ``H_x = dh/dx`` of the range-bearing prediction w.r.t. the pose (2x3).

    Needed for FastSLAM 2.0 proposals and for EKF-SLAM comparisons.
    """

    H_l = range_bearing_jacobian(pose_xy_theta, landmark_xy)
    return np.column_stack([-H_l, [0.0, -1.0]])


def inverse_range_bearing(pose_xy_theta: np.ndarray, measurement: np.ndarray) -> np.ndarray:
    """Landmark position from a laser pose and ``[range, bearing]``.

    ``l = p_L + r [cos(phi + beta), sin(phi + beta)]``.
    """

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


def inverse_range_bearing_jacobian(pose_xy_theta: np.ndarray, measurement: np.ndarray) -> np.ndarray:
    """Jacobian ``G_z = dl/dz`` of :func:`inverse_range_bearing` (2x2).

    A new landmark's covariance (conditioned on the particle pose) is
    ``P_l = G_z R G_z^T``; it equals ``H_l^{-1} R H_l^{-T}``.
    """

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


def inverse_range_bearing_pose_jacobian(pose_xy_theta: np.ndarray, measurement: np.ndarray) -> np.ndarray:
    """Jacobian ``G_x = dl/dx`` of :func:`inverse_range_bearing` w.r.t. the pose (2x3)."""

    pose = np.asarray(pose_xy_theta, dtype=np.float64).reshape(3)
    range_m, bearing = np.asarray(measurement, dtype=np.float64).reshape(2)
    angle = pose[2] + bearing
    return np.array(
        [
            [1.0, 0.0, -range_m * math.sin(angle)],
            [0.0, 1.0, range_m * math.cos(angle)],
        ],
        dtype=np.float64,
    )


def numerical_jacobian(
    function: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-6,
    angle_rows: tuple[int, ...] = (),
) -> np.ndarray:
    """Central-difference Jacobian of ``function`` at ``x``.

    Output rows listed in ``angle_rows`` are differenced with wrapping, so a
    bearing output near +/-pi does not produce a 2 pi spike. Use it to check
    every analytic Jacobian you write.
    """

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    columns = []
    for index in range(x.size):
        step = np.zeros_like(x)
        step[index] = eps
        diff = np.asarray(function(x + step), dtype=np.float64) - np.asarray(function(x - step), dtype=np.float64)
        diff = np.atleast_1d(diff).astype(np.float64)
        for row in angle_rows:
            diff[row] = wrap_angle(diff[row])
        columns.append(diff / (2.0 * eps))
    return np.column_stack(columns)


# ---------------------------------------------------------------------------
# Gaussian and particle-weight helpers
# ---------------------------------------------------------------------------


def mahalanobis_squared(residual: np.ndarray, covariance: np.ndarray) -> float:
    """Return ``residual.T @ inv(covariance) @ residual`` without forming an inverse."""

    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    covariance = np.asarray(covariance, dtype=np.float64)
    solved = np.linalg.solve(covariance, residual)
    return float(residual @ solved)


def gaussian_log_likelihood(residual: np.ndarray, covariance: np.ndarray) -> float:
    """``log N(residual; 0, covariance) = -1/2 nu^T S^-1 nu - 1/2 log det(2 pi S)``.

    Uses a Cholesky factor, so it never forms ``S^-1`` and raises
    ``numpy.linalg.LinAlgError`` if ``S`` is not positive definite.
    """

    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    covariance = np.asarray(covariance, dtype=np.float64)
    chol = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(chol, residual)
    log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
    return float(-0.5 * whitened @ whitened - 0.5 * (residual.size * math.log(2.0 * math.pi) + log_det))


def normalize_log_weights(log_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize log weights with the log-sum-exp trick.

    Returns ``(weights, normalized_log_weights)``: weights sum to one and the
    normalized log weights satisfy ``logsumexp == 0``. Subtracting the maximum
    first keeps ``exp`` from underflowing when log weights are hundreds of nats
    apart, which is normal after a few scans with many trees.
    """

    log_weights = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    if log_weights.size == 0:
        raise ValueError("log_weights must not be empty")
    peak = float(np.max(log_weights))
    if not np.isfinite(peak):
        raise ValueError("at least one log weight must be finite")
    shifted = np.exp(log_weights - peak)
    log_total = peak + math.log(float(np.sum(shifted)))
    normalized = log_weights - log_total
    return np.exp(normalized), normalized


def effective_sample_size(weights: np.ndarray) -> float:
    """Compute particle-filter effective sample size ``1 / sum(w^2)`` from weights."""

    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights.size == 0:
        raise ValueError("weights must not be empty")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("weights must have positive total mass")
    normalized = weights / total
    return float(1.0 / np.sum(normalized * normalized))


def systematic_resample_indices(weights: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return source particle indexes selected by systematic (low-variance) resampling.

    One uniform draw places ``N`` evenly spaced pointers on the cumulative
    weight. The last cumulative value is clamped to exactly ``1.0`` so that
    floating-point round-off can never push a pointer past the end.
    """

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
