"""Shared evaluation so every report measures the trajectory the same way.

GPS is the only position reference in this dataset, and it is not truth: the
receiver was failing, so there are gaps of up to 58 s and dozens of jumps. It
also lives in its own local east/north frame, while the filter starts at
``(0, 0, 0)`` in an arbitrary frame. The protocol is therefore:

1. **Valid fixes** (:func:`valid_gps_mask`): cut the fixes into runs at
   gaps and at jumps (a fix more than ``max_jump_m`` farther from the previous
   one than the vehicle could have driven, by odometry path length), then drop
   short runs and fixes outside the odometry time span.
2. **Interpolate the estimate** (the laser track, which is where the GPS
   antenna is) to the valid fix times. GPS is never interpolated across a gap.
3. **Align** the estimate to GPS with the best rigid SE(2) transform
   (:func:`align_se2`, Kabsch/Umeyama without scale).
4. **Report** the position RMSE after alignment, and the same number for dead
   reckoning over the same fixes (:func:`dead_reckoning_comparison`).

The alignment absorbs the arbitrary start frame; it cannot hide a bent
trajectory, which is the thing SLAM is supposed to fix.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .types import VehicleGeometry, VictoriaParkDataset
from .utils import encoder_to_centerline_speed, laser_pose_from_rear_axle


@dataclass(slots=True)
class TrajectoryEvaluation:
    """Result of comparing an estimated laser track with GPS."""

    rmse_m: float
    max_error_m: float
    num_fixes: int
    time_s: np.ndarray
    errors_m: np.ndarray
    aligned_xy_m: np.ndarray
    reference_xy_m: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray

    @property
    def rotation_deg(self) -> float:
        """Rotation of the SE(2) alignment, in degrees."""
        return math.degrees(math.atan2(self.rotation[1, 0], self.rotation[0, 0]))


def odometry_path_length(dataset: VictoriaParkDataset, query_time_s: np.ndarray) -> np.ndarray:
    """Distance driven (integral of |encoder speed|) up to each query time."""

    t = np.asarray(dataset.odometry_time_s, dtype=np.float64)
    speed = np.abs(np.asarray(dataset.speed_mps, dtype=np.float64))
    cumulative = np.concatenate([[0.0], np.cumsum(speed[:-1] * np.diff(t))])
    return np.interp(np.asarray(query_time_s, dtype=np.float64), t, cumulative)


def valid_gps_mask(
    dataset: VictoriaParkDataset,
    *,
    max_gap_s: float = 1.0,
    max_jump_m: float = 5.0,
    min_run_fixes: int = 5,
) -> np.ndarray:
    """Boolean mask of GPS fixes usable as a position reference.

    The fixes are cut into runs at gaps longer than ``max_gap_s`` and at
    jumps, where the distance between consecutive fixes exceeds the odometry
    path length between them by more than ``max_jump_m``. A fix is rejected
    when it lies outside the odometry time span or belongs to a run of fewer
    than ``min_run_fixes`` fixes (so isolated outliers disappear).
    """

    if dataset.gps_time_s is None or dataset.gps_xy_m is None:
        raise ValueError("dataset has no GPS")
    t = np.asarray(dataset.gps_time_s, dtype=np.float64)
    xy = dataset.gps_xy_m
    t_odo = dataset.odometry_time_s
    inside = (t >= t_odo[0]) & (t <= t_odo[-1]) & np.all(np.isfinite(xy), axis=1)
    mask = inside.copy()
    if t.size < 2:
        return mask

    driven = odometry_path_length(dataset, t)
    step = np.hypot(*np.diff(xy, axis=0).T)
    jump = step - np.diff(driven) > max_jump_m
    gap = np.diff(t) > max_gap_s

    # Runs of consecutive fixes, split at gaps and jumps. An isolated outlier
    # forms a run of one and is dropped; the fixes after it start a new run.
    breaks = np.flatnonzero(jump | gap) + 1
    for run in np.split(np.arange(t.size), breaks):
        if run.size < min_run_fixes:
            mask[run] = False
    return mask


def align_se2(source_xy: np.ndarray, target_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best rigid transform (no scale) mapping ``source`` onto ``target``.

    Returns ``(C, t)`` minimizing ``sum ||target_i - (C source_i + t)||^2``
    with ``C`` a proper rotation (Kabsch/Umeyama). Needs at least two
    distinct points.
    """

    source = np.asarray(source_xy, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target_xy, dtype=np.float64).reshape(-1, 2)
    if source.shape != target.shape or source.shape[0] < 2:
        raise ValueError("need matching (N, 2) arrays with N >= 2")
    mean_source = source.mean(axis=0)
    mean_target = target.mean(axis=0)
    cross = (source - mean_source).T @ (target - mean_target)
    U, _, Vt = np.linalg.svd(cross)
    reflection = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    C = Vt.T @ np.diag([1.0, reflection]) @ U.T
    t = mean_target - C @ mean_source
    return C, t


def apply_se2(C: np.ndarray, t: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Apply ``p -> C p + t`` to ``(N, 2)`` points."""

    return np.asarray(xy, dtype=np.float64).reshape(-1, 2) @ np.asarray(C).T + np.asarray(t).reshape(2)


def evaluate_against_gps(
    dataset: VictoriaParkDataset,
    estimate_time_s: np.ndarray,
    estimate_xy_m: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    time_range_s: tuple[float, float] | None = None,
) -> TrajectoryEvaluation:
    """SE(2)-aligned position RMSE of a laser track against valid GPS fixes.

    ``estimate_xy_m`` is the laser (= GPS antenna) position at
    ``estimate_time_s`` (any rate, increasing times). Only valid fixes inside
    the estimate's time span (and inside ``time_range_s`` if given) are used.
    """

    est_t = np.asarray(estimate_time_s, dtype=np.float64).reshape(-1)
    est_xy = np.asarray(estimate_xy_m, dtype=np.float64).reshape(-1, 2)
    if valid is None:
        valid = valid_gps_mask(dataset)
    gps_t = np.asarray(dataset.gps_time_s, dtype=np.float64)
    use = np.asarray(valid, dtype=bool) & (gps_t >= est_t[0]) & (gps_t <= est_t[-1])
    if time_range_s is not None:
        use &= (gps_t >= time_range_s[0]) & (gps_t <= time_range_s[1])
    if np.count_nonzero(use) < 2:
        raise ValueError("fewer than two valid GPS fixes overlap the estimate")

    query = gps_t[use]
    reference = dataset.gps_xy_m[use]
    estimate = np.column_stack([np.interp(query, est_t, est_xy[:, 0]), np.interp(query, est_t, est_xy[:, 1])])
    C, t = align_se2(estimate, reference)
    aligned = apply_se2(C, t, estimate)
    errors = np.hypot(*(reference - aligned).T)
    return TrajectoryEvaluation(
        rmse_m=float(np.sqrt(np.mean(errors**2))),
        max_error_m=float(np.max(errors)),
        num_fixes=int(errors.size),
        time_s=query,
        errors_m=errors,
        aligned_xy_m=aligned,
        reference_xy_m=reference,
        rotation=C,
        translation=t,
    )


def dead_reckon(
    dataset: VictoriaParkDataset,
    *,
    geometry: VehicleGeometry = VehicleGeometry(),
    use_encoder_correction: bool = True,
    use_lever_arm: bool = True,
    stop_time_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Dead-reckon from ``(0, 0, 0)`` at the first odometry sample.

    Returns ``(time_s, poses)`` at the odometry timestamps. Each sample is held
    over its interval and integrated exactly along the bicycle arc. With
    ``use_lever_arm`` the poses are laser poses (what GPS measures); without
    it they are rear-axle poses, i.e. the classic mistake of treating the GPS
    antenna as if it sat on the rear axle.
    """

    t = np.asarray(dataset.odometry_time_s, dtype=np.float64)
    if stop_time_s is not None:
        t = t[t <= stop_time_s]
    n = t.size
    speed = np.asarray(dataset.speed_mps[:n], dtype=np.float64)
    steering = np.asarray(dataset.steering_rad[:n], dtype=np.float64)
    if use_encoder_correction:
        speed = encoder_to_centerline_speed(
            speed, steering, H=geometry.encoder_offset_m, L=geometry.wheelbase_m
        )
    # Vectorized exact arcs (the same formula as predict_bicycle_pose): the
    # heading increments add up, and each position increment follows the chord
    # of its arc at the mid-interval heading.
    dt = np.diff(t)
    dtheta = speed[:-1] / geometry.wheelbase_m * np.tan(steering[:-1]) * dt
    theta = np.concatenate([[0.0], np.cumsum(dtheta)])
    half = 0.5 * dtheta
    chord = speed[:-1] * dt * np.sinc(half / np.pi)  # np.sinc(x) = sin(pi x) / (pi x)
    mid = theta[:-1] + half
    rear = np.zeros((n, 3))
    rear[1:, 0] = np.cumsum(chord * np.cos(mid))
    rear[1:, 1] = np.cumsum(chord * np.sin(mid))
    rear[:, 2] = (theta + np.pi) % (2.0 * np.pi) - np.pi
    if not use_lever_arm:
        return t, rear
    # Start the laser track at (0, 0, 0) as well, so it is the same rigid
    # frame a filter initialized there would use.
    laser = laser_pose_from_rear_axle(rear, geometry)
    laser[:, :2] -= laser[0, :2]
    return t, laser


DR_VARIANTS = {
    "DR (encoder correction + lever arm)": {"use_encoder_correction": True, "use_lever_arm": True},
    "DR (encoder correction only)": {"use_encoder_correction": True, "use_lever_arm": False},
    "DR (lever arm only)": {"use_encoder_correction": False, "use_lever_arm": True},
    "DR (no corrections)": {"use_encoder_correction": False, "use_lever_arm": False},
}


def dead_reckoning_comparison(
    dataset: VictoriaParkDataset,
    *,
    stop_time_s: float | None = None,
    valid: np.ndarray | None = None,
    geometry: VehicleGeometry = VehicleGeometry(),
) -> dict[str, TrajectoryEvaluation]:
    """Evaluate the four dead-reckoning variants of :data:`DR_VARIANTS` against GPS."""

    valid = valid_gps_mask(dataset) if valid is None else valid
    results = {}
    for name, options in DR_VARIANTS.items():
        t, poses = dead_reckon(dataset, geometry=geometry, stop_time_s=stop_time_s, **options)
        results[name] = evaluate_against_gps(dataset, t, poses[:, :2], valid=valid)
    return results
