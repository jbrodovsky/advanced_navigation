"""Tests for the Project 2 starter package.

Two kinds of tests live here:

* Support-code tests (decoding, scan geometry, vehicle model, Jacobians,
  weights, time synchronization, evaluation). These must always pass.
* Student-facing tests for the deliverables (``extract_tree_landmarks``,
  ``LandmarkEKF``, ``FastSLAM``). They skip with a message while a method is
  still a ``NotImplementedError`` stub and start checking your implementation
  as soon as it exists.

Tests that need the real dataset skip until it has been downloaded.
"""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import savemat
from scipy.stats import multivariate_normal

from victoria_park_slam import fastslam, landmarks, run_all
from victoria_park_slam.data_paths import DatasetNotFoundError
from victoria_park_slam.datasets import decode_laser_ranges, load_victoria_park_data, resolve_victoria_park_root
from victoria_park_slam.evaluation import (
    align_se2,
    apply_se2,
    dead_reckon,
    dead_reckoning_comparison,
    evaluate_against_gps,
    valid_gps_mask,
)
from victoria_park_slam.timing import scan_steps
from victoria_park_slam.types import (
    AssociationKind,
    LandmarkTrackState,
    MeasurementModelConfig,
    MotionModelConfig,
    TreeObservation,
    VehicleGeometry,
    VictoriaParkDataset,
)
from victoria_park_slam.utils import (
    effective_sample_size,
    encoder_to_centerline_speed,
    gaussian_log_likelihood,
    inverse_range_bearing,
    inverse_range_bearing_jacobian,
    inverse_range_bearing_pose_jacobian,
    laser_pose_from_rear_axle,
    laser_pose_rates,
    mahalanobis_squared,
    normalize_log_weights,
    numerical_jacobian,
    predict_bicycle_pose,
    predict_laser_pose,
    range_bearing_jacobian,
    range_bearing_pose_jacobian,
    range_bearing_prediction,
    rear_axle_pose_from_laser,
    scan_angles,
    scan_to_world,
    scan_to_xy,
    systematic_resample_indices,
    wrap_angle,
)


FETCH_HINT = "Victoria Park data not downloaded; run `uv run python scripts/fetch_data.py victoria-park`"
GEOMETRY = VehicleGeometry()
R_DEFAULT = MeasurementModelConfig().covariance


def _real_data_root() -> Path | None:
    try:
        return resolve_victoria_park_root()
    except DatasetNotFoundError:
        return None


REAL_DATA = _real_data_root()
_REAL_DATASET: VictoriaParkDataset | None = None


def real_dataset() -> VictoriaParkDataset:
    global _REAL_DATASET
    if _REAL_DATASET is None:
        _REAL_DATASET = load_victoria_park_data()
    return _REAL_DATASET


def call_or_skip(test: unittest.TestCase, function, *args, **kwargs):
    """Call a student deliverable; skip the test while it is still a stub."""
    try:
        return function(*args, **kwargs)
    except NotImplementedError as error:
        name = getattr(function, "__qualname__", repr(function))
        test.skipTest(f"{name} is not implemented yet (starter stub): {error}")


def synthetic_dataset(odometry_time_s, speed, steering, laser_time_s, gps_time_s=None, gps_xy=None, scans=None):
    """A VictoriaParkDataset built in memory (no files)."""
    laser_time_s = np.asarray(laser_time_s, dtype=np.float64)
    if scans is None:
        scans = np.full((laser_time_s.size, 361), np.nan)
    gps_xy = None if gps_xy is None else np.asarray(gps_xy, dtype=np.float64)
    return VictoriaParkDataset(
        root=Path("."),
        laser_ranges_m=np.asarray(scans, dtype=np.float64),
        laser_time_s=laser_time_s,
        speed_mps=np.asarray(speed, dtype=np.float64),
        steering_rad=np.asarray(steering, dtype=np.float64),
        odometry_time_s=np.asarray(odometry_time_s, dtype=np.float64),
        gps_lat_m=None if gps_xy is None else gps_xy[:, 1],
        gps_lon_m=None if gps_xy is None else gps_xy[:, 0],
        gps_time_s=None if gps_time_s is None else np.asarray(gps_time_s, dtype=np.float64),
    )


def synthetic_scan(circles=(), segments=()) -> np.ndarray:
    """Ray-cast a 361-beam Victoria Park scan in the vehicle frame (x forward, y left).

    ``circles``: (center_x, center_y, radius); ``segments``: ((x0, y0), (x1, y1)).
    Beams that hit nothing are NaN, as in the decoded data.
    """
    beta = scan_angles(361, frame="vehicle")
    directions = np.column_stack([np.cos(beta), np.sin(beta)])
    ranges = np.full(361, np.inf)
    for cx, cy, radius in circles:
        along = directions @ np.array([cx, cy])
        disc = along**2 - (cx * cx + cy * cy - radius * radius)
        hit = (disc >= 0.0) & (along > 0.0)
        t = along - np.sqrt(np.where(hit, disc, 0.0))
        ranges = np.where(hit & (t > 0.0), np.minimum(ranges, t), ranges)
    for (x0, y0), (x1, y1) in segments:
        edge = np.array([x1 - x0, y1 - y0])
        for k, d in enumerate(directions):
            matrix = np.column_stack([d, -edge])
            if abs(np.linalg.det(matrix)) < 1e-12:
                continue
            t, s = np.linalg.solve(matrix, np.array([x0, y0]))
            if t > 0.0 and 0.0 <= s <= 1.0:
                ranges[k] = min(ranges[k], t)
    ranges[~np.isfinite(ranges)] = np.nan
    return ranges


def polar(range_m: float, bearing_deg: float) -> tuple[float, float]:
    b = math.radians(bearing_deg)
    return range_m * math.cos(b), range_m * math.sin(b)


# ---------------------------------------------------------------------------
# Support code
# ---------------------------------------------------------------------------


class ResampleBookkeepingTests(unittest.TestCase):
    """Provided support code: the best particle survives a resample."""

    class _CopyFirst(fastslam.FastSLAM):
        def resample(self) -> None:  # minimal stand-in for the student method
            n = len(self.particles)
            self.particles = [self.particles[0].copy() for _ in range(n)]
            for particle in self.particles:
                particle.log_weight = -math.log(n)

    def test_best_particle_is_remembered_across_a_resample(self) -> None:
        slam = self._CopyFirst.from_initial_pose(4)
        for i, particle in enumerate(slam.particles):
            particle.pose_xy_theta = np.array([float(i), 0.0, 0.0])
            particle.log_weight = 0.0
        slam.particles[2].log_weight = 10.0  # dominant: N_eff ~ 1 < 0.5 N
        n_eff, resampled = slam.resample_if_needed()
        self.assertTrue(resampled)
        self.assertLess(n_eff, 2.0)
        # The resampled set holds copies of particle 0; the report must still use particle 2.
        np.testing.assert_allclose(slam.best_particle().pose_xy_theta, [2.0, 0.0, 0.0])

    def test_no_snapshot_without_resampling(self) -> None:
        slam = self._CopyFirst.from_initial_pose(4)
        for i, particle in enumerate(slam.particles):
            particle.pose_xy_theta = np.array([float(i), 0.0, 0.0])
        slam.particles[3].log_weight += 0.1  # nearly uniform: no resample
        _, resampled = slam.resample_if_needed()
        self.assertFalse(resampled)
        self.assertIs(slam.best_particle(), slam.particles[3])


class RunAllArgumentTests(unittest.TestCase):
    def test_nonpositive_scan_count_is_rejected(self) -> None:
        for value in ("0", "-5"):
            with self.subTest(scans=value), self.assertRaises(SystemExit):
                run_all.main(["--scans", value])

    @unittest.skipIf(REAL_DATA is None, FETCH_HINT)
    def test_too_short_segment_is_rejected_before_evaluation(self) -> None:
        dataset = load_victoria_park_data()
        minimum = run_all.minimum_segment_scans(dataset, valid_gps_mask(dataset))
        self.assertGreater(minimum, 1)
        self.assertLess(minimum, 200)
        with tempfile.TemporaryDirectory() as tmp_dir, self.assertRaises(SystemExit):
            run_all.main(["--scans", str(minimum - 1), "--out", tmp_dir])


class DecodingTests(unittest.TestCase):
    def test_decode_masks_flags_and_marks_no_return(self) -> None:
        raw = np.array([1234, 8000, 7999, 8191, 8192 + 1457, 0], dtype=np.uint16)
        ranges = decode_laser_ranges(raw)
        np.testing.assert_allclose(ranges[[0, 2, 4, 5]], [12.34, 79.99, 14.57, 0.0])
        self.assertTrue(np.isnan(ranges[1]) and np.isnan(ranges[3]))

    def test_load_synthetic_dataset_converts_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            laser = np.full((2, 361), 1234, dtype=np.uint16)
            laser[1, 0] = 8191
            laser[1, 1] = 8192 + 500
            savemat(root / "aa3_lsr2.mat", {"LASER": laser, "TLsr": np.array([[1000.0], [1250.0]])})
            savemat(
                root / "aa3_dr.mat",
                {"speed": np.array([[2.0], [2.5]]), "steering": np.array([[0.0], [0.1]]), "time": np.array([[1000.0], [1025.0]])},
            )
            savemat(
                root / "aa3_gpsx.mat",
                {"La_m": np.array([[1.0]]), "Lo_m": np.array([[2.0]]), "timeGps": np.array([[1100.0]])},
            )
            dataset = load_victoria_park_data(root)
        self._assert_dataset_shapes(dataset)
        self.assertAlmostEqual(dataset.laser_ranges_m[0, 0], 12.34)
        self.assertTrue(np.isnan(dataset.laser_ranges_m[1, 0]))
        self.assertAlmostEqual(dataset.laser_ranges_m[1, 1], 5.00)
        self.assertEqual(int(dataset.laser_raw[1, 1]), 8192 + 500)
        np.testing.assert_allclose(dataset.laser_time_s, [1.0, 1.25])
        np.testing.assert_allclose(dataset.odometry_time_s, [1.0, 1.025])
        np.testing.assert_allclose(dataset.gps_time_s, [1.1])
        np.testing.assert_allclose(dataset.gps_xy_m, [[2.0, 1.0]])

    def test_missing_dataset_explains_how_to_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(DatasetNotFoundError) as caught:
                load_victoria_park_data(Path(tmp_dir))
        self.assertIn("scripts/fetch_data.py victoria-park", str(caught.exception))

    def _assert_dataset_shapes(self, dataset) -> None:
        self.assertEqual(dataset.laser_ranges_m.shape[1], 361)
        self.assertEqual(dataset.laser_time_s.shape[0], dataset.laser_ranges_m.shape[0])
        self.assertEqual(dataset.speed_mps.shape[0], dataset.steering_rad.shape[0])
        self.assertIsNotNone(dataset.gps_time_s)


class ScanGeometryTests(unittest.TestCase):
    def test_wrap_angle(self) -> None:
        self.assertAlmostEqual(wrap_angle(np.pi), -np.pi)
        self.assertAlmostEqual(wrap_angle(3.0 * np.pi), -np.pi)
        values = wrap_angle(np.array([0.0, np.pi, -np.pi]))
        np.testing.assert_allclose(values, np.array([0.0, -np.pi, -np.pi]))

    def test_scan_angles_in_both_frames(self) -> None:
        vehicle = scan_angles(361)
        scanner = scan_angles(361, frame="scanner")
        np.testing.assert_allclose(vehicle[[0, 180, 360]], [-np.pi / 2, 0.0, np.pi / 2], atol=1e-12)
        np.testing.assert_allclose(scanner[[0, 180, 360]], [0.0, np.pi / 2, np.pi], atol=1e-12)
        np.testing.assert_allclose(np.diff(vehicle), np.deg2rad(0.5))
        with self.assertRaises(ValueError):
            scan_angles(361, frame="laser")

    def test_scan_to_xy_vehicle_frame_puts_beam_zero_on_the_right(self) -> None:
        scan = np.full(361, np.nan)
        scan[0], scan[180], scan[360] = 5.0, 10.0, 7.0  # A = 0 (right), 90 (ahead), 180 (left)
        np.testing.assert_allclose(scan_to_xy(scan), [[0.0, -5.0], [10.0, 0.0], [0.0, 7.0]], atol=1e-12)
        np.testing.assert_allclose(scan_to_xy(scan, frame="scanner"), [[5.0, 0.0], [0.0, 10.0], [-7.0, 0.0]], atol=1e-12)

    def test_scan_to_world_uses_the_laser_pose(self) -> None:
        scan = np.full(361, np.nan)
        scan[180] = 10.0
        np.testing.assert_allclose(scan_to_world(scan, np.array([1.0, 2.0, np.pi / 2])), [[1.0, 12.0]], atol=1e-12)


class VehicleModelTests(unittest.TestCase):
    def test_encoder_correction_follows_the_turning_geometry(self) -> None:
        self.assertAlmostEqual(encoder_to_centerline_speed(3.0, 0.0), 3.0)
        delta, v_c = 0.3, 5.0
        omega = v_c * math.tan(delta) / GEOMETRY.wheelbase_m
        turn_radius = GEOMETRY.wheelbase_m / math.tan(delta)
        v_left_wheel = omega * (turn_radius - GEOMETRY.encoder_offset_m)  # inside wheel in a left turn
        self.assertAlmostEqual(encoder_to_centerline_speed(v_left_wheel, delta), v_c, places=12)
        self.assertGreater(encoder_to_centerline_speed(3.0, 0.2), 3.0)  # left turn: encoder on the inside
        self.assertLess(encoder_to_centerline_speed(3.0, -0.2), 3.0)
        np.testing.assert_allclose(encoder_to_centerline_speed(np.array([1.0, 1.0]), np.array([0.0, 0.0])), [1.0, 1.0])

    def test_lever_arm_values_and_round_trip(self) -> None:
        np.testing.assert_allclose(laser_pose_from_rear_axle(np.zeros(3)), [3.78, 0.50, 0.0])
        np.testing.assert_allclose(
            laser_pose_from_rear_axle(np.array([0.0, 0.0, np.pi / 2])), [-0.50, 3.78, np.pi / 2], atol=1e-12
        )
        poses = np.random.default_rng(1).normal(size=(5, 3))
        np.testing.assert_allclose(rear_axle_pose_from_laser(laser_pose_from_rear_axle(poses)), poses, atol=1e-12)

    def test_bicycle_straight_turning_and_vectorized(self) -> None:
        np.testing.assert_allclose(predict_bicycle_pose(np.array([1.0, 2.0, 0.0]), 3.0, 0.0, 2.0, 2.5), [7.0, 2.0, 0.0], atol=1e-12)
        turned = predict_bicycle_pose(np.zeros(3), 2.0, 0.2, 1.0, 2.5)
        self.assertTrue(np.all(turned > 0.0))
        poses = np.zeros((4, 3))
        out = predict_bicycle_pose(poses, np.array([1.0, 2.0, 3.0, 4.0]), 0.1, 0.5)
        self.assertEqual(out.shape, (4, 3))
        np.testing.assert_allclose(out[2], predict_bicycle_pose(np.zeros(3), 3.0, 0.1, 0.5))

    def test_bicycle_full_circle_returns_to_start(self) -> None:
        speed, steering = 2.0, 0.25
        omega = speed * math.tan(steering) / GEOMETRY.wheelbase_m
        pose = np.array([1.0, -1.0, 0.3])
        out = pose
        for _ in range(100):
            out = predict_bicycle_pose(out, speed, steering, 2.0 * math.pi / omega / 100)
        np.testing.assert_allclose(out[:2], pose[:2], atol=1e-9)
        self.assertAlmostEqual(wrap_angle(out[2] - pose[2]), 0.0, places=9)

    def test_laser_kinematics_match_guivant_rates(self) -> None:
        pose, speed, steering = np.array([2.0, 1.0, 0.7]), 4.0, -0.3
        dt = 1e-6
        numeric = (predict_laser_pose(pose, speed, steering, dt) - pose) / dt
        np.testing.assert_allclose(numeric, laser_pose_rates(pose, speed, steering), rtol=1e-4, atol=1e-5)

    def test_laser_propagation_is_exact_for_constant_controls(self) -> None:
        pose, speed, steering, total = np.array([0.0, 0.0, 0.2]), 5.0, 0.35, 2.0
        euler = pose.copy()
        steps = 20000
        for _ in range(steps):
            euler = euler + laser_pose_rates(euler, speed, steering) * (total / steps)
        exact = predict_laser_pose(pose, speed, steering, total)
        np.testing.assert_allclose(exact[:2], euler[:2], atol=2e-3)
        self.assertAlmostEqual(wrap_angle(exact[2] - euler[2]), 0.0, places=6)


class ObservationModelTests(unittest.TestCase):
    POSES = [np.array([2.0, -1.0, 0.4]), np.array([0.0, 0.0, 3.1]), np.array([-5.0, 4.0, -2.0])]
    LANDMARKS = [np.array([6.0, 2.5]), np.array([-10.0, 0.05]), np.array([-1.0, 1.0])]

    def test_range_bearing_inverse_consistency(self) -> None:
        for pose, landmark in zip(self.POSES, self.LANDMARKS):
            measurement = range_bearing_prediction(pose, landmark)
            np.testing.assert_allclose(inverse_range_bearing(pose, measurement), landmark, atol=1e-12)

    def test_jacobians_match_finite_differences(self) -> None:
        for pose, landmark in zip(self.POSES, self.LANDMARKS):
            z = range_bearing_prediction(pose, landmark)
            with self.subTest(pose=pose.tolist()):
                np.testing.assert_allclose(
                    range_bearing_jacobian(pose, landmark),
                    numerical_jacobian(lambda l: range_bearing_prediction(pose, l), landmark, angle_rows=(1,)),
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    range_bearing_pose_jacobian(pose, landmark),
                    numerical_jacobian(lambda x: range_bearing_prediction(x, landmark), pose, angle_rows=(1,)),
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    inverse_range_bearing_jacobian(pose, z),
                    numerical_jacobian(lambda m: inverse_range_bearing(pose, m), z),
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    inverse_range_bearing_pose_jacobian(pose, z),
                    numerical_jacobian(lambda x: inverse_range_bearing(x, z), pose),
                    atol=1e-6,
                )
                # The inverse model's Jacobian is the inverse of the forward one.
                np.testing.assert_allclose(
                    inverse_range_bearing_jacobian(pose, z) @ range_bearing_jacobian(pose, landmark), np.eye(2), atol=1e-9
                )


class WeightTests(unittest.TestCase):
    def test_mahalanobis_squared_uses_solve(self) -> None:
        residual = np.array([2.0, 0.5])
        covariance = np.diag([4.0, 0.25])
        with patch("victoria_park_slam.utils.np.linalg.solve", wraps=np.linalg.solve) as solve:
            value = mahalanobis_squared(residual, covariance)
        self.assertAlmostEqual(value, 2.0)
        solve.assert_called_once()

    def test_gaussian_log_likelihood_matches_scipy(self) -> None:
        S = np.array([[0.3, 0.05], [0.05, 0.02]])
        nu = np.array([0.4, -0.1])
        self.assertAlmostEqual(gaussian_log_likelihood(nu, S), multivariate_normal(np.zeros(2), S).logpdf(nu), places=10)

    def test_normalize_log_weights_survives_huge_offsets(self) -> None:
        weights, normalized = normalize_log_weights(np.array([-1000.0, -1001.0, -2000.0]))
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)
        self.assertAlmostEqual(weights[0] / weights[1], math.e)
        self.assertAlmostEqual(float(np.log(np.sum(np.exp(normalized)))), 0.0)

    def test_effective_sample_size(self) -> None:
        self.assertAlmostEqual(effective_sample_size(np.array([0.25, 0.25, 0.25, 0.25])), 4.0)
        self.assertAlmostEqual(effective_sample_size(np.array([1.0, 0.0, 0.0, 0.0])), 1.0)

    def test_systematic_resampling_is_proportional_and_clamped(self) -> None:
        weights = np.random.default_rng(0).random(1000)
        indexes = systematic_resample_indices(weights, rng=np.random.default_rng(3))
        counts = np.bincount(indexes, minlength=1000)
        expected = 1000 * weights / weights.sum()
        self.assertTrue(np.all(np.abs(counts - expected) < 1.0 + 1e-9))

        class AlmostOne:
            def random(self) -> float:
                return float(np.nextafter(1.0, 0.0))

        tenths = np.full(10, 0.1)  # cumulative sum ends at 0.9999999999999999
        self.assertLess(float(np.cumsum(tenths)[-1]), 1.0)
        self.assertTrue(np.all(systematic_resample_indices(tenths, rng=AlmostOne()) < 10))


class TimingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.t_odo = 1.0 + 0.025 * np.arange(41)
        self.dataset = synthetic_dataset(self.t_odo, np.arange(41.0), np.zeros(41), [0.98, 1.1, 1.3125, 1.52])

    def test_scans_before_odometry_are_skipped_and_time_is_conserved(self) -> None:
        steps = list(scan_steps(self.dataset))
        self.assertEqual([step.scan_index for step in steps], [1, 2, 3])
        previous = self.t_odo[0]
        for step in steps:
            self.assertAlmostEqual(sum(c.dt_s for c in step.controls), step.time_s - previous)
            previous = step.time_s
            self.assertTrue(all(c.dt_s > 0.0 for c in step.controls))

    def test_last_interval_is_split_at_the_laser_time(self) -> None:
        _, second, third = list(scan_steps(self.dataset))
        self.assertEqual(second.controls[-1].odometry_index, 12)  # 1.300 s sample
        self.assertAlmostEqual(second.controls[-1].dt_s, 0.0125)
        self.assertEqual(third.controls[0].odometry_index, 12)  # the rest of the same interval
        self.assertAlmostEqual(third.controls[0].dt_s, 0.0125)
        self.assertEqual(third.controls[0].speed_mps, 12.0)  # zero-order hold of sample 12

    def test_start_and_stop_scan(self) -> None:
        steps = list(scan_steps(self.dataset, start_scan=2, stop_scan=3))
        self.assertEqual([step.scan_index for step in steps], [2])
        self.assertEqual(steps[0].controls, ())


class EvaluationTests(unittest.TestCase):
    def test_align_se2_recovers_a_rigid_transform_and_refuses_reflection(self) -> None:
        source = np.random.default_rng(2).normal(size=(50, 2)) * 10.0
        angle = 0.8
        C = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        target = apply_se2(C, [3.0, -4.0], source)
        C_hat, t_hat = align_se2(source, target)
        np.testing.assert_allclose(C_hat, C, atol=1e-12)
        np.testing.assert_allclose(t_hat, [3.0, -4.0], atol=1e-10)
        mirrored = source * np.array([1.0, -1.0])
        C_mirror, t_mirror = align_se2(source, mirrored)
        self.assertAlmostEqual(float(np.linalg.det(C_mirror)), 1.0)
        self.assertGreater(float(np.max(np.abs(apply_se2(C_mirror, t_mirror, source) - mirrored))), 1.0)

    def _straight_line_dataset(self):
        t_odo = np.arange(0.0, 60.0, 0.025)
        gps_t = np.arange(0.0, 60.0, 0.2)
        gps_xy = np.column_stack([2.0 * gps_t, np.zeros_like(gps_t)])
        gps_xy[100] += [0.0, 12.0]  # a 12 m jump
        keep = (gps_t < 30.0) | (gps_t > 40.0)  # a 10 s gap
        keep[175:178] = True  # three isolated fixes inside the gap (35.0-35.4 s)
        return synthetic_dataset(t_odo, np.full(t_odo.size, 2.0), np.zeros(t_odo.size), [0.0], gps_t[keep], gps_xy[keep])

    def test_valid_gps_mask_drops_jumps_and_isolated_fixes(self) -> None:
        dataset = self._straight_line_dataset()
        mask = valid_gps_mask(dataset)
        t = dataset.gps_time_s
        self.assertFalse(mask[np.argmin(np.abs(t - 20.0))])  # the jump
        self.assertTrue(mask[np.argmin(np.abs(t - 20.2))])  # the fix after it is fine
        self.assertFalse(np.any(mask[(t > 34.9) & (t < 35.5)]))  # isolated short run in the gap
        self.assertEqual(np.count_nonzero(~mask), 4)

    def test_evaluation_aligns_and_dead_reckoning_is_exact_on_a_line(self) -> None:
        dataset = self._straight_line_dataset()
        t, poses = dead_reckon(dataset, use_lever_arm=False)
        np.testing.assert_allclose(poses[-1], [2.0 * t[-1], 0.0, 0.0], atol=1e-9)
        rotated = apply_se2(np.array([[0.0, -1.0], [1.0, 0.0]]), [5.0, 5.0], poses[:, :2])
        result = evaluate_against_gps(dataset, t, rotated)
        self.assertLess(result.rmse_m, 1e-6)
        self.assertAlmostEqual(abs(result.rotation_deg), 90.0, places=6)


@unittest.skipIf(REAL_DATA is None, FETCH_HINT)
class RealDataTests(unittest.TestCase):
    def test_real_dataset_facts(self) -> None:
        dataset = real_dataset()
        self.assertEqual(dataset.laser_ranges_m.shape, (7249, 361))
        self.assertAlmostEqual(float(np.mean(np.isnan(dataset.laser_ranges_m))), 0.83, delta=0.01)
        self.assertEqual(int(np.count_nonzero(dataset.laser_raw > 0x1FFF)), 5)
        self.assertLess(dataset.laser_time_s[0], dataset.odometry_time_s[0])
        self.assertAlmostEqual(float(np.median(np.diff(dataset.odometry_time_s))), 0.025, places=6)

    def test_encoder_correction_halves_dead_reckoning_error(self) -> None:
        dataset = real_dataset()
        results = dead_reckoning_comparison(dataset, stop_time_s=dataset.odometry_time_s[0] + 200.0)
        corrected = results["DR (encoder correction + lever arm)"].rmse_m
        uncorrected = results["DR (no corrections)"].rmse_m
        self.assertLess(corrected, 14.0)
        self.assertGreater(uncorrected, 25.0)

    def test_first_yielded_scan_is_one(self) -> None:
        first = next(scan_steps(real_dataset()))
        self.assertEqual(first.scan_index, 1)
        self.assertAlmostEqual(sum(c.dt_s for c in first.controls), first.time_s - real_dataset().odometry_time_s[0])

    def test_run_all_does_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.assertEqual(run_all.main(["--scans", "40", "--out", tmp_dir]), 0)
            self.assertTrue((Path(tmp_dir) / "summary.md").exists())


# ---------------------------------------------------------------------------
# Student deliverables (skip while they are stubs)
# ---------------------------------------------------------------------------


class ExtractorTests(unittest.TestCase):
    def test_isolated_trunks_give_center_range_and_vehicle_bearing(self) -> None:
        left_tree = (*polar(10.0, 20.0), 0.20)  # center 10 m away, 20 deg to the LEFT, 0.4 m trunk
        right_tree = (*polar(15.0, -35.0), 0.15)  # 35 deg to the RIGHT, 0.3 m trunk
        scan = synthetic_scan(circles=[left_tree, right_tree])
        observations = call_or_skip(self, landmarks.extract_tree_landmarks, scan)
        self.assertTrue(all(isinstance(obs, TreeObservation) for obs in observations))
        self.assertEqual(len(observations), 2)
        observations = sorted(observations, key=lambda obs: obs.bearing_rad)
        for obs, (range_m, bearing_deg, diameter) in zip(observations, [(15.0, -35.0, 0.3), (10.0, 20.0, 0.4)]):
            self.assertAlmostEqual(obs.range_m, range_m, delta=0.3)
            self.assertAlmostEqual(math.degrees(obs.bearing_rad), bearing_deg, delta=1.0)
            self.assertAlmostEqual(obs.diameter_m, diameter, delta=0.2)

    def test_walls_and_empty_scans_give_nothing(self) -> None:
        wall = synthetic_scan(segments=[((20.0, -5.0), (20.0, 5.0))])
        self.assertEqual(call_or_skip(self, landmarks.extract_tree_landmarks, wall), [])
        self.assertEqual(call_or_skip(self, landmarks.extract_tree_landmarks, np.full(361, np.nan)), [])

    def test_pose_guess_does_not_change_the_observations(self) -> None:
        scan = synthetic_scan(circles=[(*polar(12.0, 5.0), 0.25)])
        plain = call_or_skip(self, landmarks.extract_tree_landmarks, scan)
        guessed = landmarks.extract_tree_landmarks(scan, pose_guess=np.array([100.0, -50.0, 2.0]))
        self.assertEqual([obs.z.tolist() for obs in plain], [obs.z.tolist() for obs in guessed])

    @unittest.skipIf(REAL_DATA is None, FETCH_HINT)
    def test_real_scans_give_plausible_trees(self) -> None:
        dataset = real_dataset()
        observations = [call_or_skip(self, landmarks.extract_tree_landmarks, dataset.laser_ranges_m[i]) for i in range(0, 1000, 5)]
        counts = np.array([len(scan) for scan in observations])
        everything = [obs for scan in observations for obs in scan]
        self.assertTrue(2 <= np.median(counts) <= 25, f"median detections per scan {np.median(counts)}")
        self.assertTrue(0.1 <= np.median([obs.diameter_m for obs in everything]) <= 1.0)
        bearings = np.array([obs.bearing_rad for obs in everything])
        self.assertTrue(np.all(np.abs(bearings) <= np.pi / 2 + 1e-9))


class LandmarkEKFTests(unittest.TestCase):
    pose = np.array([1.0, 2.0, 0.3])
    z = np.array([12.0, 0.4])

    def _initialize(self, time_s: float = 5.0):
        return call_or_skip(self, fastslam.LandmarkEKF.initialize, self.pose, self.z, R_DEFAULT, time_s=time_s)

    def test_initialize_uses_the_inverse_model(self) -> None:
        landmark = self._initialize()
        self.assertIsInstance(landmark, fastslam.LandmarkEKF)
        np.testing.assert_allclose(landmark.mean_xy_m, inverse_range_bearing(self.pose, self.z), atol=1e-12)
        G = inverse_range_bearing_jacobian(self.pose, self.z)
        np.testing.assert_allclose(landmark.covariance, G @ R_DEFAULT @ G.T, atol=1e-12)
        self.assertEqual(landmark.observation_count, 1)
        self.assertEqual(landmark.track_state, LandmarkTrackState.TENTATIVE)
        self.assertEqual((landmark.first_seen_s, landmark.last_seen_s), (5.0, 5.0))

    def test_innovation_wraps_the_bearing(self) -> None:
        pose = np.zeros(3)
        landmark = fastslam.LandmarkEKF(mean_xy_m=np.array([-10.0, 0.01]), covariance=0.01 * np.eye(2))
        nu, H_l, S = call_or_skip(self, landmark.innovation, pose, np.array([10.0, -np.pi + 0.002]), R_DEFAULT)
        self.assertLess(abs(nu[1]), 0.01)
        H = range_bearing_jacobian(pose, landmark.mean_xy_m)
        np.testing.assert_allclose(H_l, H, atol=1e-12)
        np.testing.assert_allclose(S, H @ landmark.covariance @ H.T + R_DEFAULT, atol=1e-12)

    def test_update_returns_a_new_landmark_with_the_kalman_posterior(self) -> None:
        landmark = fastslam.LandmarkEKF(
            mean_xy_m=inverse_range_bearing(self.pose, self.z) + np.array([0.3, -0.2]),
            covariance=np.array([[0.4, 0.1], [0.1, 0.3]]),
            observation_count=2,
            first_seen_s=1.0,
            last_seen_s=2.0,
        )
        before_mean, before_cov = landmark.mean_xy_m.copy(), landmark.covariance.copy()
        updated = call_or_skip(self, landmark.updated, self.pose, self.z, R_DEFAULT, time_s=3.0)
        np.testing.assert_array_equal(landmark.mean_xy_m, before_mean)  # copy-on-write
        np.testing.assert_array_equal(landmark.covariance, before_cov)
        self.assertIsNot(updated, landmark)
        self.assertEqual((updated.observation_count, updated.first_seen_s, updated.last_seen_s), (3, 1.0, 3.0))

        H = range_bearing_jacobian(self.pose, before_mean)
        nu = self.z - range_bearing_prediction(self.pose, before_mean)
        nu[1] = wrap_angle(nu[1])
        S = H @ before_cov @ H.T + R_DEFAULT
        K = before_cov @ H.T @ np.linalg.inv(S)
        np.testing.assert_allclose(updated.mean_xy_m, before_mean + K @ nu, atol=1e-10)
        np.testing.assert_allclose(updated.covariance, (np.eye(2) - K @ H) @ before_cov, atol=1e-10)
        np.testing.assert_allclose(updated.covariance, updated.covariance.T, atol=1e-14)
        self.assertLess(np.trace(updated.covariance), np.trace(before_cov))


def make_slam(num_particles: int = 4, **settings) -> fastslam.FastSLAM:
    settings.setdefault("rng", np.random.default_rng(0))
    return fastslam.FastSLAM.from_initial_pose(num_particles, (0.0, 0.0, 0.0), **settings)


def landmark_at(xy, *, covariance=0.01, count=1, state=LandmarkTrackState.TENTATIVE, seen=0.0):
    return fastslam.LandmarkEKF(
        mean_xy_m=np.asarray(xy, dtype=np.float64),
        covariance=covariance * np.eye(2),
        observation_count=count,
        first_seen_s=seen,
        last_seen_s=seen,
        track_state=state,
    )


def measurement_at_d2(pose, landmark, d2: float, R=R_DEFAULT) -> np.ndarray:
    """A measurement whose Mahalanobis distance to ``landmark`` is exactly ``d2`` (offset in range)."""
    H = range_bearing_jacobian(pose, landmark.mean_xy_m)
    S = H @ landmark.covariance @ H.T + R
    direction = np.array([1.0, 0.0])
    scale = math.sqrt(d2 / (direction @ np.linalg.solve(S, direction)))
    return range_bearing_prediction(pose, landmark.mean_xy_m) + scale * direction


class FastSLAMPredictTests(unittest.TestCase):
    def test_noise_free_prediction_uses_the_corrected_laser_model(self) -> None:
        motion = MotionModelConfig(control_noise_covariance=np.diag([1e-14, 1e-14]))
        slam = make_slam(3, motion=motion)
        call_or_skip(self, slam.predict, 4.0, 0.2, 0.5)
        expected = predict_laser_pose(np.zeros(3), encoder_to_centerline_speed(4.0, 0.2), 0.2, 0.5)
        for particle in slam.particles:
            np.testing.assert_allclose(particle.pose_xy_theta, expected, atol=1e-5)

        raw = MotionModelConfig(control_noise_covariance=np.diag([1e-14, 1e-14]), use_encoder_correction=False)
        slam = make_slam(2, motion=raw)
        slam.predict(4.0, 0.2, 0.5)
        np.testing.assert_allclose(slam.particles[0].pose_xy_theta, predict_laser_pose(np.zeros(3), 4.0, 0.2, 0.5), atol=1e-5)

    def test_noisy_prediction_spreads_reproducibly(self) -> None:
        def run(seed: int) -> np.ndarray:
            slam = make_slam(200, rng=np.random.default_rng(seed))
            for _ in range(40):
                call_or_skip(self, slam.predict, 3.0, 0.0, 0.025)
            return np.array([p.pose_xy_theta for p in slam.particles])

        first, again, other = run(7), run(7), run(8)
        np.testing.assert_array_equal(first, again)
        self.assertFalse(np.array_equal(first, other))
        self.assertGreater(np.std(first[:, 2]), 1e-4)
        self.assertAlmostEqual(float(np.mean(first[:, 0])), 3.0, delta=0.15)


class FastSLAMAssociationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.slam = make_slam(1)
        self.particle = self.slam.particles[0]
        self.pose = self.particle.pose_xy_theta
        self.a = landmark_at([10.0, 0.0])
        self.b = landmark_at([12.0, 15.0])
        self.particle.landmarks = {3: self.a, 7: self.b}

    def test_two_gates(self) -> None:
        z = np.array([
            measurement_at_d2(self.pose, self.a, 1.0),
            measurement_at_d2(self.pose, self.b, 9.0),  # between the gates
            [40.0, -1.0],  # far from everything
        ])
        result = call_or_skip(self, self.slam.associate, self.particle, z)
        self.assertEqual([r.kind for r in result], [AssociationKind.MATCH, AssociationKind.DISCARD, AssociationKind.NEW])
        self.assertEqual(result[0].landmark_id, 3)
        self.assertAlmostEqual(result[0].mahalanobis_sq, 1.0, places=6)
        self.assertEqual(result[1].landmark_id, 7)

    def test_one_landmark_absorbs_at_most_one_observation_per_scan(self) -> None:
        z = np.array([measurement_at_d2(self.pose, self.a, 2.0), measurement_at_d2(self.pose, self.a, 0.5)])
        result = call_or_skip(self, self.slam.associate, self.particle, z)
        self.assertEqual([r.kind for r in result], [AssociationKind.DISCARD, AssociationKind.MATCH])
        self.assertEqual(result[1].landmark_id, 3)

    def test_empty_map_and_empty_scan(self) -> None:
        self.particle.landmarks = {}
        result = call_or_skip(self, self.slam.associate, self.particle, np.array([[5.0, 0.1], [8.0, -0.2]]))
        self.assertEqual([r.kind for r in result], [AssociationKind.NEW, AssociationKind.NEW])
        self.assertEqual(self.slam.associate(self.particle, np.zeros((0, 2))), [])


def observation(z) -> TreeObservation:
    return TreeObservation(range_m=float(z[0]), bearing_rad=float(z[1]), diameter_m=0.4)


class FastSLAMUpdateTests(unittest.TestCase):
    def test_better_aligned_particle_gains_weight(self) -> None:
        slam = make_slam(2, resample_threshold=0.0)
        tree = landmark_at([10.0, 0.0])
        for particle in slam.particles:
            particle.landmarks = {0: tree}
        slam.next_landmark_id = 1
        slam.particles[1].pose_xy_theta = np.array([0.4, 0.0, 0.0])
        z = range_bearing_prediction(np.zeros(3), tree.mean_xy_m)
        call_or_skip(self, slam.update, [observation(z)], 1.0)
        weights = slam.weights()
        self.assertGreater(weights[0], weights[1])
        self.assertAlmostEqual(float(np.log(np.sum(np.exp(slam.log_weights())))), 0.0, places=9)
        for particle in slam.particles:
            self.assertEqual(particle.landmarks[0].observation_count, 2)
            self.assertEqual(len(particle.trajectory), 2)

    def test_new_landmarks_share_ids_across_particles(self) -> None:
        slam = make_slam(3, resample_threshold=0.0)
        call_or_skip(self, slam.update, [observation([8.0, 0.2]), observation([20.0, -0.5])], 0.5)
        self.assertEqual(slam.next_landmark_id, 2)
        for particle in slam.particles:
            self.assertEqual(sorted(particle.landmarks), [0, 1])
            self.assertTrue(all(lm.track_state == LandmarkTrackState.TENTATIVE for lm in particle.landmarks.values()))
        # Every particle spawned both landmarks, so the weights stay equal.
        np.testing.assert_allclose(slam.weights(), np.full(3, 1.0 / 3.0))

    def test_confirmation_and_tentative_timeout(self) -> None:
        slam = make_slam(1, resample_threshold=0.0, confirm_associations=3, tentative_timeout_s=1.0)
        z = np.array([10.0, 0.1])
        call_or_skip(self, slam.update, [observation(z)], 0.0)
        call_or_skip(self, slam.update, [observation([30.0, -0.8])], 0.0)  # a second, short-lived tree
        states = []
        for k in range(1, 4):
            slam.update([observation(z)], 0.2 * k)
            states.append(slam.particles[0].landmarks[0].track_state)
        self.assertEqual(states, [LandmarkTrackState.TENTATIVE, LandmarkTrackState.TENTATIVE, LandmarkTrackState.CONFIRMED])
        self.assertEqual(slam.particles[0].landmarks[0].observation_count, 4)
        self.assertIn(1, slam.particles[0].landmarks)
        slam.update([], 5.0)  # nothing seen for 5 s
        self.assertNotIn(1, slam.particles[0].landmarks)  # tentative: deleted
        self.assertIn(0, slam.particles[0].landmarks)  # confirmed: kept


class FastSLAMResampleTests(unittest.TestCase):
    def test_resampling_resets_weights_and_keeps_maps_independent(self) -> None:
        slam = make_slam(4)
        for index, particle in enumerate(slam.particles):
            particle.pose_xy_theta = np.array([float(index), 0.0, 0.0])
            particle.landmarks = {index: landmark_at([float(index), 5.0])}
        log_w = np.log(np.array([0.97, 0.01, 0.01, 0.01]))
        for particle, value in zip(slam.particles, log_w):
            particle.log_weight = float(value)
        call_or_skip(self, slam.resample)
        self.assertEqual(len(slam.particles), 4)
        np.testing.assert_allclose(slam.weights(), np.full(4, 0.25))
        self.assertGreaterEqual(sum(p.pose_xy_theta[0] == 0.0 for p in slam.particles), 3)
        first, second = slam.particles[0], slam.particles[1]
        self.assertIsNot(first.landmarks, second.landmarks)
        self.assertIsNot(first.pose_xy_theta, second.pose_xy_theta)
        first.landmarks[99] = landmark_at([0.0, 0.0])
        self.assertNotIn(99, second.landmarks)


class FastSLAMSyntheticRunTests(unittest.TestCase):
    """A small closed loop through a field of trees, with biased odometry."""

    def test_fastslam_beats_dead_reckoning_on_a_loop(self) -> None:
        rng = np.random.default_rng(11)
        center = np.array([-3.78, 16.5])  # the loop's center for the controls below
        trees = np.array([center + r * np.array([math.cos(a), math.sin(a)]) for r in (9.0, 25.0) for a in np.linspace(0, 2 * math.pi, 14, endpoint=False)])
        dt, steps_per_scan, scans = 0.025, 8, 150
        speed_true, steering_true = 4.0, 0.165  # one loop of radius ~17 m in about 133 scans
        # Q_u has to cover the odometry bias below; with (0.3 m/s, 3 deg) the
        # particles cannot keep up and the loop closes with duplicate trees.
        motion = MotionModelConfig(control_noise_covariance=np.diag([1.0**2, math.radians(5.0) ** 2]))
        slam = make_slam(30, motion=motion, rng=np.random.default_rng(5))
        truth = np.zeros(3)
        dead_reckoning = np.zeros(3)
        R = slam.measurement_covariance
        for scan in range(scans):
            for _ in range(steps_per_scan):
                truth = predict_laser_pose(truth, encoder_to_centerline_speed(speed_true, steering_true), steering_true, dt)
                speed_meas, steering_meas = speed_true * 1.04, steering_true + math.radians(1.0)  # biased odometry
                dead_reckoning = predict_laser_pose(
                    dead_reckoning, encoder_to_centerline_speed(speed_meas, steering_meas), steering_meas, dt
                )
                call_or_skip(self, slam.predict, speed_meas, steering_meas, dt)
            observations = []
            for tree in trees:
                z = range_bearing_prediction(truth, tree)
                if z[0] < 30.0 and abs(z[1]) < math.pi / 2:
                    noisy = z + rng.multivariate_normal(np.zeros(2), R)
                    observations.append(observation(noisy))
            call_or_skip(self, slam.update, observations, steps_per_scan * dt * (scan + 1))
        error = float(np.hypot(*(slam.estimate_pose()[:2] - truth[:2])))
        dr_error = float(np.hypot(*(dead_reckoning[:2] - truth[:2])))
        self.assertGreater(dr_error, 5.0)
        self.assertLess(error, 0.15 * dr_error, f"FastSLAM error {error:.2f} m vs dead reckoning {dr_error:.2f} m")
        total, confirmed = slam.landmark_counts()
        self.assertLessEqual(confirmed, 1.5 * len(trees))
        self.assertGreater(confirmed, len(trees) // 2)


if __name__ == "__main__":
    unittest.main()
