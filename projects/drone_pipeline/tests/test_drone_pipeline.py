"""Tests for the Project 1 starter package.

Two kinds of tests live here:

* Support-code tests (loaders, parameter parsing, truth interpolation,
  geometry helpers). These must always pass.
* Student-facing tests for the Part A/B/C deliverables. They skip with a
  message while a function is still a ``NotImplementedError`` stub, and start
  checking your implementation as soon as it exists.

Tests that need the real dataset skip until it has been downloaded.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
from scipy.io import savemat

from drone_pipeline import filters, mechanization, observation
from drone_pipeline.config import CAMERA_MATRIX, DIST_COEFFS, TAG_IDS
from drone_pipeline.data_paths import DATA_ENV_VAR, DatasetNotFoundError, find_dataset_dir
from drone_pipeline.datasets import (
    ParameterFileError,
    default_dataset_path,
    default_drone_parameters,
    drone_data_dir,
    interpolate_vicon_truth,
    load_drone_dataset,
    load_drone_parameters,
    parse_drone_parameters,
)
from drone_pipeline.types import DronePacket, PoseObservation, TruthSample
from drone_pipeline.utils import (
    Rx,
    Ry,
    Rz,
    invert_transform,
    make_transform,
    rotation_matrix_to_rpy,
    rpy_to_rotation_matrix,
    wrap_angle,
    wrap_angles,
)

FETCH_HINT = "drone data not downloaded; run `uv run python scripts/fetch_data.py drone-data`"

PARAMETERS_FIXTURE = """% Camera Matrix (zero-indexed):
[314.1779 0         199.4848; ...
0         314.2218  113.7838; ...
0         0         1];

% Distortion parameters (k1, k2, p1, p2, k3):
[-0.438607 0.248625 0.00072 -0.000476 -0.0911]
% the k's are radial distortion parameters
% the p's are tangential distortion parameters

% Camera-IMU Calibration (see attached images for details):
XYZ = [-0.04, 0.0, -0.03];
Yaw = pi/4;

% Tag ids:
[0, 12, 24, 36, 48, 60, 72, 84,  96;
 1, 13, 25, 37, 49, 61, 73, 85,  97;
 2, 14, 26, 38, 50, 62, 74, 86,  98;
 3, 15, 27, 39, 51, 63, 75, 87,  99;
 4, 16, 28, 40, 52, 64, 76, 88, 100;
 5, 17, 29, 41, 53, 65, 77, 89, 101;
 6, 18, 30, 42, 54, 66, 78, 90, 102;
 7, 19, 31, 43, 55, 67, 79, 91, 103;
 8, 20, 32, 44, 56, 68, 80, 92, 104;
 9, 21, 33, 45, 57, 69, 81, 93, 105;
10, 22, 34, 46, 58, 70, 82, 94, 106;
11, 23, 35, 47, 59, 71, 83, 95, 107];
"""

IMAGE_WIDTH, IMAGE_HEIGHT = 376, 240
HOVER_ACC = np.array([0.0, 0.0, 9.81])


def _real_data_dir() -> Path | None:
    try:
        return drone_data_dir()
    except DatasetNotFoundError:
        return None


REAL_DATA = _real_data_dir()


def call_or_skip(test: unittest.TestCase, function, *args, **kwargs):
    """Call a student deliverable; skip the test while it is still a stub."""
    try:
        return function(*args, **kwargs)
    except NotImplementedError as error:
        name = getattr(function, "__qualname__", repr(function))
        test.skipTest(f"{name} is not implemented yet (starter stub): {error}")


def _vicon(n: int) -> tuple[np.ndarray, np.ndarray]:
    time = np.linspace(0.0, 1.0, n)
    vicon = np.zeros((12, n))
    vicon[0] = time  # x moves at 1 m/s
    return vicon, time


def _write_legacy_dataset(path: Path, n: int = 3) -> None:
    """studentdata0 layout: a cell array of packets with ``drpy`` and an image."""
    packets = np.empty((1, n), dtype=object)
    for i in range(n):
        packets[0, i] = {
            "img": np.zeros((4, 6), dtype=np.uint8),
            "id": np.array([7], dtype=np.uint8) if i else np.array([], dtype=np.uint8),
            "p1": np.ones((2, 1)) if i else np.array([], dtype=np.uint8),
            "p2": np.ones((2, 1)) if i else np.array([], dtype=np.uint8),
            "p3": np.ones((2, 1)) if i else np.array([], dtype=np.uint8),
            "p4": np.ones((2, 1)) if i else np.array([], dtype=np.uint8),
            "t": 0.5 * i,
            "rpy": np.array([0.0, 0.0, 0.1 * i]),
            "drpy": np.array([0.01, 0.02, 0.03]),
            "acc": np.array([0.0, 0.0, 9.81]),
        }
    vicon, time = _vicon(5)
    savemat(path, {"data": packets, "vicon": vicon, "time": time})


def _write_current_dataset(path: Path, n: int = 3) -> None:
    """studentdata1-7 layout: a struct array with ``omg``, ``p0`` and ``is_ready``."""
    fields = ["is_ready", "t", "rpy", "omg", "acc", "id", "p0", "p1", "p2", "p3", "p4"]
    packets = np.zeros((1, n), dtype=[(name, object) for name in fields])
    for i in range(n):
        packets[0, i] = (
            1,
            0.5 * i,
            np.array([0.0, 0.0, 0.1 * i]),
            np.array([0.1, 0.2, 0.3]),
            np.array([0.0, 0.0, 9.81]),
            np.array([3, 4], dtype=np.uint8),
            np.ones((2, 2)),
            np.ones((2, 2)),
            np.ones((2, 2)),
            np.ones((2, 2)),
            np.ones((2, 2)),
        )
    vicon, time = _vicon(5)
    savemat(path, {"data": packets, "vicon": vicon, "time": time})


# ---------------------------------------------------------------------------
# Support code
# ---------------------------------------------------------------------------


class SyntheticDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "drone-data"
        self.root.mkdir()
        (self.root / "parameters.txt").write_text(PARAMETERS_FIXTURE)
        _write_legacy_dataset(self.root / "studentdata0.mat")
        _write_current_dataset(self.root / "studentdata1.mat")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_legacy_schema_reads_drpy_into_omg(self) -> None:
        dataset = load_drone_dataset(self.root / "studentdata0.mat")
        self.assertEqual(len(dataset.packets), 3)
        packet = dataset.packets[1]
        self.assertEqual(packet.rate_field, "drpy")
        np.testing.assert_allclose(packet.omg, [0.01, 0.02, 0.03])
        np.testing.assert_allclose(packet.drpy, packet.omg)  # backward-compatible alias
        self.assertEqual(packet.img.shape, (4, 6))
        self.assertIsNone(packet.p0)
        self.assertEqual(dataset.packets[0].id.size, 0)
        self.assertEqual(dataset.packets[0].num_tags, 0)

    def test_current_schema_reads_omg_p0_and_is_ready(self) -> None:
        dataset = load_drone_dataset(self.root / "studentdata1.mat")
        packet = dataset.packets[0]
        self.assertEqual(packet.rate_field, "omg")
        np.testing.assert_allclose(packet.omg, [0.1, 0.2, 0.3])
        self.assertIsNone(packet.img)
        self.assertTrue(packet.is_ready)
        self.assertEqual(packet.p0.shape, (2, 2))
        np.testing.assert_array_equal(packet.id, [3, 4])
        self.assertEqual(dataset.vicon.shape, (12, 5))

    def test_drpy_alias_is_read_only(self) -> None:
        packet = load_drone_dataset(self.root / "studentdata1.mat").packets[0]
        with self.assertRaises(AttributeError):
            packet.drpy = np.zeros(3)

    def test_index_and_default_parameter_lookup_use_the_data_dir(self) -> None:
        with mock.patch.dict(os.environ, {DATA_ENV_VAR: str(self.root.parent)}):
            with mock.patch("drone_pipeline.data_paths.Path.cwd", return_value=Path(self._tmp.name)):
                self.assertEqual(default_dataset_path(1), self.root / "studentdata1.mat")
                self.assertEqual(load_drone_dataset(1).path, self.root / "studentdata1.mat")
                params = load_drone_parameters()
        self.assertAlmostEqual(params.camera_matrix[0, 0], 314.1779, places=4)
        self.assertEqual(params.T_camera_body.shape, (4, 4))

    def test_interpolate_vicon_truth_matches_requested_times(self) -> None:
        dataset = load_drone_dataset(self.root / "studentdata1.mat")
        truth = interpolate_vicon_truth(dataset, np.array([0.25, 0.5]))
        self.assertEqual(len(truth), 2)
        np.testing.assert_allclose([sample.position[0] for sample in truth], [0.25, 0.5])
        self.assertTrue(all(sample.valid for sample in truth))

    def test_truth_outside_vicon_span_is_nan_and_invalid(self) -> None:
        dataset = load_drone_dataset(self.root / "studentdata1.mat")
        truth = interpolate_vicon_truth(dataset, np.array([-0.1, 0.0, 1.0, 1.2]))
        self.assertEqual([sample.valid for sample in truth], [False, True, True, False])
        self.assertTrue(np.all(np.isnan(truth[0].position)))
        self.assertTrue(np.all(np.isnan(truth[-1].rpy)))
        self.assertAlmostEqual(truth[2].position[0], 1.0)


class ParameterFileTests(unittest.TestCase):
    def test_parser_reads_every_entry(self) -> None:
        params = parse_drone_parameters(PARAMETERS_FIXTURE)
        np.testing.assert_allclose(params.camera_matrix, CAMERA_MATRIX)
        np.testing.assert_allclose(params.dist_coeffs, DIST_COEFFS)
        np.testing.assert_allclose(params.translation_m, [-0.04, 0.0, -0.03])
        self.assertAlmostEqual(params.yaw_offset_rad, np.pi / 4)
        np.testing.assert_array_equal(params.tag_ids, TAG_IDS)

    def test_camera_body_transform_matches_documented_convention(self) -> None:
        params = parse_drone_parameters(PARAMETERS_FIXTURE)
        expected = make_transform(Rz(-np.pi / 4) @ Rx(np.pi), [-0.04, 0.0, -0.03])
        np.testing.assert_allclose(params.T_camera_body, expected, atol=1e-12)
        # Body z (up) must map to camera -z: the camera looks down.
        np.testing.assert_allclose(params.T_camera_body[:3, :3] @ [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], atol=1e-12)
        # The camera origin sits about 3 cm below the IMU in body coordinates.
        camera_in_body = invert_transform(params.T_camera_body)[:3, 3]
        self.assertAlmostEqual(camera_in_body[2], -0.03)

    def test_parser_values_are_read_from_the_text(self) -> None:
        text = PARAMETERS_FIXTURE.replace("Yaw = pi/4;", "Yaw = -pi/6;").replace("314.1779", "300.5")
        params = parse_drone_parameters(text)
        self.assertAlmostEqual(params.yaw_offset_rad, -np.pi / 6)
        self.assertAlmostEqual(params.camera_matrix[0, 0], 300.5)

    def test_defaults_match_the_fixture(self) -> None:
        parsed = parse_drone_parameters(PARAMETERS_FIXTURE)
        defaults = default_drone_parameters()
        np.testing.assert_allclose(parsed.T_camera_body, defaults.T_camera_body)
        np.testing.assert_allclose(parsed.camera_matrix, defaults.camera_matrix)

    def test_missing_or_malformed_entries_raise_a_clear_error(self) -> None:
        broken = {
            "Distortion": PARAMETERS_FIXTURE.replace("% Distortion parameters", "% Lens"),
            "Yaw": PARAMETERS_FIXTURE.replace("Yaw = pi/4;", "Yaw = import os;"),
            "XYZ": PARAMETERS_FIXTURE.replace("XYZ = [-0.04, 0.0, -0.03];", "XYZ = [-0.04, 0.0];"),
            "Camera Matrix": PARAMETERS_FIXTURE.replace("0         0         1];", "0 0];"),
        }
        for item, text in broken.items():
            with self.subTest(item=item):
                with self.assertRaises(ParameterFileError) as caught:
                    parse_drone_parameters(text, source="fixture.txt")
                self.assertIn(item, str(caught.exception))
                self.assertIn("fixture.txt", str(caught.exception))


class DataPathTests(unittest.TestCase):
    def test_env_var_wins_and_missing_data_explains_how_to_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            (Path(tmp_dir) / "toy").mkdir()
            with mock.patch.dict(os.environ, {DATA_ENV_VAR: tmp_dir}):
                self.assertEqual(find_dataset_dir("toy"), Path(tmp_dir) / "toy")
            with self.assertRaises(DatasetNotFoundError) as caught:
                find_dataset_dir("toy", explicit=Path(tmp_dir) / "missing")
        self.assertIn("scripts/fetch_data.py toy", str(caught.exception))

    def test_relative_path_is_found_from_a_subfolder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "data" / "toy").mkdir(parents=True)
            (root / "deep" / "er").mkdir(parents=True)
            with mock.patch("drone_pipeline.data_paths.Path.cwd", return_value=root / "deep" / "er"):
                with mock.patch.dict(os.environ, {}, clear=True):
                    self.assertEqual(find_dataset_dir("toy"), root / "data" / "toy")


class GeometryTests(unittest.TestCase):
    def test_wrap_angles_is_vectorized_and_half_open(self) -> None:
        values = np.array([0.0, np.pi, -np.pi, 3 * np.pi, -1e-17, 7.0, -7.0])
        wrapped = wrap_angles(values)
        self.assertEqual(wrapped.shape, values.shape)
        self.assertTrue(np.all(wrapped >= -np.pi) and np.all(wrapped < np.pi))
        np.testing.assert_allclose(wrapped[[1, 2, 3]], -np.pi)
        np.testing.assert_allclose(wrapped, [wrap_angle(v) for v in values])
        self.assertEqual(wrap_angles(np.zeros((4, 3))).shape, (4, 3))

    def test_rpy_matrix_uses_the_zxy_sequence(self) -> None:
        roll, pitch, yaw = 0.2, -0.3, 1.1
        np.testing.assert_allclose(rpy_to_rotation_matrix([roll, pitch, yaw]), Rz(yaw) @ Rx(roll) @ Ry(pitch))

    def test_rotation_matrix_to_rpy_round_trips(self) -> None:
        rng = np.random.default_rng(4)
        for _ in range(100):
            rpy = rng.uniform([-1.4, -np.pi, -np.pi], [1.4, np.pi, np.pi])
            np.testing.assert_allclose(rotation_matrix_to_rpy(rpy_to_rotation_matrix(rpy)), rpy, atol=1e-10)

    def test_invert_transform(self) -> None:
        T = make_transform(rpy_to_rotation_matrix([0.1, 0.2, 0.3]), [1.0, -2.0, 0.5])
        np.testing.assert_allclose(invert_transform(T) @ T, np.eye(4), atol=1e-12)


# ---------------------------------------------------------------------------
# Part B (student deliverables; skipped while stubbed)
# ---------------------------------------------------------------------------


def synthetic_packet(
    tag_map: dict[int, np.ndarray], T_world_body: np.ndarray, T_camera_body: np.ndarray, t: float = 1.0
) -> DronePacket:
    """Project the mat into the camera at a known pose (zero distortion, no noise)."""
    T_camera_world = invert_transform(T_world_body @ invert_transform(T_camera_body))
    rvec, _ = cv2.Rodrigues(T_camera_world[:3, :3])
    tvec = T_camera_world[:3, 3]
    ids, corners = [], []
    for tag_id, world in tag_map.items():
        in_camera = (T_camera_world[:3, :3] @ world.T).T + tvec
        if np.any(in_camera[:, 2] <= 0.1):
            continue
        pixels, _ = cv2.projectPoints(world, rvec, tvec, CAMERA_MATRIX, np.zeros(5))
        pixels = pixels.reshape(4, 2)
        if np.all((pixels[:, 0] > 0) & (pixels[:, 0] < IMAGE_WIDTH) & (pixels[:, 1] > 0) & (pixels[:, 1] < IMAGE_HEIGHT)):
            ids.append(tag_id)
            corners.append(pixels)
    corners = np.array(corners).reshape(-1, 4, 2)
    fields = {f"p{k + 1}": corners[:, k, :].T for k in range(4)}
    return DronePacket(
        id=np.array(ids, dtype=np.int64),
        t=t,
        rpy=np.zeros(3),
        omg=np.zeros(3),
        acc=HOVER_ACC.copy(),
        **fields,
    )


class PartBTests(unittest.TestCase):
    def test_tag_map_geometry(self) -> None:
        tag_map = call_or_skip(self, observation.build_apriltag_corner_map)
        self.assertEqual(sorted(tag_map), list(range(108)))
        for corners in tag_map.values():
            self.assertEqual(corners.shape, (4, 3))
            np.testing.assert_allclose(corners[:, 2], 0.0)
            # p1 -> p2 runs along +y, p2 -> p3 along -x: a 0.152 m square.
            np.testing.assert_allclose(corners[1] - corners[0], [0.0, 0.152, 0.0], atol=1e-12)
            np.testing.assert_allclose(corners[2] - corners[1], [-0.152, 0.0, 0.0], atol=1e-12)
        # Tag 0's top-left corner (p4) is the world origin.
        np.testing.assert_allclose(tag_map[0][3], [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(tag_map[0][0], [0.152, 0.0, 0.0], atol=1e-12)
        # Rows step 0.304 m in x; the wide gap appears between columns 3 and 4.
        np.testing.assert_allclose(tag_map[1][3], [0.304, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(tag_map[36][3][1] - tag_map[24][3][1], 0.152 + 0.178, atol=1e-12)
        np.testing.assert_allclose(tag_map[24][3][1] - tag_map[12][3][1], 0.304, atol=1e-12)

    def test_estimate_pose_recovers_a_synthetic_pose(self) -> None:
        tag_map = call_or_skip(self, observation.build_apriltag_corner_map)
        params = default_drone_parameters()
        rpy = np.array([0.05, -0.08, 0.6])
        T_world_body = make_transform(rpy_to_rotation_matrix(rpy), [1.6, 1.3, 1.2])
        packet = synthetic_packet(tag_map, T_world_body, params.T_camera_body, t=2.5)
        self.assertGreaterEqual(packet.id.size, 4)
        pose = call_or_skip(
            self, observation.estimate_pose, packet, params.camera_matrix, np.zeros(5), tag_map, params.T_camera_body
        )
        self.assertIsInstance(pose, PoseObservation)
        self.assertAlmostEqual(pose.t, 2.5)
        np.testing.assert_allclose(pose.position, [1.6, 1.3, 1.2], atol=1e-5)
        np.testing.assert_allclose(pose.rpy, rpy, atol=1e-5)
        self.assertEqual(pose.num_points, 4 * packet.id.size)
        self.assertLess(pose.reprojection_error, 1e-3)

    def test_estimate_pose_returns_none_without_tags(self) -> None:
        tag_map = call_or_skip(self, observation.build_apriltag_corner_map)
        params = default_drone_parameters()
        empty = np.zeros((2, 0))
        packet = DronePacket(
            id=np.array([], dtype=np.int64), p1=empty, p2=empty, p3=empty, p4=empty,
            t=0.0, rpy=np.zeros(3), omg=np.zeros(3), acc=HOVER_ACC.copy(),
        )
        result = call_or_skip(
            self, observation.estimate_pose, packet, params.camera_matrix, np.zeros(5), tag_map, params.T_camera_body
        )
        self.assertIsNone(result)

    def _pairs(self, residuals: np.ndarray) -> tuple[list[PoseObservation], list[TruthSample]]:
        observations, truth = [], []
        for k, residual in enumerate(residuals):
            true_position = np.array([1.0, 2.0, 1.5])
            true_rpy = np.array([0.0, 0.1, np.pi - 0.01])  # yaw near the wrap point
            truth.append(TruthSample(t=0.02 * k, position=true_position, rpy=true_rpy, velocity=np.zeros(3), angular_velocity=np.zeros(3)))
            observations.append(
                PoseObservation(
                    t=0.02 * k,
                    position=true_position + residual[:3],
                    rpy=wrap_angles(true_rpy + residual[3:]),
                    reprojection_error=0.5,
                    num_points=8,
                )
            )
        return observations, truth

    def test_covariance_is_about_the_mean_and_wraps_angles(self) -> None:
        rng = np.random.default_rng(7)
        sigma = np.array([0.01, 0.02, 0.01, 0.005, 0.005, 0.02])
        bias = np.array([0.0, 0.05, 0.0, 0.0, 0.0, 0.0])
        residuals = rng.normal(size=(4000, 6)) * sigma + bias
        observations, truth = self._pairs(residuals)
        R = call_or_skip(self, observation.estimate_measurement_covariance, observations, truth)
        self.assertEqual(R.shape, (6, 6))
        np.testing.assert_allclose(R, R.T)
        # A constant bias must not leak into the covariance, and yaw residuals
        # that cross +/- pi must be wrapped (otherwise the yaw variance explodes).
        np.testing.assert_allclose(np.sqrt(np.diag(R)), sigma, rtol=0.06)

    def test_covariance_rejects_bad_inputs(self) -> None:
        observations, truth = self._pairs(np.zeros((3, 6)))
        with self.assertRaises(ValueError):
            call_or_skip(self, observation.estimate_measurement_covariance, observations, truth[:2])
        truth[1].valid = False
        with self.assertRaises(ValueError):
            call_or_skip(self, observation.estimate_measurement_covariance, observations, truth)


# ---------------------------------------------------------------------------
# Part A (student deliverable; skipped while stubbed)
# ---------------------------------------------------------------------------


def _state(rpy=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0), gyro_bias=(0.0, 0.0, 0.0), accel_bias=(0.0, 0.0, 0.0)) -> np.ndarray:
    return np.concatenate([[1.0, 2.0, 1.5], rpy, velocity, gyro_bias, accel_bias]).astype(np.float64)


class PartATests(unittest.TestCase):
    def propagate(self, x, acc, omg, dt):
        return call_or_skip(self, mechanization.propagate_state, x, np.asarray(acc, float), np.asarray(omg, float), dt, mechanization.PropagationParams())

    def test_hover_keeps_the_state_still(self) -> None:
        x0 = _state(rpy=(0.1, -0.2, 0.3))
        specific_force = rpy_to_rotation_matrix(x0[3:6]).T @ HOVER_ACC
        x = x0.copy()
        for _ in range(50):
            x = self.propagate(x, specific_force, np.zeros(3), 0.02)
        self.assertEqual(x.shape, (15,))
        np.testing.assert_allclose(x, x0, atol=1e-9)

    def test_level_yaw_rate_and_bias_subtraction(self) -> None:
        x = self.propagate(_state(), HOVER_ACC, [0.0, 0.0, 0.5], 0.1)
        self.assertAlmostEqual(x[5], 0.05)
        x = self.propagate(_state(gyro_bias=(0.0, 0.0, 0.5)), HOVER_ACC, [0.0, 0.0, 0.5], 0.1)
        self.assertAlmostEqual(x[5], 0.0)

    def test_euler_rates_use_the_zxy_kinematics(self) -> None:
        # With pitch = 0, W^-1 [0, 0, w] = [0, -tan(roll) w, w / cos(roll)].
        roll, w, dt = 0.3, 0.2, 1e-3
        x = self.propagate(_state(rpy=(roll, 0.0, 0.0)), rpy_to_rotation_matrix([roll, 0, 0]).T @ HOVER_ACC, [0.0, 0.0, w], dt)
        np.testing.assert_allclose(x[3:6], [roll, -np.tan(roll) * w * dt, w / np.cos(roll) * dt], atol=1e-7)

    def test_acceleration_and_accel_bias(self) -> None:
        x = self.propagate(_state(), [1.0, 0.0, 9.81], np.zeros(3), 0.1)
        self.assertAlmostEqual(x[6], 0.1)
        x = self.propagate(_state(accel_bias=(1.0, 0.0, 0.0)), [1.0, 0.0, 9.81], np.zeros(3), 0.1)
        self.assertAlmostEqual(x[6], 0.0)

    def test_angles_are_wrapped(self) -> None:
        x = self.propagate(_state(rpy=(0.0, 0.0, np.pi - 0.01)), HOVER_ACC, [0.0, 0.0, 1.0], 0.05)
        self.assertTrue(-np.pi <= x[5] < np.pi)
        self.assertAlmostEqual(x[5], -np.pi + 0.04)


# ---------------------------------------------------------------------------
# Part C (student deliverables; skipped while stubbed)
# ---------------------------------------------------------------------------

P0_TEST = np.diag(np.square([0.1] * 3 + [0.05] * 3 + [0.2] * 3 + [0.01] * 3 + [0.05] * 3))
Q_TEST = np.diag([1e-4] * 3 + [1e-4] * 3 + [1e-2] * 3 + [1e-8] * 3 + [1e-6] * 3)
R_TEST = np.diag(np.square([0.02] * 3 + [0.01] * 3))
Z_OFFSET = np.array([1.1, 2.0, 1.5, 0.0, 0.0, 0.0])


class PartCTests(unittest.TestCase):
    def _kalman_checks(self, nav) -> None:
        trace0 = np.trace(nav.covariance)
        call_or_skip(self, nav.predict, HOVER_ACC, np.zeros(3), 0.02)
        # Sigma-point filters see a small second-order velocity shift; allow it.
        np.testing.assert_allclose(nav.state, _state(), atol=1e-3)
        self.assertGreater(np.trace(nav.covariance), trace0)
        prior_var = nav.covariance[0, 0]
        accepted = call_or_skip(self, nav.update, Z_OFFSET)
        self.assertIsNot(accepted, False)
        self.assertTrue(1.0 < nav.state[0] < 1.1)
        self.assertLess(nav.covariance[0, 0], prior_var)
        np.testing.assert_allclose(nav.covariance, nav.covariance.T, atol=1e-12)
        entry = nav.history[-1]
        self.assertGreaterEqual(entry["nis"], 0.0)
        self.assertEqual(entry["innovation"].shape, (6,))

    def _gate_checks(self, nav) -> None:
        before = nav.state
        accepted = call_or_skip(self, nav.update, Z_OFFSET + np.array([5.0, 0, 0, 0, 0, 0]))
        self.assertFalse(accepted)
        np.testing.assert_allclose(nav.state, before)
        self.assertFalse(nav.history[-1]["accepted"])

    def test_ekf_predict_and_update(self) -> None:
        self._kalman_checks(filters.EKFNavigator(_state(), P0_TEST, Q_TEST, R_TEST))

    def test_ekf_gate(self) -> None:
        self._gate_checks(filters.EKFNavigator(_state(), P0_TEST, Q_TEST, R_TEST, nis_gate=22.46))

    def test_ukf_predict_and_update(self) -> None:
        self._kalman_checks(filters.UKFNavigator(_state(), P0_TEST, Q_TEST, R_TEST, alpha=0.5))

    def test_ukf_gate(self) -> None:
        self._gate_checks(filters.UKFNavigator(_state(), P0_TEST, Q_TEST, R_TEST, alpha=0.5, nis_gate=22.46))

    def test_particle_filter_cycle(self) -> None:
        rng = np.random.default_rng(3)
        particles = rng.multivariate_normal(_state(), P0_TEST, size=2000)
        nav = filters.ParticleFilterNavigator(particles, Q_TEST, R_TEST * 25.0, seed=1)
        call_or_skip(self, nav.predict, HOVER_ACC, np.zeros(3), 0.02)
        start_x = call_or_skip(self, nav.estimate, "weighted_mean")[0]
        call_or_skip(self, nav.update, Z_OFFSET)
        self.assertIn("n_eff", nav.history[-1])
        self.assertLess(nav.effective_sample_size, 2000)
        moved_x = nav.estimate("weighted_mean")[0]
        self.assertGreater(moved_x, start_x)
        for method in ("max_weight", "mean", "weighted_mean"):
            self.assertEqual(nav.estimate(method).shape, (15,))
        call_or_skip(self, nav.resample)
        np.testing.assert_allclose(nav.weights, 1.0 / 2000)
        self.assertEqual(nav.particles.shape, (2000, 15))

    def test_particle_mean_is_angle_aware(self) -> None:
        particles = np.tile(_state(), (4, 1))
        particles[:, 5] = [np.pi - 0.1, -np.pi + 0.1, np.pi - 0.05, -np.pi + 0.05]
        nav = filters.ParticleFilterNavigator(particles, Q_TEST, R_TEST)
        yaw = call_or_skip(self, nav.estimate, "mean")[5]
        self.assertAlmostEqual(abs(yaw), np.pi, places=6)


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------


@unittest.skipIf(REAL_DATA is None, FETCH_HINT)
class RealDataTests(unittest.TestCase):
    def test_every_student_dataset_loads(self) -> None:
        for index in range(8):
            with self.subTest(index=index):
                dataset = load_drone_dataset(index)
                self.assertEqual(dataset.vicon.shape[0], 12)
                self.assertGreater(len(dataset.packets), 100)
                self.assertEqual(dataset.packets[0].omg.shape, (3,))
                self.assertEqual(dataset.packets[0].rate_field, "drpy" if index == 0 else "omg")

    def test_parameters_file_matches_documented_defaults(self) -> None:
        params = load_drone_parameters()
        defaults = default_drone_parameters()
        np.testing.assert_allclose(params.camera_matrix, defaults.camera_matrix)
        np.testing.assert_allclose(params.dist_coeffs, defaults.dist_coeffs)
        np.testing.assert_allclose(params.T_camera_body, defaults.T_camera_body)
        np.testing.assert_array_equal(params.tag_ids, defaults.tag_ids)

    def test_studentdata0_truth_is_trimmed_not_clamped(self) -> None:
        dataset = load_drone_dataset(0)
        truth = interpolate_vicon_truth(dataset)
        invalid = [sample for sample in truth if not sample.valid]
        self.assertGreater(len(invalid), 0)
        self.assertTrue(all(np.all(np.isnan(sample.position)) for sample in invalid))

    def test_pnp_on_real_data_is_centimetre_level(self) -> None:
        tag_map = call_or_skip(self, observation.build_apriltag_corner_map)
        params = load_drone_parameters()
        dataset = load_drone_dataset(2)
        poses = [
            call_or_skip(self, observation.estimate_pose, packet, params.camera_matrix, np.zeros(5), tag_map, params.T_camera_body)
            for packet in dataset.packets[::10]
        ]
        poses = [pose for pose in poses if pose is not None]
        truth = interpolate_vicon_truth(dataset, np.array([pose.t for pose in poses]))
        errors = np.array([pose.position - sample.position for pose, sample in zip(poses, truth)])
        self.assertLess(np.median(np.linalg.norm(errors, axis=1)), 0.05)
        self.assertLess(np.median([pose.reprojection_error for pose in poses]), 1.5)

    def test_run_all_does_not_crash(self) -> None:
        from drone_pipeline import run_all

        with tempfile.TemporaryDirectory() as tmp_dir:
            status = run_all.main(["--datasets", "2", "--out", tmp_dir])
            self.assertEqual(status, 0)
            self.assertTrue((Path(tmp_dir) / "ds2_position.png").exists())


if __name__ == "__main__":
    unittest.main()
