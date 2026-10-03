from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drone_pipeline.data_paths import DATA_ENV_VAR, DatasetNotFoundError, find_dataset_dir
from drone_pipeline.datasets import (
    default_dataset_path,
    drone_data_dir,
    interpolate_vicon_truth,
    load_drone_dataset,
    load_drone_parameters,
)
from drone_pipeline.observation import build_apriltag_corner_map, estimate_measurement_covariance
from drone_pipeline.types import PoseObservation

FETCH_HINT = "drone data not downloaded; run `uv run python scripts/fetch_data.py drone-data`"

PARAMETERS_FIXTURE = """% Camera Matrix (zero-indexed):
[314.1779 0         199.4848; ...
0         314.2218  113.7838; ...
0         0         1];
% Camera-IMU Calibration (see attached images for details):
XYZ = [-0.04, 0.0, -0.03];
Yaw = pi/4;
% Tag ids:
[0, 12; 1, 13];
"""


def _real_data_dir() -> Path | None:
    try:
        return drone_data_dir()
    except DatasetNotFoundError:
        return None


REAL_DATA = _real_data_dir()


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
    """studentdata1-7 layout: a struct array with ``omg``, ``p0`` and ``is_ready``, no image."""
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

    def test_legacy_schema_reads_drpy_and_image(self) -> None:
        dataset = load_drone_dataset(self.root / "studentdata0.mat")
        self.assertEqual(len(dataset.packets), 3)
        packet = dataset.packets[1]
        self.assertEqual(packet.rate_field, "drpy")
        np.testing.assert_allclose(packet.drpy, [0.01, 0.02, 0.03])
        self.assertEqual(packet.img.shape, (4, 6))
        self.assertIsNone(packet.p0)
        self.assertEqual(dataset.packets[0].id.size, 0)

    def test_current_schema_reads_omg_p0_and_is_ready(self) -> None:
        dataset = load_drone_dataset(self.root / "studentdata1.mat")
        packet = dataset.packets[0]
        self.assertEqual(packet.rate_field, "omg")
        np.testing.assert_allclose(packet.drpy, [0.1, 0.2, 0.3])
        self.assertIsNone(packet.img)
        self.assertTrue(packet.is_ready)
        self.assertEqual(packet.p0.shape, (2, 2))
        np.testing.assert_array_equal(packet.id, [3, 4])
        self.assertEqual(dataset.vicon.shape, (12, 5))

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


class ObservationTests(unittest.TestCase):
    def test_build_apriltag_corner_map_contains_expected_tags(self) -> None:
        tag_map = build_apriltag_corner_map()
        self.assertIn(0, tag_map)
        self.assertIn(107, tag_map)
        self.assertEqual(tag_map[0].shape, (4, 3))

    def test_covariance_estimator_returns_six_by_six_matrix(self) -> None:
        observation = PoseObservation(
            t=0.0,
            position=np.array([1.0, 2.0, 3.0]),
            rpy=np.array([0.1, 0.2, 0.3]),
            reprojection_error=0.5,
            num_points=8,
        )
        observations = [
            observation,
            PoseObservation(
                t=0.1,
                position=observation.position + 0.1,
                rpy=observation.rpy + 0.01,
                reprojection_error=0.6,
                num_points=8,
            ),
        ]
        truth = [
            type("Truth", (), {"t": 0.0, "position": np.zeros(3), "rpy": np.zeros(3)})(),
            type("Truth", (), {"t": 0.1, "position": np.zeros(3), "rpy": np.zeros(3)})(),
        ]
        covariance = estimate_measurement_covariance(observations, truth)
        self.assertEqual(covariance.shape, (6, 6))


@unittest.skipIf(REAL_DATA is None, FETCH_HINT)
class RealDataTests(unittest.TestCase):
    def test_every_student_dataset_loads(self) -> None:
        for index in range(8):
            with self.subTest(index=index):
                dataset = load_drone_dataset(index)
                self.assertEqual(dataset.vicon.shape[0], 12)
                self.assertGreater(len(dataset.packets), 100)
                self.assertEqual(dataset.packets[0].drpy.shape, (3,))

    def test_parameters_file_matches_config(self) -> None:
        params = load_drone_parameters()
        self.assertAlmostEqual(params.camera_matrix[0, 0], 314.1779, places=4)


if __name__ == "__main__":
    unittest.main()
