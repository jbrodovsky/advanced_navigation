from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from victoria_park_slam.data_paths import DatasetNotFoundError
from victoria_park_slam.datasets import load_victoria_park_data, resolve_victoria_park_root
from victoria_park_slam.utils import (
    effective_sample_size,
    inverse_range_bearing,
    mahalanobis_squared,
    predict_bicycle_pose,
    range_bearing_jacobian,
    range_bearing_prediction,
    systematic_resample_indices,
    wrap_angle,
)


FETCH_HINT = "Victoria Park data not downloaded; run `uv run python scripts/fetch_data.py victoria-park`"


def _real_data_root() -> Path | None:
    try:
        return resolve_victoria_park_root()
    except DatasetNotFoundError:
        return None


REAL_DATA = _real_data_root()


class VictoriaParkTests(unittest.TestCase):
    def test_wrap_angle(self) -> None:
        self.assertAlmostEqual(wrap_angle(np.pi), -np.pi)
        self.assertAlmostEqual(wrap_angle(3.0 * np.pi), -np.pi)
        values = wrap_angle(np.array([0.0, np.pi, -np.pi]))
        np.testing.assert_allclose(values, np.array([0.0, -np.pi, -np.pi]))

    def test_predict_bicycle_pose_straight(self) -> None:
        pose = np.array([1.0, 2.0, 0.0])
        propagated = predict_bicycle_pose(pose, 3.0, 0.0, 2.0, 2.5)
        np.testing.assert_allclose(propagated, np.array([7.0, 2.0, 0.0]), atol=1e-12)

    def test_predict_bicycle_pose_turning(self) -> None:
        pose = np.array([0.0, 0.0, 0.0])
        propagated = predict_bicycle_pose(pose, 2.0, 0.2, 1.0, 2.5)
        self.assertGreater(propagated[0], 0.0)
        self.assertGreater(propagated[1], 0.0)
        self.assertGreater(propagated[2], 0.0)

    def test_range_bearing_inverse_consistency(self) -> None:
        pose = np.array([2.0, -1.0, 0.4])
        landmark = np.array([6.0, 2.5])
        measurement = range_bearing_prediction(pose, landmark)
        recovered = inverse_range_bearing(pose, measurement)
        np.testing.assert_allclose(recovered, landmark, atol=1e-12)

    def test_range_bearing_jacobian_shape_and_values(self) -> None:
        jacobian = range_bearing_jacobian(np.array([0.0, 0.0, 0.0]), np.array([3.0, 4.0]))
        self.assertEqual(jacobian.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(jacobian)))

    def test_mahalanobis_squared_uses_solve(self) -> None:
        residual = np.array([2.0, 0.5])
        covariance = np.diag([4.0, 0.25])
        with patch("victoria_park_slam.utils.np.linalg.solve", wraps=np.linalg.solve) as solve:
            value = mahalanobis_squared(residual, covariance)
        self.assertAlmostEqual(value, 2.0)
        solve.assert_called_once()

    def test_effective_sample_size(self) -> None:
        self.assertAlmostEqual(effective_sample_size(np.array([0.25, 0.25, 0.25, 0.25])), 4.0)
        self.assertAlmostEqual(effective_sample_size(np.array([1.0, 0.0, 0.0, 0.0])), 1.0)

    def test_systematic_resample_indices(self) -> None:
        indexes = systematic_resample_indices(
            np.array([0.1, 0.2, 0.7]),
            rng=np.random.default_rng(3),
        )
        self.assertEqual(indexes.shape, (3,))
        self.assertTrue(np.all(indexes >= 0))
        self.assertTrue(np.all(indexes < 3))

    def test_load_synthetic_dataset_converts_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            laser = np.full((2, 361), 1234, dtype=np.uint16)
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
        np.testing.assert_allclose(dataset.laser_time_s, [1.0, 1.25])
        np.testing.assert_allclose(dataset.odometry_time_s, [1.0, 1.025])
        np.testing.assert_allclose(dataset.gps_time_s, [1.1])

    def test_missing_dataset_explains_how_to_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(DatasetNotFoundError) as caught:
                load_victoria_park_data(Path(tmp_dir))
        self.assertIn("scripts/fetch_data.py victoria-park", str(caught.exception))

    @unittest.skipIf(REAL_DATA is None, FETCH_HINT)
    def test_load_real_dataset(self) -> None:
        dataset = load_victoria_park_data()
        self._assert_dataset_shapes(dataset)
        self.assertEqual(dataset.laser_ranges_m.shape[0], 7249)

    def _assert_dataset_shapes(self, dataset) -> None:
        self.assertEqual(dataset.laser_ranges_m.shape[1], 361)
        self.assertEqual(dataset.laser_time_s.shape[0], dataset.laser_ranges_m.shape[0])
        self.assertEqual(dataset.speed_mps.shape[0], dataset.steering_rad.shape[0])
        self.assertIsNotNone(dataset.gps_time_s)


if __name__ == "__main__":
    unittest.main()
