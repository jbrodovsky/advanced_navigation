from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drone_pipeline.datasets import interpolate_vicon_truth, load_drone_dataset, load_drone_parameters
from drone_pipeline.observation import build_apriltag_corner_map, estimate_measurement_covariance
from drone_pipeline.types import PoseObservation


REPO_ROOT = Path(__file__).resolve().parents[3]


class DronePipelineTests(unittest.TestCase):
    def test_load_drone_dataset_normalizes_packets(self) -> None:
        dataset = load_drone_dataset(REPO_ROOT / "data/drone-data/studentdata0.mat")
        self.assertEqual(dataset.vicon.shape[0], 12)
        self.assertGreater(len(dataset.packets), 1000)
        self.assertEqual(dataset.packets[0].img.shape, (240, 376))

    def test_load_drone_parameters_uses_expected_values(self) -> None:
        params = load_drone_parameters(REPO_ROOT / "data/drone-data/parameters.txt")
        self.assertAlmostEqual(params.camera_matrix[0, 0], 314.1779, places=4)
        self.assertEqual(params.T_camera_body.shape, (4, 4))

    def test_interpolate_vicon_truth_matches_packet_times(self) -> None:
        dataset = load_drone_dataset(REPO_ROOT / "data/drone-data/studentdata0.mat")
        truth = interpolate_vicon_truth(dataset, dataset.packet_times[:5])
        self.assertEqual(len(truth), 5)
        self.assertEqual(truth[0].position.shape, (3,))

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
        truth = interpolate_vicon_truth(load_drone_dataset(REPO_ROOT / "data/drone-data/studentdata0.mat"), np.array([88.5, 88.6]))
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
        covariance = estimate_measurement_covariance(observations, truth)
        self.assertEqual(covariance.shape, (6, 6))


if __name__ == "__main__":
    unittest.main()
