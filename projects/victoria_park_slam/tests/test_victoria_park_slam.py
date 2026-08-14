from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from victoria_park_slam.datasets import load_victoria_park_data


REPO_ROOT = Path(__file__).resolve().parents[3]


class VictoriaParkTests(unittest.TestCase):
    def test_load_victoria_park_data(self) -> None:
        dataset = load_victoria_park_data(REPO_ROOT / "dataset_victoria_park/original_MATLAB_dataset")
        self.assertEqual(dataset.laser_ranges_m.shape[1], 361)
        self.assertEqual(dataset.laser_time_s.shape[0], dataset.laser_ranges_m.shape[0])
        self.assertEqual(dataset.speed_mps.shape[0], dataset.steering_rad.shape[0])
        self.assertIsNotNone(dataset.gps_time_s)


if __name__ == "__main__":
    unittest.main()

