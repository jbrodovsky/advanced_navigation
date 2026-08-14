from __future__ import annotations

import json
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ins_gnss.datasets import load_ins_gnss_data
from ins_gnss.strapdown import IMUCalibration, StrapdownNavigator


class INSGNSSTests(unittest.TestCase):
    def test_load_ins_gnss_data_reads_conventional_csv_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "imu.csv").write_text(
                "time_s,ax,ay,az,gx,gy,gz\n"
                "0.0,0.0,0.0,9.81,0.0,0.0,0.0\n"
            )
            (root / "gnss.csv").write_text(
                "time_s,latitude_rad,longitude_rad,altitude_m,vn,ve,vd\n"
                "0.0,0.1,0.2,10.0,0.0,0.0,0.0\n"
            )
            (root / "metadata.json").write_text(json.dumps({"source": "test"}))
            dataset = load_ins_gnss_data(root)
            self.assertEqual(len(dataset.imu_samples), 1)
            self.assertEqual(len(dataset.gnss_samples), 1)
            self.assertEqual(dataset.metadata["source"], "test")

    def test_strapdown_helpers_construct(self) -> None:
        calibration = IMUCalibration(
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            accel_scale=np.eye(3),
            gyro_scale=np.eye(3),
        )
        navigator = StrapdownNavigator(
            {
                "latitude": 0.1,
                "longitude": 0.2,
                "altitude": 10.0,
                "velocity_ned": np.zeros(3),
                "attitude": np.array([1.0, 0.0, 0.0, 0.0]),
            },
            calibration,
        )
        self.assertGreater(navigator.get_meridian_radius(), 0.0)
        self.assertGreater(navigator.compute_gravity(), 0.0)


if __name__ == "__main__":
    unittest.main()
