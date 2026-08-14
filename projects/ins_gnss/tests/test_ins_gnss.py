from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ins_gnss.datasets import list_mems_nav_trials, load_ins_gnss_data, load_mems_nav_trial
from ins_gnss.strapdown import IMUCalibration, StrapdownNavigator


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_fetch_script():
    script_path = REPO_ROOT / "scripts" / "fetch_ins_gnss_data.py"
    spec = importlib.util.spec_from_file_location("fetch_ins_gnss_data", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

    def test_list_mems_nav_trials_requires_input_and_baseline_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "input").mkdir()
            (root / "baseline").mkdir()
            (root / "input" / "trial-a.csv").write_text("")
            (root / "input" / "trial-b.csv").write_text("")
            (root / "baseline" / "trial-a.csv").write_text("")

            self.assertEqual(list_mems_nav_trials(root), ["trial-a"])

    def test_load_mems_nav_trial_parses_input_and_baseline_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "input").mkdir()
            (root / "baseline").mkdir()
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "repo_url": "https://github.com/jbrodovsky/mems-nav-dataset",
                        "ref": "main",
                        "datasets": {
                            "input": [{"name": "trial-a.csv", "size": 10, "sha": "input-sha"}],
                            "baseline": [{"name": "trial-a.csv", "size": 20, "sha": "baseline-sha"}],
                        },
                    }
                )
            )
            (root / "input" / "trial-a.csv").write_text(
                "time,bearingAccuracy,speedAccuracy,verticalAccuracy,horizontalAccuracy,speed,bearing,"
                "altitude,longitude,latitude,qz,qy,qx,qw,roll,pitch,yaw,acc_z,acc_y,acc_x,"
                "gyro_z,gyro_y,gyro_x,mag_z,mag_y,mag_x,relativeAltitude,pressure,grav_z,grav_y,grav_x\n"
                "2025-06-14 21:17:02+00:00,45,1.5,0.6,11.4,0,0,58.9,-75.0,40.0,"
                "0,0,0,1,0,0,0,9.8,0.2,0.1,0.03,0.02,0.01,0,0,0,0,1005,9.8,0,0\n"
                "2025-06-14 21:17:03+00:00,,,,,,,,,,0,0,0,1,0,0,0,9.7,0.3,0.2,0.04,0.03,0.02,"
                "0,0,0,0,1005,9.8,0,0\n"
            )
            (root / "baseline" / "trial-a.csv").write_text(
                "timestamp,latitude,longitude,altitude,velocity_north,velocity_east,velocity_down,"
                "roll,pitch,yaw,acc_bias_x,acc_bias_y,acc_bias_z,gyro_bias_x,gyro_bias_y,gyro_bias_z,"
                "latitude_cov,longitude_cov,altitude_cov\n"
                "2025-06-14T21:17:03Z,40.0,-75.0,57.0,1.0,2.0,3.0,0.1,0.2,0.3,"
                "0.01,0.02,0.03,0.001,0.002,0.003,1e-6,2e-6,3.0\n"
            )

            trial = load_mems_nav_trial(root, "trial-a")

            self.assertEqual(trial.name, "trial-a")
            self.assertEqual(len(trial.imu_samples), 2)
            self.assertEqual(len(trial.gnss_samples), 1)
            self.assertEqual(len(trial.reference_states), 1)
            np.testing.assert_allclose(trial.imu_samples[0].accel_mps2, [0.1, 0.2, 9.8])
            np.testing.assert_allclose(trial.imu_samples[0].gyro_rps, [0.01, 0.02, 0.03])
            self.assertAlmostEqual(trial.imu_samples[1].time_s, 1.0)
            self.assertAlmostEqual(trial.gnss_samples[0].latitude_rad, np.deg2rad(40.0))
            self.assertAlmostEqual(trial.gnss_samples[0].longitude_rad, np.deg2rad(-75.0))
            self.assertEqual(trial.gnss_samples[0].velocity_ned_mps, None)
            reference = trial.reference_states[0]
            np.testing.assert_allclose(reference.velocity_ned_mps, [1.0, 2.0, 3.0])
            np.testing.assert_allclose(reference.attitude_rpy_rad, [0.1, 0.2, 0.3])
            self.assertEqual(reference.covariance["altitude_cov"], 3.0)
            self.assertEqual(trial.metadata["ref"], "main")

    def test_fetch_script_builds_urls_manifest_and_skip_plan(self) -> None:
        fetch_script = _load_fetch_script()
        self.assertEqual(
            fetch_script.contents_api_url("baseline", "feature/test"),
            "https://api.github.com/repos/jbrodovsky/mems-nav-dataset/contents/data/baseline?ref=feature%2Ftest",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "trial.csv"
            target.write_text("12345")
            self.assertEqual(fetch_script.planned_action(target, 5, force=False), "skip")
            self.assertEqual(fetch_script.planned_action(target, 6, force=False), "download")
            self.assertEqual(fetch_script.planned_action(target, 5, force=True), "download")

        manifest = fetch_script.build_manifest(
            "main",
            {
                "input": [
                    {
                        "name": "trial.csv",
                        "path": "data/input/trial.csv",
                        "size": 5,
                        "sha": "abc",
                        "download_url": "https://example.test/input.csv",
                    }
                ],
                "baseline": [],
            },
        )
        self.assertEqual(manifest["repo_url"], "https://github.com/jbrodovsky/mems-nav-dataset")
        self.assertEqual(manifest["datasets"]["input"][0]["sha"], "abc")

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
