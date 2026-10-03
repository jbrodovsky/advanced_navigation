"""Tests for the Project 3 starter package.

Two kinds of tests live here:

* Support-code tests (Earth model, rotations, loaders, data-quality helpers,
  evaluation, the provided filter bookkeeping). These must always pass.
* Student-facing tests for the deliverables (``mechanize``,
  ``coarse_alignment``, ``error_state_jacobian``, ``discrete_process_noise``,
  ``gnss_position_residual``, ``joseph_update``, ``inject_error_state``,
  ``reset_covariance``). They skip with a "not implemented yet" message while a
  function is still a stub and start checking your implementation as soon as
  it exists. Passing them is necessary, not sufficient.

Tests that need the real dataset skip until it has been downloaded.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy.linalg import expm

from ins_gnss import eskf, run_all, strapdown
from ins_gnss.data_paths import DATA_ENV_VAR, DatasetNotFoundError
from ins_gnss.datasets import (
    BASELINE_TIME_OFFSET_S,
    TRIAL_NOTES,
    default_mems_root,
    estimate_mount_rotation,
    gnss_arrays,
    imu_arrays,
    list_mems_nav_trials,
    load_ins_gnss_data,
    load_mems_nav_trial,
    recommended_trials,
    reference_arrays,
    repeated_fix_mask,
    rotate_imu_samples,
)
from ins_gnss.earth import (
    GRAVITY_EQUATOR,
    OMEGA_IE,
    WGS84_A,
    WGS84_E2,
    compute_gravity,
    earth_rate_ned,
    geodetic_to_ned_delta,
    gravity_ned,
    ned_to_geodetic_delta,
    radii_of_curvature,
    transport_rate_ned,
)
from ins_gnss.eskf import ACC_BIAS, ATT, GYRO_BIAS, POS, STATE_DIM, VEL, INSGNSSFilter, gnss_position_covariance
from ins_gnss.evaluation import (
    Outage,
    chi2_gate,
    chi2_interval,
    constant_velocity_outage_errors,
    error_summary,
    fraction_inside,
    gnss_only_track,
    nees,
    outage_errors,
    position_error_ned,
    reference_valid_mask,
    schedule_outages,
)
from ins_gnss.rotations import (
    attitude_error,
    dcm_to_euler,
    dcm_to_quat,
    dcm_to_rotvec,
    euler_to_dcm,
    euler_to_quat,
    quat_multiply,
    quat_normalize,
    quat_to_dcm,
    quat_to_rotvec,
    rotvec_to_quat,
    skew,
)
from ins_gnss.types import GNSSSample, IMUNoise, IMUSample, NavState

REPO_ROOT = Path(__file__).resolve().parents[3]
FETCH_HINT = "MEMS data not downloaded; run `uv run python scripts/fetch_data.py mems-nav-dataset`"
LAT_42N = math.radians(42.0)


def _real_mems_root() -> Path | None:
    try:
        return default_mems_root()
    except DatasetNotFoundError:
        return None


REAL_DATA = _real_mems_root()


def call_or_skip(test: unittest.TestCase, function, *args, **kwargs):
    """Call a student deliverable; skip the test while it is still a stub."""
    try:
        return function(*args, **kwargs)
    except NotImplementedError as error:
        name = getattr(function, "__qualname__", repr(function))
        test.skipTest(f"{name} is not implemented yet (starter stub): {error}")


def _load_fetch_script():
    script_path = REPO_ROOT / "scripts" / "fetch_ins_gnss_data.py"
    spec = importlib.util.spec_from_file_location("fetch_ins_gnss_data", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Test-side helpers (independent of the student code)
# ---------------------------------------------------------------------------


def perturb(state: NavState, delta_x: np.ndarray) -> NavState:
    """True state = nominal (+) error, with the book's error definitions."""
    d_lat, d_lon, d_h = ned_to_geodetic_delta(delta_x[POS], state.latitude_rad, state.height_m)
    return NavState(
        state.latitude_rad + d_lat,
        state.longitude_rad + d_lon,
        state.height_m + d_h,
        state.velocity_ned_mps + delta_x[VEL],
        quat_normalize(quat_multiply(rotvec_to_quat(delta_x[ATT]), state.q_bn)),
        state.accel_bias_mps2 + delta_x[ACC_BIAS],
        state.gyro_bias_rps + delta_x[GYRO_BIAS],
    )


def error_between(true: NavState, nominal: NavState) -> np.ndarray:
    """Error state (true minus nominal) between two full states."""
    dx = np.zeros(STATE_DIM)
    dx[POS] = geodetic_to_ned_delta(
        true.latitude_rad, true.longitude_rad, true.height_m, nominal.latitude_rad, nominal.longitude_rad, nominal.height_m
    )
    dx[VEL] = true.velocity_ned_mps - nominal.velocity_ned_mps
    dx[ATT] = attitude_error(quat_to_dcm(true.q_bn), quat_to_dcm(nominal.q_bn))
    dx[ACC_BIAS] = true.accel_bias_mps2 - nominal.accel_bias_mps2
    dx[GYRO_BIAS] = true.gyro_bias_rps - nominal.gyro_bias_rps
    return dx


def perfect_imu_level_east(latitude_rad: float, height_m: float, speed_mps: float, heading_rad: float = math.pi / 2):
    """Perfect IMU for level, constant-velocity motion due east along a parallel (or at rest).

    Along a parallel at constant height ``v^n`` and ``C_b^n`` are constant, so
    the specific force and the body rate are constant too:
    ``f^n = (Omega_en + 2 Omega_ie) v - g^n``, ``omega_ib^b = C_n^b (omega_ie + omega_en)``.
    """
    velocity = np.array([0.0, speed_mps, 0.0])
    C = euler_to_dcm(0.0, 0.0, heading_rad)
    w_ie = earth_rate_ned(latitude_rad)
    w_en = transport_rate_ned(latitude_rad, height_m, velocity)
    f_n = skew(w_en + 2.0 * w_ie) @ velocity - gravity_ned(latitude_rad, height_m)
    return velocity, C, C.T @ f_n, C.T @ (w_ie + w_en)


def gnss_fix(time_s, latitude_rad, longitude_rad, altitude_m, **kwargs) -> GNSSSample:
    return GNSSSample(time_s=time_s, latitude_rad=latitude_rad, longitude_rad=longitude_rad, altitude_m=altitude_m, **kwargs)


# ---------------------------------------------------------------------------
# Earth model and rotations (provided)
# ---------------------------------------------------------------------------


class EarthModelTests(unittest.TestCase):
    def test_gravity_matches_published_values(self) -> None:
        self.assertAlmostEqual(compute_gravity(0.0, 0.0), 9.7803, places=4)
        self.assertAlmostEqual(compute_gravity(math.pi / 2, 0.0), 9.8322, places=4)
        self.assertAlmostEqual(compute_gravity(0.0), GRAVITY_EQUATOR, places=9)
        # The old starter squared sin^2 L again (a sin^4 term); at 45 deg the
        # Somigliana value is 9.80620 m/s^2.
        self.assertAlmostEqual(compute_gravity(math.radians(45.0), 0.0), 9.80620, places=4)
        # Free-air gradient of about -3.086e-6 m/s^2 per meter.
        gradient = (compute_gravity(LAT_42N, 1000.0) - compute_gravity(LAT_42N, 0.0)) / 1000.0
        self.assertAlmostEqual(gradient, -3.086e-6, delta=0.01e-6)
        np.testing.assert_allclose(gravity_ned(LAT_42N, 10.0), [0.0, 0.0, compute_gravity(LAT_42N, 10.0)])
        np.testing.assert_allclose(compute_gravity(np.array([0.0, math.pi / 2])), [9.7803253359, 9.8321849379], atol=1e-8)

    def test_radii_of_curvature(self) -> None:
        r_m, r_n = radii_of_curvature(0.0)
        self.assertAlmostEqual(r_n, WGS84_A, places=6)
        self.assertAlmostEqual(r_m, WGS84_A * (1.0 - WGS84_E2), places=3)
        r_m, r_n = radii_of_curvature(math.pi / 2)
        self.assertAlmostEqual(r_m, r_n, places=3)
        self.assertAlmostEqual(r_n, WGS84_A / math.sqrt(1.0 - WGS84_E2), places=3)

    def test_earth_and_transport_rate(self) -> None:
        w = earth_rate_ned(LAT_42N)
        self.assertAlmostEqual(np.linalg.norm(w), OMEGA_IE, places=12)
        self.assertGreater(w[0], 0.0)
        self.assertLess(w[2], 0.0)  # Earth rate points up (negative down) in the northern hemisphere
        r_m, r_n = radii_of_curvature(LAT_42N)
        w_en = transport_rate_ned(LAT_42N, 100.0, np.array([10.0, 20.0, 0.0]))
        np.testing.assert_allclose(w_en, [20.0 / (r_n + 100.0), -10.0 / (r_m + 100.0), -20.0 * math.tan(LAT_42N) / (r_n + 100.0)])

    def test_ned_geodetic_round_trip(self) -> None:
        delta = np.array([120.0, -340.0, 5.0])
        d_lat, d_lon, d_h = ned_to_geodetic_delta(delta, LAT_42N, 50.0)
        self.assertAlmostEqual(d_h, -5.0)
        back = geodetic_to_ned_delta(LAT_42N + d_lat, 0.3 + d_lon, 50.0 + d_h, LAT_42N, 0.3, 50.0)
        np.testing.assert_allclose(back, delta, atol=1e-6)
        # Longitude differences wrap across the antimeridian.
        across = geodetic_to_ned_delta(0.0, -math.pi + 1e-6, 0.0, 0.0, math.pi - 1e-6, 0.0)
        self.assertAlmostEqual(across[1], 2e-6 * WGS84_A, places=3)


class RotationTests(unittest.TestCase):
    def test_quaternion_product_composes_like_matrices(self) -> None:
        rng = np.random.default_rng(1)
        for _ in range(5):
            p = quat_normalize(rng.standard_normal(4))
            q = quat_normalize(rng.standard_normal(4))
            np.testing.assert_allclose(quat_to_dcm(quat_multiply(p, q)), quat_to_dcm(p) @ quat_to_dcm(q), atol=1e-12)

    def test_rotation_vector_round_trips_and_matches_expm(self) -> None:
        for phi in (np.array([0.1, -0.2, 0.3]), np.array([1e-10, 0.0, 0.0]), np.array([0.0, 3.1, 0.0])):
            q = rotvec_to_quat(phi)
            self.assertAlmostEqual(np.linalg.norm(q), 1.0, places=12)
            np.testing.assert_allclose(quat_to_dcm(q), expm(skew(phi)), atol=1e-12)
            np.testing.assert_allclose(quat_to_rotvec(q), phi, atol=1e-9)

    def test_dcm_quaternion_round_trip_near_pi(self) -> None:
        for phi in (np.array([0.0, 0.0, math.pi - 1e-9]), np.array([2.0, -1.0, 0.5]), np.zeros(3)):
            C = expm(skew(phi))
            np.testing.assert_allclose(quat_to_dcm(dcm_to_quat(C)), C, atol=1e-9)
        np.testing.assert_allclose(dcm_to_rotvec(expm(skew([0.2, 0.1, -0.4]))), [0.2, 0.1, -0.4], atol=1e-12)

    def test_skew_is_the_cross_product(self) -> None:
        a, b = np.array([1.0, -2.0, 0.5]), np.array([0.3, 0.7, -1.1])
        np.testing.assert_allclose(skew(a) @ b, np.cross(a, b))

    def test_euler_zyx_convention(self) -> None:
        C = euler_to_dcm(0.1, -0.2, 2.5)
        np.testing.assert_allclose(dcm_to_euler(C), [0.1, -0.2, 2.5], atol=1e-12)
        # Yaw 90 deg: body x (forward) points east.
        np.testing.assert_allclose(euler_to_dcm(0.0, 0.0, math.pi / 2) @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(quat_to_dcm(euler_to_quat(0.1, -0.2, 2.5)), C, atol=1e-12)

    def test_attitude_error_is_true_minus_estimate_in_the_nav_frame(self) -> None:
        C_est = euler_to_dcm(0.3, -1.5, 2.0)  # near gimbal lock on purpose
        dtheta = np.array([0.01, -0.02, 0.03])
        C_true = expm(skew(dtheta)) @ C_est
        np.testing.assert_allclose(attitude_error(C_true, C_est), dtheta, atol=1e-12)


# ---------------------------------------------------------------------------
# Loaders and data quality (provided)
# ---------------------------------------------------------------------------

INPUT_HEADER = (
    "time,bearingAccuracy,speedAccuracy,verticalAccuracy,horizontalAccuracy,speed,bearing,"
    "altitude,longitude,latitude,qz,qy,qx,qw,roll,pitch,yaw,acc_z,acc_y,acc_x,"
    "gyro_z,gyro_y,gyro_x,mag_z,mag_y,mag_x,relativeAltitude,pressure,grav_z,grav_y,grav_x\n"
)
BASELINE_HEADER = (
    "timestamp,latitude,longitude,altitude,velocity_north,velocity_east,velocity_down,"
    "roll,pitch,yaw,acc_bias_x,acc_bias_y,acc_bias_z,gyro_bias_x,gyro_bias_y,gyro_bias_z,"
    "latitude_cov,longitude_cov,altitude_cov\n"
)


def write_synthetic_trial(root: Path) -> None:
    """A three-row MEMS trial whose baseline starts one second after the input (as the real files do)."""
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
        INPUT_HEADER
        + "2025-06-14 21:17:02+00:00,1.5,0.2,3.0,4.5,10.0,90.0,58.9,-75.0,40.0,"
        "0,0,0,1,0,0,0,9.8,0.2,0.1,0.03,0.02,0.01,0,0,0,0,1005,9.8,0,0\n"
        "2025-06-14 21:17:03+00:00,,,,,,,,,,0,0,0,1,0,0,0,9.7,0.3,0.2,0.04,0.03,0.02,0,0,0,0,1005,9.8,0,0\n"
        "2025-06-14 21:17:04+00:00,0.0,0.1,2.0,3.0,0.0,0.0,59.0,-75.0001,40.0001,"
        "0,0,0,1,0,0,0,,0.3,0.2,0.04,0.03,0.02,0,0,0,0,1005,9.8,0,0\n"
    )
    (root / "baseline" / "trial-a.csv").write_text(
        BASELINE_HEADER
        + "2025-06-14T21:17:03Z,40.0,-75.0,57.0,1.0,2.0,3.0,0.1,0.2,0.3,0.01,0.02,0.03,0.001,0.002,0.003,1e-6,2e-6,3.0\n"
        "2025-06-14T21:17:04Z,40.0,-75.0,57.0,1.0,2.0,3.0,0.1,0.2,0.3,0.01,0.02,0.03,0.001,0.002,0.003,1e-6,2e-6,3.0\n"
    )


class LoaderTests(unittest.TestCase):
    def test_default_root_comes_from_env_var(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "mems-nav-dataset"
            (root / "input").mkdir(parents=True)
            (root / "baseline").mkdir()
            with mock.patch.dict(os.environ, {DATA_ENV_VAR: tmp_dir}):
                self.assertEqual(default_mems_root(), root)
                self.assertEqual(list_mems_nav_trials(), [])
                self.assertEqual(recommended_trials(), [])

    def test_conventional_csv_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "imu.csv").write_text("time_s,ax,ay,az,gx,gy,gz\n0.0,0.0,0.0,-9.81,0.0,0.0,0.0\n")
            (root / "gnss.csv").write_text(
                "time_s,latitude_rad,longitude_rad,altitude_m,vn,ve,vd\n0.0,0.1,0.2,10.0,1.0,2.0,0.0\n"
            )
            (root / "metadata.json").write_text(json.dumps({"source": "test"}))
            dataset = load_ins_gnss_data(root)
            self.assertEqual(len(dataset.imu_samples), 1)
            np.testing.assert_allclose(dataset.gnss_samples[0].velocity_ne_mps(), [1.0, 2.0])
            self.assertEqual(dataset.metadata["source"], "test")

    def test_trial_listing_requires_input_and_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "input").mkdir()
            (root / "baseline").mkdir()
            for path in ("input/trial-a.csv", "input/trial-b.csv", "baseline/trial-a.csv"):
                (root / path).write_text("")
            self.assertEqual(list_mems_nav_trials(root), ["trial-a"])

    def test_mems_loader_uses_one_epoch_and_the_baseline_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_synthetic_trial(root)
            trial = load_mems_nav_trial(root, "trial-a")
            # IMU: the third row has a missing accelerometer value and is dropped.
            self.assertEqual([s.time_s for s in trial.imu_samples], [0.0, 1.0])
            np.testing.assert_allclose(trial.imu_samples[0].accel_mps2, [0.1, 0.2, 9.8])
            np.testing.assert_allclose(trial.imu_samples[0].gyro_rps, [0.01, 0.02, 0.03])
            # GNSS: rows 1 and 3; the empty second row is not a fix.
            self.assertEqual([g.time_s for g in trial.gnss_samples], [0.0, 2.0])
            fix = trial.gnss_samples[0]
            self.assertAlmostEqual(fix.latitude_rad, math.radians(40.0))
            self.assertAlmostEqual(fix.speed_mps, 10.0)
            self.assertAlmostEqual(fix.bearing_rad, math.pi / 2)
            self.assertEqual((fix.horizontal_accuracy_m, fix.vertical_accuracy_m), (4.5, 3.0))
            self.assertAlmostEqual(fix.speed_accuracy_mps, 0.2)
            self.assertAlmostEqual(fix.bearing_accuracy_rad, math.radians(1.5))
            np.testing.assert_allclose(fix.velocity_ne_mps(), [0.0, 10.0], atol=1e-12)
            self.assertIsNone(fix.velocity_ned_mps)
            # Baseline: same epoch as the input (first baseline row is 1 s in), plus the 1 s label offset.
            self.assertEqual(BASELINE_TIME_OFFSET_S, 1.0)
            self.assertEqual([r.time_s for r in trial.reference_states], [2.0, 3.0])
            raw = load_mems_nav_trial(root, "trial-a", baseline_time_offset_s=0.0)
            self.assertEqual([r.time_s for r in raw.reference_states], [1.0, 2.0])
            reference = trial.reference_states[0]
            np.testing.assert_allclose(reference.velocity_ned_mps, [1.0, 2.0, 3.0])
            self.assertEqual(reference.covariance["altitude_cov"], 3.0)
            self.assertEqual(trial.metadata["ref"], "main")
            self.assertEqual(trial.metadata["epoch_utc"], "2025-06-14T21:17:02+00:00")
            data = gnss_arrays(trial.gnss_samples)
            self.assertEqual(data["speed_mps"].tolist(), [10.0, 0.0])
            self.assertEqual(reference_arrays(trial.reference_states)["velocity_ned_mps"].shape, (2, 3))

    def test_repeated_fixes_are_stale_only_when_moving(self) -> None:
        fixes = [
            gnss_fix(0.0, 0.7, 0.1, 10.0, speed_mps=0.0),
            gnss_fix(1.0, 0.7, 0.1, 10.0, speed_mps=0.0),  # parked: fine
            gnss_fix(2.0, 0.7, 0.1, 10.0, speed_mps=8.0),  # moving but frozen: stale
            gnss_fix(3.0, 0.7000001, 0.1, 10.0, speed_mps=8.0),
        ]
        self.assertEqual(repeated_fix_mask(fixes).tolist(), [False, False, True, False])

    def test_mount_rotation_recovers_a_known_mounting(self) -> None:
        # A phone standing in portrait facing the driver: forward = -z, right = +x, down = -y.
        C_pv_true = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        rng = np.random.default_rng(4)
        n = 400
        t = np.arange(n, dtype=float)
        speed = 15.0 + 5.0 * np.sin(t / 15.0)
        course = 0.4 * np.sin(t / 23.0)
        a_forward = np.gradient(speed)
        a_right = speed * np.gradient(course)
        f_vehicle = np.column_stack([a_forward, a_right, np.full(n, -9.8)]) + 0.05 * rng.standard_normal((n, 3))
        imu = [IMUSample(float(k), C_pv_true.T @ f_vehicle[k], np.zeros(3)) for k in range(n)]
        gnss = [gnss_fix(float(k), 0.7, 0.1, 0.0, speed_mps=float(speed[k]), bearing_rad=float(course[k])) for k in range(n)]
        C_pv = estimate_mount_rotation(imu, gnss)
        np.testing.assert_allclose(C_pv, C_pv_true, atol=0.02)
        rotated = rotate_imu_samples(imu, C_pv)
        _, accel, _ = imu_arrays(rotated)
        self.assertLess(abs(np.mean(accel[:, 2]) + 9.8), 0.05)

    def test_trial_notes_cover_the_audit(self) -> None:
        for name in ("2024-06-20_165550", "2025-06-11_20-34-24", "2025-06-14_21-17-02"):
            self.assertEqual(TRIAL_NOTES[name].status, "excluded")
        recommended = [name for name, note in TRIAL_NOTES.items() if note.status == "recommended"]
        self.assertGreaterEqual(len(recommended), 3)

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


# ---------------------------------------------------------------------------
# Evaluation (provided)
# ---------------------------------------------------------------------------


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        # A track heading north at 10 m/s, 50 m above the ellipsoid, sampled every second.
        self.t = np.arange(0.0, 200.0)
        r_m, _ = radii_of_curvature(LAT_42N)
        self.lat = LAT_42N + 10.0 * self.t / (r_m + 50.0)
        self.lon = np.full_like(self.t, -1.3)
        self.h = np.full_like(self.t, 50.0)

    def test_errors_are_time_aligned_by_interpolation(self) -> None:
        # The estimate is sampled half a second later but on the same path: zero error.
        t_est = self.t + 0.5
        r_m, _ = radii_of_curvature(LAT_42N)
        lat_est = LAT_42N + 10.0 * t_est / (r_m + 50.0)
        t, error = position_error_ned(self.t, self.lat, self.lon, self.h, t_est, lat_est, self.lon, self.h)
        self.assertEqual(t[0], 1.0)  # t = 0 is before the estimate starts: dropped
        np.testing.assert_allclose(error, 0.0, atol=1e-6)
        # An estimate 3 m south and 2 m high has error +3 m north and +2 m down (true minus estimate).
        t, error = position_error_ned(self.t, self.lat, self.lon, self.h, self.t, self.lat - 3.0 / (r_m + 50.0), self.lon, self.h + 2.0)
        np.testing.assert_allclose(error, np.tile([3.0, 0.0, 2.0], (len(t), 1)), atol=1e-3)
        summary = error_summary(error)
        self.assertAlmostEqual(summary["horizontal_rmse_m"], 3.0, places=3)
        self.assertAlmostEqual(summary["vertical_rmse_m"], 2.0, places=3)
        self.assertEqual(summary["num_samples"], len(t))

    def test_outage_schedule_and_scoring(self) -> None:
        fixes = [gnss_fix(float(t), float(lat), -1.3, 50.0, speed_mps=10.0, bearing_rad=0.0) for t, lat in zip(self.t, self.lat)]
        outages = schedule_outages(fixes, (10.0, 30.0), first_start_s=20.0, spacing_s=50.0)
        self.assertEqual(outages, [Outage(20.0, 30.0), Outage(70.0, 100.0), Outage(120.0, 130.0)])
        self.assertTrue(outages[0].contains(30.0) and not outages[0].contains(20.0))
        # Constant velocity is exact on a straight line at constant speed.
        for result in constant_velocity_outage_errors(outages, fixes):
            np.testing.assert_allclose(result.horizontal_error_m, 0.0, atol=1e-3)
        # A track that is 5 m east of the truth everywhere scores 5 m against every withheld fix.
        _, r_n = radii_of_curvature(LAT_42N)
        lon_off = self.lon - 5.0 / ((r_n + 50.0) * math.cos(LAT_42N))
        results = outage_errors(outages, fixes, self.t, self.lat, lon_off, self.h, np.full_like(self.t, 2.0))
        self.assertEqual(results[1].elapsed_s.tolist(), list(np.arange(1.0, 31.0)))
        np.testing.assert_allclose(results[1].horizontal_error_m, 5.0, atol=0.01)
        self.assertAlmostEqual(results[1].error_at(30.0), 5.0, places=2)
        np.testing.assert_allclose(results[1].horizontal_sigma_m, 2.0)

    def test_gnss_only_track_interpolates_and_extrapolates(self) -> None:
        fixes = [gnss_fix(float(t), float(lat), -1.3, 50.0, speed_mps=10.0, bearing_rad=0.0) for t, lat in zip(self.t, self.lat)]
        del fixes[50:60]  # an 11 s gap: extrapolate at constant velocity
        lat, lon, h = gnss_only_track(fixes, np.array([10.5, 55.0, -1.0]))
        r_m, _ = radii_of_curvature(LAT_42N)
        self.assertAlmostEqual(lat[0], LAT_42N + 105.0 / (r_m + 50.0), places=12)
        self.assertAlmostEqual((lat[1] - LAT_42N) * (r_m + 50.0), 550.0, delta=0.01)
        self.assertTrue(np.isnan(lat[2]))

    def test_reference_valid_mask_drops_disagreement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_synthetic_trial(root)
            trial = load_mems_nav_trial(root, "trial-a")
            # Reference at t = 2 s is at (40, -75); the fix at 2 s is ~14 m away; t = 3 s has no fix.
            self.assertEqual(reference_valid_mask(trial).tolist(), [False, False])
            self.assertEqual(reference_valid_mask(trial, max_disagreement_m=20.0).tolist(), [True, False])

    def test_chi_square_helpers(self) -> None:
        low, high = chi2_interval(3, 0.95)
        self.assertAlmostEqual(low, 0.2158, places=3)
        self.assertAlmostEqual(high, 9.3484, places=3)
        low, high = chi2_interval(3, 0.95, num_averaged=100)
        self.assertTrue(2.4 < low < 3.0 < high < 3.6)
        self.assertAlmostEqual(chi2_gate(3, 0.999), 16.266, places=2)
        self.assertAlmostEqual(fraction_inside(np.array([0.1, 1.0, 3.0, 20.0]), 3), 0.5)
        np.testing.assert_allclose(nees(np.array([[1.0, 2.0]]), np.array([np.diag([1.0, 4.0])])), [2.0])


class FilterBookkeepingTests(unittest.TestCase):
    def test_measurement_covariance_from_receiver_accuracy(self) -> None:
        fix = gnss_fix(0.0, 0.7, 0.1, 10.0, horizontal_accuracy_m=3.03, vertical_accuracy_m=4.0)
        R = gnss_position_covariance(fix)
        np.testing.assert_allclose(np.sqrt(np.diag(R)), [2.0, 2.0, 4.0], rtol=1e-3)
        R = gnss_position_covariance(gnss_fix(0.0, 0.7, 0.1, 10.0, horizontal_accuracy_m=0.5), floor_m=1.0)
        self.assertAlmostEqual(R[0, 0], 1.0)
        self.assertAlmostEqual(R[2, 2], 64.0)  # default vertical sigma

    def test_filter_has_room_for_your_state(self) -> None:
        flt = INSGNSSFilter(NavState(0.7, 0.1, 10.0), np.eye(STATE_DIM))
        self.assertEqual(flt.P.shape, (15, 15))
        self.assertEqual(flt.delta_x.shape, (15,))
        self.assertEqual(flt.innovations, [])
        flt.P = 2.0 * flt.P
        flt.my_history = []  # a plain dataclass: add attributes freely
        self.assertEqual(flt.nis_values().size, 0)
        noise = IMUNoise()
        self.assertEqual(noise.psd().shape, (12,))

    def test_state_copy_is_deep(self) -> None:
        state = NavState(0.7, 0.1, 10.0, [1.0, 2.0, 3.0])
        copy = state.copy()
        copy.velocity_ned_mps[0] = 99.0
        self.assertEqual(state.velocity_ned_mps[0], 1.0)
        np.testing.assert_allclose(state.position, [0.7, 0.1, 10.0])


@unittest.skipIf(REAL_DATA is None, FETCH_HINT)
class RealDataTests(unittest.TestCase):
    def test_real_dataset_facts(self) -> None:
        trials = list_mems_nav_trials()
        self.assertGreaterEqual(len(trials), 10)
        name = "2023-08-09_163741"
        if name not in trials:
            self.skipTest(f"{name} not present")
        trial = load_mems_nav_trial(None, name)
        t_imu, accel, _ = imu_arrays(trial.imu_samples)
        self.assertAlmostEqual(float(np.median(np.diff(t_imu))), 1.0)  # 1 Hz IMU
        self.assertAlmostEqual(float(np.max(np.diff(t_imu))), 1.0)
        self.assertGreater(np.mean(accel[:, 1]), 9.0)  # phone upright: gravity reaction on +y
        # With one epoch and the 1 s label offset the baseline sits on the GNSS fixes.
        gnss = gnss_arrays(trial.gnss_samples)
        for offset, low, high in ((BASELINE_TIME_OFFSET_S, 0.0, 1.5), (0.0, 10.0, 1e9)):
            ref = reference_arrays(load_mems_nav_trial(None, name, baseline_time_offset_s=offset).reference_states)
            _, error = position_error_ned(
                gnss["time_s"], gnss["latitude_rad"], gnss["longitude_rad"], gnss["altitude_m"],
                ref["time_s"], ref["latitude_rad"], ref["longitude_rad"], ref["altitude_m"],
            )
            median = float(np.median(np.hypot(error[:, 0], error[:, 1])))
            self.assertTrue(low <= median <= high, f"offset {offset}: median {median:.2f} m")

    def test_recommended_trials_are_present_and_upright(self) -> None:
        names = recommended_trials()
        self.assertIn("2023-08-09_163741", names)
        trial = load_mems_nav_trial(None, "2023-08-09_163741")
        C_pv = estimate_mount_rotation(trial.imu_samples, trial.gnss_samples)
        self.assertGreater(-C_pv[0, 2], 0.98)  # vehicle forward is the phone's -z
        self.assertGreater(-C_pv[2, 1], 0.98)  # vehicle down is the phone's -y

    def test_run_all_does_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.assertEqual(run_all.main(["--trial", "2023-08-09_163741", "--out", tmp_dir]), 0)
            self.assertTrue((Path(tmp_dir) / "summary.md").exists())


# ---------------------------------------------------------------------------
# Part A deliverables (skip while they are stubs)
# ---------------------------------------------------------------------------


class MechanizationTests(unittest.TestCase):
    def run_steps(self, state: NavState, f_b: np.ndarray, w_b: np.ndarray, dt: float, steps: int) -> NavState:
        for _ in range(steps):
            state = call_or_skip(self, strapdown.mechanize, state, f_b, w_b, dt)
        return state

    def test_perfect_imu_at_rest_stays_at_rest(self) -> None:
        """60 s at 42 deg N: Earth rate must be removed from the attitude update and gravity must be right."""
        C = euler_to_dcm(math.radians(3.0), math.radians(-2.0), math.radians(30.0))
        f_b = C.T @ -gravity_ned(LAT_42N, 100.0)
        w_b = C.T @ earth_rate_ned(LAT_42N)
        start = NavState(LAT_42N, -1.3, 100.0, np.zeros(3), dcm_to_quat(C))
        for dt, steps in ((1.0, 60), (0.1, 600)):
            with self.subTest(dt=dt):
                end = self.run_steps(start, f_b, w_b, dt, steps)
                self.assertLess(np.linalg.norm(end.velocity_ned_mps), 1e-4, end.velocity_ned_mps)
                drift = geodetic_to_ned_delta(end.latitude_rad, end.longitude_rad, end.height_m, LAT_42N, -1.3, 100.0)
                self.assertLess(np.linalg.norm(drift), 1e-3)
                self.assertLess(np.linalg.norm(attitude_error(C, quat_to_dcm(end.q_bn))), 1e-8)
                self.assertAlmostEqual(np.linalg.norm(end.q_bn), 1.0, places=12)

    def test_constant_velocity_east_needs_coriolis_and_transport_rate(self) -> None:
        velocity, C, f_b, w_b = perfect_imu_level_east(LAT_42N, 100.0, 30.0)
        start = NavState(LAT_42N, 0.0, 100.0, velocity, dcm_to_quat(C))
        end = self.run_steps(start, f_b, w_b, 1.0, 60)
        np.testing.assert_allclose(end.velocity_ned_mps, velocity, atol=1e-4)
        self.assertLess(np.linalg.norm(attitude_error(C, quat_to_dcm(end.q_bn))), 1e-8)
        _, r_n = radii_of_curvature(LAT_42N)
        self.assertAlmostEqual(end.longitude_rad * (r_n + 100.0) * math.cos(LAT_42N), 1800.0, delta=0.01)
        self.assertAlmostEqual(end.latitude_rad, LAT_42N, delta=1e-9)
        self.assertAlmostEqual(end.height_m, 100.0, delta=1e-3)

    def test_biases_are_subtracted_and_input_is_not_modified(self) -> None:
        velocity, C, f_b, w_b = perfect_imu_level_east(LAT_42N, 100.0, 20.0)
        b_a, b_g = np.array([0.1, -0.05, 0.2]), np.array([0.01, 0.002, -0.003])
        start = NavState(LAT_42N, 0.0, 100.0, velocity, dcm_to_quat(C), b_a, b_g)
        before = start.copy()
        end = call_or_skip(self, strapdown.mechanize, start, f_b + b_a, w_b + b_g, 1.0)
        np.testing.assert_allclose(end.velocity_ned_mps, velocity, atol=1e-6)
        np.testing.assert_allclose(end.accel_bias_mps2, b_a)
        np.testing.assert_allclose(end.gyro_bias_rps, b_g)
        np.testing.assert_allclose(start.velocity_ned_mps, before.velocity_ned_mps)
        np.testing.assert_allclose(start.q_bn, before.q_bn)
        self.assertEqual(start.latitude_rad, before.latitude_rad)

    def test_free_fall_and_rotation_direction(self) -> None:
        start = NavState(LAT_42N, 0.0, 1000.0, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
        end = call_or_skip(self, strapdown.mechanize, start, np.zeros(3), np.zeros(3), 1.0)
        g = compute_gravity(LAT_42N, 1000.0)
        self.assertAlmostEqual(end.velocity_ned_mps[2], g, delta=1e-3)  # falls (down is positive)
        self.assertAlmostEqual(end.height_m, 1000.0 - g / 2.0, delta=1e-2)  # trapezoidal height
        # A positive body z rate (yaw right, FRD) turns the nose from north toward east.
        end = call_or_skip(self, strapdown.mechanize, start, -gravity_ned(LAT_42N, 1000.0), np.array([0.0, 0.0, 0.1]), 1.0)
        self.assertAlmostEqual(dcm_to_euler(quat_to_dcm(end.q_bn))[2], 0.1, delta=1e-4)


class AlignmentTests(unittest.TestCase):
    def test_leveling_recovers_roll_and_pitch(self) -> None:
        g = compute_gravity(LAT_42N, 0.0)
        for roll_deg, pitch_deg, yaw_deg in ((0.0, 0.0, 0.0), (5.0, -3.0, 40.0), (-20.0, 10.0, -120.0), (2.0, 30.0, 179.0)):
            with self.subTest(roll=roll_deg, pitch=pitch_deg):
                C = euler_to_dcm(*np.radians([roll_deg, pitch_deg, yaw_deg]))
                f_b = C.T @ np.array([0.0, 0.0, -g])
                q = call_or_skip(self, strapdown.coarse_alignment, f_b, math.radians(yaw_deg))
                self.assertAlmostEqual(np.linalg.norm(q), 1.0, places=9)
                np.testing.assert_allclose(quat_to_dcm(q), C, atol=1e-9)


# ---------------------------------------------------------------------------
# Part B deliverables (skip while they are stubs)
# ---------------------------------------------------------------------------


def nominal_state() -> NavState:
    """A moving, tilted, biased nominal state for Jacobian checks."""
    return NavState(
        LAT_42N,
        -1.3,
        120.0,
        np.array([12.0, -7.0, 0.5]),
        euler_to_quat(math.radians(4.0), math.radians(-6.0), math.radians(130.0)),
        np.array([0.05, -0.03, 0.02]),
        np.array([1e-3, -2e-3, 5e-4]),
    )


class ErrorModelTests(unittest.TestCase):
    def test_jacobian_has_the_short_horizon_structure(self) -> None:
        state = nominal_state()
        f_b = np.array([0.8, -0.4, -9.7])
        F = call_or_skip(self, eskf.error_state_jacobian, state, f_b)
        C = quat_to_dcm(state.q_bn)
        self.assertEqual(F.shape, (15, 15))
        np.testing.assert_allclose(F[POS, VEL], np.eye(3))
        np.testing.assert_allclose(F[VEL, ATT], -skew(C @ f_b), atol=1e-12)
        np.testing.assert_allclose(F[VEL, ACC_BIAS], -C, atol=1e-12)
        np.testing.assert_allclose(F[ATT, GYRO_BIAS], -C, atol=1e-12)

    def test_jacobian_matches_your_mechanization_numerically(self) -> None:
        """Propagate nominal and perturbed states with ``mechanize``; their error must evolve as exp(F dt)."""
        state = nominal_state()
        C = quat_to_dcm(state.q_bn)
        accel = C.T @ (np.array([0.6, -0.9, 0.2]) - gravity_ned(state.latitude_rad, state.height_m)) + state.accel_bias_mps2
        gyro = np.array([0.02, -0.01, 0.15]) + state.gyro_bias_rps
        dt = 0.01
        F = call_or_skip(self, eskf.error_state_jacobian, state, accel - state.accel_bias_mps2)
        nominal_next = call_or_skip(self, strapdown.mechanize, state, accel, gyro, dt)
        scales = np.array([1.0] * 3 + [0.1] * 3 + [1e-3] * 3 + [1e-2] * 3 + [1e-4] * 3)
        Phi = np.zeros((15, 15))
        for j in range(15):
            step = np.zeros(15)
            step[j] = scales[j]
            plus = strapdown.mechanize(perturb(state, step), accel, gyro, dt)
            minus = strapdown.mechanize(perturb(state, -step), accel, gyro, dt)
            Phi[:, j] = (error_between(plus, nominal_next) - error_between(minus, nominal_next)) / (2.0 * step[j])
        expected = expm(F * dt)
        # A sign error in any block of F shows up at the 1e-2 level here. The
        # couplings the short-horizon model neglects (Earth rate, transport
        # rate, gravity gradient) and the third-order terms are below 1e-5.
        np.testing.assert_allclose(Phi, expected, atol=2e-5, err_msg="exp(F dt) vs numerical Jacobian of mechanize")

    def test_process_noise_is_symmetric_and_scaled_by_dt(self) -> None:
        state = nominal_state()
        noise = IMUNoise(accel_vrw=0.05, gyro_arw=2e-4, accel_bias_rw=3e-4, gyro_bias_rw=2e-5)
        for dt in (0.01, 1.0):
            with self.subTest(dt=dt):
                Q = call_or_skip(self, eskf.discrete_process_noise, state, noise, dt)
                self.assertEqual(Q.shape, (15, 15))
                np.testing.assert_allclose(Q, Q.T, atol=1e-18)
                self.assertGreaterEqual(np.min(np.linalg.eigvalsh(Q)), -1e-15)
                np.testing.assert_allclose(np.diag(Q)[VEL], noise.accel_vrw**2 * dt, rtol=0.05)
                np.testing.assert_allclose(np.diag(Q)[ATT], noise.gyro_arw**2 * dt, rtol=0.05)
                np.testing.assert_allclose(np.diag(Q)[ACC_BIAS], noise.accel_bias_rw**2 * dt, rtol=0.05)
                np.testing.assert_allclose(np.diag(Q)[GYRO_BIAS], noise.gyro_bias_rw**2 * dt, rtol=0.05)


class GNSSUpdateTests(unittest.TestCase):
    def test_residual_is_the_ned_offset_of_the_fix(self) -> None:
        state = nominal_state()
        offset = np.array([4.0, -3.0, 1.5])
        d_lat, d_lon, d_h = ned_to_geodetic_delta(offset, state.latitude_rad, state.height_m)
        fix = gnss_fix(0.0, state.latitude_rad + d_lat, state.longitude_rad + d_lon, state.height_m + d_h)
        nu, H = call_or_skip(self, eskf.gnss_position_residual, state, fix, np.zeros(3))
        np.testing.assert_allclose(nu, offset, atol=1e-6)
        np.testing.assert_allclose(H[:, POS], np.eye(3))
        np.testing.assert_allclose(np.delete(H, [0, 1, 2], axis=1), 0.0, atol=1e-15)

    def test_lever_arm_residual_and_jacobian(self) -> None:
        state = nominal_state()
        lever = np.array([0.5, -0.3, -1.2])
        C = quat_to_dcm(state.q_bn)
        # A fix exactly at the antenna of the nominal state has zero residual.
        d_lat, d_lon, d_h = ned_to_geodetic_delta(C @ lever, state.latitude_rad, state.height_m)
        fix = gnss_fix(0.0, state.latitude_rad + d_lat, state.longitude_rad + d_lon, state.height_m + d_h)
        nu0, H = call_or_skip(self, eskf.gnss_position_residual, state, fix, lever)
        np.testing.assert_allclose(nu0, 0.0, atol=1e-6)
        # nu(true) ~ nu(nominal) - H dx, so -d(nu)/d(dx) = H.
        numerical = np.zeros((3, 15))
        for j in range(15):
            step = np.zeros(15)
            step[j] = 1e-4
            plus, _ = eskf.gnss_position_residual(perturb(state, step), fix, lever)
            minus, _ = eskf.gnss_position_residual(perturb(state, -step), fix, lever)
            numerical[:, j] = -(plus - minus) / 2e-4
        np.testing.assert_allclose(H, numerical, atol=1e-5)
        np.testing.assert_allclose(H[:, ATT], -skew(C @ lever), atol=1e-12)

    def test_joseph_update_matches_the_textbook_filter(self) -> None:
        rng = np.random.default_rng(7)
        A = rng.standard_normal((15, 15))
        P = A @ A.T + 15.0 * np.eye(15)
        H = np.zeros((3, 15))
        H[:, POS] = np.eye(3)
        H[:, ATT] = -skew([0.2, 0.1, -0.5])
        R = np.diag([4.0, 4.0, 9.0])
        nu = np.array([1.0, -2.0, 0.5])
        delta_x, P_plus, nis = call_or_skip(self, eskf.joseph_update, P, nu, H, R)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        np.testing.assert_allclose(delta_x, K @ nu, atol=1e-10)
        np.testing.assert_allclose(P_plus, (np.eye(15) - K @ H) @ P, atol=1e-8)
        np.testing.assert_allclose(P_plus, P_plus.T, atol=1e-12)
        self.assertAlmostEqual(nis, float(nu @ np.linalg.inv(S) @ nu), places=10)
        self.assertGreater(np.min(np.linalg.eigvalsh(P_plus)), 0.0)

    def test_injection_applies_every_block(self) -> None:
        state = nominal_state()
        before = state.copy()
        delta_x = np.concatenate([[10.0, -5.0, 2.0], [0.1, 0.2, -0.3], [0.01, -0.02, 0.03], [0.01, 0.02, 0.03], [1e-4, 2e-4, 3e-4]])
        new = call_or_skip(self, eskf.inject_error_state, state, delta_x)
        np.testing.assert_allclose(error_between(new, state), delta_x, atol=1e-7)
        np.testing.assert_allclose(quat_to_dcm(new.q_bn), expm(skew(delta_x[ATT])) @ quat_to_dcm(state.q_bn), atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(new.q_bn), 1.0, places=12)
        np.testing.assert_allclose(state.velocity_ned_mps, before.velocity_ned_mps)  # input unchanged

    def test_reset_jacobian_matches_the_exact_attitude_reset(self) -> None:
        rng = np.random.default_rng(3)
        A = rng.standard_normal((15, 15))
        P = A @ A.T
        dtheta_hat = np.radians([2.0, -3.0, 4.0])
        P_reset = call_or_skip(self, eskf.reset_covariance, P, dtheta_hat)
        np.testing.assert_allclose(P_reset, P_reset.T, atol=1e-12)
        np.testing.assert_allclose(call_or_skip(self, eskf.reset_covariance, P, np.zeros(3)), P, atol=1e-12)

        def reset(dtheta):  # exact new error: Exp(dtheta+) = Exp(dtheta) Exp(-dtheta_hat)
            return dcm_to_rotvec(expm(skew(dtheta)) @ expm(skew(-dtheta_hat)))

        G_att = np.column_stack(
            [(reset(dtheta_hat + 1e-7 * e) - reset(dtheta_hat - 1e-7 * e)) / 2e-7 for e in np.eye(3)]
        )
        G = np.eye(15)
        G[ATT, ATT] = G_att
        np.testing.assert_allclose(P_reset, G @ P @ G.T, atol=2e-3 * np.max(np.abs(P)))


class FilterIntegrationTests(unittest.TestCase):
    """The whole chain on a synthetic drive due east, perfect IMU plus biases and white noise."""

    def test_filter_is_consistent_on_a_synthetic_drive(self) -> None:
        rng = np.random.default_rng(11)
        speed, height = 20.0, 100.0
        velocity, C, f_b, w_b = perfect_imu_level_east(LAT_42N, height, speed)
        noise = IMUNoise(accel_vrw=0.02, gyro_arw=np.radians(0.01), accel_bias_rw=1e-4, gyro_bias_rw=np.radians(1e-4))
        sigma0 = np.concatenate([[2.0] * 3, [0.2] * 3, np.radians([0.5, 0.5, 1.0]), [0.02] * 3, np.radians([0.01] * 3)])
        P0 = np.diag(sigma0**2)
        truth = NavState(LAT_42N, 0.0, height, velocity, dcm_to_quat(C), rng.normal(0.0, sigma0[ACC_BIAS]), rng.normal(0.0, sigma0[GYRO_BIAS]))
        start = perturb(truth, -rng.normal(0.0, sigma0))  # nominal = truth minus a draw from P0
        start.accel_bias_mps2 = np.zeros(3)
        start.gyro_bias_rps = np.zeros(3)
        flt = INSGNSSFilter(start, P0, noise=noise)
        _, r_n = radii_of_curvature(LAT_42N)
        sigma_gnss = 1.0
        nees_values = []
        for k in range(1, 301):
            accel = f_b + truth.accel_bias_mps2 + rng.normal(0.0, noise.accel_vrw, 3)
            gyro = w_b + truth.gyro_bias_rps + rng.normal(0.0, noise.gyro_arw, 3)
            call_or_skip(self, flt.predict, accel, gyro, 1.0)
            true_lon = speed * k / ((r_n + height) * math.cos(LAT_42N))
            d_lat, d_lon, d_h = ned_to_geodetic_delta(rng.normal(0.0, sigma_gnss, 3), LAT_42N, height)
            noisy = gnss_fix(float(k), LAT_42N + d_lat, true_lon + d_lon, height + d_h)
            call_or_skip(self, flt.update_gnss_position, noisy, np.eye(3) * sigma_gnss**2)
            true_state = NavState(LAT_42N, true_lon, height, velocity, truth.q_bn, truth.accel_bias_mps2, truth.gyro_bias_rps)
            error = error_between(true_state, flt.state)[:9]
            nees_values.append(float(error @ np.linalg.solve(flt.P[:9, :9], error)))
        nis = flt.nis_values()
        self.assertEqual(len(nis), 300)
        self.assertTrue(2.0 < float(np.mean(nis)) < 4.0, f"mean NIS {np.mean(nis):.2f} (expected about 3)")
        self.assertTrue(5.0 < float(np.mean(nees_values[50:])) < 14.0, f"mean NEES {np.mean(nees_values[50:]):.2f} (9 dof)")
        np.testing.assert_allclose(flt.P, flt.P.T, atol=1e-9)
        self.assertGreater(np.min(np.linalg.eigvalsh(flt.P)), 0.0)
        self.assertTrue(np.allclose(flt.delta_x, 0.0))


if __name__ == "__main__":
    unittest.main()
