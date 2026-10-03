"""Loaders and data-quality helpers for the INS/GNSS project (Project 3).

The main dataset is the MEMS navigation dataset
(https://github.com/jbrodovsky/mems-nav-dataset): smartphone IMU and GNSS
logged in a car, one row per second, plus a ``baseline/`` reference solution
for each trial. Fetch it from the repository root with::

    uv run python scripts/fetch_data.py mems-nav-dataset

Facts about the data that the code below handles (all verified on the files;
the handout, @sec-project3-data, explains them):

* Every trial is **1 Hz**, IMU included. Timestamps are whole UTC seconds.
* IMU axes are the phone's (Android sensor frame: ``x`` right, ``y`` toward
  the top of the screen, ``z`` out of the screen); the accelerometer reads
  specific force (``+9.8`` m/s^2 on the axis pointing up when at rest).
* Input and baseline times are measured from **one** epoch, the first input
  timestamp. The baseline rows are additionally labelled one second early
  relative to the GNSS fixes they were computed from; ``baseline_time_offset_s``
  (default :data:`BASELINE_TIME_OFFSET_S`) corrects that (GitHub issue #5).
* The baseline is a GNSS-aided reference solution, not independent truth. It
  diverges by up to kilometers in some GNSS gaps.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .data_paths import find_dataset_dir
from .earth import geodetic_to_ned_delta
from .types import (
    GNSSSample,
    IMUSample,
    INSGNSSDataset,
    MEMSNavTrial,
    MEMSReferenceState,
)

MEMS_DATASET_NAME = "mems-nav-dataset"

BASELINE_TIME_OFFSET_S = 1.0
"""Seconds added to every baseline timestamp (after referencing it to the input epoch).

Measured on the data: with one common epoch, the baseline row stamped ``t``
matches the GNSS fix stamped ``t + 1 s`` (median horizontal disagreement
0.4-0.8 m on 14 of 17 trials, versus 15-30 m with no offset, while driving at
15-35 m/s). Reported as GitHub issue #5 on the dataset; pass
``baseline_time_offset_s=0.0`` to see the raw labels.
"""


@dataclass(frozen=True, slots=True)
class TrialNote:
    """Why a trial is recommended, usable with care, or excluded."""

    status: str  # "recommended", "caution", or "excluded"
    reason: str


TRIAL_NOTES: dict[str, TrialNote] = {
    "2023-08-06_144805": TrialNote("recommended", "2.5 h, no GNSS gaps, upright mount; the phone is handled in the last 15 s, after parking"),
    "2023-08-09_163741": TrialNote("recommended", "43 min, no GNSS gaps; the short one, good for development"),
    "2025-03-01_164639": TrialNote("recommended", "1.5 h, no GNSS gaps, upright mount with a ~17 deg yaw offset"),
    "2025-07-31_23-36-03": TrialNote("recommended", "3.7 h, no GNSS gaps, upright mount"),
    "2025-08-03_18-15-59": TrialNote("recommended", "4.1 h, two GNSS gaps of 7 s or less, upright mount"),
    "2025-03-01_150426": TrialNote("caution", "one 7 s GNSS gap; receiver accuracy pinned at 3.8 m most of the time"),
    "2025-06-18_16-52-32": TrialNote("caution", "phone lying flat (gravity on +z); baseline |v_D| spikes near 7131 s"),
    "2025-07-18_23-13-43": TrialNote(
        "caution", "a 37 s GNSS gap from 3410 s, during which the baseline diverges 0.7 km"
    ),
    "2023-08-04_214758": TrialNote(
        "caution", "74 GNSS gaps up to 63 s, baseline diverges up to 6 km, phone taken out of its mount near the end"
    ),
    "2023-08-09_124742": TrialNote("caution", "phone moved within the car mid-trial; baseline diverges 3.9 km near 3724 s"),
    "2025-06-18_15-09-25": TrialNote("caution", "phone lying flat; a 92 s GNSS gap after which the baseline jumps 1.9 km"),
    "2025-06-27_11-54-35": TrialNote(
        "caution", "5 h; repeated (stale) fixes while moving, receiver vertical accuracy ~19 m"
    ),
    "2025-07-04_17-24-46": TrialNote("caution", "a 43 s GNSS gap after which the baseline jumps 6.5 km"),
    "2025-07-08_14-12-53": TrialNote("caution", "phone orientation changes mid-trial; 33 short GNSS gaps"),
    "2024-06-20_165550": TrialNote("excluded", "baseline jumps 145 km (speeds of 290 km/s) near 2790 s, with GNSS present"),
    "2025-06-11_20-34-24": TrialNote("excluded", "only 443 IMU samples over 27 min; baseline 0.5-3 km from GNSS"),
    "2025-06-14_21-17-02": TrialNote(
        "excluded", "GNSS every ~5 s with jumps up to 2 km; the baseline diverges by 7 km"
    ),
}
"""Per-trial notes, checked against the data (see the handout's data section)."""


def recommended_trials(root: str | Path | None = None) -> list[str]:
    """Trials vetted for Project 3 (``TRIAL_NOTES[...].status == "recommended"``) that are on disk.

    Sorted shortest first is *not* guaranteed; the order is chronological.
    """
    available = set(list_mems_nav_trials(root))
    return [name for name, note in TRIAL_NOTES.items() if note.status == "recommended" and name in available]


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    result = float(value)
    return None if np.isnan(result) else result


def _parse_timestamp(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _relative_time_s(value: str, epoch: datetime) -> float:
    return (_parse_timestamp(value) - epoch).total_seconds()


# ---------------------------------------------------------------------------
# Conventional layout: imu.csv / gnss.csv
# ---------------------------------------------------------------------------


def load_ins_gnss_data(root: str | Path) -> INSGNSSDataset:
    """Load a conventional local INS/GNSS dataset directory.

    Expected files: ``imu.csv`` (``time_s, ax, ay, az, gx, gy, gz``),
    ``gnss.csv`` (``time_s, latitude_rad, longitude_rad, altitude_m`` and
    optionally ``vn, ve, vd``), optional ``ground_truth.csv`` and
    ``metadata.json``. Useful for your own synthetic tests.
    """
    root = Path(root)
    imu_path = root / "imu.csv"
    gnss_path = root / "gnss.csv"
    if not imu_path.exists() or not gnss_path.exists():
        raise FileNotFoundError("expected imu.csv and gnss.csv under the dataset root")

    imu_samples = [
        IMUSample(
            time_s=float(row["time_s"]),
            accel_mps2=np.array([float(row["ax"]), float(row["ay"]), float(row["az"])], dtype=np.float64),
            gyro_rps=np.array([float(row["gx"]), float(row["gy"]), float(row["gz"])], dtype=np.float64),
        )
        for row in _load_csv_rows(imu_path)
    ]
    gnss_samples = [
        GNSSSample(
            time_s=float(row["time_s"]),
            latitude_rad=float(row["latitude_rad"]),
            longitude_rad=float(row["longitude_rad"]),
            altitude_m=float(row["altitude_m"]),
            velocity_ned_mps=(
                np.array([float(row["vn"]), float(row["ve"]), float(row["vd"])], dtype=np.float64)
                if {"vn", "ve", "vd"}.issubset(row) and all(row[column] != "" for column in ("vn", "ve", "vd"))
                else None
            ),
        )
        for row in _load_csv_rows(gnss_path)
    ]

    ground_truth_path = root / "ground_truth.csv"
    metadata_path = root / "metadata.json"
    ground_truth = None
    metadata = None
    if ground_truth_path.exists():
        ground_truth = np.array(
            [
                [float(row["time_s"]), float(row["latitude_rad"]), float(row["longitude_rad"]), float(row["altitude_m"])]
                for row in _load_csv_rows(ground_truth_path)
            ],
            dtype=np.float64,
        )
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())

    return INSGNSSDataset(root=root, imu_samples=imu_samples, gnss_samples=gnss_samples, ground_truth=ground_truth, metadata=metadata)


# ---------------------------------------------------------------------------
# MEMS navigation dataset
# ---------------------------------------------------------------------------


def default_mems_root(root: str | Path | None = None) -> Path:
    """Return the fetched MEMS dataset folder (the one holding ``input/`` and ``baseline/``).

    Fetch it with ``uv run python scripts/fetch_data.py mems-nav-dataset``.
    """
    return find_dataset_dir(MEMS_DATASET_NAME, root, required=("input", "baseline"))


def list_mems_nav_trials(root: str | Path | None = None) -> list[str]:
    """Return trial names that have both an input and a baseline file.

    With no ``root``, the fetched dataset under ``data/mems-nav-dataset`` is used.
    """
    root = default_mems_root() if root is None else Path(root)
    input_dir = root / "input"
    baseline_dir = root / "baseline"
    if not input_dir.exists() or not baseline_dir.exists():
        return []
    input_trials = {path.stem for path in input_dir.glob("*.csv")}
    baseline_trials = {path.stem for path in baseline_dir.glob("*.csv")}
    return sorted(input_trials & baseline_trials)


def load_mems_nav_trial(
    root: str | Path | None,
    trial_name: str,
    *,
    baseline_time_offset_s: float = BASELINE_TIME_OFFSET_S,
) -> MEMSNavTrial:
    """Load one MEMS navigation dataset trial.

    Parameters
    ----------
    root:
        Folder holding ``input/`` and ``baseline/``; ``None`` finds the fetched
        copy under ``data/mems-nav-dataset``.
    trial_name:
        A name from :func:`list_mems_nav_trials` (a file name also works).
    baseline_time_offset_s:
        Added to every baseline time after referencing it to the input epoch.
        The default (:data:`BASELINE_TIME_OFFSET_S`, 1 s) is the measured
        labelling offset of the baseline files (GitHub issue #5).

    Every ``time_s`` (IMU, GNSS, reference) is seconds since the first input
    timestamp, ``metadata["epoch_utc"]``. IMU rows with a missing value are
    dropped; GNSS fixes are the rows with a latitude, longitude and altitude.
    """
    root = default_mems_root() if root is None else Path(root)
    normalized_name = Path(trial_name).stem
    input_path = root / "input" / f"{normalized_name}.csv"
    baseline_path = root / "baseline" / f"{normalized_name}.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"missing MEMS input file: {input_path}")
    if not baseline_path.exists():
        raise FileNotFoundError(f"missing MEMS baseline file: {baseline_path}")

    input_rows = _load_csv_rows(input_path)
    baseline_rows = _load_csv_rows(baseline_path)
    if not input_rows:
        raise ValueError(f"MEMS input file is empty: {input_path}")
    if not baseline_rows:
        raise ValueError(f"MEMS baseline file is empty: {baseline_path}")

    # One epoch for everything: the first input timestamp.
    epoch = _parse_timestamp(input_rows[0]["time"])
    metadata = _load_mems_manifest_metadata(root, normalized_name) or {}
    metadata.update(
        {
            "epoch_utc": epoch.isoformat(),
            "baseline_time_offset_s": float(baseline_time_offset_s),
            "note": TRIAL_NOTES.get(normalized_name),
        }
    )

    return MEMSNavTrial(
        name=normalized_name,
        root=root,
        input_path=input_path,
        baseline_path=baseline_path,
        imu_samples=_parse_mems_imu_samples(input_rows, epoch),
        gnss_samples=_parse_mems_gnss_samples(input_rows, epoch),
        reference_states=_parse_mems_reference_states(baseline_rows, epoch, float(baseline_time_offset_s)),
        metadata=metadata,
        auxiliary=_parse_mems_auxiliary(input_rows, epoch),
    )


def _parse_mems_auxiliary(rows: list[dict[str, str]], epoch: datetime) -> dict[str, np.ndarray]:
    def column(name: str) -> np.ndarray:
        values = [_float_or_none(row.get(name)) for row in rows]
        return np.array([np.nan if v is None else v for v in values], dtype=np.float64)

    return {
        "time_s": np.array([_relative_time_s(row["time"], epoch) for row in rows], dtype=np.float64),
        "pressure_hpa": column("pressure"),
        "relative_altitude_m": column("relativeAltitude"),
        "mag_ut": np.column_stack([column("mag_x"), column("mag_y"), column("mag_z")]),
        "gravity_mps2": np.column_stack([column("grav_x"), column("grav_y"), column("grav_z")]),
        "orientation_q": np.column_stack([column("qw"), column("qx"), column("qy"), column("qz")]),
    }


def _parse_mems_imu_samples(rows: list[dict[str, str]], epoch: datetime) -> list[IMUSample]:
    samples: list[IMUSample] = []
    for row in rows:
        accel = [_float_or_none(row.get(axis)) for axis in ("acc_x", "acc_y", "acc_z")]
        gyro = [_float_or_none(row.get(axis)) for axis in ("gyro_x", "gyro_y", "gyro_z")]
        if any(value is None for value in accel + gyro):
            continue
        samples.append(
            IMUSample(
                time_s=_relative_time_s(row["time"], epoch),
                accel_mps2=np.array(accel, dtype=np.float64),
                gyro_rps=np.array(gyro, dtype=np.float64),
            )
        )
    return samples


def _parse_mems_gnss_samples(rows: list[dict[str, str]], epoch: datetime) -> list[GNSSSample]:
    samples: list[GNSSSample] = []
    for row in rows:
        latitude_deg = _float_or_none(row.get("latitude"))
        longitude_deg = _float_or_none(row.get("longitude"))
        altitude_m = _float_or_none(row.get("altitude"))
        if latitude_deg is None or longitude_deg is None or altitude_m is None:
            continue
        bearing_deg = _float_or_none(row.get("bearing"))
        bearing_accuracy_deg = _float_or_none(row.get("bearingAccuracy"))
        samples.append(
            GNSSSample(
                time_s=_relative_time_s(row["time"], epoch),
                latitude_rad=float(np.deg2rad(latitude_deg)),
                longitude_rad=float(np.deg2rad(longitude_deg)),
                altitude_m=altitude_m,
                velocity_ned_mps=None,
                speed_mps=_float_or_none(row.get("speed")),
                bearing_rad=None if bearing_deg is None else float(np.deg2rad(bearing_deg)),
                horizontal_accuracy_m=_float_or_none(row.get("horizontalAccuracy")),
                vertical_accuracy_m=_float_or_none(row.get("verticalAccuracy")),
                speed_accuracy_mps=_float_or_none(row.get("speedAccuracy")),
                bearing_accuracy_rad=None if bearing_accuracy_deg is None else float(np.deg2rad(bearing_accuracy_deg)),
            )
        )
    return samples


def _parse_mems_reference_states(
    rows: list[dict[str, str]], epoch: datetime, time_offset_s: float
) -> list[MEMSReferenceState]:
    states: list[MEMSReferenceState] = []
    for row in rows:
        covariance = {key: float(value) for key, value in row.items() if key.endswith("_cov") and value != ""}
        states.append(
            MEMSReferenceState(
                time_s=_relative_time_s(row["timestamp"], epoch) + time_offset_s,
                latitude_rad=float(np.deg2rad(float(row["latitude"]))),
                longitude_rad=float(np.deg2rad(float(row["longitude"]))),
                altitude_m=float(row["altitude"]),
                velocity_ned_mps=np.array(
                    [float(row["velocity_north"]), float(row["velocity_east"]), float(row["velocity_down"])],
                    dtype=np.float64,
                ),
                attitude_rpy_rad=np.array([float(row["roll"]), float(row["pitch"]), float(row["yaw"])], dtype=np.float64),
                accel_bias_mps2=np.array(
                    [float(row["acc_bias_x"]), float(row["acc_bias_y"]), float(row["acc_bias_z"])], dtype=np.float64
                ),
                gyro_bias_rps=np.array(
                    [float(row["gyro_bias_x"]), float(row["gyro_bias_y"]), float(row["gyro_bias_z"])], dtype=np.float64
                ),
                covariance=covariance or None,
            )
        )
    return states


def _load_mems_manifest_metadata(root: Path, trial_name: str) -> dict[str, object] | None:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    trial_files = []
    for dataset in ("input", "baseline"):
        for entry in manifest.get("datasets", {}).get(dataset, []):
            if Path(str(entry.get("name", ""))).stem == trial_name:
                trial_files.append({"dataset": dataset, **entry})
    return {"source": manifest.get("repo_url"), "ref": manifest.get("ref"), "files": trial_files}


# ---------------------------------------------------------------------------
# Arrays
# ---------------------------------------------------------------------------


def imu_arrays(samples: list[IMUSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(time_s (N,), accel_mps2 (N, 3), gyro_rps (N, 3))``."""
    time_s = np.array([sample.time_s for sample in samples], dtype=np.float64)
    accel = np.array([sample.accel_mps2 for sample in samples], dtype=np.float64).reshape(-1, 3)
    gyro = np.array([sample.gyro_rps for sample in samples], dtype=np.float64).reshape(-1, 3)
    return time_s, accel, gyro


def gnss_arrays(samples: list[GNSSSample]) -> dict[str, np.ndarray]:
    """GNSS fields as arrays (missing optional values become NaN).

    Keys: ``time_s``, ``latitude_rad``, ``longitude_rad``, ``altitude_m``,
    ``speed_mps``, ``bearing_rad``, ``horizontal_accuracy_m``,
    ``vertical_accuracy_m``, ``speed_accuracy_mps``, ``bearing_accuracy_rad``.
    """

    def column(name: str) -> np.ndarray:
        return np.array([np.nan if getattr(s, name) is None else getattr(s, name) for s in samples], dtype=np.float64)

    names = (
        "time_s",
        "latitude_rad",
        "longitude_rad",
        "altitude_m",
        "speed_mps",
        "bearing_rad",
        "horizontal_accuracy_m",
        "vertical_accuracy_m",
        "speed_accuracy_mps",
        "bearing_accuracy_rad",
    )
    return {name: column(name) for name in names}


def reference_arrays(states: list[MEMSReferenceState]) -> dict[str, np.ndarray]:
    """Reference fields as arrays: ``time_s``, ``latitude_rad``, ``longitude_rad``, ``altitude_m``, ``velocity_ned_mps`` (N, 3)."""
    return {
        "time_s": np.array([s.time_s for s in states], dtype=np.float64),
        "latitude_rad": np.array([s.latitude_rad for s in states], dtype=np.float64),
        "longitude_rad": np.array([s.longitude_rad for s in states], dtype=np.float64),
        "altitude_m": np.array([s.altitude_m for s in states], dtype=np.float64),
        "velocity_ned_mps": np.array([s.velocity_ned_mps for s in states], dtype=np.float64).reshape(-1, 3),
    }


# ---------------------------------------------------------------------------
# Data quality and mounting
# ---------------------------------------------------------------------------


def repeated_fix_mask(gnss_samples: list[GNSSSample], *, min_speed_mps: float = 0.5) -> np.ndarray:
    """Flag stale fixes: the same latitude and longitude as the previous fix while the receiver reports motion.

    A parked car legitimately repeats its position (most repeats in the data
    happen at zero reported speed); a car moving at ``min_speed_mps`` or more
    cannot. Returns a boolean array, True for fixes you should not use.
    """
    data = gnss_arrays(gnss_samples)
    same = np.zeros(len(gnss_samples), dtype=bool)
    if len(gnss_samples) > 1:
        same[1:] = (np.diff(data["latitude_rad"]) == 0.0) & (np.diff(data["longitude_rad"]) == 0.0)
    speed = np.nan_to_num(data["speed_mps"], nan=0.0)
    return same & (speed >= min_speed_mps)


def gnss_acceleration_frd(gnss_samples: list[GNSSSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vehicle-frame horizontal acceleration implied by GNSS speed and course.

    Returns ``(time_s, a_forward, a_right)`` [m/s^2] from central differences:
    ``a_forward = d(speed)/dt`` and ``a_right = speed * d(course)/dt`` (a right
    turn has a positive course rate and a positive rightward acceleration).
    Values are NaN where the speed is below 3 m/s (the course is noise there)
    or the neighbours are more than 1.5 s away.
    """
    data = gnss_arrays(gnss_samples)
    t, speed, course = data["time_s"], data["speed_mps"], data["bearing_rad"]
    a_forward = np.full(t.shape, np.nan)
    a_right = np.full(t.shape, np.nan)
    if t.size < 3:
        return t, a_forward, a_right
    dt = t[2:] - t[:-2]
    d_course = (course[2:] - course[:-2] + np.pi) % (2 * np.pi) - np.pi
    ok = (np.abs(dt - 2.0) < 1.0) & (np.minimum(speed[2:], speed[:-2]) > 3.0) & (speed[1:-1] > 3.0)
    a_forward[1:-1] = np.where(ok, (speed[2:] - speed[:-2]) / dt, np.nan)
    a_right[1:-1] = np.where(ok, speed[1:-1] * d_course / dt, np.nan)
    return t, a_forward, a_right


def estimate_mount_rotation(
    imu_samples: list[IMUSample],
    gnss_samples: list[GNSSSample],
    *,
    max_horizontal_accel_mps2: float = 6.0,
) -> np.ndarray:
    """Estimate ``C_p^v``, the rotation from phone axes to vehicle forward-right-down axes.

    Two steps (a boresight calibration, done once per trial):

    1. **Down** is the direction of the mean accelerometer reading, negated
       (over a long drive the vehicle's mean acceleration is ~0, so the mean
       specific force points up).
    2. **Forward** is the horizontal direction (in the leveled phone frame)
       that best explains the GNSS-derived forward and rightward
       accelerations (:func:`gnss_acceleration_frd`), found as a 2-D
       orthogonal Procrustes (Wahba) fit over every moving sample.

    The result assumes the phone stays put in its mount (true for the
    recommended trials) and that the car does not slip sideways. Apply it with
    :func:`rotate_imu_samples`. It uses GNSS (an input), never the baseline.
    """
    t_imu, accel, _ = imu_arrays(imu_samples)
    up = np.mean(accel, axis=0)
    down = -up / np.linalg.norm(up)
    helper = np.array([1.0, 0.0, 0.0]) if abs(down[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_level = helper - down * (helper @ down)
    x_level /= np.linalg.norm(x_level)
    y_level = np.cross(down, x_level)
    leveled = np.vstack([x_level, y_level, down])  # rows: leveled axes in phone coordinates

    t_gnss, a_forward, a_right = gnss_acceleration_frd(gnss_samples)
    f_level = accel @ leveled.T
    # Match GNSS samples to the IMU sample at the same time (both are 1 Hz here).
    index = np.searchsorted(t_imu, t_gnss)
    index = np.clip(index, 0, len(t_imu) - 1)
    close = np.abs(t_imu[index] - t_gnss) < 0.25
    usable = close & np.isfinite(a_forward) & np.isfinite(a_right)
    usable &= np.hypot(a_forward, a_right) < max_horizontal_accel_mps2
    if np.count_nonzero(usable) < 30:
        raise ValueError("not enough moving GNSS samples to estimate the mount rotation")
    source = f_level[index[usable], :2]  # horizontal specific force in the leveled phone frame
    target = np.column_stack([a_forward[usable], a_right[usable]])
    # Find the 2-D rotation R (target ~ source R^T) by Procrustes.
    H = source.T @ target
    angle = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    c, s = np.cos(angle), np.sin(angle)
    forward = c * x_level - s * y_level
    right = np.cross(down, forward)
    return np.vstack([forward, right, down])


def rotate_imu_samples(imu_samples: list[IMUSample], rotation: np.ndarray) -> list[IMUSample]:
    """Return new samples with ``accel`` and ``gyro`` premultiplied by ``rotation`` (e.g. ``C_p^v``)."""
    C = np.asarray(rotation, dtype=np.float64)
    return [IMUSample(s.time_s, C @ s.accel_mps2, C @ s.gyro_rps) for s in imu_samples]


def gnss_ned_offsets(gnss_samples: list[GNSSSample], latitude_rad: float, longitude_rad: float, height_m: float) -> np.ndarray:
    """NED offsets [m] of every fix from a reference point (first-order; for plots and sanity checks)."""
    data = gnss_arrays(gnss_samples)
    return geodetic_to_ned_delta(data["latitude_rad"], data["longitude_rad"], data["altitude_m"], latitude_rad, longitude_rad, height_m)
