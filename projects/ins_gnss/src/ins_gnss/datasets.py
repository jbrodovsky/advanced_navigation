"""Loaders and data-quality helpers for the INS/GNSS project (Project 3).

The main dataset is the MEMS navigation dataset [@MemsNavDataset]: smartphone
IMU and GNSS logged in a car, at 10 Hz, plus a ``truth/`` reference solution
for each trajectory. It is the ``Project3-data`` folder of the course data
bundle. Fetch it from the repository root with::

    uv run python scripts/fetch_data.py mems-nav-dataset

Facts about the data that the code below handles (all verified on the files;
the handout, @sec-project3-data, explains them):

* Every row is **0.1 s** apart. Each IMU row is the mean of the phone's raw
  samples in that 0.1 s bin; GNSS fixes stay at their native ~1 Hz and sit on
  the row of the bin they fell in (the other rows leave the GNSS columns empty).
* IMU axes are the phone's (Android sensor frame: ``x`` right, ``y`` toward
  the top of the screen, ``z`` out of the screen); the accelerometer reads
  specific force (``+9.8`` m/s^2 on the axis pointing up when at rest).
* The ``truth/`` file has one row per input row, on the same timestamps. Its
  vertical velocity is positive **up**; the loader flips it to NED down.
* The reference is a GNSS-aided solution, not independent truth. It strays by
  hundreds of meters to kilometers in some long GNSS gaps.
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


@dataclass(frozen=True, slots=True)
class TrialNote:
    """Why a trial is recommended, usable with care, or excluded."""

    status: str  # "recommended", "caution", or "excluded"
    reason: str


TRIAL_NOTES: dict[str, TrialNote] = {
    # Recommended: upright mount, at most one GNSS gap of 6 s, receiver accuracy 1.5-2 m,
    # and the reference within 6 m of 99% of the fixes.
    "2025-07-31_23-36-03": TrialNote("recommended", "3.7 h, no GNSS gaps, upright mount"),
    "2025-08-03_18-15-59": TrialNote("recommended", "4.1 h, one 6 s GNSS gap, upright mount"),
    "2025-09-27_12-54-35": TrialNote("recommended", "2.3 h, no GNSS gaps, upright mount"),
    "2025-09-28_20-23-16": TrialNote("recommended", "3.3 h, one 4 s GNSS gap, upright mount"),
    "2025-11-09_17-34-01_A": TrialNote("recommended", "64 min, no GNSS gaps, upright mount; the short one, good for development"),
    # Caution: usable, with the quirk named.
    "2023-08-04_21-47-58": TrialNote(
        "caution", "42 GNSS gaps up to 63 s, the reference strays 620 m in one; the phone is handled while parked"
    ),
    "2023-08-06_14-48-05": TrialNote("caution", "2.5 h, no GNSS gaps, upright mount; the phone is handled while parked"),
    "2023-08-09_12-47-42": TrialNote("caution", "phone moved within the car mid-trial; the reference is 100 m off after a 21 s gap near 3718 s"),
    "2023-08-09_16-37-41": TrialNote("caution", "43 min, no GNSS gaps, upright mount; the reference sits up to 16 m from the fixes"),
    "2024-06-20_16-55-50": TrialNote("caution", "receiver accuracy ~8 m; a 22 s GNSS gap near 4326 s, in which the reference strays 160 m"),
    "2025-03-01_15-04-26": TrialNote("caution", "one 7 s GNSS gap; receiver accuracy pinned at 3.8 m; the reference sits 4 m (median) from the fixes"),
    "2025-03-01_16-46-39": TrialNote("caution", "no GNSS gaps, upright mount; receiver accuracy pinned at 3.8 m; the reference sits 4 m (median) from the fixes"),
    "2025-06-18_15-09-25": TrialNote("caution", "phone lying flat; a 92 s GNSS gap near 1716 s, in which the reference strays 1.7 km"),
    "2025-06-18_16-52-32": TrialNote("caution", "phone lying flat; the reference is more than 50 m from 20 fixes, from 804 s on"),
    "2025-06-27_11-54-35": TrialNote("caution", "5 h, phone lying flat; 50 repeated (stale) fixes while moving, receiver vertical accuracy ~19 m"),
    "2025-07-04_17-24-46": TrialNote("caution", "phone lying flat and tilted; a 43 s GNSS gap near 1068 s, after which the reference is 320 m off"),
    "2025-07-05_20-59-22": TrialNote("caution", "phone lying flat and tilted; receiver accuracy ~7 m"),
    "2025-07-08_14-12-53": TrialNote("caution", "phone orientation changes mid-trial; the reference is more than 50 m from 17 fixes after 6625 s"),
    "2025-07-11_13-33-16": TrialNote("caution", "clean apart from one 19 s GNSS gap near 1178 s, a natural outage to study"),
    "2025-07-18_23-13-43": TrialNote("caution", "a 37 s GNSS gap near 3410 s, in which the reference strays 320 m"),
    "2025-09-26_19-03-38": TrialNote("caution", "4.6 h, phone lying flat; receiver accuracy ~5 m horizontal, ~10 m vertical"),
    "2025-09-27_18-10-16": TrialNote("caution", "phone lying flat; receiver accuracy ~10 m horizontal, ~19 m vertical; 11 stale fixes while moving"),
    "2025-11-09_17-34-01_B": TrialNote("caution", "10 min, mostly parked, between segments A and C"),
    "2025-11-09_17-34-01_C": TrialNote("caution", "1.4 h, one 4 s GNSS gap, upright mount; a good sixth trial"),
    "2025-11-16_16-08-14": TrialNote("caution", "GNSS gaps of 29 s and 79 s, in which the reference strays 170 m and 370 m"),
    # Excluded.
    "2025-06-11_20-34-24_A": TrialNote("excluded", "7 min at parking-lot speed (11 m/s at most) with the phone in hand"),
    "2025-06-14_21-17-02": TrialNote("excluded", "10 min with a fix only every ~3 s, and the reference up to 2.5 km off"),
}
"""Per-trial notes, checked against the data (see the handout's data section)."""


def recommended_trials(root: str | Path | None = None) -> list[str]:
    """Trials vetted for Project 3 (``TRIAL_NOTES[...].status == "recommended"``) that are on disk.

    The order is chronological; the last one is the short development trial.
    """
    available = set(list_mems_nav_trials(root))
    return [name for name, note in TRIAL_NOTES.items() if note.status == "recommended" and name in available]


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _parse_timestamp(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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
    """Return the fetched MEMS dataset folder (the one holding ``input/`` and ``truth/``).

    Fetch it with ``uv run python scripts/fetch_data.py mems-nav-dataset``.
    """
    return find_dataset_dir(MEMS_DATASET_NAME, root, required=("input", "truth"))


def list_mems_nav_trials(root: str | Path | None = None) -> list[str]:
    """Return trial names that have both an input and a truth file.

    With no ``root``, the fetched dataset under ``data/mems-nav-dataset`` is used.
    """
    root = default_mems_root() if root is None else Path(root)
    input_dir = root / "input"
    truth_dir = root / "truth"
    if not input_dir.exists() or not truth_dir.exists():
        return []
    input_trials = {path.stem for path in input_dir.glob("*.csv")}
    truth_trials = {path.stem for path in truth_dir.glob("*.csv")}
    return sorted(input_trials & truth_trials)


def load_mems_nav_trial(
    root: str | Path | None,
    trial_name: str,
    *,
    reference_time_offset_s: float = 0.0,
    include_covariance: bool = False,
) -> MEMSNavTrial:
    """Load one MEMS navigation dataset trial.

    Parameters
    ----------
    root:
        Folder holding ``input/`` and ``truth/``; ``None`` finds the fetched
        copy under ``data/mems-nav-dataset``.
    trial_name:
        A name from :func:`list_mems_nav_trials` (a file name also works).
    reference_time_offset_s:
        Added to every reference time. The files need none (the reference
        rows share the input timestamps and agree best with the fixes at zero
        offset); it is here so you can check that yourself.
    include_covariance:
        Also keep each reference row's ``*_cov`` columns in
        ``MEMSReferenceState.covariance``. Off by default: at 10 Hz a long
        trial has over 100 000 rows, and a dictionary per row adds up.

    Every ``time_s`` (IMU, GNSS, reference) is seconds since the first input
    timestamp, ``metadata["epoch_utc"]``. IMU rows with a missing value are
    dropped; GNSS fixes are the rows with a latitude, longitude and altitude.
    """
    root = default_mems_root() if root is None else Path(root)
    normalized_name = Path(trial_name).stem
    input_path = root / "input" / f"{normalized_name}.csv"
    truth_path = root / "truth" / f"{normalized_name}.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"missing MEMS input file: {input_path}")
    if not truth_path.exists():
        raise FileNotFoundError(f"missing MEMS truth file: {truth_path}")

    input_times, inputs = _read_numeric_csv(input_path, "time")
    truth_times, truth = _read_numeric_csv(truth_path, "timestamp")
    if input_times.size == 0:
        raise ValueError(f"MEMS input file is empty: {input_path}")
    if truth_times.size == 0:
        raise ValueError(f"MEMS truth file is empty: {truth_path}")

    # One epoch for everything: the first input timestamp.
    epoch = input_times[0]
    t_input = (input_times - epoch) / np.timedelta64(1, "ns") * 1e-9
    t_truth = (truth_times - epoch) / np.timedelta64(1, "ns") * 1e-9 + float(reference_time_offset_s)
    epoch_utc = datetime.fromtimestamp(epoch.astype("datetime64[us]").astype(np.int64) * 1e-6, tz=timezone.utc)
    metadata: dict[str, object] = {
        "source": "course data bundle, Project3-data",
        "epoch_utc": epoch_utc.isoformat(),
        "reference_time_offset_s": float(reference_time_offset_s),
        "note": TRIAL_NOTES.get(normalized_name),
    }

    return MEMSNavTrial(
        name=normalized_name,
        root=root,
        input_path=input_path,
        truth_path=truth_path,
        imu_samples=_parse_mems_imu_samples(t_input, inputs),
        gnss_samples=_parse_mems_gnss_samples(t_input, inputs),
        reference_states=_parse_mems_reference_states(t_truth, truth, include_covariance),
        metadata=metadata,
        auxiliary=_parse_mems_auxiliary(t_input, inputs),
    )


def _parse_utc(strings: list[str]) -> np.ndarray:
    """ISO 8601 UTC strings (``Z`` or ``+00:00`` suffix, ``T`` or space) as ``datetime64[ns]``."""
    cleaned = []
    for value in strings:
        text = value.strip().replace(" ", "T")
        if text.endswith("Z"):
            text = text[:-1]
        elif text.endswith("+00:00"):
            text = text[:-6]
        elif "+" in text[10:] or "-" in text[10:]:
            text = _parse_timestamp(text).replace(tzinfo=None).isoformat()
        cleaned.append(text)
    return np.array(cleaned, dtype="datetime64[ns]")


def _read_numeric_csv(path: Path, time_column: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read a CSV whose columns are all numeric except ``time_column``.

    Returns the timestamps as ``datetime64[ns]`` and one float array per
    other column, NaN where a cell is empty. Faster and far lighter than a
    dictionary per row, which matters at 10 Hz.
    """
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header is None:
            return np.array([], dtype="datetime64[ns]"), {}
        time_index = header.index(time_column)
        times: list[str] = []
        values: list[list[float]] = []
        width = len(header)
        for row in reader:
            if not row:
                continue
            times.append(row[time_index])
            values.append([float(cell) if cell else np.nan for k, cell in enumerate(row[:width]) if k != time_index])
    names = [name for k, name in enumerate(header) if k != time_index]
    table = np.array(values, dtype=np.float64).reshape(-1, len(names))
    return _parse_utc(times), {name: table[:, k] for k, name in enumerate(names)}


def _parse_mems_auxiliary(time_s: np.ndarray, columns: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    def column(name: str) -> np.ndarray:
        return columns.get(name, np.full(time_s.shape, np.nan))

    return {
        "time_s": time_s.copy(),
        "pressure_hpa": column("pressure"),
        "relative_altitude_m": column("relativeAltitude"),
        "mag_ut": np.column_stack([column("mag_x"), column("mag_y"), column("mag_z")]),
        "gravity_mps2": np.column_stack([column("grav_x"), column("grav_y"), column("grav_z")]),
        "orientation_q": np.column_stack([column("qw"), column("qx"), column("qy"), column("qz")]),
    }


def _parse_mems_imu_samples(time_s: np.ndarray, columns: dict[str, np.ndarray]) -> list[IMUSample]:
    accel = np.column_stack([columns["acc_x"], columns["acc_y"], columns["acc_z"]])
    gyro = np.column_stack([columns["gyro_x"], columns["gyro_y"], columns["gyro_z"]])
    keep = np.all(np.isfinite(accel), axis=1) & np.all(np.isfinite(gyro), axis=1)
    return [
        IMUSample(time_s=float(time_s[k]), accel_mps2=accel[k].copy(), gyro_rps=gyro[k].copy())
        for k in np.flatnonzero(keep)
    ]


def _optional(values: np.ndarray, k: int) -> float | None:
    value = float(values[k])
    return None if np.isnan(value) else value


def _parse_mems_gnss_samples(time_s: np.ndarray, columns: dict[str, np.ndarray]) -> list[GNSSSample]:
    latitude, longitude, altitude = columns["latitude"], columns["longitude"], columns["altitude"]
    nan = np.full(time_s.shape, np.nan)
    speed, bearing = columns.get("speed", nan), columns.get("bearing", nan)
    h_acc, v_acc = columns.get("horizontalAccuracy", nan), columns.get("verticalAccuracy", nan)
    s_acc, b_acc = columns.get("speedAccuracy", nan), columns.get("bearingAccuracy", nan)
    samples: list[GNSSSample] = []
    for k in np.flatnonzero(np.isfinite(latitude) & np.isfinite(longitude) & np.isfinite(altitude)):
        bearing_deg = _optional(bearing, k)
        bearing_accuracy_deg = _optional(b_acc, k)
        samples.append(
            GNSSSample(
                time_s=float(time_s[k]),
                latitude_rad=float(np.deg2rad(latitude[k])),
                longitude_rad=float(np.deg2rad(longitude[k])),
                altitude_m=float(altitude[k]),
                velocity_ned_mps=None,
                speed_mps=_optional(speed, k),
                bearing_rad=None if bearing_deg is None else float(np.deg2rad(bearing_deg)),
                horizontal_accuracy_m=_optional(h_acc, k),
                vertical_accuracy_m=_optional(v_acc, k),
                speed_accuracy_mps=_optional(s_acc, k),
                bearing_accuracy_rad=None if bearing_accuracy_deg is None else float(np.deg2rad(bearing_accuracy_deg)),
            )
        )
    return samples


def _parse_mems_reference_states(
    time_s: np.ndarray, columns: dict[str, np.ndarray], include_covariance: bool
) -> list[MEMSReferenceState]:
    latitude = np.deg2rad(columns["latitude"])
    longitude = np.deg2rad(columns["longitude"])
    altitude = columns["altitude"]
    # The file's vertical velocity is positive up; NED wants down.
    velocity = np.column_stack([columns["velocity_north"], columns["velocity_east"], -columns["velocity_vertical"]])
    attitude = np.column_stack([columns["roll"], columns["pitch"], columns["yaw"]])
    accel_bias = np.column_stack([columns["acc_bias_x"], columns["acc_bias_y"], columns["acc_bias_z"]])
    gyro_bias = np.column_stack([columns["gyro_bias_x"], columns["gyro_bias_y"], columns["gyro_bias_z"]])
    covariance_names = [name for name in columns if name.endswith("_cov")] if include_covariance else []
    states: list[MEMSReferenceState] = []
    for k in range(time_s.size):
        covariance = None
        if covariance_names:
            covariance = {name: float(columns[name][k]) for name in covariance_names if np.isfinite(columns[name][k])} or None
        states.append(
            MEMSReferenceState(
                time_s=float(time_s[k]),
                latitude_rad=float(latitude[k]),
                longitude_rad=float(longitude[k]),
                altitude_m=float(altitude[k]),
                velocity_ned_mps=velocity[k].copy(),
                attitude_rpy_rad=attitude[k].copy(),
                accel_bias_mps2=accel_bias[k].copy(),
                gyro_bias_rps=gyro_bias[k].copy(),
                covariance=covariance,
            )
        )
    return states


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
    :func:`rotate_imu_samples`. It uses GNSS (an input), never the reference.
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
    # Average the IMU over the same +-1 s window the GNSS central difference spans.
    lo = np.searchsorted(t_imu, t_gnss - 1.0)
    hi = np.searchsorted(t_imu, t_gnss + 1.0, side="right")
    counts = hi - lo
    sums = np.vstack([np.zeros((1, 3)), np.cumsum(f_level, axis=0)])
    usable = (counts >= 3) & np.isfinite(a_forward) & np.isfinite(a_right)
    usable &= np.hypot(a_forward, a_right) < max_horizontal_accel_mps2
    if np.count_nonzero(usable) < 30:
        raise ValueError("not enough moving GNSS samples to estimate the mount rotation")
    window_mean = (sums[hi[usable]] - sums[lo[usable]]) / counts[usable, None]
    source = window_mean[:, :2]  # horizontal specific force in the leveled phone frame
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
