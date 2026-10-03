"""Flexible loader for INS/GNSS project datasets."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .data_paths import find_dataset_dir
from .types import (
    GNSSSample,
    IMUSample,
    INSGNSSDataset,
    MEMSNavTrial,
    MEMSReferenceState,
)


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _parse_timestamp(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _relative_time_s(value: str, epoch: datetime) -> float:
    return (_parse_timestamp(value) - epoch).total_seconds()


def load_ins_gnss_data(root: str | Path) -> INSGNSSDataset:
    """Load a conventional local INS/GNSS dataset directory.

    Expected files:

    - imu.csv
    - gnss.csv
    - optional ground_truth.csv
    - optional metadata.json
    """
    root = Path(root)
    imu_path = root / "imu.csv"
    gnss_path = root / "gnss.csv"
    if not imu_path.exists() or not gnss_path.exists():
        raise FileNotFoundError("expected imu.csv and gnss.csv under the dataset root")

    imu_rows = _load_csv_rows(imu_path)
    gnss_rows = _load_csv_rows(gnss_path)

    imu_samples = [
        IMUSample(
            time_s=float(row["time_s"]),
            accel_mps2=np.array([float(row["ax"]), float(row["ay"]), float(row["az"])], dtype=np.float64),
            gyro_rps=np.array([float(row["gx"]), float(row["gy"]), float(row["gz"])], dtype=np.float64),
        )
        for row in imu_rows
    ]
    gnss_samples = [
        GNSSSample(
            time_s=float(row["time_s"]),
            latitude_rad=float(row["latitude_rad"]),
            longitude_rad=float(row["longitude_rad"]),
            altitude_m=float(row["altitude_m"]),
            velocity_ned_mps=(
                np.array([float(row["vn"]), float(row["ve"]), float(row["vd"])], dtype=np.float64)
                if {"vn", "ve", "vd"}.issubset(row)
                and all(row[column] != "" for column in ("vn", "ve", "vd"))
                else None
            ),
        )
        for row in gnss_rows
    ]

    ground_truth_path = root / "ground_truth.csv"
    metadata_path = root / "metadata.json"
    ground_truth = None
    metadata = None
    if ground_truth_path.exists():
        ground_truth_rows = _load_csv_rows(ground_truth_path)
        ground_truth = np.array(
            [
                [
                    float(row["time_s"]),
                    float(row["latitude_rad"]),
                    float(row["longitude_rad"]),
                    float(row["altitude_m"]),
                ]
                for row in ground_truth_rows
            ],
            dtype=np.float64,
        )
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())

    return INSGNSSDataset(
        root=root,
        imu_samples=imu_samples,
        gnss_samples=gnss_samples,
        ground_truth=ground_truth,
        metadata=metadata,
    )


MEMS_DATASET_NAME = "mems-nav-dataset"


def default_mems_root(root: str | Path | None = None) -> Path:
    """Return the fetched MEMS dataset folder (the one holding ``input/`` and ``baseline/``).

    Fetch it with ``uv run python scripts/fetch_data.py mems-nav-dataset``.
    """
    return find_dataset_dir(MEMS_DATASET_NAME, root, required=("input", "baseline"))


def list_mems_nav_trials(root: str | Path | None = None) -> list[str]:
    """Return trial names that have both synchronized input and baseline files.

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


def load_mems_nav_trial(root: str | Path | None, trial_name: str) -> MEMSNavTrial:
    """Load one MEMS navigation dataset trial.

    The dataset root should contain the fetched ``input`` and ``baseline``
    directories from https://github.com/jbrodovsky/mems-nav-dataset. Pass
    ``None`` to use the fetched copy under ``data/mems-nav-dataset``.
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

    input_epoch = _parse_timestamp(input_rows[0]["time"])
    baseline_epoch = _parse_timestamp(baseline_rows[0]["timestamp"])
    metadata = _load_mems_manifest_metadata(root, normalized_name)

    return MEMSNavTrial(
        name=normalized_name,
        root=root,
        input_path=input_path,
        baseline_path=baseline_path,
        imu_samples=_parse_mems_imu_samples(input_rows, input_epoch),
        gnss_samples=_parse_mems_gnss_samples(input_rows, input_epoch),
        reference_states=_parse_mems_reference_states(baseline_rows, baseline_epoch),
        metadata=metadata,
    )


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

        samples.append(
            GNSSSample(
                time_s=_relative_time_s(row["time"], epoch),
                latitude_rad=np.deg2rad(latitude_deg),
                longitude_rad=np.deg2rad(longitude_deg),
                altitude_m=altitude_m,
                velocity_ned_mps=None,
            )
        )
    return samples


def _parse_mems_reference_states(rows: list[dict[str, str]], epoch: datetime) -> list[MEMSReferenceState]:
    states: list[MEMSReferenceState] = []
    for row in rows:
        covariance = {
            key: float(value)
            for key, value in row.items()
            if key.endswith("_cov") and value != ""
        }
        states.append(
            MEMSReferenceState(
                time_s=_relative_time_s(row["timestamp"], epoch),
                latitude_rad=np.deg2rad(float(row["latitude"])),
                longitude_rad=np.deg2rad(float(row["longitude"])),
                altitude_m=float(row["altitude"]),
                velocity_ned_mps=np.array(
                    [
                        float(row["velocity_north"]),
                        float(row["velocity_east"]),
                        float(row["velocity_down"]),
                    ],
                    dtype=np.float64,
                ),
                attitude_rpy_rad=np.array(
                    [float(row["roll"]), float(row["pitch"]), float(row["yaw"])],
                    dtype=np.float64,
                ),
                accel_bias_mps2=np.array(
                    [float(row["acc_bias_x"]), float(row["acc_bias_y"]), float(row["acc_bias_z"])],
                    dtype=np.float64,
                ),
                gyro_bias_rps=np.array(
                    [float(row["gyro_bias_x"]), float(row["gyro_bias_y"]), float(row["gyro_bias_z"])],
                    dtype=np.float64,
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

    return {
        "source": manifest.get("repo_url"),
        "ref": manifest.get("ref"),
        "files": trial_files,
    }
