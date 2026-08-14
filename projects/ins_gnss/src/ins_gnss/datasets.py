"""Flexible loader for INS/GNSS project datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .types import GNSSSample, IMUSample, INSGNSSDataset


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


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

