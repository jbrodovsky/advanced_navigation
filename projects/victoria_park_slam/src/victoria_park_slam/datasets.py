"""Loaders for the Victoria Park MATLAB dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .types import VictoriaParkDataset


DEFAULT_VICTORIA_PARK_ROOT = Path("data/victoria-park")
REQUIRED_MATLAB_FILES = ("aa3_lsr2.mat", "aa3_dr.mat", "aa3_gpsx.mat")


def resolve_victoria_park_root(root: str | Path = DEFAULT_VICTORIA_PARK_ROOT) -> Path:
    """Resolve the directory that contains the original Victoria Park `.mat` files.

    The course repository stores the MATLAB files directly under
    `data/victoria-park`. Legacy paths such as `dataset_victoria_park` and
    `dataset_victoria_park/original_MATLAB_dataset` are still accepted when
    passed explicitly.
    """

    candidate = Path(root).expanduser()
    if not candidate.is_absolute():
        for parent in [Path.cwd(), *Path.cwd().parents]:
            repo_relative = parent / candidate
            if repo_relative.exists():
                candidate = repo_relative
                break

    if candidate.name == "dataset_victoria_park":
        candidate = candidate / "original_MATLAB_dataset"
    elif not candidate.exists():
        for parent in [Path.cwd(), *Path.cwd().parents]:
            default_root = parent / DEFAULT_VICTORIA_PARK_ROOT
            if default_root.exists():
                candidate = default_root
                break

    missing = [name for name in REQUIRED_MATLAB_FILES if not (candidate / name).exists()]
    if missing:
        archive_hint = f" Expected the active course data at `{DEFAULT_VICTORIA_PARK_ROOT}`."
        for parent in [candidate, *candidate.parents]:
            archive = parent / "dataset_victoria_park.tar.gz"
            if archive.exists():
                archive_hint = (
                    f" Found legacy archive {archive}; prefer `data/victoria-park`, "
                    "or extract the archive and pass the extracted MATLAB directory."
                )
                break

        raise FileNotFoundError(
            "Victoria Park MATLAB files are missing from "
            f"{candidate}. Missing: {', '.join(missing)}.{archive_hint}"
        )

    return candidate


def load_victoria_park_data(root: str | Path = DEFAULT_VICTORIA_PARK_ROOT) -> VictoriaParkDataset:
    """Load the Victoria Park laser, odometry, and optional GPS files."""
    root = resolve_victoria_park_root(root)
    laser = loadmat(root / "aa3_lsr2.mat", simplify_cells=True)
    odometry = loadmat(root / "aa3_dr.mat", simplify_cells=True)
    gps_path = root / "aa3_gpsx.mat"
    gps = loadmat(gps_path, simplify_cells=True) if gps_path.exists() else None

    laser_ranges_m = np.asarray(laser["LASER"], dtype=np.float64) / 100.0
    laser_time_s = np.asarray(laser["TLsr"], dtype=np.float64).reshape(-1) / 1000.0
    speed_mps = np.asarray(odometry["speed"], dtype=np.float64).reshape(-1)
    steering_rad = np.asarray(odometry["steering"], dtype=np.float64).reshape(-1)
    odometry_time_s = np.asarray(odometry["time"], dtype=np.float64).reshape(-1) / 1000.0

    if gps is None:
        gps_lat_m = None
        gps_lon_m = None
        gps_time_s = None
    else:
        gps_lat_m = np.asarray(gps["La_m"], dtype=np.float64).reshape(-1)
        gps_lon_m = np.asarray(gps["Lo_m"], dtype=np.float64).reshape(-1)
        gps_time_s = np.asarray(gps["timeGps"], dtype=np.float64).reshape(-1) / 1000.0

    return VictoriaParkDataset(
        root=root,
        laser_ranges_m=laser_ranges_m,
        laser_time_s=laser_time_s,
        speed_mps=speed_mps,
        steering_rad=steering_rad,
        odometry_time_s=odometry_time_s,
        gps_lat_m=gps_lat_m,
        gps_lon_m=gps_lon_m,
        gps_time_s=gps_time_s,
    )
