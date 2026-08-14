"""Loaders for the Victoria Park MATLAB dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .types import VictoriaParkDataset


def load_victoria_park_data(root: str | Path) -> VictoriaParkDataset:
    """Load the Victoria Park laser, odometry, and optional GPS files."""
    root = Path(root)
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

