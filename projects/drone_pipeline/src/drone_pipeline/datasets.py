"""Loaders and truth interpolation helpers for the drone project."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat

from .config import CAMERA_MATRIX, CAMERA_BODY_TRANSLATION_M, CAMERA_BODY_YAW_RAD, DIST_COEFFS, TAG_IDS
from .types import DroneDataset, DronePacket, DroneParameters, TruthSample
from .utils import Rx, Rz, interpolate_rows, make_transform


def _as_float64_array(value: object) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _packet_array(value: object) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 0:
        return array.astype(np.float64 if array.dtype.kind == "f" else np.int64, copy=False)
    return np.atleast_1d(array)


def load_drone_dataset(path: str | Path) -> DroneDataset:
    """Load one quadrotor dataset and normalize packets into typed records."""
    path = Path(path)
    raw = loadmat(path, simplify_cells=True)
    packets: list[DronePacket] = []
    for packet in raw["data"]:
        packets.append(
            DronePacket(
                img=np.asarray(packet["img"]),
                id=np.atleast_1d(np.asarray(packet["id"], dtype=np.int64)),
                p1=_packet_array(packet["p1"]),
                p2=_packet_array(packet["p2"]),
                p3=_packet_array(packet["p3"]),
                p4=_packet_array(packet["p4"]),
                t=float(packet["t"]),
                rpy=_as_float64_array(packet["rpy"]),
                drpy=_as_float64_array(packet["drpy"]),
                acc=_as_float64_array(packet["acc"]),
            )
        )
    return DroneDataset(
        path=path,
        vicon=np.asarray(raw["vicon"], dtype=np.float64),
        time=np.asarray(raw["time"], dtype=np.float64).reshape(-1),
        packets=packets,
    )


def load_drone_parameters(path: str | Path) -> DroneParameters:
    """Parse the supplied parameter file and expose standard camera/rig values."""
    text = Path(path).read_text()
    if "Camera Matrix" not in text or "Tag ids" not in text:
        raise ValueError(f"parameter file did not match the expected drone format: {path}")

    yaw_match = re.search(r"Yaw\s*=\s*pi/4", text)
    if yaw_match is None:
        raise ValueError("expected yaw offset definition was not found in parameters.txt")

    rotation = Rz(-CAMERA_BODY_YAW_RAD) @ Rx(-np.pi)
    transform = make_transform(rotation, CAMERA_BODY_TRANSLATION_M)
    return DroneParameters(
        camera_matrix=CAMERA_MATRIX.copy(),
        dist_coeffs=DIST_COEFFS.copy(),
        translation_m=CAMERA_BODY_TRANSLATION_M.copy(),
        yaw_offset_rad=float(CAMERA_BODY_YAW_RAD),
        tag_ids=TAG_IDS.copy(),
        T_camera_body=transform,
    )


def interpolate_vicon_truth(dataset: DroneDataset, packet_times: np.ndarray | None = None) -> list[TruthSample]:
    """Interpolate Vicon truth onto packet or user-specified timestamps."""
    if packet_times is None:
        packet_times = dataset.packet_times
    interpolated = interpolate_rows(dataset.time, dataset.vicon, np.asarray(packet_times, dtype=np.float64))
    truth_samples: list[TruthSample] = []
    for index, time_s in enumerate(np.asarray(packet_times, dtype=np.float64)):
        truth_samples.append(
            TruthSample(
                t=float(time_s),
                position=interpolated[0:3, index],
                rpy=interpolated[3:6, index],
                velocity=interpolated[6:9, index],
                angular_velocity=interpolated[9:12, index],
            )
        )
    return truth_samples

