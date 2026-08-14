"""Observation-model helpers and starter interfaces."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .config import NORMAL_GAP_M, TAG_IDS, TAG_SIZE_M, WIDE_GAP_M
from .types import PoseObservation, TruthSample
from .utils import wrap_angles


def _column_origin_m(column_index: int) -> float:
    position_m = 0.0
    for col in range(column_index):
        position_m += TAG_SIZE_M
        position_m += WIDE_GAP_M if col in {2, 5} else NORMAL_GAP_M
    return position_m


def build_apriltag_corner_map() -> dict[int, np.ndarray]:
    """Construct the deterministic world-frame corner map for the AprilTag mat."""
    tag_map: dict[int, np.ndarray] = {}
    for row_idx in range(TAG_IDS.shape[0]):
        x0 = row_idx * (TAG_SIZE_M + NORMAL_GAP_M)
        x1 = x0 + TAG_SIZE_M
        for col_idx in range(TAG_IDS.shape[1]):
            y0 = _column_origin_m(col_idx)
            y1 = y0 + TAG_SIZE_M
            tag_map[int(TAG_IDS[row_idx, col_idx])] = np.array(
                [
                    [x1, y0, 0.0],
                    [x1, y1, 0.0],
                    [x0, y1, 0.0],
                    [x0, y0, 0.0],
                ],
                dtype=np.float64,
            )
    return tag_map


def estimate_pose(packet, camera_matrix, dist_coeffs, tag_map, T_camera_body):
    """Starter interface for the AprilTag PnP solver."""
    raise NotImplementedError(
        "students should implement estimate_pose(...) using build_apriltag_corner_map(), "
        "OpenCV solvePnP, and the provided camera/body transform."
    )


def estimate_measurement_covariance(
    observations: Sequence[PoseObservation | dict[str, object]],
    truth_samples: Sequence[TruthSample],
) -> np.ndarray:
    """Estimate a 6x6 covariance matrix from pose residuals."""
    if len(observations) != len(truth_samples):
        raise ValueError("observations and truth_samples must have the same length")
    if len(observations) < 2:
        raise ValueError("at least two matched samples are required to estimate covariance")

    residual_rows = []
    for observation, truth in zip(observations, truth_samples, strict=True):
        if isinstance(observation, dict):
            position = np.asarray(observation["position"], dtype=np.float64)
            rpy = np.asarray(observation["rpy"], dtype=np.float64)
        else:
            position = np.asarray(observation.position, dtype=np.float64)
            rpy = np.asarray(observation.rpy, dtype=np.float64)
        residual = np.concatenate([position - truth.position, wrap_angles(rpy - truth.rpy)])
        residual_rows.append(residual)

    residual_matrix = np.vstack(residual_rows)
    return np.cov(residual_matrix, rowvar=False)

