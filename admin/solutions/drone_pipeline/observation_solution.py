"""Reference solution for Project 1 Part B (instructor only).

Drop-in replacements for the stubs in ``drone_pipeline.observation``. The
corner-map and covariance code is the implementation that used to ship in the
starter package; ``estimate_pose`` follows the PnP chapter's skeleton.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from drone_pipeline.config import NORMAL_GAP_M, TAG_IDS, TAG_SIZE_M, WIDE_GAP_M
from drone_pipeline.types import DronePacket, PoseObservation, TruthSample
from drone_pipeline.utils import invert_transform, make_transform, rotation_matrix_to_rpy, wrap_angles

MIN_POINTS = 4


def _column_origin_m(column_index: int) -> float:
    position_m = 0.0
    for col in range(column_index):
        position_m += TAG_SIZE_M
        position_m += WIDE_GAP_M if col in {2, 5} else NORMAL_GAP_M
    return position_m


def build_apriltag_corner_map() -> dict[int, np.ndarray]:
    """World-frame corners (p1..p4 = BL, BR, TR, TL) for all 108 tags."""
    tag_map: dict[int, np.ndarray] = {}
    for row_idx in range(TAG_IDS.shape[0]):
        x0 = row_idx * (TAG_SIZE_M + NORMAL_GAP_M)
        x1 = x0 + TAG_SIZE_M
        for col_idx in range(TAG_IDS.shape[1]):
            y0 = _column_origin_m(col_idx)
            y1 = y0 + TAG_SIZE_M
            tag_map[int(TAG_IDS[row_idx, col_idx])] = np.array(
                [[x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0], [x0, y0, 0.0]],
                dtype=np.float64,
            )
    return tag_map


def assemble_correspondences(packet: DronePacket, tag_map: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Stack (N, 3) world points and (N, 2) pixels for every mapped tag in a packet."""
    ids = np.atleast_1d(np.asarray(packet.id)).astype(int)
    if ids.size == 0:
        return np.empty((0, 3)), np.empty((0, 2))
    corners = [np.asarray(getattr(packet, name), dtype=np.float64).reshape(2, -1) for name in ("p1", "p2", "p3", "p4")]
    object_points, image_points = [], []
    for column, tag_id in enumerate(ids):
        world = tag_map.get(int(tag_id))
        if world is None:
            continue
        object_points.append(world)
        image_points.append(np.stack([corner[:, column] for corner in corners]))
    if not object_points:
        return np.empty((0, 3)), np.empty((0, 2))
    return np.concatenate(object_points), np.concatenate(image_points)


def estimate_pose(
    packet: DronePacket,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    tag_map: dict[int, np.ndarray],
    T_camera_body: np.ndarray,
) -> PoseObservation | None:
    """Body pose in the world from one packet (None with fewer than 4 points)."""
    object_points, image_points = assemble_correspondences(packet, tag_map)
    if len(object_points) < MIN_POINTS:
        return None
    dist = np.asarray(dist_coeffs, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist)
    reprojection = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - image_points) ** 2, axis=1))))
    C_camera_world, _ = cv2.Rodrigues(rvec)
    T_camera_world = make_transform(C_camera_world, tvec.reshape(3))
    T_world_body = invert_transform(T_camera_world) @ T_camera_body
    return PoseObservation(
        t=float(packet.t),
        position=T_world_body[:3, 3].copy(),
        rpy=rotation_matrix_to_rpy(T_world_body[:3, :3]),
        reprojection_error=reprojection,
        num_points=int(len(object_points)),
    )


def pose_residuals(observations: Sequence[PoseObservation], truth_samples: Sequence[TruthSample]) -> np.ndarray:
    """(N, 6) residuals z_pnp - z_truth with wrapped angles."""
    if len(observations) != len(truth_samples):
        raise ValueError("observations and truth_samples must have the same length")
    if any(not getattr(sample, "valid", True) for sample in truth_samples):
        raise ValueError("drop observations whose truth sample is invalid (outside the Vicon span)")
    rows = []
    for obs, truth in zip(observations, truth_samples, strict=True):
        rows.append(
            np.concatenate(
                [
                    np.asarray(obs.position, dtype=np.float64) - truth.position,
                    wrap_angles(np.asarray(obs.rpy, dtype=np.float64) - truth.rpy),
                ]
            )
        )
    return np.vstack(rows) if rows else np.empty((0, 6))


def estimate_measurement_covariance(
    observations: Sequence[PoseObservation],
    truth_samples: Sequence[TruthSample],
) -> np.ndarray:
    """6 x 6 sample covariance of the residuals about their mean."""
    residuals = pose_residuals(observations, truth_samples)
    if len(residuals) < 2:
        raise ValueError("at least two matched samples are required to estimate covariance")
    return np.cov(residuals, rowvar=False)


def estimate_measurement_bias(
    observations: Sequence[PoseObservation],
    truth_samples: Sequence[TruthSample],
) -> np.ndarray:
    """Mean residual (6,): report it next to R."""
    return pose_residuals(observations, truth_samples).mean(axis=0)
