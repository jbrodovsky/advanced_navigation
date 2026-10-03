"""Part B: the AprilTag perception and measurement model.

The three functions here are Part B deliverables. They raise
``NotImplementedError`` until you write them; the tests that depend on them
skip with a message until then. Keep the signatures: Part C, ``run_all`` and
the tests call them exactly as written.

Conventions (see the Project 1 handout and the PnP chapter):

* World frame ``w``: the tag mat, ``x`` down the mat, ``y`` to the right,
  ``z`` up. The origin is the top-left corner of tag 0 (row 0, column 0).
* Corner order in the packets: ``p1`` bottom-left, ``p2`` bottom-right,
  ``p3`` top-right, ``p4`` top-left, where "top" and "bottom" refer to the mat
  drawing (bottom corners have the larger ``x``).
* Attitude: Z-X-Y roll, pitch, yaw, ``C_world_body = Rz(yaw) @ Rx(roll) @ Ry(pitch)``
  (``utils.rpy_to_rotation_matrix`` / ``utils.rotation_matrix_to_rpy``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .types import DronePacket, PoseObservation, TruthSample


def build_apriltag_corner_map() -> dict[int, np.ndarray]:
    """Return the world-frame corners of every tag on the mat.

    Returns
    -------
    dict[int, np.ndarray]
        Maps each tag ID (0-107) to a ``(4, 3)`` array of corner coordinates
        in metres, rows in the packet order ``p1, p2, p3, p4`` (bottom-left,
        bottom-right, top-right, top-left) and ``z = 0``.

    Notes
    -----
    The mat is a 12 x 9 grid (``config.TAG_IDS``; row index along +x, column
    index along +y). Each tag is ``config.TAG_SIZE_M`` on a side. Neighbouring
    tags are separated by ``config.NORMAL_GAP_M``, except between columns 3-4
    and 6-7 (one-based), where the gap is ``config.WIDE_GAP_M``.
    """
    raise NotImplementedError(
        "Part B: implement build_apriltag_corner_map() from config.TAG_IDS, "
        "TAG_SIZE_M, NORMAL_GAP_M and WIDE_GAP_M (see the Part B handout)."
    )


def estimate_pose(
    packet: DronePacket,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    tag_map: dict[int, np.ndarray],
    T_camera_body: np.ndarray,
) -> PoseObservation | None:
    """Estimate the body pose in the world frame from one packet's detections.

    Parameters
    ----------
    packet:
        A :class:`~drone_pipeline.types.DronePacket`. Read ``packet.id`` and the
        corner arrays ``packet.p1`` .. ``packet.p4`` (each ``(2, n_tags)``,
        rows ``u`` and ``v``), plus ``packet.t`` for the timestamp.
    camera_matrix:
        3 x 3 intrinsic matrix ``K`` (``DroneParameters.camera_matrix``).
    dist_coeffs:
        Distortion coefficients handed to ``cv2.solvePnP``. The corner
        detections are already undistorted, so pass ``np.zeros(5)``; the
        Part B handout asks you to check this claim via reprojection error.
    tag_map:
        Output of :func:`build_apriltag_corner_map`.
    T_camera_body:
        4 x 4 transform mapping body coordinates into camera coordinates
        (``DroneParameters.T_camera_body``). With OpenCV's world-to-camera
        result ``T_camera_world``, the body pose is
        ``T_world_body = inv(T_camera_world) @ T_camera_body``.

    Returns
    -------
    PoseObservation or None
        ``position`` (body origin in the world frame, m), ``rpy`` (Z-X-Y,
        rad), the RMS reprojection error in pixels, and the number of corner
        correspondences used. Return ``None`` when fewer than 4
        correspondences are available (for example, no tag in view) or when
        the solver fails.
    """
    raise NotImplementedError(
        "Part B: implement estimate_pose(...) with build_apriltag_corner_map(), "
        "cv2.solvePnP (zero distortion), and T_camera_body (see the Part B handout)."
    )


def estimate_measurement_covariance(
    observations: Sequence[PoseObservation],
    truth_samples: Sequence[TruthSample],
) -> np.ndarray:
    """Estimate the 6 x 6 PnP measurement covariance from truth residuals.

    Parameters
    ----------
    observations:
        PnP pose observations.
    truth_samples:
        Vicon truth at the same timestamps, one per observation (use
        :func:`~drone_pipeline.datasets.interpolate_vicon_truth` with the
        observation times). Drop pairs whose truth sample has
        ``valid == False`` before calling.

    Returns
    -------
    np.ndarray
        The sample covariance (about the mean) of the residuals
        ``z_pnp - z_truth`` with ``z = [px, py, pz, roll, pitch, yaw]``; angle
        residuals wrapped into [-pi, pi). Report the residual mean (the bias)
        separately: a covariance estimate says nothing about bias.

    Raises
    ------
    ValueError
        If the two sequences differ in length, contain fewer than two pairs,
        or contain an invalid truth sample.
    """
    raise NotImplementedError(
        "Part B: implement estimate_measurement_covariance(...) from wrapped "
        "pose residuals against Vicon (see the Part B handout)."
    )
