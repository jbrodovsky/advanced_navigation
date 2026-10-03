"""Documented default values for the drone project.

``load_drone_parameters()`` reads the camera matrix, distortion coefficients,
camera-IMU offset, yaw offset, and tag IDs from ``data/drone-data/parameters.txt``.
The constants below repeat what that file says so that code and tests can run
without the dataset (for example ``default_drone_parameters()``) and so the
values are visible in one place. If the file and these constants ever
disagree, the file wins.

The AprilTag geometry (tag size and gaps) is not in ``parameters.txt``; it
comes from the mat drawing in the handout.
"""

from __future__ import annotations

import numpy as np

# Pinhole intrinsics in pixels (parameters.txt, "Camera Matrix").
CAMERA_MATRIX = np.array(
    [
        [314.1779, 0.0, 199.4848],
        [0.0, 314.2218, 113.7838],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# Lens distortion [k1, k2, p1, p2, k3] (parameters.txt). The packet corner
# detections are already undistorted, so solvePnP should get zeros instead.
DIST_COEFFS = np.array(
    [-0.438607, 0.248625, 0.00072, -0.000476, -0.0911],
    dtype=np.float64,
)

# Camera-IMU calibration (parameters.txt, "XYZ" and "Yaw"). XYZ is the position
# of the body (IMU) origin in camera coordinates, i.e. the translation of
# T_camera_body; the camera sits about 3 cm below the IMU.
CAMERA_BODY_TRANSLATION_M = np.array([-0.04, 0.0, -0.03], dtype=np.float64)
CAMERA_BODY_YAW_RAD = np.pi / 4.0

# AprilTag mat geometry in metres (handout drawing).
TAG_SIZE_M = 0.152
NORMAL_GAP_M = 0.152
WIDE_GAP_M = 0.178  # between columns 3-4 and 6-7 (one-based)

# Tag IDs on the 12 x 9 mat (parameters.txt, "Tag ids"): row index runs down
# the mat (+x), column index to the right (+y).
TAG_IDS = np.array(
    [
        [0, 12, 24, 36, 48, 60, 72, 84, 96],
        [1, 13, 25, 37, 49, 61, 73, 85, 97],
        [2, 14, 26, 38, 50, 62, 74, 86, 98],
        [3, 15, 27, 39, 51, 63, 75, 87, 99],
        [4, 16, 28, 40, 52, 64, 76, 88, 100],
        [5, 17, 29, 41, 53, 65, 77, 89, 101],
        [6, 18, 30, 42, 54, 66, 78, 90, 102],
        [7, 19, 31, 43, 55, 67, 79, 91, 103],
        [8, 20, 32, 44, 56, 68, 80, 92, 104],
        [9, 21, 33, 45, 57, 69, 81, 93, 105],
        [10, 22, 34, 46, 58, 70, 82, 94, 106],
        [11, 23, 35, 47, 59, 71, 83, 95, 107],
    ],
    dtype=np.int64,
)

# Recommended split for Project 1 (see the handout). studentdata0 is excluded:
# its rate field and truth alignment are unresolved (GitHub issue #4).
DEVELOPMENT_DATASETS = (1, 2, 3, 4)
VALIDATION_DATASETS = (5, 6, 7)
