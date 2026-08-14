"""Static configuration values for the drone project."""

from __future__ import annotations

import numpy as np

CAMERA_MATRIX = np.array(
    [
        [314.1779, 0.0, 199.4848],
        [0.0, 314.2218, 113.7838],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

DIST_COEFFS = np.array(
    [-0.438607, 0.248625, 0.00072, -0.000476, -0.0911],
    dtype=np.float64,
)

CAMERA_BODY_TRANSLATION_M = np.array([-0.04, 0.0, -0.03], dtype=np.float64)
CAMERA_BODY_YAW_RAD = np.pi / 4.0

TAG_SIZE_M = 0.152
NORMAL_GAP_M = 0.152
WIDE_GAP_M = 0.178

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

