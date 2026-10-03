"""Reference solution for Project 3 Part A (instructor only).

Curved-Earth strapdown mechanization in the local navigation frame (Groves
Section 5.4, book conventions) and stationary coarse alignment. These follow
the shared model in the milestone brief and ``ins-gnss.qmd`` exactly:
attitude with exact rotation-vector quaternions, specific force rotated with
the average of the old and new ``C_b^n``, Groves eq. 5.54 velocity update,
trapezoidal position update.
"""

from __future__ import annotations

import numpy as np

from ins_gnss.earth import earth_rate_ned, gravity_ned, radii_of_curvature, transport_rate_ned
from ins_gnss.rotations import euler_to_quat, quat_multiply, quat_normalize, quat_to_dcm, rotvec_to_quat, skew
from ins_gnss.types import NavState


def mechanize(state: NavState, accel_mps2: np.ndarray, gyro_rps: np.ndarray, dt_s: float) -> NavState:
    """One local-navigation-frame strapdown step (see ``ins_gnss.strapdown``)."""
    f_b = np.asarray(accel_mps2, dtype=np.float64) - state.accel_bias_mps2
    w_ib_b = np.asarray(gyro_rps, dtype=np.float64) - state.gyro_bias_rps
    lat, lon, h = state.latitude_rad, state.longitude_rad, state.height_m
    v_minus = state.velocity_ned_mps
    r_m, r_n = radii_of_curvature(lat)
    w_ie_n = earth_rate_ned(lat)
    w_en_n = transport_rate_ned(lat, h, v_minus)

    # Attitude: body rotates by omega_ib^b dt, the NED frame by omega_in^n dt.
    C_minus = quat_to_dcm(state.q_bn)
    q_plus = quat_multiply(quat_multiply(rotvec_to_quat(-(w_ie_n + w_en_n) * dt_s), state.q_bn), rotvec_to_quat(w_ib_b * dt_s))
    q_plus = quat_normalize(q_plus)
    C_plus = quat_to_dcm(q_plus)

    # Specific force in NED, then velocity (Groves eq. 5.54).
    f_n = 0.5 * (C_minus + C_plus) @ f_b
    v_plus = v_minus + (f_n + gravity_ned(lat, h) - skew(w_en_n + 2.0 * w_ie_n) @ v_minus) * dt_s

    # Position, trapezoidal (Groves eq. 5.56).
    h_plus = h - 0.5 * dt_s * (v_minus[2] + v_plus[2])
    lat_plus = lat + 0.5 * dt_s * (v_minus[0] / (r_m + h) + v_plus[0] / (r_m + h_plus))
    _, r_n_plus = radii_of_curvature(lat_plus)
    lon_plus = lon + 0.5 * dt_s * (
        v_minus[1] / ((r_n + h) * np.cos(lat)) + v_plus[1] / ((r_n_plus + h_plus) * np.cos(lat_plus))
    )

    return NavState(lat_plus, lon_plus, h_plus, v_plus, q_plus, state.accel_bias_mps2.copy(), state.gyro_bias_rps.copy())


def coarse_alignment(mean_specific_force_b: np.ndarray, yaw_rad: float) -> np.ndarray:
    """Leveling from the mean specific force (FRD), heading supplied."""
    f_x, f_y, f_z = np.asarray(mean_specific_force_b, dtype=np.float64)
    roll = np.arctan2(-f_y, -f_z)
    pitch = np.arctan2(f_x, np.hypot(f_y, f_z))
    return euler_to_quat(roll, pitch, yaw_rad)
