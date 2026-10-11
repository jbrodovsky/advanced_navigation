"""Reference solution for Project 3 Part B (instructor only).

The 15-state loosely coupled ESKF pieces of ``ins_gnss.eskf``: the
short-horizon MEMS error model ``F``, ``Q_d = G Q_c G^T dt``, the lever-arm
position residual, the Joseph-form update, injection, and the
navigation-frame reset Jacobian. Plus the horizontal-velocity residual used by
the velocity-aiding extension.

Every sign here is checked numerically by the student tests when run through
``check_solutions.py`` (``F`` against a finite-difference Jacobian of the
mechanization, ``H`` against finite differences of the residual, the reset
Jacobian against the exact attitude reset).
"""

from __future__ import annotations

import numpy as np

from ins_gnss.earth import geodetic_to_ned_delta, ned_to_geodetic_delta
from ins_gnss.eskf import ACC_BIAS, ATT, GYRO_BIAS, POS, STATE_DIM, VEL
from ins_gnss.rotations import quat_multiply, quat_normalize, quat_to_dcm, rotvec_to_quat, skew
from ins_gnss.types import GNSSSample, IMUNoise, NavState


def error_state_jacobian(state: NavState, specific_force_b: np.ndarray) -> np.ndarray:
    C = quat_to_dcm(state.q_bn)
    f_n = C @ np.asarray(specific_force_b, dtype=np.float64)
    F = np.zeros((STATE_DIM, STATE_DIM))
    F[POS, VEL] = np.eye(3)
    F[VEL, ATT] = -skew(f_n)
    F[VEL, ACC_BIAS] = -C
    F[ATT, GYRO_BIAS] = -C
    return F


def noise_input_matrix(state: NavState) -> np.ndarray:
    """``G`` (15 x 12) for the noise vector ``[w_a, w_g, w_ba, w_bg]``."""
    C = quat_to_dcm(state.q_bn)
    G = np.zeros((STATE_DIM, 12))
    G[VEL, 0:3] = C
    G[ATT, 3:6] = C
    G[ACC_BIAS, 6:9] = np.eye(3)
    G[GYRO_BIAS, 9:12] = np.eye(3)
    return G


def discrete_process_noise(state: NavState, noise: IMUNoise, dt_s: float) -> np.ndarray:
    G = noise_input_matrix(state)
    Q_d = G @ np.diag(noise.psd()) @ G.T * dt_s
    return 0.5 * (Q_d + Q_d.T)


def gnss_position_residual(state: NavState, gnss: GNSSSample, lever_arm_b: np.ndarray):
    C = quat_to_dcm(state.q_bn)
    lever_n = C @ np.asarray(lever_arm_b, dtype=np.float64)
    z = geodetic_to_ned_delta(
        gnss.latitude_rad, gnss.longitude_rad, gnss.altitude_m, state.latitude_rad, state.longitude_rad, state.height_m
    )
    nu = z - lever_n
    H = np.zeros((3, STATE_DIM))
    H[:, POS] = np.eye(3)
    H[:, ATT] = -skew(lever_n)
    return nu, H


def gnss_velocity_residual(state: NavState, gnss: GNSSSample, lever_arm_b: np.ndarray, gyro_rps: np.ndarray):
    z = gnss.velocity_ne_mps()
    if z is None:
        raise ValueError("this fix has no speed/course")
    C = quat_to_dcm(state.q_bn)
    w_ib_b = np.asarray(gyro_rps, dtype=np.float64) - state.gyro_bias_rps
    lever_velocity_n = C @ np.cross(w_ib_b, np.asarray(lever_arm_b, dtype=np.float64))
    nu = z - (state.velocity_ned_mps + lever_velocity_n)[:2]
    H = np.zeros((2, STATE_DIM))
    H[:, VEL] = np.eye(3)[:2]
    H[:, ATT] = -skew(lever_velocity_n)[:2]
    # The lever-arm velocity uses the bias-corrected gyro rate, so it depends on
    # the gyro bias: d(C (w x l))/d(db_g) = C [l]_x.
    H[:, GYRO_BIAS] = (C @ skew(np.asarray(lever_arm_b, dtype=np.float64)))[:2]
    return nu, H


def joseph_update(P: np.ndarray, nu: np.ndarray, H: np.ndarray, R: np.ndarray):
    P = np.asarray(P, dtype=np.float64)
    nu = np.asarray(nu, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    S = H @ P @ H.T + R
    S = 0.5 * (S + S.T)
    K = np.linalg.solve(S, H @ P).T  # P H^T S^-1 (S and P symmetric)
    delta_x = K @ nu
    I_KH = np.eye(P.shape[0]) - K @ H
    P_plus = I_KH @ P @ I_KH.T + K @ R @ K.T
    nis = float(nu @ np.linalg.solve(S, nu))
    return delta_x, 0.5 * (P_plus + P_plus.T), nis


def inject_error_state(state: NavState, delta_x: np.ndarray) -> NavState:
    delta_x = np.asarray(delta_x, dtype=np.float64)
    d_lat, d_lon, d_h = ned_to_geodetic_delta(delta_x[POS], state.latitude_rad, state.height_m)
    q = quat_normalize(quat_multiply(rotvec_to_quat(delta_x[ATT]), state.q_bn))
    return NavState(
        state.latitude_rad + d_lat,
        state.longitude_rad + d_lon,
        state.height_m + d_h,
        state.velocity_ned_mps + delta_x[VEL],
        q,
        state.accel_bias_mps2 + delta_x[ACC_BIAS],
        state.gyro_bias_rps + delta_x[GYRO_BIAS],
    )


def reset_covariance(P: np.ndarray, delta_theta: np.ndarray) -> np.ndarray:
    G = np.eye(STATE_DIM)
    G[ATT, ATT] = np.eye(3) + skew(0.5 * np.asarray(delta_theta, dtype=np.float64))
    P_reset = G @ np.asarray(P, dtype=np.float64) @ G.T
    return 0.5 * (P_reset + P_reset.T)
