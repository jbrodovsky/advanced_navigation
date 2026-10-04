"""Part B: the loosely coupled INS/GNSS error-state Kalman filter.

The functions marked "yours" are Project 3 deliverables. :class:`INSGNSSFilter`
(provided) is the bookkeeping that strings them together: it propagates the
nominal state with :func:`~ins_gnss.strapdown.mechanize` and the covariance
with your ``F`` and ``Q_d``, and for every GNSS fix computes your residual,
runs your Joseph-form update, injects the correction, resets the covariance,
and records the NIS.

Error state (15), true minus estimate (@sec-ins-gnss-error-state-formulation)::

    dx = [dp (NED m), dv (NED m/s), dtheta (nav-frame rad), db_a (m/s^2), db_g (rad/s)]
    dp_N = (R_M + h) dL,  dp_E = (R_N + h) cos(L) dlambda,  dp_D = -dh
    C_b^n = (I + [dtheta]_x) C_hat_b^n,   q_b^n = dq(dtheta) (x) q_hat_b^n

Short-horizon MEMS error dynamics (Earth-rate, transport-rate and gravity-
gradient couplings neglected; see the chapter for why that is fine for a
smartphone IMU aided once a second)::

    dp_dot     = dv
    dv_dot     = -[C_hat f_hat]_x dtheta - C_hat db_a + C_hat w_a
    dtheta_dot = -C_hat db_g + C_hat w_g
    db_a_dot   = w_ba,    db_g_dot = w_bg
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import strapdown
from .types import GNSSSample, IMUNoise, NavState

STATE_DIM = 15
POS = slice(0, 3)
VEL = slice(3, 6)
ATT = slice(6, 9)
ACC_BIAS = slice(9, 12)
GYRO_BIAS = slice(12, 15)

HORIZONTAL_68_TO_SIGMA = 1.0 / 1.515
"""Android's horizontal accuracy is a 68% radius; for a circular 2-D Gaussian that is 1.515 sigma per axis."""


# ---------------------------------------------------------------------------
# Student deliverables
# ---------------------------------------------------------------------------


def error_state_jacobian(state: NavState, specific_force_b: np.ndarray) -> np.ndarray:
    """Continuous-time error-state matrix ``F`` (15 x 15) at ``state`` (Part B, yours).

    ``specific_force_b`` is the bias-corrected specific force ``f_hat^b``.
    Fill in the blocks of the module docstring's model: ``dp`` from ``dv``,
    ``dv`` from ``dtheta`` (``-[C_hat f_hat]_x``) and from ``db_a``
    (``-C_hat``), ``dtheta`` from ``db_g`` (``-C_hat``); zero elsewhere.
    The tests compare it with a numerical Jacobian of your own
    :func:`~ins_gnss.strapdown.mechanize`.
    """
    raise NotImplementedError("Part B: implement error_state_jacobian(state, specific_force_b).")


def discrete_process_noise(state: NavState, noise: IMUNoise, dt_s: float) -> np.ndarray:
    """Discrete process noise ``Q_d`` (15 x 15) over one step of ``dt_s`` (Part B, yours).

    ``Q_d ~ G Q_c G^T dt`` with ``Q_c = diag(noise.psd())`` for the noise
    vector ``[w_a, w_g, w_ba, w_bg]`` and ``G`` (15 x 12) mapping it into
    ``(dv, dtheta, db_a, db_g)`` (``C_hat`` for the first two, identity for
    the bias random walks). Van Loan's method (@sec-kf-process-noise) is a
    worthwhile refinement. Must be symmetric positive semidefinite.
    """
    raise NotImplementedError("Part B: implement discrete_process_noise(state, noise, dt_s).")


def gnss_position_residual(
    state: NavState, gnss: GNSSSample, lever_arm_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Loosely coupled position residual ``nu`` (3,) and Jacobian ``H`` (3 x 15) (Part B, yours).

    The antenna sits at ``lever_arm_b`` (body axes, meters) from the IMU. Express
    the fix in NED meters relative to the nominal IMU position
    (:func:`~ins_gnss.earth.geodetic_to_ned_delta`), predict it as
    ``C_hat_b^n l^b``, and return ``nu = z - z_hat`` with
    ``H = [I_3, 0, -[C_hat_b^n l^b]_x, 0, 0]``
    (@sec-ins-gnss-loose-gnss-measurement, @sec-ins-gnss-lever-arm-timing).
    """
    raise NotImplementedError("Part B: implement gnss_position_residual(state, gnss, lever_arm_b).")


def joseph_update(
    P: np.ndarray, nu: np.ndarray, H: np.ndarray, R: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Kalman update of the error state; return ``(delta_x, P_plus, nis)`` (Part B, yours).

    ``S = H P H^T + R``; ``K = P H^T S^-1`` (with ``np.linalg.solve``, never
    ``inv``); ``delta_x = K nu`` (the prior error-state mean is zero);
    Joseph form ``P_plus = (I - K H) P (I - K H)^T + K R K^T``; and the
    normalized innovation squared ``nis = nu^T S^-1 nu``.
    """
    raise NotImplementedError("Part B: implement joseph_update(P, nu, H, R).")


def inject_error_state(state: NavState, delta_x: np.ndarray) -> NavState:
    """Apply an estimated error to the nominal state; return a new state (Part B, yours).

    Position: convert ``dp`` back to ``(dL, dlambda, dh)``
    (:func:`~ins_gnss.earth.ned_to_geodetic_delta`) and add. Velocity and
    biases: add. Attitude: ``q <- dq(dtheta) (x) q``, then normalize.
    """
    raise NotImplementedError("Part B: implement inject_error_state(state, delta_x).")


def reset_covariance(P: np.ndarray, delta_theta: np.ndarray) -> np.ndarray:
    """Covariance after the error-state reset, ``G P G^T`` (Part B, yours).

    ``G`` is the identity except for the attitude block
    ``I + [delta_theta / 2]_x`` (navigation-frame error; @eq-ekf-reset-jacobian).
    Return a symmetric matrix.
    """
    raise NotImplementedError("Part B: implement reset_covariance(P, delta_theta).")


def gnss_velocity_residual(
    state: NavState, gnss: GNSSSample, lever_arm_b: np.ndarray, gyro_rps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Extension: horizontal GNSS velocity residual ``nu`` (2,) and ``H`` (2 x 15).

    ``z = [v_N, v_E]`` from speed and course (:meth:`GNSSSample.velocity_ne_mps`);
    the antenna moves at ``v + C_hat (omega_ib^b x l^b)`` (Earth-rate term
    neglected). With a lever arm, ``H`` has velocity, attitude, and gyro-bias
    columns (the corrected rate ``omega_ib^b = gyro - b_g`` depends on the bias).
    Only needed for the velocity-aiding extension.
    """
    raise NotImplementedError("Extension: implement gnss_velocity_residual(state, gnss, lever_arm_b, gyro_rps).")


# ---------------------------------------------------------------------------
# Provided helpers and bookkeeping
# ---------------------------------------------------------------------------


def gnss_position_covariance(
    gnss: GNSSSample,
    *,
    scale: float = 1.0,
    floor_m: float = 1.0,
    default_horizontal_m: float = 5.0,
    default_vertical_m: float = 8.0,
) -> np.ndarray:
    """A starting ``R`` (3 x 3, NED, m^2) from the receiver's own accuracy estimates.

    Horizontal sigma per axis = ``horizontal_accuracy_m / 1.515`` (Android
    reports a 68% radius), vertical sigma = ``vertical_accuracy_m``, each
    multiplied by ``scale`` and floored at ``floor_m``. Missing accuracies
    fall back to the defaults. The receiver's numbers are optimistic and its
    errors are correlated in time; tune ``scale`` with the NIS.
    """
    horizontal = gnss.horizontal_accuracy_m
    vertical = gnss.vertical_accuracy_m
    sigma_h = default_horizontal_m if not horizontal or not np.isfinite(horizontal) else horizontal * HORIZONTAL_68_TO_SIGMA
    sigma_v = default_vertical_m if not vertical or not np.isfinite(vertical) else vertical
    sigma_h = max(scale * sigma_h, floor_m)
    sigma_v = max(scale * sigma_v, floor_m)
    return np.diag([sigma_h**2, sigma_h**2, sigma_v**2])


def gnss_velocity_covariance(
    gnss: GNSSSample,
    *,
    scale: float = 1.0,
    floor_mps: float = 0.1,
    default_speed_sigma_mps: float = 0.5,
    default_course_sigma_rad: float = float(np.deg2rad(2.0)),
) -> np.ndarray:
    """A starting ``R`` (2 x 2, m^2/s^2) for :func:`gnss_velocity_residual`.

    Propagates the receiver's speed and course accuracies through
    ``[v_N, v_E] = s [cos chi, sin chi]``: ``R = J diag(sigma_s^2, sigma_chi^2) J^T``
    with ``J = [[cos chi, -s sin chi], [sin chi, s cos chi]]``
    (@sec-ins-gnss-loose-gnss-measurement), times ``scale^2``, plus
    ``floor_mps^2`` on the diagonal.
    """
    speed = float(gnss.speed_mps or 0.0)
    course = float(gnss.bearing_rad or 0.0)
    sigma_s = gnss.speed_accuracy_mps if gnss.speed_accuracy_mps and np.isfinite(gnss.speed_accuracy_mps) else default_speed_sigma_mps
    sigma_c = (
        gnss.bearing_accuracy_rad
        if gnss.bearing_accuracy_rad and np.isfinite(gnss.bearing_accuracy_rad)
        else default_course_sigma_rad
    )
    J = np.array([[np.cos(course), -speed * np.sin(course)], [np.sin(course), speed * np.cos(course)]])
    R = scale**2 * J @ np.diag([sigma_s**2, sigma_c**2]) @ J.T
    return R + floor_mps**2 * np.eye(2)


@dataclass
class InnovationRecord:
    """One measurement update, for consistency checks (@sec-kf-consistency)."""

    time_s: float
    kind: str  # "gnss_position", "gnss_velocity", ...
    nis: float
    dof: int
    accepted: bool
    residual: np.ndarray


@dataclass
class INSGNSSFilter:
    """Loosely coupled INS/GNSS ESKF (provided bookkeeping around your functions).

    Attributes
    ----------
    state:
        Nominal :class:`~ins_gnss.types.NavState`.
    P:
        15 x 15 error-state covariance, order ``[dp, dv, dtheta, db_a, db_g]``.
    noise:
        :class:`~ins_gnss.types.IMUNoise` used by :func:`discrete_process_noise`.
    lever_arm_b:
        GNSS antenna position relative to the IMU, body axes [m].
    time_s:
        Time of ``state``.
    delta_x:
        The error-state mean. It is zero between updates because every
        correction is injected and reset; ``last_correction`` keeps the most
        recent ``delta_x`` before its reset.
    innovations:
        One :class:`InnovationRecord` per update (accepted or gated out).

    The class is a plain dataclass, so add attributes freely (a history of
    ``P`` diagonals, an outage flag, ...).
    """

    state: NavState
    P: np.ndarray
    noise: IMUNoise = field(default_factory=IMUNoise)
    lever_arm_b: np.ndarray = field(default_factory=lambda: np.zeros(3))
    time_s: float = 0.0
    delta_x: np.ndarray = field(default_factory=lambda: np.zeros(STATE_DIM))
    last_correction: np.ndarray = field(default_factory=lambda: np.zeros(STATE_DIM))
    innovations: list[InnovationRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.P = np.array(self.P, dtype=np.float64).reshape(STATE_DIM, STATE_DIM)
        self.lever_arm_b = np.array(self.lever_arm_b, dtype=np.float64).reshape(3)

    def predict(self, accel_mps2: np.ndarray, gyro_rps: np.ndarray, dt_s: float) -> None:
        """Propagate the nominal state and the covariance over ``dt_s``.

        ``F`` and ``Q_d`` are evaluated at the state at the start of the
        interval, and ``P <- F_d P F_d^T + Q_d`` with ``F_d = I + F dt``.
        """
        if dt_s <= 0.0:
            raise ValueError(f"dt_s must be positive, got {dt_s}")
        accel = np.asarray(accel_mps2, dtype=np.float64)
        gyro = np.asarray(gyro_rps, dtype=np.float64)
        F = error_state_jacobian(self.state, accel - self.state.accel_bias_mps2)
        Q_d = discrete_process_noise(self.state, self.noise, dt_s)
        self.state = strapdown.mechanize(self.state, accel, gyro, dt_s)
        F_d = np.eye(STATE_DIM) + F * dt_s
        self.P = F_d @ self.P @ F_d.T + Q_d
        self.P = 0.5 * (self.P + self.P.T)
        self.time_s += dt_s

    def _apply(self, nu: np.ndarray, H: np.ndarray, R: np.ndarray, kind: str, gate: float | None) -> float:
        delta_x, P_plus, nis = joseph_update(self.P, nu, H, R)
        accepted = gate is None or nis <= gate
        self.innovations.append(InnovationRecord(self.time_s, kind, float(nis), int(len(nu)), accepted, np.array(nu)))
        if not accepted:
            return float(nis)
        self.delta_x = np.asarray(delta_x, dtype=np.float64)
        self.state = inject_error_state(self.state, self.delta_x)
        self.P = reset_covariance(P_plus, self.delta_x[ATT])
        self.last_correction = self.delta_x
        self.delta_x = np.zeros(STATE_DIM)
        return float(nis)

    def update_gnss_position(self, gnss: GNSSSample, R: np.ndarray | None = None, *, gate: float | None = None) -> float:
        """Fuse one GNSS position fix; return its NIS.

        ``R`` defaults to :func:`gnss_position_covariance`. With ``gate`` set
        (e.g. the 99.9% chi-square point for 3 dof, 16.27), fixes whose NIS
        exceeds it are recorded and skipped.
        """
        R = gnss_position_covariance(gnss) if R is None else np.asarray(R, dtype=np.float64)
        nu, H = gnss_position_residual(self.state, gnss, self.lever_arm_b)
        return self._apply(np.asarray(nu, dtype=np.float64), np.asarray(H, dtype=np.float64), R, "gnss_position", gate)

    def update_gnss_velocity(
        self, gnss: GNSSSample, gyro_rps: np.ndarray, R: np.ndarray | None = None, *, gate: float | None = None
    ) -> float:
        """Extension: fuse the horizontal GNSS velocity (speed and course); return its NIS."""
        R = gnss_velocity_covariance(gnss) if R is None else np.asarray(R, dtype=np.float64)
        nu, H = gnss_velocity_residual(self.state, gnss, self.lever_arm_b, gyro_rps)
        return self._apply(np.asarray(nu, dtype=np.float64), np.asarray(H, dtype=np.float64), R, "gnss_velocity", gate)

    def position_sigma_ned(self) -> np.ndarray:
        """One-sigma position uncertainty [m] in north, east, down."""
        return np.sqrt(np.clip(np.diag(self.P)[POS], 0.0, None))

    def nis_values(self, kind: str = "gnss_position") -> np.ndarray:
        """NIS of every update of ``kind`` (accepted or not), in order."""
        return np.array([record.nis for record in self.innovations if record.kind == kind])
