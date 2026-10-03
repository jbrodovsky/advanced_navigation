"""Part A: curved-Earth strapdown mechanization and coarse alignment.

The two functions marked "yours" are Project 3 deliverables; everything else
here is provided. The model is the one in @sec-ins-gnss-navigation-equations-local
(Groves, Section 5.4), in the book's conventions:

* state ``[L, lambda, h, v^n, q_b^n, b_a, b_g]`` (:class:`~ins_gnss.types.NavState`);
* NED navigation frame; body frame forward-right-down (after the mount
  rotation, see :func:`~ins_gnss.datasets.estimate_mount_rotation`);
* corrected measurements ``f^b = f_meas - b_a`` and
  ``omega_ib^b = omega_meas - b_g``.

One step of length ``tau`` (attitude, then velocity, then position)::

    q(+)   = q(-omega_in^n tau) (x) q(-) (x) q(omega_ib^b tau)       exact rotation-vector quaternions
    f^n    = 1/2 (C_b^n(-) + C_b^n(+)) f^b
    v(+)   = v(-) + [f^n + g^n(L, h) - (Omega_en^n + 2 Omega_ie^n) v(-)] tau
    h(+)   = h(-) - tau/2 (v_D(-) + v_D(+))
    L(+)   = L(-) + tau/2 (v_N(-) / (R_M(L(-)) + h(-)) + v_N(+) / (R_M(L(-)) + h(+)))
    lam(+) = lam(-) + tau/2 (v_E(-) / ((R_N(L(-)) + h(-)) cos L(-))
                             + v_E(+) / ((R_N(L(+)) + h(+)) cos L(+)))

with ``omega_in^n = omega_ie^n + omega_en^n`` evaluated at the start of the
step. :mod:`ins_gnss.earth` and :mod:`ins_gnss.rotations` have every helper
you need.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from .types import IMUSample, NavState


def mechanize(state: NavState, accel_mps2: np.ndarray, gyro_rps: np.ndarray, dt_s: float) -> NavState:
    """Propagate the nominal navigation state over one IMU interval (Part A, yours).

    Parameters
    ----------
    state:
        Nominal state at the start of the interval. Do not modify it; return a
        new :class:`NavState` (``state.copy()`` is handy).
    accel_mps2, gyro_rps:
        Raw specific force [m/s^2] and angular rate [rad/s] on body axes,
        representative of the interval (the driver passes the average of the
        two samples that bound it). Subtract ``state.accel_bias_mps2`` and
        ``state.gyro_bias_rps`` yourself.
    dt_s:
        Interval length [s]. Use the actual timestamp difference.

    Returns
    -------
    NavState
        The propagated state: attitude first (rotate the body by
        ``omega_ib^b dt`` and the navigation frame by ``omega_in^n dt``, with
        exact rotation-vector quaternions, then normalize), then velocity
        (specific force rotated with the average of the old and new ``C_b^n``,
        plus gravity from :func:`~ins_gnss.earth.gravity_ned`, minus the
        Coriolis and transport-rate term), then position (trapezoidal in
        ``h``, ``L`` and ``lambda``). Biases are carried over unchanged.

    The equations are in this module's docstring and in
    @sec-ins-gnss-navigation-equations-local. A perfect IMU at rest on the
    Earth must leave the velocity at zero (the tests check this at 42 deg N).
    """
    raise NotImplementedError(
        "Part A: implement mechanize(state, accel_mps2, gyro_rps, dt_s) "
        "(curved-Earth strapdown, @sec-ins-gnss-navigation-equations-local)."
    )


def coarse_alignment(mean_specific_force_b: np.ndarray, yaw_rad: float) -> np.ndarray:
    """Initial attitude ``q_b^n`` from a stationary specific-force average and a heading (Part A, yours).

    Leveling (forward-right-down body axes, ``f`` the measured specific force,
    which at rest and level is ``[0, 0, -g]``)::

        roll  = atan2(-f_y, -f_z)
        pitch = atan2(f_x, sqrt(f_y^2 + f_z^2))

    Check the signs numerically before trusting them: rotate ``[0, 0, -g]``
    into a tilted body with :func:`~ins_gnss.rotations.euler_to_dcm` and make
    sure you get the tilt back. The heading cannot come from the gyros (a
    smartphone gyro's bias is tens to hundreds of times Earth rate, so
    gyrocompassing is hopeless); it is an input here: the GNSS course over
    ground once the car moves, a magnetometer, or a second antenna
    (@sec-ins-gnss-alignment). Return the Z-Y-X quaternion of
    ``(roll, pitch, yaw_rad)``.
    """
    raise NotImplementedError(
        "Part A: implement coarse_alignment(mean_specific_force_b, yaw_rad) (@sec-ins-gnss-alignment)."
    )


def free_inertial(
    state: NavState,
    imu_samples: Iterable[IMUSample],
    *,
    mechanize_step: Callable[[NavState, np.ndarray, np.ndarray, float], NavState] | None = None,
) -> tuple[np.ndarray, list[NavState]]:
    """Run the mechanization with no aiding at all (provided).

    ``state`` is the state at the time of the first sample. Each interval
    between consecutive samples is propagated with the average of its two
    bounding samples (trapezoidal). Returns ``(times, states)`` with one state
    per sample time. ``mechanize_step`` defaults to :func:`mechanize`.
    """
    step = mechanize if mechanize_step is None else mechanize_step
    samples = list(imu_samples)
    if not samples:
        return np.zeros(0), []
    times = [samples[0].time_s]
    states = [state.copy()]
    current = state.copy()
    for previous, sample in zip(samples[:-1], samples[1:]):
        dt = sample.time_s - previous.time_s
        accel = 0.5 * (previous.accel_mps2 + sample.accel_mps2)
        gyro = 0.5 * (previous.gyro_rps + sample.gyro_rps)
        current = step(current, accel, gyro, dt)
        times.append(sample.time_s)
        states.append(current)
    return np.array(times), states
