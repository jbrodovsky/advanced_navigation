"""Part A: inertial mechanization for the 15-state drone model.

State vector (shared by Parts A, B and C)::

    x = [px, py, pz,          # position of the body origin, world frame (m)
         roll, pitch, yaw,    # Z-X-Y Euler angles (rad)
         vx, vy, vz,          # velocity, world frame (m/s)
         bgx, bgy, bgz,       # gyro bias (rad/s)
         bax, bay, baz]       # accelerometer bias (m/s^2)

Continuous-time model, with ``C = Rz(yaw) @ Rx(roll) @ Ry(pitch)``::

    p_dot   = v
    rpy_dot = W(rpy)^-1 @ (omg - b_g)
    v_dot   = C @ (acc - b_a) + gravity_world
    b_dot   = 0            (bias drift enters through process noise in Part C)

where ``omg = W(rpy) @ rpy_dot`` relates Z-X-Y Euler rates to body rates::

    W = [[ cos(pitch), 0, -cos(roll) * sin(pitch)],
         [ 0,          1,  sin(roll)             ],
         [ sin(pitch), 0,  cos(roll) * cos(pitch)]]

The Part A handout gives ``W^-1`` explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STATE_DIM = 15
POSITION = slice(0, 3)
ATTITUDE = slice(3, 6)
VELOCITY = slice(6, 9)
GYRO_BIAS = slice(9, 12)
ACCEL_BIAS = slice(12, 15)


@dataclass(slots=True)
class PropagationParams:
    """Constants used by :func:`propagate_state`.

    ``gravity_world`` is the gravity vector in the world frame. The world
    ``z`` axis points up out of the mat, so gravity is ``[0, 0, -9.81]`` and a
    hovering accelerometer reads about ``[0, 0, +9.81]``.
    """

    gravity_world: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81], dtype=np.float64))


def propagate_state(
    x: np.ndarray,
    acc: np.ndarray,
    omg: np.ndarray,
    dt: float,
    params: PropagationParams | None = None,
) -> np.ndarray:
    """Propagate the 15-state vector over one IMU interval (Part A deliverable).

    Parameters
    ----------
    x:
        Current state, shape ``(15,)`` (layout in the module docstring).
        Supporting an ``(N, 15)`` array of particles as well is optional but
        makes the Part C particle filter much faster.
    acc:
        Accelerometer specific force in the body frame (m/s^2), ``packet.acc``.
    omg:
        Gyro body angular rate (rad/s), ``packet.omg``.
    dt:
        Interval in seconds. Use the actual timestamp difference; it is not
        constant.
    params:
        :class:`PropagationParams`; ``None`` means the defaults.

    Returns
    -------
    np.ndarray
        The propagated state with the same shape as ``x``, with roll, pitch
        and yaw wrapped into [-pi, pi). Biases are held constant.
    """
    raise NotImplementedError(
        "Part A: implement propagate_state(x, acc, omg, dt, params) for the "
        "15-state model in the Part A handout."
    )
