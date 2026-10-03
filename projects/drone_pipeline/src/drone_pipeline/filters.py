"""Part C: EKF, UKF and particle-filter navigators.

All three share the 15-state vector of :mod:`drone_pipeline.mechanization`,
propagate with :func:`~drone_pipeline.mechanization.propagate_state`, and
update with the 6-vector PnP measurement ``z = [px, py, pz, roll, pitch, yaw]``
(``PoseObservation.z``), so ``h(x) = x[:6]``.

The constructors, ``state``/``covariance`` accessors and history format are
provided; ``predict``, ``update`` (and the particle filter's ``estimate`` and
``resample``) are yours. Keep the signatures so that ``run_all`` and the tests
can drive any of the three filters the same way::

    nav.predict(acc, omg, dt)      # one IMU interval
    accepted = nav.update(z)       # one PnP pose; False if the gate rejected it
    nav.state, nav.covariance      # current estimate, (15,) and (15, 15)

Noise conventions, identical for all three filters:

* ``Q`` is a (15, 15) process-noise *spectral density* (per second). Over an
  interval ``dt``, the simplest discrete covariance is ``Q * dt``; a more
  careful discretization is welcome (see the Kalman filter chapter). Either
  way, the same ``Q`` must mean the same thing in all three filters.
* ``R`` is the (6, 6) PnP measurement covariance from Part B (plus any
  inflation you justify and report).

History: every ``update`` call appends one ``dict`` to ``history`` with at
least ``"nis"`` (float) and ``"accepted"`` (bool) and, for the Kalman
filters, ``"innovation"`` (6,) and ``"S"`` (6, 6). The particle filter also
logs ``"n_eff"``, the effective sample size after reweighting and before any
resampling. ``run_all`` summarizes these keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .mechanization import STATE_DIM, PropagationParams
from .utils import wrap_angles

MEASUREMENT_DIM = 6


def _as_matrix(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


@dataclass(slots=True)
class EKFNavigator:
    """Extended Kalman filter on the 15-state model.

    Parameters
    ----------
    initial_state, P0:
        Initial mean (15,) and covariance (15, 15). Initialize from the first
        valid PnP pose, never from Vicon.
    Q, R:
        Process-noise density (15, 15) and measurement covariance (6, 6); see
        the module docstring.
    params:
        Propagation constants passed to ``propagate_state``.
    nis_gate:
        Reject an update whose NIS exceeds this value (for example
        ``scipy.stats.chi2.ppf(0.999, 6)``). ``None`` disables gating.
    """

    initial_state: np.ndarray
    P0: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    params: PropagationParams = field(default_factory=PropagationParams)
    nis_gate: float | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    x: np.ndarray = field(init=False, repr=False)
    P: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.x = _as_matrix(self.initial_state, (STATE_DIM,), "initial_state")
        self.P = _as_matrix(self.P0, (STATE_DIM, STATE_DIM), "P0")
        self.Q = _as_matrix(self.Q, (STATE_DIM, STATE_DIM), "Q")
        self.R = _as_matrix(self.R, (MEASUREMENT_DIM, MEASUREMENT_DIM), "R")

    @property
    def state(self) -> np.ndarray:
        """Current state estimate, shape (15,)."""
        return self.x.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Current state covariance, shape (15, 15)."""
        return self.P.copy()

    def predict(self, acc: np.ndarray, omg: np.ndarray, dt: float) -> None:
        """Propagate ``x`` with ``propagate_state`` and ``P`` with the Jacobian ``F``."""
        raise NotImplementedError("Part C: implement the EKF prediction step")

    def update(self, z: np.ndarray) -> bool:
        """Fuse one PnP pose ``z`` (6,); return False if the NIS gate rejected it."""
        raise NotImplementedError("Part C: implement the EKF update step")


@dataclass(slots=True)
class UKFNavigator:
    """Unscented Kalman filter on the 15-state model.

    Same parameters as :class:`EKFNavigator`, plus the scaled unscented
    transform parameters ``alpha``, ``beta`` and ``kappa``.
    """

    initial_state: np.ndarray
    P0: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    params: PropagationParams = field(default_factory=PropagationParams)
    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0
    nis_gate: float | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    x: np.ndarray = field(init=False, repr=False)
    P: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.x = _as_matrix(self.initial_state, (STATE_DIM,), "initial_state")
        self.P = _as_matrix(self.P0, (STATE_DIM, STATE_DIM), "P0")
        self.Q = _as_matrix(self.Q, (STATE_DIM, STATE_DIM), "Q")
        self.R = _as_matrix(self.R, (MEASUREMENT_DIM, MEASUREMENT_DIM), "R")

    @property
    def state(self) -> np.ndarray:
        """Current state estimate, shape (15,)."""
        return self.x.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Current state covariance, shape (15, 15)."""
        return self.P.copy()

    def predict(self, acc: np.ndarray, omg: np.ndarray, dt: float) -> None:
        """Push sigma points through ``propagate_state``; recombine with angle-aware means."""
        raise NotImplementedError("Part C: implement the UKF prediction step")

    def update(self, z: np.ndarray) -> bool:
        """Redraw sigma points from the prior, then fuse ``z``; False if gated out."""
        raise NotImplementedError("Part C: implement the UKF update step")


@dataclass(slots=True)
class ParticleFilterNavigator:
    """Bootstrap particle filter on the 15-state model.

    Parameters
    ----------
    initial_particles:
        (N, 15) array of initial particles, for example drawn around the first
        PnP pose.
    Q, R:
        Process-noise density (15, 15) and measurement covariance (6, 6).
    params:
        Propagation constants.
    weights:
        Optional initial weights (N,); uniform when ``None``. Holds the live,
        normalized weights after construction.
    resample_threshold:
        Resample when ``n_eff < resample_threshold * N``.
    seed:
        Seed for ``rng`` so runs are reproducible.
    """

    initial_particles: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    params: PropagationParams = field(default_factory=PropagationParams)
    weights: np.ndarray | None = None
    resample_threshold: float = 0.5
    seed: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    particles: np.ndarray = field(init=False, repr=False)
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        particles = np.array(self.initial_particles, dtype=np.float64, copy=True)
        if particles.ndim != 2 or particles.shape[1] != STATE_DIM:
            raise ValueError(f"initial_particles must have shape (N, {STATE_DIM}), got {particles.shape}")
        self.particles = particles
        count = particles.shape[0]
        if self.weights is None:
            self.weights = np.full(count, 1.0 / count)
        else:
            weights = np.array(self.weights, dtype=np.float64, copy=True)
            if weights.shape != (count,):
                raise ValueError(f"weights must have shape ({count},), got {weights.shape}")
            self.weights = weights / weights.sum()
        self.Q = _as_matrix(self.Q, (STATE_DIM, STATE_DIM), "Q")
        self.R = _as_matrix(self.R, (MEASUREMENT_DIM, MEASUREMENT_DIM), "R")
        self.rng = np.random.default_rng(self.seed)

    @property
    def num_particles(self) -> int:
        """Number of particles N."""
        return int(self.particles.shape[0])

    @property
    def effective_sample_size(self) -> float:
        """``N_eff = 1 / sum(w_i^2)`` for the current normalized weights."""
        return float(1.0 / np.sum(self.weights**2))

    @property
    def state(self) -> np.ndarray:
        """The weighted-mean estimate, ``estimate("weighted_mean")``."""
        return self.estimate("weighted_mean")

    @property
    def covariance(self) -> np.ndarray:
        """Weighted particle covariance about ``state``, angle residuals wrapped."""
        deviations = self.particles - self.state
        deviations[:, 3:6] = wrap_angles(deviations[:, 3:6])
        return (self.weights[:, None] * deviations).T @ deviations

    def estimate(self, method: str = "weighted_mean") -> np.ndarray:
        """Return a (15,) point estimate from the particle set.

        ``method`` is one of ``"max_weight"`` (highest-weight particle),
        ``"mean"`` (unweighted mean) or ``"weighted_mean"``. Means of the three
        angles must be angle-aware (average ``sin`` and ``cos``, then
        ``atan2``), or a yaw cloud straddling +/- pi averages to nonsense.
        """
        raise NotImplementedError("Part C: implement the particle-filter point estimates")

    def predict(self, acc: np.ndarray, omg: np.ndarray, dt: float) -> None:
        """Propagate every particle and add sampled process noise (vectorized)."""
        raise NotImplementedError("Part C: implement the particle-filter prediction step")

    def update(self, z: np.ndarray) -> bool:
        """Reweight with the PnP likelihood (in log space).

        Do not resample here: the caller records the point estimates first and
        then calls :meth:`resample_if_needed`. Log ``"n_eff"`` in ``history``.
        """
        raise NotImplementedError("Part C: implement the particle-filter update step")

    def resample(self) -> None:
        """Low-variance (systematic) resampling; reset weights to 1/N."""
        raise NotImplementedError("Part C: implement low-variance or systematic resampling")

    def resample_if_needed(self) -> bool:
        """Call :meth:`resample` when ``n_eff < resample_threshold * N``; return whether it did."""
        if self.effective_sample_size < self.resample_threshold * self.num_particles:
            self.resample()
            return True
        return False
