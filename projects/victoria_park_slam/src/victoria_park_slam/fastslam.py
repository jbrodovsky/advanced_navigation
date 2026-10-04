"""FastSLAM 1.0 contracts for the Victoria Park assignment.

The bookkeeping (construction, weights, effective sample size, pose estimate,
map queries, particle copies) is provided. The estimation logic is yours:

* :meth:`LandmarkEKF.initialize`, :meth:`LandmarkEKF.innovation`,
  :meth:`LandmarkEKF.updated` -- the per-landmark EKF;
* :meth:`FastSLAM.predict` -- sample every particle from the motion model;
* :meth:`FastSLAM.associate` -- two-gate association inside one particle;
* :meth:`FastSLAM.update` -- landmark updates, birth, deletion, weights;
* :meth:`FastSLAM.resample` -- systematic resampling.

Shared specification (the same one the SLAM chapter uses)
--------------------------------------------------------
* Particle pose = laser pose ``[x_L, y_L, phi]``; measurement ``z = [r, beta]``
  from the laser, ``beta`` in the vehicle frame. Innovation ``nu = z - h``
  with the bearing wrapped to ``[-pi, pi)``; ``S = H_l P_l H_l^T + R``.
* Matched landmark: ``log w += log N(nu; 0, S)``.
* New landmark **or discarded observation**: ``log w += log p0``
  (:attr:`FastSLAM.log_p0`), the density of a measurement that the
  particle's map does not explain. Every observation therefore contributes
  exactly one factor to every particle. Without ``p0`` the comparison is
  against an implicit density of 1 per (m rad), and whether that rewards or
  punishes inventing landmarks depends on your units.
* Two gates per particle: associate if ``d^2 < gate_threshold`` (5.99, the
  95% chi-square point for 2 dof); spawn a tentative landmark only if
  ``d^2 > birth_threshold`` (13.82, 99.9%) for every existing landmark;
  otherwise discard. One landmark may absorb at most one observation per scan
  (greedy by smallest ``d^2``).
* Tentative -> confirmed after ``confirm_associations`` (3) successful
  associations; a tentative landmark not seen for ``tentative_timeout_s`` is
  deleted.
* Weights live in log space, normalized with log-sum-exp; resample
  (systematic) when ``N_eff = 1 / sum(w^2) < resample_threshold * N``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import math

import numpy as np

from .types import (
    Association,
    LandmarkState,
    LandmarkTrackState,
    MeasurementModelConfig,
    MotionModelConfig,
    TreeObservation,
)
from .utils import effective_sample_size, normalize_log_weights

CHI2_2DOF_95 = 5.991
"""95% point of the chi-square distribution with 2 degrees of freedom."""
CHI2_2DOF_999 = 13.816
"""99.9% point of the chi-square distribution with 2 degrees of freedom."""
LASER_FIELD_OF_VIEW_RAD = math.pi
"""The SICK scanner covers 180 degrees."""


@dataclass(slots=True)
class LandmarkEKF(LandmarkState):
    """Per-landmark EKF stored inside one FastSLAM particle.

    Inherits the state and bookkeeping fields of
    :class:`~victoria_park_slam.types.LandmarkState` (``mean_xy_m``,
    ``covariance``, ``observation_count``, ``first_seen_s``, ``last_seen_s``,
    ``track_state``). The pose passed to every method is the particle's laser
    pose, treated as known: that conditioning is what makes FastSLAM's map
    factor into tiny independent 2x2 filters.

    **Copy-on-write.** After resampling, several particles hold references to
    the same ``LandmarkEKF`` object (see :meth:`FastSLAMParticle.copy`). So
    :meth:`updated` must return a *new* object and leave ``self`` untouched;
    the caller stores the result in its own particle's dictionary. That is
    what makes resampling cost O(number of landmarks) pointer copies instead
    of deep copies.
    """

    @classmethod
    def initialize(
        cls,
        pose: np.ndarray,
        measurement: np.ndarray,
        measurement_covariance: np.ndarray,
        *,
        time_s: float = 0.0,
    ) -> LandmarkEKF:
        """Create a landmark from one ``[range, bearing]`` observation.

        Use the inverse observation model
        (:func:`~victoria_park_slam.utils.inverse_range_bearing`) for the mean
        and propagate ``R`` through its Jacobian ``G_z``
        (:func:`~victoria_park_slam.utils.inverse_range_bearing_jacobian`):
        ``P = G_z R G_z^T``. The new landmark is ``TENTATIVE`` with
        ``observation_count = 1`` and ``first_seen_s = last_seen_s = time_s``.
        """
        raise NotImplementedError("students should implement landmark initialization")

    def innovation(
        self,
        pose: np.ndarray,
        measurement: np.ndarray,
        measurement_covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(nu, H_l, S)`` for this landmark seen from ``pose``.

        ``nu = z - h(pose, mean)`` with the bearing component wrapped to
        ``[-pi, pi)``, ``H_l`` the 2x2 Jacobian from
        :func:`~victoria_park_slam.utils.range_bearing_jacobian`, and
        ``S = H_l P H_l^T + R``. (Same return order as the SLAM chapter's
        ``LandmarkEKF``.)
        """
        raise NotImplementedError("students should implement the landmark innovation")

    def updated(
        self,
        pose: np.ndarray,
        measurement: np.ndarray,
        measurement_covariance: np.ndarray,
        *,
        time_s: float | None = None,
    ) -> LandmarkEKF:
        """Return the posterior landmark after fusing one observation (``self`` is unchanged).

        Compute ``nu`` and ``S`` (:meth:`innovation`), the gain
        ``K = P H^T S^-1`` with ``np.linalg.solve`` (never ``inv``), the mean
        ``m + K nu``, and the Joseph-form covariance
        ``(I - K H) P (I - K H)^T + K R K^T``. The returned object has
        ``observation_count + 1``, ``last_seen_s = time_s`` (unchanged when
        ``time_s`` is None), and the same ``track_state`` and
        ``first_seen_s``. Do **not** modify ``self``
        (``dataclasses.replace`` is handy).
        """
        raise NotImplementedError("students should implement landmark updates")


@dataclass(slots=True)
class FastSLAMParticle:
    """One FastSLAM pose hypothesis and its conditionally independent map.

    ``pose_xy_theta`` is the laser pose ``[x_L, y_L, phi]``. ``landmarks``
    maps landmark ids to :class:`LandmarkEKF`. ``trajectory`` holds the
    particle's pose after each laser update (append it in your
    :meth:`FastSLAM.update`), which makes the highest-weight particle's
    history a self-consistent path to plot.
    """

    pose_xy_theta: np.ndarray
    log_weight: float = 0.0
    landmarks: dict[int, LandmarkEKF] = field(default_factory=dict)
    trajectory: list[np.ndarray] = field(default_factory=list)

    def copy(self) -> FastSLAMParticle:
        """Copy for resampling: new pose array, dictionary and list; shared landmarks.

        The landmark *objects* are shared between the copy and the original
        (copy-on-write, see :class:`LandmarkEKF`), so the copy is cheap. Never
        modify a ``LandmarkEKF`` in place.
        """
        return FastSLAMParticle(
            pose_xy_theta=np.array(self.pose_xy_theta, dtype=np.float64, copy=True),
            log_weight=float(self.log_weight),
            landmarks=dict(self.landmarks),
            trajectory=list(self.trajectory),
        )


@dataclass(slots=True)
class FastSLAM:
    """FastSLAM 1.0 backend: motion-model proposal, per-particle landmark EKFs.

    Build one with :meth:`from_initial_pose`. The harness
    (:mod:`victoria_park_slam.run_all`) drives it like this::

        for step in scan_steps(dataset):
            for c in step.controls:
                slam.predict(c.speed_mps, c.steering_rad, c.dt_s)
            observations = extract_tree_landmarks(dataset.laser_ranges_m[step.scan_index])
            slam.update(observations, step.time_s)

    Settings (all tunable, report the values you use):

    * ``motion``: vehicle geometry, encoder correction switch, and ``Q_u``.
    * ``measurement``: range and bearing noise (``R``) and the maximum range.
    * ``gate_threshold`` / ``birth_threshold``: the two association gates.
    * ``new_landmark_log_likelihood``: ``log p0``; ``None`` uses the
      sensor-footprint default in :attr:`log_p0`.
    * ``resample_threshold``: resample when ``N_eff < resample_threshold * N``.
    * ``confirm_associations``, ``tentative_timeout_s``: landmark lifecycle.
    * ``rng``: the only source of randomness (seed it for reproducible runs).

    ``history`` is yours to fill: append one dict per :meth:`update` (for
    example ``{"time_s", "matched", "new", "discarded", "n_eff",
    "resampled"}``); ``run_all`` reports whichever of those keys it finds.
    """

    particles: list[FastSLAMParticle]
    motion: MotionModelConfig = field(default_factory=MotionModelConfig)
    measurement: MeasurementModelConfig = field(default_factory=MeasurementModelConfig)
    gate_threshold: float = CHI2_2DOF_95
    birth_threshold: float = CHI2_2DOF_999
    new_landmark_log_likelihood: float | None = None
    resample_threshold: float = 0.5
    confirm_associations: int = 3
    tentative_timeout_s: float = 2.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    next_landmark_id: int = 0
    history: list[dict[str, float]] = field(default_factory=list)
    best_snapshot: FastSLAMParticle | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Provided: construction and bookkeeping
    # ------------------------------------------------------------------

    @classmethod
    def from_initial_pose(
        cls,
        num_particles: int,
        initial_pose: Sequence[float] | np.ndarray = (0.0, 0.0, 0.0),
        **settings,
    ) -> FastSLAM:
        """Create ``num_particles`` identical particles at ``initial_pose`` (laser pose).

        Every particle starts with log weight ``-log(N)`` and an empty map.
        ``settings`` are passed to the constructor (``motion=...``,
        ``gate_threshold=...``, ``rng=...``).
        """
        if num_particles <= 0:
            raise ValueError("num_particles must be positive")
        pose = np.asarray(initial_pose, dtype=np.float64).reshape(3)
        log_weight = -math.log(num_particles)
        particles = [
            FastSLAMParticle(pose_xy_theta=pose.copy(), log_weight=log_weight, trajectory=[pose.copy()])
            for _ in range(num_particles)
        ]
        return cls(particles=particles, **settings)

    @property
    def num_particles(self) -> int:
        return len(self.particles)

    @property
    def measurement_covariance(self) -> np.ndarray:
        """``R`` from :attr:`measurement`."""
        return self.measurement.covariance

    @property
    def log_p0(self) -> float:
        """``log p0``, the weight factor for a new or discarded observation.

        ``p0`` is a density in the same units as ``N(nu; 0, S)``
        (``1 / (m rad)``). If ``new_landmark_log_likelihood`` is ``None`` the
        default treats an unexplained measurement as uniform over the sensor
        footprint, ``p0 = 1 / (max_range * field_of_view)`` with the 180-degree
        field of view (``log p0 = -log(75 pi) = -5.46`` for the defaults).
        Sanity check: with ``S = R`` the Gaussian falls below ``p0`` at
        ``d^2 = -2 log(2 pi p0 sqrt(det R))``, about 15 for the default ``R``,
        close to the 13.82 birth gate; the two should tell a consistent story.
        """
        if self.new_landmark_log_likelihood is not None:
            return float(self.new_landmark_log_likelihood)
        return float(-math.log(self.measurement.max_range_m * LASER_FIELD_OF_VIEW_RAD))

    def log_weights(self) -> np.ndarray:
        """Current particle log weights as an array."""
        return np.array([particle.log_weight for particle in self.particles], dtype=np.float64)

    def weights(self) -> np.ndarray:
        """Normalized particle weights (log-sum-exp; does not modify the particles)."""
        weights, _ = normalize_log_weights(self.log_weights())
        return weights

    def normalize_log_weights(self) -> np.ndarray:
        """Shift every log weight so that ``logsumexp == 0``; return the weights."""
        weights, normalized = normalize_log_weights(self.log_weights())
        for particle, log_weight in zip(self.particles, normalized):
            particle.log_weight = float(log_weight)
        return weights

    def effective_sample_size(self) -> float:
        """``N_eff = 1 / sum(w^2)`` of the normalized weights."""
        return effective_sample_size(self.weights())

    def estimate_pose(self) -> np.ndarray:
        """Weighted mean laser pose; the heading uses the circular mean."""
        weights = self.weights()
        poses = np.array([particle.pose_xy_theta for particle in self.particles], dtype=np.float64)
        xy = weights @ poses[:, :2]
        phi = math.atan2(float(weights @ np.sin(poses[:, 2])), float(weights @ np.cos(poses[:, 2])))
        return np.array([xy[0], xy[1], phi], dtype=np.float64)

    def best_particle(self) -> FastSLAMParticle:
        """The particle with the largest weight.

        Right after a resample every weight is equal, so the maximum would be
        an arbitrary particle; in that case this returns the copy of the best
        particle that :meth:`resample_if_needed` saved before resampling.
        """
        if self.best_snapshot is not None:
            return self.best_snapshot
        return self.particles[int(np.argmax(self.log_weights()))]

    def resample_if_needed(self) -> tuple[float, bool]:
        """Normalize the weights and resample when ``N_eff < resample_threshold * N``.

        Saves a copy of the highest-weight particle first, so
        :meth:`best_particle` (and the reported map and trajectory) still refer
        to the best particle of this update after the weights become uniform.
        Calls your :meth:`resample`. Returns ``(n_eff, resampled)``, with
        ``n_eff`` computed before resampling.
        """
        weights = self.normalize_log_weights()
        n_eff = effective_sample_size(weights)
        resampled = n_eff < self.resample_threshold * len(self.particles)
        if resampled:
            self.best_snapshot = self.particles[int(np.argmax(weights))].copy()
            self.resample()
        else:
            self.best_snapshot = None
        return n_eff, resampled

    def landmark_counts(self, particle: FastSLAMParticle | None = None) -> tuple[int, int]:
        """``(total, confirmed)`` landmarks in ``particle`` (default: the best one)."""
        particle = self.best_particle() if particle is None else particle
        confirmed = sum(
            1 for landmark in particle.landmarks.values() if landmark.track_state == LandmarkTrackState.CONFIRMED
        )
        return len(particle.landmarks), confirmed

    def landmark_means(
        self, particle: FastSLAMParticle | None = None, *, confirmed_only: bool = True
    ) -> np.ndarray:
        """``(M, 2)`` landmark means of ``particle`` (default: the best one)."""
        particle = self.best_particle() if particle is None else particle
        means = [
            landmark.mean_xy_m
            for landmark in particle.landmarks.values()
            if not confirmed_only or landmark.track_state == LandmarkTrackState.CONFIRMED
        ]
        return np.array(means, dtype=np.float64).reshape(-1, 2)

    # ------------------------------------------------------------------
    # Yours: the FastSLAM algorithm
    # ------------------------------------------------------------------

    def predict(self, speed_mps: float, steering_rad: float, dt_s: float) -> None:
        """Sample every particle's laser pose from the motion model over ``dt_s``.

        ``speed_mps`` is the raw rear-left encoder speed and ``steering_rad``
        the steering angle, as logged (one :class:`~victoria_park_slam.timing.ControlStep`).
        For each particle, add a sample of ``N(0, Q_u)``
        (``self.motion.control_noise_covariance``, drawn with ``self.rng``) to
        ``[speed, steering]``, convert the noisy encoder speed to the
        centerline speed (:func:`~victoria_park_slam.utils.encoder_to_centerline_speed`,
        when ``self.motion.use_encoder_correction``), and propagate with
        :func:`~victoria_park_slam.utils.predict_laser_pose`. Vectorize over
        particles: this runs about 8-9 times per scan.
        """
        raise NotImplementedError("students should implement the FastSLAM prediction step")

    def associate(self, particle: FastSLAMParticle, measurements: np.ndarray) -> list[Association]:
        """Associate ``measurements`` (``(M, 2)`` rows ``[r, beta]``) inside one particle.

        Return one :class:`~victoria_park_slam.types.Association` per row, in
        order:

        * ``MATCH`` with ``landmark_id`` when ``d^2 < self.gate_threshold``,
          using a greedy one-to-one assignment by smallest ``d^2`` so no
          landmark absorbs two observations from the same scan;
        * ``NEW`` when ``d^2 > self.birth_threshold`` for every landmark in
          the particle (or the particle has none);
        * ``DISCARD`` otherwise, with ``landmark_id`` the nearest landmark.

        Consider every landmark of the particle, tentative or confirmed
        (pre-filtering by range is a good speed-up). Do not modify the
        particle.
        """
        raise NotImplementedError("students should implement per-particle data association")

    def update(self, observations: Sequence[TreeObservation], time_s: float) -> None:
        """Process one scan's observations taken at ``time_s``.

        For every particle: associate, update matched landmarks with
        :meth:`LandmarkEKF.updated` (store the returned objects), initialize
        new tentative landmarks (one new id per
        observation from ``self.next_landmark_id``, shared by all particles that
        spawn from it), add the log-weight factors of the shared spec, promote
        landmarks to ``CONFIRMED`` after ``confirm_associations`` associations,
        delete tentative landmarks not seen for ``tentative_timeout_s``, and
        append the pose to ``particle.trajectory``. Then call the provided
        :meth:`resample_if_needed`, which normalizes the log weights, remembers
        the best particle, and calls :meth:`resample` when ``N_eff`` is below
        ``resample_threshold * N``. An empty observation list still needs the
        bookkeeping (timeouts, trajectory).
        """
        raise NotImplementedError("students should implement the FastSLAM update step")

    def resample(self) -> None:
        """Systematic resampling of the particle set.

        Draw indices with
        :func:`~victoria_park_slam.utils.systematic_resample_indices` (pass
        ``self.rng``), replace the particle list with ``particles[i].copy()``
        for each index, and reset every log weight to ``-log(N)``.
        """
        raise NotImplementedError("students should implement particle resampling")
