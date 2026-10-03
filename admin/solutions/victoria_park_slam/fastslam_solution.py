"""Reference FastSLAM 1.0 for Project 2 (instructor only).

Subclasses of the starter classes that fill in every student method while
keeping the public contract of ``victoria_park_slam.fastslam`` (the student
tests run against these classes unchanged; see ``check_solutions.py``).

Implementation choices worth pointing out when grading:

* Prediction is vectorized over particles: one draw of ``(N, 2)`` control
  noise per call from a Cholesky factor of ``Q_u``, then one call to
  ``predict_laser_pose`` on an ``(N, 3)`` array.
* Association is vectorized over (observations x nearby landmarks) inside each
  particle, with a batched ``np.linalg.solve`` for ``d^2`` and a greedy
  one-to-one assignment by smallest ``d^2``.
* Landmarks are immutable from the filter's point of view (copy-on-write):
  ``LandmarkEKF.updated`` returns a new object, so resampling only copies
  dictionaries of references.
* Tentative-landmark deletion scans only recently born landmarks: a
  tentative landmark that survived the previous update (time ``t_prev``) has
  at most ``k - 1`` associations, each within the timeout of the previous
  sighting, so it was born after ``t_prev - (k + 1) * timeout``. Dictionaries keep insertion (= birth) order, so a
  reverse scan can stop at the first older landmark.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import math

import numpy as np

from victoria_park_slam import fastslam
from victoria_park_slam.types import Association, AssociationKind, LandmarkTrackState, TreeObservation
from victoria_park_slam.utils import (
    encoder_to_centerline_speed,
    inverse_range_bearing,
    inverse_range_bearing_jacobian,
    predict_laser_pose,
    range_bearing_jacobian,
    range_bearing_prediction,
    systematic_resample_indices,
    wrap_angle,
)

MATCH, NEW, DISCARD = AssociationKind.MATCH, AssociationKind.NEW, AssociationKind.DISCARD
TENTATIVE, CONFIRMED = LandmarkTrackState.TENTATIVE, LandmarkTrackState.CONFIRMED


@dataclass(slots=True)
class LandmarkEKFSolution(fastslam.LandmarkEKF):
    """Reference per-landmark EKF."""

    @classmethod
    def initialize(cls, pose, measurement, measurement_covariance, *, time_s: float = 0.0):
        z = np.asarray(measurement, dtype=np.float64).reshape(2)
        R = np.asarray(measurement_covariance, dtype=np.float64)
        G = inverse_range_bearing_jacobian(pose, z)
        P = G @ R @ G.T
        return cls(
            mean_xy_m=inverse_range_bearing(pose, z),
            covariance=0.5 * (P + P.T),
            observation_count=1,
            first_seen_s=float(time_s),
            last_seen_s=float(time_s),
            track_state=TENTATIVE,
        )

    def _innovation_and_jacobian(self, pose, measurement, measurement_covariance):
        z = np.asarray(measurement, dtype=np.float64).reshape(2)
        nu = z - range_bearing_prediction(pose, self.mean_xy_m)
        nu[1] = wrap_angle(nu[1])
        H = range_bearing_jacobian(pose, self.mean_xy_m)
        S = H @ self.covariance @ H.T + np.asarray(measurement_covariance, dtype=np.float64)
        return nu, S, H

    def innovation(self, pose, measurement, measurement_covariance):
        nu, S, H = self._innovation_and_jacobian(pose, measurement, measurement_covariance)
        return nu, H, S

    def updated(self, pose, measurement, measurement_covariance, *, time_s: float | None = None):
        R = np.asarray(measurement_covariance, dtype=np.float64)
        nu, S, H = self._innovation_and_jacobian(pose, measurement, R)
        P = self.covariance
        # K = P H^T S^-1  ->  K^T = S^-1 (H P)   (P and S symmetric)
        K = np.linalg.solve(S, H @ P).T
        I_KH = np.eye(2) - K @ H
        P_new = I_KH @ P @ I_KH.T + K @ R @ K.T
        return replace(
            self,
            mean_xy_m=self.mean_xy_m + K @ nu,
            covariance=0.5 * (P_new + P_new.T),
            observation_count=self.observation_count + 1,
            last_seen_s=self.last_seen_s if time_s is None else float(time_s),
        )


@dataclass(slots=True)
class FastSLAMSolution(fastslam.FastSLAM):
    """Reference FastSLAM 1.0 (motion-model proposal, two-gate association)."""

    last_update_s: float | None = None  # time of the previous update (bounds the stale-landmark scan)

    # ------------------------------------------------------------------
    def predict(self, speed_mps: float, steering_rad: float, dt_s: float) -> None:
        if dt_s <= 0.0:
            return
        n = len(self.particles)
        chol = np.linalg.cholesky(np.asarray(self.motion.control_noise_covariance, dtype=np.float64))
        noise = self.rng.standard_normal((n, 2)) @ chol.T
        speed = speed_mps + noise[:, 0]
        steering = steering_rad + noise[:, 1]
        geometry = self.motion.geometry
        if self.motion.use_encoder_correction:
            speed = encoder_to_centerline_speed(speed, steering, H=geometry.encoder_offset_m, L=geometry.wheelbase_m)
        poses = np.array([particle.pose_xy_theta for particle in self.particles], dtype=np.float64)
        poses = predict_laser_pose(poses, speed, steering, dt_s, geometry)
        for particle, pose in zip(self.particles, poses):
            particle.pose_xy_theta = pose

    # ------------------------------------------------------------------
    def _gate(self, particle, Z: np.ndarray):
        """Association plus the arrays the update reuses.

        Returns ``(associations, cache)`` where ``cache`` is ``None`` when no
        landmark is near, else ``(ids, landmarks, nu, S, H, d2)`` over the
        nearby landmarks: ``nu`` (m, n, 2), ``S`` (n, 2, 2), ``H`` (n, 2, 2),
        ``d2`` (m, n).
        """
        m = Z.shape[0]
        if m == 0:
            return [], None
        if not particle.landmarks:
            return [Association(NEW) for _ in range(m)], None

        pose = np.asarray(particle.pose_xy_theta, dtype=np.float64)
        landmark_list = list(particle.landmarks.values())
        means = np.array([lm.mean_xy_m for lm in landmark_list], dtype=np.float64)
        delta = means - pose[:2]
        q = np.einsum("ni,ni->n", delta, delta)
        # Landmarks far beyond the sensor range cannot be within either gate.
        reach = self.measurement.max_range_m + 10.0
        near = np.flatnonzero(q < reach * reach)
        if near.size == 0:
            return [Association(NEW) for _ in range(m)], None
        all_ids = list(particle.landmarks.keys())
        ids = [all_ids[i] for i in near]
        nearby = [landmark_list[i] for i in near]
        delta, q = delta[near], q[near]
        covs = np.array([lm.covariance for lm in nearby], dtype=np.float64)

        r = np.sqrt(q)
        z_hat = np.column_stack([r, wrap_angle(np.arctan2(delta[:, 1], delta[:, 0]) - pose[2])])
        H = np.empty((near.size, 2, 2))
        H[:, 0, 0], H[:, 0, 1] = delta[:, 0] / r, delta[:, 1] / r
        H[:, 1, 0], H[:, 1, 1] = -delta[:, 1] / q, delta[:, 0] / q
        S = H @ covs @ H.transpose(0, 2, 1) + self.measurement_covariance
        nu = Z[:, None, :] - z_hat[None, :, :]
        nu[..., 1] = wrap_angle(nu[..., 1])
        solved = np.linalg.solve(np.broadcast_to(S, (m, *S.shape)), nu[..., None])[..., 0]
        d2 = np.einsum("mni,mni->mn", nu, solved)

        result: list[Association | None] = [None] * m
        # Greedy one-to-one assignment inside the gate, smallest d^2 first.
        obs_idx, lm_idx = np.nonzero(d2 < self.gate_threshold)
        order = np.argsort(d2[obs_idx, lm_idx], kind="stable")
        used_obs: set[int] = set()
        used_lm: set[int] = set()
        for k in order:
            j, i = int(obs_idx[k]), int(lm_idx[k])
            if j in used_obs or i in used_lm:
                continue
            used_obs.add(j)
            used_lm.add(i)
            result[j] = Association(MATCH, ids[i], float(d2[j, i]))
        nearest = np.argmin(d2, axis=1)
        for j in range(m):
            if result[j] is not None:
                continue
            i = int(nearest[j])
            if d2[j, i] > self.birth_threshold:
                result[j] = Association(NEW, None, float(d2[j, i]))
            else:
                result[j] = Association(DISCARD, ids[i], float(d2[j, i]))
        return result, (ids, nearby, nu, S, H, d2)

    def associate(self, particle, measurements) -> list[Association]:
        Z = np.asarray(measurements, dtype=np.float64).reshape(-1, 2)
        associations, _ = self._gate(particle, Z)
        return associations

    # ------------------------------------------------------------------
    def update(self, observations: Sequence[TreeObservation], time_s: float) -> None:
        Z = np.array([obs.z for obs in observations], dtype=np.float64).reshape(-1, 2)
        R = self.measurement_covariance
        log_p0 = self.log_p0
        timeout = self.tentative_timeout_s
        confirm = self.confirm_associations
        previous = time_s if self.last_update_s is None else min(time_s, self.last_update_s)
        horizon = previous - (confirm + 1) * timeout - 1.0
        new_ids: dict[int, int] = {}
        totals = {MATCH: 0, NEW: 0, DISCARD: 0}
        log_2pi = math.log(2.0 * math.pi)

        for particle in self.particles:
            pose = particle.pose_xy_theta
            landmarks = particle.landmarks
            associations, cache = self._gate(particle, Z)
            scored_obs: list[int] = []  # MATCH: Gaussian factor from the cache
            scored_lm: list[int] = []
            matched: list[tuple[int, int]] = []  # (observation, cache column)
            if cache is not None:
                ids, nearby, nu, S, H, d2 = cache
                column = {landmark_id: i for i, landmark_id in enumerate(ids)}
            for j, association in enumerate(associations):
                totals[association.kind] += 1
                if association.kind is NEW:
                    landmark_id = new_ids.setdefault(j, self.next_landmark_id + len(new_ids))
                    landmarks[landmark_id] = LandmarkEKFSolution.initialize(pose, Z[j], R, time_s=time_s)
                    particle.log_weight += log_p0
                elif association.kind is DISCARD:
                    particle.log_weight += log_p0  # unexplained, like a new landmark (no update)
                else:
                    i = column[association.landmark_id]
                    scored_obs.append(j)
                    scored_lm.append(i)
                    matched.append((j, i))

            if scored_obs:
                # log N(nu; 0, S) = -d2/2 - log(2 pi) - log(det S)/2 for 2-vectors.
                S_sel = S[scored_lm]
                det = S_sel[:, 0, 0] * S_sel[:, 1, 1] - S_sel[:, 0, 1] * S_sel[:, 1, 0]
                particle.log_weight += float(np.sum(-0.5 * d2[scored_obs, scored_lm] - log_2pi - 0.5 * np.log(det)))
            if matched:
                # Batched EKF update, the same equations as LandmarkEKFSolution.updated.
                js, cols = (np.array(v) for v in zip(*matched))
                H_m, S_m, nu_m = H[cols], S[cols], nu[js, cols]
                P_m = np.array([nearby[i].covariance for i in cols])
                K = np.linalg.solve(S_m, H_m @ P_m).transpose(0, 2, 1)  # P H^T S^-1
                I_KH = np.eye(2) - K @ H_m
                P_new = I_KH @ P_m @ I_KH.transpose(0, 2, 1) + K @ R @ K.transpose(0, 2, 1)
                P_new = 0.5 * (P_new + P_new.transpose(0, 2, 1))
                shift = np.einsum("kij,kj->ki", K, nu_m)
                for k, i in enumerate(cols):
                    landmark = nearby[i]
                    count = landmark.observation_count + 1
                    state = CONFIRMED if count - 1 >= confirm else landmark.track_state
                    landmarks[ids[i]] = replace(
                        landmark,
                        mean_xy_m=landmark.mean_xy_m + shift[k],
                        covariance=P_new[k],
                        observation_count=count,
                        last_seen_s=float(time_s),
                        track_state=state,
                    )

            stale = []
            for landmark_id in reversed(landmarks):
                landmark = landmarks[landmark_id]
                if landmark.first_seen_s < horizon:
                    break
                if landmark.track_state is TENTATIVE and time_s - landmark.last_seen_s > timeout:
                    stale.append(landmark_id)
            for landmark_id in stale:
                del landmarks[landmark_id]
            particle.trajectory.append(np.array(pose, dtype=np.float64, copy=True))

        self.next_landmark_id += len(new_ids)
        self.last_update_s = float(time_s)
        weights = self.normalize_log_weights()
        n_eff = float(1.0 / np.sum(weights * weights))
        resampled = n_eff < self.resample_threshold * len(self.particles)
        if resampled:
            self.resample()
        n = len(self.particles)
        self.history.append(
            {
                "time_s": float(time_s),
                "matched": totals[MATCH] / n,
                "new": totals[NEW] / n,
                "discarded": totals[DISCARD] / n,
                "n_eff": n_eff,
                "resampled": bool(resampled),
            }
        )

    # ------------------------------------------------------------------
    def resample(self) -> None:
        indices = systematic_resample_indices(self.weights(), self.rng)
        log_uniform = -math.log(len(indices))
        particles = [self.particles[i].copy() for i in indices]
        for particle in particles:
            particle.log_weight = log_uniform
        self.particles = particles
