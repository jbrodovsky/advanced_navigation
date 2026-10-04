"""Reference solution for Project 1 Part C (instructor only).

Subclasses of the starter navigators that fill in ``predict``/``update`` (and,
for the particle filter, ``estimate``/``resample``). They use the vectorized
reference mechanization so one call propagates a whole sigma-point or particle
set.
"""

from __future__ import annotations

import numpy as np

from drone_pipeline import filters
from drone_pipeline.utils import wrap_angles

from mechanization_solution import propagate_state

ANGLES = slice(3, 6)
N_STATE = 15


def _wrap_state_residual(delta: np.ndarray) -> np.ndarray:
    delta = np.array(delta, dtype=np.float64, copy=True)
    delta[..., ANGLES] = wrap_angles(delta[..., ANGLES])
    return delta


def _measurement_residual(z: np.ndarray, z_hat: np.ndarray) -> np.ndarray:
    nu = np.asarray(z, dtype=np.float64) - z_hat
    nu[..., 3:6] = wrap_angles(nu[..., 3:6])
    return nu


def numerical_jacobian(x: np.ndarray, acc: np.ndarray, omg: np.ndarray, dt: float, params, eps: float = 1e-6) -> np.ndarray:
    """Central-difference F = d f / d x, all 30 perturbations in one vectorized call."""
    perturbations = np.eye(N_STATE) * eps
    batch = np.vstack([x + perturbations, x - perturbations])
    out = propagate_state(batch, acc, omg, dt, params)
    diff = _wrap_state_residual(out[:N_STATE] - out[N_STATE:])
    return diff.T / (2.0 * eps)


class EKFSolution(filters.EKFNavigator):
    """EKF: numerical F, Q*dt, linear h(x) = x[:6], Joseph update, NIS gate."""

    def predict(self, acc, omg, dt):
        F = numerical_jacobian(self.x, acc, omg, dt, self.params)
        self.x = propagate_state(self.x, acc, omg, dt, self.params)
        self.P = F @ self.P @ F.T + self.Q * dt
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, z):
        nu = _measurement_residual(z, self.x[:6])
        S = self.P[:6, :6] + self.R
        nis = float(nu @ np.linalg.solve(S, nu))
        accepted = self.nis_gate is None or nis <= self.nis_gate
        self.history.append({"innovation": nu, "S": S, "nis": nis, "accepted": accepted})
        if not accepted:
            return False
        K = np.linalg.solve(S, self.P[:6, :]).T  # P H^T S^-1 with S symmetric
        self.x = self.x + K @ nu
        self.x[ANGLES] = wrap_angles(self.x[ANGLES])
        I_KH = np.eye(N_STATE)
        I_KH[:, :6] -= K
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return True


class UKFSolution(filters.UKFNavigator):
    """Additive-noise UKF; sigma points are redrawn from the prior before each update."""

    def _weights(self):
        n = N_STATE
        lam = self.alpha**2 * (n + self.kappa) - n
        wm = np.full(2 * n + 1, 0.5 / (n + lam))
        wc = wm.copy()
        wm[0] = lam / (n + lam)
        wc[0] = wm[0] + (1.0 - self.alpha**2 + self.beta)
        return n + lam, wm, wc

    def _sigma_points(self, x, P):
        scale, _, _ = self._weights()
        L = np.linalg.cholesky(scale * P)
        return np.vstack([x, x + L.T, x - L.T])

    def _mean_state(self, X, wm):
        # Weighted mean of residuals about the central point keeps angles sane
        # even with the large negative W0 that a small alpha produces.
        return _wrap_state_residual(X[0] + wm @ _wrap_state_residual(X - X[0]))

    def predict(self, acc, omg, dt):
        _, wm, wc = self._weights()
        X = propagate_state(self._sigma_points(self.x, self.P), acc, omg, dt, self.params)
        x = self._mean_state(X, wm)
        d = _wrap_state_residual(X - x)
        self.x = x
        self.P = (wc[:, None] * d).T @ d + self.Q * dt
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, z):
        _, wm, wc = self._weights()
        X = self._sigma_points(self.x, self.P)  # redraw: P already contains Q
        Z = X[:, :6]
        z_hat = Z[0] + wm @ _measurement_residual(Z, Z[0])
        z_hat[3:6] = wrap_angles(z_hat[3:6])
        dz = _measurement_residual(Z, z_hat)
        dx = _wrap_state_residual(X - self.x)
        S = (wc[:, None] * dz).T @ dz + self.R
        Pxz = (wc[:, None] * dx).T @ dz
        nu = _measurement_residual(z, z_hat)
        nis = float(nu @ np.linalg.solve(S, nu))
        accepted = self.nis_gate is None or nis <= self.nis_gate
        self.history.append({"innovation": nu, "S": S, "nis": nis, "accepted": accepted})
        if not accepted:
            return False
        K = np.linalg.solve(S, Pxz.T).T
        self.x = self.x + K @ nu
        self.x[ANGLES] = wrap_angles(self.x[ANGLES])
        self.P = self.P - K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return True


class ParticleFilterSolution(filters.ParticleFilterNavigator):
    """Bootstrap PF: vectorized propagation, log weights, systematic resampling."""

    def predict(self, acc, omg, dt):
        self.particles = propagate_state(self.particles, acc, omg, dt, self.params)
        std = np.sqrt(np.clip(np.diag(self.Q) * dt, 0.0, None))
        self.particles += self.rng.normal(size=self.particles.shape) * std
        self.particles[:, ANGLES] = wrap_angles(self.particles[:, ANGLES])

    def update(self, z):
        # Prior innovation and NIS, from the particles and weights BEFORE the
        # likelihood is applied (the posterior would be pulled toward z).
        z_hat = self.estimate("weighted_mean")[:6]
        dz = _measurement_residual(self.particles[:, :6], z_hat)
        S = (self.weights[:, None] * dz).T @ dz + self.R
        nu_bar = _measurement_residual(z, z_hat)
        nis = float(nu_bar @ np.linalg.solve(S, nu_bar))
        if self.nis_gate is not None and nis > self.nis_gate:
            self.history.append({"nis": nis, "accepted": False, "n_eff": self.effective_sample_size})
            return False
        nu = _measurement_residual(z, self.particles[:, :6])
        R_inv = np.linalg.inv(self.R)
        log_like = -0.5 * np.einsum("ni,ij,nj->n", nu, R_inv, nu)
        log_w = np.log(np.maximum(self.weights, 1e-300)) + log_like
        log_w -= log_w.max()
        w = np.exp(log_w)
        self.weights = w / w.sum()
        self.history.append({"nis": nis, "accepted": True, "n_eff": self.effective_sample_size})
        return True

    def estimate(self, method="weighted_mean"):
        if method == "max_weight":
            return self.particles[np.argmax(self.weights)].copy()
        if method == "mean":
            w = np.full(self.num_particles, 1.0 / self.num_particles)
        elif method == "weighted_mean":
            w = self.weights
        else:
            raise ValueError(f"unknown estimator {method!r}")
        mean = w @ self.particles
        mean[ANGLES] = np.arctan2(w @ np.sin(self.particles[:, ANGLES]), w @ np.cos(self.particles[:, ANGLES]))
        return mean

    def resample(self):
        n = self.num_particles
        positions = (self.rng.random() + np.arange(n)) / n
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        index = np.searchsorted(cumulative, positions)
        self.particles = self.particles[index]
        self.weights = np.full(n, 1.0 / n)
