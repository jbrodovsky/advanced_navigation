"""Shared evaluation helpers so every report computes errors the same way.

Errors are evaluated at packet timestamps against Vicon truth interpolated with
:func:`~drone_pipeline.datasets.interpolate_vicon_truth`, using only samples
whose truth is valid. Errors are true minus estimate; attitude errors are
wrapped into [-pi, pi).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import chi2

from .types import TruthSample
from .utils import wrap_angles


def truth_matrix(truth_samples: Sequence[TruthSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack truth samples into arrays.

    Returns ``(times, states, valid)``: ``times`` (N,), ``states`` (N, 9) with
    columns ``[px, py, pz, roll, pitch, yaw, vx, vy, vz]`` (the part of the
    15-state that Vicon observes), and the boolean ``valid`` mask (N,).
    """
    times = np.array([sample.t for sample in truth_samples], dtype=np.float64)
    states = np.array(
        [np.concatenate([sample.position, sample.rpy, sample.velocity]) for sample in truth_samples],
        dtype=np.float64,
    ).reshape(len(truth_samples), 9)
    valid = np.array([sample.valid for sample in truth_samples], dtype=bool)
    return times, states, valid


def state_errors(estimates: np.ndarray, truth_states: np.ndarray) -> np.ndarray:
    """True-minus-estimate errors on the 9 Vicon-observed states, angles wrapped.

    ``estimates`` is (N, 15) or (N, >=9); ``truth_states`` is (N, 9) from
    :func:`truth_matrix`.
    """
    estimates = np.asarray(estimates, dtype=np.float64)
    errors = np.asarray(truth_states, dtype=np.float64) - estimates[:, :9]
    errors[:, 3:6] = wrap_angles(errors[:, 3:6])
    return errors


def rmse(errors: np.ndarray, valid: np.ndarray | None = None) -> dict[str, float]:
    """Summarize (N, 9) errors from :func:`state_errors` over valid samples.

    Returns the 3-D position RMSE ``sqrt(mean ||e_p||^2)`` in metres, the
    per-axis position RMSE, the per-axis attitude RMSE in degrees, and the 3-D
    velocity RMSE.
    """
    errors = np.asarray(errors, dtype=np.float64)
    if valid is not None:
        errors = errors[np.asarray(valid, dtype=bool)]
    errors = errors[np.all(np.isfinite(errors), axis=1)]
    if errors.size == 0:
        nan = float("nan")
        return {key: nan for key in ("pos_3d_m", "x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg", "vel_3d_mps")}
    pos = errors[:, 0:3]
    att = np.degrees(errors[:, 3:6])
    vel = errors[:, 6:9]
    axis_pos = np.sqrt(np.mean(pos**2, axis=0))
    axis_att = np.sqrt(np.mean(att**2, axis=0))
    return {
        "pos_3d_m": float(np.sqrt(np.mean(np.sum(pos**2, axis=1)))),
        "x_m": float(axis_pos[0]),
        "y_m": float(axis_pos[1]),
        "z_m": float(axis_pos[2]),
        "roll_deg": float(axis_att[0]),
        "pitch_deg": float(axis_att[1]),
        "yaw_deg": float(axis_att[2]),
        "vel_3d_mps": float(np.sqrt(np.mean(np.sum(vel**2, axis=1)))),
    }


def nees(errors: np.ndarray, covariances: np.ndarray, indices: Sequence[int] = range(9)) -> np.ndarray:
    """Normalized estimation error squared ``e^T P^-1 e`` per sample.

    ``errors`` (N, 9) from :func:`state_errors`; ``covariances`` (N, 15, 15).
    ``indices`` selects the states to test (default: position, attitude and
    velocity, the 9 states Vicon observes). Each value is chi-square with
    ``len(indices)`` degrees of freedom for a consistent filter.
    """
    idx = np.asarray(list(indices))
    e = np.asarray(errors, dtype=np.float64)[:, idx]
    P = np.asarray(covariances, dtype=np.float64)[:, idx[:, None], idx[None, :]]
    try:
        return np.einsum("ni,ni->n", e, np.linalg.solve(P, e[..., None])[..., 0])
    except np.linalg.LinAlgError:
        # A collapsed particle cloud can have a singular covariance. Handle it
        # sample by sample: any error in a direction the filter claims to know
        # exactly (the covariance null space) is infinitely overconfident.
        return np.array([_nees_one(ek, Pk) for ek, Pk in zip(e, P)])


def _nees_one(error: np.ndarray, covariance: np.ndarray) -> float:
    """``e^T P^-1 e`` for one sample, returning ``inf`` for error in P's null space."""
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    scale = max(float(eigenvalues.max(initial=0.0)), 1e-300)
    null = eigenvalues <= scale * 1e-12
    components = eigenvectors.T @ error
    if np.any(np.abs(components[null]) > 1e-12 * max(1.0, float(np.linalg.norm(error)))):
        return float("inf")
    return float(np.sum(components[~null] ** 2 / eigenvalues[~null]))


def chi2_interval(dof: int, probability: float = 0.95, num_averaged: int = 1) -> tuple[float, float]:
    """Two-sided chi-square acceptance interval for an average of NEES/NIS values.

    For a single sample use ``num_averaged=1``. For the average of ``K``
    independent samples the bounds are ``chi2.ppf(., K * dof) / K``. Samples
    along one trajectory are correlated in time, so treat the averaged bounds
    as optimistic.
    """
    tail = (1.0 - probability) / 2.0
    total = dof * num_averaged
    return float(chi2.ppf(tail, total) / num_averaged), float(chi2.ppf(1.0 - tail, total) / num_averaged)


def fraction_inside(values: np.ndarray, dof: int, probability: float = 0.95) -> float:
    """Fraction of NEES/NIS samples inside the single-sample chi-square interval."""
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]  # keep +inf: a diverged sample is outside the bounds
    if values.size == 0:
        return float("nan")
    low, high = chi2_interval(dof, probability)
    return float(np.mean((values >= low) & (values <= high)))
