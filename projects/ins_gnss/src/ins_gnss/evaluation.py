"""Shared evaluation helpers so every Project 3 report measures the same way.

* Errors are **true minus estimate**, in NED meters, evaluated at the
  reference (or withheld-fix) timestamps with the estimate *interpolated* to
  them. Never compare samples by index.
* The dataset's baseline is a GNSS-aided reference solution, not truth: use
  :func:`reference_valid_mask` to drop the stretches where it disagrees with
  GNSS (it diverges by up to kilometers in some GNSS gaps).
* Outage experiments withhold GNSS fixes from the filter and score the
  estimate against those **withheld fixes** (:func:`schedule_outages`,
  :func:`outage_errors`).
* Consistency: NIS (and, on synthetic data, NEES) against chi-square bounds
  (:func:`chi2_interval`; @sec-kf-consistency).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from .datasets import gnss_arrays, reference_arrays
from .earth import geodetic_to_ned_delta, ned_to_geodetic_delta
from .types import GNSSSample, MEMSNavTrial


# ---------------------------------------------------------------------------
# Interpolation and position errors
# ---------------------------------------------------------------------------


def interpolate_track(
    t_query: np.ndarray, t: np.ndarray, latitude_rad: np.ndarray, longitude_rad: np.ndarray, height_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearly interpolate a geodetic track to ``t_query``; NaN outside ``[t[0], t[-1]]``.

    Longitude is unwrapped before interpolating, so tracks that cross the
    antimeridian are handled. ``t`` must be increasing.
    """
    t = np.asarray(t, dtype=np.float64)
    t_query = np.asarray(t_query, dtype=np.float64)
    lon = np.unwrap(np.asarray(longitude_rad, dtype=np.float64))
    out = []
    for values in (np.asarray(latitude_rad, dtype=np.float64), lon, np.asarray(height_m, dtype=np.float64)):
        v = np.interp(t_query, t, values)
        v[(t_query < t[0]) | (t_query > t[-1])] = np.nan
        out.append(v)
    out[1] = (out[1] + np.pi) % (2.0 * np.pi) - np.pi
    return out[0], out[1], out[2]


def position_error_ned(
    t_true: np.ndarray,
    latitude_true: np.ndarray,
    longitude_true: np.ndarray,
    height_true: np.ndarray,
    t_est: np.ndarray,
    latitude_est: np.ndarray,
    longitude_est: np.ndarray,
    height_est: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Position error (true minus estimate) in NED meters at the true timestamps.

    The estimate is interpolated to ``t_true``; samples outside the estimate's
    time span are dropped. Returns ``(t, error_ned)`` with ``error_ned`` of
    shape (N, 3), computed in the local NED frame of the estimate.
    """
    t_true = np.asarray(t_true, dtype=np.float64)
    lat_e, lon_e, h_e = interpolate_track(t_true, t_est, latitude_est, longitude_est, height_est)
    keep = np.isfinite(lat_e)
    error = geodetic_to_ned_delta(
        np.asarray(latitude_true, dtype=np.float64)[keep],
        np.asarray(longitude_true, dtype=np.float64)[keep],
        np.asarray(height_true, dtype=np.float64)[keep],
        lat_e[keep],
        lon_e[keep],
        h_e[keep],
    )
    return t_true[keep], error.reshape(-1, 3)


def error_summary(error_ned: np.ndarray) -> dict[str, float]:
    """Horizontal and vertical statistics of (N, 3) NED errors.

    Keys: ``horizontal_rmse_m`` (``sqrt(mean(e_N^2 + e_E^2))``),
    ``horizontal_median_m``, ``horizontal_p95_m``, ``horizontal_max_m``,
    ``vertical_rmse_m``, ``num_samples``.
    """
    error = np.asarray(error_ned, dtype=np.float64).reshape(-1, 3)
    error = error[np.all(np.isfinite(error), axis=1)]
    if error.size == 0:
        nan = float("nan")
        return {key: nan for key in ("horizontal_rmse_m", "horizontal_median_m", "horizontal_p95_m", "horizontal_max_m", "vertical_rmse_m")} | {"num_samples": 0}
    horizontal = np.hypot(error[:, 0], error[:, 1])
    return {
        "horizontal_rmse_m": float(np.sqrt(np.mean(horizontal**2))),
        "horizontal_median_m": float(np.median(horizontal)),
        "horizontal_p95_m": float(np.percentile(horizontal, 95)),
        "horizontal_max_m": float(np.max(horizontal)),
        "vertical_rmse_m": float(np.sqrt(np.mean(error[:, 2] ** 2))),
        "num_samples": int(len(error)),
    }


# ---------------------------------------------------------------------------
# The reference and GNSS-only baselines
# ---------------------------------------------------------------------------


def reference_valid_mask(
    trial: MEMSNavTrial,
    *,
    max_disagreement_m: float = 10.0,
    max_time_to_fix_s: float = 0.5,
) -> np.ndarray:
    """Reference states you can use as a yardstick.

    True where a GNSS fix exists within ``max_time_to_fix_s`` and the
    reference agrees with it horizontally to ``max_disagreement_m``. This
    drops the baseline's start-up transient and its divergences in GNSS gaps;
    it also means the reference is only checked where GNSS is available, so
    "error versus the reference" says little about the stretches in between.
    """
    ref = reference_arrays(trial.reference_states)
    gnss = gnss_arrays(trial.gnss_samples)
    valid = np.zeros(ref["time_s"].shape, dtype=bool)
    if gnss["time_s"].size == 0:
        return valid
    index = np.clip(np.searchsorted(gnss["time_s"], ref["time_s"]), 1, len(gnss["time_s"]) - 1)
    nearest = np.where(
        np.abs(gnss["time_s"][index - 1] - ref["time_s"]) <= np.abs(gnss["time_s"][index] - ref["time_s"]), index - 1, index
    )
    close = np.abs(gnss["time_s"][nearest] - ref["time_s"]) <= max_time_to_fix_s
    offset = geodetic_to_ned_delta(
        ref["latitude_rad"], ref["longitude_rad"], ref["altitude_m"],
        gnss["latitude_rad"][nearest], gnss["longitude_rad"][nearest], gnss["altitude_m"][nearest],
    )
    valid = close & (np.hypot(offset[:, 0], offset[:, 1]) <= max_disagreement_m)
    return valid


def gnss_only_track(gnss_samples: Sequence[GNSSSample], t_query: np.ndarray, *, max_gap_s: float = 1.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The "GNSS-only" navigator: what you get with no IMU at all.

    At each query time: linear interpolation between fixes no more than
    ``max_gap_s`` apart; inside longer gaps (and after the last fix),
    constant-velocity extrapolation from the last fix's speed and course
    (horizontal) and its height. NaN before the first fix.
    """
    data = gnss_arrays(list(gnss_samples))
    t = data["time_s"]
    t_query = np.asarray(t_query, dtype=np.float64)
    lat = np.full(t_query.shape, np.nan)
    lon = np.full(t_query.shape, np.nan)
    h = np.full(t_query.shape, np.nan)
    if t.size == 0:
        return lat, lon, h
    index = np.searchsorted(t, t_query, side="right") - 1  # last fix at or before t_query
    for i, (tq, k) in enumerate(zip(t_query, index)):
        if k < 0:
            continue
        if t[k] == tq or (k + 1 < t.size and t[k + 1] - t[k] <= max_gap_s):
            if t[k] == tq or k + 1 >= t.size:
                lat[i], lon[i], h[i] = data["latitude_rad"][k], data["longitude_rad"][k], data["altitude_m"][k]
            else:
                w = (tq - t[k]) / (t[k + 1] - t[k])
                lat[i] = (1 - w) * data["latitude_rad"][k] + w * data["latitude_rad"][k + 1]
                # Interpolate the wrapped longitude difference so a pair of fixes
                # straddling the antimeridian takes the short way around.
                lon_k = data["longitude_rad"][k]
                d_lon = (data["longitude_rad"][k + 1] - lon_k + np.pi) % (2.0 * np.pi) - np.pi
                lon[i] = (lon_k + w * d_lon + np.pi) % (2.0 * np.pi) - np.pi
                h[i] = (1 - w) * data["altitude_m"][k] + w * data["altitude_m"][k + 1]
            continue
        lat[i], lon[i], h[i] = _extrapolate(gnss_samples[k], tq - t[k])
    return lat, lon, h


def _extrapolate(fix: GNSSSample, tau_s: float) -> tuple[float, float, float]:
    v_ne = fix.velocity_ne_mps()
    if v_ne is None:
        return fix.latitude_rad, fix.longitude_rad, fix.altitude_m
    d_lat, d_lon, _ = ned_to_geodetic_delta(np.array([v_ne[0] * tau_s, v_ne[1] * tau_s, 0.0]), fix.latitude_rad, fix.altitude_m)
    return fix.latitude_rad + d_lat, fix.longitude_rad + d_lon, fix.altitude_m


def heading_error_vs_course(
    t_est: np.ndarray,
    yaw_est_rad: np.ndarray,
    gnss_samples: Sequence[GNSSSample],
    *,
    min_speed_mps: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Course over ground minus estimated vehicle yaw [rad], wrapped, at fixes faster than ``min_speed_mps``.

    The only attitude yardstick this dataset offers (the baseline's attitude
    columns are not usable). It assumes body ``x`` is the vehicle's forward
    axis and that the car does not slip sideways, so expect a degree or two of
    scatter in turns even from a perfect filter.
    """
    data = gnss_arrays(list(gnss_samples))
    t_est = np.asarray(t_est, dtype=np.float64)
    yaw = np.unwrap(np.asarray(yaw_est_rad, dtype=np.float64))
    keep = (np.nan_to_num(data["speed_mps"]) >= min_speed_mps) & (data["time_s"] >= t_est[0]) & (data["time_s"] <= t_est[-1])
    keep &= np.isfinite(data["bearing_rad"])
    t = data["time_s"][keep]
    error = (data["bearing_rad"][keep] - np.interp(t, t_est, yaw) + np.pi) % (2.0 * np.pi) - np.pi
    return t, error


# ---------------------------------------------------------------------------
# GNSS outages
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Outage:
    """A scheduled GNSS outage: fixes with ``start_s < t <= end_s`` are withheld from the filter."""

    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def contains(self, time_s: float) -> bool:
        return self.start_s < time_s <= self.end_s


def schedule_outages(
    gnss_samples: Sequence[GNSSSample],
    durations_s: Sequence[float] = (10.0, 30.0, 60.0, 120.0),
    *,
    first_start_s: float,
    spacing_s: float = 400.0,
    min_speed_mps: float = 5.0,
    min_fix_fraction: float = 0.9,
) -> list[Outage]:
    """Deterministic outage windows, cycling through ``durations_s``.

    Candidate starts are searched from ``first_start_s`` (give the filter a
    few minutes to converge first) in 1 s steps. A window is accepted when the
    vehicle is moving at its start (GNSS speed >= ``min_speed_mps``) and at
    least ``min_fix_fraction`` of its seconds have a fix to score against.
    The next search starts ``spacing_s`` after the accepted start, so the
    filter reconverges between outages.
    """
    data = gnss_arrays(list(gnss_samples))
    t, speed = data["time_s"], np.nan_to_num(data["speed_mps"], nan=0.0)
    outages: list[Outage] = []
    if t.size == 0:
        return outages
    start = float(first_start_s)
    j = 0
    while start + max(durations_s) <= t[-1]:
        duration = float(durations_s[j % len(durations_s)])
        k = int(np.searchsorted(t, start))
        if k < t.size and abs(t[k] - start) < 0.5 and speed[k] >= min_speed_mps:
            inside = np.count_nonzero((t > t[k]) & (t <= t[k] + duration))
            if inside >= min_fix_fraction * duration:
                outages.append(Outage(float(t[k]), float(t[k] + duration)))
                j += 1
                start = float(t[k]) + spacing_s
                continue
        start += 1.0
    return outages


@dataclass
class OutageResult:
    """Errors against the withheld fixes of one outage."""

    outage: Outage
    elapsed_s: np.ndarray  # time since the outage started
    horizontal_error_m: np.ndarray
    horizontal_sigma_m: np.ndarray | None  # sqrt(P_NN + P_EE) of the estimate, if given

    def error_at(self, elapsed_s: float) -> float:
        """Horizontal error at the withheld fix closest to ``elapsed_s`` (NaN if none within 0.5 s)."""
        if self.elapsed_s.size == 0:
            return float("nan")
        i = int(np.argmin(np.abs(self.elapsed_s - elapsed_s)))
        return float(self.horizontal_error_m[i]) if abs(self.elapsed_s[i] - elapsed_s) <= 0.5 else float("nan")


def outage_errors(
    outages: Sequence[Outage],
    gnss_samples: Sequence[GNSSSample],
    t_est: np.ndarray,
    latitude_est: np.ndarray,
    longitude_est: np.ndarray,
    height_est: np.ndarray,
    horizontal_sigma_est: np.ndarray | None = None,
) -> list[OutageResult]:
    """Score an estimated track against the fixes withheld in each outage.

    ``horizontal_sigma_est`` (same length as ``t_est``) is the filter's
    ``sqrt(P_NN + P_EE)``; it is interpolated to the fix times so you can
    plot the error against its 3-sigma envelope.
    """
    data = gnss_arrays(list(gnss_samples))
    results = []
    for outage in outages:
        inside = (data["time_s"] > outage.start_s) & (data["time_s"] <= outage.end_s)
        t, error = position_error_ned(
            data["time_s"][inside], data["latitude_rad"][inside], data["longitude_rad"][inside], data["altitude_m"][inside],
            t_est, latitude_est, longitude_est, height_est,
        )
        sigma = None
        if horizontal_sigma_est is not None:
            sigma = np.interp(t, np.asarray(t_est, dtype=np.float64), np.asarray(horizontal_sigma_est, dtype=np.float64))
        results.append(OutageResult(outage, t - outage.start_s, np.hypot(error[:, 0], error[:, 1]), sigma))
    return results


def constant_velocity_outage_errors(outages: Sequence[Outage], gnss_samples: Sequence[GNSSSample]) -> list[OutageResult]:
    """The no-IMU baseline: extrapolate the last fix before each outage at constant velocity."""
    data = gnss_arrays(list(gnss_samples))
    results = []
    for outage in outages:
        before = np.flatnonzero(data["time_s"] <= outage.start_s)
        inside = np.flatnonzero((data["time_s"] > outage.start_s) & (data["time_s"] <= outage.end_s))
        if before.size == 0 or inside.size == 0:
            results.append(OutageResult(outage, np.zeros(0), np.zeros(0), None))
            continue
        last = gnss_samples[int(before[-1])]
        errors = []
        for k in inside:
            lat, lon, h = _extrapolate(last, data["time_s"][k] - last.time_s)
            e = geodetic_to_ned_delta(data["latitude_rad"][k], data["longitude_rad"][k], data["altitude_m"][k], lat, lon, h)
            errors.append(np.hypot(e[0], e[1]))
        results.append(OutageResult(outage, data["time_s"][inside] - outage.start_s, np.array(errors), None))
    return results


# ---------------------------------------------------------------------------
# Consistency
# ---------------------------------------------------------------------------


def chi2_interval(dof: int, probability: float = 0.95, num_averaged: int = 1) -> tuple[float, float]:
    """Two-sided chi-square acceptance interval for a NIS/NEES value or an average of them.

    For a single sample use ``num_averaged=1``; for the mean of ``K``
    independent samples the bounds are ``chi2.ppf(., K dof) / K``. Samples
    along one trajectory are correlated in time (GNSS errors certainly are),
    so treat averaged bounds as optimistic. A mean NIS far *above* ``dof``
    means the filter is overconfident; far below, underconfident.
    """
    tail = (1.0 - probability) / 2.0
    total = dof * num_averaged
    return float(chi2.ppf(tail, total) / num_averaged), float(chi2.ppf(1.0 - tail, total) / num_averaged)


def chi2_gate(dof: int, probability: float = 0.999) -> float:
    """One-sided chi-square gate, e.g. 16.27 for 3 dof at 99.9%."""
    return float(chi2.ppf(probability, dof))


def fraction_inside(values: np.ndarray, dof: int, probability: float = 0.95) -> float:
    """Fraction of NIS/NEES samples inside the single-sample two-sided chi-square interval."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    low, high = chi2_interval(dof, probability)
    return float(np.mean((values >= low) & (values <= high)))


def nees(errors: np.ndarray, covariances: np.ndarray) -> np.ndarray:
    """``e^T P^-1 e`` per sample for (N, n) errors and (N, n, n) covariances (synthetic tests)."""
    e = np.asarray(errors, dtype=np.float64)
    P = np.asarray(covariances, dtype=np.float64)
    return np.einsum("ni,ni->n", e, np.linalg.solve(P, e[..., None])[..., 0])
