"""Run every Project 3 stage you have implemented and write tables and plots.

From the repository root::

    uv run python -m ins_gnss.run_all                                   # all recommended trials
    uv run python -m ins_gnss.run_all --trial 2023-08-09_163741         # one trial (repeatable)
    uv run python -m ins_gnss.run_all --help

Stages, per trial:

1. Data (provided): rates, GNSS quality, the phone-to-vehicle mount rotation
   (:func:`~ins_gnss.datasets.estimate_mount_rotation`), and how much of the
   baseline agrees with GNSS (:func:`~ins_gnss.evaluation.reference_valid_mask`).
2. Part A: coarse alignment at departure (your ``coarse_alignment``), then
   60 s of free-inertial navigation (your ``mechanize``) scored against GNSS.
3. Part B: the ESKF over the whole trial with every GNSS fix (your ``F``,
   ``Q_d``, residual, update, injection, and reset inside
   :class:`~ins_gnss.eskf.INSGNSSFilter`), compared with GNSS-only against
   the valid reference, with NIS statistics.
4. Part C: the ESKF again with scheduled GNSS outages (10, 30, 60, 120 s),
   scored against the withheld fixes, next to a constant-velocity
   extrapolation, with the filter's 3-sigma envelope.

Stages that raise ``NotImplementedError`` are reported as "not implemented"
and skipped, so the script is useful from the first week. Outputs go to
``--out`` (default ``results/project3``): ``summary.md``/``.csv``,
``outages.md``/``.csv`` and PNG plots per trial. Every number in your report
should be reproducible from a command like the ones above.

Edit :data:`TUNING` to plug in your values; the defaults are a starting
point, not a tuned solution.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np

from . import eskf, strapdown
from .datasets import (
    TRIAL_NOTES,
    estimate_mount_rotation,
    gnss_acceleration_frd,
    gnss_arrays,
    imu_arrays,
    load_mems_nav_trial,
    recommended_trials,
    reference_arrays,
    repeated_fix_mask,
    rotate_imu_samples,
)
from .earth import geodetic_to_ned_delta
from .rotations import dcm_to_euler, quat_to_dcm
from .evaluation import (
    OutageResult,
    chi2_interval,
    constant_velocity_outage_errors,
    error_summary,
    fraction_inside,
    gnss_only_track,
    heading_error_vs_course,
    outage_errors,
    position_error_ned,
    reference_valid_mask,
    schedule_outages,
)
from .types import IMUNoise, IMUSample, MEMSNavTrial, NavState

# ---------------------------------------------------------------------------
# Tuning: replace these placeholders with values you can justify.
# ---------------------------------------------------------------------------

TUNING: dict[str, Any] = {
    # IMU noise (continuous-time; see IMUNoise and @sec-ins-errors-from-imu-specs).
    # At 1 Hz these must also absorb what a 1 Hz sample misses (vibration,
    # rate changes within the second), so expect values well above a datasheet's.
    "accel_vrw_mps_rt_s": 0.1,
    "gyro_arw_deg_rt_s": 0.1,
    "accel_bias_rw_mps2_rt_s": 1.0e-3,
    "gyro_bias_rw_deg_s_rt_s": 1.0e-2,
    # Initial one-sigma uncertainties.
    "init_position_m": (3.0, 3.0, 5.0),
    "init_velocity_mps": 0.5,
    "init_roll_pitch_deg": 2.0,
    "init_yaw_deg": 5.0,
    "init_accel_bias_mps2": 0.3,
    "init_gyro_bias_deg_s": 0.2,
    # GNSS measurement noise: R from the receiver's accuracy, times this scale.
    "gnss_r_scale": 1.0,
    "gnss_r_floor_m": 1.0,
    # Innovation gate (None = accept every fix; e.g. 16.27 = chi2(3 dof, 99.9%)).
    "gnss_gate": None,
    # Antenna position relative to the IMU, vehicle FRD axes [m]. Both are in
    # the phone here, a few centimeters apart, so zero is the honest value.
    "lever_arm_b_m": (0.0, 0.0, 0.0),
    # Extension: also fuse the horizontal GNSS velocity (speed and course).
    "use_velocity_aiding": False,
    # Alignment: start when the GNSS speed exceeds this for 5 s in a row.
    "departure_speed_mps": 5.0,
}

OUTAGE_DURATIONS_S = (10.0, 30.0, 60.0, 120.0)
OUTAGE_FIRST_START_S = 300.0  # after the filter starts
OUTAGE_SPACING_S = 400.0
FREE_INERTIAL_CHECK_S = (10.0, 30.0, 60.0)


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class StageLog:
    """Collects one line per stage so the end-of-run report is readable."""

    lines: list[str] = field(default_factory=list)

    def add(self, stage: str, status: str, detail: str = "") -> None:
        line = f"{stage:<34} {status}" + (f" - {detail}" if detail else "")
        self.lines.append(line)
        print(line, flush=True)


@dataclass
class PreparedTrial:
    """A trial with the IMU on vehicle axes and the departure (alignment) point chosen."""

    trial: MEMSNavTrial
    mount_rotation: np.ndarray  # C_p^v
    imu: list[IMUSample]  # vehicle FRD axes
    start_gnss_index: int  # the fix the filter starts from
    start_imu_index: int
    stop_imu_index: int  # last sample processed (exclusive bound is stop + 1)
    leveling_imu_indices: np.ndarray  # samples averaged for leveling
    leveling_correction: np.ndarray  # (N, 3) GNSS-derived acceleration removed from them (zero when parked)
    stale_fixes: np.ndarray


@dataclass
class FilterRun:
    """Output of :func:`run_filter`: one row per IMU sample after the start."""

    time_s: np.ndarray
    latitude_rad: np.ndarray
    longitude_rad: np.ndarray
    height_m: np.ndarray
    velocity_ned_mps: np.ndarray  # (N, 3)
    sigma: np.ndarray  # (N, 15) sqrt(diag(P))
    accel_bias_mps2: np.ndarray  # (N, 3)
    gyro_bias_rps: np.ndarray  # (N, 3)
    euler_rad: np.ndarray  # (N, 3) roll, pitch, yaw of the vehicle (display only)
    nis: np.ndarray  # GNSS position NIS of every update
    nis_time_s: np.ndarray
    runtime_s: float

    @property
    def horizontal_sigma_m(self) -> np.ndarray:
        return np.hypot(self.sigma[:, 0], self.sigma[:, 1])


def _first_line(error: BaseException) -> str:
    text = str(error).strip().splitlines()
    return f"{type(error).__name__}: {text[0]}" if text else type(error).__name__


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def prepare_trial(trial: MEMSNavTrial, tuning: dict[str, Any]) -> PreparedTrial:
    """Rotate the IMU to vehicle axes and choose where the run starts and stops.

    **Start (departure).** The filter starts at the first GNSS fix after which
    the speed stays above ``departure_speed_mps`` for 5 s: there the course
    over ground is a usable heading. **Leveling** uses the samples of the
    preceding minute at which the GNSS speed is below 0.3 m/s and the gyro
    is quiet (the car waiting to leave, the phone untouched). If there are fewer than five (the log starts on the
    move), it uses the first 10 s after departure with the GNSS-derived
    forward and rightward acceleration subtracted, which is cruder.
    **Stop.** The run ends 30 s after the last fix faster than
    ``departure_speed_mps``: in some trials the phone is taken out of its
    mount once the car is parked, and a 1 Hz IMU cannot follow a phone in a
    hand.
    """
    mount = estimate_mount_rotation(trial.imu_samples, trial.gnss_samples)
    imu = rotate_imu_samples(trial.imu_samples, mount)
    t_imu, _, _ = imu_arrays(imu)
    gnss = gnss_arrays(trial.gnss_samples)
    speed = np.nan_to_num(gnss["speed_mps"], nan=0.0)
    threshold = float(tuning["departure_speed_mps"])
    start = next((k for k in range(len(speed) - 5) if np.all(speed[k : k + 5] > threshold)), None)
    if start is None:
        raise ValueError(f"{trial.name}: the vehicle never moves faster than {threshold} m/s for 5 s")
    t0 = gnss["time_s"][start]
    start_imu = int(np.searchsorted(t_imu, t0))
    if start_imu >= len(t_imu) or abs(t_imu[start_imu] - t0) > 0.5:
        raise ValueError(f"{trial.name}: no IMU sample at the departure time {t0:.1f} s")
    t_stop = gnss["time_s"][np.flatnonzero(speed > threshold)[-1]] + 30.0
    stop_imu = int(min(np.searchsorted(t_imu, t_stop, side="right") - 1, len(t_imu) - 1))

    fix_speed = {round(t): s for t, s in zip(gnss["time_s"], speed)}
    window = np.flatnonzero((t_imu < t0) & (t_imu >= t0 - 60.0))
    quiet = [i for i in window if np.linalg.norm(imu[i].gyro_rps) < np.radians(2.0)]
    leveling = np.array([i for i in quiet if fix_speed.get(round(t_imu[i]), np.inf) < 0.3], dtype=int)
    correction = np.zeros((leveling.size, 3))
    if leveling.size < 5:
        t_acc, a_forward, a_right = gnss_acceleration_frd(trial.gnss_samples)
        by_time = {round(t): (f, r) for t, f, r in zip(t_acc, a_forward, a_right) if np.isfinite(f) and np.isfinite(r)}
        leveling = np.array([i for i in range(start_imu, len(t_imu)) if t_imu[i] <= t0 + 10.0 and round(t_imu[i]) in by_time], dtype=int)
        correction = np.array([[*by_time[round(t_imu[i])], 0.0] for i in leveling]).reshape(-1, 3)
    if leveling.size == 0:
        leveling, correction = np.array([start_imu]), np.zeros((1, 3))
    return PreparedTrial(trial, mount, imu, start, start_imu, stop_imu, leveling, correction, repeated_fix_mask(trial.gnss_samples))


def initial_state(prepared: PreparedTrial) -> NavState:
    """Coarse alignment (your ``coarse_alignment``) at the departure fix."""
    fix = prepared.trial.gnss_samples[prepared.start_gnss_index]
    _, accel, _ = imu_arrays([prepared.imu[i] for i in prepared.leveling_imu_indices])
    q_bn = strapdown.coarse_alignment((accel - prepared.leveling_correction).mean(axis=0), float(fix.bearing_rad))
    v_ne = fix.velocity_ne_mps()
    velocity = np.zeros(3) if v_ne is None else np.array([v_ne[0], v_ne[1], 0.0])
    return NavState(fix.latitude_rad, fix.longitude_rad, fix.altitude_m, velocity, q_bn)


def make_noise(tuning: dict[str, Any]) -> IMUNoise:
    return IMUNoise(
        accel_vrw=float(tuning["accel_vrw_mps_rt_s"]),
        gyro_arw=math.radians(float(tuning["gyro_arw_deg_rt_s"])),
        accel_bias_rw=float(tuning["accel_bias_rw_mps2_rt_s"]),
        gyro_bias_rw=math.radians(float(tuning["gyro_bias_rw_deg_s_rt_s"])),
    )


def initial_covariance(tuning: dict[str, Any]) -> np.ndarray:
    sigmas = np.concatenate(
        [
            np.asarray(tuning["init_position_m"], dtype=np.float64),
            np.full(3, float(tuning["init_velocity_mps"])),
            np.radians([tuning["init_roll_pitch_deg"], tuning["init_roll_pitch_deg"], tuning["init_yaw_deg"]]),
            np.full(3, float(tuning["init_accel_bias_mps2"])),
            np.full(3, math.radians(float(tuning["init_gyro_bias_deg_s"]))),
        ]
    )
    return np.diag(sigmas**2)


def make_filter(prepared: PreparedTrial, tuning: dict[str, Any]) -> eskf.INSGNSSFilter:
    """Build the ESKF at the departure point. Edit freely."""
    state = initial_state(prepared)
    return eskf.INSGNSSFilter(
        state=state,
        P=initial_covariance(tuning),
        noise=make_noise(tuning),
        lever_arm_b=np.asarray(tuning["lever_arm_b_m"], dtype=np.float64),
        time_s=prepared.imu[prepared.start_imu_index].time_s,
    )


def run_filter(prepared: PreparedTrial, tuning: dict[str, Any], outages: list | None = None) -> FilterRun:
    """Drive the ESKF through the trial; withhold the fixes inside ``outages``.

    Each IMU interval is propagated with the average of its two bounding
    samples; then the GNSS fix stamped at the new sample time (if any, and not
    stale, and not withheld) is fused.
    """
    flt = make_filter(prepared, tuning)
    gnss = prepared.trial.gnss_samples
    fix_at = {round(sample.time_s): k for k, sample in enumerate(gnss)}
    gate = tuning.get("gnss_gate")
    outages = outages or []
    r_scale, r_floor = float(tuning["gnss_r_scale"]), float(tuning["gnss_r_floor_m"])
    rows: dict[str, list] = {key: [] for key in ("t", "lat", "lon", "h", "v", "sigma", "ba", "bg", "euler")}
    imu = prepared.imu
    tic = time.perf_counter()
    for i in range(prepared.start_imu_index + 1, prepared.stop_imu_index + 1):
        previous, sample = imu[i - 1], imu[i]
        dt = sample.time_s - previous.time_s
        gyro = 0.5 * (previous.gyro_rps + sample.gyro_rps)
        flt.predict(0.5 * (previous.accel_mps2 + sample.accel_mps2), gyro, dt)
        k = fix_at.get(round(sample.time_s))
        if k is not None and not prepared.stale_fixes[k] and not any(o.contains(gnss[k].time_s) for o in outages):
            R = eskf.gnss_position_covariance(gnss[k], scale=r_scale, floor_m=r_floor)
            flt.update_gnss_position(gnss[k], R, gate=gate)
            if tuning.get("use_velocity_aiding") and (gnss[k].speed_mps or 0.0) > 3.0:
                flt.update_gnss_velocity(gnss[k], gyro, gate=None)
        s = flt.state
        rows["t"].append(sample.time_s)
        rows["lat"].append(s.latitude_rad)
        rows["lon"].append(s.longitude_rad)
        rows["h"].append(s.height_m)
        rows["v"].append(s.velocity_ned_mps.copy())
        rows["sigma"].append(np.sqrt(np.clip(np.diag(flt.P), 0.0, None)))
        rows["ba"].append(s.accel_bias_mps2.copy())
        rows["bg"].append(s.gyro_bias_rps.copy())
        rows["euler"].append(dcm_to_euler(quat_to_dcm(s.q_bn)))
    runtime = time.perf_counter() - tic
    records = [r for r in flt.innovations if r.kind == "gnss_position"]
    return FilterRun(
        time_s=np.array(rows["t"]),
        latitude_rad=np.array(rows["lat"]),
        longitude_rad=np.array(rows["lon"]),
        height_m=np.array(rows["h"]),
        velocity_ned_mps=np.array(rows["v"]).reshape(-1, 3),
        sigma=np.array(rows["sigma"]).reshape(-1, 15),
        accel_bias_mps2=np.array(rows["ba"]).reshape(-1, 3),
        gyro_bias_rps=np.array(rows["bg"]).reshape(-1, 3),
        euler_rad=np.array(rows["euler"]).reshape(-1, 3),
        nis=np.array([r.nis for r in records]),
        nis_time_s=np.array([r.time_s for r in records]),
        runtime_s=runtime,
    )


def free_inertial_check(prepared: PreparedTrial, state: NavState) -> dict[str, float]:
    """Part A: free-inertial navigation from the alignment, scored against GNSS after 10, 30 and 60 s."""
    horizon = max(FREE_INERTIAL_CHECK_S)
    t0 = prepared.imu[prepared.start_imu_index].time_s
    samples = [s for s in prepared.imu[prepared.start_imu_index :] if s.time_s <= t0 + horizon + 0.5]
    times, states = strapdown.free_inertial(state, samples)
    gnss = gnss_arrays(prepared.trial.gnss_samples)
    t, error = position_error_ned(
        gnss["time_s"], gnss["latitude_rad"], gnss["longitude_rad"], gnss["altitude_m"],
        times, np.array([s.latitude_rad for s in states]), np.array([s.longitude_rad for s in states]),
        np.array([s.height_m for s in states]),
    )
    out = {}
    for check in FREE_INERTIAL_CHECK_S:
        i = np.flatnonzero(np.isclose(t - t0, check, atol=0.5))
        out[f"free_inertial_{int(check)}s_m"] = float(np.hypot(*error[i[0], :2])) if i.size else float("nan")
    return out


def stationary_check(prepared: PreparedTrial, duration_s: float = 60.0) -> dict[str, float] | None:
    """Part A on real data: free-inertial navigation of the parked car.

    Finds the longest stretch of at least ``duration_s + 10`` s with GNSS
    speed below 0.3 m/s and a quiet gyro (phone untouched), levels with the
    first 10 s (heading is irrelevant here), and mechanizes ``duration_s``
    seconds from zero velocity. The true position does not change, so the
    position drift is the navigation error. Returns ``None`` if the trial has
    no such stretch.
    """
    t_imu, accel, gyro = imu_arrays(prepared.imu)
    gnss = gnss_arrays(prepared.trial.gnss_samples)
    fix_speed = {round(t): s for t, s in zip(gnss["time_s"], np.nan_to_num(gnss["speed_mps"], nan=np.inf))}
    quiet = np.array([fix_speed.get(round(t), np.inf) < 0.3 for t in t_imu]) & (np.linalg.norm(gyro, axis=1) < np.radians(2.0))
    best: tuple[int, int] | None = None
    i = 0
    while i < len(quiet):
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(quiet) and quiet[j + 1] and t_imu[j + 1] - t_imu[j] < 1.5:
            j += 1
        if t_imu[j] - t_imu[i] >= duration_s + 10.0 and (best is None or j - i > best[1] - best[0]):
            best = (i, j)
        i = j + 1
    if best is None:
        return None
    i0 = best[0]
    level = [k for k in range(i0, best[1] + 1) if t_imu[k] < t_imu[i0] + 10.0]
    k_fix = int(np.argmin(np.abs(gnss["time_s"] - t_imu[level[-1]])))
    start = level[-1]
    state = NavState(
        gnss["latitude_rad"][k_fix], gnss["longitude_rad"][k_fix], gnss["altitude_m"][k_fix], np.zeros(3),
        strapdown.coarse_alignment(accel[level].mean(axis=0), 0.0),
    )
    samples = [prepared.imu[k] for k in range(start, best[1] + 1) if t_imu[k] <= t_imu[start] + duration_s + 0.5]
    times, states = strapdown.free_inertial(state, samples)
    out = {"stationary_start_s": float(t_imu[start])}
    for check in FREE_INERTIAL_CHECK_S:
        k = int(np.argmin(np.abs(times - times[0] - check)))
        drift = geodetic_to_ned_delta(states[k].latitude_rad, states[k].longitude_rad, states[k].height_m,
                                      state.latitude_rad, state.longitude_rad, state.height_m)
        out[f"stationary_{int(check)}s_m"] = float(np.hypot(drift[0], drift[1]))
    return out


def evaluate_against_reference(trial: MEMSNavTrial, run: FilterRun) -> dict[str, Any]:
    """Horizontal/vertical errors of the ESKF and of GNSS-only against the valid reference."""
    ref = reference_arrays(trial.reference_states)
    valid = reference_valid_mask(trial)
    inside = valid & (ref["time_s"] >= run.time_s[0]) & (ref["time_s"] <= run.time_s[-1])
    t_ref = ref["time_s"][inside]
    _, e_ins = position_error_ned(
        t_ref, ref["latitude_rad"][inside], ref["longitude_rad"][inside], ref["altitude_m"][inside],
        run.time_s, run.latitude_rad, run.longitude_rad, run.height_m,
    )
    lat_g, lon_g, h_g = gnss_only_track(trial.gnss_samples, t_ref)
    e_gnss = geodetic_to_ned_delta(
        ref["latitude_rad"][inside], ref["longitude_rad"][inside], ref["altitude_m"][inside], lat_g, lon_g, h_g
    ).reshape(-1, 3)
    _, heading = heading_error_vs_course(run.time_s, run.euler_rad[:, 2], trial.gnss_samples)
    return {
        "ins_gnss": error_summary(e_ins),
        "gnss_only": error_summary(e_gnss),
        "heading_error_deg": np.degrees(heading),
        "reference_valid_fraction": float(np.mean(valid)),
        "t_ref": t_ref,
        "error_ins": e_ins,
        "error_gnss": e_gnss,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

SUMMARY_COLUMNS = (
    "trial",
    "method",
    "horizontal_rmse_m",
    "horizontal_median_m",
    "horizontal_p95_m",
    "horizontal_max_m",
    "vertical_rmse_m",
    "num_samples",
    "mean_nis",
    "nis_inside_95",
    "heading_mad_deg",
    "runtime_s",
)

OUTAGE_COLUMNS = ("trial", "duration_s", "num_outages", "elapsed_s", "ins_median_m", "ins_max_m", "cv_median_m", "sigma3_median_m", "inside_3sigma")


def _format(value: Any) -> str:
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.4g}"
    return str(value)


def write_tables(rows: list[dict[str, Any]], out_dir: Path, stem: str, columns: tuple[str, ...]) -> None:
    """Write ``rows`` as ``<stem>.csv`` and a Markdown table ``<stem>.md``."""
    with (out_dir / f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format(row.get(key, "")) for key in columns})
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(_format(row.get(key, "")) for key in columns) + " |" for row in rows]
    (out_dir / f"{stem}.md").write_text("\n".join(lines) + "\n")


def outage_rows(name: str, ins: list[OutageResult], cv: list[OutageResult]) -> list[dict[str, Any]]:
    """Median errors at 10/30/60/120 s into each outage length."""
    rows = []
    for duration in OUTAGE_DURATIONS_S:
        pairs = [(a, b) for a, b in zip(ins, cv) if math.isclose(a.outage.duration_s, duration)]
        if not pairs:
            continue
        for elapsed in [e for e in OUTAGE_DURATIONS_S if e <= duration]:
            e_ins = np.array([a.error_at(elapsed) for a, _ in pairs])
            e_cv = np.array([b.error_at(elapsed) for _, b in pairs])
            sig = np.array(
                [a.horizontal_sigma_m[np.argmin(np.abs(a.elapsed_s - elapsed))] if a.horizontal_sigma_m is not None and a.elapsed_s.size else np.nan for a, _ in pairs]
            )
            ok = np.isfinite(e_ins)
            rows.append(
                {
                    "trial": name,
                    "duration_s": duration,
                    "num_outages": len(pairs),
                    "elapsed_s": elapsed,
                    "ins_median_m": float(np.nanmedian(e_ins)) if ok.any() else float("nan"),
                    "ins_max_m": float(np.nanmax(e_ins)) if ok.any() else float("nan"),
                    "cv_median_m": float(np.nanmedian(e_cv)) if np.isfinite(e_cv).any() else float("nan"),
                    "sigma3_median_m": 3.0 * float(np.nanmedian(sig)) if np.isfinite(sig).any() else float("nan"),
                    "inside_3sigma": float(np.mean(e_ins[ok] <= 3.0 * sig[ok])) if ok.any() and np.isfinite(sig).any() else float("nan"),
                }
            )
    return rows


def _figure(figsize: tuple[float, float]):
    from matplotlib.figure import Figure

    return Figure(figsize=figsize, layout="constrained")


def plot_trial(out_path: Path, trial: MEMSNavTrial, run: FilterRun, evaluation: dict[str, Any]) -> None:
    """Track, error versus the reference, sigmas, biases and NIS for the full-GNSS run."""
    fig = _figure((11, 12))
    axes = fig.subplots(3, 2)
    lat0, lon0, h0 = run.latitude_rad[0], run.longitude_rad[0], run.height_m[0]
    track = geodetic_to_ned_delta(run.latitude_rad, run.longitude_rad, run.height_m, lat0, lon0, h0)
    gnss = gnss_arrays(trial.gnss_samples)
    fixes = geodetic_to_ned_delta(gnss["latitude_rad"], gnss["longitude_rad"], gnss["altitude_m"], lat0, lon0, h0)
    ax = axes[0, 0]
    ax.plot(fixes[:, 1] / 1000, fixes[:, 0] / 1000, ".", ms=1.5, color="0.6", label="GNSS fixes")
    ax.plot(track[:, 1] / 1000, track[:, 0] / 1000, lw=0.8, label="INS/GNSS")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("east [km]")
    ax.set_ylabel("north [km]")
    ax.set_title(trial.name)
    ax.legend(loc="best", fontsize="small")
    ax = axes[0, 1]
    t_ref = evaluation["t_ref"]
    ax.plot(t_ref, np.hypot(*evaluation["error_gnss"][:, :2].T), ".", ms=1, color="0.6", label="GNSS only")
    ax.plot(t_ref, np.hypot(*evaluation["error_ins"][:, :2].T), ".", ms=1, label="INS/GNSS")
    ax.set_yscale("log")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("horizontal error vs reference [m]")
    ax.legend(loc="best", fontsize="small")
    ax = axes[1, 0]
    for j, label in enumerate(("N", "E", "D")):
        ax.plot(run.time_s, run.sigma[:, j], lw=0.8, label=f"sigma p_{label}")
    ax.set_yscale("log")
    ax.set_ylabel("position sigma [m]")
    ax.legend(loc="best", fontsize="small")
    ax = axes[1, 1]
    for j, label in enumerate(("roll", "pitch", "yaw")):
        ax.plot(run.time_s, np.degrees(run.sigma[:, 6 + j]), lw=0.8, label=f"sigma {label}")
    ax.set_yscale("log")
    ax.set_ylabel("attitude sigma [deg]")
    ax.legend(loc="best", fontsize="small")
    ax = axes[2, 0]
    ax.plot(run.time_s, run.accel_bias_mps2, lw=0.8)
    ax2 = ax.twinx()
    ax2.plot(run.time_s, np.degrees(run.gyro_bias_rps) * 3600.0, lw=0.8, ls="--")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("accel bias [m/s^2] (solid)")
    ax2.set_ylabel("gyro bias [deg/h] (dashed)")
    ax = axes[2, 1]
    if run.nis.size:
        low, high = chi2_interval(3, 0.95)
        ax.plot(run.nis_time_s, run.nis, ".", ms=1)
        ax.axhline(3.0, color="k", lw=1, label="dof = 3")
        ax.axhline(low, color="C3", ls=":", lw=1, label="95% interval")
        ax.axhline(high, color="C3", ls=":", lw=1)
        ax.set_yscale("log")
        ax.legend(loc="best", fontsize="small")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("GNSS position NIS")
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
    fig.savefig(out_path, dpi=110)


def plot_outages(out_path: Path, name: str, ins: list[OutageResult], cv: list[OutageResult]) -> None:
    """Error against withheld fixes versus time into each outage, with the filter's 3 sigma."""
    fig = _figure((11, 8))
    axes = fig.subplots(2, 2).flat
    for axis, duration in zip(axes, OUTAGE_DURATIONS_S):
        for k, (a, b) in enumerate((a, b) for a, b in zip(ins, cv) if math.isclose(a.outage.duration_s, duration)):
            axis.plot(a.elapsed_s, a.horizontal_error_m, color="C0", lw=1, label="INS/GNSS" if k == 0 else None)
            if a.horizontal_sigma_m is not None:
                axis.plot(a.elapsed_s, 3.0 * a.horizontal_sigma_m, color="C0", lw=0.8, ls=":", label="3 sigma" if k == 0 else None)
            axis.plot(b.elapsed_s, b.horizontal_error_m, color="C1", lw=0.8, alpha=0.7, label="constant velocity" if k == 0 else None)
        axis.set_yscale("log")
        axis.set_title(f"{name}: {duration:.0f} s outages")
        axis.set_xlabel("time since outage start [s]")
        axis.set_ylabel("horizontal error vs withheld fix [m]")
        axis.grid(True, alpha=0.25)
        if axis.lines:
            axis.legend(loc="best", fontsize="small")
    fig.savefig(out_path, dpi=110)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def process_trial(name: str, out_dir: Path, log: StageLog, tuning: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """All stages for one trial; returns (summary rows, outage rows)."""
    summary: list[dict[str, Any]] = []
    outage_table: list[dict[str, Any]] = []
    trial = load_mems_nav_trial(None, name)
    prepared = prepare_trial(trial, tuning)
    gnss = gnss_arrays(trial.gnss_samples)
    forward = prepared.mount_rotation[0]
    log.add(
        f"[{name}] data",
        "ok",
        f"{len(trial.imu_samples)} IMU / {len(trial.gnss_samples)} GNSS at 1 Hz, "
        f"{int(prepared.stale_fixes.sum())} stale fixes, median horizontal accuracy "
        f"{np.nanmedian(gnss['horizontal_accuracy_m']):.1f} m, vehicle forward = phone "
        f"[{forward[0]:+.2f}, {forward[1]:+.2f}, {forward[2]:+.2f}], "
        f"reference valid {100 * np.mean(reference_valid_mask(trial)):.0f}%",
    )

    try:
        state0 = initial_state(prepared)
        rpy = np.degrees(dcm_to_euler(quat_to_dcm(state0.q_bn)))
        log.add(f"[{name}] Part A alignment", "ok", f"start {prepared.imu[prepared.start_imu_index].time_s:.0f} s, "
                f"roll {rpy[0]:+.1f}, pitch {rpy[1]:+.1f}, yaw {rpy[2]:+.1f} deg "
                f"({len(prepared.leveling_imu_indices)} leveling samples"
                f"{', in motion' if np.any(prepared.leveling_correction) else ''})")
    except NotImplementedError as error:
        log.add(f"[{name}] Part A alignment", "not implemented", _first_line(error))
        return summary, outage_table

    try:
        free = free_inertial_check(prepared, state0)
        log.add(f"[{name}] Part A free inertial", "ok", "after departure, error vs GNSS at "
                + ", ".join(f"{k.split('_')[2]}: {v:.0f} m" for k, v in free.items()))
        parked = stationary_check(prepared)
        if parked is None:
            log.add(f"[{name}] Part A parked check", "skipped", "no quiet parked stretch of 70 s")
        else:
            log.add(f"[{name}] Part A parked check", "ok", f"from {parked['stationary_start_s']:.0f} s, drift at "
                    + ", ".join(f"{k.split('_')[1]}: {v:.1f} m" for k, v in parked.items() if k.startswith("stationary_") and k.endswith("s_m")))
    except NotImplementedError as error:
        log.add(f"[{name}] Part A free inertial", "not implemented", _first_line(error))
        return summary, outage_table

    try:
        run = run_filter(prepared, tuning)
    except NotImplementedError as error:
        log.add(f"[{name}] Part B ESKF", "not implemented", _first_line(error))
        return summary, outage_table
    evaluation = evaluate_against_reference(trial, run)
    low, high = chi2_interval(3, 0.95)
    mean_nis = float(np.mean(run.nis)) if run.nis.size else float("nan")
    inside = fraction_inside(run.nis, 3, 0.95)
    for method, key in (("GNSS only", "gnss_only"), ("INS/GNSS", "ins_gnss")):
        row = {"trial": name, "method": method, **evaluation[key]}
        if key == "ins_gnss":
            heading = evaluation["heading_error_deg"]
            heading_mad = float(np.median(np.abs(heading - np.median(heading)))) if heading.size else float("nan")
            row.update({"mean_nis": mean_nis, "nis_inside_95": inside, "heading_mad_deg": heading_mad, "runtime_s": run.runtime_s})
        summary.append(row)
    log.add(
        f"[{name}] Part B ESKF",
        "ok",
        f"horizontal RMSE vs valid reference {evaluation['ins_gnss']['horizontal_rmse_m']:.2f} m "
        f"(GNSS only {evaluation['gnss_only']['horizontal_rmse_m']:.2f} m), mean NIS {mean_nis:.2f} "
        f"(dof 3; {100 * inside:.0f}% inside [{low:.2f}, {high:.2f}]), yaw vs course median "
        f"{np.median(evaluation['heading_error_deg']):+.1f} deg, {run.runtime_s:.1f} s",
    )
    plot_trial(out_dir / f"{name}_ins_gnss.png", trial, run, evaluation)

    outages = schedule_outages(
        trial.gnss_samples, OUTAGE_DURATIONS_S, first_start_s=run.time_s[0] + OUTAGE_FIRST_START_S, spacing_s=OUTAGE_SPACING_S
    )
    if not outages:
        log.add(f"[{name}] Part C outages", "skipped", "no suitable outage windows")
        return summary, outage_table
    run_out = run_filter(prepared, tuning, outages)
    ins = outage_errors(outages, trial.gnss_samples, run_out.time_s, run_out.latitude_rad, run_out.longitude_rad, run_out.height_m, run_out.horizontal_sigma_m)
    cv = constant_velocity_outage_errors(outages, trial.gnss_samples)
    outage_table = outage_rows(name, ins, cv)
    ends = [r for r in outage_table if r["elapsed_s"] == r["duration_s"]]
    log.add(
        f"[{name}] Part C outages",
        "ok",
        f"{len(outages)} outages; median error at the end: "
        + ", ".join(f"{r['duration_s']:.0f} s: {r['ins_median_m']:.0f} m (CV {r['cv_median_m']:.0f} m)" for r in ends),
    )
    plot_outages(out_dir / f"{name}_outages.png", name, ins, cv)
    return summary, outage_table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trial", action="append", default=None, help="trial name (repeatable); default: all recommended trials")
    parser.add_argument("--out", type=Path, default=Path("results") / "project3", help="output folder")
    args = parser.parse_args(argv)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    log = StageLog()
    names = args.trial or recommended_trials()
    if not names:
        print("No trials found. Download the data with: uv run python scripts/fetch_data.py mems-nav-dataset")
        return 1
    summary: list[dict[str, Any]] = []
    outages: list[dict[str, Any]] = []
    for name in names:
        note = TRIAL_NOTES.get(name)
        if note is not None and note.status != "recommended":
            print(f"note: {name} is '{note.status}': {note.reason}")
        try:
            rows, outage_table = process_trial(name, out_dir, log, TUNING)
        except Exception as error:  # keep going: report the failure and move on to the next trial
            log.add(f"[{name}]", "error", _first_line(error))
            traceback.print_exc()
            continue
        summary.extend(rows)
        outages.extend(outage_table)
    write_tables(summary, out_dir, "summary", SUMMARY_COLUMNS)
    write_tables(outages, out_dir, "outages", OUTAGE_COLUMNS)
    print(f"\nWrote tables and plots to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
