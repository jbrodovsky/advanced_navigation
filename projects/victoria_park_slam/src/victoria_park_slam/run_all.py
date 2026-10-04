"""Run every Project 2 stage you have implemented and write tables and plots.

From the repository root::

    uv run python -m victoria_park_slam.run_all                     # first 2000 scans
    uv run python -m victoria_park_slam.run_all --sweep particles gate q_scale
    uv run python -m victoria_park_slam.run_all --all               # the whole run (slow)
    uv run python -m victoria_park_slam.run_all --help

Stages:

1. Dead reckoning (provided): the four variants of
   :data:`~victoria_park_slam.evaluation.DR_VARIANTS`, evaluated against GPS
   over the same segment.
2. Feature extraction: your ``extract_tree_landmarks`` on every scan.
3. FastSLAM: :func:`make_slam` builds your ``FastSLAM`` from :data:`TUNING`
   and :func:`run_fastslam` drives it with the event-driven time
   synchronization of :func:`~victoria_park_slam.timing.scan_steps`.
4. Sweeps (``--sweep``): FastSLAM again for each value in :data:`SWEEPS`.

Stages that raise ``NotImplementedError`` are reported as "not implemented"
and skipped, so the script is useful from the first week. Outputs go to
``--out`` (default ``results/project2``): ``summary.md``/``.csv``,
``sweep_<name>.md``/``.csv`` and PNG plots. Every number in your report
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

from . import fastslam, landmarks
from .datasets import load_victoria_park_data
from .evaluation import (
    TrajectoryEvaluation,
    apply_se2,
    dead_reckoning_comparison,
    evaluate_against_gps,
    valid_gps_mask,
)
from .timing import scan_steps
from .types import MeasurementModelConfig, MotionModelConfig, TreeObservation, VictoriaParkDataset

# ---------------------------------------------------------------------------
# Tuning: replace these placeholders with values you can justify.
# ---------------------------------------------------------------------------

TUNING: dict[str, Any] = {
    "num_particles": 50,
    "seed": 0,
    # Control noise Q_u, one-sigma per 25 ms prediction step:
    # [encoder speed (m/s), steering (deg)]. Independent at 40 Hz, so think in
    # terms of the spread it produces per second, not per sample.
    "control_std": [0.5, 3.0],
    "range_std_m": 0.5,
    "bearing_std_deg": 2.0,
    "gate_threshold": 5.99,  # chi2(2 dof, 95%)
    "birth_threshold": 13.82,  # chi2(2 dof, 99.9%)
    "new_landmark_log_likelihood": None,  # log p0; None = FastSLAM.log_p0 default
    "resample_threshold": 0.5,  # resample when N_eff < 0.5 N
    "confirm_associations": 3,
    "tentative_timeout_s": 2.0,
    "use_encoder_correction": True,
    "q_scale": 1.0,  # multiplies Q_u (used by the q_scale sweep)
}

SEGMENT_SCANS = 2000

# Required sensitivity sweeps: name -> (TUNING key, values).
SWEEPS: dict[str, tuple[str, tuple[float, ...]]] = {
    "particles": ("num_particles", (10, 25, 50, 100)),
    "gate": ("gate_threshold", (4.61, 5.99, 9.21)),
    "q_scale": ("q_scale", (0.25, 1.0, 4.0)),
}


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class StageLog:
    """Collects one line per stage so the end-of-run report is readable."""

    lines: list[str] = field(default_factory=list)

    def add(self, stage: str, status: str, detail: str = "") -> None:
        line = f"{stage:<32} {status}" + (f" - {detail}" if detail else "")
        self.lines.append(line)
        print(line, flush=True)


@dataclass
class SlamRun:
    """Per-scan record of one FastSLAM run (output of :func:`run_fastslam`)."""

    scan_index: np.ndarray
    time_s: np.ndarray
    estimate: np.ndarray  # (K, 3) weighted-mean laser pose after each update
    n_eff: np.ndarray
    landmarks_total: np.ndarray
    landmarks_confirmed: np.ndarray
    runtime_s: np.ndarray  # predict + update wall time per scan
    best_trajectory: np.ndarray | None
    final_map: np.ndarray
    history: list[dict[str, float]]


def _first_line(error: BaseException) -> str:
    text = str(error).strip().splitlines()
    return f"{type(error).__name__}: {text[0]}" if text else type(error).__name__


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def make_slam(tuning: dict[str, Any]) -> fastslam.FastSLAM:
    """Build the FastSLAM filter described by ``tuning`` at ``(0, 0, 0)``. Edit freely."""

    speed_std, steering_std_deg = tuning["control_std"]
    q_u = float(tuning.get("q_scale", 1.0)) * np.diag([speed_std**2, math.radians(steering_std_deg) ** 2])
    motion = MotionModelConfig(control_noise_covariance=q_u, use_encoder_correction=bool(tuning["use_encoder_correction"]))
    measurement = MeasurementModelConfig(
        range_std_m=float(tuning["range_std_m"]), bearing_std_rad=math.radians(float(tuning["bearing_std_deg"]))
    )
    return fastslam.FastSLAM.from_initial_pose(
        int(tuning["num_particles"]),
        (0.0, 0.0, 0.0),
        motion=motion,
        measurement=measurement,
        gate_threshold=float(tuning["gate_threshold"]),
        birth_threshold=float(tuning["birth_threshold"]),
        new_landmark_log_likelihood=tuning["new_landmark_log_likelihood"],
        resample_threshold=float(tuning["resample_threshold"]),
        confirm_associations=int(tuning["confirm_associations"]),
        tentative_timeout_s=float(tuning["tentative_timeout_s"]),
        rng=np.random.default_rng(int(tuning["seed"])),
    )


def extract_all(dataset: VictoriaParkDataset, stop_scan: int) -> tuple[list[list[TreeObservation]], float]:
    """Run ``extract_tree_landmarks`` on scans ``[0, stop_scan)``; return detections and seconds."""

    tic = time.perf_counter()
    detections = [landmarks.extract_tree_landmarks(dataset.laser_ranges_m[i]) for i in range(stop_scan)]
    return detections, time.perf_counter() - tic


def run_fastslam(
    slam: fastslam.FastSLAM,
    dataset: VictoriaParkDataset,
    detections: list[list[TreeObservation]],
    stop_scan: int,
    progress_every: int = 0,
) -> SlamRun:
    """Drive ``slam`` through scans ``[0, stop_scan)`` with event-driven synchronization.

    For every scan: one ``predict`` per odometry piece from
    :func:`~victoria_park_slam.timing.scan_steps` (the last odometry interval
    is split at the laser timestamp), then ``update`` with that scan's
    detections. Records are taken after each update.
    """

    records: dict[str, list] = {key: [] for key in ("scan", "time", "est", "neff", "total", "confirmed", "runtime")}
    history_length = len(slam.history)
    for step in scan_steps(dataset, 0, stop_scan):
        tic = time.perf_counter()
        for control in step.controls:
            slam.predict(control.speed_mps, control.steering_rad, control.dt_s)
        slam.update(detections[step.scan_index], step.time_s)
        records["runtime"].append(time.perf_counter() - tic)
        records["scan"].append(step.scan_index)
        records["time"].append(step.time_s)
        records["est"].append(slam.estimate_pose())
        latest = slam.history[-1] if len(slam.history) > history_length else {}
        history_length = len(slam.history)
        records["neff"].append(float(latest.get("n_eff", slam.effective_sample_size())))
        total, confirmed = slam.landmark_counts()
        records["total"].append(total)
        records["confirmed"].append(confirmed)
        if progress_every and step.scan_index % progress_every == 0:
            print(f"  scan {step.scan_index:5d}  t={step.time_s:7.1f} s  landmarks {total:4d}  "
                  f"N_eff {records['neff'][-1]:6.1f}", flush=True)

    best = slam.best_particle()
    best_trajectory = np.array(best.trajectory, dtype=np.float64) if best.trajectory else None
    return SlamRun(
        scan_index=np.array(records["scan"]),
        time_s=np.array(records["time"]),
        estimate=np.array(records["est"]).reshape(-1, 3),
        n_eff=np.array(records["neff"]),
        landmarks_total=np.array(records["total"]),
        landmarks_confirmed=np.array(records["confirmed"]),
        runtime_s=np.array(records["runtime"]),
        best_trajectory=best_trajectory,
        final_map=slam.landmark_means(best, confirmed_only=True),
        history=list(slam.history),
    )


def summarize_slam(name: str, run: SlamRun, evaluation: TrajectoryEvaluation, num_particles: int) -> dict[str, Any]:
    """One summary row for a FastSLAM run."""

    resampled = [entry.get("resampled") for entry in run.history if "resampled" in entry]
    return {
        "method": name,
        "rmse_m": evaluation.rmse_m,
        "max_error_m": evaluation.max_error_m,
        "num_fixes": evaluation.num_fixes,
        "particles": num_particles,
        "runtime_mean_ms": 1000.0 * float(np.mean(run.runtime_s)),
        "runtime_max_ms": 1000.0 * float(np.max(run.runtime_s)),
        "landmarks_final": int(run.landmarks_total[-1]),
        "confirmed_final": int(run.landmarks_confirmed[-1]),
        "median_n_eff": float(np.median(run.n_eff)),
        "resample_fraction": float(np.mean(resampled)) if resampled else float("nan"),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

SUMMARY_COLUMNS = (
    "method",
    "rmse_m",
    "max_error_m",
    "num_fixes",
    "particles",
    "runtime_mean_ms",
    "runtime_max_ms",
    "landmarks_final",
    "confirmed_final",
    "median_n_eff",
    "resample_fraction",
)


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


def _figure(figsize: tuple[float, float]):
    from matplotlib.figure import Figure

    return Figure(figsize=figsize, layout="constrained")


def _break_at_gaps(t: np.ndarray, y: np.ndarray, max_gap_s: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaNs where ``t`` jumps by more than ``max_gap_s`` so lines do not bridge GPS gaps."""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    gaps = np.flatnonzero(np.diff(t) > max_gap_s) + 1
    return np.insert(t, gaps, np.nan), np.insert(y, gaps, np.nan)


def plot_features(out_path: Path, detections: list[list[TreeObservation]]) -> None:
    """Features per scan and the diameter histogram."""

    fig = _figure((10, 3.5))
    left, right = fig.subplots(1, 2)
    counts = [len(scan) for scan in detections]
    left.plot(counts, lw=0.6)
    left.set_xlabel("scan index")
    left.set_ylabel("trees detected")
    left.set_title("Detections per scan")
    diameters = [obs.diameter_m for scan in detections for obs in scan]
    if diameters:
        right.hist(diameters, bins=40, range=(0.0, max(1.0, float(np.percentile(diameters, 99.5)))))
    right.set_xlabel("estimated diameter [m]")
    right.set_title("Trunk diameters")
    fig.savefig(out_path, dpi=120)


def plot_map(
    out_path: Path,
    title: str,
    evaluation: TrajectoryEvaluation,
    run: SlamRun,
    dr: dict[str, TrajectoryEvaluation],
) -> None:
    """Aligned estimate, GPS, aligned dead reckoning, and the final map."""

    fig = _figure((8, 8))
    axis = fig.subplots()
    axis.plot(*evaluation.reference_xy_m.T, ".", ms=2, color="0.6", label="GPS (valid fixes)")
    for name, result in dr.items():
        if "+" in name:
            axis.plot(*result.aligned_xy_m.T, lw=1, color="C1", label=f"{name}, RMSE {result.rmse_m:.1f} m")
    track = apply_se2(evaluation.rotation, evaluation.translation, run.estimate[:, :2])
    axis.plot(*track.T, lw=1.5, color="C0", label=f"FastSLAM, RMSE {evaluation.rmse_m:.1f} m")
    if run.final_map.size:
        landmarks_world = apply_se2(evaluation.rotation, evaluation.translation, run.final_map)
        axis.scatter(*landmarks_world.T, s=8, marker="*", color="C2", label=f"{len(run.final_map)} confirmed landmarks")
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("east [m]")
    axis.set_ylabel("north [m]")
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize="small")
    fig.savefig(out_path, dpi=120)


def plot_diagnostics(
    out_path: Path,
    run: SlamRun,
    evaluation: TrajectoryEvaluation,
    dr_evaluation: TrajectoryEvaluation | None,
    num_particles: int,
) -> None:
    """Error vs. time, N_eff, landmark count and runtime per scan."""

    fig = _figure((10, 10))
    axes = fig.subplots(4, 1, sharex=False)
    axes[0].plot(*_break_at_gaps(evaluation.time_s, evaluation.errors_m), lw=1, label="FastSLAM")
    if dr_evaluation is not None:
        axes[0].plot(*_break_at_gaps(dr_evaluation.time_s, dr_evaluation.errors_m), lw=1, label="dead reckoning")
    axes[0].set_ylabel("error vs GPS [m]")
    axes[0].legend(loc="upper left")
    axes[0].set_xlabel("time [s]")
    axes[1].plot(run.scan_index, run.n_eff, lw=0.8)
    axes[1].axhline(num_particles, color="0.5", ls=":", lw=1)
    axes[1].set_ylabel("N_eff")
    axes[2].plot(run.scan_index, run.landmarks_total, label="all")
    axes[2].plot(run.scan_index, run.landmarks_confirmed, label="confirmed")
    axes[2].set_ylabel("landmarks (best particle)")
    axes[2].legend(loc="upper left")
    axes[3].plot(run.scan_index, 1000.0 * run.runtime_s, lw=0.6)
    axes[3].set_ylabel("runtime per scan [ms]")
    axes[3].set_xlabel("scan index")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.savefig(out_path, dpi=120)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_and_evaluate(
    tuning: dict[str, Any],
    dataset: VictoriaParkDataset,
    detections: list[list[TreeObservation]],
    stop_scan: int,
    valid: np.ndarray,
    progress_every: int = 0,
) -> tuple[SlamRun, TrajectoryEvaluation]:
    """Build, run and evaluate one FastSLAM configuration."""

    slam = make_slam(tuning)
    run = run_fastslam(slam, dataset, detections, stop_scan, progress_every)
    evaluation = evaluate_against_gps(dataset, run.time_s, run.estimate[:, :2], valid=valid)
    return run, evaluation


MIN_GPS_FIXES = 2  # an SE(2) alignment needs at least two reference points


def minimum_segment_scans(dataset, valid: np.ndarray, min_gps_fixes: int = MIN_GPS_FIXES) -> int:
    """Smallest number of scans whose segment contains odometry and ``min_gps_fixes`` valid GPS fixes."""
    t_needed = float(dataset.odometry_time_s[0])
    gps_times = np.sort(np.asarray(dataset.gps_time_s, dtype=np.float64)[np.asarray(valid, dtype=bool)])
    gps_times = gps_times[gps_times >= dataset.odometry_time_s[0]]
    if gps_times.size < min_gps_fixes:
        return dataset.num_scans + 1  # never enough
    t_needed = max(t_needed, float(gps_times[min_gps_fixes - 1]))
    return int(np.searchsorted(dataset.laser_time_s, t_needed, side="left")) + 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scans", type=int, default=SEGMENT_SCANS, help=f"number of scans to process (default {SEGMENT_SCANS})")
    parser.add_argument("--all", action="store_true", help="process the whole run (overrides --scans)")
    parser.add_argument("--sweep", nargs="+", choices=sorted(SWEEPS), default=[], help="sensitivity sweeps to run")
    parser.add_argument("--out", type=Path, default=Path("results") / "project2", help="output folder")
    parser.add_argument("--progress", type=int, default=0, help="print a progress line every N scans")
    args = parser.parse_args(argv)
    if not args.all and args.scans < 1:
        parser.error("--scans must be a positive number of scans")

    dataset = load_victoria_park_data()
    valid = valid_gps_mask(dataset)
    stop_scan = dataset.num_scans if args.all else min(args.scans, dataset.num_scans)
    minimum = minimum_segment_scans(dataset, valid)
    if stop_scan < minimum:
        parser.error(
            f"--scans {args.scans} is too short to evaluate: the segment needs odometry and at least "
            f"{MIN_GPS_FIXES} valid GPS fixes, which takes at least {minimum} scans"
        )
    t_stop = float(dataset.laser_time_s[stop_scan - 1])
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    log = StageLog()
    log.add("Dataset", "ok", f"{stop_scan} scans, {dataset.laser_time_s[0]:.1f}-{t_stop:.1f} s, "
            f"{np.count_nonzero(valid)} valid GPS fixes in the whole log")

    rows: list[dict[str, Any]] = []
    dr = dead_reckoning_comparison(dataset, stop_time_s=t_stop, valid=valid)
    for name, result in dr.items():
        rows.append({"method": name, "rmse_m": result.rmse_m, "max_error_m": result.max_error_m, "num_fixes": result.num_fixes})
    log.add("Dead reckoning", "ok", ", ".join(f"{name.split('(')[1].rstrip(')')}: {r.rmse_m:.1f} m" for name, r in dr.items()))

    try:
        detections, seconds = extract_all(dataset, stop_scan)
        counts = np.array([len(scan) for scan in detections])
        log.add("Feature extraction", "ok", f"{counts.sum()} detections, median {np.median(counts):.0f} per scan, "
                f"{1000 * seconds / stop_scan:.2f} ms per scan")
        plot_features(out_dir / "features.png", detections)
    except NotImplementedError as error:
        log.add("Feature extraction", "not implemented", _first_line(error))
        detections = None

    best_dr = dr["DR (encoder correction + lever arm)"]
    if detections is None:
        log.add("FastSLAM", "skipped", "needs extract_tree_landmarks")
        args.sweep = []
    else:
        try:
            run, evaluation = run_and_evaluate(TUNING, dataset, detections, stop_scan, valid, args.progress)
            rows.append(summarize_slam("FastSLAM (weighted mean)", run, evaluation, int(TUNING["num_particles"])))
            if run.best_trajectory is not None and len(run.best_trajectory) == len(run.time_s) + 1:
                best_eval = evaluate_against_gps(dataset, run.time_s, run.best_trajectory[1:, :2], valid=valid)
                rows.append(summarize_slam("FastSLAM (best particle path)", run, best_eval, int(TUNING["num_particles"])))
            log.add("FastSLAM", "ok", f"RMSE {evaluation.rmse_m:.2f} m vs DR {best_dr.rmse_m:.2f} m, "
                    f"{1000 * np.mean(run.runtime_s):.1f} ms per scan, {run.landmarks_confirmed[-1]} confirmed landmarks")
            plot_map(out_dir / "trajectory_map.png", f"Scans 0-{stop_scan - 1}", evaluation, run, dr)
            plot_diagnostics(out_dir / "diagnostics.png", run, evaluation, best_dr, int(TUNING["num_particles"]))
        except NotImplementedError as error:
            log.add("FastSLAM", "not implemented", _first_line(error))
            args.sweep = []
        except Exception as error:  # keep going: report the failure and still write the DR table
            log.add("FastSLAM", "error", _first_line(error))
            traceback.print_exc()
            args.sweep = []

    write_tables(rows, out_dir, "summary", SUMMARY_COLUMNS)

    for sweep in args.sweep:
        key, values = SWEEPS[sweep]
        sweep_rows = []
        for value in values:
            tuning = {**TUNING, key: value}
            try:
                run, evaluation = run_and_evaluate(tuning, dataset, detections, stop_scan, valid)
            except Exception as error:
                log.add(f"Sweep {sweep}={value}", "error", _first_line(error))
                continue
            row = summarize_slam(f"{key}={value}", run, evaluation, int(tuning["num_particles"]))
            row[key] = value
            sweep_rows.append(row)
            log.add(f"Sweep {sweep}={value}", "ok", f"RMSE {evaluation.rmse_m:.2f} m, {row['runtime_mean_ms']:.1f} ms per scan")
        if sweep_rows:
            write_tables(sweep_rows, out_dir, f"sweep_{sweep}", (key, *SUMMARY_COLUMNS))

    print(f"\nWrote tables and plots to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
