"""Run every Project 1 stage on a dataset split and write comparison tables and plots.

From the repository root::

    uv run python -m drone_pipeline.run_all                       # development split
    uv run python -m drone_pipeline.run_all --split validation
    uv run python -m drone_pipeline.run_all --datasets 1 5 --pf-study
    uv run python -m drone_pipeline.run_all --help

Each stage calls your implementation through the public interfaces:

* Part A: ``propagate_state`` (dead reckoning from the Vicon state at the first
  packet with valid truth, the oracle initialization Part A allows);
* Part B: ``build_apriltag_corner_map``, ``estimate_pose`` and
  ``estimate_measurement_covariance``;
* Part C: ``EKFNavigator``, ``UKFNavigator`` and ``ParticleFilterNavigator``,
  each initialized from PnP poses alone (:func:`initial_state_from_poses`) and
  driven by :func:`run_navigator`.

Stages you have not written yet are reported as "not implemented" and skipped,
so the script is useful from the first week. Outputs go to ``--out``
(default ``results/<split>``): ``summary.csv``, ``summary.md``, one position
plot per dataset, and ``pf_study.csv`` with ``--pf-study``.

Edit :data:`TUNING` (or :func:`make_navigators`) to plug in your own noise
values. The placeholders below are deliberately untuned.
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
from scipy.stats import chi2

from . import filters, mechanization, observation
from .config import DEVELOPMENT_DATASETS, VALIDATION_DATASETS
from .datasets import interpolate_vicon_truth, load_drone_dataset, load_drone_parameters
from .evaluation import chi2_interval, fraction_inside, nees, rmse, state_errors, truth_matrix
from .mechanization import STATE_DIM, PropagationParams
from .types import DroneDataset, DroneParameters, PoseObservation, TruthSample

# ---------------------------------------------------------------------------
# Tuning: replace these placeholders with values you can justify.
# ---------------------------------------------------------------------------

TUNING: dict[str, Any] = {
    # Initial covariance diagonal (std devs): position m, attitude rad,
    # velocity m/s, gyro bias rad/s, accel bias m/s^2.
    "P0_std": [0.05] * 3 + [0.02] * 3 + [0.5] * 3 + [0.05] * 3 + [0.2] * 3,
    # Process-noise spectral density diagonal (per second), same layout.
    # TODO(Part A): justify from your IMU characterization.
    "Q_density": [1e-4] * 3 + [1e-3] * 3 + [1e-2] * 3 + [1e-6] * 3 + [1e-4] * 3,
    # Measurement covariance: None = estimate from the development split with
    # your estimate_measurement_covariance(); or paste a (6, 6) matrix here.
    "R": None,
    # Multiply R by this before filtering (report it if it is not 1).
    "R_inflation": 1.0,
    # Chi-square gate probability for the Kalman filters (None disables).
    "gate_probability": 0.999,
    "ukf": {"alpha": 1e-3, "beta": 2.0, "kappa": 0.0},
    "pf": {"num_particles": 1000, "resample_threshold": 0.5, "seed": 0},
    # Longest single prediction step; longer packet gaps are split.
    "max_step_s": 0.05,
    # Velocity initialization window for Part C (0 = zero velocity at the first pose).
    "init_window_s": 0.2,
}

PF_STUDY_COUNTS = (250, 1000, 2500, 5000)
PF_ESTIMATORS = ("max_weight", "mean", "weighted_mean")


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class StageLog:
    """Collects one line per stage so the end-of-run report is readable."""

    lines: list[str] = field(default_factory=list)

    def add(self, dataset: int | str, stage: str, status: str, detail: str = "") -> None:
        line = f"[ds{dataset}] {stage:<28} {status}" + (f" - {detail}" if detail else "")
        self.lines.append(line)
        print(line, flush=True)


@dataclass
class RunResult:
    """Output of :func:`run_navigator`."""

    times: np.ndarray
    states: np.ndarray
    covariances: np.ndarray | None
    runtime_s: float
    num_updates: int
    num_rejected: int
    extra_estimates: dict[str, np.ndarray] = field(default_factory=dict)


def _first_line(error: BaseException) -> str:
    text = str(error).strip().splitlines()
    return f"{type(error).__name__}: {text[0]}" if text else type(error).__name__


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def collect_observations(
    dataset: DroneDataset, params: DroneParameters, tag_map: dict[int, np.ndarray]
) -> list[PoseObservation | None]:
    """Run ``estimate_pose`` on every packet (``None`` where it returns ``None``).

    Zero distortion is passed: the packet corners are already undistorted.
    """
    zero_distortion = np.zeros(5)
    return [
        observation.estimate_pose(packet, params.camera_matrix, zero_distortion, tag_map, params.T_camera_body)
        for packet in dataset.packets
    ]


def state_from_truth(sample: TruthSample) -> np.ndarray:
    """15-state vector from a Vicon sample, with zero biases (Part A oracle start)."""
    x0 = np.zeros(STATE_DIM)
    x0[0:3], x0[3:6], x0[6:9] = sample.position, sample.rpy, sample.velocity
    return x0


def state_from_pose(pose: PoseObservation) -> np.ndarray:
    """15-state vector from a PnP pose: zero velocity and biases."""
    x0 = np.zeros(STATE_DIM)
    x0[0:6] = pose.z
    return x0


def initial_state_from_poses(
    poses: list[PoseObservation | None], window_s: float = 0.2, min_points: int = 8
) -> tuple[int, np.ndarray]:
    """Causal Part C initialization from PnP alone: ``(start_index, x0)``.

    Only poses with at least ``min_points`` corners are used (8 = two tags):
    single-tag poses on this dataset are biased by several centimetres and
    degrees, and a filter born from one can lock itself out behind its own
    gate. With ``window_s > 0`` the filter starts at the last usable pose
    inside the first ``window_s`` seconds, with velocity from a least-squares
    line through those poses (several flights begin while the drone is
    already moving). With ``window_s = 0`` it starts at the first usable pose
    with zero velocity. Biases start at zero either way.
    """
    indices = [k for k, pose in enumerate(poses) if pose is not None and pose.num_points >= min_points]
    if not indices:
        indices = [k for k, pose in enumerate(poses) if pose is not None]
    if not indices:
        raise ValueError("no PnP pose to initialize from")
    first = indices[0]
    window = [k for k in indices if poses[k].t - poses[first].t <= window_s]
    start = window[-1]
    x0 = state_from_pose(poses[start])
    if len(window) >= 3:
        t = np.array([poses[k].t for k in window])
        p = np.array([poses[k].position for k in window])
        x0[6:9] = np.polyfit(t - t[-1], p, 1)[0]
    return start, x0


def dead_reckon(dataset: DroneDataset, x0: np.ndarray, start: int, max_step_s: float) -> np.ndarray:
    """Propagate ``x0`` from packet ``start`` with ``propagate_state`` only."""
    packets = dataset.packets
    times = dataset.packet_times
    states = np.empty((len(packets) - start, STATE_DIM))
    states[0] = x0
    x = np.array(x0, dtype=np.float64)
    params = PropagationParams()
    for k in range(start + 1, len(packets)):
        dt = times[k] - times[k - 1]
        steps = max(1, math.ceil(dt / max_step_s))
        for _ in range(steps):
            x = mechanization.propagate_state(x, packets[k - 1].acc, packets[k - 1].omg, dt / steps, params)
        states[k - start] = x
    return states


def run_navigator(
    nav: Any,
    dataset: DroneDataset,
    observations: list[PoseObservation | None],
    start: int,
    max_step_s: float = 0.05,
    estimators: tuple[str, ...] = (),
) -> RunResult:
    """Drive one navigator through a dataset with the Project 1 timing rules.

    For each packet ``k > start``: predict from ``t[k-1]`` to ``t[k]`` with the
    IMU sample of packet ``k-1`` held constant (split into steps no longer than
    ``max_step_s``), then update with packet ``k``'s PnP pose if there is one.
    The particle filter's estimates are recorded after reweighting and before
    ``resample_if_needed()``.
    """
    packets = dataset.packets
    times = dataset.packet_times
    count = len(packets) - start
    states = np.empty((count, STATE_DIM))
    covariances: np.ndarray | None = np.empty((count, STATE_DIM, STATE_DIM))
    extra = {name: np.empty((count, STATE_DIM)) for name in estimators}
    num_updates = num_rejected = 0

    def record(index: int) -> None:
        nonlocal covariances
        states[index] = nav.state
        if covariances is not None:
            try:
                covariances[index] = nav.covariance
            except (AttributeError, NotImplementedError):
                covariances = None
        for name in estimators:
            extra[name][index] = nav.estimate(name)

    tic = time.perf_counter()
    record(0)
    for k in range(start + 1, len(packets)):
        dt = times[k] - times[k - 1]
        if dt > 0.0:
            steps = max(1, math.ceil(dt / max_step_s))
            for _ in range(steps):
                nav.predict(packets[k - 1].acc, packets[k - 1].omg, dt / steps)
        pose = observations[k]
        if pose is not None:
            accepted = nav.update(pose.z)
            num_updates += 1
            num_rejected += int(accepted is False)
        record(k - start)
        if pose is not None and hasattr(nav, "resample_if_needed"):
            nav.resample_if_needed()
    runtime = time.perf_counter() - tic
    return RunResult(times[start:], states, covariances, runtime, num_updates, num_rejected, extra)


def make_navigators(x0: np.ndarray, R: np.ndarray, tuning: dict[str, Any]) -> dict[str, Any]:
    """Build the three navigators from :data:`TUNING`. Edit freely."""
    P0 = np.diag(np.square(tuning["P0_std"]))
    Q = np.diag(tuning["Q_density"])
    R = np.asarray(R, dtype=np.float64) * tuning["R_inflation"]
    gate_probability = tuning["gate_probability"]
    gate = None if gate_probability is None else float(chi2.ppf(gate_probability, 6))
    return {
        "EKF": lambda: filters.EKFNavigator(x0, P0, Q, R, nis_gate=gate),
        "UKF": lambda: filters.UKFNavigator(x0, P0, Q, R, nis_gate=gate, **tuning["ukf"]),
        "PF": lambda: make_particle_filter(x0, P0, Q, R, tuning["pf"]),
    }


def make_particle_filter(x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, R: np.ndarray, pf_tuning: dict[str, Any]):
    """Draw ``num_particles`` around ``x0`` with covariance ``P0``."""
    rng = np.random.default_rng(pf_tuning.get("seed", 0))
    particles = rng.multivariate_normal(x0, P0, size=int(pf_tuning["num_particles"]))
    return filters.ParticleFilterNavigator(
        particles,
        Q,
        R,
        resample_threshold=pf_tuning.get("resample_threshold", 0.5),
        seed=pf_tuning.get("seed", 0),
    )


def summarize_run(result: RunResult, truth: list[TruthSample], history: list[dict[str, Any]] | None) -> dict[str, float]:
    """Error and consistency summary for one run (valid truth only)."""
    _, truth_states, valid = truth_matrix(truth)
    errors = state_errors(result.states, truth_states)
    row: dict[str, float] = dict(rmse(errors, valid))
    row["runtime_s"] = result.runtime_s
    row["num_updates"] = result.num_updates
    row["num_rejected"] = result.num_rejected
    nis_values = np.array([entry["nis"] for entry in (history or []) if "nis" in entry], dtype=np.float64)
    row["mean_nis"] = float(np.mean(nis_values)) if nis_values.size else float("nan")
    row["nis_in_95"] = fraction_inside(nis_values, 6) if nis_values.size else float("nan")
    if result.covariances is not None:
        values = nees(errors[valid], result.covariances[valid])
        row["anees"] = float(np.mean(values))
        row["nees_in_95"] = fraction_inside(values, 9)
    else:
        row["anees"] = row["nees_in_95"] = float("nan")
    return row


def estimate_R_from_split(dataset_ids: tuple[int, ...], params: DroneParameters, tag_map: dict[int, np.ndarray]) -> np.ndarray:
    """Pool PnP residuals over ``dataset_ids`` and call ``estimate_measurement_covariance``."""
    pooled_obs: list[PoseObservation] = []
    pooled_truth: list[TruthSample] = []
    for index in dataset_ids:
        dataset = load_drone_dataset(index)
        poses = [pose for pose in collect_observations(dataset, params, tag_map) if pose is not None]
        truth = interpolate_vicon_truth(dataset, np.array([pose.t for pose in poses]))
        for pose, sample in zip(poses, truth, strict=True):
            if sample.valid:
                pooled_obs.append(pose)
                pooled_truth.append(sample)
    return observation.estimate_measurement_covariance(pooled_obs, pooled_truth)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

SUMMARY_COLUMNS = (
    "dataset",
    "method",
    "pos_3d_m",
    "x_m",
    "y_m",
    "z_m",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
    "vel_3d_mps",
    "runtime_s",
    "num_updates",
    "num_rejected",
    "mean_nis",
    "nis_in_95",
    "anees",
    "nees_in_95",
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


def plot_positions(
    out_path: Path,
    title: str,
    truth: list[TruthSample],
    poses: list[PoseObservation | None] | None,
    tracks: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Truth, PnP points and every estimated track, one panel per axis."""
    from matplotlib.figure import Figure

    times, truth_states, valid = truth_matrix(truth)
    fig = Figure(figsize=(10, 8), layout="constrained")
    axes = fig.subplots(3, 1, sharex=True)
    for axis_index, (axis, label) in enumerate(zip(axes, ("x", "y", "z"))):
        axis.plot(times[valid], truth_states[valid, axis_index], color="black", lw=1.5, label="Vicon")
        if poses is not None:
            pnp = [(pose.t, pose.position[axis_index]) for pose in poses if pose is not None]
            if pnp:
                pnp_t, pnp_v = np.array(pnp).T
                axis.plot(pnp_t, pnp_v, ".", ms=2, color="0.5", label="PnP")
        for name, (track_t, track_x) in tracks.items():
            axis.plot(track_t, track_x[:, axis_index], lw=1, label=name)
        axis.set_ylabel(f"{label} [m]")
        lims = truth_states[valid, axis_index]
        if lims.size:
            pad = 0.5 + 0.2 * np.ptp(lims)
            axis.set_ylim(lims.min() - pad, lims.max() + pad)
    axes[0].set_title(title)
    axes[0].legend(loc="upper right", fontsize="small", ncol=3)
    axes[-1].set_xlabel("time [s]")
    fig.savefig(out_path, dpi=120)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_dataset(
    index: int,
    params: DroneParameters,
    tag_map: dict[int, np.ndarray] | None,
    R: np.ndarray | None,
    tuning: dict[str, Any],
    log: StageLog,
    out_dir: Path,
    pf_study: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run every available stage on one dataset; return summary and PF-study rows."""
    dataset = load_drone_dataset(index)
    truth = interpolate_vicon_truth(dataset)
    max_step = float(tuning["max_step_s"])
    rows: list[dict[str, Any]] = []
    pf_rows: list[dict[str, Any]] = []
    tracks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    times = dataset.packet_times

    # Part A: dead reckoning from the Vicon state at the first valid packet.
    first_valid = next(k for k, sample in enumerate(truth) if sample.valid)
    try:
        tic = time.perf_counter()
        dr = dead_reckon(dataset, state_from_truth(truth[first_valid]), first_valid, max_step)
        result = RunResult(times[first_valid:], dr, None, time.perf_counter() - tic, 0, 0)
        row = summarize_run(result, truth[first_valid:], None)
        rows.append({"dataset": index, "method": "DR (Vicon init)", **row})
        tracks["DR"] = (result.times, dr)
        log.add(index, "Part A dead reckoning", "ok", f"position RMSE {row['pos_3d_m']:.3f} m")
    except NotImplementedError as error:
        log.add(index, "Part A dead reckoning", "not implemented", _first_line(error))

    # Part B: PnP observations.
    poses: list[PoseObservation | None] | None = None
    if tag_map is not None:
        try:
            poses = collect_observations(dataset, params, tag_map)
            valid_poses = [(k, pose) for k, pose in enumerate(poses) if pose is not None]
            pose_states = np.zeros((len(poses), STATE_DIM))
            for k, pose in valid_poses:
                pose_states[k, 0:6] = pose.z
            mask = np.array([pose is not None for pose in poses])
            result = RunResult(times[mask], pose_states[mask], None, 0.0, len(valid_poses), 0)
            pose_truth = [sample for sample, keep in zip(truth, mask) if keep]
            row = summarize_run(result, pose_truth, None)
            row["vel_3d_mps"] = float("nan")  # PnP does not measure velocity
            rows.append({"dataset": index, "method": "PnP", **row})
            log.add(index, "Part B PnP", "ok", f"{len(valid_poses)}/{len(poses)} packets, RMSE {row['pos_3d_m']:.3f} m")
        except NotImplementedError as error:
            poses = None
            log.add(index, "Part B PnP", "not implemented", _first_line(error))
    else:
        log.add(index, "Part B PnP", "skipped", "needs build_apriltag_corner_map()")

    # Part C: filters, initialized from PnP alone (never from Vicon).
    if poses is None or R is None:
        log.add(index, "Part C filters", "skipped", "needs Part B poses and R")
    else:
        if all(pose is None for pose in poses):
            log.add(index, "Part C filters", "skipped", "no PnP pose in this dataset")
        else:
            start, x0 = initial_state_from_poses(poses, float(tuning["init_window_s"]))
            for name, factory in make_navigators(x0, R, tuning).items():
                try:
                    nav = factory()
                    result = run_navigator(nav, dataset, poses, start, max_step)
                    row = summarize_run(result, truth[start:], getattr(nav, "history", None))
                    rows.append({"dataset": index, "method": name, **row})
                    tracks[name] = (result.times, result.states)
                    log.add(index, f"Part C {name}", "ok", f"position RMSE {row['pos_3d_m']:.3f} m, {result.runtime_s:.1f} s")
                except NotImplementedError as error:
                    log.add(index, f"Part C {name}", "not implemented", _first_line(error))
                except Exception as error:  # keep going: one broken filter should not hide the others
                    log.add(index, f"Part C {name}", "error", _first_line(error))
                    traceback.print_exc()
            if pf_study:
                pf_rows.extend(particle_count_study(index, dataset, poses, start, x0, R, truth, tuning, log))

    plot_positions(out_dir / f"ds{index}_position.png", f"studentdata{index}", truth, poses, tracks)
    return rows, pf_rows


def particle_count_study(
    index: int,
    dataset: DroneDataset,
    poses: list[PoseObservation | None],
    start: int,
    x0: np.ndarray,
    R: np.ndarray,
    truth: list[TruthSample],
    tuning: dict[str, Any],
    log: StageLog,
    counts: tuple[int, ...] = PF_STUDY_COUNTS,
) -> list[dict[str, Any]]:
    """Position RMSE and runtime of the three PF point estimates for each particle count."""
    rows = []
    _, truth_states, valid = truth_matrix(truth[start:])
    P0 = np.diag(np.square(tuning["P0_std"]))
    Q = np.diag(tuning["Q_density"])
    for count in counts:
        try:
            nav = make_particle_filter(x0, P0, Q, np.asarray(R) * tuning["R_inflation"], {**tuning["pf"], "num_particles": count})
            result = run_navigator(nav, dataset, poses, start, float(tuning["max_step_s"]), estimators=PF_ESTIMATORS)
        except NotImplementedError as error:
            log.add(index, "Part C PF study", "not implemented", _first_line(error))
            return rows
        n_eff = np.array([entry["n_eff"] for entry in nav.history if "n_eff" in entry])
        for estimator in PF_ESTIMATORS:
            stats = rmse(state_errors(result.extra_estimates[estimator], truth_states), valid)
            rows.append(
                {
                    "dataset": index,
                    "num_particles": count,
                    "estimator": estimator,
                    "pos_3d_m": stats["pos_3d_m"],
                    "yaw_deg": stats["yaw_deg"],
                    "runtime_s": result.runtime_s,
                    "median_n_eff": float(np.median(n_eff)) if n_eff.size else float("nan"),
                }
            )
        log.add(index, "Part C PF study", "ok", f"N={count}: {result.runtime_s:.1f} s")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=("development", "validation", "all"), default="development")
    parser.add_argument("--datasets", type=int, nargs="+", help="explicit dataset indices (overrides --split)")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default results/<split>)")
    parser.add_argument("--pf-study", action="store_true", help=f"also run the particle-count study {PF_STUDY_COUNTS}")
    args = parser.parse_args(argv)

    if args.datasets:
        dataset_ids, label = tuple(args.datasets), "custom"
    else:
        label = args.split
        dataset_ids = {
            "development": DEVELOPMENT_DATASETS,
            "validation": VALIDATION_DATASETS,
            "all": DEVELOPMENT_DATASETS + VALIDATION_DATASETS,
        }[args.split]
    out_dir = args.out or Path("results") / label
    out_dir.mkdir(parents=True, exist_ok=True)

    log = StageLog()
    params = load_drone_parameters()
    try:
        tag_map: dict[int, np.ndarray] | None = observation.build_apriltag_corner_map()
        log.add("*", "Part B tag map", "ok", f"{len(tag_map)} tags")
    except NotImplementedError as error:
        tag_map = None
        log.add("*", "Part B tag map", "not implemented", _first_line(error))

    R: np.ndarray | None = None
    if TUNING["R"] is not None:
        R = np.asarray(TUNING["R"], dtype=np.float64)
        log.add("*", "Part B R", "ok", "from TUNING")
    elif tag_map is not None:
        try:
            R = estimate_R_from_split(DEVELOPMENT_DATASETS, params, tag_map)
            log.add("*", "Part B R", "ok", "estimated on the development split; diag std " + np.array2string(np.sqrt(np.diag(R)), precision=4))
        except NotImplementedError as error:
            log.add("*", "Part B R", "not implemented", _first_line(error))

    rows: list[dict[str, Any]] = []
    pf_rows: list[dict[str, Any]] = []
    for index in dataset_ids:
        ds_rows, ds_pf_rows = run_dataset(index, params, tag_map, R, TUNING, log, out_dir, args.pf_study)
        rows += ds_rows
        pf_rows += ds_pf_rows

    if rows:
        write_tables(rows, out_dir, "summary", SUMMARY_COLUMNS)
    if pf_rows:
        write_tables(pf_rows, out_dir, "pf_study", ("dataset", "num_particles", "estimator", "pos_3d_m", "yaw_deg", "runtime_s", "median_n_eff"))
    low, high = chi2_interval(6)
    print(f"\nNIS 95% single-sample interval (6 dof): [{low:.2f}, {high:.2f}]; NEES (9 dof): "
          f"[{chi2_interval(9)[0]:.2f}, {chi2_interval(9)[1]:.2f}]")
    print(f"Wrote {'summary tables and ' if rows else ''}plots to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
