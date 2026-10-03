"""Quick look at the Project 3 data.

From the repository root::

    uv run ins-gnss                       # every trial with its status (recommended, caution, excluded)
    uv run ins-gnss --trial 2023-08-09_163741
    uv run ins-gnss --root path/to/conventional/dataset   # an imu.csv / gnss.csv folder
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .datasets import (
    TRIAL_NOTES,
    estimate_mount_rotation,
    gnss_arrays,
    imu_arrays,
    list_mems_nav_trials,
    load_ins_gnss_data,
    load_mems_nav_trial,
    repeated_fix_mask,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect the INS/GNSS datasets.")
    parser.add_argument("--root", type=Path, default=None, help="dataset folder (default: data/mems-nav-dataset)")
    parser.add_argument("--trial", help="MEMS trial name to summarize")
    parser.add_argument("--conventional", action="store_true", help="--root is an imu.csv/gnss.csv folder")
    args = parser.parse_args(argv)

    if args.conventional:
        if args.root is None:
            parser.error("--conventional needs --root")
        dataset = load_ins_gnss_data(args.root)
        print(f"loaded {len(dataset.imu_samples)} IMU samples and {len(dataset.gnss_samples)} GNSS samples")
        return

    trials = list_mems_nav_trials(args.root)
    if not args.trial:
        print(f"found {len(trials)} MEMS trials")
        for name in trials:
            note = TRIAL_NOTES.get(name)
            status = f"{note.status:<11} {note.reason}" if note else "(no notes)"
            print(f"  {name:<22} {status}")
        return

    trial = load_mems_nav_trial(args.root, args.trial)
    t_imu, accel, _ = imu_arrays(trial.imu_samples)
    gnss = gnss_arrays(trial.gnss_samples)
    print(f"{trial.name}: {len(trial.imu_samples)} IMU samples, {len(trial.gnss_samples)} GNSS fixes, "
          f"{len(trial.reference_states)} reference states over {t_imu[-1] - t_imu[0]:.0f} s")
    print(f"  epoch {trial.metadata['epoch_utc']}, baseline time offset {trial.metadata['baseline_time_offset_s']} s")
    print(f"  IMU spacing: median {np.median(np.diff(t_imu)):.2f} s, max {np.max(np.diff(t_imu)):.2f} s")
    print(f"  GNSS gaps over 1.5 s: {int(np.sum(np.diff(gnss['time_s']) > 1.5))}, "
          f"stale fixes: {int(repeated_fix_mask(trial.gnss_samples).sum())}, "
          f"median horizontal accuracy {np.nanmedian(gnss['horizontal_accuracy_m']):.1f} m")
    up = accel.mean(axis=0) / np.linalg.norm(accel.mean(axis=0))
    print(f"  mean 'up' in phone axes: [{up[0]:+.2f}, {up[1]:+.2f}, {up[2]:+.2f}]")
    try:
        C = estimate_mount_rotation(trial.imu_samples, trial.gnss_samples)
        print(f"  vehicle forward in phone axes: [{C[0, 0]:+.2f}, {C[0, 1]:+.2f}, {C[0, 2]:+.2f}]")
    except ValueError as error:
        print(f"  mount rotation: {error}")
    note = TRIAL_NOTES.get(trial.name)
    if note:
        print(f"  status: {note.status} ({note.reason})")


if __name__ == "__main__":
    main()
