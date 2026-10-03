"""Minimal CLI for the INS/GNSS starter package."""

from __future__ import annotations

import argparse
from pathlib import Path

from .datasets import list_mems_nav_trials, load_ins_gnss_data, load_mems_nav_trial


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the INS/GNSS dataset loader.")
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=None,
        help="dataset root; with --mems, defaults to data/mems-nav-dataset at the repository root",
    )
    parser.add_argument("--mems", action="store_true", help="inspect a fetched MEMS navigation dataset root")
    parser.add_argument("--trial", help="MEMS trial name to load")
    args = parser.parse_args()

    if args.mems:
        trials = list_mems_nav_trials(args.root)
        print(f"found {len(trials)} MEMS trials")
        for trial in trials:
            print(f"- {trial}")
        if args.trial:
            dataset = load_mems_nav_trial(args.root, args.trial)
            print(
                "loaded "
                f"{len(dataset.imu_samples)} IMU samples, "
                f"{len(dataset.gnss_samples)} GNSS samples, and "
                f"{len(dataset.reference_states)} baseline reference states "
                f"for {dataset.name}"
            )
        return

    if args.root is None:
        parser.error("root is required unless --mems is given")
    dataset = load_ins_gnss_data(args.root)
    print(f"loaded {len(dataset.imu_samples)} IMU samples and {len(dataset.gnss_samples)} GNSS samples")


if __name__ == "__main__":
    main()
