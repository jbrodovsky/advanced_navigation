"""Minimal CLI for the INS/GNSS starter package."""

from __future__ import annotations

import argparse
from pathlib import Path

from .datasets import load_ins_gnss_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the INS/GNSS dataset loader.")
    parser.add_argument("root", type=Path, help="dataset directory containing imu.csv and gnss.csv")
    args = parser.parse_args()

    dataset = load_ins_gnss_data(args.root)
    print(f"loaded {len(dataset.imu_samples)} IMU samples and {len(dataset.gnss_samples)} GNSS samples")


if __name__ == "__main__":
    main()

