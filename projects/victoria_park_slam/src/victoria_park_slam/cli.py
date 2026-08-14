"""Minimal CLI for the Victoria Park starter package."""

from __future__ import annotations

import argparse
from pathlib import Path

from .datasets import load_victoria_park_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the Victoria Park dataset loader.")
    parser.add_argument("root", type=Path, help="path to the original_MATLAB_dataset directory")
    args = parser.parse_args()

    dataset = load_victoria_park_data(args.root)
    print(f"loaded {dataset.laser_ranges_m.shape[0]} scans from {args.root}")


if __name__ == "__main__":
    main()

