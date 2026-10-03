"""Minimal CLI surface for the drone starter package."""

from __future__ import annotations

import argparse
from pathlib import Path

from .datasets import load_drone_dataset, load_drone_parameters


def _dataset_arg(value: str) -> Path | int:
    return int(value) if value.isdigit() else Path(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the starter drone pipeline dataset interfaces.")
    parser.add_argument(
        "dataset",
        type=_dataset_arg,
        help="path to one studentdata*.mat file, or its index (e.g. 1 for studentdata1.mat)",
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=None,
        help="path to parameters.txt; defaults to the drone dataset folder (data/drone-data)",
    )
    args = parser.parse_args()

    dataset = load_drone_dataset(args.dataset)
    params = load_drone_parameters(args.parameters)
    print(f"loaded {dataset.path} with {len(dataset.packets)} packets")
    print(f"angular-rate field: {dataset.packets[0].rate_field}")
    print(f"camera fx = {params.camera_matrix[0, 0]:.4f}")


if __name__ == "__main__":
    main()
