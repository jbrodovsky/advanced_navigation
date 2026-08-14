"""Minimal CLI surface for the drone starter package."""

from __future__ import annotations

import argparse
from pathlib import Path

from .datasets import load_drone_dataset, load_drone_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the starter drone pipeline dataset interfaces.")
    parser.add_argument("dataset", type=Path, help="path to one studentdata*.mat file")
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("data/drone-data/parameters.txt"),
        help="path to parameters.txt",
    )
    args = parser.parse_args()

    dataset = load_drone_dataset(args.dataset)
    params = load_drone_parameters(args.parameters)
    print(f"loaded {args.dataset} with {len(dataset.packets)} packets")
    print(f"camera fx = {params.camera_matrix[0, 0]:.4f}")


if __name__ == "__main__":
    main()

