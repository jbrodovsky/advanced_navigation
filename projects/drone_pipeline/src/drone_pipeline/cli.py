"""Minimal CLI for inspecting one drone dataset.

    uv run python -m drone_pipeline.cli 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .datasets import interpolate_vicon_truth, load_drone_dataset, load_drone_parameters


def _dataset_arg(value: str) -> Path | int:
    return int(value) if value.isdigit() else Path(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one Project 1 drone dataset.")
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
    times = dataset.packet_times
    gaps = np.diff(times)
    truth = interpolate_vicon_truth(dataset)
    tags = np.array([packet.num_tags for packet in dataset.packets])
    print(f"loaded {dataset.path} with {len(dataset.packets)} packets")
    print(f"rate field: {dataset.packets[0].rate_field}")
    print(f"packet time span: {times[0]:.2f} .. {times[-1]:.2f} s; median dt {np.median(gaps):.4f} s, max {gaps.max():.3f} s")
    print(f"Vicon time span:  {dataset.time[0]:.2f} .. {dataset.time[-1]:.2f} s; "
          f"{sum(not sample.valid for sample in truth)} packets outside it")
    print(f"packets with no tags: {(tags == 0).sum()}, with exactly one tag: {(tags == 1).sum()}")
    print(f"camera fx = {params.camera_matrix[0, 0]:.4f}, yaw offset = {params.yaw_offset_rad:.4f} rad")


if __name__ == "__main__":
    main()
