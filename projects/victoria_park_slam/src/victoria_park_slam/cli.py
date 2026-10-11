"""Command line helpers for the Victoria Park starter package.

From the repository root::

    uv run victoria-park-slam summary
    uv run victoria-park-slam plot-scan --index 0 --output /tmp/scan0.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .datasets import LASER_RANGE_MASK, load_victoria_park_data
from .evaluation import valid_gps_mask
from .plotting import plot_scan_xy
from .utils import scan_to_xy


def _print_summary(root: Path | None = None) -> None:
    dataset = load_victoria_park_data(root)
    ranges = dataset.laser_ranges_m
    print(f"root: {dataset.root}")
    print(f"laser scans: {ranges.shape[0]} x {ranges.shape[1]} beams")
    print(f"laser time span: {dataset.laser_time_s[0]:.3f} s to {dataset.laser_time_s[-1]:.3f} s "
          f"(median spacing {np.median(np.diff(dataset.laser_time_s)) * 1000:.0f} ms)")
    print(f"no-return beams: {np.mean(~np.isfinite(ranges)):.1%}")
    if dataset.laser_raw is not None:
        flagged = int(np.count_nonzero(np.asarray(dataset.laser_raw) > LASER_RANGE_MASK))
        print(f"raw words with flag bits set: {flagged}")
    print(f"odometry samples: {dataset.speed_mps.size}")
    print(f"odometry time span: {dataset.odometry_time_s[0]:.3f} s to {dataset.odometry_time_s[-1]:.3f} s "
          f"(median spacing {np.median(np.diff(dataset.odometry_time_s)) * 1000:.0f} ms)")
    if dataset.gps_time_s is None:
        print("gps samples: unavailable")
    else:
        valid = valid_gps_mask(dataset)
        print(f"gps samples: {dataset.gps_time_s.size} ({np.count_nonzero(valid)} usable as a reference), "
              f"longest gap {np.max(np.diff(dataset.gps_time_s)):.1f} s")


def _plot_scan(root: Path | None, scan_index: int, output: Path | None) -> None:
    dataset = load_victoria_park_data(root)
    scan_xy_m = scan_to_xy(dataset.laser_ranges_m[scan_index], frame="vehicle")
    fig, _ = plot_scan_xy(scan_xy_m, title=f"Victoria Park scan {scan_index}")
    if output is None:
        import matplotlib.pyplot as plt

        plt.show()
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
        print(f"wrote {output}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect the Victoria Park dataset.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser("summary", help="print dataset shapes, rates and data quality")
    summary.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=None,
        help="folder with the MATLAB files; defaults to data/victoria-park at the repository root",
    )

    scan = subparsers.add_parser("plot-scan", help="plot one laser scan in the vehicle frame")
    scan.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=None,
        help="folder with the MATLAB files; defaults to data/victoria-park at the repository root",
    )
    scan.add_argument("--index", type=int, default=0, help="scan index to plot")
    scan.add_argument("--output", type=Path, help="optional image path")

    args = parser.parse_args(argv)

    if args.command == "summary":
        _print_summary(args.root)
    elif args.command == "plot-scan":
        _plot_scan(args.root, args.index, args.output)


if __name__ == "__main__":
    main()
