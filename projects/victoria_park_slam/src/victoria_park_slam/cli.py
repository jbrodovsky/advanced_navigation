"""Command line helpers for the Victoria Park starter package."""

from __future__ import annotations

import argparse
from pathlib import Path

from .datasets import load_victoria_park_data
from .plotting import plot_scan_xy
from .utils import scan_to_xy


def _print_summary(root: Path | None = None) -> None:
    dataset = load_victoria_park_data(root)
    print(f"root: {dataset.root}")
    print(f"laser scans: {dataset.laser_ranges_m.shape[0]} x {dataset.laser_ranges_m.shape[1]}")
    print(
        "laser time span: "
        f"{dataset.laser_time_s[0]:.3f} s to {dataset.laser_time_s[-1]:.3f} s"
    )
    print(f"odometry samples: {dataset.speed_mps.size}")
    print(
        "odometry time span: "
        f"{dataset.odometry_time_s[0]:.3f} s to {dataset.odometry_time_s[-1]:.3f} s"
    )
    if dataset.gps_time_s is None:
        print("gps samples: unavailable")
    else:
        print(f"gps samples: {dataset.gps_time_s.size}")


def _plot_scan(root: Path | None, scan_index: int, output: Path | None) -> None:
    dataset = load_victoria_park_data(root)
    scan_xy_m = scan_to_xy(dataset.laser_ranges_m[scan_index])
    fig, _ = plot_scan_xy(scan_xy_m, title=f"Victoria Park scan {scan_index}")
    if output is None:
        import matplotlib.pyplot as plt

        plt.show()
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
        print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the Victoria Park dataset.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser("summary", help="print dataset shapes and time spans")
    summary.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=None,
        help="folder with the MATLAB files; defaults to data/victoria-park at the repository root",
    )

    scan = subparsers.add_parser("plot-scan", help="plot one laser scan in scanner coordinates")
    scan.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=None,
        help="folder with the MATLAB files; defaults to data/victoria-park at the repository root",
    )
    scan.add_argument("--index", type=int, default=0, help="scan index to plot")
    scan.add_argument("--output", type=Path, help="optional image path")

    args = parser.parse_args()

    if args.command == "summary":
        _print_summary(args.root)
    elif args.command == "plot-scan":
        _plot_scan(args.root, args.index, args.output)


if __name__ == "__main__":
    main()
