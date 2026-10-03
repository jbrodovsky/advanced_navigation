#!/usr/bin/env python3
"""Build the dataset archives that students download (instructor only).

Run from the repository root with the original data in ``data/<name>/``::

    uv run python scripts/package_datasets.py all
    uv run python scripts/package_datasets.py victoria-park --include-rawlog

Archives are written to ``data/_upload/`` (git-ignored). Each archive holds one
top-level folder named after the dataset, so ``fetch_data.py`` can unpack it
straight into ``data/<name>/``. Zips are deterministic: the same inputs always
produce the same bytes and therefore the same SHA-256.

After uploading, paste the printed ``sha256`` and ``size_bytes`` (and the share
link) into ``scripts/data_sources.toml``.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import sys
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXED_TIMESTAMP = (2026, 1, 1, 0, 0, 0)
CHUNK_BYTES = 1024 * 1024

DRONE_FILES = ["parameters.txt", *(f"studentdata{index}.mat" for index in range(8))]
DRONE_README = """Advanced Robot Navigation - Project 1 drone dataset
=====================================================

studentdata0.mat ... studentdata7.mat
    Quadrotor flights over an AprilTag mat. Each file holds:
      data   - onboard packets: t, rpy, acc, angular rate (omg; drpy in
               studentdata0), detected tag ids and corner pixels (p0-p4),
               and the raw camera image (img)
      vicon  - 12 x N motion-capture truth: position, roll/pitch/yaw,
               velocity, angular velocity
      time   - Vicon timestamps (s)
parameters.txt
    Camera intrinsics, camera-to-body mounting, and the tag layout.

Unpack into data/drone-data/ (scripts/fetch_data.py does this for you).
For course use only. Do not redistribute.
"""

VICTORIA_PARK_FILES = [
    "aa3_dr.mat",
    "aa3_gpsx.mat",
    "aa3_lsr2.mat",
    "README.txt",
    "info.txt",
    "car.bmp",
    "detectTreesI16.m",
    "viewLsr.m",
]
VICTORIA_PARK_RAWLOG = "victoria_park_dataset.rawlog"


def _add_bytes(bundle: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=FIXED_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    bundle.writestr(info, payload)


def _add_file(bundle: zipfile.ZipFile, name: str, source: Path) -> None:
    info = zipfile.ZipInfo(name, date_time=FIXED_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    info.file_size = source.stat().st_size
    with source.open("rb") as reader, bundle.open(info, "w", force_zip64=True) as writer:
        shutil.copyfileobj(reader, writer, CHUNK_BYTES)


def build_archive(
    dataset: str,
    source_dir: Path,
    files: list[str],
    output_dir: Path,
    extra: dict[str, str] | None = None,
) -> Path:
    """Zip ``files`` from ``source_dir`` (plus ``extra`` text members) under ``<dataset>/``."""
    missing = [name for name in files if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{source_dir} is missing: {', '.join(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"{dataset}.zip"
    partial = archive.with_name(archive.name + ".part")
    members = {name: source_dir / name for name in files}
    with zipfile.ZipFile(partial, "w") as bundle:
        for name in sorted(set(members) | set(extra or {})):
            member = f"{dataset}/{name}"
            if extra and name in extra:
                _add_bytes(bundle, member, extra[name].encode("utf-8"))
            else:
                _add_file(bundle, member, members[name])
    partial.replace(archive)
    return archive


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report(dataset: str, archive: Path) -> None:
    size = archive.stat().st_size
    print(f"wrote {archive} ({size / 1e6:.1f} MB)")
    print("paste into scripts/data_sources.toml:")
    print(f"  [datasets.{dataset}]")
    print(f'  sha256 = "{sha256_of(archive)}"')
    print(f"  size_bytes = {size}")
    print()


def package_drone(data_dir: Path, output_dir: Path) -> Path:
    return build_archive(
        "drone-data",
        data_dir / "drone-data",
        DRONE_FILES,
        output_dir,
        extra={"DATASET_README.txt": DRONE_README},
    )


def package_victoria_park(data_dir: Path, output_dir: Path, include_rawlog: bool) -> Path:
    files = [*VICTORIA_PARK_FILES, *([VICTORIA_PARK_RAWLOG] if include_rawlog else [])]
    return build_archive("victoria-park", data_dir / "victoria-park", files, output_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the dataset upload archives (instructor only).")
    parser.add_argument("dataset", choices=["drone-data", "victoria-park", "all"])
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data", help="folder holding the original data")
    parser.add_argument("--output", type=Path, default=None, help="output folder (default: <data-dir>/_upload)")
    parser.add_argument(
        "--include-rawlog",
        action="store_true",
        help="also include the MRPT .rawlog in the Victoria Park archive",
    )
    args = parser.parse_args(argv)
    output_dir = args.output or args.data_dir / "_upload"

    try:
        if args.dataset in ("drone-data", "all"):
            report("drone-data", package_drone(args.data_dir, output_dir))
        if args.dataset in ("victoria-park", "all"):
            report("victoria-park", package_victoria_park(args.data_dir, output_dir, args.include_rawlog))
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
