#!/usr/bin/env python3
"""Fetch Project 3 MEMS navigation data without checking CSV files into git."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


OWNER = "jbrodovsky"
REPO = "mems-nav-dataset"
REPO_URL = f"https://github.com/{OWNER}/{REPO}"
DATASETS = ("input", "baseline")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = REPO_ROOT / "data" / "mems-nav-dataset"


def contents_api_url(dataset: str, ref: str) -> str:
    query = urlencode({"ref": ref})
    return f"https://api.github.com/repos/{OWNER}/{REPO}/contents/data/{dataset}?{query}"


def fetch_json(url: str) -> Any:
    request = Request(url, headers={"User-Agent": "advanced-navigation-fetch-ins-gnss-data"})
    with urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def list_dataset_files(dataset: str, ref: str) -> list[dict[str, Any]]:
    payload = fetch_json(contents_api_url(dataset, ref))
    if not isinstance(payload, list):
        raise RuntimeError(f"expected a directory listing for {dataset}, got {type(payload).__name__}")
    return [
        entry
        for entry in payload
        if entry.get("type") == "file" and str(entry.get("name", "")).endswith(".csv")
    ]


def planned_action(destination: Path, expected_size: int, force: bool) -> str:
    if destination.exists() and not force and destination.stat().st_size == expected_size:
        return "skip"
    return "download"


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    request = Request(url, headers={"User-Agent": "advanced-navigation-fetch-ins-gnss-data"})
    with urlopen(request) as response, temporary_path.open("wb") as stream:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            stream.write(chunk)
    temporary_path.replace(destination)


def build_manifest(ref: str, listings: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "repo_url": REPO_URL,
        "ref": ref,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": {
            dataset: [
                {
                    "name": entry["name"],
                    "path": entry["path"],
                    "size": entry["size"],
                    "sha": entry["sha"],
                    "download_url": entry["download_url"],
                }
                for entry in entries
            ]
            for dataset, entries in listings.items()
        },
    }


def fetch_datasets(destination: Path, ref: str, force: bool, dry_run: bool) -> dict[str, Any]:
    listings = {dataset: list_dataset_files(dataset, ref) for dataset in DATASETS}

    for dataset, entries in listings.items():
        for entry in entries:
            target = destination / dataset / entry["name"]
            action = planned_action(target, int(entry["size"]), force)
            if dry_run:
                print(f"{action}: {target}")
                continue
            if action == "skip":
                print(f"skip: {target}")
                continue
            print(f"download: {target}")
            download_file(str(entry["download_url"]), target)

    manifest = build_manifest(ref, listings)
    if dry_run:
        print(f"manifest: {destination / 'manifest.json'}")
    else:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Project 3 MEMS INS/GNSS input and baseline data.")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION, help="local dataset root")
    parser.add_argument("--ref", default="main", help="git ref in jbrodovsky/mems-nav-dataset")
    parser.add_argument("--force", action="store_true", help="redownload files even when size matches")
    parser.add_argument("--dry-run", action="store_true", help="show planned downloads without writing files")
    args = parser.parse_args()

    fetch_datasets(args.destination, args.ref, args.force, args.dry_run)


if __name__ == "__main__":
    main()
