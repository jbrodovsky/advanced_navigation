#!/usr/bin/env python3
"""Download the course datasets into ``data/<name>/``.

Run from the repository root::

    uv run python scripts/fetch_data.py --list
    uv run python scripts/fetch_data.py all
    uv run python scripts/fetch_data.py drone-data
    uv run python scripts/fetch_data.py victoria-park --archive ~/Downloads/victoria-park.zip

Download links, checksums and expected files live in ``scripts/data_sources.toml``.
Set ``ADVNAV_DATA_DIR`` (or pass ``--data-dir``) to keep the data somewhere other
than ``data/`` at the repository root.

The script uses only the Python standard library (Python 3.11+).
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tomllib
from typing import Any, Callable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = Path(__file__).with_name("data_sources.toml")
DATA_ENV_VAR = "ADVNAV_DATA_DIR"
PLACEHOLDER = "REPLACE_ME"
ISSUES_URL = "https://github.com/jbrodovsky/advanced_navigation/issues"
USER_AGENT = "advanced-robot-navigation-fetch-data"
STAMP_NAME = ".fetched.json"
ZIP_MAGIC = b"PK\x03\x04"
CHUNK_BYTES = 1024 * 1024

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_PLACEHOLDER = 2

Opener = Callable[[Request], Any]


class FetchError(RuntimeError):
    """A dataset could not be downloaded or installed."""


class NotAnArchiveError(FetchError):
    """The server returned a web page (or other non-zip content) instead of the archive."""


class ChecksumMismatchError(FetchError):
    """The downloaded archive does not match the expected SHA-256."""


@dataclass(frozen=True)
class DataSource:
    name: str
    description: str
    provider: str
    url: str
    archive: str
    expected_files: tuple[str, ...]
    sha256: str = ""
    size_bytes: int = 0
    ref: str = "main"
    issue: int | None = None
    credit: str = ""

    @property
    def is_placeholder(self) -> bool:
        return self.provider != "github" and (not self.url or PLACEHOLDER in self.url)

    @property
    def issue_url(self) -> str:
        return f"{ISSUES_URL}/{self.issue}" if self.issue else ISSUES_URL


def load_sources(path: Path = SOURCES_PATH) -> dict[str, DataSource]:
    """Read ``data_sources.toml`` into ``DataSource`` records keyed by dataset name."""
    with path.open("rb") as stream:
        table = tomllib.load(stream)
    sources: dict[str, DataSource] = {}
    for name, entry in table.get("datasets", {}).items():
        sources[name] = DataSource(
            name=name,
            description=str(entry.get("description", "")),
            provider=str(entry.get("provider", "auto")),
            url=str(entry.get("url", "")),
            archive=str(entry.get("archive", f"{name}.zip")),
            expected_files=tuple(entry.get("expected_files", ())),
            sha256=str(entry.get("sha256", "")).lower(),
            size_bytes=int(entry.get("size_bytes", 0)),
            ref=str(entry.get("ref", "main")),
            issue=int(entry["issue"]) if entry.get("issue") else None,
            credit=str(entry.get("credit", "")),
        )
    return sources


def default_data_dir() -> Path:
    env_root = os.environ.get(DATA_ENV_VAR)
    return Path(env_root).expanduser() if env_root else REPO_ROOT / "data"


# ---------------------------------------------------------------------------
# Share-link handling
# ---------------------------------------------------------------------------

_DRIVE_FILE_PATH = re.compile(r"/file/d/([A-Za-z0-9_-]+)")


def _google_drive_file_id(url: str) -> str | None:
    parsed = urlparse(url)
    match = _DRIVE_FILE_PATH.search(parsed.path)
    if match:
        return match.group(1)
    query = dict(parse_qsl(parsed.query))
    return query.get("id")


def _with_query(url: str, **params: str) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update(params)
    return urlunparse(parsed._replace(query=urlencode(query)))


def _detect_provider(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if "drive.google.com" in host or "drive.usercontent.google.com" in host:
        return "gdrive"
    if "sharepoint.com" in host or "onedrive.live.com" in host or host == "1drv.ms":
        return "onedrive"
    return "https"


def normalize_share_url(url: str, provider: str = "auto") -> list[str]:
    """Turn a Google Drive or OneDrive share link into direct-download candidates.

    Candidates are returned in the order they should be tried. Folder links are
    rejected because neither service offers a plain-HTTP folder download that
    works reliably; share the zip file itself instead.
    """
    if provider == "auto":
        provider = _detect_provider(url)

    if provider == "gdrive":
        if "/folders/" in urlparse(url).path:
            raise FetchError(
                "this is a Google Drive folder link; share the .zip file inside the folder "
                "and use that file's link instead"
            )
        file_id = _google_drive_file_id(url)
        if file_id is None:
            raise FetchError(f"could not find a Google Drive file id in {url}")
        # drive.usercontent.google.com with confirm=t skips the "can't scan this
        # file for viruses" page that Drive shows for files over ~100 MB.
        return [f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"]

    if provider == "onedrive":
        candidates = [_with_query(url, download="1")]
        host = urlparse(url).netloc.lower()
        if host == "1drv.ms" or "onedrive.live.com" in host:
            token = base64.urlsafe_b64encode(url.encode("utf-8")).decode("ascii").rstrip("=")
            candidates.append(f"https://api.onedrive.com/v1.0/shares/u!{token}/root/content")
        return candidates

    return [url]


# ---------------------------------------------------------------------------
# Download, verify, extract
# ---------------------------------------------------------------------------


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_bytes(count: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if count < 1024 or unit == "GB":
            return f"{count:.0f} {unit}" if unit == "B" else f"{count:.1f} {unit}"
        count /= 1024
    return f"{count:.1f} GB"


def download_archive(
    candidates: list[str],
    destination: Path,
    expected_sha256: str = "",
    opener: Opener = urlopen,
) -> Path:
    """Stream the first working candidate URL to ``destination`` and verify it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    failures: list[str] = []

    for url in candidates:
        request = Request(url, headers={"User-Agent": USER_AGENT})
        try:
            response = opener(request)
        except OSError as error:  # URLError and HTTPError are OSErrors
            failures.append(f"{url}: {error}")
            continue

        with response:
            content_type = str(response.headers.get("Content-Type", "")).lower()
            total = int(response.headers.get("Content-Length") or 0)
            if "text/html" in content_type:
                failures.append(f"{url}: returned a web page instead of the archive")
                continue

            digest = hashlib.sha256()
            received = 0
            with partial.open("wb") as stream:
                while chunk := response.read(CHUNK_BYTES):
                    stream.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
                    if total and sys.stdout.isatty():
                        print(
                            f"\r  {_format_bytes(received)} / {_format_bytes(total)}",
                            end="",
                            flush=True,
                        )
            if total and sys.stdout.isatty():
                print()

        with partial.open("rb") as stream:
            magic = stream.read(len(ZIP_MAGIC))
        if magic != ZIP_MAGIC:
            partial.unlink(missing_ok=True)
            failures.append(f"{url}: response was not a zip archive")
            continue

        actual = digest.hexdigest()
        if expected_sha256 and actual != expected_sha256:
            partial.unlink(missing_ok=True)
            raise ChecksumMismatchError(
                f"checksum mismatch for {destination.name}: expected {expected_sha256}, got {actual}"
            )
        partial.replace(destination)
        return destination

    details = "\n  ".join(failures)
    raise NotAnArchiveError(
        "could not download the archive. This usually means the link is not shared as "
        "'anyone with the link', points at a folder, or Google Drive's download quota "
        f"was exceeded.\n  {details}"
    )


def _strip_prefix(names: list[str]) -> str:
    """Return the single top-level folder shared by every member, or ''."""
    tops = {PurePosixPath(name).parts[0] for name in names if name and not name.startswith("__MACOSX")}
    if len(tops) == 1:
        top = next(iter(tops))
        if all(name == f"{top}/" or name.startswith(f"{top}/") for name in names if not name.startswith("__MACOSX")):
            return f"{top}/"
    return ""


def safe_extract(archive: Path, target: Path) -> list[Path]:
    """Extract ``archive`` into ``target``, dropping a single top-level folder.

    Members with absolute paths or ``..`` components are rejected. Files are
    unpacked into a temporary sibling folder first and then moved into place,
    so an interrupted extraction never leaves a half-written dataset.
    """
    staging = target.parent / f".tmp-{target.name}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    written: list[Path] = []
    try:
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            prefix = _strip_prefix(names)
            for info in bundle.infolist():
                if info.filename.startswith("__MACOSX") or info.is_dir():
                    continue
                raw = PurePosixPath(info.filename)
                if raw.is_absolute() or ".." in raw.parts or "\\" in info.filename:
                    raise FetchError(f"refusing unsafe path in archive: {info.filename}")
                relative = PurePosixPath(info.filename[len(prefix):])
                if not relative.parts:
                    continue
                destination = staging.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info) as source, destination.open("wb") as sink:
                    shutil.copyfileobj(source, sink, CHUNK_BYTES)
                written.append(Path(*relative.parts))

        for relative in written:
            final = target / relative
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / relative, final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return written


def _stamp_path(target: Path) -> Path:
    return target / STAMP_NAME


def dataset_is_current(source: DataSource, target: Path) -> bool:
    """True when ``target`` holds every expected file from the current archive."""
    if not target.is_dir() or not all((target / name).exists() for name in source.expected_files):
        return False
    stamp = _stamp_path(target)
    if not stamp.exists():
        # Files placed by hand (e.g. an instructor's local copy) count as present.
        return True
    try:
        recorded = json.loads(stamp.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return not source.sha256 or recorded.get("sha256") == source.sha256


def _write_stamp(target: Path, source: DataSource, sha256: str, origin: str) -> None:
    payload = {
        "dataset": source.name,
        "sha256": sha256,
        "origin": origin,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    _stamp_path(target).write_text(json.dumps(payload, indent=2) + "\n")


def _placeholder_message(source: DataSource) -> str:
    return (
        f"{source.name}: no download link has been published yet.\n"
        f"  Status: {source.issue_url}\n"
        f"  If your instructor shared the file another way, download {source.archive} and run:\n"
        f"    uv run python scripts/fetch_data.py {source.name} --archive /path/to/{source.archive}"
    )


def _load_mems_fetcher():
    script_path = Path(__file__).with_name("fetch_ins_gnss_data.py")
    spec = importlib.util.spec_from_file_location("fetch_ins_gnss_data", script_path)
    if spec is None or spec.loader is None:
        raise FetchError(f"could not import {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fetch_dataset(
    source: DataSource,
    data_dir: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
    archive: Path | None = None,
    opener: Opener = urlopen,
) -> int:
    """Install one dataset into ``data_dir / source.name``. Returns an exit status."""
    target = data_dir / source.name

    if source.provider == "github":
        if archive is not None:
            raise FetchError(f"{source.name} is fetched from GitHub; --archive is not supported")
        if not force and not dry_run and dataset_is_current(source, target):
            print(f"{source.name}: already present at {target} (use --force to re-download)")
            return EXIT_OK
        print(f"{source.name}: fetching from {source.url} ({source.ref})")
        _load_mems_fetcher().fetch_datasets(target, source.ref, force, dry_run)
        return EXIT_OK

    if archive is None and not force and dataset_is_current(source, target):
        print(f"{source.name}: already present at {target} (use --force to re-download)")
        return EXIT_OK

    if archive is not None:
        archive = archive.expanduser()
        if not archive.is_file():
            raise FetchError(f"{archive} does not exist")
        if dry_run:
            print(f"{source.name}: would install {archive} into {target}")
            return EXIT_OK
        actual = sha256_of(archive)
        if source.sha256 and actual != source.sha256:
            raise ChecksumMismatchError(
                f"checksum mismatch for {archive.name}: expected {source.sha256}, got {actual}"
            )
        if not source.sha256:
            print(f"{source.name}: warning: no checksum recorded in data_sources.toml; skipping verification")
        zip_path, origin, remove_after = archive, str(archive), False
    else:
        if source.is_placeholder:
            print(_placeholder_message(source))
            return EXIT_PLACEHOLDER
        candidates = normalize_share_url(source.url, source.provider)
        size = f" (~{_format_bytes(source.size_bytes)})" if source.size_bytes else ""
        if dry_run:
            print(f"{source.name}: would download {source.archive}{size} into {target}")
            return EXIT_OK
        print(f"{source.name}: downloading {source.archive}{size}")
        zip_path = download_archive(candidates, data_dir / ".downloads" / source.archive, source.sha256, opener)
        actual = sha256_of(zip_path)
        origin, remove_after = source.url, True

    safe_extract(zip_path, target)
    missing = [name for name in source.expected_files if not (target / name).exists()]
    if missing:
        raise FetchError(f"{source.name}: archive is missing expected files: {', '.join(missing)}")
    _write_stamp(target, source, actual, origin)
    if remove_after:
        zip_path.unlink(missing_ok=True)
    print(f"{source.name}: installed into {target}")
    return EXIT_OK


def _status(source: DataSource, data_dir: Path) -> str:
    if dataset_is_current(source, data_dir / source.name):
        return "present"
    if source.is_placeholder:
        return "link pending"
    return "not downloaded"


def list_sources(sources: dict[str, DataSource], data_dir: Path) -> None:
    for source in sources.values():
        print(f"{source.name:<18} {_status(source, data_dir):<15} {source.description}")
        if source.is_placeholder:
            print(f"{'':<18} {'':<15} link status: {source.issue_url}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the course datasets into data/<name>/.")
    parser.add_argument("names", nargs="*", help="dataset names, or 'all'")
    parser.add_argument("--list", action="store_true", help="show the datasets and whether they are present")
    parser.add_argument("--data-dir", type=Path, default=None, help=f"data folder (default: ${DATA_ENV_VAR} or ./data)")
    parser.add_argument("--force", action="store_true", help="re-download even if the data is present")
    parser.add_argument("--dry-run", action="store_true", help="show what would happen without downloading")
    parser.add_argument("--archive", type=Path, help="install from a zip you already downloaded (one dataset only)")
    parser.add_argument("--sources", type=Path, default=SOURCES_PATH, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    sources = load_sources(args.sources)
    data_dir = (args.data_dir or default_data_dir()).expanduser()

    if args.list or not args.names:
        list_sources(sources, data_dir)
        if not args.names:
            print("\nFetch with: uv run python scripts/fetch_data.py all  (or a dataset name)")
        return EXIT_OK

    names = list(sources) if args.names == ["all"] else args.names
    unknown = [name for name in names if name not in sources]
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(unknown)}; choose from {', '.join(sources)} or 'all'")
    if args.archive is not None and len(names) != 1:
        parser.error("--archive installs exactly one dataset; name it explicitly")

    status = EXIT_OK
    for name in names:
        try:
            result = fetch_dataset(
                sources[name],
                data_dir,
                force=args.force,
                dry_run=args.dry_run,
                archive=args.archive,
            )
        except FetchError as error:
            print(f"{name}: error: {error}", file=sys.stderr)
            result = EXIT_ERROR
        except OSError as error:
            print(f"{name}: error: {error}", file=sys.stderr)
            result = EXIT_ERROR
        status = max(status, result)
    return status


if __name__ == "__main__":
    sys.exit(main())
