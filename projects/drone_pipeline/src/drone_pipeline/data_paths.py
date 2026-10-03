"""Locate course datasets on disk.

Datasets are not stored in git. They live under ``data/<name>/`` at the
repository root and are downloaded with::

    uv run python scripts/fetch_data.py <name>

Set the ``ADVNAV_DATA_DIR`` environment variable to the folder that contains
``<name>/`` if you keep the data somewhere else.

This module is intentionally identical in every project package so each
starter package can be used on its own.
"""

from __future__ import annotations

from collections.abc import Iterable
import os
from pathlib import Path

DATA_ENV_VAR = "ADVNAV_DATA_DIR"


class DatasetNotFoundError(FileNotFoundError):
    """Raised when a course dataset cannot be located on disk."""


def _search_bases() -> list[Path]:
    cwd = Path.cwd()
    return [cwd, *cwd.parents]


def _default_candidates(name: str) -> list[Path]:
    candidates: list[Path] = []
    env_root = os.environ.get(DATA_ENV_VAR)
    if env_root:
        candidates.append(Path(env_root).expanduser() / name)
    candidates.extend(base / "data" / name for base in _search_bases())
    candidates.extend(base / "data" / name for base in Path(__file__).resolve().parents)
    return candidates


def _explicit_candidates(explicit: str | Path) -> list[Path]:
    path = Path(explicit).expanduser()
    if path.is_absolute():
        return [path]
    # Relative paths are resolved against the working directory and its parents
    # so ``data/<name>`` works from the repository root or any subfolder.
    return [base / path for base in _search_bases()]


def find_dataset_dir(
    name: str,
    explicit: str | Path | None = None,
    required: Iterable[str] = (),
) -> Path:
    """Return the directory holding dataset ``name``.

    Parameters
    ----------
    name:
        Dataset folder name, e.g. ``"drone-data"`` or ``"victoria-park"``.
    explicit:
        Optional user-supplied directory. When given, only this location
        (and, for a relative path, the same path under parent folders) is
        searched.
    required:
        File or folder names that must exist inside the directory.

    Search order without ``explicit``: ``$ADVNAV_DATA_DIR/<name>``, then
    ``data/<name>`` in the working directory and each parent, then
    ``data/<name>`` above this package's install location.
    """
    required = tuple(required)
    candidates = _explicit_candidates(explicit) if explicit is not None else _default_candidates(name)

    searched: list[Path] = []
    for candidate in candidates:
        if candidate in searched:
            continue
        searched.append(candidate)
        if candidate.is_dir() and all((candidate / item).exists() for item in required):
            return candidate

    needs = f" containing {', '.join(required)}" if required else ""
    looked = "\n".join(f"  - {path}" for path in searched)
    raise DatasetNotFoundError(
        f"Could not find the '{name}' dataset{needs}. Looked in:\n{looked}\n"
        f"Download it from the repository root with:\n"
        f"  uv run python scripts/fetch_data.py {name}\n"
        f"or set {DATA_ENV_VAR} to the folder that contains '{name}/'."
    )
