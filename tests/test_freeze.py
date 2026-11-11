"""Checks that the committed Quarto freeze matches the sources CI renders.

CI never has the datasets, so it relies on ``_freeze/``. Quarto reuses a frozen
result only when its stored ``hash`` equals the MD5 of the whole ``.qmd`` file;
any edit, even to prose or speaker notes, makes Quarto re-execute the code, and
code that loads a dataset then fails in CI. Fix a failure here by rendering the
file with the data present and committing its ``_freeze/`` files::

    uv run quarto render book/<chapter>.qmd --to html
    uv run quarto render book/<chapter>.qmd --to pdf
    uv run quarto render slides/decks/<deck>.qmd
"""

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Project folder -> freeze file names, one per format CI renders.
PROJECTS = {"book": ("html.json", "tex.json"), "slides": ("html.json",)}

EXECUTABLE_CELL = re.compile(r"^```\{python", re.MULTILINE)


def executable_sources() -> list[Path]:
    sources = []
    for project in PROJECTS:
        for path in sorted((ROOT / project).rglob("*.qmd")):
            parts = path.relative_to(ROOT / project).parts
            if parts[0].startswith(("_", ".")):
                continue
            if EXECUTABLE_CELL.search(path.read_text(encoding="utf-8")):
                sources.append(path)
    return sources


SOURCES = executable_sources()


def test_found_executable_sources():
    assert SOURCES, "no .qmd files with Python cells found; has the layout changed?"


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_freeze_is_current(source: Path):
    project = source.relative_to(ROOT).parts[0]
    relative = source.relative_to(ROOT / project).with_suffix("")
    results = ROOT / project / "_freeze" / relative / "execute-results"
    digest = hashlib.md5(source.read_bytes()).hexdigest()
    for name in PROJECTS[project]:
        frozen = results / name
        assert frozen.is_file(), f"no frozen output {frozen.relative_to(ROOT)}; render it with the data present"
        stored = json.loads(frozen.read_text(encoding="utf-8"))["hash"]
        assert stored == digest, (
            f"{frozen.relative_to(ROOT)} is stale: {source.relative_to(ROOT)} changed since it was "
            "rendered. Re-render it with the data present and commit the _freeze/ files."
        )
