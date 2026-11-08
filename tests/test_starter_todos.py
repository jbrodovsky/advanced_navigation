"""Checks that every unwritten student function is easy to find.

Each starter package marks the code students must write with a
``# TODO(Project N, ...)`` comment directly above the ``raise
NotImplementedError`` stub, so ``grep -rn "TODO(" projects/<pkg>/src`` (or an
editor's TODO panel) lists every piece of remaining work. Once a student
replaces a stub, the raise disappears and this test has nothing to check.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    "drone_pipeline": 1,
    "victoria_park_slam": 2,
    "ins_gnss": 3,
}


def stub_sites(package: str) -> list[tuple[Path, int, list[str]]]:
    sites = []
    for path in sorted((ROOT / "projects" / package / "src").rglob("*.py")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if line.lstrip().startswith("raise NotImplementedError"):
                sites.append((path, index, lines))
    return sites


@pytest.mark.parametrize("package", sorted(PACKAGES))
def test_every_stub_has_a_todo_marker(package: str) -> None:
    marker = re.compile(rf"^\s*# TODO\(Project {PACKAGES[package]}, (Part [A-D]|Extension)\): \S")
    missing = []
    for path, index, lines in stub_sites(package):
        previous = lines[index - 1] if index > 0 else ""
        if not marker.match(previous):
            missing.append(f"{path.relative_to(ROOT)}:{index + 1}")
    assert not missing, "stubs without a TODO(Project N, Part X) line directly above: " + ", ".join(missing)


@pytest.mark.parametrize("package", sorted(PACKAGES))
def test_todo_markers_name_their_project(package: str) -> None:
    wrong = []
    for path in sorted((ROOT / "projects" / package / "src").rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "TODO(" in line and f"TODO(Project {PACKAGES[package]}, " not in line:
                wrong.append(f"{path.relative_to(ROOT)}:{number}")
    assert not wrong, "TODO markers that do not name their project: " + ", ".join(wrong)
