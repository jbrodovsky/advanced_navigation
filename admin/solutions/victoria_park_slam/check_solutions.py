"""Run the student-facing Project 2 tests against the reference solutions.

From the repository root::

    uv run python admin/solutions/victoria_park_slam/check_solutions.py
    uv run python admin/solutions/victoria_park_slam/check_solutions.py -k Association   # extra args go to pytest

The starter package is not modified: the solutions are patched into the
already-imported ``victoria_park_slam`` modules before pytest collects the
tests, so every test that skips on the starter ("not implemented yet") runs
here and must pass. A skip in this run therefore means missing data, not a
stub.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

from solution_patch import install  # noqa: E402


def main(argv: list[str]) -> int:
    for name in install():
        print(f"patched {name}")
    tests = REPO_ROOT / "projects" / "victoria_park_slam" / "tests"
    return pytest.main([str(tests), "-ra", "-p", "no:cacheprovider", "--rootdir", str(REPO_ROOT), *argv])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
