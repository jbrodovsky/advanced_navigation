"""Run the student-facing Project 3 tests against the reference solutions.

From the repository root::

    uv run python admin/solutions/ins_gnss/check_solutions.py
    uv run python admin/solutions/ins_gnss/check_solutions.py -k Jacobian   # extra args go to pytest

The starter package is not modified: the solutions are patched into the
already-imported ``ins_gnss`` modules before pytest collects the tests, so
every test that skips on the starter ("not implemented yet") runs here and
must pass. Among them are the numerical checks of every sign: ``F`` against a
finite-difference Jacobian of the mechanization, ``H`` (with a lever arm)
against finite differences of the residual, and the reset Jacobian against
the exact attitude reset. A skip in this run therefore means missing data,
not a stub. ``test_starter_stubs.py`` (instructor only) additionally checks that
the starter source files still raise ``NotImplementedError``.
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
    tests = REPO_ROOT / "projects" / "ins_gnss" / "tests"
    starter_check = HERE / "test_starter_stubs.py"  # the shipped starter must still be stubs
    return pytest.main(
        [str(tests), str(starter_check), "-ra", "-p", "no:cacheprovider", "--rootdir", str(REPO_ROOT), *argv]
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
