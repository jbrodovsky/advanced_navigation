"""Instructor-only check: the distributed starter still ships the Project 3 stubs.

This lives outside the student suite on purpose. A student who implements a
function must not see a failing test for doing so; the instructor wants to know
that no solution leaked into ``projects/ins_gnss`` before handing it out.
``check_solutions.py`` runs this file together with the student tests.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
import unittest

STUBS = (
    "ins_gnss.strapdown.mechanize",
    "ins_gnss.strapdown.coarse_alignment",
    "ins_gnss.eskf.error_state_jacobian",
    "ins_gnss.eskf.discrete_process_noise",
    "ins_gnss.eskf.gnss_position_residual",
    "ins_gnss.eskf.joseph_update",
    "ins_gnss.eskf.inject_error_state",
    "ins_gnss.eskf.reset_covariance",
    "ins_gnss.eskf.gnss_velocity_residual",
)


class StarterStubTests(unittest.TestCase):
    def test_student_functions_are_stubs_in_the_starter(self) -> None:
        # Read the source files, not the imported objects (check_solutions patches those).
        for dotted in STUBS:
            module_name, function_name = dotted.rsplit(".", 1)
            tree = ast.parse(Path(importlib.import_module(module_name).__file__).read_text())
            with self.subTest(function=dotted):
                (function,) = [
                    node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name
                ]
                raises = [node for node in ast.walk(function) if isinstance(node, ast.Raise)]
                self.assertTrue(any("NotImplementedError" in ast.unparse(node) for node in raises))


if __name__ == "__main__":
    unittest.main()
