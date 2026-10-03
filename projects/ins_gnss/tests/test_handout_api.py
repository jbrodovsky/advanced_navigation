"""Keep the Project 3 handout and the starter package in agreement.

Every public name, attribute, and signature that ``book/assignments/
m3-ins-gnss.qmd`` shows students must exist in ``ins_gnss`` with the same
spelling and parameter names, and the numbers it quotes must match the code.
If you rename something in the package, update the handout in the same change
(and vice versa). The tests skip when the book sources are not next to the
package.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
from pathlib import Path
import re
import unittest

from ins_gnss import datasets, eskf, evaluation, run_all, types

HANDOUT = Path(__file__).resolve().parents[3] / "book" / "assignments" / "m3-ins-gnss.qmd"
MODULES = ("datasets", "earth", "rotations", "types", "strapdown", "eskf", "evaluation", "run_all", "cli")

# Signatures exactly as the handout prints them -> the callable they describe.
SIGNATURES = {
    "mechanize(state, accel_mps2, gyro_rps, dt_s)": "ins_gnss.strapdown.mechanize",
    "coarse_alignment(mean_specific_force_b, yaw_rad)": "ins_gnss.strapdown.coarse_alignment",
    "error_state_jacobian(state, specific_force_b)": "ins_gnss.eskf.error_state_jacobian",
    "discrete_process_noise(state, noise, dt_s)": "ins_gnss.eskf.discrete_process_noise",
    "gnss_position_residual(state, gnss, lever_arm_b)": "ins_gnss.eskf.gnss_position_residual",
    "joseph_update(P, nu, H, R)": "ins_gnss.eskf.joseph_update",
    "inject_error_state(state, delta_x)": "ins_gnss.eskf.inject_error_state",
    "reset_covariance(P, delta_theta)": "ins_gnss.eskf.reset_covariance",
    "gnss_velocity_residual(state, gnss, lever_arm_b, gyro_rps)": "ins_gnss.eskf.gnss_velocity_residual",
    "flt.predict(accel_mps2, gyro_rps, dt_s)": "ins_gnss.eskf.INSGNSSFilter.predict",
    "flt.update_gnss_position(gnss, R=None, *, gate=None)": "ins_gnss.eskf.INSGNSSFilter.update_gnss_position",
    "load_mems_nav_trial(..., baseline_time_offset_s=...)": "ins_gnss.datasets.load_mems_nav_trial",
}

# Objects that the handout refers to through a conventional variable name.
INSTANCE_NAMES = {
    "flt": [eskf.INSGNSSFilter],
    "trial": [types.MEMSNavTrial],
    "fix": [types.GNSSSample],
}

STUBS = (
    "ins_gnss.strapdown.mechanize",
    "ins_gnss.strapdown.coarse_alignment",
    "ins_gnss.eskf.error_state_jacobian",
    "ins_gnss.eskf.discrete_process_noise",
    "ins_gnss.eskf.gnss_position_residual",
    "ins_gnss.eskf.joseph_update",
    "ins_gnss.eskf.inject_error_state",
    "ins_gnss.eskf.reset_covariance",
)


def _resolve(dotted: str):
    """Import the longest importable module prefix, then walk attributes."""
    parts = dotted.split(".")
    for split in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:split]))
        except ModuleNotFoundError:
            continue
        for attribute in parts[split:]:
            obj = getattr(obj, attribute)
        return obj
    raise ModuleNotFoundError(dotted)


def _has_attribute(cls: type, name: str) -> bool:
    if dataclasses.is_dataclass(cls) and name in {field.name for field in dataclasses.fields(cls)}:
        return True
    return hasattr(cls, name)


def _parameter_names(obj) -> list[str]:
    names = list(inspect.signature(obj).parameters)
    return names[1:] if names and names[0] in ("self", "cls") else names


def _printed_names(printed: str) -> list[str]:
    inside = printed.split("(", 1)[1].rsplit(")", 1)[0]
    names = [part.split("=")[0].strip() for part in inside.split(",")]
    return [name for name in names if name and name not in ("*", "...")]


@unittest.skipIf(not HANDOUT.exists(), f"Project 3 handout not found at {HANDOUT}")
class HandoutApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = HANDOUT.read_text()

    def test_import_statements_resolve(self) -> None:
        pattern = re.compile(r"from\s+(ins_gnss(?:\.\w+)*)\s+import\s+(\([^)]*\)|[^\n]+)")
        found = 0
        for module_name, names in pattern.findall(self.text):
            module = importlib.import_module(module_name)
            for name in re.findall(r"\w+", re.sub(r"#[^\n]*", "", names)):
                found += 1
                with self.subTest(module=module_name, name=name):
                    self.assertTrue(hasattr(module, name), f"{module_name}.{name} is in the handout but not in the package")
        self.assertGreater(found, 30)

    def test_dotted_references_resolve(self) -> None:
        for dotted in sorted(set(re.findall(r"\bins_gnss(?:\.\w+)+", self.text))):
            with self.subTest(reference=dotted):
                _resolve(dotted)

    def test_backticked_names_exist(self) -> None:
        names = set(re.findall(r"`(\w+)(?:\(|`)", self.text))
        modules = [importlib.import_module(f"ins_gnss.{name}") for name in MODULES]
        classes = [eskf.INSGNSSFilter, types.GNSSSample, types.MEMSNavTrial, types.NavState]
        skip = {"README", "run_all", "ins_gnss", "bearing", "input", "baseline", "filterpy", "ADVNAV_DATA_DIR"}
        for name in sorted(names - skip):
            if not re.fullmatch(r"[a-z]+(_[a-z0-9]+)+|[A-Z][A-Za-z]+|[A-Z][A-Z0-9_]+", name):
                continue  # single words, file names, and plain prose
            with self.subTest(name=name):
                self.assertTrue(
                    any(hasattr(module, name) for module in modules)
                    or any(_has_attribute(cls, name) for cls in classes)
                    or name in run_all.TUNING,
                    f"`{name}` is named in the handout but not found in the package",
                )

    def test_instance_attributes_exist(self) -> None:
        for variable, classes in INSTANCE_NAMES.items():
            for attribute in sorted(set(re.findall(rf"\b{variable}\.(\w+)", self.text))):
                for cls in classes:
                    with self.subTest(reference=f"{variable}.{attribute}", cls=cls.__name__):
                        self.assertTrue(_has_attribute(cls, attribute), f"{cls.__name__} has no {attribute!r}")

    def test_printed_signatures_match_the_code(self) -> None:
        for printed, target in SIGNATURES.items():
            with self.subTest(signature=printed):
                self.assertIn(printed, self.text, "signature no longer printed in the handout; update SIGNATURES")
                expected = _printed_names(printed.replace("flt.", ""))
                actual = _parameter_names(_resolve(target))
                if "..." in printed:
                    self.assertTrue(set(expected) <= set(actual), target)
                else:
                    self.assertEqual(actual[: len(expected)], expected, target)

    def test_student_functions_are_stubs_in_the_starter(self) -> None:
        # Read the files, not the imported objects (the instructor's checker patches those).
        for dotted in STUBS:
            module_name, function_name = dotted.rsplit(".", 1)
            tree = ast.parse(Path(importlib.import_module(module_name).__file__).read_text())
            with self.subTest(function=dotted):
                (function,) = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name]
                raises = [node for node in ast.walk(function) if isinstance(node, ast.Raise)]
                self.assertTrue(any("NotImplementedError" in ast.unparse(node) for node in raises))

    def test_documented_numbers_match_the_code(self) -> None:
        self.assertEqual(datasets.BASELINE_TIME_OFFSET_S, 1.0)
        self.assertIn("`BASELINE_TIME_OFFSET_S` (1 s)", self.text)
        self.assertEqual(run_all.OUTAGE_DURATIONS_S, (10.0, 30.0, 60.0, 120.0))
        self.assertEqual((run_all.OUTAGE_FIRST_START_S, run_all.OUTAGE_SPACING_S), (300.0, 400.0))
        self.assertIn("first_start_s=start + 300, spacing_s=400", self.text)
        self.assertEqual(run_all.TUNING["departure_speed_mps"], 5.0)
        low, high = evaluation.chi2_interval(3)
        self.assertIn(f"$[{low:.2f}, {high:.2f}]$", self.text)
        self.assertIn(f"{evaluation.chi2_gate(3, 0.999):.2f}", self.text)
        valid = inspect.signature(evaluation.reference_valid_mask).parameters
        self.assertEqual((valid["max_disagreement_m"].default, valid["max_time_to_fix_s"].default), (10.0, 0.5))
        schedule = inspect.signature(evaluation.schedule_outages).parameters
        self.assertEqual((schedule["min_speed_mps"].default, schedule["min_fix_fraction"].default), (5.0, 0.9))
        recommended = [name for name, note in datasets.TRIAL_NOTES.items() if note.status == "recommended"]
        excluded = [name for name, note in datasets.TRIAL_NOTES.items() if note.status == "excluded"]
        caution = [name for name, note in datasets.TRIAL_NOTES.items() if note.status == "caution"]
        for name in recommended + excluded:
            self.assertIn(f"`{name}`", self.text)
        self.assertEqual(len(recommended), 5)
        self.assertIn("all five recommended trials", self.text)
        self.assertIn(f"the {['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'][len(caution)]} others", self.text)
        self.assertIn("recommended_trials()[1]", self.text)
        self.assertEqual(recommended[1], "2023-08-09_163741")

    def test_rubric_adds_up_to_100(self) -> None:
        rubric = self.text.split("{#sec-project3-grading-rubric}", 1)[1].split("\n## ", 1)[0]
        points = [int(match) for match in re.findall(r"^\| [A-Z][^|]*\| (\d+) +\|", rubric, flags=re.MULTILINE)]
        self.assertEqual(sum(points), 100, points)

    def test_reserved_section_ids(self) -> None:
        for section in ("logistics", "data", "evaluation-protocol", "grading-rubric"):
            self.assertIn(f"{{#sec-project3-{section}}}", self.text)


if __name__ == "__main__":
    unittest.main()
