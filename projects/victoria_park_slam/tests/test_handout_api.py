"""Keep the Project 2 handout and the starter package in agreement.

Every public name, attribute, and signature that ``book/assignments/
m2-victoria-park.qmd`` shows students must exist in ``victoria_park_slam``
with the same spelling and parameter names. If you rename something in the
package, update the handout in the same change (and vice versa). The tests
skip when the book sources are not next to the package.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
from pathlib import Path
import re
import unittest

from victoria_park_slam import evaluation, fastslam, run_all, timing, types

HANDOUT = Path(__file__).resolve().parents[3] / "book" / "assignments" / "m2-victoria-park.qmd"

# Signatures exactly as the handout prints them -> the callable they describe.
SIGNATURES = {
    "extract_tree_landmarks(scan_ranges_m, *, pose_guess=None)": "victoria_park_slam.landmarks.extract_tree_landmarks",
    "LandmarkEKF.initialize(pose, measurement, measurement_covariance, *, time_s=0.0)": "victoria_park_slam.fastslam.LandmarkEKF.initialize",
    "landmark.innovation(pose, measurement, measurement_covariance)": "victoria_park_slam.fastslam.LandmarkEKF.innovation",
    "landmark.updated(pose, measurement, measurement_covariance, *, time_s=None)": "victoria_park_slam.fastslam.LandmarkEKF.updated",
    "slam.predict(speed_mps, steering_rad, dt_s)": "victoria_park_slam.fastslam.FastSLAM.predict",
    "slam.associate(particle, measurements)": "victoria_park_slam.fastslam.FastSLAM.associate",
    "slam.update(observations, time_s)": "victoria_park_slam.fastslam.FastSLAM.update",
    "slam.resample()": "victoria_park_slam.fastslam.FastSLAM.resample",
    "predict_laser_pose(laser_pose, centerline_speed_mps, steering_rad, dt_s, geometry)": "victoria_park_slam.utils.predict_laser_pose",
}

# Objects that the handout refers to through a conventional variable name.
INSTANCE_NAMES = {
    "slam": [fastslam.FastSLAM],
    "particle": [fastslam.FastSLAMParticle],
    "landmark": [fastslam.LandmarkEKF],
    "step": [timing.ScanStep],
    "control": [timing.ControlStep],
    "dataset": [types.VictoriaParkDataset],
}

# Fields of FastSLAM the handout names in prose.
FASTSLAM_FIELDS = (
    "gate_threshold",
    "birth_threshold",
    "new_landmark_log_likelihood",
    "resample_threshold",
    "confirm_associations",
    "tentative_timeout_s",
    "rng",
    "motion",
    "measurement",
    "history",
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
    return [name for name in names if name and name != "*"]


@unittest.skipIf(not HANDOUT.exists(), f"Project 2 handout not found at {HANDOUT}")
class HandoutApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = HANDOUT.read_text()

    def test_import_statements_resolve(self) -> None:
        pattern = re.compile(r"from\s+(victoria_park_slam(?:\.\w+)*)\s+import\s+(\([^)]*\)|[^\n]+)")
        found = 0
        for module_name, names in pattern.findall(self.text):
            module = importlib.import_module(module_name)
            for name in re.findall(r"\w+", names.split("#")[0]):
                found += 1
                with self.subTest(module=module_name, name=name):
                    self.assertTrue(hasattr(module, name), f"{module_name}.{name} is in the handout but not in the package")
        self.assertGreater(found, 20)

    def test_dotted_references_resolve(self) -> None:
        for dotted in sorted(set(re.findall(r"\bvictoria_park_slam(?:\.\w+)+", self.text))):
            with self.subTest(reference=dotted):
                _resolve(dotted)

    def test_backticked_helpers_exist(self) -> None:
        names = set(re.findall(r"`(\w+)(?:\(|`)", self.text))
        modules = [importlib.import_module(f"victoria_park_slam.{name}") for name in
                   ("utils", "evaluation", "timing", "types", "datasets", "fastslam", "landmarks", "run_all")]
        skip = {"README", "run_all", "victoria_park_slam"}
        for name in sorted(names - skip):
            if not re.fullmatch(r"[a-z]+(_[a-z0-9]+)+|[A-Z][A-Za-z]+", name):
                continue  # file names, single words, and plain prose
            with self.subTest(name=name):
                self.assertTrue(
                    any(hasattr(module, name) for module in modules) or _has_attribute(fastslam.FastSLAM, name),
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
                expected = _printed_names(printed)
                actual = _parameter_names(_resolve(target))
                self.assertEqual(actual[: len(expected)], expected, target)

    def test_fastslam_fields_and_run_all_tables(self) -> None:
        for name in FASTSLAM_FIELDS:
            with self.subTest(field=name):
                self.assertTrue(_has_attribute(fastslam.FastSLAM, name))
                self.assertTrue(f"`{name}`" in self.text or f"slam.{name}" in self.text, name)
        self.assertEqual(run_all.SEGMENT_SCANS, 2000)
        self.assertEqual(run_all.SWEEPS["particles"][1], (10, 25, 50, 100))
        self.assertEqual(run_all.SWEEPS["gate"][1], (4.61, 5.99, 9.21))
        self.assertEqual(run_all.SWEEPS["q_scale"][1], (0.25, 1.0, 4.0))
        for values in ("`(10, 25, 50, 100)`", "`(4.61, 5.99, 9.21)`", "`(0.25, 1.0, 4.0)`"):
            self.assertIn(values, self.text)
        self.assertIn("--sweep particles gate q_scale", self.text)

    def test_documented_numbers_match_the_code(self) -> None:
        geometry = types.VehicleGeometry()
        self.assertEqual((geometry.wheelbase_m, geometry.encoder_offset_m), (2.83, 0.76))
        self.assertEqual((geometry.laser_forward_m, geometry.laser_left_m), (3.78, 0.50))
        slam = fastslam.FastSLAM(particles=[])
        self.assertAlmostEqual(slam.gate_threshold, 5.99, places=2)
        self.assertAlmostEqual(slam.birth_threshold, 13.82, places=2)
        self.assertEqual((slam.resample_threshold, slam.confirm_associations, slam.tentative_timeout_s), (0.5, 3, 2.0))
        self.assertIn("2 s by default", self.text)
        valid = inspect.signature(evaluation.valid_gps_mask).parameters
        self.assertEqual((valid["max_gap_s"].default, valid["max_jump_m"].default, valid["min_run_fixes"].default), (1.0, 5.0, 5))


if __name__ == "__main__":
    unittest.main()
