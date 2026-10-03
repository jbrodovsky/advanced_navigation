"""Keep the Project 1 handouts and the starter package in agreement.

Every public name, attribute, and signature that the handouts in
``book/assignments/m1-drone-pipeline*.qmd`` show students must exist in
``drone_pipeline`` with the same spelling and parameter names. If you rename
something in the package, update the handout in the same change (and vice
versa). The tests skip when the book sources are not next to the package.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
from pathlib import Path
import re
import unittest

from drone_pipeline import filters, run_all
from drone_pipeline.mechanization import PropagationParams
from drone_pipeline.types import DronePacket, DroneParameters, PoseObservation, TruthSample

HANDOUT_DIR = Path(__file__).resolve().parents[3] / "book" / "assignments"
HANDOUTS = sorted(HANDOUT_DIR.glob("m1-drone-pipeline*.qmd")) if HANDOUT_DIR.is_dir() else []
NAVIGATORS = (filters.EKFNavigator, filters.UKFNavigator, filters.ParticleFilterNavigator)

# Signatures exactly as the handouts print them -> the callable they describe.
SIGNATURES = {
    "propagate_state(x, acc, omg, dt, params)": "drone_pipeline.mechanization.propagate_state",
    "build_apriltag_corner_map()": "drone_pipeline.observation.build_apriltag_corner_map",
    "estimate_pose(packet, camera_matrix, dist_coeffs, tag_map, T_camera_body)": "drone_pipeline.observation.estimate_pose",
    "estimate_measurement_covariance(observations, truth_samples)": "drone_pipeline.observation.estimate_measurement_covariance",
    "nav.predict(acc, omg, dt)": [f"{cls.__module__}.{cls.__name__}.predict" for cls in NAVIGATORS],
    "nav.update(z)": [f"{cls.__module__}.{cls.__name__}.update" for cls in NAVIGATORS],
    "PoseObservation(t, position, rpy, reprojection_error, num_points)": "drone_pipeline.types.PoseObservation",
}

# Objects that the handouts refer to through a conventional variable name.
INSTANCE_NAMES = {
    "params": [DroneParameters],
    "packet": [DronePacket],
    "pose": [PoseObservation],
    "nav": list(NAVIGATORS),
    "sample": [TruthSample],
}


def _handout_text() -> str:
    return "\n".join(path.read_text() for path in HANDOUTS)


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
    if inspect.isclass(obj) and dataclasses.is_dataclass(obj):
        return [field.name for field in dataclasses.fields(obj) if field.init]
    names = list(inspect.signature(obj).parameters)
    return names[1:] if names and names[0] == "self" else names


@unittest.skipIf(not HANDOUTS, f"Project 1 handouts not found in {HANDOUT_DIR}")
class HandoutApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _handout_text()

    def test_import_statements_resolve(self) -> None:
        pattern = re.compile(r"from\s+(drone_pipeline(?:\.\w+)*)\s+import\s+(\([^)]*\)|[^\n]+)")
        found = 0
        for module_name, names in pattern.findall(self.text):
            module = importlib.import_module(module_name)
            for name in re.findall(r"\w+", names.split("#")[0]):
                found += 1
                with self.subTest(module=module_name, name=name):
                    self.assertTrue(hasattr(module, name), f"{module_name}.{name} is in a handout but not in the package")
        self.assertGreater(found, 10)

    def test_dotted_references_resolve(self) -> None:
        for dotted in sorted(set(re.findall(r"\bdrone_pipeline(?:\.\w+)+", self.text))):
            with self.subTest(reference=dotted):
                _resolve(dotted)

    def test_short_module_references_resolve(self) -> None:
        references = set(re.findall(r"`(utils|config|evaluation|run_all|filters|observation)\.(\w+)", self.text))
        for module_name, attribute in sorted(references):
            if attribute == "py":  # a file name such as `filters.py`
                continue
            with self.subTest(reference=f"{module_name}.{attribute}"):
                module = importlib.import_module(f"drone_pipeline.{module_name}")
                self.assertTrue(hasattr(module, attribute))

    def test_instance_attributes_exist(self) -> None:
        for variable, classes in INSTANCE_NAMES.items():
            for attribute in sorted(set(re.findall(rf"\b{variable}\.(\w+)", self.text))):
                for cls in classes:
                    with self.subTest(reference=f"{variable}.{attribute}", cls=cls.__name__):
                        self.assertTrue(_has_attribute(cls, attribute), f"{cls.__name__} has no {attribute!r}")

    def test_printed_signatures_match_the_code(self) -> None:
        for printed, targets in SIGNATURES.items():
            with self.subTest(signature=printed):
                self.assertIn(printed, self.text, "signature no longer printed in the handouts; update SIGNATURES")
                expected = [name.strip() for name in printed.split("(", 1)[1].rstrip(")").split(",") if name.strip()]
                for target in [targets] if isinstance(targets, str) else targets:
                    self.assertEqual(_parameter_names(_resolve(target)), expected, target)

    def test_packet_field_table_matches_dronepacket(self) -> None:
        overview = (HANDOUT_DIR / "m1-drone-pipeline.qmd").read_text()
        table = overview.split("Each packet is a `DronePacket`", 1)[1].split("\n\n", 2)[1]
        fields = set()
        for line in table.splitlines():
            if line.startswith("|"):
                fields.update(re.findall(r"`(\w+)`", line.split("|")[1]))
        self.assertGreaterEqual(fields, {"t", "id", "p1", "p4", "p0", "omg", "acc", "rpy", "img", "is_ready", "rate_field"})
        for name in fields:
            with self.subTest(field=name):
                self.assertTrue(_has_attribute(DronePacket, name))

    def test_named_helpers_exist(self) -> None:
        self.assertTrue(hasattr(PropagationParams(), "gravity_world"))
        self.assertTrue(hasattr(TruthSample, "__dataclass_fields__") and "valid" in TruthSample.__dataclass_fields__)
        for name in ("run_navigator", "initial_state_from_poses", "TUNING", "PF_STUDY_COUNTS"):
            self.assertTrue(hasattr(run_all, name), name)
        self.assertEqual(tuple(run_all.PF_STUDY_COUNTS), (250, 1000, 2500, 5000))
        self.assertIn(str(list(run_all.PF_STUDY_COUNTS)), self.text)


if __name__ == "__main__":
    unittest.main()
