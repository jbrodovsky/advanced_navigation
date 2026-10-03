"""Swap the reference solutions into the installed ``victoria_park_slam`` package.

``install()`` replaces the student stubs (the tree extractor and the FastSLAM
classes) in every loaded ``victoria_park_slam`` module, including modules that
already imported a stub by name, so the student tests and ``run_all`` exercise
the reference code without editing the starter package.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fastslam_solution  # noqa: E402
import landmarks_solution  # noqa: E402

REPLACEMENTS = {
    "victoria_park_slam.landmarks": {
        "extract_tree_landmarks": landmarks_solution.extract_tree_landmarks,
    },
    "victoria_park_slam.fastslam": {
        "LandmarkEKF": fastslam_solution.LandmarkEKFSolution,
        "FastSLAM": fastslam_solution.FastSLAMSolution,
    },
}

PARTS = {
    "extractor": {"victoria_park_slam.landmarks"},
    "fastslam": {"victoria_park_slam.fastslam"},
}


def install(parts: tuple[str, ...] = ("extractor", "fastslam")) -> list[str]:
    """Patch the stubs for the requested parts; return the patched names."""
    importlib.import_module("victoria_park_slam")
    importlib.import_module("victoria_park_slam.run_all")
    modules = set().union(*(PARTS[part] for part in parts))
    patched = []
    for module_name, names in REPLACEMENTS.items():
        if module_name not in modules:
            continue
        module = importlib.import_module(module_name)
        for name, replacement in names.items():
            original = getattr(module, name)
            for loaded_name, loaded in list(sys.modules.items()):
                if loaded_name.split(".")[0] == "victoria_park_slam" and getattr(loaded, name, None) is original:
                    setattr(loaded, name, replacement)
            patched.append(f"{module_name}.{name}")
    return patched
