"""Swap the reference solutions into the installed ``drone_pipeline`` package.

``install()`` replaces every stub (Parts A, B and C) in the package modules,
including any module that already imported the stub by name, so student
tests and ``run_all`` exercise the reference code without editing the
starter package.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import filters_solution  # noqa: E402
import mechanization_solution  # noqa: E402
import observation_solution  # noqa: E402

REPLACEMENTS = {
    "drone_pipeline.observation": {
        "build_apriltag_corner_map": observation_solution.build_apriltag_corner_map,
        "estimate_pose": observation_solution.estimate_pose,
        "estimate_measurement_covariance": observation_solution.estimate_measurement_covariance,
    },
    "drone_pipeline.mechanization": {
        "propagate_state": mechanization_solution.propagate_state,
    },
    "drone_pipeline.filters": {
        "EKFNavigator": filters_solution.EKFSolution,
        "UKFNavigator": filters_solution.UKFSolution,
        "ParticleFilterNavigator": filters_solution.ParticleFilterSolution,
    },
}


def install(parts: str = "ABC") -> list[str]:
    """Patch the stubs for the requested parts; return the patched names."""
    wanted = {
        "A": {"drone_pipeline.mechanization"},
        "B": {"drone_pipeline.observation"},
        "C": {"drone_pipeline.filters"},
    }
    modules = set().union(*(wanted[p] for p in parts.upper()))
    patched = []
    for module_name, names in REPLACEMENTS.items():
        if module_name not in modules:
            continue
        module = importlib.import_module(module_name)
        for name, replacement in names.items():
            original = getattr(module, name)
            for loaded_name, loaded in list(sys.modules.items()):
                if loaded_name.split(".")[0] == "drone_pipeline" and getattr(loaded, name, None) is original:
                    setattr(loaded, name, replacement)
            patched.append(f"{module_name}.{name}")
    return patched
