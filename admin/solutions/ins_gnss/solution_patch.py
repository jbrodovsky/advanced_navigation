"""Swap the reference solutions into the installed ``ins_gnss`` package.

``install()`` replaces the student stubs (mechanization, alignment, and the
ESKF pieces) in every loaded ``ins_gnss`` module, including modules that
already imported a stub by name, so the student tests and ``run_all``
exercise the reference code without editing the starter package.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import eskf_solution  # noqa: E402
import strapdown_solution  # noqa: E402

REPLACEMENTS = {
    "ins_gnss.strapdown": {
        "mechanize": strapdown_solution.mechanize,
        "coarse_alignment": strapdown_solution.coarse_alignment,
    },
    "ins_gnss.eskf": {
        "error_state_jacobian": eskf_solution.error_state_jacobian,
        "discrete_process_noise": eskf_solution.discrete_process_noise,
        "gnss_position_residual": eskf_solution.gnss_position_residual,
        "joseph_update": eskf_solution.joseph_update,
        "inject_error_state": eskf_solution.inject_error_state,
        "reset_covariance": eskf_solution.reset_covariance,
        "gnss_velocity_residual": eskf_solution.gnss_velocity_residual,
    },
}

PARTS = {
    "A": {"ins_gnss.strapdown"},
    "B": {"ins_gnss.eskf"},
}


def install(parts: str = "AB") -> list[str]:
    """Patch the stubs for the requested parts (``"A"``, ``"B"``, or ``"AB"``); return the patched names."""
    importlib.import_module("ins_gnss")
    importlib.import_module("ins_gnss.run_all")
    modules = set().union(*(PARTS[part] for part in parts.upper()))
    patched = []
    for module_name, names in REPLACEMENTS.items():
        if module_name not in modules:
            continue
        module = importlib.import_module(module_name)
        for name, replacement in names.items():
            original = getattr(module, name)
            for loaded_name, loaded in list(sys.modules.items()):
                if loaded_name.split(".")[0] == "ins_gnss" and getattr(loaded, name, None) is original:
                    setattr(loaded, name, replacement)
            patched.append(f"{module_name}.{name}")
    return patched
