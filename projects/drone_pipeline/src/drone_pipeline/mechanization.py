"""Mechanization starter interfaces for the drone pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class PropagationParams:
    gravity_world: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -9.81], dtype=np.float64)
    )


def propagate_state(x: np.ndarray, acc: np.ndarray, drpy: np.ndarray, dt: float, params: PropagationParams):
    """Starter interface for the Part A inertial propagation assignment."""
    raise NotImplementedError(
        "students should implement propagate_state(...) for the 15-state mechanization model defined in Project 1a."
    )
