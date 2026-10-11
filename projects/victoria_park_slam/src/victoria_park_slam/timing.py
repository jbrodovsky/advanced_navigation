"""Event-driven synchronization of 40 Hz odometry with ~4.7 Hz laser scans.

The vehicle logs speed and steering every 25 ms and a laser scan every
213-215 ms, on the same clock but never at the same instants. A filter that
simply grabs "the nearest odometry sample" for each scan, or that integrates
one big step per scan, puts every landmark observation at the wrong pose by up
to a few decimeters at 6 m/s, and in a turn the heading error swings the far
trees by much more. The fix is an event-driven merge:

1. Treat odometry sample ``k`` (time ``t_k``) as a zero-order hold over
   ``[t_k, t_{k+1})``.
2. Between two scans, propagate through every odometry interval in turn.
3. Split the last interval at the laser timestamp: propagate up to the scan,
   apply the measurement update, then continue with the rest of the interval.

:func:`scan_steps` yields exactly those propagation pieces for each scan. The
first laser scan (21.819 s) precedes the first odometry sample (21.940 s), so
there is no motion information for it; scans earlier than the first odometry
sample are skipped.

Noise note: each ``predict`` call in the harness draws a fresh control-noise
sample, so control noise is effectively sampled at 40 Hz. Independent errors at
40 Hz average out, which shrinks the pose spread they produce; tune ``Q_u`` for
this step rate (see :class:`~victoria_park_slam.types.MotionModelConfig`).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .types import VictoriaParkDataset


@dataclass(frozen=True, slots=True)
class ControlStep:
    """One propagation piece: hold ``(speed, steering)`` for ``dt_s`` seconds.

    ``speed_mps`` is the raw rear-left encoder speed as logged (apply the
    encoder correction in your motion model). ``odometry_index`` is the sample
    the values came from.
    """

    speed_mps: float
    steering_rad: float
    dt_s: float
    odometry_index: int


@dataclass(frozen=True, slots=True)
class ScanStep:
    """Everything needed to bring the filter to one laser scan.

    Propagate through ``controls`` in order, then update with scan
    ``scan_index`` taken at ``time_s``. The durations of ``controls`` add up to
    the time since the previous yielded scan (or since the start).
    """

    scan_index: int
    time_s: float
    controls: tuple[ControlStep, ...]


def scan_steps(
    dataset: VictoriaParkDataset,
    start_scan: int = 0,
    stop_scan: int | None = None,
) -> Iterator[ScanStep]:
    """Yield the propagation pieces and update time for scans ``[start_scan, stop_scan)``.

    The run starts at the later of the first odometry sample and the laser
    time of ``start_scan``; scans before that instant are skipped (with
    ``start_scan = 0`` this skips only scan 0). The scan at the start instant,
    if any, is yielded with no controls.
    """

    t_odo = np.asarray(dataset.odometry_time_s, dtype=np.float64)
    speed = np.asarray(dataset.speed_mps, dtype=np.float64)
    steering = np.asarray(dataset.steering_rad, dtype=np.float64)
    t_laser = np.asarray(dataset.laser_time_s, dtype=np.float64)
    if t_odo.size == 0:
        raise ValueError("dataset has no odometry")
    if np.any(np.diff(t_odo) <= 0.0):
        raise ValueError("odometry timestamps must be strictly increasing")

    stop = t_laser.size if stop_scan is None else min(int(stop_scan), t_laser.size)
    start = max(0, int(start_scan))
    if start >= stop:
        return

    t_current = max(float(t_odo[0]), float(t_laser[start]))
    # Odometry sample whose zero-order hold covers t_current.
    k = int(np.searchsorted(t_odo, t_current, side="right")) - 1
    n_odo = t_odo.size

    for scan in range(start, stop):
        t_scan = float(t_laser[scan])
        if t_scan < t_current:
            continue  # before the run start (no motion information for it)
        controls: list[ControlStep] = []
        while t_current < t_scan:
            t_next = float(t_odo[k + 1]) if k + 1 < n_odo else np.inf
            t_end = min(t_next, t_scan)
            controls.append(ControlStep(float(speed[k]), float(steering[k]), t_end - t_current, k))
            t_current = t_end
            if t_current >= t_next:
                k += 1
        yield ScanStep(scan, t_scan, tuple(controls))
