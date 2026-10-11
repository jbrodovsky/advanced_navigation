"""Reference tree extractor for Project 2 (instructor only).

A line-by-line port of the ACFR detector ``data/victoria-park/detectTreesI16.m``
(J. Guivant, 1999). The MATLAB constants live in
``victoria_park_slam.types.TreeExtractorConfig``; with the defaults this
function returns exactly what the MATLAB returns, except that bearings are
converted from the scanner angle ``A`` to the vehicle frame
(``beta = A - pi/2``).

MATLAB-to-Python notes (the places a port usually goes wrong):

* ``ii2`` (segment starts) and ``ii2u`` (segment ends) index the list of
  *valid* returns ``R1``, not the 361-beam scan. So do the mid-point indices
  used for the range.
* ``flag`` marks segments for **deletion**: both segments of a pair whose
  facing ends are closer than 3 m, and the farther segment of a pair less than
  1 degree apart. Only unflagged segments continue.
* The minimum-range (1 m) and scan-edge tests use the segment's *first* point
  (``R5``, ``A5``) and the edge test also its last angle (``A5u``).
* ``daa = 5*pi/306`` (about 2.94 degrees) is what the file says, even if
  ``5*pi/360`` was probably meant.
* The 5-degree split and 1-degree occlusion thresholds are exact multiples of
  the 0.5-degree beam spacing, so a gap of exactly 10 (or 2) beams is decided
  by floating-point rounding, differently at different beam indices. The port
  reproduces MATLAB's arithmetic (``k*pi/360``) to match it bit for bit; a
  sensible re-implementation puts thresholds between beam multiples.
* MATLAB's ``(ii4+ii4u)/2`` with 1-based indices maps to ``(s+e)/2`` with
  0-based indices, and ``floor``/``ceil`` commute with the shift.
"""

from __future__ import annotations

import math

import numpy as np

from victoria_park_slam.types import TreeExtractorConfig, TreeObservation

DEFAULT_CONFIG = TreeExtractorConfig()


def detect_trees_scanner_frame(scan_ranges_m: np.ndarray, config: TreeExtractorConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Port of ``detectTreesI16``: rows ``[range, scanner angle A, diameter, n_points]``."""

    RR = np.asarray(scan_ranges_m, dtype=np.float64).reshape(-1)
    # Same arithmetic as MATLAB's AAr = [0:360]*pi/360 (multiply, then divide):
    # gaps of exactly 10 beams (5 deg) or 2 beams (1 deg) sit on the thresholds,
    # and floating-point rounding decides them, so the order of operations matters.
    AA = np.arange(RR.size) * math.pi / 360.0
    with np.errstate(invalid="ignore"):
        ii1 = np.flatnonzero(np.isfinite(RR) & (RR < config.max_range_m))
    if ii1.size < 1:
        return np.zeros((0, 4))
    R1, A1 = RR[ii1], AA[ii1]

    # Split at range jumps or angular gaps between consecutive valid returns.
    breaks = np.flatnonzero((np.abs(np.diff(R1)) > config.range_jump_m) | (np.diff(A1) > config.angle_gap_rad))
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [ii1.size - 1]])
    L2 = starts.size
    R2, A2, R2u, A2u = R1[starts], A1[starts], R1[ends], A1[ends]
    x2, y2 = R2 * np.cos(A2), R2 * np.sin(A2)
    x2u, y2u = R2u * np.cos(A2u), R2u * np.sin(A2u)

    flag = np.zeros(L2, dtype=bool)
    # Isolation: end of segment j vs start of segments j+1..j+3.
    for k in range(1, config.isolation_neighbors + 1):
        if L2 <= k:
            break
        dl2 = (x2[k:] - x2u[:-k]) ** 2 + (y2[k:] - y2u[:-k]) ** 2
        hit = np.flatnonzero(dl2 < config.isolation_m**2)
        flag[hit] = True
        flag[hit + k] = True
    # Occlusion: adjacent segments closer than 1 degree; drop the one behind.
    if L2 > 1:
        j = np.arange(L2 - 1)
        hit = j[(A2[j + 1] - A2u[j]) < config.occlusion_angle_rad]
        behind = (R2[hit + 1] > R2u[hit]).astype(int)
        flag[hit + behind] = True

    keep = ~flag
    s, e = starts[keep], ends[keep]
    R4, R4u, A4, A4u = R2[keep], R2u[keep], A2[keep], A2u[keep]
    x4, y4, x4u, y4u = x2[keep], y2[keep], x2u[keep], y2u[keep]

    # Chord (first to last point) shorter than M5.
    ok = (x4 - x4u) ** 2 + (y4 - y4u) ** 2 < config.max_chord_m**2
    s, e, R5, R5u, A5, A5u = s[ok], e[ok], R4[ok], R4u[ok], A4[ok], A4u[ok]
    if s.size < 1:
        return np.zeros((0, 4))

    # Not too close, not touching the scan edges.
    ok = (R5 > config.min_range_m) & (A5 > config.edge_margin_rad) & (A5u < math.pi - config.edge_margin_rad)
    s, e, R5, R5u, A5, A5u = s[ok], e[ok], R5[ok], R5u[ok], A5[ok], A5u[ok]
    if s.size < 1:
        return np.zeros((0, 4))

    diameter = (A5u + config.beam_width_rad - A5) * (R5 + R5u) / 2.0
    compact = np.abs(R5 - R5u) < diameter * config.compactness_ratio
    s, e, A5, A5u, diameter = s[compact], e[compact], A5[compact], A5u[compact], diameter[compact]
    if s.size < 1:
        return np.zeros((0, 4))

    mid = (s + e) / 2.0
    range_mid = (R1[np.floor(mid).astype(int)] + R1[np.ceil(mid).astype(int)]) / 2.0
    return np.column_stack([range_mid + diameter / 2.0, (A5 + A5u) / 2.0, diameter, e - s + 1])


def extract_tree_landmarks(
    scan_ranges_m: np.ndarray,
    *,
    pose_guess: np.ndarray | None = None,
    config: TreeExtractorConfig = DEFAULT_CONFIG,
) -> list[TreeObservation]:
    """Reference implementation of the student stub (vehicle-frame bearings)."""

    rows = detect_trees_scanner_frame(scan_ranges_m, config)
    return [
        TreeObservation(
            range_m=float(r),
            bearing_rad=float(a - math.pi / 2.0),
            diameter_m=float(d),
            num_points=int(n),
        )
        for r, a, d, n in rows
    ]
