"""Laser front end: tree-trunk observations from one Victoria Park scan.

``extract_tree_landmarks`` is yours to write. The reference point is the
original ACFR detector shipped with the data (``data/victoria-park/
detectTreesI16.m``). Its logic, with the defaults collected in
:class:`~victoria_park_slam.types.TreeExtractorConfig`, is:

1. Keep returns closer than 75 m (``NaN`` = no return is dropped too).
2. Split the remaining returns into segments wherever consecutive valid
   ranges jump by more than 1.5 m **or** consecutive valid beams are more than
   5 degrees apart (a gap of missing returns).
3. Isolation test: if the end of a segment is within 3 m of the start of one
   of the next three segments, drop **both** (trunks stand alone; bushes,
   fences and parked cars do not).
4. Occlusion test: if two segments are less than 1 degree apart, drop the one
   further away (its visible edge is the occluding object's, so its width and
   center are wrong).
5. Keep segments whose chord (first point to last point) is shorter than 1 m,
   whose first point is farther than 1 m, and which do not touch the scan edges
   (start angle above about 2.9 degrees, end angle below 180 minus that).
6. Diameter: ``d = (dA + 0.5 deg) * (R_first + R_last) / 2`` where ``dA`` is
   the angle between the first and last beam; the extra 0.5 degree is the beam
   width (a trunk hit by ``n`` beams subtends about ``n``, not ``n - 1``, beam
   spacings).
7. Compactness: keep only segments with ``|R_first - R_last| < d / 3`` (a
   trunk is round; a wall seen obliquely is not).
8. Observation: bearing = midpoint of the first and last beam angles; range =
   the range at the middle beam of the segment plus ``d / 2`` (the laser hits
   the near surface; the landmark is the trunk center).

The MATLAB returns the scanner angle ``A``. This package uses vehicle-frame
bearings, ``beta = A - pi/2``, so convert before returning.
"""

from __future__ import annotations

import numpy as np

from .types import TreeObservation


def extract_tree_landmarks(
    scan_ranges_m: np.ndarray,
    *,
    pose_guess: np.ndarray | None = None,
) -> list[TreeObservation]:
    """Return tree-trunk observations from one decoded laser scan.

    Parameters
    ----------
    scan_ranges_m:
        ``(361,)`` decoded ranges in meters for one scan, beam ``k`` at scanner
        angle ``A_k = k * 0.5 deg``; ``NaN`` marks "no return" (see
        :func:`~victoria_park_slam.datasets.decode_laser_ranges`).
    pose_guess:
        Optional laser pose ``[x, y, phi]``. Use it only for diagnostics and
        plots (for example to draw detections in the world); the extraction
        itself must not depend on it, and never on GPS.

    Returns
    -------
    list[TreeObservation]
        One observation per accepted trunk, with ``range_m`` from the laser to
        the trunk center, ``bearing_rad`` in the vehicle frame
        (``beta = A - pi/2``, positive to the left), ``diameter_m``, and
        ``num_points`` (valid returns in the segment). Return an empty list
        when the scan has no trees.

    You may add keyword-only tuning parameters with defaults (for example a
    ``config: TreeExtractorConfig``), but keep this call working unchanged:
    the tests and ``run_all`` call ``extract_tree_landmarks(scan)``.
    """
    raise NotImplementedError("students should implement tree extraction from laser scans")
