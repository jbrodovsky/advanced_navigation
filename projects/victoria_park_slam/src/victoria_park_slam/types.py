"""Typed records and configuration for the Victoria Park FastSLAM assignment.

Frame conventions used everywhere in this package
--------------------------------------------------
* World frame: planar, x and y in meters, heading measured counter-clockwise
  from +x. Its origin and orientation are arbitrary (the run starts at
  ``(0, 0, 0)``); GPS is compared only after an SE(2) alignment
  (:mod:`victoria_park_slam.evaluation`).
* Vehicle frame: origin at the center of the rear axle, x forward, y to the
  vehicle's left.
* Pose carried by the filter: the **laser pose** ``[x_L, y_L, phi]``, the world
  position of the laser (which is also where the GPS antenna is mounted) and
  the vehicle heading ``phi``. See :class:`VehicleGeometry` for the lever arm.
* Laser bearings ``beta`` are in the vehicle-aligned frame centered on the
  laser: ``beta = A - pi/2`` where ``A`` (0..pi) is the scanner's own angle,
  so ``beta = 0`` is straight ahead and positive bearings are to the left.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import math
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class VictoriaParkDataset:
    """Loaded Victoria Park data with units normalized for the assignment.

    ``laser_ranges_m`` holds the decoded ranges (low 13 bits, centimeters to
    meters, ``NaN`` where the scanner reported no return); ``laser_raw`` keeps
    the original ``uint16`` words, flags included. GPS ``Lo_m``/``La_m`` are
    east/north in meters in the receiver's own local frame.
    """

    root: Path
    laser_ranges_m: np.ndarray
    laser_time_s: np.ndarray
    speed_mps: np.ndarray
    steering_rad: np.ndarray
    odometry_time_s: np.ndarray
    gps_lat_m: np.ndarray | None = None
    gps_lon_m: np.ndarray | None = None
    gps_time_s: np.ndarray | None = None
    laser_raw: np.ndarray | None = None

    @property
    def gps_xy_m(self) -> np.ndarray | None:
        """GPS fixes as an ``(N, 2)`` array of ``[east, north]`` = ``[Lo_m, La_m]``."""
        if self.gps_lon_m is None or self.gps_lat_m is None:
            return None
        return np.column_stack([self.gps_lon_m, self.gps_lat_m])

    @property
    def num_scans(self) -> int:
        return int(self.laser_ranges_m.shape[0])


@dataclass(frozen=True, slots=True)
class VehicleGeometry:
    """Victoria Park utility vehicle geometry (``info.txt`` and ``car.bmp``).

    All offsets are in the vehicle frame (origin at the rear-axle center, x
    forward, y left):

    * ``wheelbase_m`` ``L``: rear axle to front axle.
    * ``encoder_offset_m`` ``H``: the speed encoder is on the rear **left**
      wheel, ``H`` to the left of the centerline.
    * ``laser_forward_m`` ``a``: the laser (and GPS antenna) is ``0.95 m`` in
      front of the front axle, so ``a = L + 0.95 = 3.78 m`` ahead of the rear
      axle. (``car.bmp`` labels the 0.95 m front-axle-to-laser distance ``a``;
      the kinematic model needs the distance from the rear axle.)
    * ``laser_left_m`` ``b``: the laser is ``0.50 m`` to the **left** of the
      centerline, on the same side as the encoder.

    The left-side placement follows the drawing (read as a top view, front to
    the left) and is confirmed on the data: tree detections projected along
    dead reckoning form the tightest clusters during turns for
    ``(a, b) = (3.78, +0.50)``.
    """

    wheelbase_m: float = 2.83
    encoder_offset_m: float = 0.76
    laser_forward_m: float = 3.78
    laser_left_m: float = 0.50


@dataclass(slots=True)
class MotionModelConfig:
    """Vehicle model and control-noise settings for particle prediction.

    ``control_noise_covariance`` ``Q_u`` is the 2x2 covariance of additive
    noise on the *measured* controls ``[encoder speed (m/s), steering (rad)]``,
    drawn independently **once per prediction call**. The harness calls
    ``predict`` once per odometry interval (40 Hz, 25 ms), so ``Q_u`` is a
    per-25-ms-step value. Independent draws at 40 Hz average out quickly (the
    pose spread after one second grows only like ``sqrt(40)`` steps, not 40),
    so a "physically reasonable" one-sigma value is far too small; tune ``Q_u``
    to the step rate you actually use and rescale it if you change that rate.
    """

    geometry: VehicleGeometry = field(default_factory=VehicleGeometry)
    control_noise_covariance: np.ndarray = field(
        default_factory=lambda: np.diag([0.5**2, np.deg2rad(3.0) ** 2])
    )
    use_encoder_correction: bool = True


@dataclass(slots=True)
class MeasurementModelConfig:
    """Range-bearing noise for tree observations ``z = [r, beta]``."""

    range_std_m: float = 0.50
    bearing_std_rad: float = float(np.deg2rad(2.0))
    max_range_m: float = 75.0

    @property
    def covariance(self) -> np.ndarray:
        """Measurement covariance ``R = diag(range_std^2, bearing_std^2)``."""
        return np.diag([self.range_std_m**2, self.bearing_std_rad**2])


@dataclass(frozen=True, slots=True)
class TreeExtractorConfig:
    """Parameters of the original ACFR tree detector (``detectTreesI16.m``).

    The defaults reproduce the MATLAB constants (named in brackets). Angles in
    this record refer to the scanner angle ``A`` (0..pi); bearings returned by
    an extractor are converted to the vehicle frame (``beta = A - pi/2``).
    """

    max_range_m: float = 75.0  # [M11] returns at or beyond this are ignored
    min_range_m: float = 1.0  # [M10] reject segments whose first point is this close
    range_jump_m: float = 1.5  # [M2] split where consecutive valid ranges jump more
    angle_gap_rad: float = math.radians(5.0)  # [M2a] split where valid beams are this far apart
    isolation_m: float = 3.0  # [M3] drop segments whose ends are this close to a neighbor
    isolation_neighbors: int = 3  # the MATLAB checks the next 1, 2 and 3 segments
    occlusion_angle_rad: float = math.radians(1.0)  # [daMin2] adjacent segments: drop the far one
    edge_margin_rad: float = 5.0 * math.pi / 306.0  # [daa] about 2.9 deg from either scan edge
    max_chord_m: float = 1.0  # [M5] first-to-last point distance must be shorter
    beam_width_rad: float = math.radians(0.5)  # added to the angular extent for the diameter
    compactness_ratio: float = 1.0 / 3.0  # keep if |R_first - R_last| < ratio * diameter


@dataclass(slots=True)
class TreeObservation:
    """Range-bearing observation of one tree trunk center.

    ``range_m`` is measured from the laser to the trunk center and
    ``bearing_rad`` is ``beta`` in the vehicle frame (``0`` = straight ahead,
    positive to the left). ``diameter_m`` and ``num_points`` are front-end
    metadata; the FastSLAM update uses only ``z = [range_m, bearing_rad]``.
    """

    range_m: float
    bearing_rad: float
    diameter_m: float
    num_points: int | None = None

    @property
    def z(self) -> np.ndarray:
        """The measurement vector ``[range_m, bearing_rad]``."""
        return np.array([self.range_m, self.bearing_rad], dtype=np.float64)


class LandmarkTrackState(StrEnum):
    """Lifecycle labels for per-particle landmark tracks."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    INACTIVE = "inactive"


@dataclass(slots=True)
class LandmarkState:
    """Estimated 2D landmark conditioned on one particle path, plus bookkeeping.

    ``observation_count`` counts every measurement fused into the landmark,
    including the one that created it, so a landmark born and then associated
    ``k`` times has ``observation_count == k + 1``. ``first_seen_s`` and
    ``last_seen_s`` are laser timestamps.
    """

    mean_xy_m: np.ndarray
    covariance: np.ndarray
    observation_count: int = 1
    first_seen_s: float = 0.0
    last_seen_s: float = 0.0
    track_state: LandmarkTrackState = LandmarkTrackState.TENTATIVE

    @property
    def association_count(self) -> int:
        """Successful associations after birth (``observation_count - 1``)."""
        return self.observation_count - 1

    def age_s(self, now_s: float) -> float:
        """Time since the landmark was created."""
        return float(now_s - self.first_seen_s)


class AssociationKind(StrEnum):
    """Outcome of associating one observation inside one particle."""

    MATCH = "match"  # d^2 < gate for this landmark (and it won the mutual-exclusion contest)
    NEW = "new"  # d^2 > birth threshold for every landmark: spawn a tentative landmark
    DISCARD = "discard"  # ambiguous: neither matched nor clearly new


@dataclass(frozen=True, slots=True)
class Association:
    """One association decision.

    ``landmark_id`` is the matched landmark for ``MATCH``, the nearest landmark
    (smallest ``d^2``) for ``DISCARD`` (``None`` if the particle has no
    landmarks), and ``None`` for ``NEW``. ``mahalanobis_sq`` is ``d^2`` to that
    landmark (``inf`` when there is none).
    """

    kind: AssociationKind
    landmark_id: int | None = None
    mahalanobis_sq: float = math.inf
