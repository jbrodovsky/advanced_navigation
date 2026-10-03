"""WGS-84 Earth model: radii of curvature, Earth and transport rate, gravity.

Everything here is provided support code (tested). The conventions are the
book's (@sec-frames-conventions, @sec-ins-gnss-wgs84-geometry): latitude
``L`` and longitude ``lambda`` in radians, ellipsoidal height ``h`` in meters,
NED navigation frame, ``R_M`` the meridian and ``R_N`` the prime-vertical
radius of curvature. (Groves writes these ``R_N`` and ``R_E``; careful when
you copy a formula from him.)

Position errors and GNSS residuals are expressed in NED meters relative to a
nominal geodetic position::

    dp_N = (R_M + h) dL,    dp_E = (R_N + h) cos(L) dlambda,    dp_D = -dh

:func:`ned_to_geodetic_delta` and :func:`geodetic_to_ned_delta` convert
between the two. They are first-order (small-offset) conversions, accurate to
millimeters for offsets of a kilometer.
"""

from __future__ import annotations

import numpy as np

# WGS-84 defining constants and the derived values used by the course.
WGS84_A = 6378137.0
"""Semi-major axis (equatorial radius) [m]."""
WGS84_F = 1.0 / 298.257223563
"""Flattening."""
WGS84_E2 = 6.69437999014e-3
"""First eccentricity squared."""
WGS84_B = WGS84_A * (1.0 - WGS84_F)
"""Semi-minor axis (polar radius) [m]."""
WGS84_GM = 3.986004418e14
"""Earth's gravitational constant [m^3/s^2]."""
OMEGA_IE = 7.292115e-5
"""Earth rotation rate [rad/s] (about 15.04 deg/h)."""

GRAVITY_EQUATOR = 9.7803253359
"""Somigliana normal gravity on the equator [m/s^2]."""
GRAVITY_POLE = 9.8321849378
"""Somigliana normal gravity at the poles [m/s^2]."""
SOMIGLIANA_K = 0.001931852652
"""Somigliana constant k = (b g_p - a g_e) / (a g_e)."""
GRAVITY_M = OMEGA_IE**2 * WGS84_A**2 * WGS84_B / WGS84_GM
"""The ratio m = omega_ie^2 a^2 b / GM (about 0.00345) in Groves eq. 2.139."""


def radii_of_curvature(latitude_rad: float) -> tuple[float, float]:
    """Return ``(R_M, R_N)``, the meridian and prime-vertical radii [m].

    ``R_M = a (1 - e^2) / (1 - e^2 sin^2 L)^(3/2)`` (north-south curvature),
    ``R_N = a / sqrt(1 - e^2 sin^2 L)`` (east-west curvature). Works on
    scalars and arrays.
    """
    s2 = np.sin(latitude_rad) ** 2
    denom = 1.0 - WGS84_E2 * s2
    r_m = WGS84_A * (1.0 - WGS84_E2) / denom**1.5
    r_n = WGS84_A / np.sqrt(denom)
    return r_m, r_n


def earth_rate_ned(latitude_rad: float) -> np.ndarray:
    """Earth rotation rate resolved in NED, ``omega_ie [cos L, 0, -sin L]`` [rad/s]."""
    return OMEGA_IE * np.array([np.cos(latitude_rad), 0.0, -np.sin(latitude_rad)])


def transport_rate_ned(latitude_rad: float, height_m: float, velocity_ned_mps: np.ndarray) -> np.ndarray:
    """Transport rate ``omega_en^n`` [rad/s]: how fast the NED frame turns as you move.

    ``[v_E / (R_N + h), -v_N / (R_M + h), -v_E tan(L) / (R_N + h)]``.
    """
    v_n, v_e, _ = np.asarray(velocity_ned_mps, dtype=np.float64)
    r_m, r_n = radii_of_curvature(latitude_rad)
    return np.array(
        [
            v_e / (r_n + height_m),
            -v_n / (r_m + height_m),
            -v_e * np.tan(latitude_rad) / (r_n + height_m),
        ]
    )


def compute_gravity(latitude_rad: float, height_m: float = 0.0) -> float:
    """Magnitude of normal gravity (gravitation plus centrifugal), positive down [m/s^2].

    Surface value from the closed-form Somigliana formula::

        g_0(L) = 9.7803253359 (1 + 0.001931852652 sin^2 L) / sqrt(1 - e^2 sin^2 L)

    and the free-air height correction of Groves eq. 2.139::

        g(L, h) = g_0(L) [1 - (2 h / a)(1 + f + m - 2 f sin^2 L) + 3 h^2 / a^2]

    The correction is about -3.086e-6 m/s^2 per meter of height. Gives
    9.7803 m/s^2 on the equator and 9.8322 m/s^2 at the poles (h = 0). Works on
    scalars and arrays.
    """
    s2 = np.sin(latitude_rad) ** 2
    g_0 = GRAVITY_EQUATOR * (1.0 + SOMIGLIANA_K * s2) / np.sqrt(1.0 - WGS84_E2 * s2)
    h = np.asarray(height_m, dtype=np.float64)
    correction = 1.0 - (2.0 * h / WGS84_A) * (1.0 + WGS84_F + GRAVITY_M - 2.0 * WGS84_F * s2) + 3.0 * h**2 / WGS84_A**2
    result = g_0 * correction
    return float(result) if np.ndim(result) == 0 else result


def gravity_ned(latitude_rad: float, height_m: float = 0.0) -> np.ndarray:
    """Gravity vector in NED, ``[0, 0, g(L, h)]`` [m/s^2] (it points down)."""
    return np.array([0.0, 0.0, compute_gravity(latitude_rad, height_m)])


def ned_to_geodetic_delta(delta_ned_m: np.ndarray, latitude_rad: float, height_m: float) -> np.ndarray:
    """Convert a small NED offset [m] at ``(L, h)`` into ``[dL, dlambda, dh]`` [rad, rad, m]."""
    d_n, d_e, d_d = np.asarray(delta_ned_m, dtype=np.float64)
    r_m, r_n = radii_of_curvature(latitude_rad)
    return np.array(
        [
            d_n / (r_m + height_m),
            d_e / ((r_n + height_m) * np.cos(latitude_rad)),
            -d_d,
        ]
    )


def geodetic_to_ned_delta(
    latitude_rad: float,
    longitude_rad: float,
    height_m: float,
    ref_latitude_rad: float,
    ref_longitude_rad: float,
    ref_height_m: float,
) -> np.ndarray:
    """NED offset [m] of the point ``(L, lambda, h)`` from the reference point.

    First-order: ``[(R_M + h_r) dL, (R_N + h_r) cos(L_r) dlambda, -dh]`` with
    the radii at the reference latitude. Use it for residuals and errors (meters
    to a few kilometers), not to map a whole trip onto one tangent plane.
    Longitude differences are wrapped into [-pi, pi). Works on arrays.
    """
    r_m, r_n = radii_of_curvature(ref_latitude_rad)
    d_lat = np.asarray(latitude_rad, dtype=np.float64) - ref_latitude_rad
    d_lon = (np.asarray(longitude_rad, dtype=np.float64) - ref_longitude_rad + np.pi) % (2.0 * np.pi) - np.pi
    d_h = np.asarray(height_m, dtype=np.float64) - ref_height_m
    return np.stack(
        [
            (r_m + ref_height_m) * d_lat,
            (r_n + ref_height_m) * np.cos(ref_latitude_rad) * d_lon,
            -d_h,
        ],
        axis=-1,
    )
