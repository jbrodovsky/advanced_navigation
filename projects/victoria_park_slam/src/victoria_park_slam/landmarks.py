"""Starter front-end interfaces for Victoria Park tree extraction."""

from __future__ import annotations


def extract_tree_landmarks(scan, pose_guess):
    """Return range-bearing tree observations from one laser scan.

    Students are expected to implement this front end. A complete solution
    should reject invalid ranges, segment compact scan clusters, estimate tree
    diameters, and return observations in scanner/vehicle coordinates. The
    `pose_guess` is useful for diagnostics and plotting, but feature extraction
    should not depend on GPS truth.
    """
    raise NotImplementedError("students should implement tree extraction from laser scans")
