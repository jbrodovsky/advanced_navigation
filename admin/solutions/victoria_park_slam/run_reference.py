"""Run the reference solution through the student ``run_all`` driver.

From the repository root::

    uv run python admin/solutions/victoria_park_slam/run_reference.py                       # first 2000 scans
    uv run python admin/solutions/victoria_park_slam/run_reference.py --sweep particles gate q_scale
    uv run python admin/solutions/victoria_park_slam/run_reference.py --all                 # whole run, ~80 s

Outputs go to ``results/project2-reference`` unless ``--out`` is given.
Expected numbers are in this folder's README.
"""

from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from solution_patch import install  # noqa: E402

# Reference tuning: the starter defaults except for Q_u, which is 4x larger in
# covariance (2x in sigma). With the starter's (0.5 m/s, 3 deg) per 25 ms step,
# one seed in five duplicates trees on the first 2000 scans; with
# (1.0 m/s, 6 deg) all five seeds land within 1.3-1.4 m RMSE.
REFERENCE_TUNING = {
    "num_particles": 50,
    "control_std": [1.0, 6.0],
    "range_std_m": 0.5,
    "bearing_std_deg": 2.0,
    "gate_threshold": 5.99,
    "birth_threshold": 13.82,
    "new_landmark_log_likelihood": None,
    "resample_threshold": 0.5,
}


def main(argv: list[str]) -> int:
    install()
    from victoria_park_slam import run_all

    run_all.TUNING.update(REFERENCE_TUNING)
    if "--out" not in argv:
        argv = [*argv, "--out", "results/project2-reference"]
    return run_all.main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
