"""Run the reference solution through the student ``run_all`` driver.

From the repository root::

    uv run python admin/solutions/ins_gnss/run_reference.py                          # all recommended trials
    uv run python admin/solutions/ins_gnss/run_reference.py --trial 2023-08-09_163741

Outputs go to ``results/project3-reference`` unless ``--out`` is given.
Expected numbers are in this folder's README.
"""

from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from solution_patch import install  # noqa: E402

# Reference tuning (see README): the starter's IMU noise is a factor of 2-10
# too pessimistic, which makes the outage 3-sigma envelope useless.
REFERENCE_TUNING = {
    "accel_vrw_mps_rt_s": 0.05,
    "gyro_arw_deg_rt_s": 0.01,
    "accel_bias_rw_mps2_rt_s": 3.0e-4,
    "gyro_bias_rw_deg_s_rt_s": 1.0e-3,
}


def main(argv: list[str]) -> int:
    install()
    from ins_gnss import run_all

    run_all.TUNING.update(REFERENCE_TUNING)
    if "--out" not in argv:
        argv = [*argv, "--out", "results/project3-reference"]
    return run_all.main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
