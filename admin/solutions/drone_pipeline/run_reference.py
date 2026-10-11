"""Run the reference solution through the student ``run_all`` driver.

From the repository root::

    uv run python admin/solutions/drone_pipeline/run_reference.py                 # all of ds1-7
    uv run python admin/solutions/drone_pipeline/run_reference.py --pf-study      # + particle-count study
    uv run python admin/solutions/drone_pipeline/run_reference.py --datasets 2 7  # any run_all arguments

Outputs go to ``results/reference`` unless ``--out`` is given. Expected numbers
are in this folder's README.
"""

from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from solution_patch import install  # noqa: E402

# Reference tuning. Q is a spectral density (per second); see the README for
# where each number comes from.
REFERENCE_TUNING = {
    # pos, att, vel, gyro bias, accel bias (std devs)
    "P0_std": [0.05] * 3 + [0.02] * 3 + [0.3] * 3 + [0.02] * 3 + [0.2] * 3,
    # attitude: gyro white noise ~0.1 rad/s at 50 Hz -> 0.1^2 * 0.02 = 2e-4
    # velocity: 2 m/s^2 effective accel error (hover noise is ~0.12 m/s^2; the
    #           aggressive flights need ~16x more to cover model/timing error)
    "Q_density": [1e-6] * 3 + [2e-4] * 3 + [8e-2] * 3 + [1e-7] * 3 + [1e-4] * 3,
    "R": None,  # estimated on the development split by the reference Part B
    "R_inflation": 1.0,
    "gate_probability": 0.999,
    "init_window_s": 0.2,
}


def main(argv: list[str]) -> int:
    install("ABC")
    from drone_pipeline import run_all

    run_all.TUNING.update(REFERENCE_TUNING)
    if "--out" not in argv:
        argv = [*argv, "--out", "results/reference"]
    if "--datasets" not in argv and "--split" not in argv:
        argv = [*argv, "--split", "all"]
    return run_all.main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
