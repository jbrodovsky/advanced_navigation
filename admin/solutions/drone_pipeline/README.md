# Project 1 reference solutions (instructor only)

Reference implementations of every student deliverable in
`projects/drone_pipeline/`, plus scripts that check them against the same tests
and driver the students use. Nothing here ships with the starter package.

| File | Replaces (in `drone_pipeline`) |
|---|---|
| `observation_solution.py` | Part B: `build_apriltag_corner_map`, `estimate_pose`, `estimate_measurement_covariance` (+ `estimate_measurement_bias`). The corner map and covariance code is what used to ship pre-solved in the starter. |
| `mechanization_solution.py` | Part A: `propagate_state` (forward Euler, Z-X-Y kinematics, vectorized over particles), plus `euler_rate_matrix` / `euler_rate_matrix_inverse`. |
| `filters_solution.py` | Part C: `EKFNavigator`, `UKFNavigator`, `ParticleFilterNavigator` as subclasses of the starter classes. |
| `solution_patch.py` | `install()` swaps the solutions into the imported package (no edits to the starter). |
| `check_solutions.py` | Runs the student test suite against the solutions. |
| `run_reference.py` | Runs `drone_pipeline.run_all` with the solutions and the reference tuning. |

## How to check them

From the repository root, with the drone data downloaded
(`uv run python scripts/fetch_data.py drone-data`):

```bash
# 1. The starter suite must stay green (student stubs skip, support code passes).
uv run pytest projects/drone_pipeline

# 2. The same tests against the reference solutions: nothing may skip except
#    for missing data, and everything must pass (38 tests at the time of writing).
uv run python admin/solutions/drone_pipeline/check_solutions.py

# 3. End-to-end: all stages on studentdata1-7 through the student driver
#    (about 15 s; add --pf-study for the particle-count study, about 75 s).
uv run python admin/solutions/drone_pipeline/run_reference.py --out /tmp/p1-reference
```

Step 3 writes `summary.md`, `summary.csv`, per-dataset position plots and,
with `--pf-study`, `pf_study.md`.

## Expected results (reference tuning, verified 2026-10-03)

Part B, zero distortion, all multi-tag and single-tag frames:

| | ds1-4 | ds5-7 | ds0 |
|---|---|---|---|
| Median reprojection error | 0.65-0.73 px | 0.84-0.89 px | 0.81 px |
| ... with the file's distortion coefficients | 4.3-4.5 px | 3.9-4.1 px | 4.2 px |
| PnP position RMSE vs Vicon (3-D) | 1.7-2.7 cm | 1.9-2.1 cm | ~15 cm (bad truth) |
| Position bias (mean residual) | <= 1.6 cm per axis; y ~ +1.3 cm on ds1-3 | <= 0.9 cm | (+12.6, -4.0, +6.1) cm |
| Position std | 0.5-1.3 cm | 0.6-1.7 cm | 1.6-3.2 cm |
| Yaw bias / std | +0.28 to +0.49 deg / 0.25-0.40 deg | +0.46 to +0.51 deg / ~0.1 deg | +5.3 deg |

Passing the distortion coefficients instead adds an 8-12 cm *downward* bias in
z and inflates the horizontal position std to 5-8 cm: a modelling error masquerading as
sensor bias. Single-tag frames (ds5-7 have 14-28 each) have 5-6 cm errors and
2-6 deg pitch errors (vs ~1.5 cm multi-tag); the second IPPE solution is
45-85 cm away with ~10x the reprojection error, so the flip is usually
detectable. R estimated on the development split (pooled): position std
(1.23, 0.78, 0.70) cm, attitude std (0.53, 0.60, 0.38) deg.

Part C (`run_reference.py`, R from the development split, 99.9 % NIS gate):

| | ds1-4, ds6 | ds5, ds7 |
|---|---|---|
| EKF / UKF position RMSE | 1.7-2.5 cm | ~9 cm overall, ~2.1 cm while tags are visible |
| PF (1000 particles) | 1.6-2.7 cm | 14-15 cm overall |
| Mean NIS | 0.4-3.1 | 0.4-0.5 |
| ANEES (9 states: pos, att, vel) | 29-67 | ~31 |

* ds5 and ds7 end with ~1.5 s (ds0: 2.9 s) without tags (ds5-7 start and ds5, ds7 end parked beside the mat corner at about (-0.35, -0.05, 0.95) m; ds0 starts and ends too low over the mat); the filters dead-reckon
  through it and that tail dominates their RMSE. ds5-7 and ds0 also start with
  ~1-2 s without tags. There are no mid-flight dropouts.
* EKF and UKF agree to well under a millimetre: the measurement model is linear
  and 20 ms steps keep the propagation nearly linear. A good report says so.
* Mean NIS below 6 means the truth-derived R is conservative (it contains
  Vicon noise, interpolation and between-flight bias differences). ANEES far
  above 9 means the filter is overconfident about truth: the PnP bias
  (especially the consistent +0.45 deg yaw) is common to every measurement, so
  the innovations cannot see it. Students who notice this contrast deserve
  credit.
* Dead reckoning from the Vicon state without bias compensation reaches
  6-87 m RMSE (ds2 shortest, ds5 longest). Subtracting the hover-segment bias
  estimate helps a lot; growth is still roughly quadratic.

### Pitfalls the reference run surfaced (and the handouts now warn about)

* **Process noise from hover is too small.** Gyro and accelerometer residuals
  against Vicon on the hover-like flights (ds5-7) have std ~0.06-0.10 rad/s and
  ~0.10-0.13 m/s^2, mean (bias) ~0.1 m/s^2 on x/y. On the aggressive flights
  (ds1-4) the effective accelerometer error is 0.2-0.9 m/s^2; the best-fit
  IMU-to-Vicon time lag on ds1-4 is about 40-50 ms, which may explain part of
  it (worth an investigation, not a fact to state). With Q at hover level the
  EKF locks out on ds2 (286 consecutive gate rejections) and the PF collapses
  (median N_eff = 1 of 1000 on ds3, 32 m RMSE). An effective accelerometer
  noise of ~2 m/s^2 (density 0.08) fixes both.
* **No flight starts at rest.** ds1-4 start in flight; ds5-7 have about 0.3 s
  nearly at rest. Quiet hover stretches of 1-2.3 s exist mid-flight in ds5-7
  only (around t = 7-11 s).
* **Initialization lockout.** ds5-7 start with single-tag frames while moving
  at ~0.5 m/s. A filter initialized from a biased single-tag pose with zero
  velocity rejects the first good multi-tag poses (NIS 30-90) and never
  recovers. `run_all.initial_state_from_poses` uses only poses with >= 2 tags
  and a 0.2 s line fit for velocity.
* **studentdata0.** 33 Hz, gaps up to 0.49 s, 75 packets outside the Vicon
  span, `drpy` correlates only 0.1-0.7 with any candidate rate signal, and PnP
  disagrees with Vicon by a constant (+12.6, -4.0, +6.1) cm and +5.3 deg yaw.
  It is excluded from the split pending GitHub issue #4.

## Grading notes

* Part B code is substantially given in the PnP chapter (students may adapt it
  with attribution). Grade Part B on the evidence: the distortion check, the
  bias/covariance split, single-tag handling, and the camera-body convention.
* A filter that "beats PnP" by more than a few millimetres in visual coverage
  is suspicious: check for Vicon leaking into the update path.
* Any R or Q inflation must be reported with the diagnostic that motivated it
  (N_eff history for the PF, NIS/NEES for the KFs).
