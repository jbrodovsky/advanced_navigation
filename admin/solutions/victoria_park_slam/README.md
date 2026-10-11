# Project 2 reference solutions (instructor only)

Reference implementations of every student deliverable in
`projects/victoria_park_slam/`, plus scripts that check them against the same
tests and driver the students use. Nothing here ships with the starter package.

| File | Replaces (in `victoria_park_slam`) |
|---|---|
| `landmarks_solution.py` | `extract_tree_landmarks`: a bit-for-bit port of `data/victoria-park/detectTreesI16.m` (ACFR, 1999), returning vehicle-frame bearings. |
| `fastslam_solution.py` | `LandmarkEKF` (`initialize`, `innovation`, `updated`) and `FastSLAM` (`predict`, `associate`, `update`, `resample`) as subclasses of the starter classes. |
| `solution_patch.py` | `install()` swaps the solutions into the imported package (no edits to the starter). |
| `check_solutions.py` | Runs the student test suite against the solutions. |
| `run_reference.py` | Runs `victoria_park_slam.run_all` with the solutions and the reference tuning. |

## How to check them

From the repository root, with the data downloaded
(`uv run python scripts/fetch_data.py victoria-park`):

```bash
# 1. The starter suite must stay green (student stubs skip, support code passes).
uv run pytest projects/victoria_park_slam

# 2. The same tests against the reference solutions: nothing may skip except
#    for missing data, and everything must pass (54 tests at the time of writing).
uv run python admin/solutions/victoria_park_slam/check_solutions.py

# 3. End-to-end through the student driver.
uv run python admin/solutions/victoria_park_slam/run_reference.py --out /tmp/p2-ref               # first 2000 scans, ~25 s
uv run python admin/solutions/victoria_park_slam/run_reference.py --sweep particles gate q_scale  # + required sweeps, ~4 min
uv run python admin/solutions/victoria_park_slam/run_reference.py --all --out /tmp/p2-ref-all     # whole log, ~80 s
```

Step 3 writes `summary.md`/`.csv`, `sweep_<name>.md`/`.csv`, `features.png`,
`trajectory_map.png`, and `diagnostics.png` (error vs time, N_eff, landmark
count, runtime per scan).

## Expected results (reference tuning, verified 2026-10-03)

Reference tuning (`run_reference.REFERENCE_TUNING`): 50 particles,
$\mathbf{Q}_u$ one-sigma (1.0 m/s, 6 deg) per 25 ms prediction step,
$\mathbf{R}$ = diag(0.5 m, 2 deg)^2, gates 5.99 / 13.82,
$\log p_0 = -\log(75\pi) = -5.46$, resample at $N_\text{eff} < N/2$,
confirm after 3 associations, tentative timeout 2 s. Runtimes on the
development workstation (Python 3.14, one core).

**First 2000 scans (21.8-448.9 s, 1.1 km, 1128 valid GPS fixes):**

| Method | SE(2)-aligned RMSE vs GPS | max error |
|---|---|---|
| DR, encoder correction + lever arm | 32.4 m | 88 m |
| DR, encoder correction only | 32.2 m | 84 m |
| DR, lever arm only | 62.4 m | 142 m |
| DR, no corrections | 61.5 m | 137 m |
| FastSLAM (weighted mean), seed 0 | 1.34 m | 3.5 m |

FastSLAM: 182 confirmed landmarks at the end, about 10 ms per scan (max
~35 ms), median N_eff 28 of 50, resampling on 41% of scans, about 8.2 matched,
0.25 new, and 0.1 discarded observations per scan per particle. Seeds 0-4 give
1.29-1.38 m.

Required sweeps (seed 0):

| particles | 10 | 25 | 50 | 100 |
|---|---|---|---|---|
| RMSE (m) | 1.24 | 1.33 | 1.34 | 1.27 |
| ms per scan | 2.5 | 5.7 | 10.3 | 20.9 |

| gate | 4.61 | 5.99 | 9.21 |
|---|---|---|---|
| RMSE (m) | 1.31 | 1.34 | 1.29 |

| q_scale (relative to the reference Q_u) | 0.25 | 1.0 | 4.0 |
|---|---|---|---|
| RMSE (m) | 1.25 | 1.34 | 1.30 |

The particle count and the gate barely matter on this segment (the vehicle
sees 5-10 trees in every scan, so the map pins the pose down locally); the
process noise is the parameter that matters. Relative to the *starter*
`TUNING` (0.5 m/s, 3 deg), which is what students sweep:

| Q_u one-sigma per step | (0.25 m/s, 1.5 deg) | (0.5, 3) | (1.0, 6) |
|---|---|---|---|
| RMSE, seed 0 | 8.3 m (318 confirmed: duplicates) | 1.25 m | 1.30 m |
| seeds 0-4 | | 1.23-1.26 m, except seed 1: 3.2 m (276 confirmed) | 1.29-1.38 m |

With `q_scale = 0.01` the filter degenerates to dead reckoning (28.6 m, 500
confirmed landmarks). So students running the required `q_scale` sweep on the
starter tuning should see the overconfidence failure at 0.25 and, if they run
several seeds, the occasional failure at 1.0. A report that spots this and
explains it with the duplicate-landmark count and the 40 Hz noise argument
deserves credit.

**Full log (7249 scans, 1548 s, 4.0 km, 4443 valid fixes):** dead reckoning
93.1 m RMSE (no corrections 128.8 m); FastSLAM 3.76 m (max 11.5 m),
420 confirmed landmarks, 11 ms per scan, 80 s total. With the starter Q_u
(0.5 m/s, 3 deg) the same seed gives 2.57 m (414 confirmed). The residual
error is concentrated in the north-west branch (around east -150 m, north
50-180 m), where the estimate runs a few meters west of GPS. Treat a single
full run as one sample: the two settings above differ by more than their
2000-scan results do.

## Verified dataset facts (and how)

* **Laser decoding.** Low 13 bits = cm (`viewLsr.m` uses the mask);
  83.0% of beams are >= 8000 cm ("no return"); 5 raw words have flag bits set,
  and their masked values (14.6-16.9 m) are ordinary returns.
* **Bearing convention, beta = A - pi/2.** Tree detections projected along
  dead reckoning cluster 3-4x more tightly in turns than with the mirrored
  convention (`mapping.qmd`, convention-check cell: 0.40 m vs 1.49 m median
  nearest-re-observation distance over the 12 sharpest turns; the same
  holds in every 150-scan window of the log).
* **Steering sign.** Positive = left (CCW): the mirrored dead-reckoning track
  fits GPS with 24 m instead of 12.6 m RMSE over the first 200 s.
* **Encoder on the rear-left wheel.** The correction with `1 - (H/L) tan(delta)`
  halves the error (28.1 -> 13.8 m, first 200 s); with the sign for a right
  wheel it gets worse (41 m).
* **Laser lever arm (3.78, +0.50).** GPS RMSE cannot resolve the lateral
  offset (12.6 vs 12.5 m for b = +/-0.5), but map sharpness in turns can.
  The chapter's convention-check cell (12 sharpest turns, chapter extractor):
  0.40 m for (3.78, +0.50), 0.64 m for b = -0.50, 1.26 m for a = 0.95 (the
  drawing's label), 1.38 m with no lever arm, 1.49 m with a mirrored scan.
  A finer scan with the MATLAB detector over the 25 sharpest 60-scan windows:
  0.41 m (3.78, +0.50), 0.45 (b = +1.0), 0.49 (b = 0), 0.59 (b = -0.50),
  0.45 (a = 4.5), 0.49 (a = 3.0), 0.84 (a = 0.95), 0.94 (no lever arm). This
  matches `car.bmp` read as a top view with the front on the left (encoder
  and laser both on the vehicle's left side).
* **GPS.** 4466 fixes, gaps up to 58 s, 76 inter-fix steps > 5 m (most across
  gaps); `valid_gps_mask` keeps 4444. One fix at 1244 s is 141 m off.
* **Timing.** First scan 21.819 s, first odometry 21.940 s; laser every
  213-215 ms, odometry every 25 ms. Map sharpness is best with the laser
  timestamps shifted by about -50 ms (0.380 vs 0.400 m; see `mapping.md`
  exercise 6): the laser stamps appear late by roughly 50 +/- 25 ms relative
  to odometry. Applying -50 ms improves the reference RMSE on 2000 scans by
  only 0.03-0.06 m (three seeds). Not in the handout; worth credit if a
  student finds it.
* **Tree detector.** 52974 detections in the log (7.3 per scan), 17117 in the
  first 2000 scans; median diameter 0.41 m (5-95%: 0.14-0.85 m); median range
  23 m. The MATLAB's 5-degree split and 1-degree occlusion thresholds are exact
  multiples of the beam spacing, so gaps of exactly 10 (or 2) beams are
  decided by floating-point rounding (`landmarks_solution.py` reproduces it).

## Grading notes

* The FastSLAM back end can be adapted from `slam.qmd` and the extractor
  logic from `mapping.qmd`/`detectTreesI16.m`, with attribution. Grade those
  on correctness and evidence; put the weight on the dataset-specific parts
  (rubric rows 1-3) and on the evaluation.
* A FastSLAM RMSE well below 1 m on the 2000-scan segment is suspicious
  (GPS noise alone is around a meter): check that GPS is not leaking into the
  filter and that the evaluation follows the protocol (one SE(2) alignment
  for the whole segment, no per-window re-alignment).
* Landmark counts far above ~190 confirmed on the first 2000 scans mean
  duplicates (usually Q_u too small, a missing p0, or no mutual exclusion).
  Counts far below ~170 usually mean an over-aggressive extractor.
* Watch for in-place landmark updates combined with shallow particle copies:
  the provided test `test_better_aligned_particle_gains_weight` catches it
  (observation counts of 3 instead of 2).
* Nearest-sample time synchronization or one prediction per scan costs
  little RMSE on the straight stretches and a lot in turns; the rubric's first
  row asks for evidence either way.
