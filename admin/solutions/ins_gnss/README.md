# Project 3 reference solutions (instructor only)

Reference implementations of every student deliverable in
`projects/ins_gnss/`, plus scripts that check them against the same tests and
driver the students use. Nothing here ships with the starter package. The
model is the shared one of `ins-gnss.qmd` and the handout's System Model
section (Groves Section 5.4 mechanization, 15-state short-horizon error model,
lever-arm position update, Joseph form, navigation-frame reset).

| File | Replaces (in `ins_gnss`) |
|---|---|
| `strapdown_solution.py` | Part A: `mechanize` (exact rotation-vector quaternions, `q(-omega_in tau) (x) q (x) q(omega_ib tau)`, averaged `C_b^n` for the specific force, Groves eq. 5.54 velocity, trapezoidal position) and `coarse_alignment`. |
| `eskf_solution.py` | Part B: `error_state_jacobian`, `discrete_process_noise` (`G Q_c G^T dt`), `gnss_position_residual` (lever arm), `joseph_update`, `inject_error_state`, `reset_covariance`, and the extension's `gnss_velocity_residual`. |
| `solution_patch.py` | `install()` swaps the solutions into the imported package (no edits to the starter). |
| `check_solutions.py` | Runs the student test suite against the solutions. |
| `run_reference.py` | Runs `ins_gnss.run_all` with the solutions and `REFERENCE_TUNING`. |

## How to check them

From the repository root, with the data downloaded
(`uv run python scripts/fetch_data.py mems-nav-dataset`):

```bash
# 1. The starter suite must stay green (student stubs skip, support code passes).
uv run pytest projects/ins_gnss

# 2. The same tests against the reference solutions: nothing may skip except
#    for missing data, and everything must pass (52 tests at the time of writing).
uv run python admin/solutions/ins_gnss/check_solutions.py

# 3. End-to-end through the student driver, all five recommended trials (~25 s).
uv run python admin/solutions/ins_gnss/run_reference.py --out /tmp/p3-ref
```

Step 2 includes the numerical sign checks: `F` against a finite-difference
Jacobian of `mechanize` (`exp(F dt)` vs the numerical transition matrix at
`dt = 0.01 s`, atol 2e-5), `H` with a 1.3 m lever arm against finite
differences of the residual, and the reset Jacobian against the exact
`Exp(dtheta) Exp(-dtheta_hat)` reset. A mutation run (scratch, not shipped)
confirmed that the suite fails for each of: `+[Cf]_x` in `F`, `+C` on the gyro
bias, no `omega_in` in the attitude update, `dq (x) q` with a body-rate
increment on the left, Coriolis without transport rate, the old `sin^4`
gravity, a rad/m mix plus body-frame `dq` in the injection, a residual that
ignores the lever arm, the body-frame (Sola) reset sign, and a flipped roll
sign in the alignment.

Step 3 writes `summary.md`/`.csv`, `outages.md`/`.csv`,
`<trial>_ins_gnss.png` (track, error vs reference, sigmas, biases, NIS) and
`<trial>_outages.png` (error vs withheld fixes with 3 sigma, constant
velocity) for every trial.

## Expected results (reference tuning, verified 2026-10-03)

Reference tuning (`run_reference.REFERENCE_TUNING`): `N_a` = 0.05 m/s/sqrt(s),
`N_g` = 0.01 deg/sqrt(s) (0.6 deg/sqrt(h)), accelerometer bias random walk
3e-4 m/s^2/sqrt(s), gyro bias random walk 1e-3 deg/s/sqrt(s); everything else
as in the starter `TUNING` (R from the receiver accuracies with scale 1 and a
1 m floor, no gate, zero lever arm). Runtimes on the development workstation
(Python 3.14, one core): 0.4-2.2 s per trial for the full-GNSS run.

**Full-GNSS run, horizontal error against the valid reference**
(`reference_valid_mask`: 97-100% of baseline rows):

| Trial | GNSS only RMSE / median | INS/GNSS RMSE / median | mean NIS (dof 3) | NIS inside 95% | yaw - course MAD |
|---|---|---|---|---|---|
| 2023-08-06_144805 (2.5 h) | 1.15 / 0.46 m | 1.51 / 0.66 m | 0.82 | 62% | 1.3 deg |
| 2023-08-09_163741 (43 min) | 1.62 / 0.59 m | 1.93 / 0.92 m | 0.80 | 67% | 1.5 deg |
| 2025-03-01_164639 (1.5 h) | 1.50 / 0.74 m | 2.76 / 1.78 m | 2.56 | 85% | 2.9 deg |
| 2025-07-31_23-36-03 (3.7 h) | 1.00 / 0.39 m | 1.16 / 0.53 m | 0.35 | 51% | 0.8 deg |
| 2025-08-03_18-15-59 (4.1 h) | 1.04 / 0.37 m | 1.25 / 0.56 m | 0.67 | 64% | 1.1 deg |

GNSS only wins against this reference on every trial, by 0.2-1.3 m RMSE. That
is expected and is a point the report should make: the baseline is itself a
GNSS-aided solution that hugs the fixes (median 0.4-0.8 m from them), so the
comparison measures closeness to GNSS. Vertical errors against the baseline
(3-34 m RMSE for both methods) say nothing about either method: the baseline
height sits 2-14 m (median, per trial) away from the GNSS height.

**Scheduled outages, pooled over the five trials** (107 outages; error
against the withheld fix at the end of each outage; CV = constant-velocity
extrapolation from the last fix):

| Outage | n | INS median | INS p90 | INS max | CV median | CV p90 | INS beats CV |
|---|---|---|---|---|---|---|---|
| 10 s | 29 | 4 m | 10 m | 21 m | 11 m | 49 m | 79% |
| 30 s | 27 | 26 m | 57 m | 107 m | 103 m | 252 m | 81% |
| 60 s | 26 | 106 m | 222 m | 440 m | 248 m | 509 m | 73% |
| 120 s | 25 | 462 m | 1.1 km | 3.9 km | 684 m | 2.1 km | 64% |

Per-trial medians at the end of the outage (`outages.md`): 3-11 m after
10 s, 15-68 m after 30 s, 46-209 m after 60 s, 226-878 m after 120 s. Over all
withheld fixes, 64% of the INS errors are inside `sqrt(P_NN + P_EE)` (63% is
the Gaussian value for a circular 2-D error) and 95% inside three times it:
the reference tuning is close to consistent in outages, with heavier tails
than Gaussian. Mean NIS is 0.35-2.6 (below 3): the receiver's accuracy
estimates are conservative fix-to-fix, while its errors are correlated in
time, so a low NIS does not mean the absolute R is too large.

With the *starter* `TUNING` (0.1, 0.1 deg/sqrt(s), 1e-3, 1e-2 deg/s/sqrt(s)),
which students start from: 95% of outage errors fall inside one sigma (badly
underconfident), and the 120 s errors get worse (median 920 m, max 13.7 km;
INS beats CV in only 36% of 120 s outages). A report that finds this from the
sigma check and tunes toward the numbers above deserves credit.

**Part A checks** (printed by `run_all`): free inertial for 60 s after
departure reaches 0.3-2.2 km (alignment at departure is the weak point:
two trials level at a pitch of -7 deg, presumably a sloped driveway, and the
car accelerates hard while the initial velocity comes from one fix). The
parked check drifts 14-118 m in 60 s on the three trials that have a quiet
parked stretch, i.e. an effective acceleration error of 0.01-0.07 m/s^2.

**Extension, horizontal velocity aiding** (`use_velocity_aiding = True`,
`R` from speed and course accuracies): outage medians at the end improve to
3 / 21 / 102 / 359 m (10 / 30 / 60 / 120 s); RMSE against the reference
changes by -0.04 to +0.17 m and stays worse than GNSS only.

## Verified dataset facts (and how)

* **Rates.** All 17 trials are 1 Hz in every column, IMU included
  (median and maximum row spacing 1.00 s); timestamps are whole UTC seconds.
  `2025-06-11_20-34-24` has IMU values in only 443 of 1648 rows.
* **Time skew.** Each baseline file starts 1 s after its input file. With
  one epoch, the baseline row stamped `t` matches the fix stamped `t + 1`:
  median horizontal disagreement 0.37-0.81 m on 13 trials (1.1-1.6 m on
  `2025-07-08` and `2025-06-27`) at a +1 s shift, versus a median over these
  15 trials of 24 m at 0 s and 47 m at -1 s (the old two-epoch loader was
  effectively -1 s). The exceptions are the excluded `2025-06-11` (no offset fits: the baseline is
  0.5-3 km off) and `2025-06-14` (best at 0 s, 10 m). Reported as GitHub issue
  #5; confirm at the source.
* **IMU/GNSS alignment in time.** Gyro rate about the vehicle down axis
  against the GNSS course rate correlates at r = 0.87-0.99 with the best lag
  0 s on all eight trials checked; forward specific force against d(speed)/dt
  also peaks at lag 0 (r = 0.61-0.86).
* **Axes and mount.** Mean specific force points along the phone's +y in 13
  trials (upright mount; `2025-06-11` and `2025-06-14` tilted back about
  27 deg), along +z in four 2025 trials (flat or nearly so), and changes
  mid-trial in `2023-08-04` (phone taken out), `2023-08-09_124742`,
  `2025-07-08`. `estimate_mount_rotation` gives vehicle forward = phone
  (+0.03...+0.30, -0.05...+0.08, -0.95...-1.00) on the recommended trials
  (yaw offsets 2-17 deg); for the flat `2025-06-18_16-52-32` forward is the
  phone's -x.
  The vehicle yaw estimated by the reference filter tracks the GNSS course
  with a median absolute deviation of 0.8-2.9 deg.
* **Height datum.** Ellipsoidal (Android `Location.getAltitude()`): the
  lowest fixes, at places a few meters above sea level, read -33 m near JFK
  (40.67 N, 73.83 W) and -41 to -49 m near the Jersey Shore (39.2-39.3 N),
  where the geoid undulation is about -33 to -34 m. Orthometric (MSL) heights
  would read near 0 m there. The `grav_*` sign convention (+9.8 on the up
  axis) is Android's too.
* **Baseline quality.** Start-up transient with |v_D| above 50 m/s in the
  first ~20 s of eight trials (up to 244 m/s); position divergences inside
  GNSS gaps: 6.1 km (`2023-08-04`, gaps of 23-51 s), 3.9 km
  (`2023-08-09_124742`, 21 s gap), 1.9 km (`2025-06-18_15-09-25`, 92 s gap),
  6.5 km (`2025-07-04`, 43 s gap), 0.7 km (`2025-07-18`, 37 s gap); and a
  145 km jump in `2024-06-20` near 2790 s *with* GNSS present. Baseline height
  minus GNSS height: median +14 m and +6 m on the two 2023 recommended trials,
  -2 to +2 m on the 2025 ones.
* **Baseline attitude.** Not usable. Read as Z-Y-X angles of `C_b^n` with
  `b` = phone axes, it puts gravity upside down (178 deg off); none of eight
  conventions tried (Z-Y-X, X-Y-Z, Z-X-Y, transposed, ENU) agrees with the
  reference filter's attitude to better than ~47 deg median. The phone's own
  orientation quaternion (`qw..qz`) is device-to-ENU (gravity within
  0.1-0.25 deg) but its heading is 70-110 deg (median) off the filter's;
  probably a magnetometer-free game rotation vector. Possibly worth a second
  GitHub issue.
* **Stale fixes.** Identical consecutive lat/lon occur in most trials, almost
  all at zero reported speed (chipset static hold). Repeats while the
  receiver reports >= 0.5 m/s: 50 in `2025-06-27`, none in the recommended
  trials. (The audit's 293 counts repeats while the *baseline* speed exceeds
  0.5 m/s, which is inflated by the baseline's own errors when parked.)
* **Gyro and accelerometer biases.** The gyro output looks bias-compensated
  by the phone: parked, it averages 1-64 deg/h per axis with 0.01-0.09 deg/s
  sample noise at 1 Hz. With the reference tuning the filter's gyro-bias
  estimates wander within about +/-250 deg/h (5-95% per axis, after the first
  10 minutes) and its accelerometer-bias estimates within -0.04...+0.16 m/s^2;
  the wander absorbs 1 Hz integration, road-grade, and mount errors as much
  as any physical bias. A student who reads the bias plots as a calibration
  of the phone deserves a gentle correction.

## Grading notes

* The book (`ins-gnss.qmd`) contains a complete mechanization, error model,
  and loose filter on a simulated drive. Students may adapt it with
  attribution; grade correctness and evidence, and put the weight on `Q_d`
  justification, consistency evidence, and the outage study.
* An INS/GNSS RMSE against the reference well *below* GNSS only (say under
  0.8 m on the 2025 trials) is suspicious: check that the baseline is not
  leaking into the filter.
* Mean NIS in the 0.3-3 range with the receiver's accuracies is normal here.
  What matters is that the student checks outage errors against the
  predicted sigma, not only the NIS.
* Common errors the tests catch are listed above; errors they do not catch:
  using the raw baseline time (2 s skew, if the student wrote their own
  loader), evaluating by sample index instead of interpolating, filtering on
  phone axes with FRD leveling formulas (pitch near -90 deg), forgetting to
  skip velocity aiding below a few m/s.
