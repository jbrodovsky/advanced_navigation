# INS/GNSS Integration

Starter package for Project 3: a curved-Earth strapdown INS aided by loosely
coupled GNSS through a 15-state error-state Kalman filter, on smartphone MEMS
data. The handout is `../../book/assignments/m3-ins-gnss.qmd`; the background
is in `book/chapters/05-global/` (`ins-gnss.qmd`, `ins-errors.qmd`).

The package provides the data loaders and data-quality helpers, the WGS-84
Earth model, rotation helpers, the filter bookkeeping, the evaluation
protocol, and a driver. The mechanization, the alignment, and the ESKF model
and update are yours.

## Layout

```text
src/ins_gnss/
  datasets.py     load_mems_nav_trial (one epoch, baseline offset), recommended_trials,
                  TRIAL_NOTES, estimate_mount_rotation, rotate_imu_samples,
                  repeated_fix_mask, gnss_acceleration_frd, array helpers (provided)
  types.py        IMUSample, GNSSSample (speed, course, accuracies), NavState,
                  IMUNoise, MEMSNavTrial, MEMSReferenceState (provided)
  earth.py        radii_of_curvature, earth_rate_ned, transport_rate_ned,
                  compute_gravity (Somigliana + free air), gravity_ned,
                  ned_to_geodetic_delta, geodetic_to_ned_delta (provided)
  rotations.py    quat_multiply, quat_normalize, quat_to_dcm, dcm_to_quat,
                  rotvec_to_quat, quat_to_rotvec, skew, euler_to_dcm,
                  attitude_error (provided)
  strapdown.py    mechanize, coarse_alignment (yours); free_inertial (provided)
  eskf.py         error_state_jacobian, discrete_process_noise,
                  gnss_position_residual, joseph_update, inject_error_state,
                  reset_covariance (yours); gnss_velocity_residual (extension);
                  INSGNSSFilter, gnss_position_covariance (provided)
  evaluation.py   position_error_ned (time-aligned), error_summary,
                  reference_valid_mask, gnss_only_track, schedule_outages,
                  outage_errors, constant_velocity_outage_errors,
                  heading_error_vs_course, chi2_interval, nees (provided)
  run_all.py      every stage on the recommended trials -> tables and plots
  cli.py, plotting.py, data_paths.py
tests/
  test_ins_gnss.py      support-code tests + tests of your deliverables
  test_handout_api.py   the handout and the code agree
```

## Conventions (verified on the data)

* NED navigation frame; latitude, longitude in radians; height above the
  WGS-84 ellipsoid (the fixes read -30 to -49 m at the shore, where the geoid
  is about 33 m below the ellipsoid).
* Hamilton scalar-first `q_b^n`, `C(q_b^n) = C_b^n`; errors are true minus
  estimate; attitude error in the navigation frame,
  `q = dq(dtheta) (x) q_hat`. Error state `[dp (NED m), dv, dtheta, db_a, db_g]`.
* Every trial is 1 Hz, IMU included. IMU axes are the phone's (Android: x
  right, y top of screen, z out of screen). `estimate_mount_rotation` finds the
  phone-to-vehicle rotation (down from the mean specific force, forward from
  GNSS acceleration); in the recommended trials vehicle forward is about the
  phone's -z.
* One time epoch per trial (the first input row). The baseline is labelled
  1 s early relative to the GNSS fixes; `load_mems_nav_trial` adds
  `BASELINE_TIME_OFFSET_S = 1.0` (GitHub issue #5 on the dataset).
* The baseline is a GNSS-aided reference, not truth: compare horizontal
  position only, where `reference_valid_mask` holds. Its attitude columns
  are not in the book's convention. Never use it as a measurement.

## Commands

From the repository root:

```bash
uv sync
uv run python scripts/fetch_data.py mems-nav-dataset
uv run pytest projects/ins_gnss                       # support code passes, your stubs skip
uv run ins-gnss                                       # trials and their status
uv run ins-gnss --trial 2023-08-09_163741             # one trial: rates, gaps, mount
uv run python -m ins_gnss.run_all                     # all recommended trials -> results/project3/
uv run python -m ins_gnss.run_all --trial 2023-08-09_163741
```

The tests of your deliverables skip with "not implemented yet" until you
write them. They check signs and Jacobians numerically (a stationary perfect
IMU, a perfect IMU driving east, `F` against your own mechanization, `H` with
a lever arm, the reset against the exact attitude reset, and a consistency
check on a synthetic drive). Passing them is necessary, not sufficient.

## Suggested order

1. Look at the data: `uv run ins-gnss --trial NAME`, plot GNSS, read the handout's data section.
2. `mechanize`, `coarse_alignment`; run the Part A checks in `run_all`.
3. `error_state_jacobian`, `discrete_process_noise`.
4. `gnss_position_residual`, `joseph_update`, `inject_error_state`, `reset_covariance`.
5. `run_all` on one short trial; tune `TUNING`; then all five recommended trials and the outage study.

## Data

`uv run python scripts/fetch_data.py mems-nav-dataset` writes
`data/mems-nav-dataset/{input,baseline}/` and `manifest.json` (about 150 MB,
17 trials). Set `ADVNAV_DATA_DIR` to the folder containing
`mems-nav-dataset/` if you keep the data elsewhere. Use `input/` as estimator
input and `baseline/` only for comparison.
