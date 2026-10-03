# Victoria Park FastSLAM

Starter package for Project 2: FastSLAM 1.0 on the Victoria Park dataset. The
handout is `../../book/assignments/m2-victoria-park.qmd`; the background is in
the mapping and SLAM chapters (`book/chapters/04-slam/`).

The package provides the dataset loader, the Victoria Park geometry (laser
decoding, scan frames, encoder correction, lever arm), time synchronization,
evaluation against GPS, plotting, and the FastSLAM bookkeeping. The tree
extractor and the FastSLAM algorithm are yours.

## Layout

```text
src/victoria_park_slam/
  datasets.py     load_victoria_park_data, decode_laser_ranges (provided)
  types.py        VehicleGeometry, MotionModelConfig, MeasurementModelConfig,
                  TreeExtractorConfig, TreeObservation, LandmarkState, Association (provided)
  utils.py        scan_angles / scan_to_xy (vehicle or scanner frame),
                  encoder_to_centerline_speed, laser_pose_from_rear_axle,
                  predict_laser_pose, range-bearing model and Jacobians,
                  numerical_jacobian, Gaussian log-likelihood, log-sum-exp,
                  effective sample size, systematic resampling (provided)
  timing.py       scan_steps: event-driven odometry/laser merge (provided)
  evaluation.py   valid_gps_mask, align_se2, evaluate_against_gps,
                  dead_reckon, dead_reckoning_comparison (provided)
  landmarks.py    extract_tree_landmarks (yours)
  fastslam.py     LandmarkEKF and FastSLAM: bookkeeping provided,
                  initialize / innovation / updated / predict / associate /
                  update / resample yours
  run_all.py      runs every implemented stage on a scan segment, writes tables and plots
  plotting.py, cli.py
tests/
  test_victoria_park_slam.py   support-code tests + tests of your deliverables
  test_handout_api.py          the handout and the code agree
```

## Conventions (all verified on the data)

* Laser words: `raw & 0x1FFF` is the range in cm; values >= 8000 cm mean
  "no return" and decode to `NaN` (about 83% of beams).
* Scanner angle `A = 0..180 deg` in 0.5 deg steps with `A = 0` to the
  vehicle's **right**; vehicle-frame bearing `beta = A - pi/2` (positive left).
  `scan_angles()` and `scan_to_xy()` default to the vehicle frame.
* Vehicle (`VehicleGeometry`): wheelbase 2.83 m; speed encoder on the
  rear-left wheel, 0.76 m left of the centerline; laser (and GPS antenna)
  3.78 m ahead of the rear axle and 0.50 m to the left.
* Filter state: the laser pose `[x_L, y_L, theta]`, propagated with
  `predict_laser_pose` (exact for constant controls) from the centerline speed
  `encoder_to_centerline_speed(v_e, delta)`.
* Odometry 40 Hz, laser every ~214 ms; `scan_steps` splits the last odometry
  interval at each laser timestamp. The first scan precedes the first
  odometry sample and is skipped.
* GPS: reference only, compared after one SE(2) alignment over valid fixes.

## Commands

From the repository root:

```bash
uv sync
uv run python scripts/fetch_data.py victoria-park
uv run pytest projects/victoria_park_slam                 # support code passes, your stubs skip
uv run victoria-park-slam summary
uv run victoria-park-slam plot-scan --index 760 --output /tmp/scan760.png
uv run python -m victoria_park_slam.run_all               # first 2000 scans -> results/project2/
uv run python -m victoria_park_slam.run_all --sweep particles gate q_scale
uv run python -m victoria_park_slam.run_all --all         # whole log
```

The tests of your deliverables skip with "not implemented yet" until you
write them. Passing them is necessary, not sufficient: they check interfaces
and small synthetic cases, not whether your Victoria Park map is good.

## Suggested order

1. Plot scans in the vehicle frame and run `run_all` to see the dead-reckoning
   comparison.
2. Write `extract_tree_landmarks` and check it on many scans.
3. Write `LandmarkEKF.initialize`, `innovation`, `updated`.
4. Write `FastSLAM.predict`, `associate`, `update`, `resample`; pass the
   synthetic-loop test.
5. Run the first 2000 scans, read `diagnostics.png`, tune, run the sweeps.

## Data

`uv run python scripts/fetch_data.py victoria-park` installs the original
files in `data/victoria-park/` (`aa3_lsr2.mat`, `aa3_dr.mat`, `aa3_gpsx.mat`,
plus `info.txt`, `car.bmp`, `detectTreesI16.m`, `viewLsr.m`). Set
`ADVNAV_DATA_DIR` to the folder containing `victoria-park/` if you keep the
data elsewhere. Do not use GPS as a filter measurement.
