# Drone Visual-Inertial Pipeline

Starter package for Project 1, the staged drone visual-inertial navigation
pipeline. It provides a dataset loader, a `parameters.txt` parser, Vicon truth
interpolation, Z-X-Y attitude and transform helpers, typed records, shared
evaluation metrics, a driver that runs every stage, and the public interfaces
for the parts you implement.

The project builds one pipeline in three checkpoints:

1. Part A: IMU characterization and inertial dead reckoning.
2. Part B: the AprilTag PnP perception model and its measurement noise.
3. Part C: EKF, UKF, and particle-filter fusion of A and B.

The handouts are in `../../book/assignments/` (`m1-drone-pipeline.qmd` and
`m1-drone-pipeline-a/b/c.qmd`). They are the specification; this README is
the map.

## Code surface

```text
src/drone_pipeline/
  datasets.py       load_drone_dataset, load_drone_parameters (parses parameters.txt),
                    interpolate_vicon_truth (NaN + valid=False outside the Vicon span)
  types.py          DronePacket, DroneDataset, TruthSample, DroneParameters, PoseObservation
  utils.py          rpy_to_rotation_matrix / rotation_matrix_to_rpy (Z-X-Y), wrap_angles,
                    make_transform / invert_transform, interpolate_rows
  config.py         documented defaults (camera, mounting, tag geometry), dataset split
  evaluation.py     truth_matrix, state_errors, rmse, nees, chi2_interval, fraction_inside
  mechanization.py  PropagationParams, propagate_state            <- Part A (yours)
  observation.py    build_apriltag_corner_map, estimate_pose,
                    estimate_measurement_covariance               <- Part B (yours)
  filters.py        EKFNavigator, UKFNavigator, ParticleFilterNavigator
                    (constructors/accessors given; predict/update yours) <- Part C
  run_all.py        runs every implemented stage on a split; writes tables and plots
  plotting.py       small plotting helpers
  cli.py            dataset inspection
tests/
  test_drone_pipeline.py   support-code tests + tests of your deliverables
  test_handout_api.py      keeps the handouts and this package in agreement
```

Keep the public signatures stable (the tests, `run_all`, and the graders call
them); add helpers and modules freely.

## Conventions in one paragraph

World frame `w` = the tag mat (`x` down the mat, `y` right, `z` up; origin at
the top-left corner of tag 0). Body frame `b` = the IMU, `z` up. Camera frame
`c` = optical (`x` right, `y` down, `z` along the optical axis, pointing at the
mat). Attitude is Z-X-Y: `C_world_body = Rz(yaw) @ Rx(roll) @ Ry(pitch)`.
`T_camera_body` maps body coordinates into camera coordinates, so
`T_world_body = inv(T_camera_world) @ T_camera_body`. The packet corner
detections are already undistorted: pass `np.zeros(5)` to `cv2.solvePnP`.
`packet.omg` is the gyro body rate (`packet.drpy` is a read-only alias kept
for old code; in `studentdata0` the slot holds that file's unexplained `drpy`
field). The onboard `packet.rpy` may be used for sanity checks, never as a
filter measurement.

## Getting started

From the repository root:

```bash
uv sync
uv run python scripts/fetch_data.py drone-data   # about 570 MB into data/drone-data/
uv run pytest projects/drone_pipeline            # support code passes; your stubs skip
uv run python -m drone_pipeline.cli 1            # inspect studentdata1
uv run python -m drone_pipeline.run_all          # run whatever you have implemented
```

```python
from drone_pipeline.datasets import interpolate_vicon_truth, load_drone_dataset, load_drone_parameters

dataset = load_drone_dataset(1)          # data/drone-data/studentdata1.mat
params = load_drone_parameters()         # data/drone-data/parameters.txt
truth = interpolate_vicon_truth(dataset)
valid = [sample for sample in truth if sample.valid]
print(len(dataset.packets), dataset.packets[0].omg, valid[0].position)
```

Start Part A in `mechanization.py`, Part B in `observation.py` (after reading
`config.py`), and Part C in `filters.py`.

## Tests

The deliverable tests skip with a "not implemented yet (starter stub)" message
until you write the corresponding function, then start checking it. Passing
them means your interfaces and simple cases are right, not that your
navigation solution is good. Tests that need the real data skip until it has
been downloaded.

## run_all

`uv run python -m drone_pipeline.run_all [--split development|validation|all]
[--datasets 1 5] [--pf-study] [--out DIR]` runs dead reckoning (Vicon-
initialized, as Part A allows), PnP, and the three filters (PnP-initialized) on
each dataset, using the Part C processing loop (`run_navigator`). It writes
`summary.csv`/`summary.md` (RMSE, NIS, NEES, runtime, gated updates), one
position plot per dataset, and `pf_study.csv` with `--pf-study`. Stages you
have not written are reported and skipped. Put your noise values in its
`TUNING` block; the defaults are deliberately untuned placeholders. Results
go to `results/<split>/` by default; keep that folder out of version control.

## Data

`uv run python scripts/fetch_data.py drone-data` installs `parameters.txt`
and `studentdata0.mat` ... `studentdata7.mat` under `data/drone-data/`. Set
`ADVNAV_DATA_DIR` to the folder containing `drone-data/` if you keep it
elsewhere. Recommended split: develop and tune on 1-4, validate on 5-7;
`studentdata0` is excluded pending GitHub issue #4 (see the handout).

Vicon truth is for evaluation and sensor characterization. It must not be used
as a recurring measurement in the navigation filter.
