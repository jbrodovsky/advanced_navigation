# Drone Visual-Inertial Pipeline

This package is the starter code surface for Project 1: the staged drone
visual-inertial navigation pipeline. It gives students a common dataset loader,
camera/tag geometry helpers, typed records, plotting utilities, and public
interfaces for the implementations they are expected to write.

The project is organized around one end-to-end navigation pipeline:

1. Part A: inertial dead reckoning from IMU packets.
2. Part B: AprilTag pose observations from camera images.
3. Part C: EKF, UKF, and particle-filter fusion of inertial propagation and
   visual pose measurements.

The corresponding assignment handouts are in `../../book/assignments/`:

- `m1-drone-pipeline.qmd`
- `m1-drone-pipeline-a.qmd`
- `m1-drone-pipeline-b.qmd`
- `m1-drone-pipeline-c.qmd`

## Current Code Surface

```text
projects/drone_pipeline/
  pyproject.toml
  src/drone_pipeline/
    cli.py
    config.py
    datasets.py
    filters.py
    mechanization.py
    observation.py
    plotting.py
    types.py
    utils.py
  tests/
    test_drone_pipeline.py
```

The implemented support code is:

- `datasets.py`: loads `studentdata*.mat`, parses `parameters.txt`, and
  interpolates Vicon truth onto packet timestamps.
- `types.py`: dataclasses for packets, datasets, Vicon truth samples, camera
  parameters, and pose observations.
- `config.py`: fixed camera intrinsics, distortion coefficients, camera/body
  offset, AprilTag size, gaps, and tag IDs.
- `observation.py`: builds the AprilTag world-corner map and estimates
  measurement covariance from pose residuals.
- `utils.py`: angle wrapping, roll-pitch-yaw rotation matrices, homogeneous
  transforms, and row-wise interpolation.
- `plotting.py`: simple position and attitude comparison plots.
- `cli.py`: a minimal dataset inspection command.
- `tests/`: regression tests for data loading, tag-map geometry, Vicon
  interpolation, and covariance estimation.

The intentionally incomplete student implementation points are:

- `mechanization.propagate_state(...)`
- `observation.estimate_pose(...)`
- `filters.EKFNavigator`
- `filters.UKFNavigator`
- `filters.ParticleFilterNavigator`

Those functions and classes define the public contracts students should build
against. Add helpers as needed, but keep these interfaces stable so notebooks,
tests, and reports can call the same code.

## Where To Start

Start with the top-level Project 1 handout, then work through the staged files
in order. In code, start by loading one dataset and checking the packet and
truth conventions:

```python
from drone_pipeline.datasets import load_drone_dataset, interpolate_vicon_truth

dataset = load_drone_dataset(1)  # data/drone-data/studentdata1.mat
truth = interpolate_vicon_truth(dataset)

print(len(dataset.packets))
print(dataset.packets[0].t)
print(truth[0].position)
```

For Part A, begin in `src/drone_pipeline/mechanization.py`. For Part B, begin
in `src/drone_pipeline/observation.py` after inspecting `config.py`. For Part C,
begin in `src/drone_pipeline/filters.py` and reuse the same state and
measurement conventions from the assignment.

## Useful Commands

From the repository root:

```bash
uv sync
uv run python scripts/fetch_data.py drone-data
uv run pytest projects/drone_pipeline/tests
uv run python -m drone_pipeline.cli 1
```

The tests check the starter surface, not a full navigation solution. Passing
them means the provided loaders and utilities still work; it does not mean the
student implementation is complete. Tests that need the real data are skipped
until it has been downloaded.

## Data

Download the course drone data (about 570 MB) with
`uv run python scripts/fetch_data.py drone-data`. It is installed under:

```text
data/drone-data/
  parameters.txt
  studentdata0.mat
  ...
  studentdata7.mat
```

Vicon truth is available for evaluation and covariance estimation. It should
not be used as a recurring measurement in the navigation filter.
