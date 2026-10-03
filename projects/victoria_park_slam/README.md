# Victoria Park FastSLAM

This package is the starter code surface for the Victoria Park SLAM project. It
centers on laser feature extraction, planar vehicle motion, per-particle
landmark EKFs, data association, particle weighting, and resampling.

The assignment handout is:

- `../../book/assignments/m2-victoria-park.qmd`

The package provides dataset loading, geometry utilities, plotting hooks, and
public contracts. It intentionally does not implement the complete FastSLAM
solution.

## Current Code Surface

```text
projects/victoria_park_slam/
  pyproject.toml
  src/victoria_park_slam/
    cli.py
    datasets.py
    fastslam.py
    landmarks.py
    plotting.py
    types.py
    utils.py
  tests/
    test_victoria_park_slam.py
```

The implemented support code is:

- `datasets.py`: resolves and loads the Victoria Park MATLAB files, converting
  laser ranges from centimeters to meters and timestamps from milliseconds to
  seconds.
- `types.py`: dataclasses for the loaded dataset, motion and measurement
  configuration, tree observations, landmark states, particle diagnostics, and
  map diagnostics.
- `utils.py`: angle wrapping, scan angle generation, polar-to-XY scan
  conversion, bicycle-model prediction, range-bearing models and Jacobians,
  Mahalanobis distance, effective sample size, and systematic resampling.
- `plotting.py`: scan, observation, trajectory, effective-sample-size, and
  landmark-count plotting helpers.
- `cli.py`: dataset summary and single-scan plotting commands.
- `tests/`: regression tests for geometry, motion propagation, Mahalanobis
  distance, resampling helpers, and dataset loading when the data is present.

The intentionally incomplete student implementation points are:

- `landmarks.extract_tree_landmarks(...)`
- `fastslam.LandmarkEKF.initialize(...)`
- `fastslam.LandmarkEKF.update(...)`
- `fastslam.FastSLAM.predict(...)`
- `fastslam.FastSLAM.update(...)`
- `fastslam.FastSLAM.resample(...)`

These define the expected algorithm contracts. Students can add internal helper
modules, but the public surface should stay stable enough for tests, notebooks,
and reports to call it.

## Where To Start

Start with `m2-victoria-park.qmd`, then confirm you can load and inspect the
dataset:

```python
from victoria_park_slam.datasets import load_victoria_park_data
from victoria_park_slam.utils import scan_to_xy

dataset = load_victoria_park_data()  # finds data/victoria-park
scan_xy = scan_to_xy(dataset.laser_ranges_m[0])

print(dataset.laser_ranges_m.shape)
print(scan_xy.shape)
```

For the front end, begin in `src/victoria_park_slam/landmarks.py`. For the SLAM
backend, begin in `src/victoria_park_slam/fastslam.py` and reuse the tested
geometry utilities in `src/victoria_park_slam/utils.py`.

A practical implementation order is:

1. Plot a few raw scans and verify the scanner angle convention.
2. Implement range filtering and scan clustering.
3. Convert accepted clusters into `TreeObservation` records.
4. Implement landmark initialization and update.
5. Add per-particle association, weighting, and resampling.
6. Track diagnostics so the final report can explain failures, not just show a
   final map.

## Useful Commands

From the repository root:

```bash
uv sync
uv run python scripts/fetch_data.py victoria-park
uv run pytest projects/victoria_park_slam/tests
uv run victoria-park-slam summary
uv run victoria-park-slam plot-scan --index 0 --output /tmp/victoria-park-scan-0.png
```

The tests check the starter utilities and data loader. They do not represent a
complete SLAM solution.

## Data

Download the course data with `uv run python scripts/fetch_data.py victoria-park`.
It is installed under:

```text
data/victoria-park/
  aa3_lsr2.mat
  aa3_dr.mat
  aa3_gpsx.mat
```

GPS is useful for rough plotting and sanity checks. Do not use GPS as a
recurring SLAM measurement unless the assignment explicitly changes that rule.
