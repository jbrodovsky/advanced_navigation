# INS/GNSS Integration

This package is the starter code surface for the INS/GNSS integration project.
It focuses on curved-earth strapdown mechanization, error-state Kalman filtering,
and comparison against a baseline reference solution.

The assignment handout is:

- `../../book/assignments/m3-ins-gnss.qmd`

The package is deliberately not a complete navigation system. It provides data
loaders, typed records, plotting hooks, and the public classes students should
complete.

## Current Code Surface

```text
projects/ins_gnss/
  pyproject.toml
  src/ins_gnss/
    cli.py
    datasets.py
    eskf.py
    plotting.py
    strapdown.py
    types.py
  tests/
    test_ins_gnss.py
```

The implemented support code is:

- `datasets.py`: loads a conventional local `imu.csv` / `gnss.csv` layout and
  the fetched MEMS navigation dataset layout with paired `input/` and
  `baseline/` files.
- `types.py`: dataclasses for IMU samples, GNSS samples, generic datasets, MEMS
  reference states, and MEMS trials.
- `strapdown.py`: `IMUCalibration` plus `StrapdownNavigator` state
  initialization and WGS84 helper methods for meridian radius, prime vertical
  radius, and gravity.
- `eskf.py`: the public `INSGNSSFilter` contract for prediction and GNSS
  correction.
- `plotting.py`: a small covariance-diagonal plotting helper.
- `cli.py`: a minimal dataset inspection command.
- `tests/`: regression tests for CSV parsing, MEMS trial discovery, MEMS trial
  loading, fetch-script helpers, and strapdown helper construction.

The intentionally incomplete student implementation points are:

- `strapdown.StrapdownNavigator.integrate_imu(...)`
- `eskf.INSGNSSFilter.predict_with_imu(...)`
- `eskf.INSGNSSFilter.update_with_gnss(...)`

These are the main algorithmic work areas. The loader keeps estimator inputs
and reference/baseline outputs separate so students can evaluate honestly
without accidentally feeding truth back into the filter.

## Where To Start

Start with the architecture and Part A sections in `m3-ins-gnss.qmd`. Then load
one trial and inspect the normalized records:

```python
from pathlib import Path

from ins_gnss.datasets import list_mems_nav_trials, load_mems_nav_trial

root = Path("data/mems-nav-dataset")
trial_name = list_mems_nav_trials(root)[0]
trial = load_mems_nav_trial(root, trial_name)

print(trial.name)
print(len(trial.imu_samples), len(trial.gnss_samples))
print(trial.reference_states[0].latitude_rad)
```

For mechanization work, begin in `src/ins_gnss/strapdown.py`. For filtering
work, begin in `src/ins_gnss/eskf.py` after the strapdown propagation behaves
sensibly.

## Useful Commands

From the repository root:

```bash
uv sync
uv run python scripts/fetch_ins_gnss_data.py
uv run python -m unittest discover -s projects/ins_gnss/tests
uv run --package ins-gnss python -m ins_gnss.cli data/mems-nav-dataset --mems
```

To inspect one fetched trial:

```bash
uv run --package ins-gnss python -m ins_gnss.cli data/mems-nav-dataset --mems --trial TRIAL_NAME
```

Replace `TRIAL_NAME` with one of the names printed by the summary command.

## Data

The MEMS dataset is not checked directly into this repository. Fetch it with:

```bash
uv run python scripts/fetch_ins_gnss_data.py
```

By default it writes:

```text
data/mems-nav-dataset/
  input/
  baseline/
  manifest.json
```

Use `input/` as estimator input. Use `baseline/` for comparison and reporting,
not as a measurement source.
