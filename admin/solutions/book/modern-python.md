# Solutions: Modern Python Refresher (`sec-python-exercises`)

Instructor-only. All code below was run with pytest in the course environment
(Python 3.14, NumPy 2.4); every test shown passes.

## 1. Broadcasting bug hunt

**(a)** With `x` of shape `(n, 1)`, `H @ x` has shape `(2, 1)`. Subtracting it from
`z` of shape `(2,)` broadcasts to `(2, 2)`, `np.linalg.solve(S, y)` is `(2, 2)`,
and `y @ ...` is a `(2, 2)` matrix instead of a scalar. No error is raised.

**(b)**

```python
def nis(z: np.ndarray, H: np.ndarray, x: np.ndarray, S: np.ndarray) -> float:
    """Normalized innovation squared y^T S^-1 y for one measurement."""
    z = np.asarray(z, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    y = z - H @ x
    if y.shape != z.shape:
        raise ValueError(f"innovation shape {y.shape} does not match measurement {z.shape}")
    return float(y @ np.linalg.solve(S, y))
```

**(c)**

```python
def test_nis_is_scalar_for_column_state():
    z = np.array([1.0, 2.0])
    H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    x_col = np.zeros((3, 1))              # the teammate's convention
    S = np.diag([1.0, 4.0])
    result = nis(z, H, x_col, S)
    assert np.ndim(result) == 0
    assert result == pytest.approx(1.0**2 / 1.0 + 2.0**2 / 4.0)
```

On the original function the `np.ndim(result) == 0` assertion fails (the result
is `(2, 2)`); on the fix it passes with value 2.0. Accept any test that pins both
the shape and a hand-computed value.

## 2. Reproducible Monte Carlo

```python
TRUTH = np.array([100.0, -40.0, -5.0])

def run_trial(rng: np.random.Generator) -> float:
    fixes = simulate_gnss_fixes(TRUTH, 3.0, 100, rng)
    err = fixes[:, :2] - TRUTH[:2]
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))

# (a) reproducible
a = [run_trial(r) for r in np.random.default_rng(595).spawn(200)]
b = [run_trial(r) for r in np.random.default_rng(595).spawn(200)]
assert a == b

# (b) order independent: trial k always uses child stream k
rngs = np.random.default_rng(595).spawn(200)
reversed_results = [run_trial(r) for r in reversed(rngs)][::-1]
assert reversed_results == a

# (c) legacy global state
def run_trial_legacy() -> float:
    fixes = TRUTH + np.random.normal(0.0, 3.0, size=(100, 3))
    err = fixes[:, :2] - TRUTH[:2]
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))

np.random.seed(595)
forward = [run_trial_legacy() for _ in range(200)]
np.random.seed(595)
backward = [run_trial_legacy() for _ in range(200)][::-1]   # trial 199 ran first
assert forward != backward
```

The mean RMS horizontal error is close to $\sqrt{2}\times 3 \approx 4.24$ m. With
spawned generators each trial owns an independent stream, so its result depends
only on its index, not on when it runs (which also makes parallel execution
reproducible). With the legacy interface all trials share one global stream, so a
trial's numbers depend on how many draws happened before it; reordering, adding a
diagnostic draw, or running in parallel changes every result.

## 3. Course over ground, tested

```python
import math

def course_over_ground_deg(v_north: float, v_east: float, min_speed: float = 0.5) -> float:
    """Course over ground in degrees clockwise from north, in [0, 360)."""
    if math.hypot(v_north, v_east) < min_speed:
        raise ValueError("course over ground is undefined below min_speed")
    course = math.degrees(math.atan2(v_east, v_north)) % 360.0
    return 0.0 if course >= 360.0 else course   # a tiny negative angle rounds to 360.0


@pytest.mark.parametrize(("v", "expected"), [
    ((5.0, 0.0), 0.0), ((0.0, 5.0), 90.0), ((-5.0, 0.0), 180.0),
    ((0.0, -5.0), 270.0), ((5.0, -5.0), 315.0),
])
def test_cog(v, expected):
    assert course_over_ground_deg(*v) == pytest.approx(expected)


def test_cog_low_speed():
    with pytest.raises(ValueError):
        course_over_ground_deg(0.1, 0.1)
```

Swapping the arguments (`atan2(v_north, v_east)`) measures the angle
counterclockwise from east. All five listed cases catch it (north becomes 90°,
east 0°, south 270°, west 180°, 315° becomes 135°). A test at 45° or 225° would
*not*, because the diagonal is symmetric under the swap. Test cases should be
chosen off symmetry lines. Bonus point for noticing the wrap edge case:
`(-1e-20) % 360.0 == 360.0` in floating point, so a naive implementation can
return 360.0 for a vehicle heading due north with a tiny negative east velocity;
the guard in the return line handles it.

## 4. Package it

```bash
uv init --lib imu_tools
cd imu_tools
uv add numpy
uv add --dev pytest
```

`src/imu_tools/__init__.py` (or a module re-exported from it):

```python
from dataclasses import dataclass

import numpy as np


@dataclass
class ImuSample:
    t: float
    accel: np.ndarray
    gyro: np.ndarray

    def __post_init__(self) -> None:
        for name in ("accel", "gyro"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,):
                raise ValueError(f"{name} must have shape (3,), got {value.shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")
            setattr(self, name, value)
```

`tests/test_imu.py`:

```python
import numpy as np
import pytest

from imu_tools import ImuSample


def test_valid_sample():
    s = ImuSample(0.0, np.array([0.0, 0.0, -9.81]), np.zeros(3))
    assert s.accel.shape == (3,)


@pytest.mark.parametrize("accel, gyro", [
    (np.zeros(2), np.zeros(3)),                  # wrong shape
    (np.zeros(3), np.array([0.0, np.nan, 0.0])), # non-finite
])
def test_invalid_sample(accel, gyro):
    with pytest.raises(ValueError):
        ImuSample(0.0, accel, gyro)
```

`uv run pytest` passes. The import works from `tests/` because uv installed the
project into `.venv` in editable mode: the environment's `site-packages` points
at `src/`, so `import imu_tools` resolves through the installed package, not the
current directory. Running `python -m pytest` with a system Python fails with
`ModuleNotFoundError` (for `imu_tools`, and probably for `numpy` or `pytest`),
because that interpreter has never had the package installed. With a flat layout
(package directory at the project root) the import can succeed by accident when
run from the root, hiding packaging mistakes; the `src/` layout makes "is my
package actually installable?" part of every test run.

## 5. Hints are not enforced

```python
def add(x: str, y: str) -> str:
    return x + y  # concatenates two strings


print(add(1, 2))
```

`uv run python add_demo.py` prints `3`. `uv run --with mypy mypy add_demo.py`
prints

```text
add_demo.py:5: error: Argument 1 to "add" has incompatible type "int"; expected "str"  [arg-type]
add_demo.py:5: error: Argument 2 to "add" has incompatible type "int"; expected "str"  [arg-type]
Found 2 errors in 1 file (checked 1 source file)
```

Both are correct. The interpreter stores annotations but never checks them, and
`1 + 2` is valid Python, so the program runs. The type checker analyzes the code
statically against the annotations and reports that the call violates the
declared contract. Hints are documentation that a tool can verify, not a runtime
guard; if a runtime guard is needed, write an explicit check (as in Exercise 4).
