# Solutions: Sensing and Perception (`sec-sensing-exercises`)

Instructor-only. Numerical answers were checked with short Python scripts; the
code for Exercises 3 and 4 runs as written with NumPy only.

## 1. Equip an AUV

A defensible suite, with the role of each sensor:

| Sensor | Measures | Resolved in | Dominant error | Role |
|---|---|---|---|---|
| IMU (navigation or tactical grade, FOG or RLG gyros) | Specific force, angular rate | Body | Bias drift, random walk | Propagation at 100+ Hz |
| DVL (bottom track) | Velocity over the seafloor | Instrument, rotated to body | Misalignment, sound-speed scale factor, loss of bottom lock | Bounds velocity error, so position error grows roughly linearly with distance instead of with $t^2$ or $t^3$ |
| Pressure (depth) sensor | Water pressure, hence depth | Vertical | Water density and tide model, small bias | Bounds the vertical channel absolutely |
| Magnetometer or gyrocompass heading | Field direction / Earth-rate heading | Body | Hull and thruster magnetic distortion | Helps bound heading; often weak on a steel vehicle |
| Acoustic positioning (USBL from the ship, or LBL transponders) | Range and bearing to beacons | Ship or beacon frame | Sound-speed refraction, multipath, low rate | Absolute position fixes at depth |
| GNSS | Position before diving and on surfacing | Earth-fixed | Unavailable underwater | Initialization and recovery |
| Multibeam or imaging sonar (optional) | Ranges to seafloor and pipeline | Sensor | Multipath, sound speed | Terrain or feature (pipeline) relative corrections |

Propagation: IMU, with DVL velocity as the main aiding. Drift bounding: DVL
(velocity), depth sensor (vertical), heading reference, and, most importantly for
the 50 m recovery requirement, acoustic positioning. At 500 m depth there is no
GNSS; the only absolute horizontal references are acoustic beacons or the support
ship's USBL, or terrain/pipeline-relative matching against a map. A good answer
notes that a DVL-aided INS drifting at a few tenths of a percent of distance
traveled will exceed 50 m over a six-hour survey of tens of kilometers, so an
acoustic or map-relative fix is needed before recovery.

## 2. Timing on a drone and on a ship

**(a)** Position error $v\,\delta t = 4 \times 0.040 = 0.16$ m. Heading error
$\omega\,\delta t = 60 \times 0.040 = 2.4^\circ$. Both are far larger than the
few-centimeter PnP accuracy: the timing error, not the vision algorithm, would
dominate the error budget. At 1.5 m altitude, a $2.4^\circ$ attitude error also
moves the projected ground point by about $1.5 \tan 2.4^\circ \approx 6$ cm.

**(b)** Earth turns $15^\circ$ per hour, i.e. $0.25'$ of arc per second of time.
A 2 minute (120 s) clock error gives $120 \times 15/3600 = 0.5^\circ$ of longitude,
which is $30'$ of arc. One minute of arc of longitude is one nautical mile at the
equator and $\cos L$ nautical miles at latitude $L$, so the error is 30 nmi at the
equator and $30 \cos 40^\circ \approx 23$ nmi at $40^\circ$ latitude. Because the
chronometer is fast, the navigator believes Greenwich time is later than it is,
and computes a position $0.5^\circ$ too far west.

## 3. A measurement contract in code

```python
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Measurement:
    """A sensor measurement and the context an estimator needs to use it."""
    t_valid: float            # s, when the quantity was true (estimator clock)
    t_arrival: float          # s, when the packet reached the computer
    value: np.ndarray         # measurement vector, SI units
    frame: str                # frame the components are resolved in, e.g. "n", "b", "c"
    covariance: np.ndarray    # R, units of value squared
    sensor_id: str
    valid: bool = True

    @property
    def latency(self) -> float:
        return self.t_arrival - self.t_valid


def interpolate_state(times: np.ndarray, states: np.ndarray, t: float) -> np.ndarray:
    """Linearly interpolate a buffer of states (rows of `states`) to time t."""
    times = np.asarray(times, dtype=float)
    if not (times[0] <= t <= times[-1]):
        raise ValueError(f"t = {t:.3f} s is outside the state buffer "
                         f"[{times[0]:.3f}, {times[-1]:.3f}] s")
    i = int(np.searchsorted(times, t, side="right")) - 1
    i = min(i, len(times) - 2)                 # t == times[-1] uses the last interval
    alpha = (t - times[i]) / (times[i + 1] - times[i])
    return (1.0 - alpha) * states[i] + alpha * states[i + 1]


# Test: constant-velocity truth sampled at 100 Hz; measurement valid 37 ms before arrival.
times = np.arange(0.0, 2.0, 0.01)
velocity = np.array([1.5, -0.5])
states = np.column_stack([times * velocity[0], times * velocity[1]])
z = Measurement(t_valid=1.2345, t_arrival=1.2715, value=np.array([1.85, -0.62]),
                frame="n", covariance=0.05**2 * np.eye(2), sensor_id="cam0")
x_at_valid = interpolate_state(times, states, z.t_valid)
np.testing.assert_allclose(x_at_valid, z.t_valid * velocity, atol=1e-12)
print(z.latency, x_at_valid)          # 0.037, [ 1.85175 -0.61725]
```

Linear interpolation is exact for a constant-velocity trajectory, which makes the
test sharp. Points to look for in student work: the function refuses to
extrapolate, the validity time (not arrival time) is used, and the frame and
covariance travel with the value. Attitude should not be interpolated linearly in
a real system (use SLERP on quaternions); a strong student will say so.

## 4. Find the clock offset

```python
import numpy as np

rng = np.random.default_rng(0)
true_delay = 0.065                                   # s
t_gyro = np.arange(0.0, 60.0, 1.0 / 200.0)


def yaw_rate(t):
    return (0.4 * np.sin(2 * np.pi * 0.2 * t) + 0.25 * np.sin(2 * np.pi * 0.7 * t + 1.0)
            + 0.15 * np.sin(2 * np.pi * 1.3 * t + 2.0))


gyro = yaw_rate(t_gyro) + rng.normal(0.0, 0.02, t_gyro.size)
t_vo = np.arange(0.0, 60.0, 1.0 / 20.0)             # VO's own timestamps
vo = yaw_rate(t_vo - true_delay) + rng.normal(0.0, 0.05, t_vo.size)   # stamped late

t_common = np.arange(1.0, 59.0, 0.001)              # 1 ms grid over the overlap
g = np.interp(t_common, t_gyro, gyro); g -= g.mean()
v = np.interp(t_common, t_vo, vo); v -= v.mean()
lags = np.arange(-300, 301)                          # +/- 300 ms in 1 ms steps
corr = [np.dot(g[300:-300], np.roll(v, -lag)[300:-300]) for lag in lags]
print(lags[int(np.argmax(corr))])                    # 64 (ms); truth 65
```

The estimate lands within a millisecond or two of the truth. If the motion is
nearly constant, the cross-correlation is flat (a constant signal correlates
equally well at every lag) and the peak is set by noise, so the offset becomes
unobservable. Clock calibration needs *excitation*: rotate the platform briskly,
with several frequencies, during the calibration run. This is the same
observability idea that returns in the filtering chapters.

## 5. Which box?

- **Wheel odometry**: proprioceptive by mechanism and in behavior. Surprise if
  treated as exteroceptive: wheel slip on a wet floor or loose gravel produces
  large, correlated errors that are not white measurement noise, and the error
  integrates into unbounded drift.
- **DVL in bottom track**: exteroceptive by mechanism (acoustic returns from the
  seafloor) but used as a velocity input, so its errors integrate like odometry.
  Surprise if treated as an absolute measurement: misalignment and scale errors
  produce position drift proportional to distance traveled; surprise if treated as
  infallible proprioception: it loses bottom lock over rough terrain or when the
  seafloor is out of range, and can silently switch to water track (velocity
  relative to a moving current).
- **Airspeed sensor**: measures the air, not the ground. Used for propagation it
  behaves like odometry, but its "drift" is the wind: integrating airspeed gives
  position relative to the air mass, which can be wrong by the wind speed times
  the elapsed time. A designer who treats it as ground velocity is surprised by
  steady crosswind drift.
- **Visual odometry**: exteroceptive by mechanism, proprioceptive in behavior once
  integrated. Surprise if treated as an absolute fix: it drifts without loop
  closure, scale drifts for a monocular system, and it fails in textureless or
  dynamic scenes, often abruptly.

The conclusion students should reach: for filter design, what matters is whether
the sensor's errors are bounded (absolute reference) or integrate (relative
increment), and what its failure modes look like, not the label.
