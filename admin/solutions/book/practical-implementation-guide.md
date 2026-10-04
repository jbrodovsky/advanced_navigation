# Solutions: From Prototype to Vehicle: Navigation System Integration (`practical-implementation-guide.qmd`)

Instructor-only. Exercise numbers match `## Exercises {#sec-practical-exercises}`.
Code snippets assume the chapter's cells have been run (in particular `C_NED_TO_ENU`,
`C_FRD_TO_FLU`, `skew`, `wxyz_to_matrix`, `to_ros_odometry`, and `StateBuffer`). The
numbers quoted below were produced with that code (NumPy/SciPy, seeds as shown).

## Exercise 1: A time-sync and output contract for a delivery drone

This is a design exercise; grade the reasoning, not the exact numbers. A strong answer
looks roughly like the following.

**(a) Clocks.** The GNSS receiver is the time master. Its PPS goes to the companion
computer (chrony with a PPS reference clock, labeled by the receiver's time messages) and,
ideally, to a GPIO on the flight controller as well, so both computers share GPS-traceable
time to a few microseconds. The estimator runs on GPS time (or a monotonic clock with a
known offset to it); UTC appears only in logs. Without the PPS line on the flight
controller, its clock is related to the companion's by a round-trip time-sync exchange over
the serial or UDP link (as MAVLink's TIMESYNC and the PX4-ROS 2 bridge do), good to roughly
0.1-1 ms depending on link jitter. Per sensor:

| Device | How it is stamped | Residual timing error |
|---|---|---|
| IMU (flight controller) | data-ready interrupt in FC time, converted with the FC-companion offset | ~10 µs with PPS on the FC; 0.1-1 ms with round-trip sync |
| GNSS | receiver's time of fix (GPS time) | < 1 µs |
| Camera | hardware trigger from the FC (or the camera's strobe into an FC GPIO); stamp = trigger + half the 8 ms exposure | ~0.1 ms |
| Lidar altimeter | free-running; stamped on arrival at the FC, then corrected by an estimated offset and drift | 1-2 ms (arrival jitter) |

Justification with $v\,\delta t$ and $\omega\,\delta t$: at 15 m/s, 1 ms is 1.5 cm, negligible
against 1-2 m standalone GNSS (RTK at 2 cm would demand sub-millisecond timing). For the
landing camera the binding constraint is rotation: in gusts the body rate can reach 100°/s,
so 1 ms is 0.1°, or 1.7 cm at 10 m height, comparable to the marker-pose accuracy; 5 ms would
be 9 cm. Hence the hardware trigger. The lidar altimeter at 3 m/s vertical speed and 2 ms
error is 6 mm against a few centimeters of range noise: fine without hardware sync.

**(b) Offsets.** Offline: camera-IMU extrinsics and time offset with Kalibr (state the
exposure-midpoint convention); GNSS-antenna and lidar lever arms by measurement; lidar
latency by cross-correlating lidar range rate with INS vertical velocity during climbs and
descents. Online: the camera time offset as a filter state with a small random walk (it
drifts with exposure settings); the lidar clock offset and drift by the lower-envelope fit of
(arrival time - sensor time). The GNSS needs no offset state because its stamp is GPS time.

**(c) Buffer.** Worst latency: camera 4 ms (half exposure) + 60 ms processing + ~10 ms
transport ≈ 75 ms; GNSS ≈ 80 ms plus jitter, say 150 ms worst case. Retain 300 ms (a 2×
margin): 120 entries at 400 Hz, a trivial amount of memory.

**(d) Outputs.**

| Consumer | Content and frame | Rate / latency | May it jump? |
|---|---|---|---|
| Rate loop (FC) | bias-corrected gyro, FRD body | 1 kHz, < 1 ms | no |
| Attitude and velocity loops (FC) | attitude; velocity in the odom (local-level) frame, continuous | 250 Hz attitude, 50-100 Hz velocity; < 5 ms | no |
| Landing-marker tracker | camera pose at each exposure stamp (tf lookup at that time), odom frame | 30 Hz, stamp-consistent rather than low-latency | no |
| Mission planner and geofence | map frame (ENU at home or geodetic) with covariance | 5-10 Hz | yes, with reset notification |

**(e) Health and degraded modes.** Fields: mode (INITIALIZING, GNSS_AIDED, VISION_AIDED for
landing, DEAD_RECKONING, INERTIAL_ONLY, FAULT), active aids, time since the last GNSS fix,
horizontal and vertical σ, per-aid NIS flags, output latency, time-sync status (PPS locked,
FC offset σ), and the camera offset estimate with its σ. Policies: GNSS lost en route →
widen geofence margins with σ; if σ_h exceeds about 10 m or GNSS has been absent for 30 s,
land at the nearest safe site rather than attempt a long dead-reckoned return (a quadrotor
with a MEMS IMU and no velocity aid drifts quickly). Vision lost during landing → abort,
climb to a safe height, retry, then divert. Lidar lost → fall back to the barometer, raise
minimum altitudes, prohibit precision landing. PPS lost → keep the free-running offset,
inflate the camera offset σ; if it exceeds ~2 ms, disable precision landing. Stale output →
the FC's own watchdog switches to its onboard attitude estimate and lands.

**Where to put the filter.** If the companion computer's latency is unpredictable, the
attitude/velocity loops cannot depend on it. Put the INS propagation and an output predictor
on the flight controller (deterministic, RTOS), and let the companion send timestamped
measurements (vision poses, marker detections) that the FC applies at their own times
through its buffer or a delayed fusion horizon. This is PX4's architecture (EKF2 on the FC,
vision from the companion as external measurements). The heavy perception stays on the
companion; the time-critical estimation does not.

## Exercise 2: Coming back from ROS

**(a) Derivation.** With $\mathbf{C}_b^n = (\mathbf{I} + [\delta\boldsymbol{\theta}]_\times)\hat{\mathbf{C}}_b^n$,
transposing gives $\mathbf{C}_n^b = \hat{\mathbf{C}}_n^b(\mathbf{I} - [\delta\boldsymbol{\theta}]_\times)$
(the transpose of a skew matrix is its negative). Then, to first order,

$$
\mathbf{v}^b = \hat{\mathbf{C}}_n^b(\mathbf{I} - [\delta\boldsymbol{\theta}]_\times)(\hat{\mathbf{v}}^n + \delta\mathbf{v}^n)
\approx \hat{\mathbf{v}}^b + \hat{\mathbf{C}}_n^b\delta\mathbf{v}^n - \hat{\mathbf{C}}_n^b(\delta\boldsymbol{\theta}\times\hat{\mathbf{v}}^n),
$$

and $-\delta\boldsymbol{\theta}\times\hat{\mathbf{v}}^n = \hat{\mathbf{v}}^n\times\delta\boldsymbol{\theta} = [\hat{\mathbf{v}}^n]_\times\delta\boldsymbol{\theta}$.
So $\delta\mathbf{v}^b \approx \hat{\mathbf{C}}_n^b(\delta\mathbf{v}^n + [\hat{\mathbf{v}}^n]_\times\delta\boldsymbol{\theta})$,
and multiplying by $\mathbf{C}_{\text{FRD}}^{\text{FLU}}$ gives @eq-practical-twist-error. The chapter's
Monte Carlo check confirms the sign: the twist covariance matches to about 3% only with
this term (without it the sideways σ is 0.025 m/s instead of 0.049 m/s).

**(b) Inverse conversion.**

```python
def from_ros_odometry(ros):
    """Inverse of to_ros_odometry for the state (not the covariance)."""
    T, B = C_NED_TO_ENU, C_FRD_TO_FLU
    p_ned = T @ np.asarray(ros["position"])                 # T is its own inverse
    C_flu_enu = Rotation.from_quat(ros["orientation_xyzw"]).as_matrix()
    C_bn = T @ C_flu_enu @ B                                 # undo C_flu_enu = T C_bn B
    x, y, z, w = Rotation.from_matrix(C_bn).as_quat()
    q_bn = np.array([w, x, y, z])                            # back to scalar first
    v_ned = C_bn @ (B @ np.asarray(ros["linear"]))
    omega_nb_frd = B @ np.asarray(ros["angular"])
    return p_ned, v_ned, q_bn, omega_nb_frd


rng = np.random.default_rng(2)
worst = 0.0
for trial in range(2000):
    ypr = np.r_[rng.uniform(-180, 180), rng.uniform(-90, 90), rng.uniform(-180, 180)]
    if trial < 10:                                           # pitch within 1e-7 deg of +/-90
        ypr[1] = 90.0 * (-1) ** trial - (1e-7 if trial % 2 == 0 else -1e-7)
    x, y, z, w = Rotation.from_euler("ZYX", ypr, degrees=True).as_quat()
    q = np.array([w, x, y, z])
    p, v, om = rng.normal(0, 100, 3), rng.normal(0, 5, 3), rng.normal(0, 0.5, 3)
    back = from_ros_odometry(to_ros_odometry(p, v, q, om, np.eye(15)))
    worst = max(worst, np.max(np.abs(back[0] - p)), np.max(np.abs(back[1] - v)),
                abs(abs(back[2] @ q) - 1.0), np.max(np.abs(back[3] - om)))
print(f"worst round-trip error over 2000 random states: {worst:.2e}")   # 1.1e-14
```

Pitch near ±90° causes no trouble because nothing passes through Euler angles; the
quaternion comparison uses $|\mathbf{q}_1\cdot\mathbf{q}_2| = 1$ to allow for the sign
ambiguity.

**(c) Why the covariance cannot be recovered.** The two published covariances are
$\mathbf{J}_{pose}\mathbf{P}\mathbf{J}_{pose}^T$ and $\mathbf{J}_{twist}\mathbf{P}\mathbf{J}_{twist}^T$,
12 rows of maps from 15 states, and they are published separately. Lost: (i) the
accelerometer-bias block entirely (no row involves $\delta\mathbf{b}_a$); (ii) the gyro-bias
block separately from rate noise (it appears only as the rate covariance); (iii) every
pose-twist cross-covariance, including position-velocity, which is exactly the correlation
a downstream fusion or prediction step needs; (iv) the velocity block itself, because the
twist covariance mixes $\delta\mathbf{v}$ with $\delta\boldsymbol{\theta}$ through
$[\hat{\mathbf{v}}^n]_\times$, and without the cross-covariance between velocity and
attitude errors the mixture cannot be undone. A consumer that needs the full joint
uncertainty must get it from a dedicated message, not from `nav_msgs/Odometry`.

**(d) IMU conversion.** The swap is its own inverse:

```python
def imu_flu_to_frd(gyro_flu, accel_flu):
    return C_FRD_TO_FLU @ gyro_flu, C_FRD_TO_FLU @ accel_flu
```

Level at rest: FLU reads $[0, 0, +9.81]$, FRD $[0, 0, -9.81]$, as the book's specific-force
convention requires. Rolled 90° right (right side down): the specific force points up, which
is the FLU $+y$ (left) axis, so FLU reads $[0, +9.81, 0]$ and FRD $[0, -9.81, 0]$. Check:
$\mathbf{C}_n^b[0, 0, -g]^T$ with roll 90° gives $[0, -g, 0]$.

## Exercise 3: Estimating a GNSS time offset online

The filter state is $[\mathbf{p}, \mathbf{v}, t_d]$ (2D), propagated with the accelerometer;
the measurement is $\mathbf{h} = \hat{\mathbf{p}} - \hat{\mathbf{v}}\hat{t}_d$ with
$\mathbf{H} = [\mathbf{I},\ -\hat{t}_d\mathbf{I},\ -\hat{\mathbf{v}}]$.

```python
import numpy as np

dt, T_total, true_td = 0.01, 240.0, 0.080
sigma_acc, sigma_gnss = 0.05, 0.5
n = round(T_total / dt)
t = np.arange(n + 1) * dt


def trajectory(kind):
    """Position, velocity, acceleration (navigation axes) for the two test drives."""
    if kind == "straight":
        p = np.stack([15.0 * t, np.zeros_like(t)], axis=1)
        v = np.tile([15.0, 0.0], (t.size, 1))
        a = np.zeros_like(p)
    else:   # figure-eight (lemniscate of Gerono), 60 s period, mean speed ~15 m/s
        w = 2 * np.pi / 60.0
        A = 15.0 / (w * 0.97)
        p = np.stack([A * np.sin(w * t), A * np.sin(w * t) * np.cos(w * t)], axis=1)
        v = np.stack([A * w * np.cos(w * t), A * w * np.cos(2 * w * t)], axis=1)
        a = np.stack([-A * w**2 * np.sin(w * t), -2 * A * w**2 * np.sin(2 * w * t)], axis=1)
    return p, v, a


def run(kind, seed=3):
    rng = np.random.default_rng(seed)
    p, v, a = trajectory(kind)
    # An IMU reports velocity increments: use the interval-average acceleration.
    accel = np.diff(v, axis=0) / dt
    accel = np.vstack([accel, accel[-1]]) + sigma_acc * rng.standard_normal(a.shape)
    x = np.r_[p[0] + rng.normal(0, 3.0, 2), v[0] + rng.normal(0, 0.3, 2), 0.0]
    P = np.diag([3.0**2, 3.0**2, 0.3**2, 0.3**2, 0.1**2])
    F = np.eye(5); F[0, 2] = F[1, 3] = dt
    G = np.zeros((5, 2)); G[0, 0] = G[1, 1] = 0.5 * dt**2; G[2, 0] = G[3, 1] = dt
    Q = sigma_acc**2 * G @ G.T
    Q[4, 4] = (1e-5) ** 2 * dt
    R = sigma_gnss**2 * np.eye(2)
    log = []
    for k in range(n):
        x = F @ x + G @ accel[k]
        P = F @ P @ F.T + Q
        if (k + 1) % 100 == 0:                        # 1 Hz fix, stamped t, valid at t - t_d
            t_valid = t[k + 1] - true_td
            z = np.array([np.interp(t_valid, t, p[:, 0]), np.interp(t_valid, t, p[:, 1])])
            z += sigma_gnss * rng.standard_normal(2)
            H = np.zeros((2, 5))
            H[:, 0:2] = np.eye(2)
            H[:, 2:4] = -x[4] * np.eye(2)
            H[:, 4] = -x[2:4]
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ (z - (x[0:2] - x[2:4] * x[4]))
            I_KH = np.eye(5) - K @ H
            P = I_KH @ P @ I_KH.T + K @ R @ K.T
            log.append((t[k + 1], x[4], np.sqrt(P[4, 4])))
    return np.array(log)
```

Results (seed 3), $\hat{t}_d \pm 1\sigma$:

| Drive | after 60 s | after 240 s |
|---|---|---|
| straight, 15 m/s | 60.0 ± 89.3 ms | 62.1 ± 89.2 ms |
| figure-eight, mean 15 m/s | 81.7 ± 5.4 ms | 78.8 ± 2.5 ms |

Over 30 seeds the figure-eight's final normalized error $(\hat{t}_d - t_d)/\sigma$ has mean
0.4 and standard deviation 0.8: close to consistent, slightly conservative.

**Observability.** On the straight line, $\mathbf{z}_k = \mathbf{p}_0 + \mathbf{v}t_k - \mathbf{v}t_d$:
the data determine $\mathbf{v}$ (the slope) and the combination $\mathbf{p}_0 - \mathbf{v}t_d$,
but not $\mathbf{p}_0$ and $t_d$ separately. The $t_d$ standard deviation never moves far
from its 100 ms prior; what little it learns comes from the 3 m prior on the initial
position. The position estimate absorbs the remainder, ending a fraction of a meter off
along track. On the figure-eight, the velocity changes direction and magnitude and the
accelerometer tells the filter exactly how, so $-\mathbf{v}(t)t_d$ varies in a way no
constant position error can mimic, and $t_d$ converges within a minute. Ask the test driver
for turns in both directions and speed changes (a slalom or a figure-eight in a parking lot),
not a highway run.

**A subtle point worth discussing.** A first attempt that feeds the filter point samples of
the acceleration at the start of each interval, $\mathbf{a}(t_k)$, biases $\hat{t}_d$ low by
about 5 ms (over 30 seeds the normalized error averaged −1.2). Rectangular integration of
left-point samples makes the integrated trajectory lag by half a sample, $\Delta t/2 = 5$ ms,
and the offset state dutifully absorbs it. Real IMUs report velocity increments (integrals
over the interval), which is what the code above simulates. An offset estimator measures
every timing convention in the system, including the ones you did not know you had.

## Exercise 4: The late USBL fix

Geometry: slant range $\sqrt{1500^2 + 1000^2} = 1803$ m, so the USBL σ is
$0.002 \times 1803 = 3.6$ m per horizontal axis.

**(a) Applied at arrival time.** In 6 s the AUV moves 12 m east, so the fix (which describes
where the AUV was) sits 12 m west of the state it is compared with: an innovation of about
−12 m east plus noise. With a horizontal state σ of about 2 m, $S \approx 3.6^2 + 2^2 = 17$ m²
per axis. The bias contributes $12^2/17 \approx 8.5$ to the NIS, so the expected NIS is about
10.5 against a 99% gate of $\chi^2_2(0.99) = 9.21$. The NIS is noncentral chi-square, and
about 48% of fixes pass the gate. The gate does not save you: each accepted fix pulls the
estimate westward, the next innovations shrink, more fixes pass, and the estimate settles
behind the truth. With the gate disabled, the along-track error obeys
$e^+ = (1 - K)e - 12K$ at each fix (DVL propagation keeps it unchanged between fixes), whose
fixed point is $e^* = -12$ m for any gain $K > 0$. The filter ends up tracking where the
vehicle was 6 s ago while its covariance shrinks to a couple of meters: a 12 m bias reported
as 2 m σ. During the transient the filter will also try to explain the persistent
"the vehicle is behind" evidence with the DVL scale factor, biasing it low. At a line turn,
the bias rotates with the velocity, and the filter may misattribute part of it to heading.
The fix is event-time processing, not a wider gate or a larger R.

**(b) Buffer size.** 200 Hz × 12 s = 2400 entries. Each entry: 1 (time) + 23 (nominal) +
484 (22 × 22 covariance) + 6 (IMU increments) = 514 doubles = 4112 bytes, so about 9.9 MB.
Reductions: store the IMU increments at every sample (7 doubles, 134 kB for 2400 samples)
and full checkpoints (state plus covariance upper triangle, 1 + 23 + 253 doubles = 2.2 kB)
only once per second (13 checkpoints, 29 kB). To process a late fix, restore the checkpoint
just before $t_m$, re-propagate the increments to $t_m$, update, and re-propagate to the
present. Total about 0.16 MB, a factor of about 60. Storing only the covariance's upper
triangle alone saves a factor of two.

**(c) Ship-side errors.** Heading: $1000 \times 0.08° \times \pi/180 = 1.4$ m, perpendicular to
the ship-to-AUV bearing. Timing: 15 ms at 5°/s of roll is a 0.075° attitude error at the
ping; in the worst geometry it acts on the full slant range,
$1803 \times 0.075° \times \pi/180 = 2.4$ m. Both are comparable to the USBL's own 3.6 m noise
(RSS of all three about 4.5 m), and both are worse than they look: the heading error is a
bias that never averages out, and the roll-timing error is correlated over the roll period
(several seconds), so consecutive fixes share it. A filter that models USBL errors as white
with σ = 3.6 m will be overconfident. Either fix the ship (better heading, common PPS) or
inflate R and add a slowly varying USBL bias state.

## Exercise 5: Testing the buffer, and adding attitude

A module with slerp (the chapter's buffer plus an attitude field):

```python
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass

import numpy as np


def slerp(q0, q1, alpha):
    """Spherical interpolation between unit scalar-first quaternions, 0 <= alpha <= 1."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    dot = float(q0 @ q1)
    if dot < 0.0:                 # q and -q are the same rotation: take the short way
        q1, dot = -q1, -dot
    if dot > 0.9995:              # nearly identical: normalized lerp avoids 0/0
        q = (1.0 - alpha) * q0 + alpha * q1
        return q / np.linalg.norm(q)
    theta = np.arccos(dot)
    return (np.sin((1.0 - alpha) * theta) * q0 + np.sin(alpha * theta) * q1) / np.sin(theta)


@dataclass(frozen=True)
class NavState:
    time_s: float
    position_m: np.ndarray
    velocity_mps: np.ndarray
    q_bn: np.ndarray              # Hamilton, scalar first


class StateBuffer:
    def __init__(self, max_age_s):
        self.max_age_s = max_age_s
        self._times, self._states = deque(), deque()

    def push(self, state):
        if self._times and state.time_s <= self._times[-1]:
            raise ValueError("states must arrive in strictly increasing event time")
        self._times.append(state.time_s)
        self._states.append(state)
        while state.time_s - self._times[0] > self.max_age_s:
            self._times.popleft()
            self._states.popleft()

    @property
    def interval(self):
        return self._times[0], self._times[-1]

    def __len__(self):
        return len(self._times)

    def interpolate(self, time_s):
        if not self._times or not (self._times[0] <= time_s <= self._times[-1]):
            raise ValueError(f"t = {time_s} s is outside the state buffer")
        i = bisect_right(self._times, time_s)
        if i == len(self._times):
            return self._states[-1]
        a, b = self._states[i - 1], self._states[i]
        alpha = (time_s - a.time_s) / (b.time_s - a.time_s)
        return NavState(time_s,
                        (1 - alpha) * a.position_m + alpha * b.position_m,
                        (1 - alpha) * a.velocity_mps + alpha * b.velocity_mps,
                        slerp(a.q_bn, b.q_bn, alpha))
```

Tests (`uv run pytest test_nav_buffer.py`; all 11 pass in about 0.3 s):

```python
from bisect import bisect_left

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from nav_buffer import NavState, StateBuffer, slerp

V = np.array([30.0, 0.0, 0.0])
OMEGA = np.deg2rad([0.0, 0.0, 20.0])


def q_at(t):
    """Attitude after rotating at the constant body rate OMEGA for t seconds."""
    x, y, z, w = Rotation.from_rotvec(OMEGA * t).as_quat()
    return np.array([w, x, y, z])


@pytest.fixture
def buffer():
    buf = StateBuffer(max_age_s=2.0)
    for k in range(301):                      # 3 s at 100 Hz; keeps [11, 13]
        t = 10.0 + 0.01 * k
        buf.push(NavState(t, V * (t - 10.0), V, q_at(t - 10.0)))
    return buf


def test_eviction_keeps_max_age(buffer):
    oldest, newest = buffer.interval
    assert newest == pytest.approx(13.0)
    assert newest - oldest <= 2.0 + 1e-9
    assert len(buffer) == 201


@pytest.mark.parametrize("which", ["oldest", "newest"])
def test_edges_are_valid_queries(buffer, which):
    t = buffer.interval[0 if which == "oldest" else 1]
    np.testing.assert_allclose(buffer.interpolate(t).position_m, V * (t - 10.0), atol=1e-9)


def test_interior_linear_motion_is_exact(buffer):
    for t in np.linspace(*buffer.interval, 37):
        np.testing.assert_allclose(buffer.interpolate(t).position_m, V * (t - 10.0), atol=1e-9)


@pytest.mark.parametrize("dt", [-1e-6, 1e-6])
def test_outside_raises(buffer, dt):
    oldest, newest = buffer.interval
    with pytest.raises(ValueError):
        buffer.interpolate(oldest + dt if dt < 0 else newest + dt)


def test_non_monotonic_push_raises(buffer):
    with pytest.raises(ValueError):
        buffer.push(NavState(12.5, V, V, q_at(0.0)))
    with pytest.raises(ValueError):
        buffer.push(NavState(13.0, V, V, q_at(0.0)))      # equal time is also rejected


def test_bisect_left_version_fails_at_oldest(buffer):
    """The old lookup rejected i == 0, which bisect_left returns for t == oldest."""
    oldest = buffer.interval[0]
    assert bisect_left(buffer._times, oldest) == 0       # so the old code raised here
    buffer.interpolate(oldest)                           # the fixed code does not


def test_slerp_exact_for_constant_rate(buffer):
    for t in np.linspace(*buffer.interval, 23):
        q = buffer.interpolate(t).q_bn
        assert abs(abs(q @ q_at(t - 10.0)) - 1.0) < 1e-12


def test_slerp_double_cover():
    mid = slerp(q_at(0.0), -q_at(0.1), 0.5)              # same rotation, opposite sign
    assert abs(abs(mid @ q_at(0.05)) - 1.0) < 1e-12      # short way, not the long way round
```

The edge test `test_edges_are_valid_queries[oldest]` is the one that fails for the old
`bisect_left` version (it raised "outside the buffer" for a query at exactly the oldest
sample). Slerp is exact for a constant body rate because the relative rotation between two
samples is then a rotation about a fixed axis at a constant rate, which is exactly the path
slerp follows. Without the sign flip, interpolating between $\mathbf{q}$ and a stored
$-\mathbf{q}'$ goes the long way round (through a rotation of nearly 360°), and the double-cover
test catches it.

**(c) Why not interpolate the covariance.** Between samples the covariance evolves as
$\mathbf{P}(t) = \boldsymbol{\Phi}\mathbf{P}\boldsymbol{\Phi}^T + \mathbf{Q}$, which is not
a linear blend of its endpoint values; a blend is positive semidefinite but bears no
particular relation to the true covariance, and in particular it gets the cross-covariances
between position, velocity, and attitude errors wrong, which is exactly what makes a
position update correct velocity and attitude. If an update happened between the two
samples, the covariance jumps there and interpolation smears that jump across the interval.
The right procedure is to rewind to the stored state and covariance at the last sample at or
before $t_m$, propagate the fractional step with the stored IMU increment, update, and
re-propagate.
