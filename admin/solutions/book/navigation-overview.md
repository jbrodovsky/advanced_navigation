# Solutions: Guidance, Navigation, and Control (`navigation-overview.qmd`)

Instructor-only. Numbers for Exercise 4 come from the chapter's closed-loop
simulation (seed 4, 100 Hz, 30 s, $k_p = 16$, $k_d = 5.6$, metrics computed
for $t \ge 2$ s). The Exercise 4 snippets assume the chapter's
`fig-gnc-nav-quality-control` cell has been run, so `t`, `n`, `dt`, `r`,
`r_dot`, `r_ddot`, `kp`, `kd`, `gust`, `white`, and `drift_sign` exist.

## Exercise 1: Dead reckoning off Cape Cod

**(a)** Longitudes are east-positive in the code (70° W is $-70$).

```python
import numpy as np


def mid_latitude_sailing(lat0_deg, lon0_deg, course_deg, dist_nmi):
    """Advance a position by mid-latitude sailing.

    lat0_deg, lon0_deg: start position [deg], north and east positive.
    course_deg: true course [deg clockwise from north].
    dist_nmi: distance run [nmi].
    Returns (lat_deg, lon_deg).
    """
    c = np.radians(course_deg)
    dlat_arcmin = dist_nmi * np.cos(c)          # 1 arcmin of latitude = 1 nmi
    departure = dist_nmi * np.sin(c)            # east-west distance [nmi]
    lat1 = lat0_deg + dlat_arcmin / 60.0
    lat_mid = np.radians(0.5 * (lat0_deg + lat1))
    dlon_arcmin = departure / np.cos(lat_mid)   # meridians converge
    return lat1, lon0_deg + dlon_arcmin / 60.0


print(mid_latitude_sailing(42.0, -70.0, 90.0, 120.0))   # (42.0, -67.3087)
```

Distance run is $12 \times 10 = 120$ nmi, all departure. The change in
longitude is $120/\cos 42^\circ = 161.5'$ $= 2.691^\circ$ east, so the DR
position is **42° 00.0′ N, 67° 18.5′ W**.

**(b)** The current adds 10 nmi toward 180° over 10 hours. The ground track is
120 nmi east and 10 nmi south (course over ground 094.8°, 120.4 nmi). The
actual position is **41° 50.0′ N, 67° 18.7′ W**, about **10.0 nmi** south of the
DR position. (The 0.2′ longitude difference comes from the slightly different
mean latitude and is negligible.)

**(c)** A 2° heading error (actual course 092°) puts the ship
$120 \sin 2^\circ = 4.19$ nmi off track (and 0.07 nmi short along track). The
1-in-60 rule gives $2 \times 120/60 = 4.0$ nmi: close, because
$\sin 1^\circ \approx 1/57.3$. A log reading 5% high means the true distance is
$120/1.05 = 114.3$ nmi: **5.7 nmi** along-track error.

**(d)** The arc-chord difference is $s - 2R\sin(s/2R) \approx s^3/(24R^2)$. With
$s = 222.2$ km and $R = 6371$ km this is **11 m**, or 0.006 nmi: utterly
negligible. Ignoring the convergence of the meridians gives a change in
longitude of $120' = 2.000^\circ$ instead of $2.691^\circ$, an east-west error
of $0.691^\circ \times 60 \times \cos 42^\circ = $ **30.8 nmi**.

Ranking: meridian convergence ignored (30.8 nmi) > current (10.0) > log scale
factor (5.7) > heading (4.2) >> arc versus chord (0.006). The point students
should make: the largest number on the list is not an error any navigator
since the sixteenth century makes. Curvature effects are either negligible
(arc versus chord) or deterministic and exactly correctable in the model
(convergence of meridians, the $1/\cos L$ in the position update, the same
term that appears in @sec-ins-gnss-wgs84-geometry). They do not grow randomly.
Heading, speed, and current errors are unknown at the time of computation, so
they must be modeled as noise or estimated as states. That is a noise-modeling
question, and it is the one that dominates DR error once the model is right.

**(e)** Over 10 hours, the current is roughly constant: a bias whose position
error grows linearly with time. Compass deviation on a fixed heading is also a
constant bias (it changes with heading, so a ship that alternates courses sees
it partially average). The log calibration is a scale factor: error
proportional to distance run. Helmsman wander about the ordered course is
closer to zero-mean, short-correlation noise and averages out, with a
contribution growing like $\sqrt{t}$. Implication: a DR motion model with only
white process noise predicts $\sqrt{t}$ growth and will be badly overconfident
at long horizons, where the biases grow like $t$. The fix is to augment the
state with the biases (current set and drift, log scale factor) or model them
as Gauss-Markov processes, rather than inflating $\mathbf{Q}$, which makes the
filter too pessimistic at short horizons. The navigator's practice of computing
"set and drift" from the difference between successive fixes and DR positions
is exactly bias estimation.

## Exercise 2: Longitude is a clock

**(a)** One second of time is 15 arcseconds of longitude, so
$5.1 \times 15'' = 76.5'' = 1.275'$ of longitude. At 18° N an arcminute of
longitude is $\cos 18^\circ = 0.951$ nmi, so the error is about **1.2 nmi**.

**(b)** $30'$ of longitude is $30/15 = 2$ minutes of time $= 120$ s. Over six
weeks (42 days) the unpredicted rate error must stay below
$120/42 = $ **2.9 s/day** (on average; a rate error that changes sign could do
better). A known, constant rate is acceptable because it can be measured before
departure ("rating" the chronometer against astronomical time) and subtracted,
exactly like a deterministic IMU bias calibrated at turn-on. What cannot be
subtracted is the unpredictable part: rate changes with temperature, motion,
and age, the analog of in-run bias instability and temperature sensitivity
(@sec-ins-errors). Harrison's bimetallic temperature compensation is the
eighteenth-century version of temperature-compensated bias.

Growth: a clock-rate error $\delta r$ integrates once, so the time error and
the longitude error grow linearly, $\delta\lambda = 15''\,\delta r\,t$. An
accelerometer bias integrates twice, giving
$\delta p = \tfrac12 b t^2$ (and a gyro bias, through tilt, faster still). A
clock is a single integrator; an INS is a double (or triple) integrator, which
is why inertial errors explode so much faster.

**(c)** $c \times 5.1\ \text{s} = 1.53 \times 10^6$ km, about four times the
Earth-Moon distance. In 1 ns light travels **0.30 m**. A receiver's quartz
oscillator can drift by microseconds within seconds, which is hundreds of
meters of range per microsecond, so the receiver cannot trust its clock.
Instead it solves for its clock bias (and, in a filter, clock drift) as
unknowns alongside position, which is why a 3-D fix needs at least four
satellites (@sec-global-nav). H4 solved longitude by carrying an excellent
clock; GNSS solves position by carrying a cheap clock and estimating its error.

## Exercise 3: Write the contract

This is a design exercise; grade for completeness, internal consistency, and
whether each field is tied to a consumer. A strong answer for the Project 1
quadrotor looks like this.

| Field | Frame, units | Rate / latency budget | Consumer and use |
|---|---|---|---|
| `t_valid` | vehicle clock, s | every message | all; control predicts forward by `now - t_valid` |
| position | mat world frame $w$, m | 50 Hz (IMU rate), < 5 ms after the IMU sample | planner, position loop |
| velocity | resolved in $w$, m/s | same | position/velocity loop (D term) |
| attitude | quaternion, body to $w$, scalar first | same | attitude loop, gimbal |
| angular rate | body frame, rad/s, bias-corrected | same | rate loop (or the loop uses raw gyro) |
| covariance | 9x9 over position, velocity, attitude error | same | planner margins, mode logic |
| mode | enum | same | everyone |
| aids, time since last fix | tag count, s | same | mode logic, operator |
| NIS flag | bool + windowed statistic | same | mode logic, logging |

Frames must follow the project's own world frame $w$ and $z$-up body frame
(@sec-pnp defines both). Modes and transitions (thresholds are illustrative,
the reasoning is what matters): **nominal** (PnP updates arriving, windowed NIS
within its 95% bounds); **coasting** (no accepted PnP update for more than
about 0.2 s, i.e. several camera frames); **degraded** (windowed NIS above bound
for several consecutive windows: the filter is inconsistent, so stop trusting
its covariance and reject aids until reinitialized); **failed** (coasting for
more than about 2 to 3 s, or horizontal one-sigma above a threshold such as
0.5 m; with a MEMS IMU, double-integrated bias makes position meaningless within
seconds). Consumer actions: the controller holds attitude and altitude and
stops translating when coasting, and lands on failed; the planner refuses
aggressive maneuvers and inflates margins as the covariance grows; the operator
sees the mode at all times.

The survey-boat variant should mention NED versus ROS ENU/FLU conversion at the
interface (REP-103), GNSS antenna lever arm, heading source (dual-antenna GNSS,
magnetometer, or gyrocompass), RTK fixed/float as mode information, and radar
or camera piloting against charted features as an aid.

**(d)** KAL 007: the navigation system must publish which source is steering
the vehicle as part of its health output, and guidance must continuously
compare the commanded track with navigation's estimated track (cross-track
error) and raise an alarm when they diverge. Mode annunciation must be
persistent and unmissable to the operator. Responsibility is shared, and good
answers say so: navigation must report mode and health truthfully, guidance
must check that the mode it is in matches the mission, and the operator is the
last line of defense, not the first. A design that relies only on a human
noticing a mode light for five hours has already failed.

## Exercise 4: Breaking the loop

A helper that flies the loop with an arbitrary navigation function:

```python
def fly_with(nav_fn):
    """Closed loop with nav_fn(k, p, v, state) -> (p_hat, v_hat)."""
    p, v = np.zeros(n), np.zeros(n)
    p[0], v[0] = r[0], r_dot[0]
    p_hat, u, state = np.zeros(n), np.zeros(n), {}
    for k in range(n):
        p_hat[k], v_hat = nav_fn(k, p, v, state)
        u[k] = r_ddot[k] + kp * (r[k] - p_hat[k]) + kd * (r_dot[k] - v_hat)
        if k + 1 < n:
            a = u[k] + gust[k]
            p[k + 1] = p[k] + v[k] * dt + 0.5 * a * dt**2
            v[k + 1] = v[k] + a * dt
    return p, p_hat, u


keep = t >= 2.0
track = lambda p: 100 * np.sqrt(np.mean((p - r)[keep] ** 2))           # cm
chatter = lambda u: np.sqrt(np.mean(np.diff(u)[keep[1:]] ** 2))       # m/s^2
```

**(a)** Delay sweep with `nav_fn = lambda k, p, v, s: (p[max(k - lag, 0)], v[max(k - lag, 0)])`:

| delay [ms] | 0 | 50 | 100 | 150 | 160 | 170 | 180 | 190 | 200 |
|---|---|---|---|---|---|---|---|---|---|
| tracking RMSE [cm] | 1.7 | 2.7 | 4.6 | 6.6 | 7.0 | 7.5 | 9.8 | 1077 | diverged |

The simulated loop goes unstable between 180 and 190 ms (the discrete closed
loop's spectral radius crosses 1 at 18 steps; at 180 ms the growth is so slow
that 30 s is not enough to see it clearly). Theory: crossover satisfies
$\omega^2 = |k_p + j k_d \omega|$, i.e. $\omega^4 - k_d^2\omega^2 - k_p^2 = 0$,
so $\omega_c^2 = \tfrac12\left(k_d^2 + \sqrt{k_d^4 + 4k_p^2}\right) = 38.08$ and
$\omega_c = 6.17$ rad/s. The phase margin is
$\phi_m = \arctan(k_d\omega_c/k_p) = \arctan(2.16) = 65.2^\circ = 1.137$ rad,
and $\tau_{\max} = \phi_m/\omega_c = $ **0.184 s**. Agreement with simulation is
good; the small difference is the discretization (sample-and-hold adds roughly
half a sample of effective delay) and the semi-implicit integrator. At 150 ms,
$\omega_c\tau = 0.926$ rad $= 53.0^\circ$ of the $65.2^\circ$ margin is gone,
leaving about $12^\circ$: the number quoted in the chapter.

**(b)** First-order low-pass on the noisy feed,
$\hat{p} \leftarrow \hat{p} + \alpha(z - \hat{p})$ with
$\alpha = \Delta t/(T_f + \Delta t)$, applied to position and velocity:

| $T_f$ [s] | 0 | 0.02 | 0.05 | 0.10 | 0.20 |
|---|---|---|---|---|---|
| estimate RMSE seen by controller [cm] | 8.0 | 3.7 | 3.2 | 4.4 | 8.2 |
| tracking RMSE [cm] | 2.2 | 2.4 | 3.1 | 4.8 | 8.8 |
| chatter (RMS step in $u$) [m/s²] | 1.90 | 0.49 | 0.23 | 0.12 | 0.07 |

Chatter falls monotonically and tracking error rises monotonically: the filter
trades noise for lag, and lag is what the loop cannot tolerate. Note that the
estimate RMSE is minimized near $T_f = 0.05$ s, but the closed-loop optimum
depends on how you weigh chatter against tracking, and is not the same point.
The better answer, which the rest of the course builds, is a model-based filter
(for example a Kalman filter that uses the known command $u$ as its input), which
removes noise without adding the lag of a blind low-pass filter.

**(c)** Two reasonable blends; full credit for either if the velocity choice is
explained.

```python
def nav_blend(k, p, v, s, T_blend=0.5, include_rate=False):
    """Ramp each fix's position correction in over T_blend seconds."""
    i, m = divmod(k, 200)
    b = 0.05 * drift_sign[i]
    e, ve = b * m * dt, b
    nb = int(T_blend / dt)
    if i > 0 and m < nb:
        prev = 0.05 * drift_sign[i - 1] * 200 * dt     # error at the fix
        e += prev * (1 - m / nb)
        if include_rate:
            ve -= prev / T_blend                       # velocity consistent with ramp
    return p[k] + e, v[k] + ve
```

| version | estimate RMSE [cm] | tracking RMSE [cm] | chatter | largest step in $u$ [m/s²] |
|---|---|---|---|---|
| instantaneous jump (chapter) | 5.8 | 7.5 | 0.137 | 2.18 |
| blend, velocity includes ramp rate | 6.5 | 7.5 | 0.130 | 1.70 |
| blend, velocity excludes ramp rate | 6.5 | 8.4 | 0.031 | 0.58 |

Blending removes most of the kick (the remaining 0.58 m/s² step is the
velocity-bias change at each fix, $k_d \times 0.1$ m/s), at the cost of a
slightly larger estimate error because the correction arrives later. If the
published velocity includes the ramp rate, the ramp itself appears as a
velocity step and much of the benefit is lost. Control prefers the blended
stream; planning, mapping, and anything comparing the vehicle with a global map
prefer the instantly corrected, most accurate pose. That is why systems publish
both: a smooth odometry-frame state for control and a corrected map-frame pose
that may jump (the ROS `odom`/`map` split).

## Exercise 5: Separation, derived and broken

**(a)** Substituting $\hat{\mathbf{x}} = \mathbf{x} - \mathbf{e}$ into the plant,
$\dot{\mathbf{x}} = \mathbf{A}\mathbf{x} - \mathbf{B}\mathbf{K}_c(\mathbf{x} - \mathbf{e})
= (\mathbf{A} - \mathbf{B}\mathbf{K}_c)\mathbf{x} + \mathbf{B}\mathbf{K}_c\mathbf{e}$.
Subtracting the observer from the plant (the $\mathbf{B}\mathbf{u}$ terms
cancel because both see the same input),
$\dot{\mathbf{e}} = \mathbf{A}\mathbf{e} - \mathbf{L}\mathbf{H}\mathbf{e} = (\mathbf{A} - \mathbf{L}\mathbf{H})\mathbf{e}$.
Stacking gives @eq-gnc-separation. For a block upper-triangular matrix,
$\det(s\mathbf{I} - \mathbf{M}) = \det(s\mathbf{I} - \mathbf{A} + \mathbf{B}\mathbf{K}_c)\,\det(s\mathbf{I} - \mathbf{A} + \mathbf{L}\mathbf{H})$,
so the eigenvalues are the union of the two sets. Students should note the
cancellation of $\mathbf{B}\mathbf{u}$ is the crux: it requires the observer to
know the applied input exactly and at the same time.

**(b)** With $x(t)$ known perfectly but used $\tau$ late,
$\ddot{x}(t) = -k_p x(t-\tau) - k_d \dot{x}(t-\tau)$. Laplace transform:
$s^2 + (k_d s + k_p)e^{-s\tau} = 0$, a transcendental equation with infinitely
many roots. At $\tau = 0$ both roots are in the left half plane. As $\tau$
grows, roots cross the imaginary axis when $s = j\omega$ satisfies the
magnitude condition $\omega^2 = \sqrt{k_p^2 + k_d^2\omega^2}$ and the phase
condition $\omega\tau = \arctan(k_d\omega/k_p)$. For $k_p = 16$, $k_d = 5.6$:
$\omega_c = 6.17$ rad/s, $\arctan(2.16) = 1.137$ rad, so
$\tau^\ast = 0.184$ s. Both pieces are individually perfect (zero estimation
error, a well-damped controller), yet the combination is unstable beyond
184 ms. The violated assumption is that the estimate is available at the
instant the state exists: the control law uses $\mathbf{x}(t-\tau)$, the
closed loop becomes a delay-differential equation, and the block-triangular
structure is gone. The controller must be designed knowing the estimator's
timing (or the estimate must be predicted forward to the present using the
known inputs, which restores most of the margin).

**(c)** Any two, explained concretely:

1. The estimator gain depends on the covariance. An overconfident $\mathbf{P}$
   (from too-small $\mathbf{Q}$ or initial $\mathbf{P}$) gives small Kalman
   gains, so the filter discounts measurements and its estimate lags and drifts
   from the truth. The controller then tracks a wrong state, exactly like the
   delayed and jumpy feeds of the chapter's experiment, and when a large
   innovation finally gets through, the correction arrives as a jump.
2. Innovation gating uses $\mathbf{S} = \mathbf{H}\mathbf{P}\mathbf{H}^T + \mathbf{R}$.
   If $\mathbf{P}$ is too small the gates are too tight, valid measurements are
   rejected as outliers, the filter coasts, and it can diverge while reporting
   high confidence (@sec-kf-consistency).
3. Consumers other than the feedback law do use the covariance: planners size
   obstacle margins and chance constraints from it, mission logic decides
   whether to proceed (for example, begin a precision landing only if
   $\sigma < 0.2$ m), and integrity monitors compare it with alert limits.
   Overconfidence makes all of these decisions unsafe.
4. Certainty equivalence is optimal only in the linear-Gaussian case. For
   nonlinear systems, a wrong covariance means wrong linearization points and
   wrong sigma-point spreads, which corrupt the estimate itself.
