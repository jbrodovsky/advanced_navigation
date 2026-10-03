# Solutions: Curved-Earth INS and Loose GNSS Integration (`ins-gnss.qmd`)

Instructor-only. Exercise numbers match `## Exercises {#sec-ins-gnss-exercises}`.
Code snippets assume the chapter's cells have been run (in particular `strapdown_step`,
`error_state_model`, `error_between`, `inject`, `gnss_position_residual`, `eskf_update`,
and the simulated-drive variables). Numbers below were produced with those cells.

## Exercise 1: The attitude error equation in full

Identities needed:

- For a rotation matrix $\mathbf{C}$: $\mathbf{C}[\mathbf{a}]_\times\mathbf{C}^T = [\mathbf{C}\mathbf{a}]_\times$.
- $[\mathbf{a}]_\times[\mathbf{b}]_\times - [\mathbf{b}]_\times[\mathbf{a}]_\times = [\mathbf{a} \times \mathbf{b}]_\times$.
- Products of two small quantities are dropped.

Write $\boldsymbol{\Theta} = [\delta\boldsymbol{\theta}]_\times$, so $\mathbf{C} = (\mathbf{I} + \boldsymbol{\Theta})\hat{\mathbf{C}}$
(dropping the sub/superscripts $b$, $n$). The truth obeys
$\dot{\mathbf{C}} = \mathbf{C}\boldsymbol{\Omega}_{ib} - \boldsymbol{\Omega}_{in}\mathbf{C}$ and the estimate
$\dot{\hat{\mathbf{C}}} = \hat{\mathbf{C}}\hat{\boldsymbol{\Omega}}_{ib} - \hat{\boldsymbol{\Omega}}_{in}\hat{\mathbf{C}}$.
Differentiate the error definition:

$$
\dot{\mathbf{C}} = \dot{\boldsymbol{\Theta}}\hat{\mathbf{C}} + (\mathbf{I} + \boldsymbol{\Theta})\dot{\hat{\mathbf{C}}}.
$$

Equate with the truth dynamics and write $\boldsymbol{\Omega}_{ib} = \hat{\boldsymbol{\Omega}}_{ib} + \delta\boldsymbol{\Omega}_{ib}$,
$\boldsymbol{\Omega}_{in} = \hat{\boldsymbol{\Omega}}_{in} + \delta\boldsymbol{\Omega}_{in}$:

$$
\dot{\boldsymbol{\Theta}}\hat{\mathbf{C}}
= (\mathbf{I} + \boldsymbol{\Theta})\hat{\mathbf{C}}\,\delta\boldsymbol{\Omega}_{ib}
- \boldsymbol{\Omega}_{in}(\mathbf{I} + \boldsymbol{\Theta})\hat{\mathbf{C}}
+ (\mathbf{I} + \boldsymbol{\Theta})\hat{\boldsymbol{\Omega}}_{in}\hat{\mathbf{C}}
\approx \hat{\mathbf{C}}\,\delta\boldsymbol{\Omega}_{ib} - \delta\boldsymbol{\Omega}_{in}\hat{\mathbf{C}}
- (\hat{\boldsymbol{\Omega}}_{in}\boldsymbol{\Theta} - \boldsymbol{\Theta}\hat{\boldsymbol{\Omega}}_{in})\hat{\mathbf{C}}.
$$

Multiply on the right by $\hat{\mathbf{C}}^T$ and use the two identities:

$$
[\delta\dot{\boldsymbol{\theta}}]_\times = [\hat{\mathbf{C}}\delta\boldsymbol{\omega}_{ib}^b]_\times - [\delta\boldsymbol{\omega}_{in}^n]_\times - [\boldsymbol{\omega}_{in}^n \times \delta\boldsymbol{\theta}]_\times,
$$

which is the stated result. The chapter's finite-difference cell confirms the sign of the
$-\boldsymbol{\omega}_{in}^n\times\delta\boldsymbol{\theta}$ term: the attitude-attitude mismatch
it prints is exactly $-[\boldsymbol{\omega}_{in}^n]_\times$.

Numbers at $L = 42.27°$, stationary ($\boldsymbol{\omega}_{in}^n = \boldsymbol{\omega}_{ie}^n$),
1° tilt. The relative rate of change is $\|\boldsymbol{\omega}_{ie}^n \times \delta\boldsymbol{\theta}\|/\|\delta\boldsymbol{\theta}\|$:

- tilt about north, $\delta\boldsymbol{\theta} \parallel [1, 0, 0]$: $\omega_{ie}\sin L = 4.9 \times 10^{-5}$ s⁻¹, so 1% takes about 204 s;
- tilt about east, $\delta\boldsymbol{\theta} \parallel [0, 1, 0]$: $\omega_{ie} = 7.3 \times 10^{-5}$ s⁻¹, so 1% takes about 137 s.

(The neglected term mostly rotates the error vector rather than growing it.) A 200°/h gyro
bias ($9.7 \times 10^{-4}$ rad/s) adds 1% of 1° ($1.75 \times 10^{-4}$ rad) in 0.18 s, roughly
a thousand times faster. That is the quantitative case for dropping the term in a MEMS
filter, and also why it must come back for a navigation-grade gyro: at 0.01°/h the bias
needs an hour to add 1% to a 1° error, while the Earth-rate coupling does it in about two
minutes, so for that sensor the neglected term is the larger one.

## Exercise 2: Mount Washington gravity

Ellipsoidal height: $h = H + N_g = 1917 + (-28) \approx 1889$ m.

With the chapter's `gravity`:

| Quantity | Value |
|---|---|
| $g(44.27°, 1889\ \text{m})$ | 9.799711 m/s² |
| $g(44.27°, 0)$ | 9.805537 m/s² |
| summit minus ellipsoid | −5.83 mm/s² (about 0.59 mg) |
| $9.80665 - g_\text{summit}$ | 6.94 mm/s² = 0.71 mg |

An INS using 9.80665 m/s² on the summit adds 6.94 mm/s² too much downward acceleration, so
it "sinks" by $\tfrac12(6.94\times10^{-3})(60)^2 \approx 12.5$ m in a minute. A 2 milli-g
vertical bias gives $\tfrac12(0.0196)(60)^2 \approx 35.3$ m, so the gravity-model error is
about a third of a typical MEMS bias: not negligible, and in an aided filter it would be
absorbed into the vertical accelerometer bias estimate (which then changes when the vehicle
changes altitude, a trap worth pointing out to students).

Using $H = 1917$ m instead of $h = 1889$ m changes $g$ by $8.6 \times 10^{-5}$ m/s²
(8.8 micro-g), about 16 cm of vertical error after a minute. Small for MEMS, but the
position error from feeding orthometric height to the geodetic conversion (28 m vertically)
is a much bigger problem than the gravity error.

```python
lat_w = np.radians(44.27)
h_ell = 1917.0 - 28.0
g_top, g_ell = gravity(lat_w, h_ell), gravity(lat_w, 0.0)
dg = 9.80665 - g_top
print(g_top, g_ell, dg, 0.5 * dg * 60**2, 0.5 * 2e-3 * 9.80665 * 60**2)
print(gravity(lat_w, 1917.0) - g_top)
```

## Exercise 3: Course over ground

Inside the GNSS block of the simulated-drive loop, after the position update:

```python
s_true = np.hypot(*truth_states[k + 1].v[:2])
if s_true > 3.0:
    sig_s, sig_chi = 0.1, 0.1 / s_true
    v_t = truth_states[k + 1].v
    chi = np.arctan2(v_t[1], v_t[0]) + sig_chi * rng.standard_normal()
    s = s_true + sig_s * rng.standard_normal()
    z_v = s * np.array([np.cos(chi), np.sin(chi)])
    J = np.array([[np.cos(chi), -s * np.sin(chi)], [np.sin(chi), s * np.cos(chi)]])
    R_v = J @ np.diag([sig_s**2, sig_chi**2]) @ J.T
    H_v = np.zeros((2, 15)); H_v[0, 3] = H_v[1, 4] = 1.0
    x_est, P, nis_v = eskf_update(x_est, P, z_v - x_est.v[:2], H_v, R_v)
```

Representative results (same IMU data as the chapter, independent GNSS noise; heading
sigma in degrees):

| t [s] | position only | with velocity |
|---|---|---|
| 12 (accelerating) | 9.7 | 6.7 |
| 20 (end of acceleration) | 5.5 | 4.0 |
| 30 (cruise) | 2.6 | 1.4 |
| 50 (end of cruise) | 3.6 | 3.1 |
| 60 (after turn) | 1.5 | 0.7 |
| 120 | 1.1 | 0.6 |

Roll/pitch sigma at 60 s drops from 0.44° to 0.16°, and the forward accelerometer-bias
sigma from 5.7 to 2.0 mg. The velocity NIS averages about 2.3 for $m = 2$, the position NIS
stays near 2.7: consistent.

The velocity update helps most when the vehicle accelerates and turns, because that is
when tilt, heading, and accelerometer bias produce velocity errors, and a velocity
measurement sees them immediately instead of after integration into position. It does
**not** make heading observable during the straight, constant-speed cruise: the heading
sigma still grows from 1.4° to 3.1° between 30 and 50 s (the vertical gyro bias
integrating). In this error model, heading enters $\delta\dot{\mathbf{v}}$ only through
$-[\hat{\mathbf{f}}^n]_\times\delta\boldsymbol{\theta}$, which has no yaw term when
$\hat{\mathbf{f}}^n$ is vertical; a velocity measurement cannot see what does not affect
velocity.

What would make course over ground informative about heading: the car's non-holonomic
constraint. A car does not slide sideways or jump, so its body-frame velocity has
(approximately) zero lateral and vertical components. As a pseudo-measurement
$\mathbf{z} = [0, 0]^T$ of rows 2 and 3 of $\mathbf{v}^b = \mathbf{C}_n^b\mathbf{v}^n$, with
$\mathbf{C}_b^n = (\mathbf{I} + [\delta\boldsymbol{\theta}]_\times)\hat{\mathbf{C}}_b^n$:

$$
\mathbf{v}^b \approx \hat{\mathbf{C}}_n^b\hat{\mathbf{v}}^n + \hat{\mathbf{C}}_n^b\delta\mathbf{v}^n + \hat{\mathbf{C}}_n^b[\hat{\mathbf{v}}^n]_\times\delta\boldsymbol{\theta},
$$

so $\mathbf{H}_\text{NHC}$ is rows 2-3 of $[\mathbf{0},\ \hat{\mathbf{C}}_n^b,\ \hat{\mathbf{C}}_n^b[\hat{\mathbf{v}}^n]_\times,\ \mathbf{0},\ \mathbf{0}]$,
with noise of a few tenths of a meter per second for skidding and suspension motion. The
attitude block couples yaw to the velocity direction whenever the car moves, which is what
makes course over ground (or any velocity) informative about heading. (For a phone in a car,
this also requires the phone-to-car mounting rotation; see Exercise 6.)

## Exercise 4: The lever arm with the wrong sign

The filter's lever arm differs from the truth by $2\boldsymbol{\ell}^b = [0.8, 0, -2.4]$ m:
the predicted antenna is 0.8 m behind and 2.4 m below where it really is.

**1.5 m / 3 m GNSS noise.** The NIS does not notice: mean 2.74, the same as with the correct
lever arm, with no segment of the drive leaving the 95% band in a meaningful way (the turn
raises the segment mean only to about 3.3). The error goes into the position estimate: the
INS reports a point 2.4 m above the IMU (height error about +2.4 m against a height sigma
under 1 m by the end) and 0.8 m along-track. A constant body-frame offset is indistinguishable
from the IMU simply being elsewhere, so without truth (or a surveyed lever arm) there is
nothing to see. Roll and pitch are essentially unaffected; meter-level GNSS cannot resolve a
rotating 0.8 m horizontal offset.

**RTK fixes (2 cm / 4 cm).** The vertical part is still silently absorbed (height error
+2.4 m against a 2 cm sigma: a 100-sigma error the filter is perfectly happy with, because a
constant vertical offset is unobservable from position fixes). The horizontal 0.8 m offset,
however, rotates with the vehicle during the turn and cannot be explained by a position
error. Segment-mean NIS: parked 2.1, acceleration 3.0, cruise 2.6, **turn 10.2**, after the
turn 5.7. The filter tries to explain the rotating offset with the states it has: after the
turn the heading error is about −0.5° (3 sigma), roll −0.1° (5 sigma), and the forward
accelerometer bias about −2 mg (8 sigma). Lesson: lever-arm errors hide in position until
the vehicle rotates, and then they corrupt attitude and bias estimates.

**Correct lever arm, RTK, attitude block of H zeroed.** Results are close to the baseline
(mean NIS 2.79 vs 2.73), but the heading errors during the acceleration exceed their bounds:
about +1.7° against a 1.1° sigma at 12 s and −1.2° against 0.65° at 20 s (roughly 1.5-2
sigma, where the baseline stays under 1 sigma). The covariance is overconfident because the
filter ignores how attitude moves the antenna, but the effect is small for a 1.3 m lever arm
and transient, so NIS barely moves. With a 10 m lever arm (ship's mast, aircraft tail
antenna) it would be obvious.

```python
# Driver used for the numbers above: copy the chapter loop into a function with
# arguments lever_filter, zero_att_block, sigma_gnss, and replace inside the update
#     nu, H = gnss_position_residual(..., lever_filter)
#     if zero_att_block: H[:, 6:9] = 0.0
# while the simulated GNSS still uses the true lever arm in antenna_position.
```

## Exercise 5: Why the order of the updates matters

```python
def strapdown_step_stale(x, f_meas, w_meas, tau):
    """Velocity first, resolving specific force with C_b^n(-) only."""
    f_b, w_ib = f_meas - x.b_a, w_meas - x.b_g
    R_M, R_N = radii(x.lat)
    w_ie, w_en = earth_rate_ned(x.lat), transport_rate_ned(x.lat, x.h, x.v)
    g_n = np.array([0.0, 0.0, gravity(x.lat, x.h)])
    v = x.v + (quat_to_dcm(x.q) @ f_b + g_n - np.cross(w_en + 2.0 * w_ie, x.v)) * tau
    q = quat_multiply(quat_from_rotvec(-(w_ie + w_en) * tau),
                      quat_multiply(x.q, quat_from_rotvec(w_ib * tau)))
    q = q / np.linalg.norm(q)
    h = x.h - 0.5 * tau * (x.v[2] + v[2])
    lat = x.lat + 0.5 * tau * (x.v[0] / (R_M + x.h) + v[0] / (R_M + h))
    lon = x.lon + 0.5 * tau * (x.v[1] / ((R_N + x.h) * np.cos(x.lat))
                               + v[1] / ((radii(lat)[1] + h) * np.cos(lat)))
    return replace(x, lat=lat, lon=lon, h=h, v=v, q=q)
```

Generate the IMU data for a steady 20°/s turn at 15 m/s with the truth-generation loop of the
simulated drive (constant yaw rate, centripetal acceleration $v\omega \approx 5.2$ m/s²), feed
the identical data to both functions, and compare with `error_between`:

| IMU rate | horizontal difference after 60 s |
|---|---|
| 50 Hz | 3.0 m |
| 200 Hz | 0.75 m |

At 50 Hz the difference grows roughly linearly in time (0.9, 1.7, 2.4, 3.0 m at 15, 30, 45,
60 s), and the speeds differ by a few centimeters per second.

Mechanism: with $\mathbf{C}_b^n(-)$ alone, the specific force is resolved along axes that
are, on average over the step, half a step old, so in a turn its direction lags by
$\omega\tau/2$ (0.2° at 50 Hz). The 5.2 m/s² centripetal force then has a spurious
along-track component of about $a\,\omega\tau/2 \approx 0.018$ m/s², which perturbs the
speed and the phase of the velocity vector, and the error accumulates in position. The
averaged-attitude form removes the first-order term. The error is first order in $\tau$:
four times the rate, a quarter of the error. Moral: when you reorder the steps, you change
the integrator, not just the code.

## Exercise 6: Aligning a parked phone

**Leveling.** With @eq-ins-gnss-leveling,
$\phi = \operatorname{atan2}(0.3408, 9.7600) = 2.00°$ and
$\theta = \operatorname{atan2}(0.8544, \sqrt{0.3408^2 + 9.7600^2}) = 5.00°$;
$\|\tilde{\mathbf{f}}\| = 9.8033$ m/s², consistent with local gravity at WPI.

**Tilt error.** $b_a/g = 0.020 \times 9.80665/9.8033$ rad $\approx 1.15°$ per horizontal axis.

**Gyrocompassing.** The gyro reading has magnitude about 532°/h; the horizontal Earth rate
the gyrocompass needs to resolve is $\omega_{ie}\cos L \approx 11.1$°/h, about 50 times
smaller. The reading is bias, not Earth rate. Running `gyrocompass` on it returns
about 29°, which is simply the direction of the bias's horizontal projection; it has nothing
to do with north.

**Phone heading from course over ground.** Course gives the car's heading; the phone's
heading also requires the **mounting yaw** (the phone's yaw relative to the car's forward
axis), plus the knowledge that the car is not sideslipping. During the first acceleration
from rest, the car accelerates along its own forward axis. Subtract the stationary average
from the accelerometer readings during the acceleration (or resolve them in the leveled
frame): the horizontal part of the difference points along the car's forward axis in phone
coordinates, and $\psi_\text{mount} = \operatorname{atan2}(\Delta f_y^h, \Delta f_x^h)$.

Accuracy: the constant accelerometer bias cancels in the difference with the stationary
reading (equivalently, the leveling tilt error already cancels the horizontal bias), so the
20 mg bias costs little. If one naively used raw readings with the full bias acting
sideways, the error would be about $\arctan(0.196/1.5) \approx 7.4°$, worse than the
magnetometer. What actually limits the differenced estimate is anything lateral during the
averaging window: steering, road camber, vibration, the car pitching under acceleration
changing the gravity projection (that one is along the forward axis, so it mostly does not
affect yaw). A lateral disturbance of 0.05 m/s² against 1.5 m/s² of forward acceleration is
about 2°, and averaging over the whole acceleration helps with the noise but not with a
steady lateral slope. In practice the mounting yaw is refined continuously by the filter,
often with the non-holonomic constraint from Exercise 3.
