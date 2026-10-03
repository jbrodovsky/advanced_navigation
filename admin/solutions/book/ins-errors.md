# Solutions: INS Error Models and IMU Characterization (`ins-errors.qmd`)

Instructor-only. Exercise numbers match `## Exercises {#sec-ins-errors-exercises}`.
Code snippets assume the chapter's cells have been run (in particular `drift_terms`,
`overlapping_allan_deviation`, the unit-conversion helpers, `error_state_model`,
`van_loan`, `stacked_observability`, and `report`). Numbers below were produced with them.

## Exercise 1: Noise growth laws

Velocity random walk. With white accelerometer noise $w(t)$ of spectral density $q = N_a^2$,

$$
p(t) = \int_0^t\!\int_0^s w(r)\,dr\,ds = \int_0^t (t - r)\,w(r)\,dr,
$$

by swapping the order of integration. For white noise,
$\operatorname{Var}\left[\int_0^t k(r)w(r)\,dr\right] = q\int_0^t k(r)^2\,dr$, so

$$
\sigma_p^2 = N_a^2\int_0^t (t - r)^2\,dr = \frac{N_a^2 t^3}{3}.
$$

Angle random walk. The tilt is $\delta\theta(t) = \int_0^t w_g$, the acceleration error
$g_0\delta\theta$, and position is two more integrals:

$$
p(t) = g_0\int_0^t \frac{(t - r)^2}{2}\,w_g(r)\,dr,
\qquad
\sigma_p^2 = g_0^2 N_g^2\int_0^t \frac{(t - r)^4}{4}\,dr = \frac{g_0^2 N_g^2 t^5}{20}.
$$

(The kernel of an $n$-fold integral is $(t - r)^{n-1}/(n-1)!$.)

Crossovers for @tbl-ins-errors-datasheet ($N_a = 1.0\times10^{-3}$ m/s/√s,
$N_g = 8.73\times10^{-5}$ rad/√s):

- ARW overtakes VRW when $g_0 N_g t^{5/2}/\sqrt{20} = N_a t^{3/2}/\sqrt{3}$, i.e.
  $t = \sqrt{20}N_a/(\sqrt{3}g_0N_g) \approx 3.0$ s.
- Calibrated gyro bias (10°/h) overtakes calibrated accelerometer bias (1 mg) when
  $g_0 b_g t^3/6 = b_a t^2/2$, i.e. $t = 3b_a/(g_0 b_g) \approx 62$ s.

The chapter's Monte Carlo already checks both formulas (ratios within a few percent of 1
with 400 runs, which is the sampling scatter of a standard deviation estimate).

## Exercise 2: Read a bias instability

```python
from scipy.signal import lfilter

rng = np.random.default_rng(7)
fs, hours = 100.0, 4.0
dt = 1.0 / fs
n = round(hours * 3600 * fs)


def flicker(sigma_each, time_constants=(1.0, 10.0, 100.0, 1000.0, 10000.0)):
    """Sum of equal-variance first-order Gauss-Markov processes (approximate 1/f noise)."""
    out = np.zeros(n)
    for T in time_constants:
        phi = np.exp(-dt / T)
        w = sigma_each * np.sqrt(1 - phi**2) * rng.standard_normal(n)
        out += lfilter([1.0], [1.0, -phi], w)       # x_k = phi x_{k-1} + w_k
    return out


K_small = np.radians(3.0 / 3600.0) / 60.0
rate = (N_true / np.sqrt(dt) * rng.standard_normal(n)
        + np.cumsum(K_small * np.sqrt(dt) * rng.standard_normal(n))
        + flicker(np.radians(4.3 / 3600.0)))
taus2, adev2 = overlapping_allan_deviation(rate, dt)
```

Equal variances make each Gauss-Markov process contribute a Lorentzian whose level at its
corner frequency scales as $1/f$, so together they mimic flicker noise between about
$1/(2\pi\cdot 10^4)$ and $1/(2\pi)$ Hz. A common standard deviation of about 4.3°/h puts the
floor near 3°/h; empirically the flicker floor is roughly $0.77\,\sigma_\text{each}$ for this
spacing, before the white and random-walk terms are added.

Representative readings (seed 7):

- Allan curve: 18.1°/h at 1 s, 6.8 at 9 s, 4.6 at 31 s, 3.6 at 105 s, 3.0 at 290 s,
  2.9 at about 800-960 s. It flattens only slowly.
- $B = \sigma_\text{min}/0.664 \approx 2.89/0.664 \approx 4.4$°/h, against the roughly 5°/h
  intended.
- $N$ from the −1/2 line: 0.307°/√h (injected 0.3). Reliable.
- $K$: not readable. The +1/2 region has been pushed past the end of a four-hour record by
  the flicker floor, and a fit to the last two points returns about 9°/h/√h, three times the
  injected value.

Least reliable: $K$, then $B$. $N$ comes from the short-τ end where there are thousands of
independent clusters and nothing else contributes. $B$ depends on where you decide the
"flat" bottom is and on the record being long enough to see it; a curve that is still
falling gives a low reading of the floor but a too-early minimum. $K$ rests on the few
clusters at the right end, and with flicker present it may never emerge at all. That is
also why data-sheet RRW values are often omitted or quoted loosely.

## Exercise 3: Stop-and-go observability

Segments: for each 20 s cycle, 3 s at +1 m/s² along track, 14 s cruise, 3 s at −1 m/s²,
heading fixed at 0.3 rad; six cycles is two minutes. Then append ten 1-s segments of a
90° turn at 9°/s with 0.5 m/s² centripetal acceleration.

```python
def stop_and_go(cycles=6, heading=0.3, turn=False):
    c, s = np.cos(heading), np.sin(heading)
    segs = []
    for _ in range(cycles):
        segs += [(heading, [c, s, -G0], 3.0), (heading, [0.0, 0.0, -G0], 14.0),
                 (heading, [-c, -s, -G0], 3.0)]
    if turn:
        for k in range(10):
            psi = heading + np.radians(9.0) * (k + 0.5)
            right = np.array([-np.sin(psi), np.cos(psi), 0.0])
            segs.append((psi, list(0.5 * right + np.array([0.0, 0.0, -G0])), 1.0))
    return segs

null = report("stop-and-go straight", stacked_observability(stop_and_go()))
report("stop-and-go + 90 deg turn", stacked_observability(stop_and_go(turn=True)))
```

Results: stop-and-go straight gives rank 14 of 15 with yaw observable; the one remaining
null direction is about $0.10\,\delta\theta_\parallel + 0.99\,\delta b_{a,y}$, a roll tilt
(about the along-track axis) traded against the lateral (body $y$) accelerometer bias. Adding
the turn gives rank 15.

Explanation from $-[\hat{\mathbf{f}}^n]_\times\delta\boldsymbol{\theta} - \hat{\mathbf{C}}_b^n\delta\mathbf{b}_a$:
the changing along-track specific force $f_\parallel$ puts yaw into the cross-track velocity
error ($f_\parallel\delta\theta_D$) with a time variation no constant bias can mimic, so yaw
and the along-track accelerometer bias separate. But the lateral specific force is always
zero and the body never rotates, so a roll tilt $\delta\theta_\parallel$ produces a constant
lateral acceleration error $g\,\delta\theta_\parallel$ that is indistinguishable from a
lateral accelerometer bias of the same size. The turn rotates the body-fixed bias relative to
the navigation frame (and adds lateral specific force), which breaks the tie.

## Exercise 4: A tactical data sheet

Conversions: $N_g = 0.05\pi/180/60 = 1.45\times10^{-5}$ rad/√s,
$N_a = 0.03/60 = 5.0\times10^{-4}$ m/s/√s; with $T_c = 100$ s,
$\sqrt{q_{bg}} = B_g/\sqrt{T_c} = 2.4\times10^{-7}$ rad/s/√s and
$\sqrt{q_{ba}} = B_a/\sqrt{T_c} = 4.9\times10^{-5}$ m/s²/√s.

```python
N_g4, N_a4 = arw_to_rad_per_sqrt_s(0.05), vrw_to_m_per_s_per_sqrt_s(0.03)
q_bg4 = deg_per_hour_to_rad_per_second(0.5) ** 2 / T_c
q_ba4 = milli_g_to_mps2(0.05) ** 2 / T_c
Q_c4 = np.diag(np.concatenate([[N_a4**2] * 3, [N_g4**2] * 3, [q_ba4] * 3, [q_bg4] * 3]))
Q_d4 = G @ Q_c4 @ G.T / 200.0                       # first order at 200 Hz
P0 = np.zeros((15, 15))
P0[9:12, 9:12] = milli_g_to_mps2(0.3) ** 2 * np.eye(3)
P0[12:15, 12:15] = deg_per_hour_to_rad_per_second(1.0) ** 2 * np.eye(3)
Phi_1s, Q_1s = van_loan(F, G, Q_c4, 1.0)
P = P0.copy()
for _ in range(60):
    P = Phi_1s @ P @ Phi_1s.T + Q_1s
print(np.sqrt(P[0, 0]))
```

Per-step standard deviations at 200 Hz: velocity $3.5\times10^{-5}$ m/s, angle
$1.0\times10^{-6}$ rad, accelerometer bias $3.5\times10^{-6}$ m/s², gyro bias
$1.7\times10^{-8}$ rad/s.

North position sigma after 60 s: about 5.7 m, against about 640 m for the smartphone
example. Contributions (each propagated alone): accelerometer turn-on bias 5.3 m, gyro
turn-on bias 1.7 m, all white noise and bias instability 1.0 m.

What to improve first: the accelerometer turn-on bias. At one minute, $t < 3b_a/(g_0b_g)$
($\approx 185$ s for 0.3 mg and 1°/h), so the quadratic accelerometer term still dominates
the cubic gyro term. For longer outages the gyro takes over, and for heading (and
gyrocompassing) the gyro bias is what matters. A good answer notes that the priority depends
on the outage length the application must survive.

## Exercise 5: Urban canyon outage

At $t = 30$ s with $b_a = 1$ mg and $b_g = 10$°/h on the same axis:

- accelerometer: $\tfrac12(9.81\times10^{-3})(30)^2 \approx 4.4$ m;
- gyro via tilt: $g_0 b_g t^3/6 \approx 2.1$ m;
- worst case (aligned) 6.6 m; root-sum-square (independent axes or signs) 4.9 m;
- white-noise terms for an industrial part are tens of centimeters at 30 s and do not change
  the answer.

So the IMU alone is marginal-to-insufficient for a 5 m requirement. A better gyro barely
helps: even a perfect gyro leaves the 4.4 m accelerometer term (2°/h gives 4.8 m aligned,
1°/h 4.6 m). At 30 s the accelerometer dominates, because the crossover
$3b_a/(g_0b_g)$ is about 62 s. The better levers are:

- the filter's own bias estimate before the outage: after a few minutes of GNSS-aided
  driving with turns, the residual accelerometer bias is a few tenths of a milli-g (0.3 mg
  gives 3.5 m total with the same gyro), which is why the outage behavior depends on the
  trajectory before it;
- wheel odometry or the vehicle's speed signal, which bounds along-track error directly and
  with the non-holonomic constraint bounds cross-track drift too, making the outage almost
  irrelevant for a van;
- map matching to the street network, which solves exactly the "which side of the street"
  question.
