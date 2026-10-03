# Solutions: Measurement Models (`sec-measurement-models-exercises`)

Instructor-only. Code assumes the functions and variables defined in the
chapter's cells (`range_bearing`, `range_bearing_jacobian`, `beam_components`,
`beam_likelihood`, `weights`, `likelihood_field_log`, `scan`, `beam_angles`,
`true_pose`, `lidar_max`, `pressure_altitude`). All numbers below were produced
by running that code.

## 1. Range-bearing Jacobian

With $\Delta x = m_x - x$, $\Delta y = m_y - y$, $q = \Delta x^2 + \Delta y^2$,
$r = \sqrt q$:

- Range: $\partial r/\partial x = \frac{1}{2\sqrt q}\cdot 2\Delta x \cdot (-1) = -\Delta x / r$,
  similarly $\partial r/\partial y = -\Delta y/r$, and $\partial r/\partial\theta = 0$.
- Bearing: $\beta = \operatorname{atan2}(\Delta y, \Delta x) - \theta$, and
  $d\,\operatorname{atan2}(Y, X) = (X\,dY - Y\,dX)/(X^2 + Y^2)$. With
  $dX = -dx$, $dY = -dy$: $\partial\beta/\partial x = \Delta y/q$,
  $\partial\beta/\partial y = -\Delta x/q$, $\partial\beta/\partial\theta = -1$.

Landmark Jacobian ($dX = dm_x$, $dY = dm_y$):

$$
\frac{\partial h}{\partial \mathbf{m}} =
\begin{bmatrix} \Delta x/r & \Delta y/r \\ -\Delta y/q & \Delta x/q \end{bmatrix},
$$

the negative of the first two columns of $\mathbf{H}$, as it must be, since moving
the landmark and moving the robot the opposite way are equivalent.

At $(0, 0, 0)$:

$$
\mathbf{H}_{(2,0)} = \begin{bmatrix} -1 & 0 & 0 \\ 0 & -0.5 & -1 \end{bmatrix},
\qquad
\mathbf{H}_{(200,0)} = \begin{bmatrix} -1 & 0 & 0 \\ 0 & -0.005 & -1 \end{bmatrix}.
$$

The lateral ($y$) entry of the bearing row is 0.5 rad/m for the near landmark and
0.005 rad/m for the far one. With the same bearing noise, the near landmark
constrains lateral position 100 times more strongly; the distant landmark's
bearing essentially measures heading only (the $-1$). Range constrains only the
along-line-of-sight position in both cases.

As $r \to 0$ the bearing row grows like $1/r$ and is undefined at $r = 0$. The
linearization is accurate only for state errors small compared with $r$, so near a
landmark a modest position uncertainty makes the EKF's linear model badly wrong,
leading to inconsistent (overconfident or biased) updates. A particle filter never
linearizes: it evaluates $p(\mathbf{z} \mid \mathbf{x}^{[i]})$ at each particle,
and only the single point $r = 0$ (bearing undefined) needs care.

## 2. The point-mass problem

```python
def two_readings(max_bin, scale=1.0, point_mass=False):
    """Likelihoods of a correct hit (6 m) and a max reading (10 m), lengths in units of `scale` m."""
    z_max_s, z_star_s = 10.0 / scale, 6.0 / scale
    prm = {"sigma_hit": 0.2 / scale, "lambda_short": 0.3 * scale, "max_bin": max_bin / scale}
    hit = beam_likelihood(z_star_s, z_star_s, z_max_s, prm, weights)[0]
    if point_mass:                       # bare PMF value 1 for the max term
        w = dict(weights); w_max = w.pop("max")
        comps = beam_components(z_max_s, z_star_s, z_max_s, **prm)
        mx = sum(w[k] * comps[k][0] for k in w) + w_max * 1.0
    else:
        mx = beam_likelihood(z_max_s, z_star_s, z_max_s, prm, weights)[0]
    return hit, mx
```

| Treatment | Units | Hit likelihood | Max likelihood | Max / hit |
|---|---|---:|---:|---:|
| Bin, $\Delta = 0.1$ m | m | 1.415 /m | 0.5 /m | 0.353 |
| Bin, $\Delta = 0.01$ m | m | 1.415 /m | 5.0 /m | 3.53 |
| Point mass = 1 | m | 1.415 /m | 0.05 | 0.0353 |
| Bin, $\Delta = 0.1$ m | mm | 0.001415 /mm | 0.0005 /mm | 0.353 |
| Point mass = 1 | mm | 0.001415 /mm | 0.05 | 35.3 |

The ratio of the two likelihoods says how much a max reading should count
relative to a hit, which is physics, so it must not depend on whether ranges are
stored in meters or millimeters. The bin treatment passes that test (both terms
are densities, so the units cancel). The bare point mass fails by a factor of
1000. The status-flag treatment, which handles max readings as a discrete event
with probability $w_{\max}$ and continuous readings with density
$(1 - w_{\max})\,p_{\text{cont}}$, is also unit-consistent, because all particles
see the same set of max-flagged beams and the unit factor cancels when weights are
normalized.

With the bin, $w_{\max}/\Delta$ is the effective height of the max-range term, so
$\Delta$ and $w_{\max}$ are not separately identifiable; pick $\Delta$ as the
sensor's range resolution (or the width of the "no return" value in the driver's
output, often one quantization step) and then fit $w_{\max}$ from the observed
fraction of max readings in logged data, as Probabilistic Robotics does for all
the mixture weights.

## 3. Likelihood-field localization

```python
def grid_search(scan_ranges, angles, center, half_width=0.6, step=0.03,
                heading_half=np.deg2rad(15.0), heading_step=np.deg2rad(1.0), sigma_hit=0.1):
    best = (-np.inf, None)
    for dth in np.arange(-heading_half, heading_half + 1e-9, heading_step):
        for dx in np.arange(-half_width, half_width + 1e-9, step):
            for dy in np.arange(-half_width, half_width + 1e-9, step):
                pose = center + np.array([dx, dy, dth])
                ll = likelihood_field_log(pose, angles, scan_ranges, lidar_max, sigma_hit=sigma_hit)
                if ll > best[0]:
                    best = (ll, pose)
    return best

ll, pose = grid_search(scan, beam_angles, true_pose + np.array([0.2, -0.1, np.deg2rad(5)]))
print(pose, true_pose)     # [3.49 3.99 0.349] vs [3.5 4.0 0.349]


def peak_drop(angles, ranges, sigma_hit, h=0.15):
    """Mean log-likelihood drop for +/- h steps in x and y from the truth.
    (The grid lookup is piecewise constant at the 5 cm cell scale, so use a wide step.)"""
    f0 = likelihood_field_log(true_pose, angles, ranges, lidar_max, sigma_hit=sigma_hit)
    steps = [np.array([s * h, 0, 0]) for s in (-1, 1)] + [np.array([0, s * h, 0]) for s in (-1, 1)]
    return f0 - np.mean([likelihood_field_log(true_pose + d, angles, ranges, lidar_max,
                                              sigma_hit=sigma_hit) for d in steps])

for s in [0.03, 0.1, 0.5]:
    print(s, peak_drop(beam_angles, scan, s), peak_drop(beam_angles[::4], scan[::4], s))
```

The search recovers the pose to within one grid step (1 cm, 0 degrees here).

**(a)** Log-likelihood drop for a 15 cm step:

| $\sigma_{\text{hit}}$ | all 60 beams | every 4th beam |
|---:|---:|---:|
| 0.03 m | 86.9 | 21.3 |
| 0.1 m | 18.7 | 4.6 |
| 0.5 m | 0.7 | 0.2 |

Small $\sigma_{\text{hit}}$ gives a sharp, narrow peak with a small basin of
attraction (a search or particle filter started 20 cm away may see almost no
gradient, and the surface becomes jagged from the 5 cm grid). Large
$\sigma_{\text{hit}}$ gives a broad, forgiving peak that localizes poorly. A
practical choice is a little larger than the combined range noise, pose
uncertainty, and grid resolution, and many systems anneal it from large to small.

**(b)** The drop scales almost exactly with the number of beams (ratio about 4),
because the model multiplies beam likelihoods as if independent. Real neighboring
beams are correlated (same surfaces, same pose error), so the true information
gain from 60 beams is less than four times that of 15 beams. Using all beams with
an independent model makes the posterior overconfident; downsampling or tempering
the likelihood (raising it to a power less than one) compensates.

**(c)** In front of a single long wall, moving parallel to the wall leaves every
endpoint on the wall, so the surface is a long ridge: sharp across the wall, flat
along it. Position along the wall is unobservable from that scan. Heading is still
constrained (rotating moves endpoints off the wall). This is the corridor problem;
a filter there must rely on odometry along the corridor until a doorway or corner
appears.

## 4. A rolling survey boat

In FRD body axes the lever arm is $\boldsymbol{\ell}_a^b = [-1.0, 0, -2.0]^T$ m
(aft is $-x$, up is $-z$). Rolling 12 degrees to starboard is $\phi = +12^\circ$,
and heading 090 is $\psi = 90^\circ$:

$$
\mathbf{C}_b^n\boldsymbol{\ell}_a^b = \mathbf{R}_z(90^\circ)\mathbf{R}_x(12^\circ)\boldsymbol{\ell}_a^b
= [-0.416,\ -1.000,\ -1.956]^T \text{ m (N, E, D)}.
$$

**(a)** The antenna is 1.00 m west (the fixed aft offset, with the bow pointing
east) and 0.42 m south of the IMU, for a total horizontal displacement of 1.08 m.
The roll-induced part is $2.0\sin 12^\circ = 0.42$ m toward starboard (south).

**(b)** $\phi(t) = A\sin(2\pi t/T)$ with $A = 0.209$ rad and $T = 6$ s has a peak
roll rate $A\cdot 2\pi/T = 0.219$ rad/s (12.6 deg/s), reached as the boat passes
upright. The lever-arm velocity is $\boldsymbol{\omega}_{nb}^b \times \boldsymbol{\ell}_a^b
= [p, 0, 0]^T \times [-1, 0, -2]^T = [0, 2p, 0]^T$, a peak of 0.44 m/s sideways.

**(c)** No. The roll-induced displacement (up to 0.42 m) is eight times the 5 cm
requirement and forty times the RTK noise; even a $1^\circ$ roll moves the antenna
3.5 cm. The lever arm must be measured (to a centimeter or so) and applied with the
estimated attitude. An attitude error of $0.5^\circ$ then contributes only
$2.2\text{ m} \times 0.0087 \approx 2$ cm, which is acceptable. The velocity term
matters too if GNSS velocity is fused.

## 5. Weather and the barometer

**(a)** `pressure_altitude(101325 - 600)` gives 50.1 m: a 6 hPa fall reads as
about 50 m of climb. A UAV holding barometric altitude therefore *descends* by
about 50 m over the three hours while believing it is level, which matters a great
deal for terrain clearance.

**(b)** The drift rate is $r = 50.1/10800 = 4.6\times10^{-3}$ m/s. A random walk
with spectral density $q$ has standard deviation $\sqrt{q T}$ after time $T$; to
follow a linear drift over a correction horizon $T$, ask that this be comparable to
the drift $rT$, giving $q \approx r^2 T$ and a per-step standard deviation
$\sigma_w = \sqrt{q\,\Delta t}$ at $\Delta t = 0.1$ s:

| Horizon $T$ | $q$ (m²/s) | $\sigma_w$ at 10 Hz |
|---:|---:|---:|
| 1 min | $1.3\times10^{-3}$ | 1.1 cm |
| 10 min | $1.3\times10^{-2}$ | 3.6 cm |
| 1 h | $7.7\times10^{-2}$ | 8.8 cm |

A value of a few centimeters per step is a reasonable starting point; the horizon
should match how often the aiding sensor in (c) corrects the bias. Students who
model the drift as a constant-rate state (bias plus bias rate) instead of a pure
random walk should receive full credit if they justify it.

**(c)** GNSS height. It is noisy (several meters) but unbiased with respect to
weather, so the difference between barometric altitude and GNSS height (after
accounting for the geoid and the pressure-versus-ellipsoidal height difference,
which is itself slowly varying) observes $b_{\text{baro}}$. The barometer supplies
smooth short-term altitude; GNSS anchors it over the long term. A laser altimeter
over terrain of known elevation would also work, as would a known landing-site
elevation at the end of the flight.
