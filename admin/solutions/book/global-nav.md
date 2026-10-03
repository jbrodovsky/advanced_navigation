# Solutions: Global Navigation and GNSS Measurements (`global-nav.qmd`)

Instructor-only. Exercise numbers match `## Exercises {#sec-global-nav-exercises}`.
Code snippets assume the chapter's cells have been run (they use `radii`,
`geodetic_to_ecef`, `C_e_to_n`, `place_satellites`, `geometry_matrix`,
`solve_pvt`, `dop`, `r_true`, `lat0`, `lon0`, `clock_true_m`, `sats_good`,
`el_good`). Numbers below were produced with those cells.

## Exercise 1: DOP by hand, then by code

In local NED axes a satellite at azimuth $a$ and elevation $e$ has line of sight
$\mathbf{u} = [\cos e\cos a,\ \cos e\sin a,\ -\sin e]^T$, so its row of
$\mathbf{G}$ is $[-\cos e\cos a,\ -\cos e\sin a,\ \sin e,\ 1]$.

Zenith satellite: row $[0, 0, 1, 1]$. Three satellites at elevation $e$ and
azimuths $0^\circ, 120^\circ, 240^\circ$: because
$\sum\cos a_k = \sum\sin a_k = \sum\sin a_k\cos a_k = 0$ and
$\sum\cos^2 a_k = \sum\sin^2 a_k = 3/2$ for three equally spaced azimuths,

$$
\mathbf{G}^T\mathbf{G} =
\begin{bmatrix}
\tfrac32\cos^2 e & 0 & 0 & 0 \\
0 & \tfrac32\cos^2 e & 0 & 0 \\
0 & 0 & 1 + 3\sin^2 e & 1 + 3\sin e \\
0 & 0 & 1 + 3\sin e & 4
\end{bmatrix}.
$$

The horizontal block decouples from the vertical-plus-clock block because the
horizontal components of the lines of sight sum to zero. With $e = 30^\circ$:
horizontal diagonal $1.125$; vertical-clock block
$\begin{bmatrix}1.75 & 2.5\\ 2.5 & 4\end{bmatrix}$ with determinant $0.75$, so its
inverse is $\frac{1}{0.75}\begin{bmatrix}4 & -2.5\\ -2.5 & 1.75\end{bmatrix}$.

| | $e = 30^\circ$ | $e = 5^\circ$ |
|---|---|---|
| HDOP $=\sqrt{2/(1.5\cos^2 e)}$ | 1.333 | 1.159 |
| VDOP | 2.309 | 1.265 |
| TDOP | 1.528 | 0.640 |
| PDOP | 2.667 | 1.716 |
| GDOP | 3.073 | 1.831 |

(For $e = 5^\circ$: horizontal diagonal $1.4886$, vertical-clock block
$\begin{bmatrix}1.0228 & 1.2615\\1.2615 & 4\end{bmatrix}$, determinant $2.4998$.)

**Why VDOP improves.** The vertical and clock columns of $\mathbf{G}$ are
$[\sin e_k]$ and $[1]$. When every satellite is high, $\sin e_k \approx 1$ and the
two columns are nearly parallel: raising the receiver and advancing its clock
change all pseudoranges almost identically, so least squares cannot tell them
apart. Low satellites have $\sin e \approx 0$, so their pseudoranges respond to the
clock but hardly to height, which breaks the degeneracy.

**Why clock and height are correlated.** The off-diagonal
$1 + 3\sin e$ of the information matrix is large and positive, so the covariance
has a large negative correlation between the down and clock estimates: raising
the receiver by 1 m shortens the ranges to high satellites by almost 1 m, which a
clock bias 1 m larger cancels, so an estimate that is too high (too small in the
down coordinate) pairs with a clock-bias estimate that is too large. The correlation coefficient is
$-2.5/\sqrt{4\cdot1.75} = -0.94$ at $30^\circ$ and $-0.63$ at $5^\circ$.

Code check (matches the table to rounding):

```python
for el_low in (30.0, 5.0):
    az = np.array([0.0, 0.0, 120.0, 240.0])
    el = np.array([90.0, el_low, el_low, el_low])
    sats = place_satellites(r_true, lat0, lon0, az, el)
    _, G, _ = solve_pvt(sats, np.linalg.norm(sats - r_true, axis=1) + clock_true_m)
    print(el_low, dop(G, lat0, lon0)[0])
```

## Exercise 2: Weighted least squares

```python
sigma0 = 1.5
sig_s = sigma0 / np.sin(np.radians(el_good))      # [1.58 3.2 2.12 5.8 1.73 4.0 2.44 7.21] m
W = np.diag(1.0 / sig_s**2)

def solve_wls(sats, rho, W, x0, iters=4):
    x = np.array(x0, dtype=float)
    for _ in range(iters):
        G, ranges = geometry_matrix(sats, x[:3])
        x = x + np.linalg.solve(G.T @ W @ G, G.T @ W @ (rho - ranges - x[3]))
    return x, geometry_matrix(sats, x[:3])[0]
```

Monte Carlo (3000 epochs, `default_rng(5)`), horizontal / vertical RMS:

| Solver | Monte Carlo RMS H / V [m] | Predicted H / V [m] |
|---|---|---|
| Unweighted LS | 4.22 / 6.08 | $\mathbf{A}\mathbf{R}\mathbf{A}^T$: 4.21 / 6.12 |
| Unweighted, naive $\sigma_0^2(\mathbf{G}^T\mathbf{G})^{-1}$ | | 1.37 / 2.08 (wrong) |
| Weighted LS | 2.85 / 4.30 | $(\mathbf{G}^T\mathbf{R}^{-1}\mathbf{G})^{-1}$: 2.88 / 4.33 |

The naive unweighted covariance assumes every pseudorange has the same
$\sigma$. With heteroscedastic noise the unweighted estimator's true covariance
is $\mathbf{A}\mathbf{R}\mathbf{A}^T$ with $\mathbf{A} = (\mathbf{G}^T\mathbf{G})^{-1}\mathbf{G}^T$;
any single "$\sigma^2$" plugged into $(\mathbf{G}^T\mathbf{G})^{-1}$ is wrong in
shape as well as size (using $\sigma_0$ underestimates by a factor of three; even
the mean variance underestimates). Weighting is the best linear unbiased estimator
here, about a third better in both axes, and its formula covariance is correct.

**Elevation mask.** A $15^\circ$ mask drops only satellite 8 ($12^\circ$). DOP
rises from (H, V) = (0.92, 1.39) to (1.13, 1.75), and the unweighted
covariance with the remaining noise becomes 3.89 / 5.25 m: better than unweighted
with all satellites, worse than weighting. Weighting keeps the geometric benefit
of low satellites while discounting their noise, so it is preferred when the
noise model is trustworthy; a mask is the robust choice when low satellites
suffer non-Gaussian errors (multipath, NLOS) that no $\sigma$ describes. Real
receivers commonly do both: a 5 to 15 degree mask plus elevation or C/N0
weighting.

## Exercise 3: What the receiver's MSL height does to your filter

(a) $H = h - N_g$ with $N_g = -28$ m, so $H = h + 28$ m: the receiver's MSL
altitude is 28 m larger than the ellipsoidal height. Fed in as $h$, every GNSS
height measurement is biased 28 m **high** (in NED, $-28$ m in $\delta p_D$). The
filter has no state for this bias, so the nominal height converges to
$h + 28$ m, at the rate set by the vertical gain. After convergence the
innovations are zero-mean again, the NIS sits at its expected value, and nothing
flags the problem: a constant bias in the only height source is unobservable.
Horizontal position is unaffected (the ECEF position moves 28 m along the
ellipsoid normal).

(b) Normal gravity decreases by about $3.086\times10^{-6}$ m/s² per meter of
height, so a 28 m height error changes $g$ by $8.6\times10^{-5}$ m/s², about
9 $\mu$g. A smartphone accelerometer bias is a few mg to tens of mg, so this is
three orders of magnitude below the sensor error and irrelevant to the
propagation. (For a navigation-grade INS with ~25 $\mu$g accelerometers it would be
comparable to the sensor error, though still small.)

(c) At the switch the measured height drops by 28 m (the measurement is now
correct, and the state is 28 m too high). The vertical innovation jumps to about
$+28$ m (down positive). With $\sigma_v = 5$ m and a small vertical state
covariance, the NIS is about $(28/5)^2 \approx 31$, well beyond the 99% gate for
3 degrees of freedom (11.3). The gate rejects the fix. Because the height is wrong
every subsequent fix is rejected too, and the filter coasts on the INS alone. The
vertical covariance then grows (fast for a MEMS IMU, whose vertical channel is
unbounded without aiding) until the gate opens, at which point the filter snaps
down by 28 m in one update, possibly corrupting vertical velocity and the $z$
accelerometer bias on the way. Horizontal updates are rejected along with the
vertical unless the gate is applied per axis. Lesson: gate per measurement
component where possible, log rejections, and treat a long run of consecutive
rejections as a fault to investigate, not as normal operation.

(d) Any independent height reference exposes it: a barometric altimeter
referenced to a known datum (or a known starting elevation), a digital elevation
model or map of the road surface, a surveyed benchmark or known start point, a
second receiver or NMEA source that reports $h$ and $N_g$ separately (GGA field
11), or RTK/PPP solutions that are always ellipsoidal. On a long route the
undulation itself changes by meters per hundred kilometers, which shows up as a
slow vertical drift that no IMU explains.

## Exercise 4: The masthead antenna

Take the ship heading north, level, rolling by $\phi(t) = \phi_0\sin(2\pi t/T)$ with
$\phi_0 = 15^\circ$, $T = 10$ s. With $\mathbf{C}_b^n = \mathbf{R}_x(\phi)$ and
$\boldsymbol{\ell}^b = [-10, 0, -25]^T$,

$$
\mathbf{C}_b^n\boldsymbol{\ell}^b = [-10,\ 25\sin\phi,\ -25\cos\phi]^T .
$$

(a) Sideways (east) displacement amplitude $25\sin 15^\circ = 6.47$ m; the
antenna also bobs vertically by $25(1-\cos 15^\circ) = 0.85$ m twice per roll
cycle. Sideways velocity $25\cos\phi\,\dot\phi$ peaks at $\phi = 0$ with
$\dot\phi_{\max} = \phi_0\,2\pi/T = 0.1645$ rad/s, so 4.11 m/s.

(b) Ignoring the lever arm, the filter sees: a constant 25 m vertical offset
(the IMU height converges to the antenna height, mostly harmless on its own);
a 10 m horizontal offset that rotates with heading (a 10 m innovation that changes
whenever the ship turns); and a 0.1 Hz sideways oscillation of 6.5 m amplitude
(6.5$\sigma$ with $\sigma = 1$ m). The velocity innovations are worse: 4.1 m/s
against $\sigma = 0.1$ m/s is 41$\sigma$. An ungated filter tries to explain
these with velocity, attitude, and accelerometer-bias corrections and corrupts
all three (roll and the lateral accelerometer bias in particular); a gated filter
rejects most velocity updates near mid-roll, exactly when the ship moves fastest.

(c) The attitude block is $-[\hat{\mathbf{C}}_b^n\boldsymbol{\ell}^b]_\times$. A
roll error is a rotation about north, $\delta\boldsymbol\theta = [\delta\phi, 0, 0]^T$,
and moves the predicted antenna by
$\delta\boldsymbol\theta\times\hat{\mathbf{C}}_b^n\boldsymbol{\ell}^b \approx [0,\ 25\,\delta\phi,\ 0]^T$
when upright. So $1^\circ$ of roll error is 0.44 m of east position, visible
against 1 m GNSS noise after a handful of updates, and the filter can estimate
roll from GNSS position alone. For a car with the antenna 0.2 m above the IMU the
same $1^\circ$ is 3.5 mm, hopelessly below GNSS noise, so roll and pitch there must
come from the accelerometers (gravity) instead.

## Exercise 5: Where the centimeters come from

(a) Loose: nothing. The receiver cannot form a 3-D fix with three satellites
(four unknowns), so it reports no position (or a stale or 2-D fix with an assumed
height, which should be treated with suspicion). The filter coasts on the INS.
Tight: three pseudoranges (and three Doppler measurements) each epoch. Each
constrains the position error along its own line of sight, combined with the
clock error. In a north-south street the visible satellites lie roughly north and
south, high in the sky, so the constraints are on northing and height; easting
(cross-street) is unconstrained by GNSS and stays INS-limited.

(b) Each pseudorange row has a 1 in the clock column. If the clock is uncertain,
it absorbs the common part of the residuals, so three satellites give roughly two
position constraints, and two give one. A well-modeled clock propagated by its
process model (@sec-adv-gnss-clock-model) turns each satellite into a full
position constraint, which is the difference between a useful and a marginal
update in the canyon. A TCXO's clock uncertainty grows to tens of meters within a
minute, so clock modeling matters most precisely during the long partial outages
that motivate tight coupling.

(c) Questions for the vendor: "Is the centimeter figure from carrier-phase RTK or
PPP, and under what baseline, correction service, and sky conditions?" and "What
is the accuracy and integrity when the carrier-phase fix is lost (float or code
only), and how long does it take to re-fix after an outage?" Centimeter accuracy
comes from carrier-phase measurements with correctly resolved integer ambiguities
(RTK with a base within roughly 10 to 20 km, network RTK, or converged PPP).
Tight coupling helps such a system bridge partial outages, reject bad satellites,
and re-fix faster, but standalone code pseudoranges with tight coupling still give
meter-level accuracy.
