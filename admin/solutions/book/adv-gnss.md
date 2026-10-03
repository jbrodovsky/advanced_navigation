# Solutions: GNSS Signals, Receivers, and Deep Integration (`adv-gnss.qmd`)

Instructor-only. The chapter is an optional enrichment chapter and is not
assessed; these solutions support instructors who assign it anyway. Exercise
numbers match `## Exercises {#sec-adv-gnss-exercises}`. Code snippets assume the
chapter's cells have been run (they use `clock_model`, `clock_psds`,
`OSCILLATORS`, `predict_pseudoranges`, `solve_pvt`, `sats`, `az`, `el`, `nominal`,
`r_imu`, `C_en`, `rho_clean`, `x0`, `slopes`, `lam`, `sigma_rho`). Numbers below
were produced with those cells.

## Exercise 1: Clock process noise

With $\mathbf{A} = \begin{bmatrix}0 & 1\\0 & 0\end{bmatrix}$,
$\boldsymbol\Phi(t) = \begin{bmatrix}1 & t\\0 & 1\end{bmatrix}$, and
$\mathbf{G} = \mathbf{I}$,

$$
\boldsymbol\Phi(t)\mathbf{Q}_c\boldsymbol\Phi(t)^T =
\begin{bmatrix} S_f + S_g t^2 & S_g t \\ S_g t & S_g \end{bmatrix},
$$

and integrating from 0 to $\tau$ gives @eq-adv-gnss-clock-q:
$Q_{11} = S_f\tau + S_g\tau^3/3$, $Q_{12} = S_g\tau^2/2$, $Q_{22} = S_g\tau$.

Units: the bias is in m and the drift in m/s, so $w_f$ is in m/s and $w_g$ in
m/s². Their PSDs are $S_f$ [m²/s] and $S_g$ [m²/s³]. Then $S_f\tau$ and
$S_g\tau^3$ are m², $S_g\tau^2$ is m²/s, and $S_g\tau$ is m²/s², as they must be
for $\operatorname{var}(b)$, $\operatorname{cov}(b, d)$, $\operatorname{var}(d)$.

Coasting time until $\sigma_b > 3$ m from $(\sigma_b, \sigma_d) = (1\text{ m}, 0.05\text{ m/s})$,
propagated in 0.1 s steps:

```python
for name, c in OSCILLATORS.items():
    F1, Q1 = clock_model(0.1, c["h0"], c["h_m2"])
    P, t = np.diag([1.0, 0.05**2]), 0.0
    while np.sqrt(P[0, 0]) <= 3.0:
        P, t = F1 @ P @ F1.T + Q1, t + 0.1
    print(name, round(t, 1))
```

| Oscillator | Time to $\sigma_b = 3$ m | Variance contributions at that time [m²] |
|---|---|---|
| TCXO | 8.7 s | initial 1.0, initial drift $(\sigma_d t)^2$ 0.19, $S_f t$ 0.08, $S_g t^3/3$ 7.8 |
| OCXO | 46.7 s | initial 1.0, initial drift 5.4, $S_f t$ 0.17, $S_g t^3/3$ 2.4 |

For the TCXO, the random-walk-frequency term $S_g t^3/3$ (from $h_{-2}$)
dominates almost immediately: the frequency itself wanders, so the bias error
grows like $t^{3/2}$. For the OCXO, $h_{-2}$ is three orders of magnitude smaller
and the growth is dominated by the initial drift uncertainty, i.e. by how well the
drift was estimated before the outage, with $h_{-2}$ second. The white-frequency
term $h_0$ barely matters for either at these horizons. Datasheet lesson: for
coasting, look at the long-term stability (the random-walk or aging part of the
Allan deviation at tens of seconds to minutes), not the short-term (1 s) Allan
deviation, and estimate the drift well while satellites are available.

## Exercise 2: The tightly coupled Jacobian

(a) The predicted pseudorange is $\hat\rho_s = \|\mathbf{r}_s' - \hat{\mathbf{r}}_a\| + \hat b + \dots$
The true antenna position in NED, relative to the nominal IMU position, is
$\delta\mathbf{p} + \mathbf{C}_b^n\boldsymbol\ell^b$, with
$\mathbf{C}_b^n \approx (\mathbf{I} + [\delta\boldsymbol\theta]_\times)\hat{\mathbf{C}}_b^n$
(@sec-frames-conventions). So the antenna error is

$$
\delta\mathbf{r}_a^n = \delta\mathbf{p} + [\delta\boldsymbol\theta]_\times\hat{\mathbf{C}}_b^n\boldsymbol\ell^b
= \delta\mathbf{p} - [\hat{\mathbf{C}}_b^n\boldsymbol\ell^b]_\times\delta\boldsymbol\theta .
$$

The gradient of $\|\mathbf{r}_s - \mathbf{r}_a\|$ with respect to $\mathbf{r}_a$ is
$-\mathbf{u}_s^T$ (in any frame, here NED), so
$\nu_s \approx -\mathbf{u}_s^{nT}\delta\mathbf{p} + \mathbf{u}_s^{nT}[\hat{\mathbf{C}}_b^n\boldsymbol\ell^b]_\times\delta\boldsymbol\theta + \delta b$,
giving @eq-adv-gnss-tight-h. The attitude block is $+\mathbf{u}^T[\cdot]_\times$
because two minus signs meet: moving toward the satellite shortens the range, and
$[\delta\boldsymbol\theta]_\times\mathbf{a} = -[\mathbf{a}]_\times\delta\boldsymbol\theta$.
The chapter's finite-difference check confirms the sign numerically.

(b) Let $\mathbf{U} = [\mathbf{u}_1, \dots, \mathbf{u}_4]$ (3 x 4) and suppose the
clock is perfectly known (drop the clock column). Stacking four rows gives
$\mathbf{H}_\rho = -\mathbf{U}^T[\mathbf{I}_3,\ \mathbf{0},\ -[\hat{\mathbf{C}}\boldsymbol\ell]_\times,\ \mathbf{0},\ \mathbf{0}]
= -\mathbf{U}^T\mathbf{H}_{loose}$, and the pseudorange residual vector is
$-\mathbf{U}^T$ times the antenna-position innovation. If $\mathbf{U}^T$ (4 x 3) has
full column rank (lines of sight not all in one plane), the four residuals
determine the antenna-position innovation uniquely, with one redundant
combination left over for checking; the information content is the same as a
loose position measurement whose covariance is
$(\mathbf{U}\mathbf{R}_\rho^{-1}\mathbf{U}^T)^{-1}$. With an unknown clock,
four satellites are needed just to recover the loose measurement, which is the
receiver's PVT problem.

(c) The Sagnac term $(\omega_{ie}/c)(x_s y_a - y_s x_a) = (\omega_{ie}/c)\,[\mathbf{r}_s\times\mathbf{r}_a]_z$
is largest when both vectors lie in the equatorial plane and are as far apart in
longitude as visibility allows: a receiver on the equator and a satellite on its
eastern or western horizon. Then the angle between them satisfies
$\cos\gamma = R_e/R_s$ and the magnitude is
$(\omega_{ie}/c)R_sR_e\sin\gamma \approx 40$ m (positive or negative depending
on east or west).

Ignoring it for the chapter's eight-satellite sky (pseudoranges generated with the
Sagnac rotation, solved without it):

```python
rho_true = predict_pseudoranges(nominal, np.eye(3), np.zeros(3), 9e4, sats)[0]
x_ns, _, _ = solve_pvt(sats, rho_true, np.r_[r_imu + 100, 0.0])
print(C_e_to_n(*nominal[:2]) @ (x_ns[:3] - r_imu))   # [-0.46, 26.29, -0.27] m
```

The error is about 26 m **east** with sub-meter north and down components and a
0.25 m clock change. The common part of the correction is equivalent to rotating
all satellites (hence the solution) by $\omega_{ie}\tau \approx 5\times10^{-6}$ rad
about the polar axis, which at latitude $42^\circ$ is
$\omega_{ie}\tau R\cos L \approx 26$ m of longitude. The solver places the receiver
east because it uses satellite positions that are, in the receive-time frame,
rotated east of where they really are.

## Exercise 3: Double differences

Carrier phase in meters for receiver $r$ and satellite $j$:

$$
\Phi_r^j = \rho_r^j + c\,\delta t_r - c\,\delta t^j + T_r^j - I_r^j + \lambda_c N_r^j + \epsilon_r^j .
$$

Single difference between receivers (rover $r$, base $b$):

$$
\Delta\Phi_{rb}^j = \Delta\rho_{rb}^j + c(\delta t_r - \delta t_b) + \Delta T_{rb}^j - \Delta I_{rb}^j + \lambda_c\Delta N_{rb}^j + \Delta\epsilon_{rb}^j .
$$

The satellite clock $\delta t^j$ cancels exactly. Orbit error projects almost
identically onto both lines of sight on a short baseline and largely cancels, as
do $T$ and $I$ when both receivers see the same atmosphere.

Double difference between satellites $j$ and $k$:

$$
\nabla\Delta\Phi_{rb}^{jk} = \nabla\Delta\rho_{rb}^{jk} + \nabla\Delta T - \nabla\Delta I
+ \lambda_c\nabla\Delta N_{rb}^{jk} + \nabla\Delta\epsilon .
$$

The receiver clock difference $c(\delta t_r - \delta t_b)$ is common to both
satellites and cancels exactly. Exact cancellations: satellite clocks, receiver
clocks (and common receiver hardware delays). Approximate cancellations on a short
baseline: orbit errors, tropospheric and ionospheric delays. Survivors: the
geometry (which contains the unknown baseline), residual atmosphere, multipath
and noise (amplified by about a factor of 2 in variance per differencing step),
and $\nabla\Delta N = (N_r^j - N_b^j) - (N_r^k - N_b^k)$, an integer because it is a
sum and difference of integers (provided the fractional initial phases and
hardware phase biases cancel in the double difference, which they do for
identical signal types). With longer baselines the two receivers see the
ionosphere and troposphere through increasingly different paths; the residual
$\nabla\Delta I$ and $\nabla\Delta T$ grow to a significant fraction of the
19 cm wavelength beyond roughly 10 to 20 km, and the integer search can no longer
pick the right candidate reliably.

## Exercise 4: How big a fault can hide?

Because the PVT problem is almost linear near the solution, the test statistic
for a bias $\beta$ on satellite $i$ can be computed directly from the residual
projection: $T = \|\mathbf{S}(\boldsymbol\epsilon + \beta\mathbf{e}_i)\|^2/\sigma^2$
(solving the full nonlinear problem gives the same answer to many digits and is
slower).

```python
_, G, _ = solve_pvt(sats, rho_clean, x0)
A = np.linalg.solve(G.T @ G, G.T)
S = np.eye(8) - G @ A
T_D = chi2.ppf(1 - 1e-5, 4)
rng = np.random.default_rng(3)
eps = sigma_rho * rng.standard_normal((500, 8))
biases = np.arange(0, 101, 5.0)
for i in range(8):
    pd = [np.mean(np.sum(((eps + b * np.eye(8)[i]) @ S.T) ** 2, axis=1) / sigma_rho**2 > T_D)
          for b in biases]
    plt.plot(biases, pd, label=f"sat {i + 1}")
```

| Satellite (az, el) | $S_{ii}$ | Slope | Bias at $P_D \ge 99.9\%$ (MC, 5 m grid) | $\sigma\sqrt{\lambda/S_{ii}}$ |
|---|---|---|---|---|
| 1 (10°, 72°) | 0.525 | 0.22 | 35 | 33.9 |
| 2 (60°, 28°) | 0.479 | 0.60 | 35 | 35.6 |
| 3 (105°, 45°) | 0.664 | 0.30 | 30 | 30.2 |
| 4 (150°, 15°) | 0.292 | 0.75 | 50 | 45.5 |
| 5 (200°, 60°) | 0.573 | 0.32 | 35 | 32.5 |
| 6 (245°, 22°) | 0.483 | 0.56 | 40 | 35.4 |
| 7 (290°, 38°) | 0.663 | 0.33 | 35 | 30.2 |
| 8 (330°, 12°) | 0.321 | 0.65 | 45 | 43.4 |

With 500 trials, "99.9%" means no misses, so the Monte Carlo column is coarse,
but it tracks $\sigma\sqrt{\lambda/S_{ii}}$ ($\lambda = 67.2$ for 4 degrees of
freedom, $P_{FA} = 10^{-5}$, $P_{MD} = 10^{-3}$) to within the grid. The worst
satellite is 4, low in the south-east, followed by 8, low in the north-west. Low
satellites have the largest horizontal leverage (their lines of sight are nearly
horizontal, so a bias moves the horizontal fix a lot) and the smallest $S_{ii}$
(no other satellite duplicates their direction, so the fit absorbs much of their
bias instead of showing it in the residuals). The high satellite 1 has the
smallest slope: a bias on it mostly moves height and clock.

Removing satellite 8 (lowest, $12^\circ$) leaves seven satellites and 3 degrees of
freedom ($\lambda = 64.4$). Slopes become
[0.21, 1.23, 0.37, 0.84, 0.72, 0.56, 0.70]; satellite 2, which lost its partner
direction, now has slope 1.23, and the simplified HPL rises from 18.4 m to 29.6 m.
Losing one well-placed satellite can raise the protection level more than it raises
DOP, which is why aviation integrity availability is so sensitive to constellation
outages.

## Exercise 5: A spoofer meets an INS

(a) Without bias states, the INS prediction follows the true trajectory and the
spoofed position departs from it by $\frac12 a_s t^2$. The innovation reaches
$3\sigma = 9$ m at $t = \sqrt{18/a_s}$: 42.4 s for $a_s = 0.01$ m/s² and 13.4 s for
$a_s = 0.1$ m/s². (A real filter's innovation covariance also grows between
updates, so these are optimistic lower bounds on detection time.)

(b) With accelerometer-bias states, the filter can explain a steady apparent
acceleration as a bias of the same size. If $a_s$ is well below the bias
uncertainty, successive updates simply pull the bias estimate toward $-a_s$ (in the
appropriate axis), the innovations stay small, and the filter follows the spoofer
smoothly: the attack is absorbed, not detected. For the phone
($\sigma_{b_a} \approx 0.05$ m/s²), $a_s = 0.01$ m/s² is a fifth of the bias
uncertainty and will be absorbed; $a_s = 0.1$ m/s² is twice it, so the bias state
moves until the NIS rises, and detection becomes possible but slow and
threshold-dependent. For the airliner ($\sigma_{b_a} \approx 5\times10^{-4}$ m/s²),
both accelerations are 20 to 200 times the bias uncertainty; the bias estimate
cannot move that far without large normalized innovations, so both are detected
(the slower in roughly the time from part (a), the faster sooner). A smart spoofer
therefore drags with an acceleration below the victim IMU's bias uncertainty; the
better the IMU, the slower the spoofer must go and the longer it takes to move the
victim a useful distance. Bias random walk lets even a good filter follow a slow
enough drag eventually, which is why inertial checks bound, rather than eliminate,
spoofing.

(c) Receiver-level cues that do not depend on the IMU: uniform or unusually high
C/N0 across satellites, or a step in received power or AGC level; a jump in
receiver clock bias or time; all signals arriving from one direction (needs a
multi-element antenna); disagreement between constellations or frequencies;
failure of navigation-message authentication (Galileo OSNMA); position or time
inconsistent with an independent source (cellular or Wi-Fi location, network
time). A smartphone can realistically check C/N0 statistics, clock and time jumps
(Android exposes raw GNSS measurements and clock fields on many devices),
constellation consistency, and agreement with network location and time.
Angle-of-arrival needs antenna hardware phones lack, and OSNMA support depends on
the chipset.
