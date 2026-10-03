# Solutions: The Kalman Filter (`kf.qmd`)

Instructor-only. Exercise numbers match `## Exercises {#sec-kf-exercises}`.
Code snippets assume the chapter's cells have been run (`cv_model`,
`simulate_cv`, `kf_predict`, `kf_update`, `run_kf`, `van_loan`,
`rts_smoother`, `normalized_squares`, `autocorrelation` and the scenario
variables are defined).

## Exercise 1: Steady state by hand

Let $P$ be the prior variance at steady state. The update gives the posterior
$P^+ = PR/(P + R)$ (since $K = P/(P+R)$ and $P^+ = (1-K)P$), and the next
prediction must return to $P$:

$$
P = \frac{PR}{P + R} + Q
\;\Rightarrow\;
P(P + R) = PR + Q(P + R)
\;\Rightarrow\;
P^2 - QP - QR = 0.
$$

Taking the positive root,

$$
P_\text{ss} = \frac{Q + \sqrt{Q^2 + 4QR}}{2},
\qquad
K_\infty = \frac{P_\text{ss}}{P_\text{ss} + R}.
$$

With $Q = 0.01$ and $R = 0.25$ m²: $\sqrt{0.0001 + 0.01} = 0.10050$, so
$P_\text{ss} = 0.05525$ m², $K_\infty = 0.1810$, and the posterior variance is
$(1 - K_\infty)P_\text{ss} = 0.04525$ m² (about 21 cm standard deviation, versus
50 cm per raw barometer reading).

**Effective averaging.** The steady-state filter is an exponentially weighted
moving average $\hat{x}_k = (1 - K)\hat{x}_{k-1}^{\text{pred}} + Kz_k$ with time
constant about $1/K_\infty \approx 5.5$ samples. The variance reduction says the
same thing: $R/P^+ = 0.25/0.04525 = 5.5$, so the filter is worth about five or six
independent measurements. Accept anything in the range 5 to 10 with a sensible
argument (the EWMA "equivalent window" $(2-K)/K \approx 10$ is also a common
answer). The physical point: the filter cannot average longer than the
altitude's random walk lets it, which is what $Q$ encodes.

## Exercise 2: Why Joseph

With $\mathbf{e}_{k|k} = (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{e}_{k|k-1} - \mathbf{K}\mathbf{v}_k$
and $\mathbb{E}[\mathbf{e}_{k|k-1}\mathbf{v}_k^T] = \mathbf{0}$,

$$
\mathbf{P}_{k|k}
= (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}_{k|k-1}(\mathbf{I} - \mathbf{K}\mathbf{H})^T
+ \mathbf{K}\mathbf{R}\mathbf{K}^T.
$$

Expanding,
$\mathbf{P}_{k|k} = \mathbf{P}^- - \mathbf{K}\mathbf{H}\mathbf{P}^- - \mathbf{P}^-\mathbf{H}^T\mathbf{K}^T + \mathbf{K}\mathbf{S}\mathbf{K}^T$
with $\mathbf{S} = \mathbf{H}\mathbf{P}^-\mathbf{H}^T + \mathbf{R}$. For the
optimal gain, $\mathbf{K}\mathbf{S} = \mathbf{P}^-\mathbf{H}^T$, so
$\mathbf{K}\mathbf{S}\mathbf{K}^T = \mathbf{P}^-\mathbf{H}^T\mathbf{K}^T$ and the
last two terms cancel, leaving $(\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}^-$.

**Numbers.** $P^- = 4$, $H = 1$, $R = 1$, so $K_\text{opt} = 0.8$. With the
deweighted $K = 0.4$:

- Joseph: $(1 - 0.4)^2 \cdot 4 + 0.4^2 \cdot 1 = 1.44 + 0.16 = 1.60$ (correct)
- short form: $(1 - 0.4)\cdot 4 = 2.40$ (wrong)

Here the short form is *pessimistic*: it reports more uncertainty than the
filter actually has. That is not a general rule. For an over-weighted gain the
short form becomes optimistic, for example $K = 1$ gives Joseph $= 1$ (the
estimate is just the measurement) but short form $= 0$, a filter claiming
perfect knowledge. Full credit requires identifying Joseph as correct and the
direction of the short form's error for this gain.

## Exercise 3: Constant-acceleration process noise

State $[s, \dot{s}, \ddot{s}]^T$, $\mathbf{A}$ with ones on the superdiagonal,
$\mathbf{G} = [0, 0, 1]^T$, white jerk with density $q$. Then
$\boldsymbol{\Phi}(\tau)\mathbf{G} = [\tau^2/2, \tau, 1]^T$ and

$$
\mathbf{Q}_d = q\int_0^{\Delta t}
\begin{bmatrix}
\tau^4/4 & \tau^3/2 & \tau^2/2 \\
\tau^3/2 & \tau^2 & \tau \\
\tau^2/2 & \tau & 1
\end{bmatrix} d\tau
= q
\begin{bmatrix}
\Delta t^5/20 & \Delta t^4/8 & \Delta t^3/6 \\
\Delta t^4/8 & \Delta t^3/3 & \Delta t^2/2 \\
\Delta t^3/6 & \Delta t^2/2 & \Delta t
\end{bmatrix}.
$$

Verification:

```python
A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0.0]])
G = np.array([[0.0], [0.0], [1.0]])
dt_c, q_c = 0.1, 2.0
Phi, Q_d = van_loan(A, G, np.array([[q_c]]), dt_c)
Q_closed = q_c * np.array([[dt_c**5 / 20, dt_c**4 / 8, dt_c**3 / 6],
                           [dt_c**4 / 8,  dt_c**3 / 3, dt_c**2 / 2],
                           [dt_c**3 / 6,  dt_c**2 / 2, dt_c]])
assert np.allclose(Q_d, Q_closed, rtol=1e-9, atol=1e-15)
```

For $\Delta t = 0.1$, $q = 2$: diagonal $1.0\times10^{-6}$, $6.67\times10^{-4}$,
$0.2$; off-diagonals $2.5\times10^{-5}$, $3.33\times10^{-4}$, $0.01$.

## Exercise 4: Diagnose the filter

Add two entries to the Monte Carlo loop, passing `R_f = R * 0.1` and
`R_f = R * 10` to `run_kf` with the correct `Q`:

```python
cases = {"correct": (1, 1), "q small": (1/50, 1), "q large": (50, 1),
         "R small": (1, 0.1), "R large": (1, 10)}
for label, (fq, fr) in cases.items():
    F_f, Q_f = cv_model(dt, fq * q_true)
    R_f = R * fr
    rng_mc = np.random.default_rng(7)
    nees = np.zeros((50, n_steps)); nis = np.zeros((50, n_steps))
    for i in range(50):
        x_t, z_i = simulate_cv(rng_mc, n_steps, dt, q_true, sigma_z, x0_true)
        x0_i = x0_true + np.linalg.cholesky(P0) @ rng_mc.standard_normal(2)
        run = run_kf(z_i, F_f, Q_f, H, R_f, x0_i, P0)
        nees[i] = normalized_squares((x_t - run["x"])[1:], run["P"][1:])
        nis[i] = normalized_squares(run["nu"][1:], run["S"][1:])
        if i == 0:
            rho = autocorrelation(run["nu"][1:, 0], 3)
    print(label, nees.mean(), nis.mean(), rho[0])
```

Expected output (seed 7, 50 runs):

| Case | ANEES (n = 2) | ANIS (m = 1) | lag-1 $\hat\rho$ (run 0) |
|---|---|---|---|
| correct | 2.10 | 0.99 | -0.12 (inside $\pm 0.16$) |
| $q$ / 50 | 46.0 | 2.56 | +0.67 |
| $q \times 50$ | 1.12 | 0.73 | -0.42 |
| $\mathbf{R}$ / 10 | 11.8 | 8.17 | -0.32 |
| $\mathbf{R} \times 10$ | 1.15 | 0.16 | +0.35 |

All four rows of the diagnostic table are confirmed. Points to look for in the
write-up: NIS is far more sensitive to a wrong $\mathbf{R}$ than to a wrong
$\mathbf{Q}$ in this scenario, because $\mathbf{S}$ is dominated by
$\mathbf{R}$; NEES catches both; and whiteness distinguishes the two
"NIS too high" cases ($q$ small: positive correlation; $\mathbf{R}$ small:
negative).

## Exercise 5: GNSS outage under a bridge

Set `z_gap = z.copy(); z_gap[(t >= 60) & (t < 90), 0] = np.nan` and modify the
loop so that a NaN measurement skips the update (posterior = prior). Keep the
priors for the smoother.

Expected behavior with the chapter's seed and the correct $q$:

- $\sigma_\text{pos}$ grows from about 2.3 m at $t = 59$ s to about 37 m at
  $t = 89$ s. The growth is roughly cubic in time early on and dominated by the
  $q\,t^3/3$ and $t^2 P_{vv}$ terms: with $q = 0.1$ the vessel's speed really
  can wander by more than 1.5 m/s over 30 s.
- The NIS of the first fix after the gap ($t = 90$ s) is small (0.01 for this
  seed). Over 200 Monte Carlo runs the mean reacquisition NIS is about 1.0, as it
  should be for a consistent filter: the huge innovation covariance properly
  accounts for the large prediction error.
- With $q$/50 the bounds grow only to about 7 m while the actual error is
  similar; over 200 runs the mean reacquisition NIS is about 35 and 77% of
  reacquisition fixes exceed the 95% chi-square gate (3.84). An overconfident
  filter would reject the very fix it needs. This is the practical argument for
  honest $\mathbf{Q}$.

**Smoother.** Run `rts_smoother` on the gapped output. The smoothed position
standard deviation is about 2.35 m at both edges of the gap and peaks at about
5.9 m near $t = 74$ s, the middle. The smoother has measurements on both sides of
the gap: the forward pass constrains the start, the backward pass constrains the
end, and the middle is farthest from both, like the sag of a rope tied at two
ends. The filter's uncertainty peaks at the end of the gap because it only has
the past.

## Exercise 6: What a Doppler log cannot see

$\mathbf{H} = [0, 1]$, $\mathbf{H}\mathbf{F} = [0, 1]$, so

$$
\mathcal{O} = \begin{bmatrix} 0 & 1 \\ 0 & 1 \end{bmatrix},
$$

rank 1, null space spanned by $[1, 0]^T$: position is unobservable. A constant
position offset produces exactly the same speed measurements forever.

Prediction: position variance never drops below its initial value and grows
without bound, because position error is the integral of the (small but
nonzero) speed error plus the process noise. With a precise DVL
($\sigma = 0.05$ m/s) the growth is slow. In a test with the chapter's $q$ and
$P_0$, $\sigma_\text{pos}$ goes from 10.0 m to 10.2 m after 300 s and 10.3 m
after 600 s while $\sigma_\text{speed}$ settles near a few cm/s.

Adding a position fix with $\sigma = 5$ m every 60 s makes the system observable
(stacking both measurement types gives rank 2). The position standard deviation
then settles into a sawtooth between roughly 2.0 m (just after a fix) and 2.5 m
(just before). Code:

```python
F1, Q1 = cv_model(1.0, q_true)
H_dvl, R_dvl = np.array([[0.0, 1.0]]), np.array([[0.05**2]])
H_fix, R_fix = np.array([[1.0, 0.0]]), np.array([[5.0**2]])
rng6 = np.random.default_rng(0)
x_t = np.array([0.0, 5.0])
x_h = x_t + np.linalg.cholesky(P0) @ rng6.standard_normal(2)
P_k, sig = P0.copy(), []
for k in range(1, 601):
    x_t = F1 @ x_t + np.linalg.cholesky(Q1) @ rng6.standard_normal(2)
    x_h, P_k = kf_predict(x_h, P_k, F1, Q1)
    z_dvl = np.array([x_t[1] + 0.05 * rng6.standard_normal()])
    x_h, P_k, _, _ = kf_update(x_h, P_k, z_dvl, H_dvl, R_dvl)
    if k % 60 == 0:          # remove this block for the DVL-only case
        z_fix = np.array([x_t[0] + 5.0 * rng6.standard_normal()])
        x_h, P_k, _, _ = kf_update(x_h, P_k, z_fix, H_fix, R_fix)
    sig.append(np.sqrt(P_k[0, 0]))
```

Real-world connection: this is why AUVs surface for GNSS or use USBL/LBL
acoustic fixes; a DVL-aided INS alone drifts at a fraction of a percent of
distance traveled, forever.
