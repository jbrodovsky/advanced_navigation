# Solutions: Probability Theory Refresher (`probability-theory.qmd`)

Instructor-only. Exercise numbers match `## Exercises {#sec-probability-exercises}`.
All code runs with numpy, scipy, and matplotlib; each block is self-contained.

---

## Exercise 1: Which "center" goes on the data sheet?

**(a) Rayleigh density.** The north and east errors are independent
$\mathcal{N}(0, \sigma^2)$, so their joint density is
$\frac{1}{2\pi\sigma^2}\exp\!\left(-\frac{e_N^2 + e_E^2}{2\sigma^2}\right)$.
Change to polar coordinates $e_N = r\cos\phi$, $e_E = r\sin\phi$ (Jacobian $r$)
and integrate out $\phi \in [0, 2\pi)$:

$$
p(r) = \int_0^{2\pi} \frac{1}{2\pi\sigma^2} e^{-r^2/2\sigma^2}\, r\, d\phi = \frac{r}{\sigma^2} e^{-r^2/2\sigma^2}, \qquad r \ge 0 .
$$

The CDF is $F(r) = 1 - e^{-r^2/2\sigma^2}$.

**(b) Mode, median, mean.**

- Mode: $\frac{d}{dr}\left(r e^{-r^2/2\sigma^2}\right) = \left(1 - r^2/\sigma^2\right)e^{-r^2/2\sigma^2} = 0 \Rightarrow r = \sigma$. For $\sigma = 2$ m: **2.00 m**.
- Median: $F(r) = 1/2 \Rightarrow r = \sigma\sqrt{2\ln 2} \approx 1.1774\sigma$: **2.35 m**.
- Mean: $\mathbb{E}[r] = \int_0^\infty \frac{r^2}{\sigma^2}e^{-r^2/2\sigma^2}dr = \sigma\sqrt{\pi/2} \approx 1.2533\sigma$: **2.51 m**.

Mode < median < mean, as expected for a right-skewed distribution.

**(c)** CEP = median = 2.35 m. The 95% radius solves $1 - e^{-r^2/2\sigma^2} = 0.95$:
$r = \sigma\sqrt{-2\ln 0.05} \approx 2.4477\sigma =$ **4.90 m** (sometimes called R95).

**(d)** See the code. Marketing prefers CEP because it is the smallest of the
standard numbers (2.35 m versus 4.90 m for R95) and still sounds like a
guarantee; it actually says half of all fixes are worse than this. Navigation
engineers sizing a safety margin want R95 or higher, and integrity monitoring
cares about the far tail (protection levels at $10^{-7}$), where the Gaussian
assumption itself is suspect.

```python
import numpy as np

sigma = 2.0
rng = np.random.default_rng(1)
e = rng.normal(0.0, sigma, size=(100_000, 2))
r = np.hypot(e[:, 0], e[:, 1])

counts, edges = np.histogram(r, bins=200)
mode_mc = 0.5 * (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1])
print(f"mode   theory {sigma:.3f}  MC (histogram peak) {mode_mc:.3f}")
print(f"median theory {sigma*np.sqrt(2*np.log(2)):.3f}  MC {np.median(r):.3f}")
print(f"mean   theory {sigma*np.sqrt(np.pi/2):.3f}  MC {r.mean():.3f}")
print(f"R95    theory {sigma*np.sqrt(-2*np.log(0.05)):.3f}  MC {np.quantile(r, 0.95):.3f}")
```

Expected output (approximately): mode 2.0 (histogram peak is noisy, within about
0.1 m), median 2.355, mean 2.507, R95 4.895.

---

## Exercise 2: Fusing fixes on a coastal passage

Prior $\mathcal{N}(1200, 1600)$; $z_1 = 1260$, $R_1 = 900$; $z_2 = 1230$, $R_2 = 625$
(all in m and m$^2$).

**(a) $z_1$ then $z_2$.**

- $K_1 = 1600/2500 = 0.64$; $\mu = 1200 + 0.64(60) = 1238.4$ m; $\sigma^2 = 0.36(1600) = 576$ m$^2$ ($\sigma = 24.0$ m).
- $K_2 = 576/1201 = 0.4796$; $\mu = 1238.4 + 0.4796(1230 - 1238.4) = 1234.37$ m; $\sigma^2 = 0.5204(576) = 299.75$ m$^2$ (**$\sigma = 17.31$ m**).

**(b) $z_2$ then $z_1$.**

- $K = 1600/2225 = 0.7191$; $\mu = 1221.57$ m; $\sigma^2 = 449.44$ m$^2$.
- $K = 449.44/1349.44 = 0.3331$; $\mu = 1221.57 + 0.3331(38.43) = 1234.37$ m; $\sigma^2 = 299.75$ m$^2$.

Same answer. In information form the result is obviously order-independent:
precisions add ($1/1600 + 1/900 + 1/625 = 1/299.75$) and the posterior mean is
the precision-weighted average
$299.75\,(1200/1600 + 1260/900 + 1230/625) = 1234.37$ m. Order independence
requires the measurements to be conditionally independent given $x$; correlated
fixes (two radars seeing the same anomalous propagation) break it.

**(c)** Combined measurement: $R_c = (1/900 + 1/625)^{-1} = 368.85$ m$^2$,
$z_c = R_c(1260/900 + 1230/625) = 1242.30$ m. Fusing with the prior:
$K = 1600/1968.85 = 0.8127$, $\mu = 1200 + 0.8127(42.30) = 1234.37$ m,
$\sigma^2 = 0.1873(1600) = 299.75$ m$^2$. Identical, because the product of the
two likelihood Gaussians is itself (proportional to) the Gaussian
$\mathcal{N}(x; z_c, R_c)$ by the Gaussian product formula (`eq-probability-gaussian-product`).

**(d)** With $z_1 = x + v_1$: $\boldsymbol{\mu} = [1200, 1200]^T$,
$\boldsymbol{\Sigma} = \begin{bmatrix} 1600 & 1600 \\ 1600 & 2500 \end{bmatrix}$.
Conditioning: $\mu_{x|z} = 1200 + (1600/2500)(1260 - 1200) = 1238.4$ m and
$\sigma^2_{x|z} = 1600 - 1600^2/2500 = 576$ m$^2$, matching the first update.

```python
import numpy as np

def fuse(m1, v1, m2, v2):
    k = v1 / (v1 + v2)
    return m1 + k * (m2 - m1), (1 - k) * v1

m, v = fuse(*fuse(1200, 1600, 1260, 900), 1230, 625)
print("z1 then z2:", round(m, 2), round(v, 2), round(np.sqrt(v), 2))
m, v = fuse(*fuse(1200, 1600, 1230, 625), 1260, 900)
print("z2 then z1:", round(m, 2), round(v, 2))
Rc = 1 / (1/900 + 1/625); zc = Rc * (1260/900 + 1230/625)
print("combined  :", round(zc, 2), round(Rc, 2), [round(t, 2) for t in fuse(1200, 1600, zc, Rc)])
```

---

## Exercise 3: Correcting what you never measured

$\mathbf{P} = \begin{bmatrix} 4 & 1.5 \\ 1.5 & 1 \end{bmatrix}$, mean $[10, 2]^T$,
$\mathbf{H} = [1, 0]$, $R = 1$, $z = 12.5$.

**(a)** $\boldsymbol{\Sigma}_{xz} = \mathbf{P}\mathbf{H}^T = [4, 1.5]^T$,
$S = 4 + 1 = 5$, gain $\mathbf{K} = [0.8, 0.3]^T$, innovation $12.5 - 10 = 2.5$.

$$
\hat{\mathbf{x}}^+ = \begin{bmatrix} 10 + 0.8(2.5) \\ 2 + 0.3(2.5) \end{bmatrix} = \begin{bmatrix} 12.0 \\ 2.75 \end{bmatrix},
\qquad
\mathbf{P}^+ = \mathbf{P} - \mathbf{K}S\mathbf{K}^T = \begin{bmatrix} 4 - 3.2 & 1.5 - 1.2 \\ 1.5 - 1.2 & 1 - 0.45 \end{bmatrix} = \begin{bmatrix} 0.8 & 0.3 \\ 0.3 & 0.55 \end{bmatrix}.
$$

**(b)** Velocity standard deviation goes from 1.0 to $\sqrt{0.55} = 0.742$ m/s, a
26% reduction. It shrinks because velocity and the measurement are correlated
through the prior: $\rho_{vz} = 1.5/\sqrt{1 \cdot 5} = 0.671$, and for jointly
Gaussian variables $\sigma^2_{v|z} = \sigma_v^2(1 - \rho_{vz}^2) = 1(1 - 0.45) = 0.55$.
Physically, the prior correlation exists because position error accumulated from
velocity error; a position surprise is evidence about velocity.

**(c)** With a diagonal prior, $\boldsymbol{\Sigma}_{vz} = 0$, so $K_v = 0$: the
velocity mean and variance are unchanged. This is why a filter that zeros
off-diagonal covariance terms "for stability" quietly loses the ability to
estimate unmeasured states.

**(d)**

```python
import numpy as np

P = np.array([[4.0, 1.5], [1.5, 1.0]]); x = np.array([10.0, 2.0])
H = np.array([[1.0, 0.0]]); R = 1.0; z = 12.5
S = (H @ P @ H.T).item() + R
K = (P @ H.T / S).ravel()
print("analytic mean", x + K * (z - x[0]), "\nanalytic cov\n", P - np.outer(K, K) * S)

rng = np.random.default_rng(0)
xs = rng.multivariate_normal(x, P, size=2_000_000)
zs = xs[:, 0] + rng.normal(0, np.sqrt(R), size=xs.shape[0])
keep = np.abs(zs - z) < 0.02
print("kept", keep.sum(), "\nslice mean", xs[keep].mean(axis=0).round(3), "\nslice cov\n", np.cov(xs[keep].T).round(3))
```

---

## Exercise 4: There is no betting system

**(a)** The top line covers 5 of 38 pockets and pays 6 to 1:
$\mathbb{E}[W] = \frac{5}{38}(6) - \frac{33}{38}(1) = \frac{30 - 33}{38} = -\frac{3}{38} \approx -7.89\%$.
A "fair to the wheel" payout would be $36/5 - 1 = 6.2$ to 1; rounding down to 6
adds an extra 2.63% edge on top of the double zero.

**(b)** Simulation (the session ends when the next stake exceeds the bankroll,
or after 200 spins):

```python
import numpy as np

def martingale_sessions(n_sessions=20_000, bankroll0=255, max_spins=200, p_win=18/37, seed=5):
    rng = np.random.default_rng(seed)
    profits = np.zeros(n_sessions)
    wagered = np.zeros(n_sessions)
    for s in range(n_sessions):
        bankroll, stake, total = bankroll0, 1, 0
        wins = rng.random(max_spins) < p_win
        for w in wins:
            if stake > bankroll:
                break
            total += stake
            if w:
                bankroll += stake
                stake = 1
            else:
                bankroll -= stake
                stake *= 2
        profits[s] = bankroll - bankroll0
        wagered[s] = total
    return profits, wagered

profits, wagered = martingale_sessions()
print(f"fraction of sessions ahead : {np.mean(profits > 0):.3f}")
print(f"mean profit per session    : {profits.mean():+.2f} units")
print(f"mean amount wagered        : {wagered.mean():.1f} units")
print(f"-(1/37) * mean wagered     : {-wagered.mean()/37:+.2f} units")
d = profits + wagered / 37          # should average to zero (Wald's identity)
print(f"mean(profit + wagered/37)  : {d.mean():+.2f} +/- {d.std() / np.sqrt(d.size):.2f}")
print(f"winners average {profits[profits > 0].mean():+.1f}, losers average {profits[profits <= 0].mean():+.1f}")
```

Results with the seed above: about 62% of sessions end ahead, by roughly +96
units on average (one unit per completed doubling cycle). The other 38% hit a run
of 8 losses, cannot cover the next stake, and lose about 210 units on average.
The mean profit is about $-19$ units on a mean of about 685 units wagered, and
$\text{profit} + \text{wagered}/37$ averages to zero within its Monte Carlo
standard error of about 1 unit. Students' numbers will differ with the seed, but
should show the same pattern: most sessions win a little, a minority lose a lot,
and the mean is $-1/37$ of the action. (With 1000-spin sessions only about 16% of
sessions end ahead, because a run of 8 losses becomes very likely.)

**(c)** Let $s_n$ be the stake on spin $n$ and $X_n \in \{+1, -1\}$ the outcome,
with $\mathbb{E}[X_n] = -1/37$. The stake is chosen before the spin, from past
outcomes only, so $s_n$ is independent of $X_n$. Profit is $\sum_n s_n X_n$, and
by linearity of expectation

$$
\mathbb{E}\Big[\sum_n s_n X_n\Big] = \sum_n \mathbb{E}[s_n]\,\mathbb{E}[X_n] = -\frac{1}{37}\,\mathbb{E}\Big[\sum_n s_n\Big].
$$

(Formally this is Wald's identity / optional stopping; it needs bounded stakes
or a bounded number of spins, which the bankroll and table limits provide.) Any
staking rule only reshapes the *distribution* of outcomes, trading many small
wins for rare large losses in the martingale's case, without changing its mean.

---

## Exercise 5: Beating the crowd, not the track

Pool $P = \$100{,}000$, $\tau = 0.18$.

**(a)** $q = [0.40, 0.30, 0.20, 0.10]$. Payout per \$1 if the horse wins:
$(1-\tau)/q_i = [2.05, 2.733, 4.10, 8.20]$.

**(b)** Expected profit $0.82\,p_i/q_i - 1$ with $p = [0.35, 0.30, 0.25, 0.10]$:

| Horse | $p_i/q_i$ | Expected profit per \$1 |
|---|---|---|
| A | 0.875 | $-0.2825$ |
| B | 1.000 | $-0.18$ |
| C | 1.250 | $+0.025$ |
| D | 1.000 | $-0.18$ |

Only C clears the threshold $p/q > 1/0.82 = 1.2195$, by a thin margin. Note
that B and D, where the crowd is right, cost exactly the take; A, which the
crowd overrates, is worse than the take.

**(c)** Total expected payout
$= \sum_i P_i \cdot p_i\,(1-\tau)/q_i = (1-\tau)P\sum_i p_i = 0.82 \times 100{,}000 = \$82{,}000$,
independent of $p$ because $P_i/q_i = P$ for every horse.

**(d)** With \$5,000 more on C: pool \$105,000, $P_C = \$25{,}000$, payout per
\$1 $= 0.82 \times 105{,}000/25{,}000 = 3.444$, expected profit
$0.25 \times 3.444 - 1 = -0.139$. The bet destroyed its own edge. The break-even
bet $b$ solves $0.25 \times 0.82\,(100{,}000 + b)/(20{,}000 + b) = 1$, giving
$b \approx \$629$, and the expected profit is maximized near
$b \approx \$300$ for an expected gain of only a few dollars (see code). Edges
in a parimutuel market scale with pool size, which is why a serious operation
needs large pools, many races, and careful bet sizing (Kelly-style), not a single
big wager.

```python
import numpy as np

pools = np.array([40_000, 30_000, 20_000, 10_000.0]); tau = 0.18
p = np.array([0.35, 0.30, 0.25, 0.10])
q = pools / pools.sum()
print("q:", q, "\npayout per $1:", ((1 - tau) / q).round(3))
print("EV per $1:", ((1 - tau) * p / q - 1).round(4))
print("total expected payout:", np.sum(pools * p * (1 - tau) / q))

b = np.linspace(1, 3000, 3000)
ev = b * (0.25 * (1 - tau) * (pools.sum() + b) / (pools[2] + b) - 1)
print(f"break-even bet ~ ${b[np.argmin(np.abs(ev[100:])) + 100]:.0f}; best bet ~ ${b[np.argmax(ev)]:.0f} "
      f"with expected profit ${ev.max():.2f}")
b5k = 5000
print("EV per $1 with $5000 bet:", round(0.25 * (1 - tau) * (pools.sum() + b5k) / (pools[2] + b5k) - 1, 4))
```

---

## Exercise 6: Drift versus diffusion

**(a)** Per spin: mean $-\varepsilon$ with $\varepsilon = 1/37$, variance
$1 - \varepsilon^2$. After $n$ spins: mean $-n\varepsilon$, standard deviation
$\sqrt{n(1-\varepsilon^2)}$. Setting $n\varepsilon = c\sqrt{n(1-\varepsilon^2)}$ gives
$n = c^2(1-\varepsilon^2)/\varepsilon^2 = c^2(37^2 - 1)$:
$n^*_{1\sigma} = 1368$ spins and $n^*_{2\sigma} = 5472$ spins. At $n^*_{2\sigma}$
the bettor is ahead only if their luck is better than two standard deviations,
which (by the CLT) happens with probability about $P(Z > 2) \approx 2.3\%$; at a
typical 40 spins per hour that is about 140 hours of play.

**(b)** $N = 0.15^\circ/\sqrt{\text{h}}$, $b = 3^\circ/\text{h}$:
$t^* = (N/b)^2 = (0.05)^2$ h $= 0.0025$ h $= 9$ s. After 10 min ($1/6$ h):
random walk $1\sigma = 0.15\sqrt{1/6} = 0.061^\circ$; bias error
$3 \times 1/6 = 0.50^\circ$. The bias is eight times larger.

**(c)** Expected answer: the trajectories fan out like $\sqrt{t}$ around a mean
that drifts linearly at $3^\circ$/h. By about one minute the mean is more than
2.5 standard deviations from zero, so essentially every trajectory is on the
same side of zero. The heading error budget is set almost
entirely by the uncompensated bias. If the application needs, say, $1^\circ$
heading accuracy, this gyroscope alone is good for about 20 minutes; to do
better, the bias must be estimated (from GNSS course over ground, a
magnetometer, zero-velocity updates, or filter state augmentation), and it must
be re-estimated on a time scale shorter than that on which the bias itself
wanders. Reducing white noise would barely help.

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

eps = 1 / 37
for c in (1, 2):
    print(f"n*({c} sigma) = {c**2 * (1 - eps**2) / eps**2:.0f} spins")

N_arw = 0.15 / 60.0          # deg/sqrt(s)  (0.15 deg/sqrt(h))
bias = 3.0 / 3600.0          # deg/s        (3 deg/h)
print(f"t* = {(N_arw / bias)**2:.1f} s")
T = 600.0
print(f"after 10 min: ARW 1-sigma {N_arw*np.sqrt(T):.3f} deg, bias error {bias*T:.3f} deg")

dt, n_traj = 0.01, 200
n_steps = int(T / dt)
rng = np.random.default_rng(6)
t = np.arange(1, n_steps + 1) * dt
heading_err = np.zeros((n_traj, n_steps))
for i in range(n_traj):   # loop over trajectories keeps memory modest
    increments = bias * dt + N_arw * np.sqrt(dt) * rng.standard_normal(n_steps)
    heading_err[i] = np.cumsum(increments)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t[::100], heading_err[:, ::100].T, color="0.7", linewidth=0.5)
ax.plot(t, bias * t, "C3", label="mean (bias)")
ax.fill_between(t, bias * t - 2 * N_arw * np.sqrt(t), bias * t + 2 * N_arw * np.sqrt(t), color="C3", alpha=0.2, label="±2 sigma")
ax.set_xlabel("time (s)"); ax.set_ylabel("heading error (deg)"); ax.legend()
fig.savefig("ex6_heading.png", dpi=80)
print("final error mean / std:", heading_err[:, -1].mean().round(3), heading_err[:, -1].std().round(3))
```
