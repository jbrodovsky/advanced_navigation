# Solutions: The Unscented Kalman Filter (`ukf.qmd`)

Instructor-only. Notation follows the chapter: spread $s = \alpha\sqrt{n+\kappa}$,
outer weights $W = 1/(2s^2)$, $W_0^{(m)} = 1 - n/s^2 = 1 - 2nW$,
$W_0^{(c)} = W_0^{(m)} + 1 - \alpha^2 + \beta$.

## Exercise 1: The covariance identity

Let $\mathbf{d}_i = \mathcal{Y}^{(i)} - \mathcal{Y}^{(0)}$ for $i = 1, \ldots, 2n$.

**Mean.** Because the mean weights sum to one,

$$
\boldsymbol{\mu}_y = W_0^{(m)}\mathcal{Y}^{(0)} + W\sum_{i\ge1}\mathcal{Y}^{(i)}
= \mathcal{Y}^{(0)}\Big(W_0^{(m)} + 2nW\Big) + W\sum_{i\ge 1}\mathbf{d}_i
= \mathcal{Y}^{(0)} + W\sum_{i \ge 1}\mathbf{d}_i,
$$

so $\bar{\mathbf{m}} = \boldsymbol{\mu}_y - \mathcal{Y}^{(0)} = W\sum_{i\ge1}\mathbf{d}_i$.

**Covariance.** The residuals are $\mathcal{Y}^{(i)} - \boldsymbol{\mu}_y = \mathbf{d}_i - \bar{\mathbf{m}}$
for $i \ge 1$ and $\mathcal{Y}^{(0)} - \boldsymbol{\mu}_y = -\bar{\mathbf{m}}$. Then

$$
\begin{aligned}
\mathbf{P}_y
&= W_0^{(c)}\bar{\mathbf{m}}\bar{\mathbf{m}}^T + W\sum_{i\ge1}(\mathbf{d}_i - \bar{\mathbf{m}})(\mathbf{d}_i - \bar{\mathbf{m}})^T \\
&= W\sum_{i\ge1}\mathbf{d}_i\mathbf{d}_i^T
 - \Big(W\sum_i\mathbf{d}_i\Big)\bar{\mathbf{m}}^T - \bar{\mathbf{m}}\Big(W\sum_i\mathbf{d}_i\Big)^T
 + \big(2nW + W_0^{(c)}\big)\bar{\mathbf{m}}\bar{\mathbf{m}}^T \\
&= W\sum_{i\ge1}\mathbf{d}_i\mathbf{d}_i^T + \big(-2 + 2nW + W_0^{(c)}\big)\bar{\mathbf{m}}\bar{\mathbf{m}}^T.
\end{aligned}
$$

Now $2nW + W_0^{(c)} = (1 - W_0^{(m)}) + W_0^{(m)} + 1 - \alpha^2 + \beta = 2 - \alpha^2 + \beta$,
so the coefficient is $\beta - \alpha^2$, which is the chapter's identity.

**Positive semidefiniteness.** $W = 1/(2s^2) > 0$ whenever $s^2 = \alpha^2(n+\kappa) > 0$,
so the first term is a positive combination of outer products. The second term
is PSD iff $\beta - \alpha^2 \ge 0$. Hence $\mathbf{P}_y \succeq 0$ whenever
$\beta \ge \alpha^2$ (and $n + \kappa > 0$), regardless of the sign of $W_0^{(c)}$.

**Counterexample for the unscaled transform** ($\alpha = 1$, $\beta = 0$, so the
second coefficient is $-1$). Take $n = 1$, $\kappa = -0.5$ (so $s^2 = 0.5$),
$x \sim \mathcal{N}(0, \sigma^2)$, $y = x^2$. The points are $0, \pm s\sigma$, so
$d_\pm = s^2\sigma^2 = 0.5\sigma^2$ and $\bar{m} = W(d_+ + d_-) = \sigma^2$. Then

$$
P_y = \frac{1}{2s^2}\,2\,(s^2\sigma^2)^2 - \bar{m}^2 = s^2\sigma^4 - \sigma^4 = -0.5\,\sigma^4 < 0.
$$

Any $s^2 < 1$ (that is, $-1 < \kappa < 0$) gives a negative variance.

## Exercise 2: How big should beta be?

Let $x \sim \mathcal{N}(\mu, \sigma^2)$, $y = x^2$, $n = 1$, $s^2 = \alpha^2(1+\kappa)$.
Points: $\mu$, $\mu \pm s\sigma$. Transformed: $\mathcal{Y}^{(0)} = \mu^2$,
$\mathcal{Y}^{(\pm)} = \mu^2 \pm 2\mu s\sigma + s^2\sigma^2$.

**Mean.** $\mu_y = W_0\mu^2 + W(2\mu^2 + 2s^2\sigma^2) = \mu^2(W_0 + 2W) + 2Ws^2\sigma^2 = \mu^2 + \sigma^2$,
since $W_0 + 2W = 1$ and $2Ws^2 = 1$. This equals $\mathbb{E}[x^2]$ for every
$\alpha, \kappa$: the scaled UT is exact for the mean of a quadratic.

**Variance** (use Exercise 1). $d_\pm = \pm 2\mu s\sigma + s^2\sigma^2$,
$\sum d_\pm^2 = 8\mu^2 s^2\sigma^2 + 2 s^4\sigma^4$, so
$W\sum d^2 = 4\mu^2\sigma^2 + s^2\sigma^4$, and $\bar{m} = \sigma^2$. Hence

$$
P_y = 4\mu^2\sigma^2 + (s^2 - \alpha^2 + \beta)\sigma^4.
$$

The true variance is $\mathbb{E}[x^4] - (\mathbb{E}[x^2])^2 = (\mu^4 + 6\mu^2\sigma^2 + 3\sigma^4) - (\mu^2 + \sigma^2)^2 = 4\mu^2\sigma^2 + 2\sigma^4$.
So the UT is exact iff $s^2 - \alpha^2 + \beta = 2$.

- $\alpha \to 0$ with $\kappa$ fixed: $s^2 - \alpha^2 = \alpha^2\kappa \to 0$, so $\beta = 2$. This is the
  origin of the "beta = 2 is optimal for Gaussians" rule: it is optimal in the
  small-spread limit.
- $\alpha = 1$, $\kappa = 3 - n$ (so $s^2 = 3$): $3 - 1 + \beta = 2$, so $\beta = 0$.

The condition depends only on the spread along the direction of curvature, so
the same reasoning applies along one axis of the two-state range-bearing
example. There $s^2 = 3$ ($\alpha = 1$, $\kappa = 1$, $n = 2$); the sweep shows
$\beta = 0$ matching the Monte Carlo north variance (about 86 vs 84 m$^2$) and
$\beta = 2$ inflating it (about 156 m$^2$). Larger $\beta$ only adds
$(\beta - \alpha^2)\bar{\mathbf{m}}\bar{\mathbf{m}}^T$, so it errs on the
conservative side.

## Exercise 3: Redraw or reuse?

**Setup.** $\mathbf{F} = \begin{bmatrix}1 & \Delta t\\ 0 & 1\end{bmatrix}$,
$\mathbf{Q} = q\begin{bmatrix}\Delta t^3/3 & \Delta t^2/2\\ \Delta t^2/2 & \Delta t\end{bmatrix}$,
$\mathbf{H} = [1\ 0]$, $R = 1$, $\Delta t = 1$, $\mathbf{P}_{0|0} = \mathbf{I}$, 200 runs of 100 steps.
Use `ukf_predict`/`ukf_update` for the redrawn filter and `ukf_step_shortcut`
(with this $\mathbf{Q}$, $R$) for the shortcut. ANEES at each step is the
average over runs of $\mathbf{e}^T\mathbf{P}_{k|k}^{-1}\mathbf{e}$ with
$\mathbf{e} = \mathbf{x} - \hat{\mathbf{x}}_{k|k}$. For $n_x = 2$ and $N = 200$
runs, the 95% band is
$[\chi^2_{0.025}(400), \chi^2_{0.975}(400)]/200 \approx [1.73, 2.28]$.

```python
import numpy as np
from scipy.stats import chi2

lo, hi = chi2.ppf([0.025, 0.975], 2 * 200) / 200
# for q in (0.01, 0.1, 1, 10): build Q, run both filters on the same simulated
# data, accumulate NEES per step, average over runs, plot against (lo, hi).
```

**Expected result.** The redrawn UKF is the Kalman filter (to rounding error),
so its ANEES sits inside the band for every $q$. The shortcut's ANEES falls
*below* the band, increasingly so as $q$ grows; at $q = 0.01$ the two are
nearly indistinguishable.

**Why (scalar insight).** For a random walk $x_k = x_{k-1} + w_k$, $z_k = x_k + v_k$,
the shortcut's propagated points have spread $P$ (its own posterior), so it
uses $S = P + R$ and $K = P/(P+R)$ while reporting
$P_{\text{new}} = P + Q - P^2/(P+R)$. Its steady state solves
$P^2 = Q(P + R)$, which is exactly the steady-state Kalman *prior*
$\bar{P}^\star$; its gain $\bar{P}^\star/(\bar{P}^\star + R)$ is the optimal
gain. So in this case the estimates are optimal but the reported covariance is
the prior, too large by $1 + \bar{P}^\star/R$; with $Q = R$ the NEES is
$P^\star/\bar{P}^\star = 0.618/1.618 \approx 0.38$ instead of 1. In general
(including the CV model) the gain is also wrong during transients and when
$\mathbf{F} \ne \mathbf{I}$, and the mismatch grows with
$\mathbf{Q}$ relative to $\mathbf{P}_{k|k-1}$.

## Exercise 4: A buoy abeam

Sketch:

```python
buoy = np.array([0.0, 0.0])                         # north, east
dt, n_steps = 1.0, 120
F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1.0]])
Q = 0.05 * np.diag([dt**3 / 3, dt**3 / 3, dt, dt])  # plus off-diagonals if desired
R = np.diag([2.0**2, np.deg2rad(1.0) ** 2])

def h_rb(x):
    d = buoy - x[:2]
    return np.array([np.hypot(*d), np.arctan2(d[1], d[0])])

# truth: start at (50, 300) north/east, velocity (0, -5): passes 50 m north of the buoy at t = 60 s
z_ops = make_angle_ops([1])
# run ukf_predict / ukf_update twice: once with z_ops=EUCLIDEAN, once with z_ops=z_ops
```

**Expected result.** Bearings start near $-100^\circ$ (the buoy is ahead and to
port, roughly west by south), pass through $\pm 180^\circ$ at closest approach
($t \approx 60$ s), where the buoy is due south, and end near $+100^\circ$. With angle-aware residuals the filter tracks smoothly and the bearing
innovations stay within a few degrees. Without them, as soon as the predicted
sigma-point bearings straddle $\pm 180^\circ$ (a few seconds before closest
approach, depending on the position uncertainty), $\hat{\mathbf{z}}$ and
$\mathbf{S}$ are corrupted: the mean bearing jumps toward $0^\circ$ and the
bearing variance balloons, so the filter effectively ignores bearings; and once
the measured bearing wraps but the prediction does not, the innovation is near
$\pm 360^\circ$, which drags the position estimate badly off. The position error
plot shows a sharp jump at that step. Students should identify the step at which
the sigma points first straddle the wrap, not just where the truth crosses it.

## Exercise 5: One vector is not enough

**Unobservability.** Let $\hat{\mathbf{m}} = \mathbf{m}^n/\|\mathbf{m}^n\|$ and
$\mathbf{q}' = \operatorname{Exp}(\phi\hat{\mathbf{m}}) \otimes \mathbf{q}$. Then
$\mathbf{C}(\mathbf{q}') = \mathbf{C}_\phi\mathbf{C}(\mathbf{q})$, where
$\mathbf{C}_\phi$ rotates about $\hat{\mathbf{m}}$, and

$$
h(\mathbf{q}') = \mathbf{C}(\mathbf{q})^T\mathbf{C}_\phi^T\mathbf{m}^n = \mathbf{C}(\mathbf{q})^T\mathbf{m}^n = h(\mathbf{q}),
$$

because a rotation about $\hat{\mathbf{m}}$ leaves $\mathbf{m}^n$ unchanged. So
the navigation-frame error direction $\hat{\mathbf{m}}$ is unobservable from the
magnetometer alone. With $\mathbf{m}^n \propto [0.44, 0, 0.90]$ this direction is
mostly "down" (heading) with a north component (roll about north). In the
magnetometer-only posterior, the eigenvector of $\mathbf{P}_{k|k}$ with the
largest eigenvalue is close to $\pm\hat{\mathbf{m}}$ and its variance is barely
reduced from the prior's projection
$\hat{\mathbf{m}}^T\mathbf{P}\hat{\mathbf{m}} \approx 0.19\cdot 3^2 + 0.81\cdot 10^2 \approx 82\ \text{deg}^2$
(about 9 degrees), while the two orthogonal directions shrink to around a degree.
Gravity is not parallel to $\mathbf{m}^n$, which is why the two-vector update
observes all three angles (as long as the field is not vertical, which is the
classic reason magnetic heading degrades near the magnetic poles).

**Consistency vs prior width.** Monte Carlo: draw
$\delta\boldsymbol{\theta} \sim \mathcal{N}(\mathbf{0}, \mathbf{P})$,
$\mathbf{q}_\text{true} = \operatorname{Exp}(\delta\boldsymbol{\theta}) \otimes \hat{\mathbf{q}}$,
simulate the two-vector measurement, run `ukf_update(..., x_ops=ATTITUDE)`, and
compute $\mathbf{e}^T\mathbf{P}_{k|k}^{-1}\mathbf{e}$ with
$\mathbf{e} = \mathbf{q}_\text{true} \boxminus \hat{\mathbf{q}}_{k|k}$. A reference
run with 400 trials (roll and pitch prior 3 degrees) gave average NEES of about
3.2, 3.5, 3.9, and 5.6 for heading priors of 5, 10, 15, and 25 degrees
(expected value 3; the 95% band for 400 trials is roughly [2.76, 3.24]).

**Explanation.** A single Gaussian update is a statistical linearization over the
*prior* spread. When the measurement is far more precise than the prior, the
posterior occupies a small region, and the curvature of $h$ across the wide prior
displaces the update's mean by an amount its covariance does not account for. The
error grows with the prior width, so the filter becomes overconfident.

**Fixes at start-up.** (1) Initialize attitude deterministically from the two
vectors (TRIAD or QUEST), so the filter starts with a small prior; (2) iterate
the update (an iterated UKF/EKF relinearizes about the posterior); (3) anneal:
apply the first few updates with inflated $\mathbf{R}$, shrinking it as the
attitude converges; (4) as a crude last resort, inflate the posterior covariance
after large updates. (1) is what most AHRS units do.
