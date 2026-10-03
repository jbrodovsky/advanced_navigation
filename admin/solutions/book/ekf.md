# Solutions: The Extended Kalman Filter (`ekf.qmd`)

Instructor-only. Exercise numbers match `## Exercises {#sec-ekf-exercises}`.
Code snippets assume the chapter's cells have been run.

## Exercise 1: Range-bearing Jacobian

With $\delta_x = m_x - x$, $\delta_y = m_y - y$, $q = \delta_x^2 + \delta_y^2$,
$r = \sqrt{q}$:

- Range: $\partial r/\partial x = -\delta_x/r$, $\partial r/\partial y = -\delta_y/r$,
  $\partial r/\partial\theta = 0$.
- Bearing $\beta = \operatorname{atan2}(\delta_y, \delta_x) - \theta$. Using
  $\partial \operatorname{atan2}(v, u) = (u\,dv - v\,du)/(u^2 + v^2)$ with
  $du = -dx$, $dv = -dy$:
  $\partial\beta/\partial x = \delta_y/q$, $\partial\beta/\partial y = -\delta_x/q$,
  and $\partial\beta/\partial\theta = -1$ because the bearing is measured
  relative to the robot's heading.

$$
\mathbf{H} =
\begin{bmatrix}
-\delta_x/r & -\delta_y/r & 0 \\
\delta_y/q & -\delta_x/q & -1
\end{bmatrix}.
$$

The position part of the bearing row has magnitude
$\sqrt{\delta_y^2 + \delta_x^2}/q = 1/r$. A position error $\delta p$
perpendicular to the line of sight changes the bearing by about
$\delta p / r$ radians, and the neglected second-order term is of order
$(\delta p/r)^2$. At $r = 1$ m a 0.5 m position uncertainty means bearing
changes of about $27^\circ$ across the uncertainty region and a badly curved
measurement function, so a linearized update can be far off. The key word is
*uncertainty*: what matters is $\sigma_p/r$, not $r$ alone. A converged filter
with $\sigma_p \approx 7$ cm can pass 1 m from a landmark without trouble
(students who try it in the example will see only a small NEES bump); a filter
with meter-level uncertainty, or a true path that passes nearly *over* the
landmark ($r \to 0$, bearing undefined), cannot. Practical mitigations: inflate
the bearing variance by about $(\sigma_p/r)^2$, or ignore bearings inside a
minimum range.

## Exercise 2: The IEKF is Gauss-Newton

Write $\bar{\mathbf{x}} = \hat{\mathbf{x}}_{k|k-1}$, $\bar{\mathbf{P}} = \mathbf{P}_{k|k-1}$.
Linearize $h(\mathbf{x}) \approx h(\mathbf{x}_i) + \mathbf{H}_i(\mathbf{x} - \mathbf{x}_i)$.
The cost becomes quadratic:

$$
J(\mathbf{x}) \approx \tfrac12(\mathbf{x} - \bar{\mathbf{x}})^T\bar{\mathbf{P}}^{-1}(\mathbf{x} - \bar{\mathbf{x}})
+ \tfrac12(\mathbf{r}_i - \mathbf{H}_i(\mathbf{x} - \mathbf{x}_i))^T\mathbf{R}^{-1}(\mathbf{r}_i - \mathbf{H}_i(\mathbf{x} - \mathbf{x}_i)),
\qquad \mathbf{r}_i = \mathbf{z} - h(\mathbf{x}_i).
$$

Setting the gradient to zero and writing $\mathbf{x} = \bar{\mathbf{x}} + \boldsymbol{\delta}$:

$$
(\bar{\mathbf{P}}^{-1} + \mathbf{H}_i^T\mathbf{R}^{-1}\mathbf{H}_i)\boldsymbol{\delta}
= \mathbf{H}_i^T\mathbf{R}^{-1}\left[\mathbf{r}_i - \mathbf{H}_i(\bar{\mathbf{x}} - \mathbf{x}_i)\right].
$$

The push-through form of the matrix inversion lemma gives
$(\bar{\mathbf{P}}^{-1} + \mathbf{H}^T\mathbf{R}^{-1}\mathbf{H})^{-1}\mathbf{H}^T\mathbf{R}^{-1}
= \bar{\mathbf{P}}\mathbf{H}^T(\mathbf{H}\bar{\mathbf{P}}\mathbf{H}^T + \mathbf{R})^{-1} = \mathbf{K}_i$.
(Verify by multiplying both sides on the left by
$\bar{\mathbf{P}}^{-1} + \mathbf{H}^T\mathbf{R}^{-1}\mathbf{H}$ and on the right
by $\mathbf{H}\bar{\mathbf{P}}\mathbf{H}^T + \mathbf{R}$.) Hence

$$
\mathbf{x}_{i+1} = \bar{\mathbf{x}} + \mathbf{K}_i\left[\mathbf{z} - h(\mathbf{x}_i) - \mathbf{H}_i(\bar{\mathbf{x}} - \mathbf{x}_i)\right],
$$

which is the IEKF iteration. With $\mathbf{x}_0 = \bar{\mathbf{x}}$ the bracket is
the ordinary innovation and $\mathbf{K}_0$ is the EKF gain, so the first
iteration is the EKF update. Because this is Gauss-Newton (no second
derivatives of $h$, full step), it can overshoot when the residual is large or
$h$ is strongly curved, which motivates the line search.

## Exercise 3: Reset Jacobian sign

**Navigation-frame error** (the book's convention): truth
$\mathbf{C} = \operatorname{Exp}(\delta\boldsymbol\theta)\hat{\mathbf{C}}$. Injection
$\hat{\mathbf{C}}^+ = \operatorname{Exp}(\delta\hat{\boldsymbol\theta})\hat{\mathbf{C}}$.
New error:
$\operatorname{Exp}(\delta\boldsymbol\theta^+) = \mathbf{C}(\hat{\mathbf{C}}^+)^T
= \operatorname{Exp}(\delta\boldsymbol\theta)\operatorname{Exp}(-\delta\hat{\boldsymbol\theta})$.
With $\mathbf{a} = \delta\boldsymbol\theta$, $\mathbf{b} = -\delta\hat{\boldsymbol\theta}$:

$$
\delta\boldsymbol\theta^+ \approx \delta\boldsymbol\theta - \delta\hat{\boldsymbol\theta}
- \tfrac12\delta\boldsymbol\theta\times\delta\hat{\boldsymbol\theta}
= \delta\boldsymbol\theta - \delta\hat{\boldsymbol\theta} + \tfrac12[\delta\hat{\boldsymbol\theta}]_\times\delta\boldsymbol\theta,
\qquad
\frac{\partial\delta\boldsymbol\theta^+}{\partial\delta\boldsymbol\theta} = \mathbf{I} + \tfrac12[\delta\hat{\boldsymbol\theta}]_\times.
$$

**Body-frame error** (Solà): truth $\mathbf{C} = \hat{\mathbf{C}}\operatorname{Exp}(\delta\boldsymbol\theta)$,
injection $\hat{\mathbf{C}}^+ = \hat{\mathbf{C}}\operatorname{Exp}(\delta\hat{\boldsymbol\theta})$, so
$\operatorname{Exp}(\delta\boldsymbol\theta^+) = (\hat{\mathbf{C}}^+)^T\mathbf{C}
= \operatorname{Exp}(-\delta\hat{\boldsymbol\theta})\operatorname{Exp}(\delta\boldsymbol\theta)$,
giving $\delta\boldsymbol\theta^+ \approx \delta\boldsymbol\theta - \delta\hat{\boldsymbol\theta}
- \tfrac12[\delta\hat{\boldsymbol\theta}]_\times\delta\boldsymbol\theta$ and
$\mathbf{I} - \tfrac12[\delta\hat{\boldsymbol\theta}]_\times$, Solà's sign.

**Size.** The spectral norm of $[\mathbf{a}]_\times$ is $\|\mathbf{a}\|$, so
$\|\mathbf{G} - \mathbf{I}\| = \tfrac12\|\delta\hat{\boldsymbol\theta}\|$:
$0.0044$ for a $0.5^\circ$ correction and $0.087$ for $10^\circ$. The reset
rotates the attitude covariance by half the correction angle. For a $0.5^\circ$
correction that is a 0.4% effect, far below typical tuning uncertainty, so
skipping it is defensible. For $10^\circ$ it is a 9% effect on the attitude
cross-covariances, which matters, especially for heading after a GNSS outage or
during in-motion alignment. Rule of thumb: include it whenever corrections can
exceed a degree or two; it costs one 3 x 3 skew matrix.

## Exercise 4: Iterate the landmark updates

A general IEKF update for the landmark model (residual and heading wrapped):

```python
def iekf_update_rb(x_prior, P_prior, z, m, R, max_iter=30, tol=1e-9):
    def resid(x):
        nu = z - h_range_bearing(x, m); nu[1] = wrap(nu[1]); return nu
    def dprior(x):
        d = x - x_prior; d[2] = wrap(d[2]); return d
    def cost(x):
        d, r = dprior(x), resid(x)
        return 0.5 * d @ np.linalg.solve(P_prior, d) + 0.5 * r @ np.linalg.solve(R, r)
    x_i = x_prior.copy()
    for _ in range(max_iter):
        H = H_range_bearing(x_i, m)
        S = H @ P_prior @ H.T + R
        K = np.linalg.solve(S, H @ P_prior).T
        step = K @ (resid(x_i) + H @ dprior(x_i)) - dprior(x_i)   # x_GN - x_i
        alpha, c0 = 1.0, cost(x_i)
        while alpha > 1e-4 and cost(x_i + alpha * step) > c0:
            alpha *= 0.5
        x_i = x_i + alpha * step
        x_i[2] = wrap(x_i[2])
        if np.linalg.norm(alpha * step) < tol:
            break
    I_KH = np.eye(3) - K @ H                     # last linearization
    return x_i, I_KH @ P_prior @ I_KH.T + K @ R @ K.T
```

(`x_GN - x_i` is written with `dprior` so the heading difference is wrapped;
it equals $\bar{\mathbf{x}} + \mathbf{K}[\mathbf{r}_i - \mathbf{H}(\bar{\mathbf{x}} - \mathbf{x}_i)] - \mathbf{x}_i$.)
Note the covariance must use the gain and Jacobian of the last linearization;
with `max_iter=1` this reproduces the chapter's EKF exactly, which is a good
unit test.

Expected ANEES (40 runs, seed 1, same as the chapter):

| Time | 1 s | 2 s | 5 s | 10 s | 20 s | 40 s |
|---|---|---|---|---|---|---|
| EKF | 95.4 | 32.4 | 9.1 | 4.3 | 2.7 | 2.9 |
| IEKF | 5.5 | 3.6 | 3.4 | 3.0 | 2.7 | 2.9 |

(95% bounds for $n = 3$, $M = 40$: [2.29, 3.81].) Iteration removes almost all
of the initial overconfidence: it came from linearizing the first precise
range-bearing measurements at a poor estimate. What remains (ANEES about 5 at
1 s) comes from the prediction linearization with a $40^\circ$ heading
uncertainty and from the posterior genuinely not being Gaussian at first.
Cost: about 15 evaluations of $h$ per update with the line search (counting
cost-function evaluations) versus 1 for the EKF, i.e., roughly five Gauss-Newton
iterations, and only during the first few seconds; once converged, the IEKF
stops after one or two iterations.

## Exercise 5: Acoustic range-only positioning

Use the same IEKF function with a scalar range model
($h = \|\mathbf{m} - \mathbf{p}\|$, $\mathbf{H} = [-\delta_x/r, -\delta_y/r, 0]$,
no angle wrapping of the residual), the three transponders always in range,
$\sigma_r = 0.5$ m, and $P_0 = \operatorname{diag}(10^2, 10^2, (40^\circ)^2)$.

Expected (40 runs, seed 1):

| Time | 1 s | 2 s | 5 s | 10 s | 20 s | 40 s |
|---|---|---|---|---|---|---|
| EKF ANEES | 120 | 102 | 110 | 615 | 1626 | 367 |
| EKF median NEES | 19.8 | 16.3 | 9.5 | 6.6 | 3.1 | 2.5 |
| IEKF ANEES | 117 | 87 | 53 | 38 | 9.8 | 2.9 |
| IEKF median NEES | 14.3 | 10.9 | 11.0 | 6.4 | 3.3 | 2.5 |

Interpretation:

- The IEKF becomes consistent by about 40 s; the EKF's *average* never does,
  because two of the 40 EKF runs diverge (final NEES above 20) and dominate the
  mean, while the median EKF run looks fine. This is exactly why Monte Carlo
  averages matter: a single "typical" run would have hidden the failure mode.
  No IEKF run diverged.
- Both filters start badly inconsistent, and iteration helps much less early on
  than in Exercise 4. With ranges only, heading is not measured at all: it
  becomes observable only through motion, as the vehicle's displacement over
  several updates reveals which way it is pointing. Until then the large
  heading error lives in the *prediction*, which the IEKF does not relinearize.
  Range-bearing measurements observe heading directly (the $-1$ in the
  Jacobian), so they pin it down in a single update.
- The 10 m initial uncertainty is also comparable to the transponder spacing,
  so the early range circles intersect in strongly curved, sometimes nearly
  ambiguous geometry. LBL systems in practice initialize from a surface GNSS
  fix or a coarse grid search before handing off to an EKF.
