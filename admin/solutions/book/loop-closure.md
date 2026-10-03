# Solutions: Loop Closure and Long-Term Consistency (`loop-closure.qmd`)

Instructor-only. All numbers below come from the chapter's simulation (seed 7,
51 poses, 1.25 laps of a 10 m square, odometry noise
$\sigma = (0.05\text{ m}, 0.05\text{ m}, 1^\circ)$ plus an unmodeled
$0.6^\circ$/step heading bias, loop closure $(40, 0)$ with
$\sigma = (0.05, 0.05, 0.5^\circ)$, false closure $(48, 2)$). The code snippets
assume the chapter's cells have been run. Pose 0 is held fixed throughout.

## Exercise 1: Pose-graph Jacobians

**(a)** With $\boldsymbol{\tau} = \mathbf{C}_z^T\left[\mathbf{C}_i^T(\mathbf{t}_j - \mathbf{t}_i) - \mathbf{t}_z\right]$
and $\phi = \operatorname{wrap}(\theta_j - \theta_i - \theta_z)$:

- $\partial\boldsymbol{\tau}/\partial\mathbf{t}_i = -\mathbf{C}_z^T\mathbf{C}_i^T$ and
  $\partial\boldsymbol{\tau}/\partial\mathbf{t}_j = \mathbf{C}_z^T\mathbf{C}_i^T$
  (the expression is linear in $\mathbf{t}_j - \mathbf{t}_i$);
- $\partial\boldsymbol{\tau}/\partial\theta_i = \mathbf{C}_z^T\,(\partial\mathbf{C}_i^T/\partial\theta_i)\,(\mathbf{t}_j - \mathbf{t}_i)$,
  with $\partial\mathbf{C}(\theta)^T/\partial\theta = \begin{bmatrix} -\sin\theta & \cos\theta \\ -\cos\theta & -\sin\theta \end{bmatrix}$;
  $\partial\boldsymbol{\tau}/\partial\theta_j = \mathbf{0}$;
- $\partial\phi/\partial\theta_i = -1$, $\partial\phi/\partial\theta_j = +1$
  (away from the wrap point, where $\phi$ jumps by $2\pi$ but the derivative
  is unchanged), and $\phi$ does not depend on the translations.

Stacking these gives $\mathbf{A}_{ij}$ and $\mathbf{B}_{ij}$ of
@eq-loop-closure-jacobians. Note $\mathbf{B}_{ij}$ does not depend on
$\theta_j$: pose $j$'s heading enters only through $\phi$.

**(b)**

```python
rng = np.random.default_rng(0)
worst, eps = 0.0, 1e-6
for _ in range(1000):
    pi = np.append(rng.uniform(-20, 20, 2), rng.uniform(-np.pi, np.pi))
    pj = np.append(rng.uniform(-20, 20, 2), rng.uniform(-np.pi, np.pi))
    z = np.append(rng.uniform(-5, 5, 2), rng.uniform(-np.pi, np.pi))
    e0, A, B = edge_residual_and_jacobians(pi, pj, z)
    if abs(abs(e0[2]) - np.pi) < 1e-3:   # a central difference would straddle the wrap
        continue
    f = lambda a, b: edge_residual_and_jacobians(a, b, z)[0]
    A_num = np.column_stack([(f(pi + d, pj) - f(pi - d, pj)) / (2 * eps) for d in eps * np.eye(3)])
    B_num = np.column_stack([(f(pi, pj + d) - f(pi, pj - d)) / (2 * eps) for d in eps * np.eye(3)])
    worst = max(worst, np.abs(A_num - A).max(), np.abs(B_num - B).max())
print(worst)   # about 1e-8
```

The worst disagreement is about $10^{-8}$, finite-difference roundoff. Students
who do not skip the samples next to $\phi = \pm\pi$ will see occasional
disagreements of order $\pi/\epsilon$ in the heading row: that is the finite
difference jumping across the wrap, not a Jacobian error, and is worth a
sentence in their answer.

**(c)** Expand $\sin\phi/\phi = 1 - \phi^2/6 + O(\phi^4)$ and
$(1 - \cos\phi)/\phi = \phi/2 + O(\phi^3)$, so
$\mathbf{V}(\phi) = \mathbf{I} + \tfrac{\phi}{2}\mathbf{J} + O(\phi^2)$ with
$\mathbf{J} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$, and
$\mathbf{V}^{-1} = \mathbf{I} - \tfrac{\phi}{2}\mathbf{J} + O(\phi^2)$. Hence

$$
\operatorname{Log}(\mathbf{E}) - [\boldsymbol{\tau};\ \phi]
= \begin{bmatrix} (\mathbf{V}^{-1} - \mathbf{I})\boldsymbol{\tau} \\ 0 \end{bmatrix}
= \begin{bmatrix} -\tfrac{\phi}{2}\mathbf{J}\boldsymbol{\tau} + O(\phi^2\|\boldsymbol{\tau}\|) \\ 0 \end{bmatrix},
$$

whose size is about $|\phi|\,\|\boldsymbol{\tau}\|/2$: a product of two
error quantities, so second order. At the solution the largest leftover
rotations are a fraction of a degree ($\sim 10^{-2}$ rad) and the leftover
translations a few millimeters, so the gap should be of order
$10^{-2} \times 3\times10^{-3}/2 \approx 10^{-5}$ m. The chapter prints
$1.2\times10^{-5}$: consistent.

## Exercise 2: Gauge freedom

**(a)** `np.linalg.eigvalsh(H.toarray())` gives three eigenvalues of order
$10^{-13}$, then 1.8, 3.9, ...; the largest is about $3.2\times10^4$. Three
are zero to numerical precision.

**(b)** $\mathbf{H} = \sum \mathbf{J}_k^T\boldsymbol{\Omega}_k\mathbf{J}_k$ is
positive semidefinite, so $\mathbf{H}\mathbf{v} = \mathbf{0}$ iff
$\mathbf{J}_k\mathbf{v} = \mathbf{0}$ for every edge, i.e. iff moving the
poses along $\mathbf{v}$ leaves every residual unchanged to first order.

- Translation by $\boldsymbol{\delta}$: $\mathbf{t}_j - \mathbf{t}_i$ and
  $\theta_j - \theta_i$ are unchanged, so $\mathbf{v}_x = [1, 0, 0, 1, 0, 0, \ldots]$
  and $\mathbf{v}_y = [0, 1, 0, 0, 1, 0, \ldots]$ are null vectors.
- Rotation of everything about the world origin by $\delta$:
  $\mathbf{t}_i \to \mathbf{C}(\delta)\mathbf{t}_i$,
  $\theta_i \to \theta_i + \delta$. Then
  $\mathbf{C}_i^T(\mathbf{t}_j - \mathbf{t}_i) \to \mathbf{C}_i^T\mathbf{C}(\delta)^T\mathbf{C}(\delta)(\mathbf{t}_j - \mathbf{t}_i)$,
  unchanged, and $\phi$ is unchanged. To first order
  $\delta\mathbf{t}_i = \delta\,[-y_i, x_i]^T$ and $\delta\theta_i = \delta$, so
  $\mathbf{v}_\psi = [-y_0, x_0, 1, -y_1, x_1, 1, \ldots]$.

Numerically, with each vector normalized,
$\|\mathbf{H}\mathbf{v}_x\| \approx \|\mathbf{H}\mathbf{v}_y\| \approx 4\times10^{-14}$
and $\|\mathbf{H}\mathbf{v}_\psi\| \approx 2\times10^{-10}$ (the latter scaled
by pose coordinates of order 10 m).

```python
H_full, P, n = H.toarray(), optimized, len(optimized)
v_psi = np.column_stack([-P[:, 1], P[:, 0], np.ones(n)]).ravel()
print(np.linalg.norm(H_full @ (v_psi / np.linalg.norm(v_psi))))
```

**(c)** A global rotation moves each pose by an amount proportional to its
distance from the rotation center, so the direction depends on where the poses
currently are (the linearization point). A translation moves every pose by the
same amount regardless of the estimate. (Rotating about a different center
gives the same rotation direction plus a combination of the two translations,
so the null space itself, a 3D subspace, does not depend on that choice.)

**(d)** Anchoring only $(x_0, y_0)$ (e.g. adding $10^{12}$ to those two
diagonal entries) leaves the rotation about pose 0's position free: one
eigenvalue stays at zero ($\sim 10^{-8}$ against an anchor of $10^{12}$), the
reduced system is singular, and `spsolve` either warns or returns garbage.
A gauge needs three constraints in 2D (six in 3D; seven for monocular
$\mathrm{Sim}(3)$ problems).

## Exercise 3: What a loop closure buys

```python
def position_sigma(edges_used):
    poses, _ = gauss_newton(dead_reckoning, edges_used)
    H_used, _, _ = build_normal_equations(poses, edges_used)
    cov = np.linalg.inv(H_used[3:, 3:].toarray())        # pose 0 fixed
    sigma = np.zeros(len(poses))
    for k in range(1, len(poses)):
        block = cov[3 * (k - 1):3 * (k - 1) + 2, 3 * (k - 1):3 * (k - 1) + 2]
        sigma[k] = np.sqrt(np.trace(block))
    return sigma

with_loop, without_loop = position_sigma(edges), position_sigma(edges[:-1])
```

Results: without the loop closure the one-sigma position uncertainty grows
monotonically, to 0.73 m at pose 20, 1.05 m at pose 40, and 1.11 m at pose 50.
With it, the uncertainty peaks at pose 20 (0.43 m), halfway around the loop;
it is 0.31 m and 0.34 m at poses 10 and 30, only 0.07 m at pose 40, then grows
again on the second lap to 0.38 m at pose 50.

Pose 40 is not the worst because the loop closure ties it directly to the
anchored pose 0 with a 5 cm measurement. The most uncertain pose is the one
farthest *in the graph* from both anchors, the middle of the loop, exactly the
$kq(1 - k/N)$ profile of Exercise 6. The second-lap poses hang off pose 40
through odometry alone, so their uncertainty grows like dead reckoning from a
well-known start. (These sigmas come from the noise model; they know nothing
about the unmodeled heading bias, which is why the actual errors in
@fig-loop-closure-pose-graph exceed them in places. A good follow-up question:
compute the NEES.)

## Exercise 4: Breaking the robust kernel

A helper that rebuilds the problem for a given bias, Cauchy width, optional
graduated schedule, and optional extra false closures:

```python
def run_case(bias_deg, c, schedule=False, extra_false=0, seed=7):
    rng_ = np.random.default_rng(seed)
    e_ = []
    for i in range(n_poses - 1):
        z = relative_pose(truth[i], truth[i + 1]) + rng_.normal(0.0, sigma_odo)
        z[2] += np.deg2rad(bias_deg)
        e_.append((i, i + 1, z, Omega_odo))
    e_.append((40, 0, relative_pose(truth[40], truth[0]) + rng_.normal(0.0, sigma_loop), Omega_loop))
    dr = [truth[0]]
    for i, j, z, _ in e_[: n_poses - 1]:
        dr.append(compose(dr[-1], z))
    dr = np.array(dr)
    clean, _ = gauss_newton(dr, e_)
    e_.append((48, 2, rng_.normal(0.0, sigma_loop), Omega_loop))
    loops = [n_poses - 1, n_poses]
    rng_f = np.random.default_rng(100)
    while len(loops) < 2 + extra_false:              # random false closures
        a, b = rng_f.integers(0, n_poses, 2)
        if abs(a - b) >= 10 and np.linalg.norm(truth[a, :2] - truth[b, :2]) > 2.0:
            e_.append((int(a), int(b), rng_f.normal(0.0, sigma_loop), Omega_loop))
            loops.append(len(e_) - 1)
    state = {"it": 0}

    def weight_fn(poses, edges_):
        width = max(c, 100.0 * 0.5 ** state["it"]) if schedule else c
        state["it"] += 1
        w = np.ones(len(edges_))
        for k in loops:
            i, j, z, Om = edges_[k]
            ee, _, _ = edge_residual_and_jacobians(poses[i], poses[j], z)
            w[k] = cauchy_weight(ee @ Om @ ee, width)
        return w

    poses, _ = gauss_newton(dr, e_, weight_fn, max_iterations=100)
    return position_rmse(dr), position_rmse(clean), position_rmse(poses)
```

**(a)** Bias $0.6^\circ$ (clean optimum RMSE 0.32 m):

| Cauchy width $c$ | RMSE [m] | final weights (true, false) |
|---|---|---|
| 0.5 | 1.06 | $10^{-4}$, $\approx 0$ |
| 1.0 | 0.76 | $1.2\times10^{-3}$, $10^{-4}$ |
| 2.8 | 0.32 | 0.997, $6\times10^{-4}$ |
| 10 | 0.33 | 0.992, $8\times10^{-3}$ |

Widths of 1 or less reject the true closure too, for two different reasons
that are worth separating by evaluating the robust cost
$\sum_\text{odo} s_k + \sum_\text{loops} c^2\ln(1 + s_k/c^2)$ at the clean
optimum and at the IRLS answer:

- $c = 0.5$: the cost is 7.80 at the clean optimum but 5.07 at the IRLS
  answer, so the *global* minimum of this robust cost ignores the true closure.
  Ignoring a closure costs only $c^2\ln(1 + s/c^2)$, which is scaled by $c^2$
  and grows logarithmically, while honoring it forces the odometry to absorb
  the misclosure (about 5 units of squared residual here, mostly from the
  unmodeled bias). A kernel that narrow is a bad model, not just a bad start.
- $c = 1$: the clean optimum has the lower cost (14.6 versus 17.2), so IRLS is
  stuck in a local minimum. At the first iteration the true closure's weight is
  about $c^2/s \approx 5\times10^{-4}$; its pull, roughly $c^2/\|\mathbf{e}\|$
  for large residuals, is too weak to start closing the loop, so its residual
  never shrinks.

The true closure's residual at the clean optimum is tiny
($s \approx 0.02$), so this is not a matter of the closure being noisy. The
width must reflect the real residual distribution and leave the solver a basin
of attraction; $c = \sqrt{\chi^2_{3,0.95}}$ is a principled default. A wide
kernel ($c = 10$) still works here but lets the false closure keep about 1%
weight.

**(b)** With $c = 2.8$ and a $1.5^\circ$/step bias, dead reckoning has drifted
to 3.4 m RMSE; the clean optimum is 0.38 m but Cauchy returns 2.67 m, with
final weights $6\times10^{-4}$ (true) and $2\times10^{-4}$ (false). At the first
iteration both closures have residuals of dozens of sigma, so both get weights
near zero. Odometry alone does not move the map toward the true closure, the
true closure's residual never shrinks, and IRLS sits in a local minimum where
both closures are ignored. (At $1.2^\circ$ with this seed Cauchy still
succeeds; the failure point depends on the seed.)

**(c)** With the schedule $c_k = \max(2.8,\ 100 \cdot 2^{-k})$: at $1.5^\circ$
it succeeds (RMSE 0.37 m; weights 0.96 and $5\times10^{-4}$). Early iterations
are essentially least squares, which closes the loop (partly satisfying the
false closure too); as $c$ shrinks the false closure's large residual is
downweighted first. At $2.0^\circ$ and above it fails like plain Cauchy
(4.06 m at $2.0^\circ$). Reasons and what real GNC [@yang2020gnc] does
differently: this schedule takes one Gauss-Newton step per stage, so no
intermediate problem is solved to convergence; it shrinks $c$ by a factor of 2
per step where GNC increases the nonconvexity slowly (a factor of about 1.4)
and re-solves each stage; GNC uses surrogates (Geman-McClure or truncated least
squares) with closed-form weight updates and a known inlier threshold. A better
initial guess (e.g. chordal relaxation of the rotations) also helps more than
any schedule when drift is large.

**(d)** `run_case(0.6, 2.8, extra_false=5)` adds false closures
$(6, 30), (48, 30), (36, 9), (7, 29), (6, 25)$ on top of $(48, 2)$. RMSE:
plain least squares 5.90 m, Huber ($k = 2.8$) 6.36 m, Cauchy 0.33 m (clean
0.32 m). All six false closures end with Cauchy weights below $10^{-3}$.
Huber is *worse* than plain least squares here: each false closure's pull is
capped but never removed, six of them overwhelm one true closure, and the
weight cap also shrinks the true closure's influence. Bounded influence is not
enough when outliers are numerous; redescending kernels or explicit
consistency checks (PCM) are needed.

## Exercise 5: Constellation place recognition

A complete solution (about 0.2 s):

```python
import numpy as np


def rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def local_scan(rng, xy, diam, center, heading, radius=15.0, p_miss=0.2, n_spurious=2):
    """Trees within `radius` of `center`, expressed in a vehicle frame at (center, heading)."""
    near = np.linalg.norm(xy - center, axis=1) < radius
    keep = near & (rng.random(len(xy)) > p_miss)
    pts = (xy[keep] - center) @ rot2(heading) + rng.normal(0, 0.1, (keep.sum(), 2))
    d = diam[keep] + rng.normal(0, 0.05, keep.sum())
    r, a = radius * np.sqrt(rng.random(n_spurious)), rng.uniform(0, 2 * np.pi, n_spurious)
    pts = np.vstack([pts, np.column_stack([r * np.cos(a), r * np.sin(a)])])
    return pts, np.append(d, rng.uniform(0.2, 0.8, n_spurious))


def fit_se2(src, dst):
    """Least-squares rotation and translation with dst ~ C src + t (2D Kabsch)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S = (src - mu_s).T @ (dst - mu_d)
    theta = np.arctan2(S[0, 1] - S[1, 0], S[0, 0] + S[1, 1])
    C = rot2(theta)
    return C, mu_d - C @ mu_s, theta


def count_inliers(C, t, new_xy, new_d, old_xy, old_d, tol_xy=0.4, tol_d=0.15):
    dist = np.linalg.norm((new_xy @ C.T + t)[:, None] - old_xy[None], axis=2)
    j = dist.argmin(1)
    ok = (dist[np.arange(len(new_xy)), j] < tol_xy) & (np.abs(new_d - old_d[j]) < tol_d)
    return ok, j


def match_constellation(new_xy, new_d, old_xy, old_d, rng, n_iter=500, tol_len=0.3, tol_d=0.15):
    def pairs(xy):
        i, j = np.triu_indices(len(xy), 1)
        return i, j, np.linalg.norm(xy[i] - xy[j], axis=1)
    ni, nj, nl = pairs(new_xy)
    oi, oj, ol = pairs(old_xy)
    cand = []                                   # pair correspondences that agree in length and diameters
    for a in range(len(nl)):
        for b in np.flatnonzero(np.abs(ol - nl[a]) < tol_len):
            for p, q in [(oi[b], oj[b]), (oj[b], oi[b])]:
                if abs(new_d[ni[a]] - old_d[p]) < tol_d and abs(new_d[nj[a]] - old_d[q]) < tol_d:
                    cand.append((ni[a], nj[a], p, q))
    best_count, best = 0, None
    for _ in range(min(n_iter, len(cand))):
        i1, i2, j1, j2 = cand[rng.integers(len(cand))]
        C, t, _ = fit_se2(new_xy[[i1, i2]], old_xy[[j1, j2]])
        ok, _ = count_inliers(C, t, new_xy, new_d, old_xy, old_d)
        if ok.sum() > best_count:
            best_count, best = ok.sum(), (C, t)
    if best is None:
        return 0, None
    ok, j = count_inliers(*best, new_xy, new_d, old_xy, old_d)
    C, t, theta = fit_se2(new_xy[ok], old_xy[j[ok]])        # refine on all inliers
    ok, _ = count_inliers(C, t, new_xy, new_d, old_xy, old_d)
    return ok.sum(), (C, t, theta)


rng = np.random.default_rng(5)
xy, diam = rng.uniform(0, 60, (60, 2)), rng.uniform(0.2, 0.8, 60)
A = np.array([30.0, 30.0])
old = np.linalg.norm(xy - A, axis=1) < 15.0
B, heading = A + np.array([3.0, 0.0]), 0.7
new_xy, new_d = local_scan(rng, xy, diam, B, heading)
print(match_constellation(new_xy, new_d, xy[old], diam[old], rng))
```

**(a)** With this seed the old map has 11 trees and the scan 9 (7 real, 2
spurious), giving 45 candidate pair correspondences. RANSAC finds 6 inliers and
recovers heading 0.707 rad (true 0.70) and the scan origin at (32.93, 30.04) m
(true (33, 30)).

**(b)** Scans of five patches of an independent random forest give 2 to 3
inliers each. Two inliers come free with every sample (the sample itself), and
a third occasionally by chance. A reasonable acceptance rule here is at least 5
inliers *and* at least half of the scan's trees explained, followed by a check
that the implied relative pose is plausible given odometry. Expect students to
note that the margin (6 versus 3) is thin with only about 10 trees in view,
which is the argument for matching submaps accumulated over several scans.

**(c)** In the real data (@sec-mapping): trunk centers are biased by partial
returns (the laser sees only the near face of a trunk) and by the 0.5° beam
spacing at range; diameter estimates are noisy and range dependent; the 180°
forward field of view means a revisit from a different direction sees a
different subset of trees; some areas are open grass with too few trees for a
constellation, while planted rows are repetitive (perceptual aliasing); non-tree
objects pass the detector; and building local maps from several scans requires
odometry, which drifts. Any of these should be accepted if argued well.

## Exercise 6: The surveyor's compass rule

**(a)** Write $x_k = x_k^\text{DR} + \sum_{m=1}^{k} w_m$. Then $x_k$ and $z$
are jointly Gaussian with $\operatorname{Cov}(x_k, z) = kq$ and
$\operatorname{Var}(z) = Nq + r$. Conditioning (@sec-kf-derivation):

$$
\hat{x}_k = x_k^\text{DR} + \frac{kq}{Nq + r}\left(z - x_N^\text{DR}\right).
$$

Equivalently, minimizing
$\sum_k (x_k - x_{k-1} - u_k)^2/q + (z - x_N)^2/r$ adjusts every step by the
same amount $q(z - x_N^\text{DR})/(Nq + r)$, so the correction to $x_k$ is
linear in $k$. With unequal steps and $q$ proportional to step length, the
correction is proportional to distance traveled: Bowditch's compass rule for
traverse adjustment.

**(b)** $\sigma_k^2 = \operatorname{Var}(x_k) - \operatorname{Cov}(x_k, z)^2/\operatorname{Var}(z) = kq - (kq)^2/(Nq + r)$.
As $r \to 0$, $\sigma_k^2 = kq(1 - k/N)$, which peaks at $k = N/2$ with value
$Nq/4$, and vanishes at both ends. This is the profile of Exercise 3.

**(c)** A filter carrying only $x_N$ has prior $\mathcal{N}(x_N^\text{DR}, Nq)$
and measurement $z = x_N + v$; its update is
$\hat{x}_N = x_N^\text{DR} + \frac{Nq}{Nq + r}(z - x_N^\text{DR})$, identical
to (a) at $k = N$, with the same variance. It has no estimates of past poses to
update; they stay at dead reckoning. The RTS smoother (@sec-kf-smoothing)
recovers them with one backward pass and reproduces (a) exactly, because the
problem is linear and Gaussian.

For a planar robot with an uncertain heading the map from odometry to positions
is nonlinear (each step is rotated by the current heading). The filter
linearizes each step at the then-current, drifted estimate and those Jacobians
are frozen into its covariance; a smoother relinearizes every pose about the
corrected trajectory. So even $\hat{x}_N$ differs, and the filter tends to be
overconfident: the EKF-SLAM inconsistency of @sec-loop-closure-why-filtering-slam
[@huang2010].
