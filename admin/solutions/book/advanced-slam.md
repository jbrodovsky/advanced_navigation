# Solutions: Factor Graphs, Manifolds, and Modern Back-ends (`advanced-slam.qmd`)

Instructor-only. Numbers come from the chapter's cells as written (orbit:
seed 5, 90 keyframes, odometry noise 0.1 m and $0.5^\circ$ per axis, GNSS every
sixth keyframe with $(1.5, 1.5, 3)$ m noise and lever arm $(0.3, 0, -0.2)$ m
FRD, initial heading error $40^\circ$; marginalization example seed 4; FEJ
example continuing the same generator). Code snippets assume the chapter's cells
have been run in order. They were checked with the course environment (NumPy
2.4, SciPy 1.17).

## Exercise 1: Left and right Jacobians of $SO(3)$

**(a)** Start from the right-Jacobian definition and move the small rotation to
the left with the adjoint identity
$\mathbf{C}\operatorname{Exp}(\mathbf{v})\mathbf{C}^T = \operatorname{Exp}(\mathbf{C}\mathbf{v})$:

$$
\operatorname{Exp}(\boldsymbol{\phi} + \delta\boldsymbol{\phi})
\approx \operatorname{Exp}(\boldsymbol{\phi})\operatorname{Exp}(\mathbf{J}_r\delta\boldsymbol{\phi})
= \operatorname{Exp}\left(\operatorname{Exp}(\boldsymbol{\phi})\mathbf{J}_r\delta\boldsymbol{\phi}\right)\operatorname{Exp}(\boldsymbol{\phi}).
$$

Comparing with the left-Jacobian definition gives
$\mathbf{J}_l(\boldsymbol{\phi}) = \operatorname{Exp}(\boldsymbol{\phi})\mathbf{J}_r(\boldsymbol{\phi})$.
For the second identity, transpose the left-Jacobian definition:
$\operatorname{Exp}(-\boldsymbol{\phi} - \delta\boldsymbol{\phi}) \approx \operatorname{Exp}(-\boldsymbol{\phi})\operatorname{Exp}(-\mathbf{J}_l(\boldsymbol{\phi})\delta\boldsymbol{\phi})$.
The right-Jacobian definition at $-\boldsymbol{\phi}$ with increment
$-\delta\boldsymbol{\phi}$ says the same left-hand side is
$\operatorname{Exp}(-\boldsymbol{\phi})\operatorname{Exp}(-\mathbf{J}_r(-\boldsymbol{\phi})\delta\boldsymbol{\phi})$,
so $\mathbf{J}_l(\boldsymbol{\phi}) = \mathbf{J}_r(-\boldsymbol{\phi})$.
Finally, in the closed form
$\mathbf{J}_r = \mathbf{I} - a[\boldsymbol{\phi}]_\times + b[\boldsymbol{\phi}]_\times^2$
the skew term changes sign under transposition and the square does not, so
$\mathbf{J}_r^T = \mathbf{I} + a[\boldsymbol{\phi}]_\times + b[\boldsymbol{\phi}]_\times^2 = \mathbf{J}_r(-\boldsymbol{\phi})$.
Numerically, `so3_left_jacobian(phi) - so3_right_jacobian(phi).T` is exactly
zero, and the chapter's cell already checks $\mathbf{J}_l = \operatorname{Exp}(\boldsymbol{\phi})\mathbf{J}_r$
to $10^{-16}$.

**(b)** Write $\delta\boldsymbol{\theta} = \delta\hat{\boldsymbol{\theta}} + \boldsymbol{\epsilon}$.
By the left-Jacobian definition,
$\operatorname{Exp}(\delta\hat{\boldsymbol{\theta}} + \boldsymbol{\epsilon})\operatorname{Exp}(-\delta\hat{\boldsymbol{\theta}}) \approx \operatorname{Exp}(\mathbf{J}_l(\delta\hat{\boldsymbol{\theta}})\boldsymbol{\epsilon})$,
so $\delta\boldsymbol{\theta}^+ \approx \mathbf{J}_l(\delta\hat{\boldsymbol{\theta}})\boldsymbol{\epsilon}$
and the reset Jacobian is $\mathbf{J}_l(\delta\hat{\boldsymbol{\theta}})$.
To first order $\mathbf{J}_l(\boldsymbol{\phi}) = \mathbf{I} + \tfrac{1}{2}[\boldsymbol{\phi}]_\times + O(\phi^2)$,
which is @eq-ekf-reset-jacobian. For Solà's body-frame error,
$\operatorname{Exp}(\delta\boldsymbol{\theta}^+) = \operatorname{Exp}(-\delta\hat{\boldsymbol{\theta}})\operatorname{Exp}(\delta\boldsymbol{\theta})$,
and the right-Jacobian definition gives $\mathbf{J}_r(\delta\hat{\boldsymbol{\theta}}) \approx \mathbf{I} - \tfrac{1}{2}[\delta\hat{\boldsymbol{\theta}}]_\times$,
Solà's sign. A finite-difference check of the body-frame reset against
`so3_right_jacobian(dtheta_hat)` agrees to $6\times10^{-12}$.

**(c)** Use $(1 + \cos\phi)/\sin\phi = \cot(\phi/2)$ and
$\cot x = 1/x - x/3 - x^3/45 - \cdots$ with $x = \phi/2$:

$$
\frac{1}{\phi^2} - \frac{\cot(\phi/2)}{2\phi}
= \frac{1}{\phi^2} - \frac{1}{2\phi}\left(\frac{2}{\phi} - \frac{\phi}{6} - \frac{\phi^3}{360} - \cdots\right)
= \frac{1}{12} + \frac{\phi^2}{720} + \cdots
$$

so $\mathbf{J}_r^{-1} = \mathbf{I} + \tfrac{1}{2}[\boldsymbol{\phi}]_\times + \tfrac{1}{12}[\boldsymbol{\phi}]_\times^2 + O(\phi^4)$
(the next term, $\tfrac{\phi^2}{720}[\boldsymbol{\phi}]_\times^2$, is fourth
order). Dropping the quadratic term leaves an error of norm
$\|[\boldsymbol{\phi}]_\times^2\|/12 = \phi^2/12$, which reaches 1% at
$\phi = \sqrt{0.12} \approx 0.35$ rad, about $20^\circ$. Numerically, the
relative spectral-norm error of $\mathbf{I} + \tfrac{1}{2}[\boldsymbol{\phi}]_\times$
is 0.0083% at $5.7^\circ$, 0.75% at $17^\circ$, 0.99% at $19.8^\circ$, and 2.1%
at $28.6^\circ$. Students should note the practical lesson: small-angle
Jacobians are fine for converged residuals and wrong for the first iterations
of a badly initialized problem.

## Exercise 2: Factor Jacobians on the manifold

**(a)** With $\mathbf{d} = \mathbf{p}_j - \mathbf{p}_i$ and
$\mathbf{E} = \Delta\tilde{\mathbf{C}}_{ij}^T\mathbf{C}_i^T\mathbf{C}_j$,
$\mathbf{e}_\theta = \operatorname{Log}\mathbf{E}$:

- Positions: $\mathbf{e}_p$ is linear in $\mathbf{d}$, so
  $\partial\mathbf{e}_p/\partial\delta\mathbf{p}_i = -\mathbf{C}_i^T$ and
  $\partial\mathbf{e}_p/\partial\delta\mathbf{p}_j = \mathbf{C}_i^T$;
  $\mathbf{e}_\theta$ does not involve positions.
- $\mathbf{e}_p$ with respect to $\delta\boldsymbol{\theta}_i$:
  $\mathbf{C}_i^T \to (\operatorname{Exp}(\delta\boldsymbol{\theta}_i)\mathbf{C}_i)^T = \mathbf{C}_i^T\operatorname{Exp}(-\delta\boldsymbol{\theta}_i) \approx \mathbf{C}_i^T(\mathbf{I} - [\delta\boldsymbol{\theta}_i]_\times)$,
  so $\mathbf{e}_p$ changes by $-\mathbf{C}_i^T[\delta\boldsymbol{\theta}_i]_\times\mathbf{d} = \mathbf{C}_i^T[\mathbf{d}]_\times\delta\boldsymbol{\theta}_i$.
  $\mathbf{e}_p$ does not depend on $\mathbf{C}_j$.
- $\mathbf{e}_\theta$ with respect to $\delta\boldsymbol{\theta}_j$:
  $\mathbf{C}_i^T\operatorname{Exp}(\delta\boldsymbol{\theta}_j)\mathbf{C}_j = \mathbf{C}_i^T\mathbf{C}_j\operatorname{Exp}(\mathbf{C}_j^T\delta\boldsymbol{\theta}_j)$,
  so $\mathbf{E} \to \mathbf{E}\operatorname{Exp}(\mathbf{C}_j^T\delta\boldsymbol{\theta}_j)$ and,
  by the $\operatorname{Log}$ rule, $\mathbf{e}_\theta$ changes by
  $\mathbf{J}_r^{-1}(\mathbf{e}_\theta)\mathbf{C}_j^T\delta\boldsymbol{\theta}_j$.
- $\mathbf{e}_\theta$ with respect to $\delta\boldsymbol{\theta}_i$:
  $\mathbf{C}_i^T\operatorname{Exp}(-\delta\boldsymbol{\theta}_i)\mathbf{C}_j = \mathbf{C}_i^T\mathbf{C}_j\operatorname{Exp}(-\mathbf{C}_j^T\delta\boldsymbol{\theta}_i)$,
  giving $-\mathbf{J}_r^{-1}(\mathbf{e}_\theta)\mathbf{C}_j^T$.

Stacking gives @eq-advanced-slam-relpose-jacobians.

**(b)** A body-frame perturbation $\mathbf{C}\operatorname{Exp}(\delta\boldsymbol{\theta}^b)$
equals the navigation-frame perturbation $\operatorname{Exp}(\mathbf{C}\delta\boldsymbol{\theta}^b)\mathbf{C}$
(adjoint), i.e. $\delta\boldsymbol{\theta}^n = \mathbf{C}\delta\boldsymbol{\theta}^b$.
By the chain rule each attitude column block is the navigation-frame block times
$\mathbf{C}_i$ (for pose $i$) or $\mathbf{C}_j$ (for pose $j$); the position
columns are unchanged because position is still perturbed additively in NED.
Simplifying:

$$
\begin{aligned}
\frac{\partial\mathbf{e}_p}{\partial\delta\boldsymbol{\theta}_i^b} &= \mathbf{C}_i^T[\mathbf{d}]_\times\mathbf{C}_i = [\mathbf{C}_i^T\mathbf{d}]_\times, \\
\frac{\partial\mathbf{e}_\theta}{\partial\delta\boldsymbol{\theta}_i^b} &= -\mathbf{J}_r^{-1}(\mathbf{e}_\theta)\,\mathbf{C}_j^T\mathbf{C}_i, \\
\frac{\partial\mathbf{e}_\theta}{\partial\delta\boldsymbol{\theta}_j^b} &= \mathbf{J}_r^{-1}(\mathbf{e}_\theta).
\end{aligned}
$$

The body-frame form is the one in Forster et al. and in most VIO code, which is
why the rotation residual's Jacobian with respect to the later keyframe is
"just" $\mathbf{J}_r^{-1}$ there.

**(c)**

```python
def oplus_body(C, p, dx):
    """Body-frame attitude perturbation C <- C Exp(dtheta); position still additive in NED."""
    dx = dx.reshape(-1, 6)
    return np.array([Ck @ so3_exp(dth) for Ck, dth in zip(C, dx[:, 3:])]), p + dx[:, :3]

i, j, dC, dp = odometry[17]
_, A, B = odometry_factor(C_dr[i], p_dr[i], C_dr[j], p_dr[j], dC, dp)

def residual_body(which, dx):
    C2, p2 = C_dr.copy(), p_dr.copy()
    C2[[which]], p2[[which]] = oplus_body(C_dr[[which]], p_dr[[which]], dx)
    return odometry_factor(C2[i], p2[i], C2[j], p2[j], dC, dp)[0]

A_body = numerical_jacobian(lambda dx: residual_body(i, dx), np.zeros(6))
B_body = numerical_jacobian(lambda dx: residual_body(j, dx), np.zeros(6))
A_pred, B_pred = A.copy(), B.copy()
A_pred[:, 3:] = A[:, 3:] @ C_dr[i]
B_pred[:, 3:] = B[:, 3:] @ C_dr[j]
print(np.abs(A_body - A_pred).max(), np.abs(B_body - B_pred).max())   # about 2.4e-9
```

The simplified forms in (b) also check to $10^{-10}$.

**(d)** The GNSS factor's attitude Jacobian is $-[\mathbf{C}_k\boldsymbol{\ell}^b]_\times$,
so with the position held fixed a single fix carries attitude information
$[\mathbf{C}\boldsymbol{\ell}]_\times^T\boldsymbol{\Omega}_\text{gnss}[\mathbf{C}\boldsymbol{\ell}]_\times$.
Its eigenvalues for $|\boldsymbol{\ell}| = 0.36$ m are 0, 0.031, and 0.058
rad$^{-2}$ (zero about the lever-arm axis itself), i.e. at best
$\sigma \approx 1/\sqrt{0.058} \approx 4.2$ rad, about $240^\circ$: no useful
information. Roughly, the attitude information scales as $(|\boldsymbol{\ell}|/\sigma_\text{gnss})^2$.
Recomputing $\mathbf{H}$ at the solution with the lever-arm term removed from
the Jacobian changes the median heading sigma from $1.256^\circ$ to
$1.255^\circ$. Heading in the orbit example comes from the motion (GNSS
displacements over tens of meters compared with odometry displacements in the
body frame), not from the lever arm. A lever arm becomes useful for attitude
only with precise positioning (RTK at centimeter level) or with a multi-antenna
baseline of meters, which is exactly how GNSS compasses work.

## Exercise 3: The drone flies under a bridge deck

A reusable solver with optional Levenberg-Marquardt damping (Marquardt's
diagonal scaling, as in @sec-math-reference-pose-graph-optimization):

```python
def solve(gnss_used, C0, p0, damping=None, max_iterations=200):
    """Gauss-Newton (damping=None) or Levenberg-Marquardt on the orbit problem.

    Returns the estimate, the final H, and the cost history (LM: accepted steps only).
    """
    global gnss
    gnss_saved, gnss = gnss, gnss_used        # normal_equations reads the global list
    try:
        C, p, lam = C0.copy(), p0.copy(), damping
        H, b, cost = normal_equations(C, p)
        history = [cost]
        for _ in range(max_iterations):
            if lam is None:
                dx = spla.spsolve(H, -b)
            else:
                dx = spla.spsolve((H + lam * sp.diags(H.diagonal())).tocsc(), -b)
            C_new, p_new = oplus(C, p, dx)
            H_new, b_new, cost_new = normal_equations(C_new, p_new)
            if lam is None or cost_new < cost:     # GN always accepts; LM only if better
                C, p, H, b, cost = C_new, p_new, H_new, b_new, cost_new
                history.append(cost)
                lam = None if lam is None else max(lam / 10, 1e-9)
            else:
                lam *= 10
            if np.abs(dx).max() < 1e-8:
                break
    finally:
        gnss = gnss_saved
    return C, p, H, history
```

**(a)**

```python
gnss_out = [(k, z) for k, z in gnss if not 30 <= k <= 65]     # keeps 0-24 and 66-84
C_o, p_o, H_o, _ = solve(gnss_out, C_dr, p_dr)
P_o = np.linalg.inv(H_o.toarray())
h_sigma = np.sqrt([np.trace(P_o[6 * k:6 * k + 2, 6 * k:6 * k + 2]) for k in range(n_kf)])
h_err = np.linalg.norm((p_true - p_o)[:, :2], axis=1)
```

The horizontal one-sigma position uncertainty peaks at keyframe 47, 1.88 m,
almost exactly in the middle of the gap between the last fix before it
(keyframe 24) and the first after it (keyframe 66). It is 1.48 m at keyframe 30
and 1.33 m at keyframe 65, compared with a median of 1.0 m and a maximum of
1.70 m with all fixes. A smoother uses the fixes on *both* sides of the outage,
so the worst point is the one farthest from both, as in the RTS exercise of
@sec-kf-smoothing and the compass rule of the loop-closure chapter. A filter
would instead peak at the end of the outage. The trajectory's last keyframes are
also high (1.82 m at keyframe 89) because the last fix is at keyframe 84 and the
final seconds are pure extrapolation. The horizontal error stays inside
$3\sigma$ at every keyframe (largest 3.3 m, at keyframe 89). The heading sigma
grows to $2.3^\circ$, largest at the end of the trajectory.

**(b)** Results (iterations are accepted steps; "worst cost" is the largest cost
visited):

| Initial heading error | GN iterations | GN worst cost | LM iterations | LM worst cost | Final cost |
|---|---|---|---|---|---|
| $40^\circ$ | 11 | 8,763 | 12 | 4,577 (initial) | 44.03 |
| $90^\circ$ | 14 | 60,163 | 17 | 20,717 (initial) | 44.03 |
| $150^\circ$ | 16 | 101,428 | 23 | 39,853 (initial) | 44.03 |

Gauss-Newton needs fewer iterations but its cost spikes by an order of magnitude
on the first step; LM never increases the cost (by construction it rejects such
steps and raises the damping) and takes a few more iterations. Neither fails;
both reach the same solution even from $150^\circ$ (and, as a check, from
$170^\circ$, $179^\circ$, and $180^\circ$). This is not luck. The initial
guess is an *accurate trajectory rotated rigidly* about the first fix, so the
problem is essentially the alignment of two point sets (dead-reckoned and GNSS
positions) by a rotation, whose cost has a single minimum and a single maximum
as a function of the heading offset; the only stationary point to fear is the
exact maximum, and noise keeps the iteration off it. A genuinely bad
initialization is one where the *shape* is wrong, for example dead reckoning
with a large gyro bias, so that the trajectory spirals; then local minima appear
and LM, GNC-style continuation, or a global initializer (rotation averaging
first, then positions) earns its keep. Students who find this answer
"suspiciously easy" should be credited for saying so.

**(c)**

```python
from scipy.stats import chi2
gnss_bad = [(k, z + np.array([15.0, 0.0, 0.0]) * (k == 48)) for k, z in gnss]
C_b, p_b, H_b, hist_b = solve(gnss_bad, C_dr, p_dr)
P_b = np.linalg.inv(H_b.toarray())
scores = []
for k, z in gnss_bad:
    e, J = gnss_factor(C_b[k], p_b[k], z)
    block_k = slice(6 * k, 6 * k + 6)
    S_e = np.linalg.inv(Omega_gnss) - J @ P_b[block_k, block_k] @ J.T   # residual covariance
    scores.append((float(e @ np.linalg.solve(S_e, e)), k))
```

The final cost is 130.6 against a one-sided 95% threshold of
$\chi^2_{0.95}(39) = 54.6$ ($p \approx 10^{-11}$), so the global test fails.
Normalizing each GNSS residual by its own residual covariance
$\boldsymbol{\Sigma}_\text{gnss} - \mathbf{J}\mathbf{P}\mathbf{J}^T$
(not just by $\boldsymbol{\Sigma}_\text{gnss}$; the solution has already pulled
toward every fix) gives a statistic of 91.7 for keyframe 48, then 24.3 for
keyframe 60 and 8.2 for keyframe 12. The residual at 48 is $-12.9$ m north
(sign: predicted minus measured, and the fix was pushed north). Keyframe 60 is
flagged too because the bad fix drags its neighbors: the error smears. That is
why fault exclusion removes only the largest offender and tests again:
without keyframe 48 the cost is 38.9 against $\chi^2_{0.95}(36) = 51.0$, and the
test passes. This is RAIM's logic (@sec-adv-gnss-raim) carried over to a
smoother: a global residual test for detection, then per-measurement normalized
residuals for identification, then exclusion and retest. The smoother has more
redundancy than snapshot RAIM, because odometry ties the fixes together over
time, which is why a single 15 m fault is so obvious here.

## Exercise 4: Marginalize, condition, and first estimates

**(a)** By the block inverse,
$(\mathbf{H}^{-1})_{rr} = (\mathbf{H}_{rr} - \mathbf{H}_{rm}\mathbf{H}_{mm}^{-1}\mathbf{H}_{mr})^{-1} = (\mathbf{H}^\star)^{-1} = \mathbf{P}_\text{marg}$,
while $\mathbf{P}_\text{cond} = \mathbf{H}_{rr}^{-1}$. Since $\mathbf{H}_{mm} \succ 0$,
$\mathbf{H}_{rr} - \mathbf{H}^\star = \mathbf{H}_{rm}\mathbf{H}_{mm}^{-1}\mathbf{H}_{mr} \succeq 0$,
so $\mathbf{H}_{rr} \succeq \mathbf{H}^\star \succ 0$, and matrix inversion
reverses the order on positive definite matrices:
$\mathbf{P}_\text{marg} = (\mathbf{H}^\star)^{-1} \succeq \mathbf{H}_{rr}^{-1} = \mathbf{P}_\text{cond}$.
The difference has rank at most $\dim\mathbf{x}_m$, which is why the chapter
prints exactly two nonzero eigenvalues (0.07 each, equal because the $x$ and $y$
problems are identical) for the two coordinates of $k_0$. Interpretation:
conditioning claims the information $\mathbf{H}_{rm}\mathbf{H}_{mm}^{-1}\mathbf{H}_{mr}$
about $k_0$'s neighbors that would only be justified if $k_0$ were known exactly.

**(b)**

```python
rng = np.random.default_rng(0)
q, r_var, N, W = 0.04, 0.25, 40, 5
x0_mean, x0_var = 0.0, 1.0
x = np.cumsum(rng.normal(0, np.sqrt(q), N))
z = x + rng.normal(0, np.sqrt(r_var), N)

kf_mean, kf_var, m_k, v_k = [], [], x0_mean, x0_var        # scalar Kalman filter
for k in range(N):
    v_k += q if k > 0 else 0.0
    gain = v_k / (v_k + r_var)
    m_k, v_k = m_k + gain * (z[k] - m_k), (1 - gain) * v_k
    kf_mean.append(m_k); kf_var.append(v_k)

def batch(n):
    """Information form of the first n states with all their factors."""
    H = np.zeros((n, n)); b = np.zeros(n)
    H[0, 0] += 1 / x0_var; b[0] += x0_mean / x0_var
    for k in range(n):
        H[k, k] += 1 / r_var; b[k] += z[k] / r_var
        if k > 0:
            J = np.zeros(n); J[k - 1], J[k] = -1, 1
            H += np.outer(J, J) / q
    return H, b

Hw, bw, first_index = np.array([[1 / x0_var + 1 / r_var]]), np.array([x0_mean / x0_var + z[0] / r_var]), 0
worst_newest = worst_window = 0.0
for k in range(1, N):
    n = len(bw)                                   # add x_k: dynamics factor and measurement
    Hn = np.zeros((n + 1, n + 1)); Hn[:n, :n] = Hw
    bn = np.append(bw, 0.0)
    J = np.zeros(n + 1); J[-2], J[-1] = -1, 1
    Hn += np.outer(J, J) / q
    Hn[-1, -1] += 1 / r_var; bn[-1] += z[k] / r_var
    Hw, bw = Hn, bn
    if len(bw) > W:                               # marginalize the oldest state
        rest = np.arange(1, len(bw))
        Hw, bw = (Hw[np.ix_(rest, rest)] - np.outer(Hw[rest, 0], Hw[0, rest]) / Hw[0, 0],
                  bw[rest] - Hw[rest, 0] * bw[0] / Hw[0, 0])
        first_index += 1
    mean_w, cov_w = np.linalg.solve(Hw, bw), np.linalg.inv(Hw)
    Hb, bb = batch(k + 1)
    mean_b, cov_b = np.linalg.solve(Hb, bb), np.linalg.inv(Hb)
    window = slice(first_index, k + 1)
    worst_newest = max(worst_newest, abs(mean_w[-1] - kf_mean[k]), abs(cov_w[-1, -1] - kf_var[k]))
    worst_window = max(worst_window, np.abs(mean_w - mean_b[window]).max(),
                       np.abs(cov_w - cov_b[window, window]).max())
print(worst_newest, worst_window)                 # about 4e-16 and 2e-16
```

The newest state matches the Kalman filter, and every state in the window
matches the batch smoother on the data received so far, to rounding. For a
linear-Gaussian model marginalization is exact, so a fixed-lag smoother loses
nothing relative to full smoothing *for the states it still holds*; what it
gives up is the ability to revise states that have left the window. (Note that
the window does not equal the *final* batch smoother: future data would still
change those states.)

**(c)** The null vector for a global rotation assigns pose $i$ the direction
$[-y_i, x_i, 1]^T$ and landmark $m$ the direction $[-y_m, x_m]^T$; it depends on
positions but not on headings. If every position moves by the same translation
$\mathbf{t}$, the rotation null vector at the new estimate is the old one plus
$[-t_y, t_x, 0]$ for every pose and $[-t_y, t_x]$ for every landmark, which is a
combination of the two translation null vectors, and those are null vectors at
*every* estimate. So the prior (linearized at the old estimate) and the new
factors (linearized at the new one) still share a three-dimensional null space,
whatever happens to the headings. Running the chapter's FEJ setup with a common
shift of $(0.3, -0.2)$ m and $+5^\circ$ on every heading gives three eigenvalues
of order $10^{-13}$. Independent perturbations change the *relative* positions,
the two rotation null vectors differ by more than a translation, and the common
null space shrinks to the two translations. This is also why a careless test of
FEJ can "prove" it unnecessary.

**(d)** Information adds, so after $n$ equal contributions the spurious heading
information is $n$ times one step's and the claimed standard deviation scales as
$\sigma_n = \sigma_1/\sqrt{n}$. With the chapter's $\sigma_1 \approx 121^\circ$
for an update of typical size, $\pm 5^\circ$ is reached after
$n = (121/5)^2 \approx 580$ marginalizations. At a few keyframes per second that
is a few minutes of operation, after which the filter believes a heading that
nothing has measured. Real systems are worse or better depending on how far
estimates move after marginalization, which is why the published
inconsistency results are so dependent on trajectory and initialization.

## Exercise 5: Is the learned covariance honest?

```python
from scipy.stats import chi2
gate = chi2.ppf(0.95, 2)                              # 5.99

def simulate(n, rng):
    turning = rng.random(n) < 0.3
    sigma_true = np.where(turning, 0.12, 0.05)
    return turning, rng.normal(size=(n, 2)) * sigma_true[:, None]

turning, err = simulate(2000, np.random.default_rng(0))
nis = np.sum(err**2, axis=1) / 0.05**2
```

**(a)** Overall mean NIS 4.92 and 20.9% of windows above the gate. Straight
walking: mean 1.95, 3.9% above (correct for two degrees of freedom). Turns:
mean 11.8, 60.1% above. Analytically, NIS for a 2D Gaussian scaled by
$s = \sigma_\text{true}/\hat{\sigma}$ has mean $2s^2$, so the mean is
$0.7 \times 2 + 0.3 \times 2 \times 2.4^2 = 4.86$, and the exceedance probability
for turns is $\exp(-5.99/(2 \times 2.4^2)) = 0.594$, overall
$0.7 \times 0.05 + 0.3 \times 0.594 = 0.213$. Simulation and prediction agree.

**(b)** Estimate a sigma scale per motion type on an independent calibration set
(seed 1), $\hat{s}^2 = \overline{\|\mathbf{e}\|^2}/(2\hat{\sigma}^2)$, and test
on a third set (seed 2):

```python
turning_cal, err_cal = simulate(2000, np.random.default_rng(1))
turning_test, err_test = simulate(2000, np.random.default_rng(2))
scale = {t: np.sqrt(np.mean(np.sum(err_cal[turning_cal == t]**2, axis=1)) / (2 * 0.05**2))
         for t in (False, True)}
sigma_corr = 0.05 * np.where(turning_test, scale[True], scale[False])
nis_test = np.sum(err_test**2, axis=1) / sigma_corr**2
```

The scales are 1.007 (straight) and 2.32 (turning). On the test set the mean NIS
is 2.01 and 4.8% exceed the gate (straight 4.2%, turning 5.9%): calibrated in
both regimes. The essential point is the three-way split: fitting the correction
on the data used to evaluate it would make any correction look perfect.

**(c)** An average hides regime dependence. A single global scale fitted on the
calibration set (1.52) gives a test-set mean NIS of 2.12, which looks fine, but
0.2% of straight windows and 30.7% of turning windows exceed the gate: the
network is now underconfident when walking straight and still badly
overconfident in turns, exactly where errors hurt most. Other things a good
average can hide: heavy tails (a few huge errors balanced by many small ones),
correlation between consecutive outputs, dependence on speed, device placement,
or user, and a test set drawn from the same recordings as the training set.
Students should propose checking NIS by regime, the NIS histogram against the
$\chi^2_2$ density, exceedance rates, and lag correlation.

**(d)** If the error accumulates roughly uniformly over the window, consecutive
2 s windows that share 1 s share half their error: the lag-one correlation is
about 0.5 (a simulation with independent 1 s error increments gives 0.499). An
EKF that treats them as independent counts the shared second twice, so its
covariance is too small (overconfident) and its NIS on other aids grows. Fixes:
use non-overlapping windows, inflate the noise (by about a factor of 2 in
variance for 50% overlap), or model the correlation explicitly, for example
with a state that carries the shared error, or by fusing relative displacements
between cloned states only once.

## Exercise 6: Designing the back end for a survey AUV

A strong answer covers the following; there is no single correct design.

**(a) Graph.** Variables: vehicle states at keyframes (attitude on $SO(3)$,
velocity, position, gyro and accelerometer biases), plus slowly varying
calibration (DVL scale factor and DVL-to-IMU misalignment) as random-walk
states. Factors and rates: IMU preintegration between keyframes (IMU at
100-400 Hz). For a navigation-grade INS the preintegration must include Earth
rate and Coriolis terms; the chapter's simplified preintegration neglects Earth
rotation, which is fine for MEMS but not when the gyro bias is $0.01^\circ$/h
and Earth rate is $15^\circ$/h. Then: DVL body-frame velocity factors with lever
arm and misalignment (1-5 Hz, only with bottom lock); depth factors from
pressure (1-10 Hz, corrected for water density, atmospheric pressure, and
tide); USBL position factors every 10 s with noise growing with slant range
(typically a fraction of a percent of range), attached to the state at the
time the acoustic signal was received, not the time the fix arrived; GNSS
position factors at the surface; terrain-matching loop-closure factors between
keyframes on crossing survey lines (often three degrees of freedom: horizontal
position and heading, since depth, roll, and pitch are already well observed);
bias and calibration random-walk factors; an initial prior from surface GNSS and
alignment.

**(b) What each consumer gets.** The controller needs a smooth, low-latency
state at a high rate: the output of a filter, or the newest state of a fixed-lag
smoother propagated forward with the IMU, published at, say, 50-100 Hz with
covariance, validity time, latency, and health flags (DVL bottom lock, age of
the last USBL fix, NIS alarms, mode). Corrections from late USBL fixes or loop
closures must not appear as steps in the control frame; they go into a slowly
updated correction to the global frame (the `map`-to-`odom` pattern), consistent
with the navigation-to-control contract of @sec-practical. The survey product
should be built from a full batch smoother run after the dive, with both surface
GNSS fixes, all USBL fixes, and the vetted loop closures: the bathymetry's
georeferencing accuracy is limited by this trajectory, and there is no latency
constraint.

**(c) Late USBL fixes.** In a factor graph a measurement with timestamp $t$ is
simply a factor on the state at $t$ (a keyframe created or interpolated at that
time), added whenever it arrives; the smoother re-solves and nothing has to be
rolled back. A filter must either buffer measurements and re-run from $t$,
augment its state with a delayed clone, or apply the fix at the wrong time,
which biases the estimate by roughly velocity times delay (1.5 m/s times 2 s is
3 m, larger than a good USBL error). A fixed-lag window of tens of seconds
easily covers a 2 s delay.

**(d) Marginalization and consistency.** Marginalize keyframe states older than
the window (for example 30-60 s) into the prior, together with the IMU, DVL, and
depth factors that only touch them; keep a sparse set of keyframes (with their
sonar submaps) in the offline graph for loop closures. With a navigation-grade
INS, heading is actually observable through gyrocompassing (Earth rate), so the
VIO-style unobservable yaw does not arise in the same form, but heading is
*weakly* observable and still vulnerable to spurious information. Use FEJ (or an
invariant formulation) for every variable touched by the prior, hold the prior at
its linearization point, and verify consistency in simulation with NEES and in
operation with the NIS of USBL fixes, which are an independent check on
horizontal position. A small, deliberate inflation of the prior is a common
practical hedge.

**(e) Loop-closure checks.** Before trusting a terrain match: require enough
overlap and enough terrain relief (a flat, sandy seabed constrains depth but not
horizontal position; the match's cost surface should have a single, well-curved
minimum, and its covariance should reflect any along-track ambiguity); gate the
relative pose against the smoother's predicted relative covariance between the
two keyframes (Mahalanobis test); check pairwise consistency among several
candidate closures (PCM); robustify the closures in the back end (Cauchy, DCS,
or switchable constraints, @sec-loop-closure-robust-kernels); and make sure the
two bathymetric patches were corrected consistently for sound speed and tide,
since an uncorrected tide difference looks exactly like a vertical offset. A
false closure on repetitive terrain such as sand waves is the underwater version
of the perceptual aliasing in the loop-closure chapter.
