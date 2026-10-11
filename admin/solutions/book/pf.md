# Solutions: Nonparametric Filters and Particle Filters (`pf.qmd`)

Instructor-only. Particle weights are $w^{[i]}$; the chapter's terrain-aided
navigation (TAN) example uses $N = 2000$, $\sigma_v = 3$ m/s, $\sigma_z = 12$ m.

## Exercise 1: Sequential importance sampling

**(a)** Target: $\pi_k(\mathbf{x}_{0:k}) = p(\mathbf{x}_{0:k} \mid \mathbf{z}_{1:k}, \mathbf{u}_{1:k})$.
Bayes' rule on the newest measurement, then the chain rule on the newest state:

$$
\pi_k(\mathbf{x}_{0:k})
= \frac{p(\mathbf{z}_k \mid \mathbf{x}_{0:k}, \mathbf{z}_{1:k-1}, \mathbf{u}_{1:k})\,
        p(\mathbf{x}_k \mid \mathbf{x}_{0:k-1}, \mathbf{z}_{1:k-1}, \mathbf{u}_{1:k})\,
        p(\mathbf{x}_{0:k-1} \mid \mathbf{z}_{1:k-1}, \mathbf{u}_{1:k})}
       {p(\mathbf{z}_k \mid \mathbf{z}_{1:k-1}, \mathbf{u}_{1:k})}.
$$

Assumptions: (i) measurement conditional independence,
$p(\mathbf{z}_k \mid \mathbf{x}_{0:k}, \mathbf{z}_{1:k-1}, \mathbf{u}_{1:k}) = p(\mathbf{z}_k \mid \mathbf{x}_k)$;
(ii) Markov motion, $p(\mathbf{x}_k \mid \mathbf{x}_{0:k-1}, \ldots) = p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k)$;
(iii) future controls carry no information about past states,
$p(\mathbf{x}_{0:k-1} \mid \mathbf{z}_{1:k-1}, \mathbf{u}_{1:k}) = \pi_{k-1}(\mathbf{x}_{0:k-1})$.
The denominator does not depend on the state. So
$\pi_k \propto p(\mathbf{z}_k \mid \mathbf{x}_k)\,p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k)\,\pi_{k-1}$.

Proposal: (iv) $q_k(\mathbf{x}_{0:k}) = q(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{z}_k, \mathbf{u}_k)\,q_{k-1}(\mathbf{x}_{0:k-1})$,
which means old samples are kept and only extended. The importance weight of a
trajectory is $w_k \propto \pi_k / q_k$, so

$$
w_k \propto \frac{p(\mathbf{z}_k \mid \mathbf{x}_k)\,p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k)}{q(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{z}_k, \mathbf{u}_k)}\cdot\frac{\pi_{k-1}}{q_{k-1}}
= w_{k-1}\,\frac{p(\mathbf{z}_k \mid \mathbf{x}_k)\,p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k)}{q(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{z}_k, \mathbf{u}_k)}.
$$

Because the weight only involves $\mathbf{x}_{k-1}$ and $\mathbf{x}_k$, a filter
never needs to store whole trajectories. (Also required: $q > 0$ wherever the
target is positive.)

**(b)** With $q = p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k)$ the transition terms cancel:
$w_k \propto w_{k-1}\,p(\mathbf{z}_k \mid \mathbf{x}_k)$.

**(c)** With $q^\star = p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{z}_k, \mathbf{u}_k)$, Bayes' rule gives
$q^\star = p(\mathbf{z}_k \mid \mathbf{x}_k)\,p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k)/p(\mathbf{z}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k)$, so

$$
w_k \propto w_{k-1}\,p(\mathbf{z}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k)
= w_{k-1}\int p(\mathbf{z}_k \mid \mathbf{x}_k)\,p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k)\,d\mathbf{x}_k,
$$

which does not depend on the sampled $\mathbf{x}_k$. Conditional on
$\mathbf{x}_{k-1}$, the new weight therefore has zero variance, which is why this
proposal minimizes the weight variance.

Scalar model $x_k = x_{k-1} + w_k$, $w_k \sim \mathcal{N}(0, Q)$; $z_k = x_k + v_k$, $v_k \sim \mathcal{N}(0, R)$:

$$
q^\star(x_k \mid x_{k-1}, z_k) = \mathcal{N}\!\left(\Sigma\left(\frac{x_{k-1}}{Q} + \frac{z_k}{R}\right),\ \Sigma\right),
\quad \Sigma = \frac{QR}{Q + R},
\qquad
w_k \propto w_{k-1}\,\mathcal{N}(z_k;\ x_{k-1},\ Q + R).
$$

## Exercise 2: What N_eff measures

With $\sum_i w^{[i]} = 1$ and $w^{[i]} \ge 0$: $\sum_i (w^{[i]})^2 \le \max_i w^{[i]}\sum_i w^{[i]} \le 1$, so
$N_\text{eff} \ge 1$, with equality iff one particle has all the weight. By
Cauchy-Schwarz, $1 = (\sum_i w^{[i]} \cdot 1)^2 \le N\sum_i (w^{[i]})^2$, so
$N_\text{eff} \le N$, with equality iff all weights are $1/N$.

If $M$ particles have weight $1/M$ and the rest zero, $\sum (w^{[i]})^2 = M/M^2 = 1/M$,
so $N_\text{eff} = M$. This is the sense in which $N_\text{eff}$ counts "useful"
particles.

Let $u_i = N w^{[i]}$, so $\bar{u} = 1$ and
$\mathrm{CV}^2 = \frac{1}{N}\sum_i (u_i - 1)^2 = \frac{1}{N}\sum_i u_i^2 - 1 = N\sum_i (w^{[i]})^2 - 1$.
Hence $\sum_i (w^{[i]})^2 = (1 + \mathrm{CV}^2)/N$ and $N_\text{eff} = N/(1 + \mathrm{CV}^2)$.

## Exercise 3: Resampling variance

```python
import numpy as np
# (use cumulative_weights, multinomial_resample, stratified_resample,
#  low_variance_resample from the chapter)
rng = np.random.default_rng(0)
N = 50
w = rng.gamma(0.5, size=N); w /= w.sum()
ids = np.arange(N)
counts = {name: np.zeros((10_000, N)) for name in ("multinomial", "stratified", "systematic")}
for r in range(10_000):
    for name, fn in (("multinomial", multinomial_resample),
                     ("stratified", stratified_resample),
                     ("systematic", low_variance_resample)):
        out, _ = fn(ids, w, rng)
        counts[name][r] = np.bincount(out, minlength=N)
variances = {name: c.var(axis=0) for name, c in counts.items()}
# plot variances[name] against N * w for each method
```

Expected: every method is unbiased (mean copies $= N w^{[i]}$). Multinomial
variance is binomial, $N w^{[i]}(1 - w^{[i]})$, close to $N w^{[i]}$, so it grows
with the weight. Stratified is lower. Systematic is lowest and always gives each
particle either $\lfloor N w^{[i]} \rfloor$ or $\lceil N w^{[i]} \rceil$ copies.

For $N w^{[i]} = 2.3$, an unbiased integer count with minimum variance takes the
values 2 and 3 with probabilities 0.7 and 0.3, giving variance
$0.7 \cdot 0.3 = 0.21$. (Multinomial gives about
$2.3(1 - 2.3/50) \approx 2.19$, ten times more.) Systematic resampling attains
0.21 for every particle individually; its weakness is that the counts of
neighboring particles are strongly correlated, which can matter if particles are
ordered in a structured way.

## Exercise 4: An unmodeled speed bias

Add `v_dr = v_true + 0.8 + rng.normal(0, sigma_v, n_steps)` and compute
NEES $= e_k^2/\hat{\sigma}_k^2$ after convergence, averaged over time and over 20
seeds.

Expected: without the bias, average NEES is close to 1 (a reference run of the
chapter's model over 20 seeds gave about 0.95). With the bias, the position error
accumulates $0.8$ m/s faster than the model allows. Over rough terrain the
measurements pull it back, but over the 6 km lake (about 55 s) it drifts about
44 m, while the filter's standard deviation is only 20-30 m, so NEES rises well
above 1 and the 2-sigma bound is violated systematically during and just after
the lake.

Augmented state $[x_k, b_k]$ with
$x_k = x_{k-1} + (\tilde{v}_k - b_{k-1} + w_k)\Delta t$ and
$b_k = b_{k-1} + w^b_k$, for example $\sigma_b = 0.02$ m/s per step, and an
initial $b_0 \sim \mathcal{N}(0, 1^2)$. The bias becomes observable from the
terrain over the ridges, the filter learns it, and NEES for position returns
close to 1. Expect the convergence to need somewhat more particles, because the
state space is now 2D.

Rao-Blackwellization: conditional on a particle's position history,
$x_k - x_{k-1} - \tilde{v}_k\Delta t = -b_{k-1}\Delta t + w_k\Delta t$ is a linear
measurement of $b$ with Gaussian noise, and $b$ evolves linearly. So each
particle can carry a scalar Kalman filter for $b$ (mean and variance) and only
the position needs to be sampled; the weight update then uses the predictive
likelihood of the motion increment from that per-particle filter. This is the
marginalized particle filter structure used in operational TAN.

## Exercise 5: Kidnapping an aircraft

Implementation: after $t = 120$ s, set `x_true[k:] += 3000` before generating
the measurements. For the augmented version, compute the per-step mean
likelihood as `np.mean(np.exp(-0.5 * (residual / sigma_z) ** 2))` (the same
unnormalized Gaussian in every step, so the ratio $w_\text{fast}/w_\text{slow}$ is
meaningful), initialize `state = {"w_slow": m0, "w_fast": m0}` with the first
step's mean likelihood, resample with `augmented_mcl_resample` (reshape the 1D
particles to `(N, 1)` and use `bounds=[(0, 30_000)]`), and resample every step or
whenever $N_\text{eff} < N/2$.

Expected: the plain filter does not recover. All particles are about 3 km
behind the truth; the mean likelihood collapses; $N_\text{eff}$ drops to a few
particles at nearly every step; repeated resampling concentrates the cloud on
whatever local feature fits least badly, and because the process noise moves
particles only a few meters per step, nothing ever reaches the true position.
The error stays near 3 km (or wanders), while the reported standard deviation
stays small: a confidently wrong filter.

With augmented MCL, the injection probability climbs over the next few tens of
steps; random particles spread over the whole map, and those landing within the
likelihood width of the truth (tens of meters) quickly dominate. Recovery
typically takes on the order of tens of seconds, similar to the initial
convergence, because the filter must again disambiguate ridges. Students should
report their measured recovery time and how it depends on $N$ and $\gamma_\text{fast}$.

The lake does not trigger a false alarm: over flat terrain every particle (and
the truth) predicts the same elevation, so the mean likelihood stays at its
typical value for a correct fit. The flip side is that a kidnapping *over* the
lake is undetectable until the aircraft reaches informative terrain again.

## Exercise 6: How many particles?

Loop over $N$ and 20 seeds, rerunning the TAN cell with `n_particles = N`.
Declare "locked on the wrong ridge" if, after $t = 60$ s, the absolute error
exceeds, say, 500 m (half the ridge spacing is 750 m) for most of the remaining
flight. Time each run with `time.perf_counter()`.

Expected trends (the exact numbers depend on the machine and seeds):

- Failure rate falls steeply with $N$. The governing ratio is prior width over
  likelihood width: the prior spans 10 km, and a single reading localizes a
  particle to within roughly $\sigma_z$ divided by the terrain slope, a few tens
  of meters on the ridge flanks. With $N = 100$ the initial particle spacing is
  100 m, so the correct hypothesis is often poorly represented and the filter
  can lock onto a wrong ridge; by $N \approx 1000$-$2000$ failures should be rare.
- RMS error of successful runs improves only modestly beyond the point where
  the filter converges reliably; once locked on, a few hundred particles track
  a 1D state well.
- Runtime is linear in $N$ (vectorized NumPy; on the order of ten milliseconds
  per run at $N = 2000$ for 180 steps, excluding plotting).

Connection to Part C: the particle count is set by the *initial* (or
worst-case) uncertainty relative to the measurement's resolving power, not by
steady-state tracking accuracy. Part C's 15-state drone filter has a far larger
state space, which is why its particle-count study shows much slower
improvement with $N$ and why highest-weight-particle estimates can be noisy.
