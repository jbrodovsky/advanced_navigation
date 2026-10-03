# Solutions: Simultaneous Localization and Mapping (`slam.qmd`)

Instructor-only. Notation follows the chapter: pose $\mathbf{x} = [x, y, \theta]^T$,
landmarks $\boldsymbol{\ell}_j$, $N$ landmarks, $M$ particles, odometry noise
$\mathbf{Q}_u$, measurement noise $\mathbf{R}$. Numerical answers were produced by
running the chapter's cells (seeded) plus the short additions described below; the
toy world is the chapter's (three trees, 12 scans, 1 m to 8 m sensor,
$p_0 = 1/(8 \cdot 2\pi)$).

## Exercise 1: EKF-SLAM bookkeeping

**(a)** Write the full motion model as $\mathbf{y}_k = f_y(\mathbf{y}_{k-1}, \mathbf{u}_k)$ with
$f_y = [f(\mathbf{x}, \mathbf{u});\ \mathbf{m}]$. Its Jacobians are

$$
\mathbf{F}_y = \begin{bmatrix} \mathbf{F} & \mathbf{0} \\ \mathbf{0} & \mathbf{I} \end{bmatrix},
\qquad
\mathbf{L}_y = \begin{bmatrix} \mathbf{L} \\ \mathbf{0} \end{bmatrix}.
$$

Then $\mathbf{P} \leftarrow \mathbf{F}_y\mathbf{P}\mathbf{F}_y^T + \mathbf{L}_y\mathbf{Q}_u\mathbf{L}_y^T$.
Multiplying out the blocks:
$\mathbf{P}_{xx} \leftarrow \mathbf{F}\mathbf{P}_{xx}\mathbf{F}^T + \mathbf{L}\mathbf{Q}_u\mathbf{L}^T$,
$\mathbf{P}_{xm} \leftarrow \mathbf{F}\mathbf{P}_{xm}\mathbf{I} = \mathbf{F}\mathbf{P}_{xm}$,
$\mathbf{P}_{mm} \leftarrow \mathbf{I}\mathbf{P}_{mm}\mathbf{I} = \mathbf{P}_{mm}$. The map block is
unchanged because the landmarks have identity dynamics and no process noise (static
world); only $3 \times 3 + 3 \times 2N$ entries change, so prediction is $O(N)$.

**(b)** Let $\mathbf{z} = \mathbf{z}^\star + \mathbf{v}$ with $\mathbf{v} \sim \mathcal{N}(\mathbf{0}, \mathbf{R})$
independent of $\mathbf{y}$. Linearize
$\boldsymbol{\ell} = g(\mathbf{x}, \mathbf{z}) \approx g(\hat{\mathbf{x}}, \mathbf{z}) + \mathbf{G}_x(\mathbf{x} - \hat{\mathbf{x}}) + \mathbf{G}_z\mathbf{v}'$,
where $\mathbf{v}'$ is the measurement error. Then, using the independence of $\mathbf{v}'$ from the state,

- $\operatorname{Cov}(\boldsymbol{\ell}) = \mathbf{G}_x\mathbf{P}_{xx}\mathbf{G}_x^T + \mathbf{G}_z\mathbf{R}\mathbf{G}_z^T$;
- $\operatorname{Cov}(\boldsymbol{\ell}, \mathbf{x}) = \mathbf{G}_x\operatorname{Cov}(\mathbf{x}, \mathbf{x}) = \mathbf{G}_x\mathbf{P}_{xx}$;
- $\operatorname{Cov}(\boldsymbol{\ell}, \mathbf{m}) = \mathbf{G}_x\operatorname{Cov}(\mathbf{x}, \mathbf{m}) = \mathbf{G}_x\mathbf{P}_{xm}$.

**(c)** Let $n = 3 + 2N$. Counting multiply-adds (2 flops each):

| Step | Work | Order |
|---|---|---|
| $\mathbf{S}$ from $\mathbf{P}_{xx}, \mathbf{P}_{xj}, \mathbf{P}_{jj}$ | $\approx 2 \cdot (5\cdot 5 \cdot 2 + 5 \cdot 2 \cdot 2)$ | $O(1)$ |
| $\mathbf{P}\mathbf{H}^T$ (only 5 columns of $\mathbf{P}$ used) | $2 \cdot 5 \cdot 2 \cdot n = 20n$ | $O(N)$ |
| $\mathbf{K} = (\mathbf{P}\mathbf{H}^T)\mathbf{S}^{-1}$ | $\approx 8n$ | $O(N)$ |
| $\hat{\mathbf{y}} \mathrel{+}= \mathbf{K}\boldsymbol{\nu}$ | $4n$ | $O(N)$ |
| $\mathbf{P} \mathrel{-}= (\mathbf{K}\mathbf{S})\mathbf{K}^T$ | $8n + 4n^2$ (half with symmetry) | $O(N^2)$ |

For large $N$ the cost is $\approx 4n^2 \approx 16N^2$ flops per observation (about
$2n^2$ exploiting symmetry); the Joseph form roughly triples it. Sparsity cannot
remove the $O(N^2)$ term because $\mathbf{K}$ is dense: after a few steps every landmark
is correlated with the robot, so every row of $\mathbf{P}\mathbf{H}^T$ is nonzero and
$\mathbf{K}\mathbf{S}\mathbf{K}^T$ is a dense rank-2 update. The posterior covariance is
dense, so even writing it down is $O(N^2)$.

## Exercise 2: Correspondences and the factorization

**(a)** Assumptions: (i) the world is static and the landmark prior factors,
$p(\mathbf{m} \mid \mathbf{x}_{1:k}, n_{1:k}) = \prod_j p(\boldsymbol{\ell}_j)$ (the prior does not depend
on the path or correspondences); (ii) each measurement depends only on the pose at
its time and on the landmark it came from,
$p(\mathbf{z}_\tau \mid \mathbf{x}_{1:k}, \mathbf{m}, \mathbf{z}_{\neq\tau}, n_{1:k}, \mathbf{u}_{1:k}) = p(\mathbf{z}_\tau \mid \mathbf{x}_\tau, \boldsymbol{\ell}_{n_\tau})$;
(iii) controls carry no information about the map beyond the path.

By the chain rule,
$p(\mathbf{x}_{1:k}, \mathbf{m} \mid \mathbf{z}, \mathbf{u}, n) = p(\mathbf{x}_{1:k} \mid \mathbf{z}, \mathbf{u}, n)\,p(\mathbf{m} \mid \mathbf{x}_{1:k}, \mathbf{z}, \mathbf{u}, n)$.
For the second factor, (iii) drops $\mathbf{u}$, and Bayes' rule with (i) and (ii) gives

$$
p(\mathbf{m} \mid \mathbf{x}_{1:k}, \mathbf{z}_{1:k}, n_{1:k})
\propto \prod_j p(\boldsymbol{\ell}_j) \prod_{\tau=1}^k p(\mathbf{z}_\tau \mid \mathbf{x}_\tau, \boldsymbol{\ell}_{n_\tau})
= \prod_j \Big[ p(\boldsymbol{\ell}_j) \prod_{\tau:\, n_\tau = j} p(\mathbf{z}_\tau \mid \mathbf{x}_\tau, \boldsymbol{\ell}_j) \Big].
$$

Each bracket depends on $\boldsymbol{\ell}_j$ only and, normalized, is
$p(\boldsymbol{\ell}_j \mid \mathbf{x}_{1:k}, \mathbf{z}_{1:k}, n_{1:k})$ (the same Bayes computation
restricted to landmark $j$'s measurements). A product of functions of disjoint
variables normalizes factor by factor, which proves the result.

**(b)** Take scalar landmark positions on a line through the known robot position
for simplicity: priors $p_1(a) = \mathcal{N}(a; \mu_1, \sigma_1^2)$, $p_2(b) = \mathcal{N}(b; \mu_2, \sigma_2^2)$,
and $z = \ell_n + v$, $v \sim \mathcal{N}(0, \sigma_v^2)$, $P(n=1) = P(n=2) = 1/2$. Then

$$
p(a, b \mid z) \propto \tfrac12 A(a)\,p_2(b) + \tfrac12 p_1(a)\,B(b),
\qquad A(a) = p_1(a)\mathcal{N}(z; a, \sigma_v^2),\ B(b) = p_2(b)\mathcal{N}(z; b, \sigma_v^2).
$$

A positive function $F(a, b)$ factors as $f(a)g(b)$ only if
$F(a,b)F(a',b') = F(a,b')F(a',b)$ for all arguments. Writing $F = AP + QB$ with
$P = p_2$, $Q = p_1$ and expanding,

$$
F(a,b)F(a',b') - F(a,b')F(a',b) = \big(A(a)Q(a') - Q(a)A(a')\big)\big(P(b)B(b') - P(b')B(b)\big).
$$

The first factor vanishes for all $a, a'$ only if $A/Q = \mathcal{N}(z; a, \sigma_v^2)$ is
constant in $a$, which it is not; likewise for the second. So the posterior is a
mixture of two product terms and does not factor. (Physically: learning that
landmark 1 is near $z$ makes it less likely that landmark 2 is.)

## Exercise 3: Units, $p_0$, and the birth gate

**(a)** With $\mathbf{z}_{cm} = \mathbf{D}\mathbf{z}_m$, $\mathbf{D} = \operatorname{diag}(100, 1)$, densities
transform by the Jacobian: $p_{cm}(\mathbf{z}_{cm}) = p_m(\mathbf{z}_m)/|\det\mathbf{D}| = p_m/100$.
For the Gaussian, $\boldsymbol{\nu}_{cm} = \mathbf{D}\boldsymbol{\nu}$ and $\mathbf{S}_{cm} = \mathbf{D}\mathbf{S}\mathbf{D}^T$, so
$\boldsymbol{\nu}_{cm}^T\mathbf{S}_{cm}^{-1}\boldsymbol{\nu}_{cm} = \boldsymbol{\nu}^T\mathbf{D}^T\mathbf{D}^{-T}\mathbf{S}^{-1}\mathbf{D}^{-1}\mathbf{D}\boldsymbol{\nu} = d^2$,
while $\det(2\pi\mathbf{S}_{cm}) = 10^4\det(2\pi\mathbf{S})$ divides the density by $\sqrt{10^4} = 100$.
The chapter's cell shows the log shift of exactly $\ln 100 = 4.605$.

**(b)** If every particle's log weight for observation $a$ changes by
$f_{ia} + c_a$, the normalized weights are
$w_i e^{f_{ia} + c_a}/\sum_l w_l e^{f_{la} + c_a} = w_i e^{f_{ia}}/\sum_l w_l e^{f_{la}}$.
This needs every particle to receive exactly one factor for observation $a$, which is
the book's rule. Without $p_0$ in centimeters, the factor is
$\ln\mathcal{N}_m - \ln 100$ for a match and $0$ otherwise. Add $c_a = \ln 100$ to every
particle's factor: matches get $\ln\mathcal{N}_m$ and everything else gets $\ln 100$, i.e.
$p_0 = 100\ \text{m}^{-1}\text{rad}^{-1}$ in meters.

**(c)** $p_0 = 1/(30\,\text{m}\cdot\pi) = 0.0106\ \text{m}^{-1}\text{rad}^{-1}$;
$\sqrt{\det\mathbf{R}} = 0.3 \times 0.01745 = 5.24\times10^{-3}$;
$d^2_\times = -2\ln(2\pi \cdot 0.0106 \cdot 5.24\times 10^{-3}) = 15.9$
(a 99.965% chi-square point). The 13.82 birth gate sits *below* the crossover, so an
observation with $13.82 < d^2 < 15.9$ spawns a tentative landmark although the
match is still the better explanation. The tentative stage makes this fairly
harmless, but it is inconsistent. Acceptable fixes, any of which earns full credit
with a reason: raise the birth gate to about 16; or use a footprint that reflects
where trees are actually detected (crossover 13.82 needs
$p_0 = e^{-6.91}/(2\pi\cdot 5.24\times10^{-3}) = 0.030\ \text{m}^{-1}\text{rad}^{-1}$, e.g. a
10.5 m effective range over $\pi$); or keep both and document the band. The point
is to compute both numbers.

## Exercise 4: Hunting the residual duplicates

Running `run_fastslam(FastSLAM, seed, log_p0=LOG_P0)` for seeds 0 to 199 gives
**5 failures** (2.5%), all with four confirmed trees: seeds 20, 26, 53, 58, 118.
The duplicated tree in the best particle is tree 0 (seed 20), tree 2 (26, 58, 118),
and tree 1 (53).

An oracle (per-landmark EKFs fed the true poses and true labels) has worst
re-observation $d^2$ of 12.6 (seed 20), **16.5** (26; scan 6, tree 2, 6.3 m),
**13.9** (53; scan 5, tree 1, 4.4 m), 9.7 (58), **17.7** (118; scan 7, tree 2, 4.8 m).
So the oracle reproduces three of the five (26, 53, 118): the measurement itself
lands beyond the 99.9% gate, every particle sees the same noisy measurement, and
every particle spawns the duplicate together. The duplicated tree matches the
oracle's offending tree in all three. The other two (20, 58) are FastSLAM 1.0's
doing: the oracle's innovation is borderline (9.7 to 12.6) and the particles'
accumulated pose error pushes it over 13.82. Expected rate from the gate alone:
about 30 re-observations per run times $10^{-3}$, i.e. about 3% of runs have a
breach; not every breach yields a confirmed duplicate.

Remedies over the same 200 seeds:

| Filter | Failures / 200 |
|---|---|
| FastSLAM 1.0, chapter settings | 5 |
| FastSLAM 1.0, filter $\mathbf{R} \times 1.5$ | 0 |
| FastSLAM 2.0 | 7 (one of them with only two trees) |

Inflating $\mathbf{R}$ widens both gates relative to the true noise, so a
one-in-a-thousand breach becomes much rarer, and it also absorbs some
linearization error; the price is a less confident map. FastSLAM 2.0 does not help
here: the failures are about the gate statistics of a shared measurement, not about
the proposal, and in this world (sensor not much sharper than odometry) 2.0's
proposal advantage is small. Students should notice that the 2.0 failures overlap
the oracle seeds (26, 53, 118).

## Exercise 5: The FastSLAM 2.0 proposal

**(a)** With $\boldsymbol{\delta} = \mathbf{x} - \bar{\mathbf{x}}$ and $\boldsymbol{\nu} = \mathbf{z} - \hat{\mathbf{z}}$,
the product of $\mathcal{N}(\boldsymbol{\delta}; \mathbf{0}, \mathbf{Q}_x)$ and
$\mathcal{N}(\boldsymbol{\nu}; \mathbf{H}_x\boldsymbol{\delta}, \mathbf{S}_\ell)$ has exponent

$$
-\tfrac12\left[\boldsymbol{\delta}^T(\mathbf{Q}_x^{-1} + \mathbf{H}_x^T\mathbf{S}_\ell^{-1}\mathbf{H}_x)\boldsymbol{\delta}
- 2\boldsymbol{\delta}^T\mathbf{H}_x^T\mathbf{S}_\ell^{-1}\boldsymbol{\nu} + \boldsymbol{\nu}^T\mathbf{S}_\ell^{-1}\boldsymbol{\nu}\right].
$$

Completing the square in $\boldsymbol{\delta}$ gives a Gaussian with
$\boldsymbol{\Sigma}_x = (\mathbf{Q}_x^{-1} + \mathbf{H}_x^T\mathbf{S}_\ell^{-1}\mathbf{H}_x)^{-1}$ and
mean $\boldsymbol{\delta}^\star = \boldsymbol{\Sigma}_x\mathbf{H}_x^T\mathbf{S}_\ell^{-1}\boldsymbol{\nu}$, i.e.
$\boldsymbol{\mu}_x = \bar{\mathbf{x}} + \boldsymbol{\Sigma}_x\mathbf{H}_x^T\mathbf{S}_\ell^{-1}(\mathbf{z} - \hat{\mathbf{z}})$.

**(b)** The matrix inversion lemma gives
$(\mathbf{Q}^{-1} + \mathbf{H}^T\mathbf{S}^{-1}\mathbf{H})^{-1} = \mathbf{Q} - \mathbf{Q}\mathbf{H}^T(\mathbf{H}\mathbf{Q}\mathbf{H}^T + \mathbf{S})^{-1}\mathbf{H}\mathbf{Q} = (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{Q}$
with $\mathbf{K} = \mathbf{Q}\mathbf{H}^T(\mathbf{H}\mathbf{Q}\mathbf{H}^T + \mathbf{S})^{-1}$, and the push-through
identity $(\mathbf{Q}^{-1} + \mathbf{H}^T\mathbf{S}^{-1}\mathbf{H})^{-1}\mathbf{H}^T\mathbf{S}^{-1} = \mathbf{Q}\mathbf{H}^T(\mathbf{H}\mathbf{Q}\mathbf{H}^T + \mathbf{S})^{-1} = \mathbf{K}$
gives the same mean. The Kalman form needs only $\mathbf{H}\mathbf{Q}\mathbf{H}^T + \mathbf{S}$ to be
invertible, which holds because $\mathbf{S}_\ell \succeq \mathbf{R} \succ 0$. For singular
$\mathbf{Q}_x$ it is the limit of the information form with $\mathbf{Q}_x + \epsilon\mathbf{I}$,
$\epsilon \to 0$. The resulting $\boldsymbol{\Sigma}_x$ is singular too, which is correct: the
proposal cannot move the pose in a direction no process noise can push it.

**(c)** $\int \mathcal{N}(\mathbf{z}; \hat{\mathbf{z}} + \mathbf{H}_x\boldsymbol{\delta}, \mathbf{S}_\ell)\,\mathcal{N}(\boldsymbol{\delta}; \mathbf{0}, \mathbf{Q}_x)\,d\boldsymbol{\delta}$
is the marginal of $\mathbf{z} = \hat{\mathbf{z}} + \mathbf{H}_x\boldsymbol{\delta} + \boldsymbol{\epsilon}$ with independent
Gaussian $\boldsymbol{\delta}$ and $\boldsymbol{\epsilon}$: mean $\hat{\mathbf{z}}$, covariance
$\mathbf{H}_x\mathbf{Q}_x\mathbf{H}_x^T + \mathbf{S}_\ell$. This is the optimal-proposal weight
$p(\mathbf{z} \mid \mathbf{x}_{k-1}, \mathbf{u}_k, \mathbf{m}, n)$, independent of the sampled pose.

## Exercise 6: A softer discard rule

Implementation: in `update`, for `decision == DISCARD` add
`max(max_j gaussian_log_likelihood(*cache[a, j]), self.log_p0)` instead of
`self.log_p0`. The weight factor versus $d^2$ (for $\mathbf{S} = \mathbf{R}$) is the Gaussian
$0.91\,e^{-(d^2 - 5.99)/2}$ up to the gate under both rules. Beyond it the hard rule
drops to $p_0 = 0.020$, a factor of about 45 at the gate (about 23 for a young
landmark with $\mathbf{S} = 2\mathbf{R}$); the soft rule continues the Gaussian until it
meets $p_0$ at $d^2 \approx 13.6$, then stays flat.

Results (best-particle confirmed-tree counts, final position error, distinct
scan-1 ancestors):

| World, rule | 20 seeds exact | 100 seeds exact | Final error [m] (20 seeds) | Ancestors (20 seeds) |
|---|---|---|---|---|
| 3 trees, hard ($p_0$) | 20/20 | 96/100 | 0.29 | 3.0 |
| 3 trees, soft | 20/20 | 96/100 | 0.22 | 4.3 |
| 4 trees (extra tree 0.8 m from tree 1), hard | 20/20 | 97/100 | 0.27 | 2.5 |
| 4 trees, soft | 20/20 | 99/100 | 0.26 | 3.4 |

It barely matters for the map here, but the softer rule keeps more path diversity
(fewer large weight jumps, so fewer resampling-driven collapses) and slightly lower
position error. It matters when discards are frequent: dense trees, a tight
association gate, clutter, or poses that hover near the gate edge. Then the hard
rule's factor-of-45 cliff at $d^2 = 5.99$ makes particle weights hypersensitive to
which side of the gate an innovation falls on.
