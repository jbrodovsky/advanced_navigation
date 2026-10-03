# Solutions: The Information Filter (`information-filter.qmd`)

Instructor-only. Prior quantities are written $\bar{\mathbf{x}} = \hat{\mathbf{x}}_{k|k-1}$,
$\bar{\mathbf{P}} = \mathbf{P}_{k|k-1}$; posterior $\hat{\mathbf{x}}$, $\mathbf{P}$.

## Exercise 1: Two roads to the same update

**(a)** With $\mathbf{S} = \mathbf{H}\bar{\mathbf{P}}\mathbf{H}^T + \mathbf{R}$ and
$\mathbf{K} = \bar{\mathbf{P}}\mathbf{H}^T\mathbf{S}^{-1}$, the matrix inversion lemma
$(\mathbf{A} + \mathbf{U}\mathbf{C}\mathbf{V})^{-1} = \mathbf{A}^{-1} - \mathbf{A}^{-1}\mathbf{U}(\mathbf{C}^{-1} + \mathbf{V}\mathbf{A}^{-1}\mathbf{U})^{-1}\mathbf{V}\mathbf{A}^{-1}$
with $\mathbf{A} = \bar{\mathbf{P}}^{-1}$, $\mathbf{U} = \mathbf{H}^T$, $\mathbf{C} = \mathbf{R}^{-1}$, $\mathbf{V} = \mathbf{H}$ gives

$$
(\bar{\mathbf{P}}^{-1} + \mathbf{H}^T\mathbf{R}^{-1}\mathbf{H})^{-1}
= \bar{\mathbf{P}} - \bar{\mathbf{P}}\mathbf{H}^T(\mathbf{R} + \mathbf{H}\bar{\mathbf{P}}\mathbf{H}^T)^{-1}\mathbf{H}\bar{\mathbf{P}}
= \bar{\mathbf{P}} - \mathbf{K}\mathbf{H}\bar{\mathbf{P}} = (\mathbf{I} - \mathbf{K}\mathbf{H})\bar{\mathbf{P}} = \mathbf{P}.
$$

So $\mathbf{P}^{-1} = \bar{\mathbf{P}}^{-1} + \mathbf{H}^T\mathbf{R}^{-1}\mathbf{H}$.

**(b)** First, $\mathbf{K} = \mathbf{P}\mathbf{H}^T\mathbf{R}^{-1}$:

$$
\mathbf{P}\mathbf{H}^T\mathbf{R}^{-1}
= \bar{\mathbf{P}}\mathbf{H}^T\mathbf{R}^{-1} - \mathbf{K}\mathbf{H}\bar{\mathbf{P}}\mathbf{H}^T\mathbf{R}^{-1}
= \mathbf{K}\mathbf{S}\mathbf{R}^{-1} - \mathbf{K}\mathbf{H}\bar{\mathbf{P}}\mathbf{H}^T\mathbf{R}^{-1}
= \mathbf{K}(\mathbf{S} - \mathbf{H}\bar{\mathbf{P}}\mathbf{H}^T)\mathbf{R}^{-1} = \mathbf{K},
$$

using $\bar{\mathbf{P}}\mathbf{H}^T = \mathbf{K}\mathbf{S}$. Then

$$
\mathbf{P}^{-1}\hat{\mathbf{x}}
= \mathbf{P}^{-1}\bar{\mathbf{x}} + \mathbf{P}^{-1}\mathbf{K}(\mathbf{z} - \mathbf{H}\bar{\mathbf{x}})
= \mathbf{P}^{-1}\bar{\mathbf{x}} + \mathbf{H}^T\mathbf{R}^{-1}\mathbf{z} - \mathbf{H}^T\mathbf{R}^{-1}\mathbf{H}\bar{\mathbf{x}}
= \bar{\mathbf{P}}^{-1}\bar{\mathbf{x}} + \mathbf{H}^T\mathbf{R}^{-1}\mathbf{z},
$$

using part (a) in the last step. This is $\boldsymbol{\xi} = \bar{\boldsymbol{\xi}} + \mathbf{H}^T\mathbf{R}^{-1}\mathbf{z}$.

## Exercise 2: Where does the information go?

With $\mathbf{P} = \boldsymbol{\Omega}^{-1}$, $\mathbf{F}\boldsymbol{\Omega}^{-1}\mathbf{F}^T = (\mathbf{F}^{-T}\boldsymbol{\Omega}\mathbf{F}^{-1})^{-1} = \mathbf{M}^{-1}$.
The lemma with $\mathbf{A} = \mathbf{M}^{-1}$, $\mathbf{U} = \mathbf{V} = \mathbf{I}$, $\mathbf{C} = \mathbf{Q}$:

$$
\bar{\boldsymbol{\Omega}} = (\mathbf{M}^{-1} + \mathbf{Q})^{-1} = \mathbf{M} - \mathbf{M}(\mathbf{Q}^{-1} + \mathbf{M})^{-1}\mathbf{M} = (\mathbf{I} - \mathbf{G})\mathbf{M},
\qquad \mathbf{G} = \mathbf{M}(\mathbf{M} + \mathbf{Q}^{-1})^{-1}.
$$

For the information vector, $\bar{\boldsymbol{\xi}} = \bar{\boldsymbol{\Omega}}(\mathbf{F}\hat{\mathbf{x}} + \mathbf{B}\mathbf{u}) = (\mathbf{I} - \mathbf{G})(\mathbf{M}\mathbf{F}\hat{\mathbf{x}} + \mathbf{M}\mathbf{B}\mathbf{u})$
and $\mathbf{M}\mathbf{F}\hat{\mathbf{x}} = \mathbf{F}^{-T}\boldsymbol{\Omega}\hat{\mathbf{x}} = \mathbf{F}^{-T}\boldsymbol{\xi}$, giving the chapter's
equation.

**Scalar random walk** ($F = 1$, $M = \Omega$):

$$
\bar{\Omega} = \Omega - \frac{\Omega^2}{\Omega + 1/Q} = \frac{\Omega\,(1/Q)}{\Omega + 1/Q} = \left(\frac{1}{\Omega} + Q\right)^{-1}.
$$

This is the "parallel resistors" combination of $\Omega$ and $1/Q$, so it is
smaller than both. Prediction can only lose information, and one step can never
leave more than $1/Q$ of it, however well the state was known. For a GNSS
receiver in a tunnel (no updates), $1/\bar{\Omega}_k = 1/\Omega_0 + kQ$: the
variance grows linearly and the information decays like $1/(kQ)$, which is the
dead-reckoning random walk. The first fix after the tunnel adds $1/R$ to a
small number, so it dominates the estimate.

## Exercise 3: Marginalization by Schur complement

The block inverse of $\boldsymbol{\Omega}$ has upper-left block
$\mathbf{P}_{aa} = (\boldsymbol{\Omega}_{aa} - \boldsymbol{\Omega}_{ab}\boldsymbol{\Omega}_{bb}^{-1}\boldsymbol{\Omega}_{ba})^{-1}$,
so the marginal information is $\mathbf{P}_{aa}^{-1}$, the Schur complement. For the vector, write
$\boldsymbol{\Omega}\hat{\mathbf{x}} = \boldsymbol{\xi}$ by blocks:

$$
\boldsymbol{\Omega}_{aa}\hat{\mathbf{a}} + \boldsymbol{\Omega}_{ab}\hat{\mathbf{b}} = \boldsymbol{\xi}_a,
\qquad
\boldsymbol{\Omega}_{ba}\hat{\mathbf{a}} + \boldsymbol{\Omega}_{bb}\hat{\mathbf{b}} = \boldsymbol{\xi}_b.
$$

Eliminate $\hat{\mathbf{b}} = \boldsymbol{\Omega}_{bb}^{-1}(\boldsymbol{\xi}_b - \boldsymbol{\Omega}_{ba}\hat{\mathbf{a}})$:

$$
(\boldsymbol{\Omega}_{aa} - \boldsymbol{\Omega}_{ab}\boldsymbol{\Omega}_{bb}^{-1}\boldsymbol{\Omega}_{ba})\hat{\mathbf{a}}
= \boldsymbol{\xi}_a - \boldsymbol{\Omega}_{ab}\boldsymbol{\Omega}_{bb}^{-1}\boldsymbol{\xi}_b,
$$

which identifies $\boldsymbol{\xi}_a^\text{marg}$. (This elimination is exactly one step of Gaussian
elimination; variable ordering in sparse solvers is the art of choosing which
variables to eliminate first to minimize fill-in.)

**Coding part.** One convenient construction: $N$ poses, one landmark per pose
index $j$ seen by poses $j$, $j+1$, $j+2$ (truncate at the end), odometry between
consecutive poses, and a prior on pose 0. Build $\boldsymbol{\Omega}$ with
`add_relative_factor` from the chapter, count `np.count_nonzero(np.abs(Omega) > 1e-12)`,
then marginalize all poses except the last with the Schur complement and count
again.

Expected: smoothing has $3N - 2$ pose-pose entries, about $N$ landmark diagonal
entries, and about $2 \cdot 3N$ pose-landmark entries, so roughly $10N$
nonzeros: linear growth. Filtering leaves a dense $(L + 1) \times (L + 1)$ block
with $L \approx N$ landmarks: $\approx N^2$ nonzeros, quadratic growth. At
$N = 100$ the counts are on the order of $10^3$ versus $10^4$, and the gap
widens linearly with $N$.

## Exercise 4: Cocked hats and cut angles

Two LOPs with unit normals $\mathbf{a}_1, \mathbf{a}_2$ separated by the cut
angle $\theta$, each with cross-line standard deviation $\sigma_d = r\sigma_\vartheta$
(equal ranges). Then
$\boldsymbol{\Omega} = (\mathbf{a}_1\mathbf{a}_1^T + \mathbf{a}_2\mathbf{a}_2^T)/\sigma_d^2$.
The matrix $\mathbf{a}_1\mathbf{a}_1^T + \mathbf{a}_2\mathbf{a}_2^T$ has eigenvalues
$1 \pm |\cos\theta|$ (eigenvectors along the bisectors), so the fix covariance has
semi-axes

$$
\sigma_{\pm} = \frac{\sigma_d}{\sqrt{1 \mp |\cos\theta|}},
\qquad
\text{semi-major axis} = \frac{\sigma_d}{\sqrt{1 - |\cos\theta|}}.
$$

The semi-major axis is minimized at $\theta = 90^\circ$, where both semi-axes equal
$\sigma_d$. At $10^\circ$ (or $170^\circ$) it is $\sigma_d/\sqrt{1 - \cos 10^\circ} \approx 8.1\,\sigma_d$.
The plot is symmetric about $90^\circ$ and blows up at $0^\circ$ and $180^\circ$, where the
two lines coincide and the along-line direction is unobserved (rank-1
$\boldsymbol{\Omega}$, as after the first LOP in the chapter).

A celestial line of position is perpendicular to the star's azimuth, so the
same geometry applies with azimuth differences: two stars about $90^\circ$ apart in
azimuth give the roundest error ellipse. A third star close in azimuth to one of
the first two contributes a normal nearly parallel to an existing one; its
information lands almost entirely in an already well-determined direction and
barely shrinks the semi-major axis. (Navigators use three stars about $120^\circ$
apart, which also lets the cocked hat reveal a constant bias.)

```python
import numpy as np

sigma_d = 1.0
for theta_deg in (10, 30, 60, 90, 120, 150, 170):
    th = np.deg2rad(theta_deg)
    a1, a2 = np.array([1.0, 0.0]), np.array([np.cos(th), np.sin(th)])
    Omega = (np.outer(a1, a1) + np.outer(a2, a2)) / sigma_d**2
    semi_major = np.sqrt(np.linalg.eigvalsh(np.linalg.inv(Omega)).max())
    print(theta_deg, round(semi_major, 2))
```

## Exercise 5: Distributed fusion and double counting

**(a)** Each ship computes $\boldsymbol{\Omega}_i = \boldsymbol{\Omega}_0 + \mathbf{H}_i^T\mathbf{R}_i^{-1}\mathbf{H}_i$
and $\boldsymbol{\xi}_i = \boldsymbol{\xi}_0 + \mathbf{H}_i^T\mathbf{R}_i^{-1}\mathbf{z}_i$. The centralized
posterior uses the prior once:
$\boldsymbol{\Omega} = \boldsymbol{\Omega}_0 + \sum_i\mathbf{H}_i^T\mathbf{R}_i^{-1}\mathbf{H}_i = \boldsymbol{\Omega}_1 + \boldsymbol{\Omega}_2 - \boldsymbol{\Omega}_0$,
and likewise $\boldsymbol{\xi} = \boldsymbol{\xi}_1 + \boldsymbol{\xi}_2 - \boldsymbol{\xi}_0$. Naive addition counts the
prior twice.

**(b)** Scalar illustration (the 2D simulation behaves the same way): prior
variance $P_0 = 1$, two measurements with $R = 1$. Centralized: $\Omega = 3$,
$P = 1/3$, weights $1/3$ on the prior and on each measurement, so the actual error
variance is $1/3$ and NEES averages 1. Naive: $\Omega = 4$, claimed $P = 0.25$,
weights $1/2$ on the prior and $1/4$ on each measurement, so the actual error
variance is $0.25 + 2(0.0625) = 0.375$ and NEES averages $0.375/0.25 = 1.5$. In
2D with isotropic noise the average NEES is about 3 instead of 2. The fused
estimate is overconfident and pulled toward the prior.

The 1000-trial simulation should show the centralized NEES inside its chi-square
band and the naive NEES well above it. The general difficulty: in a network,
nodes exchange estimates that already contain shared information (common priors,
and each other's earlier messages), and a node cannot tell what is common just
from $(\boldsymbol{\Omega}_i, \boldsymbol{\xi}_i)$. Exact remedies track the common information explicitly (channel
filters on tree-structured networks); conservative remedies such as covariance
intersection fuse without knowing the correlation, at the price of optimality.
