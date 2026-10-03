# Solutions: Mathematical Reference and Derivations (`mathematical-reference.qmd`)

Instructor-only. Exercise numbers match
`## Exercises {#sec-math-reference-exercises}`. Each code block is
self-contained and uses only numpy and scipy.

---

## Exercise 1: Reading the NIS

**(a)** $m = 3$, $N = 200$, so $Nm = 600$ degrees of freedom. The two-sided 95%
interval for the time-averaged NIS is

$$
\left[\frac{\chi^2_{600}(0.025)}{200},\; \frac{\chi^2_{600}(0.975)}{200}\right] \approx [2.67,\; 3.35].
$$

(A normal approximation gives the same thing: $\chi^2_{600}$ has mean 600 and
standard deviation $\sqrt{1200} = 34.6$, so $600 \pm 1.96 \times 34.6$ over 200.)
The observed 4.6 is far above the upper bound, so the filter fails the test.

**(b)** NIS above the interval means the innovations are larger than
$\mathbf{S}_k$ predicts: the filter is **overconfident**. Since
$\mathbf{S} = \mathbf{H}\mathbf{P}_{k|k-1}\mathbf{H}^T + \mathbf{R}$, either the
predicted covariance or $\mathbf{R}$ is too small. In a loosely coupled GNSS/INS
filter, the usual first suspect is $\mathbf{Q}$ (IMU noise and bias instability
underestimated, so $\mathbf{P}_{k|k-1}$ collapses between fixes), with the GNSS
position $\mathbf{R}$ a close second (receiver-reported accuracies are often
optimistic, especially under multipath). A practical way to tell them apart is to
look at how NIS behaves with the update interval: if NIS grows when fixes are
sparser, the prediction ($\mathbf{Q}$) is the problem; if it is high even at the
full rate, look at $\mathbf{R}$. Students may also note unmodeled lever arms or
time offsets, which produce biased innovations and therefore high NIS.

**(c)** The interval is derived from the fact that a sum of $N$ *independent*
$\chi^2_m$ variables is $\chi^2_{Nm}$; independence follows from white
innovations. With positively correlated innovations the mean of each term can
still be $m$, but the sum has a larger variance than $\chi^2_{Nm}$ (fewer
effective independent samples), so the interval is too narrow and the test
raises false alarms. Worse, time-correlated measurement errors such as slowly
varying multipath violate the filter's white-noise model: the filter treats
repeated fixes as independent, shrinks $\mathbf{P}$ too quickly, and becomes
genuinely overconfident. The remedy is to test innovation whiteness
(autocorrelation) alongside the NIS mean (@sec-kf-consistency), and either model
the correlated error as a state or inflate/decimate the measurements.

```python
from scipy.stats import chi2
N, m = 200, 3
lo, hi = chi2.ppf(0.025, N * m) / N, chi2.ppf(0.975, N * m) / N
print(f"95% interval for mean NIS: [{lo:.3f}, {hi:.3f}]; observed 4.6 -> "
      f"{'overconfident' if 4.6 > hi else 'ok'}")
```

---

## Exercise 2: Why accelerometer leveling cannot find north

**(a)** The measurement is $\mathbf{f}^b = \mathbf{C}_n^b\mathbf{u}^n$ with
$\mathbf{u}^n = -\mathbf{g}^n$. By @eq-math-reference-jacobian-nav-transpose,

$$
\frac{\partial\mathbf{f}^b}{\partial\delta\boldsymbol{\theta}} = \hat{\mathbf{C}}_n^b[-\mathbf{g}^n]_\times = -\hat{\mathbf{C}}_n^b[\mathbf{g}^n]_\times .
$$

Derivation from scratch: with the book's convention
$\mathbf{C}_b^n \approx (\mathbf{I} + [\delta\boldsymbol{\theta}]_\times)\hat{\mathbf{C}}_b^n$,
we get $\mathbf{C}_n^b \approx \hat{\mathbf{C}}_n^b(\mathbf{I} - [\delta\boldsymbol{\theta}]_\times)$,
so $\mathbf{f}^b \approx \hat{\mathbf{f}}^b + \hat{\mathbf{C}}_n^b[\delta\boldsymbol{\theta}]_\times\mathbf{g}^n = \hat{\mathbf{f}}^b - \hat{\mathbf{C}}_n^b[\mathbf{g}^n]_\times\delta\boldsymbol{\theta}$.

**(b)** With $\mathbf{g}^n = [0, 0, g]^T$,

$$
[\mathbf{g}^n]_\times = \begin{bmatrix} 0 & -g & 0 \\ g & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix},
$$

whose third column is zero. Multiplying on the left by any
$-\hat{\mathbf{C}}_n^b$ keeps it zero. The third component of
$\delta\boldsymbol{\theta}$ (in NED) is a rotation about the down axis, i.e. a
heading error. Rotating the vehicle about the gravity vector does not change the
gravity vector, so the accelerometer cannot see heading error: roll and pitch
are observable from leveling, yaw is not. Heading must come from another source
(magnetometer, GNSS velocity during acceleration or dual-antenna GNSS, a known
initial heading, or gyrocompassing with a gyroscope good enough to sense Earth
rate, which MEMS gyroscopes generally are not).

**(c)**

```python
import numpy as np
from scipy.spatial.transform import Rotation

def skew(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])

g_n = np.array([0.0, 0.0, 9.81])
C_bn_hat = Rotation.from_euler("ZYX", [0.7, 0.1, -0.05]).as_matrix()   # yaw, pitch, roll -> C_b^n

def f_b(dtheta):
    C_bn = Rotation.from_rotvec(dtheta).as_matrix() @ C_bn_hat
    return -C_bn.T @ g_n

eps = 1e-6
J_num = np.column_stack([(f_b(eps * e) - f_b(-eps * e)) / (2 * eps) for e in np.eye(3)])
J_ana = -C_bn_hat.T @ skew(g_n)
print("analytic == numerical:", np.allclose(J_num, J_ana, atol=1e-6))
print("third column (heading):", J_num[:, 2].round(9))
```

---

## Exercise 3: Filter versus smoother in the batch view

**(a)** With $p_0 = q = r = 1$ and $\hat{x}_{0|0} = 0$,

$$
J = x_0^2 + (x_1 - x_0)^2 + (x_2 - x_1)^2 + (z_1 - x_1)^2 + (z_2 - x_2)^2 .
$$

Setting $\partial J/\partial x_i = 0$ (and dividing by 2):

$$
\begin{aligned}
x_0 - (x_1 - x_0) &= 0 &&\Rightarrow\; 2x_0 - x_1 = 0, \\
(x_1 - x_0) - (x_2 - x_1) - (z_1 - x_1) &= 0 &&\Rightarrow\; -x_0 + 3x_1 - x_2 = z_1, \\
(x_2 - x_1) - (z_2 - x_2) &= 0 &&\Rightarrow\; -x_1 + 2x_2 = z_2 .
\end{aligned}
$$

For general $p_0, q, r$ the same system has coefficients
$\begin{bmatrix} 1/p_0 + 1/q & -1/q & 0 \\ -1/q & 2/q + 1/r & -1/q \\ 0 & -1/q & 1/q + 1/r \end{bmatrix}$
and right-hand side $[0, z_1/r, z_2/r]^T$: a tridiagonal information matrix,
as expected for a chain.

**(b)** With $z_1 = 1$, $z_2 = 3$: $x_0 = x_1/2$; substituting,
$2.5x_1 - x_2 = 1$ and $x_2 = (3 + x_1)/2$, so $2x_1 = 2.5$:

$$
x_0^* = 0.625, \qquad x_1^* = 1.25, \qquad x_2^* = 2.125 .
$$

Kalman filter: predict $P = 2$, $K = 2/3$, $\hat{x}_{1|1} = 2/3 \approx 0.667$,
$P_{1|1} = 2/3$; predict $P = 5/3$, $K = 5/8$,
$\hat{x}_{2|2} = 2/3 + (5/8)(3 - 2/3) = 2.125$.

So $x_2^* = \hat{x}_{2|2} = 2.125$ but $x_1^* = 1.25 \ne \hat{x}_{1|1} = 0.667$.

**(c)** The batch estimate of $x_1$ uses $z_2 = 3$ (which, through the random
walk, says $x_1$ was probably larger than $z_1$ suggested), while the filter's
$\hat{x}_{1|1}$ only had $z_1$; for the last state both use exactly the same
data, so they agree.

```python
import numpy as np

A = np.array([[2.0, -1.0, 0.0], [-1.0, 3.0, -1.0], [0.0, -1.0, 2.0]])
b = np.array([0.0, 1.0, 3.0])
print("batch MAP x0, x1, x2:", np.linalg.solve(A, b))

x, P = 0.0, 1.0
for z in (1.0, 3.0):
    P += 1.0
    K = P / (P + 1.0)
    x += K * (z - x)
    P *= 1 - K
    print(f"KF estimate after z = {z}: {x:.4f}")
```

---

## Exercise 4: Joseph form under a wrong gain

**(a)** Expand the Joseph form with $\mathbf{S} = \mathbf{H}\mathbf{P}\mathbf{H}^T + \mathbf{R}$:

$$
\begin{aligned}
(\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}(\mathbf{I} - \mathbf{K}\mathbf{H})^T + \mathbf{K}\mathbf{R}\mathbf{K}^T
&= (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P} - \mathbf{P}\mathbf{H}^T\mathbf{K}^T + \mathbf{K}\mathbf{H}\mathbf{P}\mathbf{H}^T\mathbf{K}^T + \mathbf{K}\mathbf{R}\mathbf{K}^T \\
&= (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P} - \mathbf{P}\mathbf{H}^T\mathbf{K}^T + \mathbf{K}\mathbf{S}\mathbf{K}^T .
\end{aligned}
$$

With the optimal gain $\mathbf{K} = \mathbf{P}\mathbf{H}^T\mathbf{S}^{-1}$,
$\mathbf{K}\mathbf{S}\mathbf{K}^T = \mathbf{P}\mathbf{H}^T\mathbf{S}^{-1}\mathbf{S}\mathbf{K}^T = \mathbf{P}\mathbf{H}^T\mathbf{K}^T$,
so the last two terms cancel and the Joseph form equals
$(\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}$.

**(b)** For an arbitrary gain $\mathbf{K}'$, the true posterior error covariance
is the Joseph expression (it follows from
$\mathbf{e}^+ = (\mathbf{I} - \mathbf{K}'\mathbf{H})\mathbf{e}^- - \mathbf{K}'\mathbf{v}$
with no optimality assumption). From the expansion above, the short form's error
is

$$
\text{Joseph}(\mathbf{K}') - (\mathbf{I} - \mathbf{K}'\mathbf{H})\mathbf{P} = \mathbf{K}'\mathbf{S}\mathbf{K}'^T - \mathbf{P}\mathbf{H}^T\mathbf{K}'^T .
$$

Write $\mathbf{K}' = \mathbf{K} + \delta\mathbf{K}$ with
$\delta\mathbf{K} = \varepsilon\|\mathbf{K}\|_F\mathbf{D}$, and use
$\mathbf{P}\mathbf{H}^T = \mathbf{K}\mathbf{S}$. The error becomes
$\delta\mathbf{K}\,\mathbf{S}\mathbf{K}^T + \delta\mathbf{K}\,\mathbf{S}\,\delta\mathbf{K}^T$:
**first order** in $\varepsilon$, and since $\delta\mathbf{K}\,\mathbf{S}\mathbf{K}^T$
is not symmetric in general, the short form's result is asymmetric to first
order as well. The code shows both shrinking tenfold per tenfold reduction in
$\varepsilon$.

**(c)** Joseph is quadratic in the gain,
$\text{Joseph}(\mathbf{K}') = \mathbf{P} - \mathbf{K}'\mathbf{H}\mathbf{P} - \mathbf{P}\mathbf{H}^T\mathbf{K}'^T + \mathbf{K}'\mathbf{S}\mathbf{K}'^T$,
and its first-order variation vanishes at the optimal gain (the terms
$-\delta\mathbf{K}\mathbf{H}\mathbf{P} + \delta\mathbf{K}\mathbf{S}\mathbf{K}^T$ cancel
because $\mathbf{S}\mathbf{K}^T = \mathbf{H}\mathbf{P}$, and likewise for their
transposes). Exactly,

$$
\text{Joseph}(\mathbf{K} + \delta\mathbf{K}) - \mathbf{P}^+_{\text{opt}} = \delta\mathbf{K}\,\mathbf{S}\,\delta\mathbf{K}^T,
$$

which is **second order** and positive semidefinite (any gain other than the
optimal one does worse, as it should). The code confirms a hundredfold reduction
per tenfold reduction in $\varepsilon$. Interpretation: when the gain carries
round-off error, the Joseph form still reports, to second order, the covariance
the filter actually achieves, and it is symmetric by construction. The short
form reports a covariance that is wrong to first order and asymmetric, and those
errors accumulate over many updates until $\mathbf{P}$ can become indefinite.

**(d)** With $\mathbf{K}' = (1+\varepsilon)\mathbf{K}$, the short form is
$\mathbf{P} - (1+\varepsilon)\mathbf{K}\mathbf{H}\mathbf{P} = \mathbf{P} - (1+\varepsilon)\mathbf{P}\mathbf{H}^T\mathbf{S}^{-1}\mathbf{H}\mathbf{P}$,
which is symmetric because $\mathbf{P}\mathbf{H}^T\mathbf{S}^{-1}\mathbf{H}\mathbf{P}$
is. Its error is $\varepsilon(1+\varepsilon)\mathbf{K}\mathbf{S}\mathbf{K}^T$, still
first order, but symmetric. Asymmetry is therefore a symptom of only some gain
errors, and checking symmetry is not a substitute for using the Joseph form.

```python
import numpy as np

rng = np.random.default_rng(1977)       # same draws as the chapter's square-root check
n_x, n_z = 4, 2
A = rng.standard_normal((n_x, n_x))
P = A @ A.T + n_x * np.eye(n_x)
H = rng.standard_normal((n_z, n_x))
R = np.diag([0.5, 2.0])
S = H @ P @ H.T + R
K = np.linalg.solve(S, H @ P).T
I = np.eye(n_x)

def joseph(K_):
    return (I - K_ @ H) @ P @ (I - K_ @ H).T + K_ @ R @ K_.T

P_opt = joseph(K)
D = np.random.default_rng(2).standard_normal(K.shape)
D /= np.linalg.norm(D)

for label, perturb in (("general perturbation", lambda e: K + e * np.linalg.norm(K) * D),
                       ("pure scaling", lambda e: (1 + e) * K)):
    print(label)
    print(f"{'eps':>8} {'short-form error':>18} {'short-form asym':>16} {'Joseph - P_opt':>15}")
    for eps in (1e-1, 1e-2, 1e-3, 1e-4):
        Kp = perturb(eps)
        P_true = joseph(Kp)
        P_short = (I - Kp @ H) @ P
        print(f"{eps:>8.0e} {np.linalg.norm(P_short - P_true):>18.3e} "
              f"{np.linalg.norm(P_short - P_short.T):>16.3e} {np.linalg.norm(P_true - P_opt):>15.3e}")
```
