# Solutions: Linear Algebra Refresher (`sec-linalg-exercises`)

Instructor-only. Numbers below were produced by running the code shown with the
course environment (NumPy 2.4, SciPy 1.17); timings will vary by machine.

## 1. Skew-matrix identities

**(a)** By inspection of $[\mathbf{a}]_\times$: the diagonal is zero and entry
$(i,j)$ is the negative of entry $(j,i)$, so $[\mathbf{a}]_\times^T = -[\mathbf{a}]_\times$.

**(b)** Anti-commutativity gives $[\mathbf{a}]_\times\mathbf{b} = \mathbf{a}\times\mathbf{b} = -\mathbf{b}\times\mathbf{a} = -[\mathbf{b}]_\times\mathbf{a}$.
The expression is linear in $\mathbf{a}$ with matrix $-[\mathbf{b}]_\times$, so
$\partial(\mathbf{a}\times\mathbf{b})/\partial\mathbf{a} = -[\mathbf{b}]_\times$ (and
$\partial(\mathbf{a}\times\mathbf{b})/\partial\mathbf{b} = [\mathbf{a}]_\times$).

**(c)** For any $\mathbf{b}$, the vector triple product ("BAC minus CAB") gives
$[\mathbf{a}]_\times^2\mathbf{b} = \mathbf{a}\times(\mathbf{a}\times\mathbf{b}) = \mathbf{a}(\mathbf{a}\cdot\mathbf{b}) - \mathbf{b}(\mathbf{a}\cdot\mathbf{a}) = (\mathbf{a}\mathbf{a}^T - \|\mathbf{a}\|^2\mathbf{I})\mathbf{b}$.

**(d)** $[\mathbf{a}]_\times^3 = [\mathbf{a}]_\times(\mathbf{a}\mathbf{a}^T - \|\mathbf{a}\|^2\mathbf{I}) = ([\mathbf{a}]_\times\mathbf{a})\mathbf{a}^T - \|\mathbf{a}\|^2[\mathbf{a}]_\times = -\|\mathbf{a}\|^2[\mathbf{a}]_\times$,
since $\mathbf{a}\times\mathbf{a} = \mathbf{0}$.

With $\boldsymbol{\theta} = \theta\mathbf{u}$, $\|\mathbf{u}\| = 1$, (d) gives
$[\boldsymbol{\theta}]_\times^{2k+1} = (-1)^k\theta^{2k+1}[\mathbf{u}]_\times$ and
$[\boldsymbol{\theta}]_\times^{2k} = (-1)^{k+1}\theta^{2k}[\mathbf{u}]_\times^2$ for
$k \ge 1$. Every power reduces to $[\mathbf{u}]_\times$ or $[\mathbf{u}]_\times^2$, so
the exponential series collapses to
$\mathbf{I} + (\theta - \theta^3/3! + \cdots)[\mathbf{u}]_\times + (\theta^2/2! - \theta^4/4! + \cdots)[\mathbf{u}]_\times^2 = \mathbf{I} + \sin\theta[\mathbf{u}]_\times + (1-\cos\theta)[\mathbf{u}]_\times^2$.

## 2. Why dead reckoning drifts

$\mathbf{F}^2 = \begin{bmatrix} 1 & 2\Delta t & 2\Delta t^2 \\ 0 & 1 & 2\Delta t \\ 0 & 0 & 1 \end{bmatrix}$.

**(a)** $\mathbf{H} = [1\ 0\ 0]$:
$\mathcal{O} = \begin{bmatrix} 1 & 0 & 0 \\ 1 & \Delta t & \Delta t^2/2 \\ 1 & 2\Delta t & 2\Delta t^2 \end{bmatrix}$,
$\det\mathcal{O} = 2\Delta t^3 - \Delta t^3 = \Delta t^3 \neq 0$. Rank 3, null space
$\{\mathbf{0}\}$: positions alone determine velocity and acceleration (two and three
samples, respectively, by finite differences).

**(b)** $\mathbf{H} = [0\ 1\ 0]$:
$\mathcal{O} = \begin{bmatrix} 0 & 1 & 0 \\ 0 & 1 & \Delta t \\ 0 & 1 & 2\Delta t \end{bmatrix}$.
The first column is zero and the other two are independent: rank 2, null space
spanned by $[1, 0, 0]^T$. A constant position offset never affects a velocity
measurement, so odometry or Doppler alone cannot fix absolute position; position
error from integrating velocity noise accumulates without bound.

**(c)** $\mathbf{H} = [0\ 0\ 1]$: every row is $[0, 0, 1]$, rank 1, null space spanned
by $[1, 0, 0]^T$ and $[0, 1, 0]^T$. Neither initial position nor initial velocity
can be recovered from an accelerometer; they must be supplied as initial
conditions, and any error in them (plus integrated accelerometer noise and bias)
persists and grows. That is dead reckoning with an IMU in one line of linear
algebra.

Numerical confirmation with $\Delta t = 0.1$:

```python
F = np.array([[1, dt, dt**2 / 2], [0, 1, dt], [0, 0, 1.0]])
for H in ([[1, 0, 0]], [[0, 1, 0]], [[0, 0, 1]]):
    O = observability_matrix(F, np.array(H, float))
    print(np.linalg.matrix_rank(O))      # 3, 2, 1
```

The SVD null-space basis returned is empty, $[1,0,0]$, and $\{[0,1,0],[1,0,0]\}$
(up to sign) respectively.

## 3. Which update form is cheaper?

```python
import timeit

def update_covariance_form(P, H, R):
    S = H @ P @ H.T + R
    K = np.linalg.solve(S, H @ P).T          # K = P H^T S^-1 (S, P symmetric)
    return P - K @ H @ P

def update_information_form(P, H, R):
    R_inv = np.diag(1.0 / np.diag(R))        # R is diagonal
    return np.linalg.inv(np.linalg.inv(P) + H.T @ R_inv @ H)

rng = np.random.default_rng(3)
for n, m in [(15, 3), (15, 400)]:
    G = rng.standard_normal((n, n)); P = G @ G.T + n * np.eye(n)
    H = rng.standard_normal((m, n)); R = np.diag(rng.uniform(0.5, 2.0, m))
    assert np.allclose(update_covariance_form(P, H, R), update_information_form(P, H, R))
    for f in (update_covariance_form, update_information_form):
        print(n, m, f.__name__, min(timeit.repeat(lambda: f(P, H, R), number=200, repeat=5)) / 200)
```

Representative results: for $(15, 3)$ both take about 13–16 µs (Python call
overhead dominates at these sizes, and the two are essentially tied). For
$(15, 400)$ the covariance form takes about 5.6 ms and the information form about
0.14 ms, roughly 40 times faster.

Explanation: the covariance form solves with the $m \times m$ innovation
covariance $\mathbf{S}$, costing $O(m^3)$ plus $O(m^2 n)$ to form it; the
information form inverts $n \times n$ matrices, costing $O(n^3)$, plus
$O(n^2 m)$ to accumulate $\mathbf{H}^T\mathbf{R}^{-1}\mathbf{H}$ (cheap because
$\mathbf{R}$ is diagonal). With $m \ll n$ (a GNSS fix on an INS) the covariance
form wins asymptotically; with $m \gg n$ (hundreds of image features on a small
pose state) the information form wins. This is the Woodbury identity used as an
engineering decision, and one reason information-form filters are attractive for
vision and multi-sensor problems.

## 4. Repairing a covariance

**(a)** For $\begin{bmatrix} 1 & \rho & 0 \\ \rho & 1 & \rho \\ 0 & \rho & 1 \end{bmatrix}$,
expanding along the first row,
$\det(\mathbf{P} - \lambda\mathbf{I}) = (1-\lambda)[(1-\lambda)^2 - \rho^2] - \rho\cdot\rho(1-\lambda) = (1-\lambda)[(1-\lambda)^2 - 2\rho^2]$.
The roots are $\lambda = 1$ and $\lambda = 1 \pm \rho\sqrt2$. With $\rho = 0.9$:
$1$, $2.2728$, and $-0.2728$. The negative eigenvalue means some linear
combination of the states would have negative variance (the direction is
$[1, -\sqrt2, 1]/2$).

**(b)**

```python
def nearest_psd(P: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    """Symmetrize, clip eigenvalues at floor, and reassemble."""
    P_sym = 0.5 * (P + P.T)
    lam, V = np.linalg.eigh(P_sym)
    return (V * np.maximum(lam, floor)) @ V.T

P_fix = nearest_psd(P_bad, floor=1e-6)
np.linalg.cholesky(P_fix)                    # succeeds
```

Result (rounded): $\begin{bmatrix} 1.068 & 0.804 & 0.068 \\ 0.804 & 1.136 & 0.804 \\ 0.068 & 0.804 & 1.068 \end{bmatrix}$,
eigenvalues $[10^{-6}, 1, 2.273]$. This is the nearest matrix in the Frobenius
norm with eigenvalues $\ge$ `floor`. It is a repair for a log or a diagnostic,
not a substitute for fixing the filter: it changes the variances, so the result
is no longer what the filter "believed."

**(c)** Acceptable answers include: (1) round-off in the short-form update
$\mathbf{P} \leftarrow (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}$, which loses
symmetry and definiteness, especially with a suboptimal $\mathbf{K}$, large
information gain, or float32 (prevent with the Joseph form, symmetrizing after
each update, or a square-root/UD filter [@bierman1977]); (2) a hand-built
$\mathbf{Q}$ or $\mathbf{R}$ with individually plausible correlations that are
jointly impossible, as here (prevent by building noise from a factor, e.g.
$\mathbf{L}\mathbf{Q}\mathbf{L}^T$, and checking with Cholesky); (3) manual
covariance edits such as inflating or zeroing individual off-diagonal terms
after a reset; (4) mixing covariance entries from different time steps when
logging asynchronously.

## 5. Exact versus first-order discretization

**(a)** $\mathbf{A}^2 = \begin{bmatrix} 0 & 1 \\ -\omega_0^2 & 0 \end{bmatrix}^2 = \begin{bmatrix} -\omega_0^2 & 0 \\ 0 & -\omega_0^2 \end{bmatrix} = -\omega_0^2\mathbf{I}$.
So $\mathbf{A}^{2k} = (-1)^k\omega_0^{2k}\mathbf{I}$ and
$\mathbf{A}^{2k+1} = (-1)^k\omega_0^{2k}\mathbf{A}$, and the series splits:
$e^{\mathbf{A}\Delta t} = \left(\sum_k \frac{(-1)^k(\omega_0\Delta t)^{2k}}{(2k)!}\right)\mathbf{I} + \frac{1}{\omega_0}\left(\sum_k \frac{(-1)^k(\omega_0\Delta t)^{2k+1}}{(2k+1)!}\right)\mathbf{A} = \cos(\omega_0\Delta t)\,\mathbf{I} + \frac{\sin(\omega_0\Delta t)}{\omega_0}\mathbf{A}$,
which is the stated matrix.

**(b)**

```python
w0, dt = 2 * np.pi, 0.01
A = np.array([[0.0, 1.0], [-w0**2, 0.0]])
closed = np.array([[np.cos(w0 * dt), np.sin(w0 * dt) / w0],
                   [-w0 * np.sin(w0 * dt), np.cos(w0 * dt)]])
assert np.allclose(closed, expm(A * dt))
```

**(c)**

```python
F_exact, F_euler = expm(A * dt), np.eye(2) + A * dt
x_exact = x_euler = np.array([1.0, 0.0])
for _ in range(1000):
    x_exact, x_euler = F_exact @ x_exact, F_euler @ x_euler
```

After 10 s (exactly 10 periods) the exact propagation returns to $[1, 0]$ to
rounding. The first-order propagation ends at about $[7.15, 3.71]$, an amplitude
$\sqrt{x^2 + (\dot x/\omega_0)^2} \approx 7.17$ times the initial one. The
eigenvalues of $\mathbf{I} + \mathbf{A}\Delta t$ are $1 \pm i\omega_0\Delta t$ with
modulus $\sqrt{1 + \omega_0^2\Delta t^2} = 1.00197$, so each step inflates the
amplitude by that factor: $1.00197^{1000} = (1 + \omega_0^2\Delta t^2)^{500} = 7.17$,
matching the simulation. The exact transition matrix has eigenvalues
$e^{\pm i\omega_0\Delta t}$ on the unit circle and conserves energy. The lesson
for filters is that $\mathbf{F} \approx \mathbf{I} + \mathbf{A}\Delta t$ is only safe
when $\|\mathbf{A}\|\Delta t \ll 1$ over the horizon that matters, and that
`expm` is cheap insurance.
