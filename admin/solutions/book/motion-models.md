# Solutions: Motion Models (`sec-motion-models-exercises`)

Instructor-only. Code assumes the functions defined in the chapter's cells
(`unicycle_step`, `euler_step`, `kinematic_F`, `wrap_angle`,
`sample_motion_model_odometry`). All numbers below were produced by running that
code.

## 1. Exact discretization

For the constant-acceleration model with state $[p, \dot p, \ddot p]^T$ and white
jerk,

$$
\mathbf{A} = \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{bmatrix},
\qquad
\mathbf{A}^2 = \begin{bmatrix} 0 & 0 & 1 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix},
\qquad
\mathbf{A}^3 = \mathbf{0}.
$$

$\mathbf{A}$ is strictly upper triangular, so each multiplication pushes the
nonzero band one place further from the diagonal, and after three it is gone. The
exponential series therefore terminates:

$$
e^{\mathbf{A}\Delta t} = \mathbf{I} + \mathbf{A}\Delta t + \tfrac{1}{2}\mathbf{A}^2\Delta t^2
= \begin{bmatrix} 1 & \Delta t & \tfrac12\Delta t^2 \\ 0 & 1 & \Delta t \\ 0 & 0 & 1 \end{bmatrix}.
$$

**Semigroup property.** Because $\mathbf{A}t_1$ and $\mathbf{A}t_2$ commute,
$e^{\mathbf{A}t_1}e^{\mathbf{A}t_2} = e^{\mathbf{A}(t_1+t_2)}$. Directly: the
product of the two upper-triangular matrices has diagonal ones, superdiagonal
$t_1 + t_2$, and top-right entry
$\tfrac12 t_2^2 + t_1 t_2 + \tfrac12 t_1^2 = \tfrac12 (t_1 + t_2)^2$.

**Euler counterexample.** Take $\mathbf{x}_0 = [0, 0, 0]^T$ and
$\mathbf{u} = [1\text{ m/s}, 1\text{ rad/s}]^T$. One Euler step of 2 s gives
$[2, 0, 2]^T$. Two Euler steps of 1 s give $[1, 0, 1]^T$ and then
$[1 + \cos 1, \sin 1, 2]^T = [1.540, 0.841, 2]^T$. The results differ, so the
Euler map is not a one-parameter semigroup in $\Delta t$; the exact arc model is
(two arcs at the same $v$ and $\omega$ are one longer arc).

## 2. Reading a gyro data sheet

**(a)** ARW $= 0.004 \times 60 = 0.24\ \text{deg}/\sqrt{\text{h}}$. Per-sample
standard deviation at 400 Hz:
$\sigma_d = 0.004\sqrt{400} = 0.08$ deg/s $= 1.40 \times 10^{-3}$ rad/s.

**(b)** Heading error from ARW after 600 s:
$\sigma_\psi = 0.004\ (\text{deg/s})/\sqrt{\text{Hz}} \times \sqrt{600\ \text{s}} = 0.098^\circ$
(equivalently $0.24\ \text{deg}/\sqrt{\text{h}} \times \sqrt{1/6\ \text{h}}$).
Heading error from the 10 deg/h bias over 1/6 h: $1.67^\circ$, about 17 times
larger. The designer should worry about bias: estimate it (for example, at rest
before takeoff, or as a filter state while GNSS is available) and keep the
GNSS-denied interval short. The white noise is a rounding error by comparison.

## 3. Gravity in two frames

**(a)** NED: $\mathbf{g}^n = [0, 0, +g]^T$. Level FRD aligned with NED means
$\mathbf{C}_b^n = \mathbf{I}$, and with $\mathbf{a}^n = \mathbf{0}$,
$\mathbf{f}^b = -\mathbf{g}^n = [0, 0, -g]^T$: the down-pointing $z$ accelerometer
reads $-9.81\ \text{m/s}^2$, i.e. specific force points up.

**(b)** ENU: $\mathbf{g}^n = [0, 0, -g]^T$. A level forward-left-up body frame
facing east is aligned with ENU, so again $\mathbf{C}_b^n = \mathbf{I}$ and
$\mathbf{f}^b = [0, 0, +g]^T$: the up-pointing $z$ axis reads $+9.81$. Then
$\mathbf{a}^n = \mathbf{C}_b^n\mathbf{f}^b + \mathbf{g}^n = [0, 0, g]^T + [0, 0, -g]^T = \mathbf{0}$.
The equation is the same; only the vertical sign of $\mathbf{g}^n$ (and of what the
accelerometer reports along "up" versus "down") changes.

**(c)** $\delta\theta = 0.5^\circ = 8.73 \times 10^{-3}$ rad, so the fictitious
horizontal acceleration is $g\,\delta\theta = 0.0856\ \text{m/s}^2$ (about 8.7
milli-g). Position error $\tfrac12 (0.0856) t^2$: 38.5 m after 30 s and 616 m
after 2 minutes. A half-degree tilt error is a disaster for an unaided INS, which
is why leveling (initial attitude from the accelerometers) and attitude aiding
matter so much.

## 4. The banana versus the ellipse

```python
rng = np.random.default_rng(42)
odom_prev = np.array([0.0, 0.0, 0.0])
odom_curr = np.array([0.95, 0.25, np.deg2rad(30.0)])
alpha_rot = np.array([0.1, 0.1, 0.001, 0.001])
samples = sample_motion_model_odometry(np.zeros((5000, 3)), odom_prev, odom_curr,
                                       alpha_rot, rng)
xy = samples[:, :2]
mean, cov = xy.mean(axis=0), np.cov(xy.T)
cov_inv = np.linalg.inv(cov)

d2 = np.einsum("ij,jk,ik->i", xy - mean, cov_inv, xy - mean)
print(np.mean(d2 <= 4.0))                 # 0.891; a Gaussian gives 0.865

# Fraction of the 2-sigma ellipse with no samples nearby.
from scipy.ndimage import maximum_filter
ex = np.linspace(mean[0] - 1, mean[0] + 1, 201)
ey = np.linspace(mean[1] - 1, mean[1] + 1, 201)
counts, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[ex, ey])
cx, cy = 0.5 * (ex[:-1] + ex[1:]), 0.5 * (ey[:-1] + ey[1:])
GX, GY = np.meshgrid(cx, cy, indexing="ij")
cells = np.column_stack([GX.ravel(), GY.ravel()]) - mean
inside = np.einsum("ij,jk,ik->i", cells, cov_inv, cells) <= 4.0
occupied = maximum_filter(counts > 0, size=5).ravel()
print(np.mean(~occupied[inside]))         # about 0.18

# Sample density at the mean versus at the odometry endpoint (3 cm radius).
for pt in (mean, odom_curr[:2]):
    print(np.sum(np.linalg.norm(xy - pt, axis=1) < 0.03))   # 101 vs 193
```

About 89 percent of samples fall inside the $2\sigma$ ellipse, close to the 86.5
percent a Gaussian would have, so the ellipse is not wildly wrong in *coverage*.
It is wrong in *shape*: roughly a fifth of its area holds essentially no samples
(the region inside the curve and beyond the arc's radial spread), the sample mean
sits about 5 cm inside the arc where the density is half that along the arc, and
the tips of the banana poke outside. Larger rotational noise or longer motions make
all of these worse. For an EKF, which propagates only a mean and covariance, this
means the predicted mean is a pose the robot is relatively unlikely to occupy, the
linearized measurement model is evaluated at that pose, and subsequent updates can
be biased or overconfident. The UKF captures more of the curvature; the particle
filter represents the banana directly.

## 5. Steering by a biased gyro

**(a)** The autopilot holds the *estimated* heading, so the true heading error is
$\psi_e(t) = b_g t$. The cross-track rate is $\dot y = v \sin\psi_e \approx v b_g t$
for small angles; integrating, $y(t) \approx \tfrac12 v b_g t^2$. The exact result
(constant turn rate $b_g$, an arc of radius $v/b_g$) is
$y(t) = (v/b_g)(1 - \cos b_g t)$.

**(b)**

```python
v, b_g, dt, duration = 5.0, np.deg2rad(0.01), 0.1, 600.0
x = np.zeros(3)                     # planar pose; the true heading turns at b_g
history = [x]
for _ in range(int(duration / dt)):
    x = unicycle_step(x, np.array([v, b_g]), dt)
    history.append(x)
history = np.array(history)
print(history[-1, 1], 0.5 * v * b_g * duration**2)    # 156.9 m vs 157.1 m
```

After 10 minutes the simulated cross-track error is 156.9 m, against 157.1 m from
the formula. The heading error is then $6^\circ$.

**(c)** The relative error of the small-angle formula is
$1 - (1 - \cos x)/(x^2/2) \approx x^2/12$ with $x = b_g t$. Solving
$(1 - \cos x)/(x^2/2) = 0.9$ numerically gives $x = 1.118$ rad ($64^\circ$), i.e.
$t \approx 6400$ s, about 107 minutes. The formula fails when the heading error is
no longer small: the vessel is then flying a visible arc of radius
$v/b_g \approx 28.6$ km, and $\sin\psi_e \approx \psi_e$ no longer holds. In practice
the navigator would have noticed long before.
