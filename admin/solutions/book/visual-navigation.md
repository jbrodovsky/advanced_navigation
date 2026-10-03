# Solutions: Visual Navigation and Visual SLAM (`visual-navigation.qmd`)

Instructor-only. Notation follows the chapter: camera 1 is the reference,
$\mathbf{p}^{c_2} = \mathbf{C}\mathbf{p}^{c_1} + \mathbf{r}$ with
$\mathbf{C} = \mathbf{C}_{c_1}^{c_2}$ and $\mathbf{r} = \mathbf{r}_{c_2c_1}^{c_2}$.
Numerical answers come from the chapter's two-view setup (seed 11, VGA camera
with $f = 500$ px, points 4 to 12 m away, 0.5 px noise, 20% bad matches,
`cv2.USAC_ACCURATE`, threshold 1.5 px). The coding snippets assume the
chapter's cells have been run.

A helper used below regenerates the two views for any relative pose:

```python
def make_views(r_c1c2_c1, C_c2_c1, seed=11, n_points=200, outlier_rate=0.2):
    rng = np.random.default_rng(seed)
    p1 = np.column_stack([rng.uniform(-5, 5, n_points), rng.uniform(-3, 3, n_points),
                          rng.uniform(4, 12, n_points)])
    C12 = C_c2_c1.T
    r21 = -C12 @ r_c1c2_c1
    u1, u2 = project(K, p1), project(K, (C12 @ p1.T).T + r21)
    vis = in_image(u1) & in_image(u2)
    p1, u1, u2 = p1[vis], u1[vis], u2[vis]
    m = len(p1)
    u1n, u2n = u1 + rng.normal(0, 0.5, (m, 2)), u2 + rng.normal(0, 0.5, (m, 2))
    bad = rng.random(m) < outlier_rate
    u2n[bad] = rng.uniform([0, 0], [WIDTH, HEIGHT], (bad.sum(), 2))
    return p1, u1n, u2n, bad, C12, r21
```

## Exercise 1: The epipolar constraint

**(a)** From $\mathbf{p}^{c_2} = \mathbf{C}\mathbf{p}^{c_1} + \mathbf{r}$, take
the cross product with $\mathbf{r}$:
$\mathbf{r} \times \mathbf{p}^{c_2} = \mathbf{r} \times \mathbf{C}\mathbf{p}^{c_1}$.
Dot with $\mathbf{p}^{c_2}$; the left side vanishes because
$\mathbf{r} \times \mathbf{p}^{c_2}$ is perpendicular to $\mathbf{p}^{c_2}$:
$\mathbf{p}^{c_2\,T}[\mathbf{r}]_\times\mathbf{C}\,\mathbf{p}^{c_1} = 0$.
Dividing by $Z_1 Z_2 > 0$ gives
$\bar{\mathbf{x}}_2^T\mathbf{E}\bar{\mathbf{x}}_1 = 0$ with
$\mathbf{E} = [\mathbf{r}]_\times\mathbf{C}$. Geometrically this is the
statement that the two viewing rays and the baseline are coplanar.

**(b)** $[\mathbf{r}]_\times[\mathbf{r}]_\times^T = \|\mathbf{r}\|^2\mathbf{I} - \mathbf{r}\mathbf{r}^T$,
whose eigenvalues are $\|\mathbf{r}\|^2$ (twice, on the plane perpendicular to
$\mathbf{r}$) and 0 (along $\mathbf{r}$). Since $\mathbf{C}$ is orthogonal,
$\mathbf{E}\mathbf{E}^T = [\mathbf{r}]_\times\mathbf{C}\mathbf{C}^T[\mathbf{r}]_\times^T = [\mathbf{r}]_\times[\mathbf{r}]_\times^T$,
so the singular values of $\mathbf{E}$ are $\|\mathbf{r}\|, \|\mathbf{r}\|, 0$:
rank two with two equal nonzero singular values.

**(c)** $\mathbf{C}$ has 3 degrees of freedom and $\mathbf{r}$ has 3, but the
constraint is homogeneous in $\mathbf{E}$ (equivalently, in $\mathbf{r}$), so
the overall scale is unobservable: $3 + 3 - 1 = 5$. Each correspondence gives
one scalar equation, so five correspondences give finitely many solutions (up
to ten); fewer leave a continuous family.

**(d)** $\bar{\mathbf{x}}_k \propto \mathbf{K}^{-1}\tilde{\mathbf{u}}_k$, so
$\tilde{\mathbf{u}}_2^T\mathbf{K}^{-T}\mathbf{E}\mathbf{K}^{-1}\tilde{\mathbf{u}}_1 = 0$,
i.e. $\mathbf{F} = \mathbf{K}^{-T}\mathbf{E}\mathbf{K}^{-1}$ (up to scale).
$\mathbf{F}$ has rank two but its two nonzero singular values are no longer
equal, and it has 7 degrees of freedom.

## Exercise 2: A hovering drone that only yaws

```python
p1, u1n, u2n, bad, C12, r21 = make_views(np.zeros(3), C_c2_c1)
E0, m0 = cv2.findEssentialMat(u1n, u2n, K, method=cv2.USAC_ACCURATE, prob=0.999, threshold=1.5)
n_front, C_r, t_r, _ = cv2.recoverPose(E0[:3], u1n, u2n, K, mask=m0.copy())

H_img, h_mask = cv2.findHomography(u1n, u2n, cv2.RANSAC, 1.5)
C_h = np.linalg.inv(K) @ H_img @ K
C_h /= np.cbrt(np.linalg.det(C_h))           # remove the arbitrary scale of H
U, _, Vt = np.linalg.svd(C_h)
C_h = U @ Vt                                  # nearest rotation matrix
```

**(a)** OpenCV does not complain. `findEssentialMat` returns a valid-looking
$\mathbf{E}$ (singular values $(1, 1, 0)$) with 132 "inliers", and
`recoverPose` returns a rotation within $0.02^\circ$ of the truth, a
translation direction of about $[-0.21, 0.46, 0.86]$ that means nothing (the
true translation is zero), and only 2 points in front of both cameras. With
$\mathbf{r} = \mathbf{0}$, $\bar{\mathbf{x}}_2 \propto \mathbf{C}\bar{\mathbf{x}}_1$,
so $\bar{\mathbf{x}}_2^T[\mathbf{t}]_\times\mathbf{C}\bar{\mathbf{x}}_1 \propto \bar{\mathbf{x}}_2^T[\mathbf{t}]_\times\bar{\mathbf{x}}_2 = 0$
for *every* $\mathbf{t}$: the constraint cannot determine a translation
direction, and the true $\mathbf{E} = [\mathbf{0}]_\times\mathbf{C}$ is the zero
matrix. Triangulation is impossible because both rays start from the same
center. The lesson: a solver returning an answer is not evidence of
observability; the cheirality count (2 of 132) is the warning light.

**(b)** $\lambda\bar{\mathbf{x}}_2 = \mathbf{C}\bar{\mathbf{x}}_1$ gives
$\tilde{\mathbf{u}}_2 \propto \mathbf{K}\mathbf{C}\mathbf{K}^{-1}\tilde{\mathbf{u}}_1$,
a homography. `findHomography` keeps 102 of 167 matches (38 were bad), and the
rotation recovered from $\mathbf{K}^{-1}\mathbf{H}\mathbf{K}$, normalized and
projected onto $SO(3)$, is within about $0.3^\circ$ of the truth.

**(c)** Detect low parallax: after compensating rotation (from the gyro, or
from a homography), the residual feature motion is a pixel or two; or compare
how well a homography and an essential (or fundamental) matrix explain the
matches, as ORB-SLAM's initializer does with its model-selection score. While
it lasts: do not insert keyframes, do not triangulate new points, keep tracking
orientation with the homography or gyro and position against existing map
points with PnP, and wait for translation. A VIO system simply keeps
propagating with the IMU.

## Exercise 3: Parallax buys depth

```python
direction = np.array([1.5, -0.1, 0.3]) / np.linalg.norm([1.5, -0.1, 0.3])
for b in [0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0]:
    p1, u1n, u2n, bad, C12, r21 = make_views(b * direction, C_c2_c1)
    E1, m1 = cv2.findEssentialMat(u1n, u2n, K, method=cv2.USAC_ACCURATE, prob=0.999, threshold=1.5)
    _, C_e, t_e, _ = cv2.recoverPose(E1[:3], u1n, u2n, K, mask=m1.copy())
    inl = m1.ravel().astype(bool)
    p_e = triangulate(C_e, t_e.ravel(), u1n[inl], u2n[inl]) * np.linalg.norm(r21)
    # record rotation error, direction error, and np.median(np.linalg.norm(p_e - p1[inl], axis=1))
```

| baseline [m] | rotation err [deg] | direction err [deg] | median 3D err [m] | predicted $\sigma_Z$ [m] |
|---|---|---|---|---|
| 0.05 | 0.37 | 73.4 | 21.4 | 2.22 |
| 0.10 | 0.52 | 32.3 | 38.6 | 1.14 |
| 0.20 | 0.14 | 5.9 | 0.41 | 0.57 |
| 0.50 | 0.02 | 0.55 | 0.15 | 0.23 |
| 1.00 | 0.03 | 0.42 | 0.070 | 0.11 |
| 1.50 | 0.13 | 0.20 | 0.12 | 0.076 |
| 2.00 | 0.06 | 0.06 | 0.042 | 0.059 |
| 3.00 | 0.08 | 0.18 | 0.029 | 0.042 |

The prediction uses $Z = $ median depth (about 8 m), $f = 500$ px, and
$\sigma_d = \sqrt{2} \times 0.5$ px. For a Gaussian the median absolute error is
$0.67\sigma$, so from 0.5 m upward the measured medians agree with the
prediction to within the scatter of a single run (the 1.5 m case is inflated by
its pose error). Below about 0.2 m the translation *direction* collapses (tens
of degrees of error), and then triangulation is meaningless no matter what the
depth formula says: with $b/Z$ around 1%, noise dominates the epipolar
geometry. A defensible keyframe rule here is $b \gtrsim 0.5$ m, i.e.
$b/Z \gtrsim 1/16$, or equivalently a median parallax of a few degrees; real
systems use thresholds of that order (often expressed as a median parallax in
pixels after rotation compensation).

## Exercise 4: Scale drift on a pipeline survey

**(a)** 2 km at 5 m per keyframe is $N = 400$ handoffs.

```python
rng = np.random.default_rng(0)
s = np.cumprod(1.0 + rng.normal(0.0, 0.01, (1000, 400)), axis=1)
along_track_error = 5.0 * (s - 1.0).sum(axis=1)
```

The spread of $s_k$ is 0.102 at $k = 100$ and 0.201 at $k = 400$, matching
$0.01\sqrt{k}$ = 0.10 and 0.20: the log-scale is a random walk.

**(b)** To first order $s_k - 1 \approx \sum_{m \le k}\varepsilon_m$. The
along-track error is $e = \Delta\sum_{k=1}^{N}(s_k - 1) = \Delta\sum_{m=1}^{N}(N - m + 1)\varepsilon_m$
with $\Delta = 5$ m, so

$$
\sigma_e = \Delta\,\sigma_\varepsilon\sqrt{\sum_{j=1}^{N} j^2}
= \Delta\,\sigma_\varepsilon\sqrt{\frac{N(N+1)(2N+1)}{6}}
= 5 \times 0.01 \times 4627 \approx 231\ \text{m},
$$

about 12% of the distance flown. The simulation gives 235 m. The error grows
like $N^{3/2}$, the same doubly integrated random walk as a heading error in
@sec-loop-closure-why-drift-accumulates.

**(c)** On the way back the scale keeps drifting, so when the drone recognizes
its launch point the current keyframe and the first keyframe disagree in
position *and* in scale (the current scale may be 20% off). The trajectory is
the right shape at the wrong size, piece by piece. A rigid $SE(3)$ pose graph
can only rotate and translate keyframes; its relative-pose edges carry the
wrongly scaled translations as if they were metric, so the scale inconsistency
becomes distortion. A $\mathrm{Sim}(3)$ pose graph gives each keyframe a scale
and each edge a relative scale, so the correction can distribute a smooth scale
change along the trajectory [@strasdat2010scale]. (An IMU, wheel odometry, or a
known pipe diameter would prevent the problem in the first place.)

## Exercise 5: What VIO cannot see

Neglect Earth rotation. The IMU measures
$\mathbf{f}_{ib}^b = \mathbf{C}_n^b(\mathbf{a}^n - \mathbf{g}^n)$ and
$\boldsymbol{\omega}_{ib}^b$; the camera measures
$\pi\left(\mathbf{C}_n^c(\mathbf{p}_\ell - \mathbf{p}_c)\right)$ for each
landmark $\ell$.

**(a)** *Translation* $\mathbf{p} \to \mathbf{p} + \boldsymbol{\delta}$ for every
position (trajectory and landmarks): accelerations, attitudes, and the
differences $\mathbf{p}_\ell - \mathbf{p}_c$ are unchanged, so nothing changes.
*Rotation about gravity* by $\mathbf{R} = \mathbf{R}_z(\delta\psi)$ applied to
everything: $\mathbf{p} \to \mathbf{R}\mathbf{p}$, $\mathbf{C}_b^n \to \mathbf{R}\mathbf{C}_b^n$,
$\mathbf{a}^n \to \mathbf{R}\mathbf{a}^n$. Then
$\mathbf{f}_{ib}^b \to \mathbf{C}_n^b\mathbf{R}^T(\mathbf{R}\mathbf{a}^n - \mathbf{g}^n) = \mathbf{C}_n^b(\mathbf{a}^n - \mathbf{R}^T\mathbf{g}^n) = \mathbf{C}_n^b(\mathbf{a}^n - \mathbf{g}^n)$
because a rotation about the vertical leaves $\mathbf{g}^n = [0, 0, g]^T$
unchanged. Body rates are unchanged by a constant rotation of the navigation
frame, and the camera sees
$\mathbf{C}_n^c\mathbf{R}^T\mathbf{R}(\mathbf{p}_\ell - \mathbf{p}_c)$,
unchanged. That is $3 + 1 = 4$ unobservable directions.

**(b)** For a rotation $\mathbf{R}$ about a horizontal axis,
$\mathbf{R}^T\mathbf{g}^n \ne \mathbf{g}^n$, and the accelerometer output
changes by $\mathbf{C}_n^b(\mathbf{R}^T\mathbf{g}^n - \mathbf{g}^n)$, which is
of order $g\,\delta\theta$: about 0.17 m/s² per degree, easily measured. Roll
and pitch are observable. (A constant accelerometer bias can mimic a small tilt
over short windows, which is why attitude and bias need motion to separate.)

**(c)** Trust: roll and pitch, metric scale and velocity over short intervals,
and the local shape of the trajectory. Expect drift: horizontal and vertical
position (typically a fraction of a percent to a few percent of the distance
flown, depending on the system and the scene) and yaw, which drifts slowly with
gyro bias and also rotates the direction of the position error. Altitude
usually drifts less if a barometer is fused; heading needs a magnetometer,
GNSS-derived course, or a map.

**(d)** Loop-closure drift lives only in the unobservable directions: position
and yaw. Roll and pitch are already referenced to gravity at every keyframe, so
a 6-DOF correction could only make them worse (or let one bad loop closure tilt
the map). VINS-Mono therefore optimizes $(x, y, z, \psi)$ per keyframe.

## Exercise 6: Schur complement bookkeeping

**(a)** The normal equations are
$\mathbf{H}_{cc}\Delta\mathbf{c} + \mathbf{H}_{cp}\Delta\mathbf{p} = -\mathbf{g}_c$ and
$\mathbf{H}_{pc}\Delta\mathbf{c} + \mathbf{H}_{pp}\Delta\mathbf{p} = -\mathbf{g}_p$.
From the second, $\Delta\mathbf{p} = -\mathbf{H}_{pp}^{-1}(\mathbf{g}_p + \mathbf{H}_{pc}\Delta\mathbf{c})$.
Substituting into the first:
$(\mathbf{H}_{cc} - \mathbf{H}_{cp}\mathbf{H}_{pp}^{-1}\mathbf{H}_{pc})\Delta\mathbf{c} = -\mathbf{g}_c + \mathbf{H}_{cp}\mathbf{H}_{pp}^{-1}\mathbf{g}_p$.
Because $\mathbf{H}_{pp}$ is block diagonal, the back-substitution splits per
point:
$\Delta\mathbf{p}_i = -\mathbf{H}_{p_ip_i}^{-1}(\mathbf{g}_{p_i} + \mathbf{H}_{p_ic}\Delta\mathbf{c})$.

**(b)** $\mathbf{H}$ is $(6m + 3n)^2 = 153{,}000 \times 153{,}000$.
$\mathbf{H}_{pp}$ is $150{,}000 \times 150{,}000$ with 50,000 diagonal
$3 \times 3$ blocks (450,000 nonzeros). The reduced camera system is
$3000 \times 3000$. There are $50{,}000 \times 8 = 400{,}000$ observations, each
contributing one $6 \times 3$ block to $\mathbf{H}_{cp}$: about $7.2$ million
nonzeros (and the same again in $\mathbf{H}_{pc}$). Forming the Schur
complement costs, per point, $8^2 = 64$ products of $6 \times 3$ by
$3 \times 3$ by $3 \times 6$ blocks, about $3.2$ million block products in all:
linear in $n$.

**(c)** There are far more points than camera parameters ($150{,}000$ versus
$3000$ unknowns), and each point's block is independent, so eliminating points
costs one $3 \times 3$ inverse each and leaves a small system. Eliminating the
cameras instead would couple every pair of points seen by a common camera and
leave a $150{,}000$-dimensional system that is far denser.

**(d)** Block $(j, k)$ of the reduced camera system is nonzero whenever
keyframes $j$ and $k$ observe a common point. For a video sequence explored
once, points are seen only by nearby keyframes, so the system is banded and
sparse Cholesky is cheap. Loop closures, revisits, and unordered photo
collections in which many cameras see the same points make it dense; then a
dense Cholesky on the (still small) reduced system, or a preconditioned
conjugate-gradient solver, is used.
