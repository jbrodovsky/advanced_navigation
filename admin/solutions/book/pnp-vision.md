# Solutions: Perspective-n-Point Vision (`book/chapters/02-sensing/pnp-vision.qmd`)

Instructor only. Numerical results below were produced with the chapter's own
functions (all cells up to `pnp-covariance-monte-carlo`, run in order) on
2026-10-03; the code is included so the numbers can be regenerated.

## Exercise 1: Euler-rate kinematics

**(a)** Write $\mathbf{C} = \mathbf{A}\mathbf{B}\mathbf{D}$ with
$\mathbf{A} = \mathbf{R}_z(\psi)$, $\mathbf{B} = \mathbf{R}_x(\phi)$,
$\mathbf{D} = \mathbf{R}_y(\theta)$. For an elementary rotation about unit axis
$\mathbf{e}$, $\dot{\mathbf{R}} = \mathbf{R}[\mathbf{e}]_\times\dot\alpha$.
Then

$$
\mathbf{C}^T\dot{\mathbf{C}}
= \mathbf{D}^T\mathbf{B}^T[\mathbf{e}_z]_\times\mathbf{B}\mathbf{D}\,\dot\psi
+ \mathbf{D}^T[\mathbf{e}_x]_\times\mathbf{D}\,\dot\phi
+ [\mathbf{e}_y]_\times\dot\theta ,
$$

and with $\mathbf{M}^T[\mathbf{a}]_\times\mathbf{M} = [\mathbf{M}^T\mathbf{a}]_\times$
for a rotation $\mathbf{M}$,

$$
\boldsymbol{\omega}_{wb}^b
= \mathbf{R}_y^T\mathbf{R}_x^T\mathbf{e}_z\dot\psi + \mathbf{R}_y^T\mathbf{e}_x\dot\phi + \mathbf{e}_y\dot\theta .
$$

$\mathbf{R}_x^T(\phi)\mathbf{e}_z = [0, s_\phi, c_\phi]^T$, and applying
$\mathbf{R}_y^T(\theta)$ gives $[-c_\phi s_\theta,\ s_\phi,\ c_\phi c_\theta]^T$;
$\mathbf{R}_y^T\mathbf{e}_x = [c_\theta, 0, s_\theta]^T$. Collecting columns in
the order $(\dot\phi, \dot\theta, \dot\psi)$:

$$
\mathbf{W} = \begin{bmatrix} c_\theta & 0 & -c_\phi s_\theta \\ 0 & 1 & s_\phi \\ s_\theta & 0 & c_\phi c_\theta \end{bmatrix},
\qquad
\mathbf{W}^{-1} = \begin{bmatrix} c_\theta & 0 & s_\theta \\ s_\theta t_\phi & 1 & -c_\theta t_\phi \\ -s_\theta/c_\phi & 0 & c_\theta/c_\phi \end{bmatrix}.
$$

(Check: $\mathbf{W}^{-1}\mathbf{W} = \mathbf{I}$; verified numerically to
$10^{-16}$, and $\mathbf{W}$ against finite differences to $10^{-10}$.)

**(b)** Expanding along the first row,
$\det\mathbf{W} = c_\theta(c_\phi c_\theta) + (-c_\phi s_\theta)(-s_\theta) = c_\phi$.
Singular at $\phi = \pm 90°$ (roll gimbal lock).

**(c)** For Z-Y-X, $\mathbf{C} = \mathbf{R}_z(\psi)\mathbf{R}_y(\theta)\mathbf{R}_x(\phi)$,
$$
\mathbf{W}_{zyx} = \begin{bmatrix} 1 & 0 & -s_\theta \\ 0 & c_\phi & s_\phi c_\theta \\ 0 & -s_\phi & c_\phi c_\theta \end{bmatrix},
\qquad \det\mathbf{W}_{zyx} = c_\theta ,
$$
singular at pitch $\pm 90°$. A quadrotor over a tag mat stays within about
$25°$ of level in both roll and pitch (max 0.39 rad in the dataset), so neither
singularity is near; Z-X-Y is harmless. It would be a poor choice for a vehicle
that rolls to $90°$: a knife-edge pass or steep banked turn of an aerobatic
aircraft, or a fixed-wing drone in a $90°$ bank. (Z-Y-X fails instead for
vertical climbs, e.g. a tail-sitter VTOL in hover.) Quaternions avoid both.

## Exercise 2: Bias, covariance, and second moment

**(a)** With $\boldsymbol{\nu}_t = (\boldsymbol{\nu}_t - \hat{\mathbf{b}}) + \hat{\mathbf{b}}$,

$$
\frac{1}{n}\sum_t \boldsymbol{\nu}_t\boldsymbol{\nu}_t^T
= \frac{1}{n}\sum_t (\boldsymbol{\nu}_t - \hat{\mathbf{b}})(\boldsymbol{\nu}_t - \hat{\mathbf{b}})^T
+ \frac{1}{n}\sum_t \left[(\boldsymbol{\nu}_t - \hat{\mathbf{b}})\hat{\mathbf{b}}^T + \hat{\mathbf{b}}(\boldsymbol{\nu}_t - \hat{\mathbf{b}})^T\right]
+ \hat{\mathbf{b}}\hat{\mathbf{b}}^T .
$$

The cross terms vanish because $\sum_t(\boldsymbol{\nu}_t - \hat{\mathbf{b}}) = \mathbf{0}$,
and the first term is $\frac{n-1}{n}\hat{\mathbf{R}}$.

**(b)** $b^2/(b^2 + \sigma^2) = 0.46^2/(0.46^2 + 0.11^2) = 0.2116/0.2237 \approx 0.95$:
about 95% of the yaw mean-square error is bias.

**(c)** Averaging $N$ independent measurements with variance
$\sigma^2 = (0.11°)^2$, the filter claims $\sigma_{\hat\psi} = 0.11°/\sqrt{N}$
(ignoring process noise). The actual error is the bias plus the averaged noise,
so $E[e^2] = b^2 + \sigma^2/N$ and

$$
E[\text{NEES}] = \frac{b^2 + \sigma^2/N}{\sigma^2/N} = 1 + N\frac{b^2}{\sigma^2}
= 1 + 100 \times 17.5 \approx 1750
$$

for $N = 100$ (expected value 1 for this one-dimensional state). The NIS
cannot see this: the innovation compares the new measurement with an estimate
built from earlier measurements that carry the same bias, so the bias cancels
and NIS stays near its nominal value. NEES compares with truth and exposes any
common-mode error. This is exactly the NIS-below-6 / ANEES-far-above-9 pattern
the Project 1 reference run shows (`admin/solutions/drone_pipeline/README.md`).

## Exercise 3: Undistorting twice

```python
for h in (0.6, 1.0, 1.4, 2.0):
    T = make_transform(rpy_to_rotation_zxy([0.03, -0.02, 0.3]), [1.8, 1.4, h])
    pk = make_synthetic_packet(T, tag_corners, T_camera_body, K, pixel_sigma=0.0)
    a = estimate_pose(pk, K, np.zeros(5), tag_corners, T_camera_body)
    b = estimate_pose(pk, K, DIST_COEFFS, tag_corners, T_camera_body)
    print(h, pk.id.size, a.reprojection_error, a.position[2] - h, b.reprojection_error, b.position[2] - h)
```

| Height | Tags | Zero distortion: reproj., height error | File coefficients: reproj., height error |
|---|---|---|---|
| 0.6 m | 1 | 0.00 px, 0.0 cm | 0.53 px, -1.1 cm (and 4-8 cm horizontal) |
| 1.0 m | 5 | 0.00 px, 0.0 cm | 2.5 px, -5.6 cm |
| 1.4 m | 13 | 0.00 px, 0.0 cm | 4.3 px, -11.3 cm |
| 2.0 m | 32 | 0.00 px, 0.0 cm | 5.1 px, -18.6 cm |

(A sweep of 10-20 heights plotted as two curves is the expected deliverable.)
Sign: passing the coefficients tells the solver the pixels are distorted, so it
undistorts them. The lens has barrel distortion ($k_1 < 0$, @fig-pnp-distortion
shows points pulled inward), so undistortion pushes points outward, away from
the principal point. The tag pattern then looks larger than it is, and a larger
image means a closer camera: the height is underestimated. The error grows with
height because more tags sit near the image edge, where the distortion model is
strongest. This matches the real data (8-12 cm too low at 1-1.4 m).

Tests on a new vehicle: (i) look for detections outside the image bounds
(only undistorted coordinates can be there); (ii) image a target with long
straight edges and check whether they are straight in the images the detector
sees; (iii) compute reprojection error on a known target both ways, as here;
(iv) read the driver configuration (in ROS, whether the detector subscribes to
`image_raw` or `image_rect`, and what `camera_info` says).

## Exercise 4: When does a single marker flip?

Camera at distance $d$ from a tag at the origin, displaced by angle $\alpha$
from the tag normal and looking at the tag centre; 200 trials of 0.5 px corner
noise each; a "flip" is when the lower-reprojection IPPE candidate is the one
with the larger rotation error.

```python
SQUARE = 0.5 * 0.152 * np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], float)
def camera_pose(d, alpha_deg):
    a = np.radians(alpha_deg)
    cam = np.array([d * np.sin(a), 0.0, d * np.cos(a)])
    z_c = -cam / np.linalg.norm(cam)
    x_c = np.cross([0.0, 1.0, 0.0], z_c); x_c /= np.linalg.norm(x_c)
    return make_transform(np.column_stack([x_c, np.cross(z_c, x_c), z_c]), cam)  # T_world_camera
# for each (d, alpha): project SQUARE, add noise, cv2.solvePnPGeneric(..., SOLVEPNP_IPPE_SQUARE),
# compare argmin(reprojection error) with argmin(rotation error).
```

| Distance (tag size in image) | 0 deg | 10 deg | 20 deg | 40 deg |
|---|---|---|---|---|
| 0.5 m (74 px) | 24% | 0% | 0% | 0% |
| 1.0 m (37 px) | 25% | 12% | 1.5% | 0% |
| 2.0 m (18 px) | 31% | 39% | 22% | 13% |
| 4.0 m (9 px) | 27% | 43% | 50% | 46% |

At exactly 0 deg the two candidates are mirror tilts on either side of the true
(zero) tilt, so "wrong" is ill-defined and both are close; ignore that column or
say so. Elsewhere the flip rate rises sharply as the tag shrinks below ~20 px
and approaches a coin toss (50%) at 9 px. For a landing, the marker is smallest
(most ambiguous) at the start of the approach and largest at the end, so the
single-marker pose is least trustworthy exactly when the vehicle is farthest
away and most dependent on it; markers with internal structure, multiple
markers, or a larger marker help. An IMU-propagated attitude (good to a degree
or so over short intervals) separates candidates that differ by 5-10 deg in
tilt: pick the candidate closer to the prediction, or gate both through the
filter's innovation test and keep the one with the smaller NIS.

## Exercise 5: Lever arms rotate

**(a)** $\delta\mathbf{p}^w = \mathbf{C}_b^w\mathbf{C}_c^b\,\delta\mathbf{r}$.
For a level vehicle $\mathbf{C}_b^w = \mathbf{R}_z(\psi)$, and
$\mathbf{u} = \mathbf{C}_c^b\delta\mathbf{r}$ is a fixed body-frame vector, so
$\delta\mathbf{p}^w = \mathbf{R}_z(\psi)\mathbf{u}$: the vertical component is
constant, and the horizontal component has constant magnitude
$\|\mathbf{u}_h\|$ and rotates with yaw.

**(b)** Same Monte Carlo, but yaw sweeps $-\pi \to \pi$ over the 300 frames
(position held at (1.8, 1.3, 1.1) m, 1 cm lever-arm error along camera $x$, 0.5 px noise):

```python
rpy_k = np.array([0.04 * np.sin(6 * s), 0.04 * np.cos(5 * s), -np.pi + 2 * np.pi * s])
```

Result: bias (-0.04, -0.03, -0.01) cm, $\sqrt{\operatorname{diag}\hat{\mathbf{R}}}$ =
(0.96, 0.89, 0.12) cm, against a bias of about (-0.9, 0.4, 0) cm and
scatter (0.5, 0.6, 0.1) cm when yaw is nearly constant. A rotating 1 cm vector
has zero mean and per-axis variance $(1\ \text{cm})^2/2$; adding $(0.71\ \text{cm})^2$
to the ~0.55 cm pixel-noise scatter gives ~0.9 cm, as observed. The calibration
error moved from $\hat{\mathbf{b}}$ into $\hat{\mathbf{R}}$.

**(c)** A lever-arm error is not white noise: it is a deterministic,
attitude-dependent error, strongly correlated in time. Modeling it as noise
violates the filter's whiteness assumption (innovations become correlated with
heading) and forces an inflated $\mathbf{R}$ that discards information in
straight-line legs. It is a constant, so it can be added to the state with no
process noise, and it is observable whenever the vehicle turns (the error
changes direction while the true trajectory does not), so the filter can
estimate it to sub-centimeter level after a few maneuvers. On a ship the
antenna-to-IMU lever arm can be tens of meters and roll/pitch make it sweep,
which is why surveyed lever arms are entered in the INS and residual errors are
often estimated online.
