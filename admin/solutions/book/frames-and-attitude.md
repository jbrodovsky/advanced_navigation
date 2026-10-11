# Solutions: Coordinate Frames and Attitude (`sec-frames-exercises`)

Instructor-only. Code assumes the functions defined in the chapter's cells
(`skew`, `radii`, `geodetic_to_ecef`, `ecef_to_geodetic`, `C_e_to_n`,
`make_transform`, `invert_transform`, and the variables of the SE(3) cell). All
numbers below were produced by running that code.

## 1. Rodrigues backward

**(a)** Take the trace of the Rodrigues formula term by term. $\operatorname{tr}\mathbf{I} = 3$;
$[\mathbf{u}]_\times$ has a zero diagonal, so its trace is 0; and
$\operatorname{tr}[\mathbf{u}]_\times^2 = \operatorname{tr}(\mathbf{u}\mathbf{u}^T - \mathbf{I}) = \|\mathbf{u}\|^2 - 3 = -2$.
Hence $\operatorname{tr}\mathbf{C} = 3 + 0 - 2(1-\cos\theta) = 1 + 2\cos\theta$.

For the antisymmetric part, $\mathbf{I}$ and $[\mathbf{u}]_\times^2 = \mathbf{u}\mathbf{u}^T - \mathbf{I}$
are symmetric and cancel in $\mathbf{C} - \mathbf{C}^T$, while
$[\mathbf{u}]_\times^T = -[\mathbf{u}]_\times$, so
$\mathbf{C} - \mathbf{C}^T = \sin\theta\,([\mathbf{u}]_\times - [\mathbf{u}]_\times^T) = 2\sin\theta\,[\mathbf{u}]_\times$.

**(b)**

```python
import matplotlib.pyplot as plt

def vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])

def log_so3(C: np.ndarray) -> np.ndarray:
    """Rotation vector of C for 0 <= theta < pi."""
    cos_t = np.clip((np.trace(C) - 1.0) / 2.0, -1.0, 1.0)  # clip: rounding can exceed 1
    theta = np.arccos(cos_t)
    if theta < 1e-8:
        return 0.5 * vee(C - C.T)                           # first order
    return theta / (2.0 * np.sin(theta)) * vee(C - C.T)

rng = np.random.default_rng(1)
rotations = Rotation.random(1000, random_state=rng)
theta = rotations.magnitude()
err = np.array([np.linalg.norm(log_so3(R.as_matrix()) - R.as_rotvec()) for R in rotations])
plt.semilogy(np.degrees(theta), err + 1e-18, ".")
plt.xlabel("rotation angle [deg]"); plt.ylabel("|log_so3 - as_rotvec| [rad]")
```

With seed 1 the maximum error is about $5\times10^{-16}$ rad below $90^\circ$,
$2\times10^{-14}$ rad up to $170^\circ$, $1\times10^{-12}$ rad up to $179^\circ$, and
$3\times10^{-9}$ rad for the dozen samples above $179^\circ$. Targeted tests show
the blow-up: $6\times10^{-12}$ rad at $179.9^\circ$, $5\times10^{-9}$ at
$179.99^\circ$, $1\times10^{-6}$ at $179.999^\circ$.

Why: as $\theta \to \pi$, $\sin\theta \to 0$ and $\mathbf{C} - \mathbf{C}^T \to \mathbf{0}$,
so the axis is computed as a ratio of two vanishing quantities. The antisymmetric
part is known only to absolute precision $\sim\epsilon$, so the relative axis
error grows like $\epsilon/(\pi - \theta)$, and $\arccos$ is itself ill-conditioned
near $-1$. At exactly $\pi$ the formula is $0/0$ and the sign of $\mathbf{u}$ is
genuinely ambiguous ($\pm\mathbf{u}$ give the same rotation). Robust
implementations switch near $\pi$ to the symmetric part,
$\mathbf{C} + \mathbf{I} \approx 2\mathbf{u}\mathbf{u}^T$, taking $\mathbf{u}$ from its
largest column. Students may also note that the small-angle end is fine: although
$\arccos$ loses precision near 1, the factor $\theta/\sin\theta \approx 1$ is
insensitive to that error, and the first-order branch handles $\theta \approx 0$.

## 2. Earth-referenced attitude rate

**(a)** A proper rotation preserves cross products:
$(\mathbf{C}\mathbf{a}) \times (\mathbf{C}\mathbf{b}) = \mathbf{C}(\mathbf{a} \times \mathbf{b})$
(lengths, angles, and handedness are preserved because $\det\mathbf{C} = +1$).
For any $\mathbf{w}$, write $\mathbf{w} = \mathbf{C}\mathbf{C}^T\mathbf{w}$:
$[\mathbf{C}\mathbf{a}]_\times\mathbf{w} = (\mathbf{C}\mathbf{a}) \times (\mathbf{C}\mathbf{C}^T\mathbf{w}) = \mathbf{C}(\mathbf{a} \times \mathbf{C}^T\mathbf{w}) = \mathbf{C}[\mathbf{a}]_\times\mathbf{C}^T\mathbf{w}$.
Since this holds for every $\mathbf{w}$, the matrices are equal.

**(b)** From the DCM kinematics equation and linearity of $[\cdot]_\times$,
$\dot{\mathbf{C}}_b^n = \mathbf{C}_b^n[\boldsymbol{\omega}_{ib}^b]_\times - \mathbf{C}_b^n[\mathbf{C}_n^b\boldsymbol{\omega}_{in}^n]_\times$.
Apply (a) with $\mathbf{C} = \mathbf{C}_n^b$:
$[\mathbf{C}_n^b\boldsymbol{\omega}_{in}^n]_\times = \mathbf{C}_n^b[\boldsymbol{\omega}_{in}^n]_\times\mathbf{C}_b^n$.
Then $\mathbf{C}_b^n\mathbf{C}_n^b = \mathbf{I}$ leaves
$\dot{\mathbf{C}}_b^n = \mathbf{C}_b^n[\boldsymbol{\omega}_{ib}^b]_\times - [\boldsymbol{\omega}_{in}^n]_\times\mathbf{C}_b^n$.
The body-rate term multiplies on the right (body axes) and the frame-rate term on
the left (navigation axes), matching the form used in the INS/GNSS chapter.

**(c)** $\|\boldsymbol{\omega}_{ie}^n\| = \omega_{ie} = 7.292115\times10^{-5}$ rad/s
$= 15.04$ °/h at any latitude; at $L = 42.3^\circ$ the components are
$11.12$ °/h north and $-10.12$ °/h down (that is, 10.12 °/h up). For the ship,
$v_N = 20 \times 1852/3600 = 10.29$ m/s and $v_E = 0$, so only the east component
of the transport rate survives: $\|\boldsymbol{\omega}_{en}^n\| = v_N/(R_M + h)$
with $R_M = 6.364\times10^6$ m, giving $1.62\times10^{-6}$ rad/s $= 0.33$ °/h.

Against a 10 °/h bias: Earth rate is only 1.5 times the bias, so it is visible in
principle but cannot be separated from an uncalibrated bias; the transport rate
is 30 times smaller than the bias and completely buried. This is why
$\boldsymbol{\omega}_{in}^n$ matters for navigation- and tactical-grade INS but is
often dropped for consumer MEMS over short intervals.

## 3. The MSL bug

**(a)** $h = H + N_g = 160.0 + (-28.5) = 131.5$ m.

**(b)**

```python
lat, lon = np.radians(42.2742), np.radians(-71.8080)
r_ok = geodetic_to_ecef(lat, lon, 131.5)
r_bad = geodetic_to_ecef(lat, lon, 160.0)
d_ned = C_e_to_n(lat, lon) @ (r_bad - r_ok)
```

The ECEF difference is $[6.58, -20.03, 19.17]$ m, magnitude 28.50 m. In NED it is
$[0, 0, -28.5]$ m to rounding ($<10^{-9}$ m horizontally): the wrong position is
28.5 m *up*. Both points share latitude and longitude, and
the geodetic-to-ECEF formula moves a point along the ellipsoid normal when only $h$
changes, which is exactly the local down axis.

**(c)** Converting the wrong ECEF position back gives the same latitude and
longitude, so any horizontal check (map overlays, north/east errors against a
survey point, distances between fixes) is unaffected, and relative positions
between nearby fixes cancel the common bias as well. Only absolute vertical
comparisons (a barometer, a terrain or chart datum, a surveyed height, or the
INS vertical channel in a fused solution) expose it. Students should conclude
that unit tests need at least one absolute-height case with a known geoid
separation.

## 4. Compass, current, and GNSS

**(a)** With east positive: magnetic = compass + deviation $= 045° - 3° = 042°$ M;
true = magnetic + variation $= 042° - 14° = 028°$ T. ("Can Dead Men Vote Twice At
Elections": westerly corrections are subtracted going from compass to true.)

**(b)** Course minus heading $= 025° - 028° = -3°$: the ship moves 3° to port of
where its bow points. With speed over ground 12.0 kn, the velocity component
perpendicular to the keel is $12.0\sin(-3°) = -0.63$ kn (0.63 kn toward port)
and the along-keel component is $12.0\cos 3° = 11.98$ kn.

**(c)** The ship is being set to port. What this data determines is only the
cross-keel component: 0.63 kn of current and/or leeway toward $298°$ T (heading
minus $90°$). The along-keel component of the set is unknown without the speed
through the water from a log, and it can change both the total set's direction
and its magnitude, so no overall set bearing or drift can be stated from COG,
SOG, and heading alone.
Full credit requires noting that GNSS course is not heading, so initializing an
INS yaw from this COG would put a 3° error into the attitude.

## 5. Composition order

**(a)** `T_c_b` is $\mathbf{T}_c^b$ (camera coordinates into body) and `T_b_w` is
$\mathbf{T}_b^w$ (body into world). The product $\mathbf{T}_c^b\mathbf{T}_b^w$ feeds
world coordinates (the output of $\mathbf{T}_b^w$) into a transform that expects
camera coordinates: the inner indices ($c$ and $w$) do not match. The correct
chain is $\mathbf{T}_c^w = \mathbf{T}_b^w\mathbf{T}_c^b$, with the shared index $b$ in
the middle.

**(b)**

```python
T_c_w_wrong = T_c_b @ T_b_w
p_c_wrong = (invert_transform(T_c_w_wrong) @ np.append(landmark_w, 1.0))[:3]
```

The correct camera coordinates are $[1, 0.5, 10]$ m; the wrong product gives
$[-11.03, -1.14, 30.12]$ m, a difference of 23.5 m. The landmark would project
far outside the image.

**(c)** The wrong order gives the right answer whenever the two transforms
commute. The most common case is the vehicle at the world origin with identity
attitude ($\mathbf{T}_b^w = \mathbf{I}$), which is exactly the pose a quick unit test
tends to use. Less obviously, $\mathbf{C}_c^b$ for the forward camera is a
$120^\circ$ rotation about $(1,1,1)/\sqrt3$, so a vehicle that is only translated
along $(1,1,1)$ (north, east, and down equally) with identity attitude also
commutes with it. Tests built from identity or "nice" poses therefore pass, and
the bug appears only with a general pose. The lesson is to test transforms with
a random or deliberately asymmetric pose.
