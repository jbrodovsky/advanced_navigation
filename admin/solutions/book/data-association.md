# Solutions: Features, Maps, and Data Association (`sec-feature-association-exercises`)

Instructor-only. Code assumes the functions and variables defined in the
chapter's cells (`tag_corners`, `TAG_PITCH`, `K`, `dist_coeffs`, `signed_area`,
`validate_tag_detections`, plus `numpy`, `cv2`, and SciPy's `Rotation`). All
numbers below were produced by running that code.

## 1. Counting hypotheses

**(a)** Choose which $k$ of the $n$ measurements are real detections of mapped
landmarks ($\binom{n}{k}$ ways); the rest are clutter. Assign those $k$
measurements to distinct landmarks, an ordered selection of $k$ from $N$
($N!/(N-k)!$ ways). Summing over $k$:

$$
\sum_{k=0}^{5}\binom{5}{k}\frac{20!}{(20-k)!}
= 1 + 100 + 3{,}800 + 68{,}400 + 581{,}400 + 1{,}860{,}480 = 2{,}514{,}181.
$$

Two and a half million joint hypotheses for five measurements and a modest map.

**(b)** With unique IDs there is exactly one hypothesis: each measurement goes to
the landmark with its ID, or is discarded if the ID is not in the map.

**(c)** Detections: $10 \times 60 \times 30 \times 8 = 144{,}000$. Expected misreads:
$144{,}000 \times 10^{-4} = 14.4$. That is small but not zero, so the pipeline must
handle them: discard IDs not in the map; for misreads that land on a valid ID, rely
on geometry (the tag's corners will be inconsistent with the other tags in a
multi-tag PnP solve, so a per-tag reprojection gate rejects it) and on temporal
consistency (a tag that should not be visible from the predicted pose, or that
appears for a single frame, is suspect).

## 2. Why a cyclic shift is invisible

Place the tag's center at the origin with corners
$\mathbf{c}_k = \tfrac{L}{2}\mathbf{R}_z(k\pi/2)[1, 1, 0]^T$, $k = 0, \dots, 3$ (any
consistent labeling works). A cyclic relabeling pairs world corner $\mathbf{c}_k$
with the image of $\mathbf{c}_{k+1} = \mathbf{R}_z(\pi/2)\mathbf{c}_k$. If the true
pose is $(\mathbf{C}, \mathbf{t})$ (world to camera), the shifted image points are
$\pi(\mathbf{C}\mathbf{R}_z(\pi/2)\mathbf{c}_k + \mathbf{t})$, which are exactly the
projections of the *correctly labeled* corners under the pose
$(\mathbf{C}\mathbf{R}_z(\pi/2), \mathbf{t})$. That is a valid rigid pose, so it
reproduces the detections as well as the true pose does, and the minimum
reprojection error is unchanged. The recovered camera is rotated $90^\circ$ about
the tag normal (and, because the rotation is about the tag center rather than the
world origin, displaced).

Reversing the order swaps top and bottom corners, a reflection of the square across
a line through its center lying in the plane $z = 0$. For points with $z = 0$, that
reflection equals a $180^\circ$ rotation about the same in-plane line (the rotation
also negates $z$, which is zero). So again a rigid pose explains the data, but the
rotation flips the plane's normal: the camera now sees the tag's back, and its
position in the mat frame has the opposite sign of $z$, below the floor. The
chapter's experiment shows exactly this ($z = -1.20$ m).

With four tags, the shift rotates each tag by $90^\circ$ about *its own* center (or
flips each about its own axis). Rotations about different centers are different
rigid motions, and their combination is not a rigid motion of the whole layout, so
no single camera pose explains all the corners and the reprojection error becomes
large. (The argument would fail only for degenerate layouts, such as a single tag,
or tags sharing a center.)

## 3. Catch every fault

```python
rng = np.random.default_rng(5)
ids_grid = np.array([[0, 12, 24], [1, 13, 25], [2, 14, 26]])
patch = {int(ids_grid[r, c]): tag_corners(r * TAG_PITCH, c * TAG_PITCH)
         for r in range(3) for c in range(3)}


def random_camera():
    """A camera 1.0-1.6 m above the patch, looking down with small tilts."""
    base = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    C = Rotation.from_euler("xyz", rng.uniform(-8, 8, 3), degrees=True).as_matrix() @ base
    pos = np.array([0.38, 0.38, 0.0]) + np.array([*rng.uniform(-0.1, 0.1, 2), rng.uniform(1.0, 1.6)])
    rvec, _ = cv2.Rodrigues(C)
    return rvec, -C @ pos


def simulate_frame(rvec, tvec, pixel_sigma=0.3):
    frame = []
    for tid, P in patch.items():
        uv, _ = cv2.projectPoints(P, rvec, tvec, K, dist_coeffs)
        frame.append((tid, uv.reshape(-1, 2) + rng.normal(0.0, pixel_sigma, (4, 2))))
    return frame


def inject(frame, fault):
    frame = [(tid, c.copy()) for tid, c in frame]
    if fault == "shift all":
        frame = [(tid, np.roll(c, 1, axis=0)) for tid, c in frame]
    elif fault == "shift one":
        i = rng.integers(len(frame)); frame[i] = (frame[i][0], np.roll(frame[i][1], 1, axis=0))
    elif fault == "corner 15 px":
        i, j = rng.integers(len(frame)), rng.integers(4)
        d = rng.normal(size=2); frame[i][1][j] += 15.0 * d / np.linalg.norm(d)
    elif fault == "duplicate":
        i = rng.integers(len(frame)); frame.append((frame[i][0], frame[i][1] + rng.uniform(30, 80, 2)))
    return frame


def check_frame(frame, gate_px):
    """Stage 1: per-tag checks. Stage 2: multi-tag PnP and a per-tag reprojection gate."""
    accepted, rejected = validate_tag_detections(frame, patch)
    if rejected:
        return True
    obj = np.vstack([patch[t] for t, _ in accepted])
    img = np.vstack([c for _, c in accepted])
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return True
    reproj = cv2.projectPoints(obj, rvec, tvec, K, dist_coeffs)[0].reshape(-1, 2)
    per_tag = np.sqrt(np.mean(np.sum((reproj - img) ** 2, axis=1).reshape(-1, 4), axis=1))
    return bool(np.any(per_tag > gate_px))


faults = ["clean", "shift all", "shift one", "corner 15 px", "duplicate"]
for sigma in [0.3, 1.5]:
    frames = [simulate_frame(*random_camera(), pixel_sigma=sigma) for _ in range(200)]
    for gate_px in [2.0, 4.0, 6.0]:
        rates = {f: np.mean([check_frame(inject(fr, f), gate_px) for fr in frames]) for f in faults}
        print(sigma, gate_px, rates)
```

Results (flag rate over 200 frames):

| Corner noise | Gate | Clean (false alarm) | Shift all | Shift one | Corner +15 px | Duplicate |
|---:|---:|---:|---:|---:|---:|---:|
| 0.3 px | 2 px | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.3 px | 4 px | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.3 px | 6 px | 0.00 | 1.00 | 1.00 | 0.975 | 1.00 |
| 1.5 px | 2 px | 0.995 | 1.00 | 1.00 | 1.00 | 1.00 |
| 1.5 px | 4 px | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 1.5 px | 6 px | 0.00 | 1.00 | 1.00 | 0.94 | 1.00 |

Duplicates are caught by stage 1; all shifts and the displaced corner by the
reprojection gate. With clean 0.3 px corners, gates of 2 to 4 px are perfect, and even a 6 px gate misses only a few displaced corners.
With noisier 1.5 px corners, the per-tag RMS of a clean tag is about
$\sqrt{2}\times1.5 \approx 2$ px, so a 2 px gate flags almost every clean frame,
while a 6 px gate starts missing the 15 px corner outlier (its per-tag RMS is
about $15/2 = 7.5$ px before the solve absorbs part of it). A gate of 2 to 3 times
the expected clean per-tag RMS, here about 4 px, balances the two. The threshold
should therefore be set from the measured corner noise, not hard-coded.

## 4. Buoys at night

**(a)** The two buoys are 300 m apart and the along-track position uncertainty is
200 m (one sigma). Predicted range and bearing to each buoy differ mostly along
track by an amount comparable to the uncertainty, so a radar return near either
predicted location passes a reasonable gate for *both* buoys. Radar sees two point
targets with no identity; range and bearing alone cannot say which is which until
the position uncertainty shrinks.

**(b)** The light characteristic is a partial ID: both are green, so color does
not help, but the flash periods (4 s and 6 s) differ. The camera must time at least
two flashes of each light, so it needs at least 6 s, and about 12 s (the least
common multiple, three flashes of one and two of the other) to be confident.
What can go wrong: an extinguished or obscured light; shore lights, other
vessels, or reflections that flash irregularly; the camera's frame rate or
exposure missing short flashes; ship motion moving the light out of frame; and
associating the light seen by the (bearing-only) camera with the wrong radar
return if the two buoys appear at similar bearings.

**(c)** For each radar return $\mathbf{z} = [r, \beta]^T$ and buoy $j$, compute
$\hat{\mathbf{z}}_j = h(\hat{\mathbf{x}}, \mathbf{m}_j)$ and the innovation
$\boldsymbol{\nu}_j = \mathbf{z} - \hat{\mathbf{z}}_j$, with the bearing component
wrapped to $[-\pi, \pi)$. Form $\mathbf{S}_j = \mathbf{H}_j\mathbf{P}\mathbf{H}_j^T + \mathbf{R}$
and $d_j^2 = \boldsymbol{\nu}_j^T\mathbf{S}_j^{-1}\boldsymbol{\nu}_j$. Accept the pairing
only if $d_j^2$ is below the 99 percent chi-square gate for two degrees of freedom
(9.21) for exactly one buoy *and* the timed light characteristic matches that
buoy; if both buoys pass, or the characteristic disagrees, defer the update rather
than guess.

## 5. The wrapped bearing

Without wrapping, the residual is $-179^\circ - 179^\circ = -358^\circ$, and
$(358/2)^2 = 32{,}041 \gg 6.63$: the correct match is rejected. With wrapping, the
residual is $+2^\circ$, $(2/2)^2 = 1 < 6.63$: accepted.

```python
wrap = lambda a: (a + 180.0) % 360.0 - 180.0       # degrees; use np.pi for radians

for a in [181, -181, 359, -359, 721]:
    print(a, wrap(a))      # -179, 179, -1, 1, 1
```

Python's `%` returns a result with the sign of the divisor, so this works for
negative inputs too (unlike C's `fmod`). The result lies in $[-180, 180)$; an input
of exactly $180^\circ$ maps to $-180^\circ$, which is the same angle.
