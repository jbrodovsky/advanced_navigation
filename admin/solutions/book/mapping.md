# Solutions: Mapping for SLAM (`mapping.qmd`)

Instructor-only. Vehicle numbers: $L = 2.83$ m, $H = 0.76$ m, laser at
$(a, b) = (3.78, +0.50)$ m in the vehicle frame (x forward, y left, origin at
the rear-axle center). Inverse sensor model of the demo:
$l_\text{occ} = \log(0.7/0.3) = 0.8473$, $l_\text{free} = -0.8473$,
clamp $[-2, 3.5]$. Numbers for the coding exercises were produced with the
chapter's own functions on the Victoria Park data (2026-10-03).

## Exercise 1: The encoder correction

**(a)** With steering $\delta$ the front wheel rolls perpendicular to the line
from the instantaneous center of rotation (ICR); the rear wheels have no
steering, so the ICR lies on the rear-axle line. The front wheel is $L$ ahead
of the axle and turned by $\delta$, so the ICR is at distance
$R = L/\tan\delta$ from the axle center, on the left for $\delta > 0$. Every
point of the rigid body moves at $\omega$ times its distance from the ICR, and
the axle center moves at $v_c$, so $\omega = v_c/R = (v_c/L)\tan\delta$. The
rear-left wheel is at lateral offset $+H$, i.e. at distance $R - H$ from the
ICR:

$$
v_e = \omega (R - H) = v_c\left(1 - \frac{H}{R}\right) = v_c\left(1 - \frac{H}{L}\tan\delta\right)
\;\Rightarrow\;
v_c = \frac{v_e}{1 - \frac{H}{L}\tan\delta}.
$$

The formula is signed: for $\delta < 0$ (ICR on the right) the left wheel is
on the outside, $R - H$ with $R < 0$ has magnitude $|R| + H$, and the encoder
reads fast. For an encoder on the rear-*right* wheel (offset $-H$):
$v_e = \omega(R + H)$, so $v_c = v_e / (1 + \frac{H}{L}\tan\delta)$.

**(b)** The fractional error of the uncorrected encoder is
$k = (H/L)\tan\delta = 0.26855\tan\delta$ (the encoder reads $v_c(1-k)$):

| $\delta$ | $\pm 10°$ | $\pm 20°$ | $\pm 30°$ |
|---|---|---|---|
| $k$ | $\pm 4.7\%$ | $\pm 9.8\%$ | $\pm 15.5\%$ |

Left turns ($+$) read slow, right turns read fast.

**(c)** At $\delta = 15°$, $R = 2.83/\tan 15° = 10.56$ m and $k = 0.0720$.
The dead reckoning uses $v_e = v_c(1-k)$ both for distance and for the yaw
rate $\omega' = v_e\tan\delta/L = \omega(1 - k)$, so its radius
$v_e/\omega' = L/\tan\delta$ is **exactly right**. The error is along the
track: when the real vehicle has completed the circle, the dead-reckoned one
has turned only $2\pi(1-k)$ and is short of closing by the chord
$2R\sin(\pi k) = 4.7$ m. The circle "closes" only after an extra 7.8% of
time. This is why the uncorrected error in the chapter grows with every
left-hand loop rather than appearing as a wrong curvature: on this dataset it
roughly doubles the aligned RMSE (about 28 m vs 13 m over the first 200 s).

## Exercise 2: Lever-arm kinematics

**(a)** $\mathbf{p}_L = \mathbf{p}_r + \mathbf{C}(\theta)\mathbf{d}$ with
$\mathbf{d} = [a, b]^T$ and $\mathbf{C}(\theta) = \begin{bmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{bmatrix}$.
Since $\frac{d}{dt}\mathbf{C}(\theta) = \dot\theta\,\mathbf{C}(\theta)\mathbf{J}$
with $\mathbf{J} = \begin{bmatrix}0 & -1\\ 1 & 0\end{bmatrix}$,

$$
\dot{\mathbf{p}}_L = v_c\begin{bmatrix}\cos\theta\\ \sin\theta\end{bmatrix} + \omega\,\mathbf{C}(\theta)\begin{bmatrix}-b\\ a\end{bmatrix}
= \begin{bmatrix} v_c\cos\theta - \omega(a\sin\theta + b\cos\theta)\\ v_c\sin\theta + \omega(a\cos\theta - b\sin\theta)\end{bmatrix},
$$

and $\dot\theta = \omega$ is unchanged because heading is a property of the
whole body.

**(b)** In the vehicle frame the laser velocity is
$\mathbf{C}(\theta)^T\dot{\mathbf{p}}_L = [v_c - \omega b,\; \omega a]^T$, so
$|\dot{\mathbf{p}}_L| = \sqrt{(v_c - \omega b)^2 + (\omega a)^2}$. At
$v_c = 5$ m/s and $\delta = 25°$: $\omega = 5\tan 25°/2.83 = 0.824$ rad/s,
$[v_c - \omega b, \omega a] = [4.59, 3.11]$ m/s, speed 5.55 m/s, direction
$\operatorname{atan2}(3.11, 4.59) = 34°$ to the **left** of the heading. (The
chapter's numerical check uses an encoder reading of 5 m/s, i.e.
$v_c = 5.72$ m/s, and prints a laser speed of 6.34 m/s.)

**(c)** $30°/\text{min} = 0.00873$ rad/s; a point 20 m ahead of the pivot moves
sideways at $0.00873 \times 20 = 0.175$ m/s (0.34 kn) toward the inside of the
turn. A dead-reckoning or set-and-drift computation that treats the antenna
velocity as the ship's velocity through the water attributes that sideways
motion to current or leeway: the plot shows an apparent 0.17 m/s cross-current
that appears and vanishes with every turn, and a dynamic-positioning controller
fed that velocity would "correct" a drift that does not exist. The fix is the
same as here: transform the antenna measurement to a common reference point
with the lever arm and the measured rate of turn (or carry the lever arm in the
measurement model).

## Exercise 3: Log-odds bookkeeping

**(a)** For a static binary cell with measurements conditionally independent
given the cell, Bayes' rule in odds form is
$o_t = o_{t-1}\cdot\frac{p(m \mid z_t)}{1 - p(m \mid z_t)}\cdot\frac{1 - p_0}{p_0}$
(the inverse-sensor form follows from $p(z_t \mid m) = p(m \mid z_t)p(z_t)/p(m)$;
the $p(z_t)$ terms cancel in the ratio of occupied to free). Taking logs gives
@eq-mapping-log-odds-update. The final log-odds are
$l_0 + \sum_t [\operatorname{logit} p(m \mid z_t) - l_0]$, a sum, which does not
depend on the order of the terms. (Clamping breaks this: a clamped sequence is
order dependent. That is the price of a map that can forget.)

**(b)** After $n$ "occupied" readings the odds are $(7/3)^n$ and
$1 - p = 1/(1 + (7/3)^n)$. Double precision rounds $p$ to exactly 1.0 once
$1 - p$ falls below half an ulp of 1, about $2^{-53}$:
$(7/3)^n > 2^{53}$, i.e. $n > 53\ln 2/\ln(7/3) = 43.4$. So **44** readings,
which is what the chapter's cell prints (log-odds 37.28 at that point). After
that $p/(1-p) = \infty$ and the next update is `nan`.

**(c)** With clamping the cell sits at 3.5; each free reading subtracts 0.847,
so $\lceil 3.5/0.847\rceil = 5$ readings bring it below 0 ($p < 0.5$).
Without clamping, 1000 occupied readings give $l = 847.3$ and it takes 1000
free readings to get back to $p = 0.5$ (1001 to go below it): a parked car seen
for three minutes at 5 Hz would haunt the map for three more minutes after it
left.

## Exercise 4: Complete the extractor

Reference implementation (adds the two tests on the segment list, then reuses
the chapter's per-segment logic by masking the scan to one segment at a time):

```python
def extract_full(scan_m, isolation=True, occlusion=True):
    scan_m = np.asarray(scan_m, float)
    beams = np.arange(scan_m.size)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(scan_m) & (scan_m < 75.0)
    index, ranges = beams[valid], scan_m[valid]
    if index.size < 2:
        return []
    bearings = index * BEAM_STEP - math.pi / 2
    breaks = np.flatnonzero((np.abs(np.diff(ranges)) > 1.5) | (np.diff(index) > 10)) + 1
    segs = np.split(np.arange(index.size), breaks)
    first = np.array([s[0] for s in segs]); last = np.array([s[-1] for s in segs])
    xy = lambda i: np.column_stack([ranges[i] * np.cos(bearings[i]), ranges[i] * np.sin(bearings[i])])
    p_first, p_last = xy(first), xy(last)
    drop = np.zeros(len(segs), bool)
    if isolation:                       # ends within 3 m of the next 1..3 segments: drop both
        for k in (1, 2, 3):
            if len(segs) > k:
                close = np.flatnonzero(np.linalg.norm(p_first[k:] - p_last[:-k], axis=1) < 3.0)
                drop[close] = True
                drop[close + k] = True
    if occlusion and len(segs) > 1:     # less than 1 degree (2 beams) apart: drop the far one
        j = np.flatnonzero(index[first[1:]] - index[last[:-1]] < 2)
        drop[j + (ranges[first[j + 1]] > ranges[last[j]])] = True
    out = []
    for keep, seg in zip(~drop, segs):
        if keep:
            out += extract_tree_landmarks(np.where(np.isin(beams, index[seg]), scan_m, np.nan))
    return out
```

Results on scans 1000-1999 and the `map_sharpness` check (12 sharpest turns):

| Extractor | Detections per scan | Sharpness (m) |
|---|---|---|
| chapter | 5.42 | 0.400 |
| + isolation | 5.11 | 0.399 |
| + occlusion | 5.14 | 0.399 |
| + both | 4.94 | 0.406 |

Isolation removes slightly more candidates than occlusion (about 6% vs 5%);
together they remove about 9%. Neither changes the sharpness metric
meaningfully, and a good answer explains why: the metric measures whether
re-observed *detections* agree, and hedge fragments or partly occluded trunks
that are seen consistently from nearby poses agree with themselves too. Their
value shows up in the SLAM back end (fewer spurious landmarks, fewer ambiguous
associations, centers that are not biased by occlusion), which is the
experiment Project 2 asks for. Full-credit answers also notice that the
occlusion test as written in MATLAB fires for *every* break between adjacent
beams, because consecutive valid beams are 0.5 degrees apart.

## Exercise 5: Thresholds on a grid

With `A = np.arange(361) * np.pi / 360` and the threshold `10 * np.pi / 360`,
the 351 differences `A[k+10] - A[k]` compare as: 170 greater, 6 equal, 175
less. For 2-beam gaps against `2 * np.pi / 360`: 198 greater, 3 equal, 158
less. Each `A[k]` is the correctly rounded value of $k\pi/360$, and the
difference of two rounded numbers is not the rounded difference, so an
"exact" 5-degree gap lands a few ulps either side of the threshold depending
on $k$. The MATLAB detector therefore splits (or does not) at 10-beam gaps
essentially at random, and the 1-degree occlusion test behaves the same way
for a single missing beam. (The reference port in
`admin/solutions/victoria_park_slam/landmarks_solution.py` reproduces the
MATLAB arithmetic so it matches bit for bit.)

The robust fixes: compare integer beam indices (`np.diff(index) > 10`), or
place the threshold between multiples of the quantum (4.75 degrees). Other
places it bites: timestamps. The odometry is logged every 25 ms in integer
milliseconds, but in float seconds `np.diff(t_odo) > 0.025` is true for
27357 of the 61944 intervals and `== 0.025` for none, so any "gap" detector
with a 25 ms threshold fires on half the log. The same happens with encoder
tick counts, ADC codes, GNSS week/time-of-week rollovers, and range bins.

## Exercise 6: Calibrating a time offset

`map_sharpness` (12 sharpest turns, laser at $(3.78, +0.50)$) against a shift
added to the laser timestamps:

| shift (ms) | -150 | -125 | -100 | -75 | -50 | -25 | 0 | +25 | +50 | +75 | +100 | +125 | +150 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sharpness (m) | 0.469 | 0.438 | 0.414 | 0.386 | 0.380 | 0.384 | 0.400 | 0.426 | 0.456 | 0.484 | 0.494 | 0.512 | 0.531 |

The minimum is **not** at zero but near $-50$ ms, with a flat bottom from
about $-75$ to $-25$ ms: the scans are best explained as having been taken
about 50 ms *before* their timestamps, i.e. the laser stamp is late relative to
the odometry by roughly $50 \pm 25$ ms. Plausible causes: the stamp is applied
when the scan arrives over the serial link rather than at the middle of the
sweep, or the steering and speed stream has its own (negative) latency. Only
the *relative* offset is observable from this test. Turns carry the
information because a timing error $\Delta t$ displaces a tree at range $r$ by
roughly $v\,\Delta t$ along track (absorbed by the same error on every
re-observation, so nearly invisible on a straight) plus $r\,\omega\,\Delta t$
across track, which changes direction as the vehicle turns and therefore
smears the cluster. A good answer quantifies the uncertainty (the 25 ms grid
and the flatness of the curve), checks robustness to the window selection,
and notes that a 50 ms error at 0.3 rad/s moves a tree 40 m away by 0.6 m,
comparable to the measurement noise assumed in Project 2.
