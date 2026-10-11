# Solutions: Bayes Filtering (`bayes.qmd`)

Instructor-only. Exercise numbers match `## Exercises {#sec-bayes-exercises}`.

## Exercise 1: Where the assumptions enter

Write $\mathcal{D}_t = \{\mathbf{z}_{1:t}, \mathbf{u}_{1:t}\}$.

**Update.** Bayes' rule on the newest measurement:

$$
p(\mathbf{x}_t \mid \mathbf{z}_t, \mathbf{z}_{1:t-1}, \mathbf{u}_{1:t})
=
\frac{p(\mathbf{z}_t \mid \mathbf{x}_t, \mathbf{z}_{1:t-1}, \mathbf{u}_{1:t})\,
      p(\mathbf{x}_t \mid \mathbf{z}_{1:t-1}, \mathbf{u}_{1:t})}
     {p(\mathbf{z}_t \mid \mathbf{z}_{1:t-1}, \mathbf{u}_{1:t})}.
$$

The **measurement conditional-independence assumption** replaces
$p(\mathbf{z}_t \mid \mathbf{x}_t, \mathbf{z}_{1:t-1}, \mathbf{u}_{1:t})$ by
$p(\mathbf{z}_t \mid \mathbf{x}_t)$. (Strictly, the stated assumption conditions
on the whole state history $\mathbf{x}_{0:t}$; marginalizing the earlier states
out of a quantity that does not depend on them gives the form used here.) The
denominator does not depend on $\mathbf{x}_t$ and becomes $\eta^{-1}$. The
second factor in the numerator is $\overline{\operatorname{bel}}(\mathbf{x}_t)$.

**Prediction.** Total probability over the previous state:

$$
p(\mathbf{x}_t \mid \mathbf{z}_{1:t-1}, \mathbf{u}_{1:t})
=
\int p(\mathbf{x}_t \mid \mathbf{x}_{t-1}, \mathbf{z}_{1:t-1}, \mathbf{u}_{1:t})\,
p(\mathbf{x}_{t-1} \mid \mathbf{z}_{1:t-1}, \mathbf{u}_{1:t})\, d\mathbf{x}_{t-1}.
$$

- The **state Markov assumption** reduces the first factor to
  $p(\mathbf{x}_t \mid \mathbf{x}_{t-1}, \mathbf{u}_t)$.
- The second factor still conditions on $\mathbf{u}_t$. Assuming the input is
  chosen independently of the past state given past data (inputs are not
  "informative about where we were"), we drop it and get
  $p(\mathbf{x}_{t-1} \mid \mathbf{z}_{1:t-1}, \mathbf{u}_{1:t-1}) = \operatorname{bel}(\mathbf{x}_{t-1})$.
  This is the third, usually unstated, assumption the exercise asks for. It can
  fail if, say, a controller picks $\mathbf{u}_t$ based on information the
  filter does not have.

Grading: full credit requires all three locations identified correctly.

## Exercise 2: A five-cell world by hand

Doors in cells 0 and 1. $p(\text{door}\mid\text{door cell}) = 0.8$,
$p(\text{door}\mid\text{wall}) = 0.1$. Move: advance one cell with 0.8, stay with
0.2, circular.

**Step 1, sense "door."** Unnormalized: $0.2 \times [0.8, 0.8, 0.1, 0.1, 0.1]
= [0.16, 0.16, 0.02, 0.02, 0.02]$, sum 0.38:

$$
\operatorname{bel}_1 = \tfrac{1}{19}[8, 8, 1, 1, 1] = [0.421, 0.421, 0.053, 0.053, 0.053].
$$

**Step 2, move.** $\overline{\operatorname{bel}}(i) = 0.8\,\operatorname{bel}_1(i-1) + 0.2\,\operatorname{bel}_1(i)$:

$$
\overline{\operatorname{bel}}_2 = \tfrac{1}{19}[2.4, 8, 6.6, 1, 1] = [0.126, 0.421, 0.347, 0.053, 0.053].
$$

(Cell 0 receives from cell 4, which wraps around.)

**Step 3, sense "door."** Multiply by $[0.8, 0.8, 0.1, 0.1, 0.1]$:
$\tfrac{1}{19}[1.92, 6.4, 0.66, 0.1, 0.1]$, sum $9.18/19$:

$$
\operatorname{bel}_2 = [0.209, 0.697, 0.072, 0.011, 0.011].
$$

Most probable: **cell 1**. Interpretation: two consecutive "door" readings with a
move in between are best explained by having started in cell 0 and moved to
cell 1; the 0.2 "stay" probability is why cell 0 keeps about 21%.

## Exercise 3: Boundary conditions

```python
b0 = gaussian_pdf(x, 9.0, 0.3); b0 /= b0.sum()
T_col = gaussian_pdf(x[:, None], x[None, :] + commanded_motion, motion_sigma)
T_col /= T_col.sum(axis=0, keepdims=True)
T_leak = gaussian_pdf(x[:, None], x[None, :] + commanded_motion, motion_sigma) * dx
L = x[-1] - x[0] + dx
d = (x[:, None] - x[None, :] - commanded_motion + L / 2) % L - L / 2
T_circ = gaussian_pdf(d, 0.0, motion_sigma); T_circ /= T_circ.sum(axis=0, keepdims=True)
for name, T_ in [("column", T_col), ("leak", T_leak), ("circular", T_circ)]:
    b = b0.copy()
    for _ in range(10):
        b = T_ @ b; b /= b.sum()
    print(name, b[x >= 9.5].sum())
```

Expected (grid $[0, 10]$ m, 1000 points, motion 1 m with $\sigma = 0.55$ m):

| Model | Mass in last 0.5 m | Mean | Physical meaning |
|---|---|---|---|
| column-normalized | 0.86 | 9.75 m | robot is stopped by the wall and stays near it |
| leak + renormalize | 0.70 | 9.61 m | robot is *known* to still be in the hallway; we condition on the hypotheses that moved least |
| circular | 0.10 | 6.16 m | ring corridor: the robot reappears at the start (after ten 1 m steps on a 10 m ring it is back near 9 m with a 1.8 m spread) |

For a robot that genuinely stops at a wall, the column-normalized matrix is the
closest of the three, but it is still an approximation (it rescales the whole
truncated Gaussian rather than putting the overflow exactly at the wall). The
best model puts the overflow probability into the last cell explicitly. The
lesson the chapter wants: every option is a modeling decision, so make it on
purpose.

## Exercise 4: Terrain-relative navigation

A working solution with 5 m cells, using a shift plus convolution for prediction
(the transition is shift-invariant, so a 4000 x 4000 dense matrix is
unnecessary; this is the "exploit sparsity" advice in practice), and
log-likelihoods for the update:

```python
import numpy as np

rng = np.random.default_rng(1)
cell = 5.0
x = np.arange(0.0, 20_000.0, cell)
depth = 120.0 - 0.001 * x
for k in range(1, 30):
    depth += rng.normal(0, 3.0 / k) * np.sin(2 * np.pi * k * x / 20_000.0 + rng.uniform(0, 2 * np.pi))
seamount = lambda c: 15.0 * np.exp(-0.5 * ((x - c) / 150.0) ** 2)
depth -= seamount(6_000.0) + seamount(14_000.0)       # two identical features

step, step_sigma, sigma_z = 50.0, 5.0, 0.5
kernel_x = np.arange(-30, 31) * cell
kernel = np.exp(-0.5 * (kernel_x / step_sigma) ** 2); kernel /= kernel.sum()
shift = int(step / cell)

def predict(bel):
    moved = np.zeros_like(bel); moved[shift:] = bel[:-shift]
    out = np.convolve(moved, kernel, mode="same")
    return out / out.sum()

def update(bel, z):
    log_l = -0.5 * ((z - depth) / sigma_z) ** 2
    post = bel * np.exp(log_l - log_l.max())
    return post / post.sum()

x_true, bel = 5_000.0, np.full(x.size, 1.0 / x.size)
for t in range(1, 121):
    x_true += step + step_sigma * rng.standard_normal()
    z = np.interp(x_true, x, depth) + sigma_z * rng.standard_normal()
    bel = update(predict(bel), z)
```

With this seed: about six modes hold more than 5% of the mass after the first
measurement (one depth value matches many places), three after five steps, one
after ten, and the mass within 50 m of the truth is essentially 1 by step 20
(1 km of travel). It then stays near 0.9, limited by the 5 m motion noise per
step and the flat stretches of seabed.

Discussion points: a single depth reading is wildly ambiguous, so the belief is
initially many-peaked; it is the *sequence* of depths that identifies position.
Repeated features (the twin seamounts) keep two hypotheses alive whenever the
vehicle is on or near one of them and has not yet seen distinguishing terrain;
starting the vehicle at 13.5 km instead of 5 km makes that visible. Flat terrain
gives no information at all, and the belief spreads along it like dead
reckoning. Students who use the dense transition matrix with 5 m cells will run
out of memory or patience; that is part of the lesson.

The ethical aside about TERCOM is intentional and connects to the course theme
that the same mathematics is used for good or ill.

## Exercise 5: How lucky was that run?

```python
def localization_curve(doors, p_false=0.05, width=0.25, n_seeds=40, n=8):
    mass = np.zeros((n_seeds, n))
    lik = p_door(hallway, doors=doors, p_false=p_false, width=width)
    for s in range(n_seeds):
        rng_s = np.random.default_rng(s)
        x_t, bel = 0.0, np.full(hallway.size, 1.0 / hallway.size)
        for t in range(n):
            x_t += step + step_sigma * rng_s.standard_normal()
            z = rng_s.random() < p_door(np.array([x_t]), doors=doors,
                                        p_false=p_false, width=width)[0]
            bel = update_bayes(predict_bayes(bel, transition), lik if z else 1 - lik)
            mass[s, t] = bel[np.abs(hallway - x_t) < 0.5].sum()
    return mass.mean(axis=0), (mass[:, -1] > 0.5).mean()
```

Expected results (40 seeds):

| Configuration | Mean mass near truth, $t = 1 \ldots 8$ | Localized at $t = 8$ |
|---|---|---|
| chapter settings | 0.05, 0.10, 0.11, 0.30, 0.31, 0.34, 0.38, 0.43 | 35% |
| false alarms 1% | 0.05, 0.11, 0.12, 0.41, 0.40, 0.42, 0.45, 0.50 | 50% |
| door width doubled | 0.05, 0.09, 0.10, 0.27, 0.31, 0.41, 0.47, 0.51 | 58% |

Seed 7 reaches 0.93 at $t = 8$, so it is one of the best runs, not a typical
one.

Interpretation: the dominant source of ambiguity is that the robot samples the
hallway only once per meter while its position random-walks, so it frequently
steps past a 0.5 m door without seeing it. Making the door detectable over a
wider region attacks that directly and helps most; reducing false alarms helps
less because false alarms were not the main problem. A good answer also notes
that a "no door" reading only down-weights the door cells (likelihood 0.95 for
wall versus 0.10 at a door center) and treats every wall cell identically, so it
says almost nothing about *where along the walls* the robot is. Most of the
information arrives in the few steps where the robot is actually in front of a
door.
