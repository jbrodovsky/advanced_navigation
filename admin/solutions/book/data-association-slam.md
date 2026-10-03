# Solutions: Data Association for Landmark SLAM (`data-association.qmd`, `sec-slam-association-exercises`)

Instructor-only. (The sensing chapter's `data-association.qmd` has its own file,
`data-association.md`.) Code assumes the chapter's cells (`range_bearing`,
`landmark_jacobian`, `gated_d2`, `joint_d2`, `nearest_neighbor`, `jcbb`,
`ekf_update`, `R`, `P_tree`, `LOG_P0`) and, for Exercises 4 and 6, the FastSLAM
classes of `slam.qmd`. All numbers were produced by running that code with the
seeds stated.

## Exercise 1: Chi-square gates

**(a)** In polar coordinates $u_1 = \rho\cos\phi$, $u_2 = \rho\sin\phi$, the density of
$\mathbf{u}$ is $\frac{1}{2\pi}e^{-\rho^2/2}\rho\,d\rho\,d\phi$. Integrating out $\phi$ and
substituting $s = \rho^2$ ($ds = 2\rho\,d\rho$) gives $f(s) = \tfrac12 e^{-s/2}$, an
exponential with mean 2. So $P(\|\mathbf{u}\|^2 \le \gamma) = 1 - e^{-\gamma/2}$.

**(b)** Write $\boldsymbol{\nu} = \mathbf{S}^{1/2}\mathbf{u}$ with $\mathbf{u} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ (any
square root, e.g. Cholesky). Then $\boldsymbol{\nu}^T\mathbf{S}^{-1}\boldsymbol{\nu} = \mathbf{u}^T\mathbf{u}$.

**(c)** $\gamma = -2\ln(1-P)$ reproduces 4.605, 5.991, 9.210, 13.816. For a 3-D
measurement (range, bearing, elevation), `chi2.ppf(0.95, 3)` $= 7.81$ and
`chi2.ppf(0.999, 3)` $= 16.27$. The chi-square CDF has an elementary closed form
only for even degrees of freedom (a finite Poisson sum); for odd degrees it
involves the error function, so its inverse has no tidy expression. Use the
library.

## Exercise 2: Joint compatibility by hand

**(a)** Both bearings depend on heading with derivative $-1$, so
$\mathbf{H}_\theta = [-1, -1]^T$ and
$\mathbf{S} = \mathbf{H}_\theta\sigma_\theta^2\mathbf{H}_\theta^T + \sigma_\beta^2\mathbf{I} = \sigma_\theta^2\mathbf{1}\mathbf{1}^T + \sigma_\beta^2\mathbf{I}$.
In degrees, $\mathbf{S} = \begin{bmatrix} 26 & 25 \\ 25 & 26 \end{bmatrix}$ deg$^2$.

**(b)** Individually, $d^2 = 36/26 = 1.38$ for each, below the 1-dof 95% gate
of 3.84: both compatible. Jointly,
$\mathbf{S}^{-1} = \frac{1}{51}\begin{bmatrix} 26 & -25 \\ -25 & 26 \end{bmatrix}$ and
$D^2 = \frac{36}{51}(26 + 26 + 2\cdot 25) = 72.0 \gg 5.99$: jointly incompatible.

**(c)** For $(+6, +6)$: $D^2 = \frac{36}{51}(26 + 26 - 50) = 1.41 < 5.99$: compatible,
and smaller than the sum of the individual values (2.77).

Explanation via the eigenvectors of $\mathbf{S}$: the common mode
$(\nu_1 + \nu_2)$ has variance $4\sigma_\theta^2 + 2\sigma_\beta^2 = 102$, because a heading
error moves both bearings together, while the differential mode $(\nu_1 - \nu_2)$
has variance only $2\sigma_\beta^2 = 2$. In (b) the innovations are purely
differential, $12^2/2 = 72$: no heading error can explain opposite shifts, so only
the small sensor noise is left to explain them. In (c) they are purely common
mode, $12^2/102 = 1.41$: one heading error explains both at once, which is more
plausible than two independent 1.2-sigma coincidences.

## Exercise 3: How often is JCBB fooled?

Scene: the chapter's construction as a function of the seed, with the true heading
error drawn from $\mathcal{N}(0, (6^\circ)^2)$ and the position offset kept at
$(0.1, -0.1)$ m. "Wrong" means at least one pairing that disagrees with the truth,
including pairing the clutter return with a tree. 200 scenes each:

| Grove | Method | Scenes with a wrong pairing | Mean (median) heading error after update |
|---|---|---|---|
| $[5,10]\times[-4,4]$ m | nearest neighbor | 168 (84%) | 2.35° (1.69°) |
| | JCBB | 68 (34%) | 1.34° (1.07°) |
| $[5,10]\times[-2,2]$ m (denser) | nearest neighbor | 180 (90%) | 2.68° (1.78°) |
| | JCBB | 100 (50%) | 1.33° (1.12°) |

When JCBB is wrong, its hypothesis is jointly compatible by construction, and in
63 of 68 cases (93 of 100 in the dense grove) the *true* hypothesis was jointly
compatible too. JCBB chose the other one for one of two reasons. In 30 of the 68
(27 of 100 dense), the only error is the clutter return paired with one of the
undetected mapped trees: six pairings beat five, and clutter that happens to fall
where an undetected tree should be is geometrically indistinguishable from a
detection. In 26 (49 dense), it chose a hypothesis with the same number of
pairings but a swapped tree and a smaller $D^2$: an aliased but consistent
explanation. The remaining 12 (24) mix both.

Lesson: joint compatibility removes *inconsistent* hypotheses; it cannot choose
among *consistent* ones, and the "most pairings" rule actively prefers explaining
clutter with an unseen landmark. The guarantees end where geometry stops
discriminating. Remedies students may propose: negative information (a mapped
tree that should have been visible but was not), a visibility model, more
observations per scan, a tighter pose prior, or appearance descriptors. Despite
its errors, JCBB still halves the post-update heading error, because its wrong
hypotheses are by construction close to consistent.

## Exercise 4: Hungarian inside FastSLAM

Implementation: subclass `FastSLAM` and override `associate`. Build a
$K \times (N + K)$ cost with $-2\ln\mathcal{N}(\boldsymbol{\nu}; \mathbf{0}, \mathbf{S})$ for gated pairs, a
prohibitive cost for out-of-gate pairs, and $-2\ln p_0$ on the diagonal of the
$K$ "unmatched" columns; call `linear_sum_assignment`; then apply the birth gate
to unmatched observations (NEW if $d^2 > 13.82$ for every landmark, else
DISCARD).

| World | Greedy, exact maps / 100 | Hungarian, exact maps / 100 |
|---|---|---|
| original (3 trees) | 96 | 97 |
| plus a tree at (7, -0.7), 0.8 m from (7, -1.5) | 97 | 96 |

No meaningful difference on either world (the 20-seed tables are 20/20 versus
20/20, and 20/20 versus 19/20). For two observations to compete for one tree, a
particle's pose error must move one observation of tree B closer (in Mahalanobis
terms) to tree A than A's own observation is, which for trees 0.8 m apart means a
pose error of roughly half the spacing, about 0.4 m or 1.5 to 2 sigma of the
innovation, in the direction joining the trees. Particles that wrong carry tiny
weights and rarely survive resampling. The added tree also sits almost exactly
along the line of sight from the path (the robot passes at $y \approx 1.6$, so the
two trees are at about 2.3 m and 3.1 m range when abreast): they are separated by
$d^2 \approx 9$ in range alone, and when seen from farther back, by about 12° in
bearing. Hungarian pays off when trees are side by side across the line of sight
at long range and the particle cloud is wide, e.g. right after a long stretch
without landmarks.

## Exercise 5: Buoys in a channel

Pose covariance $\operatorname{diag}(5^2, 5^2, (2^\circ)^2)$, radar $\mathbf{R} = \operatorname{diag}(10^2, (1^\circ)^2)$,
buoy at $(1500, 100)$ m, chart positions treated as exact.

**(a)** $\mathbf{S} = \mathbf{H}_x\mathbf{P}_{xx}\mathbf{H}_x^T + \mathbf{R}$ gives a range standard deviation of
11.2 m ($\sqrt{10^2 + 5^2}$) and a bearing standard deviation of 2.24°, i.e. 59 m
across the line of sight at 1.5 km. The neighbor 150 m farther along the
channel is at $d^2 = 179$: no confusion, range separates it. The neighbor 200 m
across the channel is at $d^2 = 11.6$: outside the 95% association gate (5.99) but
inside the 99.9% birth gate (13.82). With one observation, the across-channel
pair is individually ambiguous; the two-gate rule would discard rather than
guess.

**(b)** A port-starboard swap at 1.5 km needs a 200 m lateral shift, i.e. a heading
error of $200/1500$ rad $= 7.6°$ (3.8 sigma), or a 200 m position error, which the
GNSS rules out. With buoys observed at two ranges (say 750 m and 1.5 km), a
heading error shifts them laterally in proportion to range (a swap would need
15.3° for the near pair and 7.6° for the far pair), while the swap requires the
same 200 m at both. No single pose error explains it, so the swapped hypothesis is
jointly incompatible as soon as two pairs at different ranges are tested
together. Individual gates cannot see this. Worth noting: magnetic compass
deviation on a steel hull can reach several degrees, so a 7.6° heading error is
not fantasy, which is exactly when the joint test earns its keep.

**(c)** A racon (radar transponder) paints a Morse identifier on the radar
display, and an AIS aid-to-navigation broadcasts its identity and position. Either
turns association into a lookup, exactly as AprilTag IDs did for Project 1
(@sec-feature-association): the remaining job is gating for outliers and misreads,
not deciding which buoy is which.

## Exercise 6: Sampling versus maximum likelihood

Implementation: in each particle, process observations in order of their best
$d^2$; for each, collect the unclaimed landmarks inside the 5.99 gate; if any,
sample among them plus a "discard" option with likelihood $p_0$ (spawning is still
reserved for observations beyond 13.82 from every landmark, so the two-gate rule
is respected), and add $\ln(\sum_j \mathcal{N}_j + p_0)$ to the log weight; otherwise
apply the birth gate as before.

Four-tree world (extra tree 0.8 m from tree 1):

| Association | 20 seeds | 100 seeds exact | Final error [m] | Scan-1 ancestors |
|---|---|---|---|---|
| maximum likelihood (greedy) | 20/20 exact | 97 | 0.27 | 2.5 |
| sampled | 19/20 exact (one with 5) | 95 | 0.28 | 2.4 |

Sampling does not help here and costs a little. Observations in this world are
rarely ambiguous (Exercise 4 explains why), so sampling picks the maximum
likelihood answer almost every time, and when it does not, it occasionally picks
the discard option or the worse tree, which can seed a duplicate. Its advantage
appears when ambiguity is real and persists for several scans, and when a wrong
hard decision would be irreversible: then keeping both answers alive in different
particles lets later scans decide. Full credit for an explanation along these
lines that is supported by the student's numbers.
