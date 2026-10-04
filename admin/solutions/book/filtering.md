# Solutions: Filtering and State Estimation (`filtering.qmd`)

Instructor-only. Exercise numbers match `## Exercises {#sec-filtering-exercises}`.

## Exercise 1: One step by hand

**Prediction.** For a Gaussian belief and additive Gaussian motion noise, means
add and variances add:

- predicted mean: $\mu^- = 2.0 + 1.2 = 3.2$
- predicted variance: $(\sigma^-)^2 = 0.7^2 + 0.85^2 = 0.49 + 0.7225 = 1.2125$
  (standard deviation 1.101)

**Update.** The posterior density is proportional to the product of the
predicted density and the likelihood. The product of two Gaussians in $x$ is
proportional to a Gaussian whose precision is the sum of the precisions and
whose mean is the precision-weighted average of the means:

$$
\frac{1}{\sigma^{+2}} = \frac{1}{1.2125} + \frac{1}{0.3025} = 0.8247 + 3.3058 = 4.1305,
\qquad
\sigma^{+2} = 0.2421 \;(\sigma^+ = 0.492),
$$

$$
\mu^+ = \sigma^{+2}\left(\frac{3.2}{1.2125} + \frac{4.1}{0.3025}\right)
= 0.2421 \times (2.6392 + 13.5537) = 3.920.
$$

Equivalently, with the scalar Kalman gain $K = 1.2125/(1.2125 + 0.3025) = 0.8003$:
$\mu^+ = 3.2 + 0.8003(4.1 - 3.2) = 3.920$ and $\sigma^{+2} = (1 - 0.8003)\,1.2125 = 0.2421$.

**Numerical check.** Appending

```python
mean = np.trapezoid(x * posterior, x)
var = np.trapezoid((x - mean) ** 2 * posterior, x)
```

to the figure cell gives mean 3.9203 and variance 0.2421, matching the
analytical result. Students should notice the posterior is much closer to the
measurement than to the prediction because the measurement variance (0.30) is
four times smaller than the predicted variance (1.21).

## Exercise 2: Dead reckoning between fixes

```python
import numpy as np
import matplotlib.pyplot as plt

var_u, var_z = 0.25, 1.0       # nmi^2 per hour of DR, and per celestial fix
P = 0.1
t, P_hist = [0.0], [P]
for hour in range(1, 49):
    P += var_u                         # prediction: one hour of dead reckoning
    t.append(hour); P_hist.append(P)
    if hour % 6 == 0:                  # celestial fix
        P = P * var_z / (P + var_z)    # scalar update, H = 1
        t.append(hour); P_hist.append(P)

plt.plot(t, P_hist)
plt.xlabel("time [h]"); plt.ylabel("position variance [nmi$^2$]")
plt.show()
```

The plot is a sawtooth: the variance grows linearly between fixes (each hour of
dead reckoning adds 0.25 nmi²) and drops at each fix. Prediction only ever adds
uncertainty; updates only ever remove it.

**Steady state.** Let $P_a$ be the variance just after a fix. Six hours later
it is $P_b = P_a + 1.5$, and the fix gives $P_a = P_b/(P_b + 1)$. Substituting,

$$
P_a(P_a + 2.5) = P_a + 1.5 \;\Rightarrow\; P_a^2 + 1.5P_a - 1.5 = 0
\;\Rightarrow\; P_a = \frac{-1.5 + \sqrt{8.25}}{2} = 0.686\ \text{nmi}^2,
$$

and $P_b = 2.186\ \text{nmi}^2$. The simulation reaches these values after two
or three fixes (it prints 0.6861 and 2.1861 at hours 42 and 48). In standard
deviations, the navigator's position is good to about 0.83 nmi right after a
fix and degrades to about 1.48 nmi just before the next one, which is a fair
description of pre-GNSS celestial navigation with a good dead-reckoning plot.

## Exercise 3: Design questions for a real vehicle

Open-ended; grade on whether each answer is specific, frame-aware, and
internally consistent. A model answer for the AUV:

- **State and frames.** Position in a local NED frame (or latitude, longitude,
  depth), velocity in NED, attitude $\mathbf{q}_b^n$, IMU biases, and a DVL
  misalignment or scale-factor state if the mission is long. Depth is measured
  along the down axis; DVL velocity is measured in the DVL frame and must be
  rotated through the DVL-to-body mounting and $\mathbf{C}_b^n$.
- **Propagation vs. update.** The IMU drives propagation (it is the input). The
  DVL (body-frame velocity), the depth sensor (down position), and the acoustic
  fixes (USBL or LBL position or ranges) are measurement updates. A student who
  propagates with the DVL and treats the IMU as a measurement should be asked to
  justify it; it is possible but unusual.
- **Noise.** IMU white noise and bias random walk enter as process noise through
  the inputs; DVL noise and depth noise are measurement noise; unmodeled
  currents (if DVL bottom lock is lost and it measures water-relative velocity)
  are hidden in process noise unless a current state is added.
- **Modality.** Navigation with DVL and acoustic fixes is normally unimodal. It
  can become multimodal if acoustic ranges have ambiguous geometry (a single
  transponder gives a circle of positions) or if terrain-relative navigation is
  used over repetitive seabed. A Gaussian filter is appropriate for the core
  problem; a particle or grid filter might be used for terrain matching.
- **Consistency checks.** NIS on DVL, depth, and acoustic updates during the
  mission; NEES against a post-processed smoothed trajectory (with USBL) after
  the mission.

For the drone: IMU drives propagation; barometer, GNSS, and camera
(visual-odometry velocity or feature positions) are updates; the belief is
unimodal except possibly visual relocalization in repetitive scenes; GNSS
multipath and barometric drift from prop wash are the noise sources most often
hidden in $\mathbf{Q}$ or $\mathbf{R}$.
