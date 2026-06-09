# Advanced Navigation: Core 16-Week Course Outline

## Course Identity

**Working Title:** Advanced Autonomous Navigation
**Level:** Graduate (assumes undergraduate navigation, probability, and programming background)
**Format:** Lecture slides + recorded video + Jupyter notebook demonstrations + projects
**Language:** Python 3.x (NumPy, SciPy, Matplotlib, OpenCV)

### Prerequisites
- Basic Bayes filter and Kalman filter (undergraduate navigation or equivalent)
- Intermediate probability theory (Gaussian distributions, conditional probability)
- Intermediate linear algebra (matrix operations, eigenvalues)
- Intermediate Python programming

### Recommended Companion Courses
- Computer Vision (RBE 549 or equivalent) — sensor models and image-based observation models
- Motion Planning (RBE 550 or equivalent) — guidance and path planning

### Learning Outcomes
By the end of this course a student should be able to:
1. Identify where the linear/small-scale approximations of introductory navigation break down and select appropriate techniques to address each failure mode
2. Implement an Extended Kalman Filter, Unscented Kalman Filter, and Particle Filter for a nonlinear navigation problem and justify the choice of filter for a given application
3. Derive and implement observation models for real sensors (cameras, IMUs, range/bearing sensors) as probabilistic measurement functions
4. Implement EKF-SLAM and explain the GraphSLAM and FastSLAM alternatives and their computational tradeoffs
5. Describe the architecture of a multi-sensor navigation system and analyze the observability of a given sensor suite relative to a navigation state definition
6. Connect navigation state estimation to the broader guidance, navigation, and control stack: how navigation uncertainty constrains guidance decisions and what control feedback requires from the navigation solution

---

## Textbooks and References

| Abbreviation | Reference |
|---|---|
| PR | *Probabilistic Robotics* — Thrun, Burgard, Fox |
| KBFP | *Kalman and Bayesian Filters in Python* — Labbe |
| RVC | *Robotics, Vision and Control* (3rd ed., Python) — Corke |
| GNSS | *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems* (2nd ed.) — Groves |
| SHR | *Springer Handbook of Robotics* — Siciliano & Khatib |

---

## Projects Overview

| Project | Topic | Weight | Due (suggested) |
|---|---|---|---|
| P1 | Dead Reckoning and Sensor Characterization | 15% | End of Block 1 (Week 3) |
| P2 | Drone Observation Model (PnP) | 20% | End of Block 2 (Week 9) |
| P3 | Nonlinear Navigation Filters | 35% | End of Block 3 (Week 13) |
| P4 | Victoria Park SLAM | 30% | End of semester (Week 16) |

---

## Block 1 — Foundations (Weeks 1–3)

*Organizing question: What does "knowing where you are" actually require, and where do the simple approximations break down?*

---

### Week 1 — What Is Navigation?

**Topics**
- Guidance, Navigation, and Control as a discipline: definitions, history, and the relationship between the three
- The navigation problem: observer, state, measurement, and the role of uncertainty
- Survey of navigation domains: ground vehicles, UAVs, marine surface, submarines, spacecraft — same mathematical structure, different physical constraints
- The GNC stack interfaces: what navigation provides to guidance (state + uncertainty), what navigation requires from perception (observation models), what control requires from navigation (state feedback, latency, integrity)
- A brief history of navigation: dead reckoning, the longitude problem, inertial navigation, GNSS, and where probabilistic robotics fits in the timeline
- Why this is hard: error accumulation, sensor noise, nonlinearity, unknown environments

**Readings**
- KBFP: Chapter 0 (Preface and Introduction)
- PR: Chapters 1.1–1.3 (Introduction)
- GNSS: Chapter 1 (Introduction to Navigation — selected sections)

**Notebook Demo**
- 1D dead reckoning: integrate velocity to get position, add noise, watch error accumulate. Motivates the entire course in ~30 lines of code.

---

### Week 2 — Coordinate Frames and Rotation Representations

**Topics**
- Reference frames: body frame, navigation frame, Earth-centered Earth-fixed (ECEF), Earth-centered inertial (ECI), local level frame. When each is appropriate.
- Rotation matrices: properties, composition, active vs. passive convention
- Euler angles: roll, pitch, yaw (and heading vs. course), intrinsic vs. extrinsic rotations, the gimbal lock singularity
- Quaternions: motivation (no singularity, efficient composition), definition, conversion to/from rotation matrix and Euler angles, the double-cover problem
- Direction cosine matrices (DCM): the rotation matrix by another name, how aerospace conventions differ from robotics conventions
- Practical implications: why heading and yaw are not the same thing, why you cannot simply average Euler angles, and why the choice of representation matters for filter design

**Readings**
- RVC: Appendix A (Coordinate Frames and Transformations)
- GNSS: Chapter 2.1–2.3 (Reference Frames)

**Notebook Demo**
- Implement rotation matrix, Euler angle, and quaternion representations. Demonstrate gimbal lock. Convert between representations and verify round-trip consistency. Visualize 3D orientation.

---

### Week 3 — Sensors and Observation Models

**Topics**
- Sensor taxonomy: proprioceptive (inertial, odometry) vs. exteroceptive (GPS, camera, lidar, sonar)
- The observation model as the mathematical bridge from hardware to state estimate: p(z|x)
- Inertial Measurement Units (IMUs): what accelerometers and gyroscopes actually measure (specific force, angular rate — not acceleration and orientation), basic error model (bias, noise, scale factor)
- GNSS fundamentals: the pseudorange measurement, satellite geometry and DOP, typical error sources (ionosphere, multipath). Going from "GPS gives me a position" to understanding what that position is based on.
- Range and bearing sensors: lidar, sonar, radar — the beam model and the range/bearing observation model
- Camera as a navigation sensor: the projection model and why pose estimation from images is an observation model problem
- Sensor noise modeling: white noise, random walk, bias instability. Why Gaussian is often assumed, when it fails.

**Readings**
- PR: Chapters 5, 6 (Robot Motion, Robot Perception)
- SHR: Chapter 4 (Sensors), Chapter 20.1–20.4 (IMU sections)
- GNSS: Chapters 4–5 (INS and GNSS error modeling — selected sections)

**Notebook Demo**
- Simulate IMU measurements from a known trajectory: add bias, random walk, and white noise. Integrate raw IMU and show error growth. Compute Allan deviation from a static IMU dataset.

**Project 1 Released:** Dead Reckoning and Sensor Characterization

---

## Block 2 — Estimation (Weeks 4–9)

*Organizing question: How do we maintain a principled estimate of state given noisy, nonlinear sensors?*

---

### Week 4 — The Bayesian Estimation Framework

**Topics**
- Review: the Bayes filter as the theoretical foundation for all estimation in this course
- The belief, the prediction step (motion model), and the update step (measurement model)
- The Markov assumption and why it makes the filter tractable
- Why the full Bayes filter is intractable for navigation: state space dimensionality, memory, computation
- Two families of approximation: parametric (Kalman-style: represent the belief as a Gaussian) and non-parametric (particle-style: represent the belief as a set of samples)
- The tradeoff space: accuracy, computational cost, memory, applicability — setting up the next six weeks

**Readings**
- PR: Chapters 1.1–1.3, 2, 3.1–3.2
- KBFP: Chapters 2–4

**Notebook Demo**
- Implement the Bayes filter on a discrete 1D example. Visualize the belief as it evolves. Show how the belief collapses with good measurements and spreads with poor ones.

---

### Week 5 — The Kalman Filter

**Topics**
- Review: the Kalman filter as the optimal parametric Bayes filter for linear Gaussian systems
- The two required assumptions: linear process and measurement models, additive Gaussian noise
- Prediction step: propagating the mean and covariance through the process model
- Update step: the Kalman gain as a principled weighting between prediction and measurement
- Interpretations of the Kalman gain: R→0 (trust the sensor), Q→0 (trust the model)
- The process noise Q and measurement noise R as design parameters: physical meaning and tuning intuition
- Extended state vectors: augmenting the state with velocity, bias terms
- A worked navigation example: 2D constant-velocity target tracking with position measurements

**Readings**
- PR: Chapter 3.2 (Kalman Filter)
- KBFP: Chapters 5–8
- RVC: Chapter 6.2, Appendix H.1

**Notebook Demo**
- Implement the KF for 2D position+velocity tracking. Visualize the covariance ellipse at each step. Show filter behavior under mistuned Q and R.

---

### Week 6 — Nonlinearities: Where the Kalman Filter Breaks Down

**Topics**
- Why real navigation is nonlinear: the bicycle/unicycle motion model with heading, range/bearing measurement models, the coordinate frame rotation problem
- What happens when you apply a linear KF to a nonlinear system: the Gaussian is no longer preserved, bias is introduced, covariance underestimates true uncertainty
- Two-dimensional navigation with heading: the nonlinear process model, why the Kalman filter cannot directly handle it
- Linearization as a strategy: approximate the nonlinear function by its first-order Taylor expansion
- The Jacobian: definition, physical meaning as a local linear approximation, how to compute it analytically and numerically
- Propagation of a Gaussian through a nonlinear function: first-order and second-order approximations
- Setting up the Extended Kalman Filter derivation

**Readings**
- KBFP: Chapter 9 (Nonlinear Filtering)
- GNSS: Chapter 2 (selected nonlinearity sections)
- PR: Chapter 3.3 (preamble to EKF)

**Notebook Demo**
- Visualize what happens to a Gaussian pushed through a nonlinear function (e.g., polar to Cartesian). Compare true distribution vs. linearized approximation. Show where the approximation is good and where it fails.

---

### Week 7 — The Extended Kalman Filter

**Topics**
- EKF derivation: replace the linear model matrices A, C with Jacobians F, H evaluated at the current estimate
- The EKF prediction step: propagate mean through the nonlinear function, propagate covariance through the linearized function
- The EKF update step: same structure as KF, innovation computed in measurement space
- The 2D navigation example: bicycle model process Jacobian, GPS observation Jacobian, range/bearing observation Jacobian — derive all three
- EKF implementation patterns: separating the model functions from the filter loop, numerical Jacobians as a debugging tool
- EKF failure modes: linearization error with high nonlinearity, inconsistency, divergence. When to be suspicious.
- Consistency analysis: normalized innovation squared (NIS) and normalized estimation error squared (NEES) as diagnostic tools

**Readings**
- PR: Chapters 3.3, 7.4–7.5
- RVC: Chapter 6.2
- KBFP: Chapter 11

**Notebook Demo**
- Implement EKF for a ground robot with bicycle model kinematics and GPS+compass measurements. Simulate with and without linearization errors. Run NIS/NEES consistency check.

**Project 1 Due. Project 2 Released:** Drone Observation Model (PnP).

---

### Week 8 — The Unscented Kalman Filter

**Topics**
- Motivation: the EKF linearizes before propagating. What if we propagated first and then characterized the result?
- The unscented transform: sigma points as a deterministic sampling strategy, weighted mean and covariance from propagated sigma points
- Sigma point selection: the scaled unscented transform, the parameters α, β, κ and their physical meaning
- UKF prediction and update: the full algorithm without a single Jacobian
- UKF vs. EKF: accuracy comparison for highly nonlinear systems, computational cost comparison, implementation complexity comparison
- When to choose UKF over EKF: the practical tradeoff in real navigation system design
- Variants: the square-root UKF for numerical stability, the Cubature Kalman Filter as a related approach

**Readings**
- PR: Chapters 3.4, 7.7–7.8
- KBFP: Chapter 10

**Notebook Demo**
- Implement UKF for the same ground robot problem from Week 7. Compare EKF and UKF performance as a function of increasing nonlinearity. Profile computational cost of each.

---

### Week 9 — The Particle Filter

**Topics**
- Motivation: what if the belief is genuinely multimodal, or the noise is non-Gaussian? Neither EKF nor UKF can represent this.
- Sequential Monte Carlo: representing a distribution as a set of weighted samples
- The particle filter algorithm: sampling from the prior, weighting by the likelihood, resampling
- Resampling strategies: multinomial, systematic, residual. Particle degeneracy and the resampling dilemma.
- The particle filter for navigation: global localization (the "kidnapped robot" problem), recovery from filter divergence
- Computational cost: particles vs. covariance matrix — when the particle filter is worth it
- Rao-Blackwellization: exploiting conditional independence to reduce the particle count needed

**Readings**
- PR: Chapters 4, 8.3–8.5
- KBFP: Chapter 12 (if available)

**Notebook Demo**
- Implement a particle filter for 2D localization on a known map. Demonstrate global localization from uniform prior. Show particle degeneracy and the effect of different resampling strategies.

**Filter Design Lecture (end of week):** When to use KF, EKF, UKF, PF — a structured decision framework based on nonlinearity, noise characteristics, computational budget, and whether the problem is local or global localization.

**Project 2 Due. Project 3 Released:** Nonlinear Navigation Filters.

---

## Block 3 — Integration (Weeks 10–13)

*Organizing question: How do we navigate in a world where the map is not given?*

---

### Week 10 — Mapping

**Topics**
- The mapping problem: estimating the environment given a known trajectory — p(m|x₁:ₜ, z₁:ₜ)
- Why mapping is the inverse problem of localization
- Map representations: occupancy grids, feature-based (landmark) maps, topological maps — tradeoffs in memory, accuracy, and utility
- Occupancy grid mapping: the log-odds update rule, inverse sensor model for a range finder
- Feature-based maps: landmark representation, association between observations and map features, the data association problem
- The data association problem: nearest-neighbor, maximum likelihood, joint compatibility. This is one of the hardest problems in navigation.
- Sensor fusion for mapping: combining information from multiple sensor types into a single consistent map

**Readings**
- PR: Chapters 9, 10 (introductory mapping sections)
- RVC: Chapter 6.3

**Notebook Demo**
- Implement occupancy grid mapping from a simulated lidar sensor on a known trajectory. Visualize the map building in real time. Show the effect of sensor model parameters.

---

### Week 11 — SLAM and EKF-SLAM

**Topics**
- The SLAM problem: build a map and localize simultaneously — the chicken-and-egg problem
- SLAM as joint estimation: augmenting the robot state with landmark positions, p(xₜ, m | z₁:ₜ, u₁:ₜ)
- EKF-SLAM: the landmark-augmented state vector, the structure of the covariance matrix, the observation and update equations
- The correlation structure in EKF-SLAM: why landmarks become correlated over time and why this is both the power and the scaling problem of EKF-SLAM
- Data association in SLAM: Mahalanobis distance gating, managing new vs. seen landmarks
- Computational complexity: EKF-SLAM scales as O(n²) in the number of landmarks — why this limits applicability
- Consistency in EKF-SLAM: why the filter tends to be overconfident and the implications

**Readings**
- PR: Chapter 10
- SHR: Chapter 37
- RVC: Chapters 6.3–6.4

**Notebook Demo**
- Implement EKF-SLAM in a simulated 2D environment with range/bearing measurements to point landmarks. Visualize the growing covariance ellipses. Show the correlation structure of the joint covariance matrix.

---

### Week 12 — GraphSLAM and Factor Graphs

**Topics**
- Motivation: can we do better than EKF-SLAM's quadratic scaling and consistency issues?
- Graph-based SLAM: pose nodes, measurement edges, the SLAM posterior as a factor graph
- The information matrix (inverse covariance): why the SLAM graph is sparse in information form
- GraphSLAM: the offline batch optimization formulation, solving the linear system
- Factor graphs: a general framework for sparse probabilistic inference that unifies SLAM formulations
- Loop closure: what it is, why it is transformative for long-duration navigation, the graph structure before and after loop closure
- Solving the optimization: Gauss-Newton, Levenberg-Marquardt, the connection to least squares
- Modern backends: GTSAM and g2o as implementations of these ideas — connecting the math to tools practitioners use

**Readings**
- PR: Chapter 11
- RVC: Chapter 6.5

**Notebook Demo**
- Implement a simple pose-graph SLAM: build a graph from simulated odometry and loop closure measurements, solve with a least-squares backend, visualize the trajectory before and after optimization.

---

### Week 13 — FastSLAM and the Victoria Park Dataset

**Topics**
- Motivation for particle-based SLAM: EKF-SLAM scales poorly, GraphSLAM is offline. Can we do online SLAM with a particle filter?
- Rao-Blackwellized particle filters: exploit the conditional independence of landmarks given the trajectory
- FastSLAM: each particle carries a full trajectory and a set of per-landmark EKFs. Scales as O(n log n) in landmarks.
- FastSLAM 2.0: incorporating the most recent observation into the proposal distribution
- The Victoria Park dataset: a canonical outdoor SLAM benchmark — vehicle with GPS ground truth, lidar, and odometry
- Data association in practice: the challenges of the real dataset, handling spurious detections
- Comparing SLAM algorithms: EKF-SLAM, GraphSLAM, FastSLAM on the same problem — accuracy, consistency, computational cost

**Readings**
- PR: Chapter 13

**Notebook Demo**
- Work through the Victoria Park dataset with a FastSLAM implementation. Compare estimated trajectory against GPS ground truth.

**Project 3 Due. Project 4 Released:** Victoria Park SLAM.

---

## Block 4 — Advanced Topics (Weeks 14–16)

*Organizing question: How do real navigation systems work, and what does it take to move from a notebook to a deployed system?*

---

### Week 14 — Multi-Sensor Fusion Architectures

**Topics**
- Why multi-sensor fusion: no single sensor is sufficient for robust navigation. Each sensor has failure modes that others do not share.
- The integration architecture design space:
  - **Loosely-coupled:** sensors are processed independently into position/velocity/attitude estimates, which are then fused. Simple, modular, robust to individual sensor processor failures. Loses information.
  - **Tightly-coupled:** raw sensor measurements (pseudoranges, IMU delta-v) are fused directly in a single filter. More accurate, especially in degraded environments. More complex.
  - **Deeply-coupled (ultra-tight):** the navigation filter feeds back into the sensor signal processing (e.g., GPS receiver tracking loops). Maximum performance in jamming/spoofing environments. Highest complexity.
- INS/GNSS integration as the canonical example of each architecture
- Aiding sources: what sensors can you use when GPS is unavailable? (visual odometry, lidar odometry, barometric altitude, magnetometer, DVL, terrain matching)
- Sensor failure detection: chi-squared consistency tests, residual monitoring, graceful degradation
- The error-state (indirect) Kalman filter: why navigation engineers often filter the *error* in the navigation solution rather than the state directly

**Readings**
- GNSS: Chapters 13–14 (Integration architectures — selected sections)

**Notebook Demo**
- Simulate INS/GPS loosely-coupled integration. Remove GPS periodically to simulate outages. Compare performance vs. tightly-coupled. Show how error grows during GPS outage and is corrected on re-acquisition.

---

### Week 15 — Navigation in the GNC Stack

**Topics**
- Observability in navigation systems: a state is observable only if the sensor suite provides information about it. Review the linear systems definition, extend to nonlinear systems intuitively.
- Worked examples of unobservable states: yaw from accelerometers alone, position from IMU alone, velocity from GPS alone (static case). Each has engineering significance.
- Navigation performance metrics: CEP (Circular Error Probable), DRMS, 2DRMS, HDOP/VDOP — the language of navigation system specifications. Connecting filter covariance to operational requirements.
- How navigation uncertainty feeds guidance: chance-constrained path planning, safety margins, the cost of navigation uncertainty in a mission. A conceptual treatment without teaching planning.
- Navigation integrity: the difference between accuracy (how close is the estimate to truth?) and integrity (how well does the system know when it is wrong?). RAIM as an integrity concept.
- Initialization: how do you start a navigation filter? Cold start vs. warm start, attitude alignment, the role of early motion in observability
- Latency and update rates: real sensors have different rates (IMU at 200 Hz, GPS at 1 Hz, camera at 30 Hz). How do you handle asynchronous measurements?

**Notebook Demo**
- Observability analysis: simulate a sensor suite, check the observability Gramian, identify unobservable state directions. Show how adding a sensor makes previously unobservable states observable.

---

### Week 16 — Where the Field Is Going

**Topics**
- Modern SLAM systems: brief survey of deployed systems (ORB-SLAM3, LOAM/LeGO-LOAM, MSCKF, OpenVINS) — what algorithms they use and why, connecting back to the course material
- Visual-Inertial Odometry as a case study in tight coupling: IMU pre-integration, the VIO factor graph, why it has become the dominant approach for MAV navigation
- Learning-based navigation: where deep learning is genuinely useful (feature extraction, place recognition, loop closure detection, outlier rejection) vs. where classical estimation remains superior
- Learned uncertainty: neural networks that output calibrated covariances — connecting deep learning outputs back to Bayesian filters
- Open problems: long-duration autonomy, navigation in truly unknown environments, certifiable/safe navigation systems, multi-robot collaborative SLAM
- What a navigation engineer needs to know in 2025 and beyond: the skills the course has built and the directions they lead

**Notebook Demo**
- Run a public VIO system (e.g., OpenVINS or MSCKF on EuRoC MAV dataset). Visualize the factor graph. Connect each component back to a topic from the course.

**Project 4 Due (end of semester):** Victoria Park SLAM.

---

## Assessment Philosophy

Projects are graded on four levels: **Excellent / Acceptable / Marginal / Unacceptable**.

An **Acceptable** rating on all deliverables corresponds to a B. This reflects the philosophy that a solid, correct, well-documented implementation that demonstrates genuine understanding of the material is the baseline of graduate-level competence.

**In the agentic AI era**, coding agents can produce working implementations of all algorithms in this course. The assessment therefore places increasing weight on the *written report* component, which requires:
- Justifying design choices (why this filter? why this state definition? why this observation model?)
- Analyzing failure modes (what breaks and why? how does the covariance behave?)
- Connecting results to the underlying theory
- Proposing and evaluating alternatives

Code that runs is necessary but no longer sufficient for an Excellent rating. Understanding demonstrated in writing is the differentiator.

---

## Project Descriptions (Summary)

### P1: Dead Reckoning and Sensor Characterization
Implement IMU integration (dead reckoning) from a real or simulated IMU dataset. Characterize sensor noise using Allan deviation analysis. Demonstrate error growth over time. Discuss what aiding sources would be needed and at what rate to bound the error to a specified tolerance. *This project builds physical intuition for why estimation is necessary before any filter is introduced.*

### P2: Drone Observation Model (PnP)
Using data from a quadrotor flying over an AprilTag mat, implement the Perspective-N-Point observation model to estimate drone pose from camera images. Estimate the measurement covariance empirically from ground truth. Document the observation model in a form suitable for direct use as the measurement function in P3. *This project is intentionally a component, not a standalone result — it feeds directly into P3.*

### P3: Nonlinear Navigation Filters
Using the observation model from P2 and IMU data from the same drone dataset, implement at least two of: EKF, UKF, Particle Filter for drone navigation. Compare filter performance quantitatively (RMSE, NIS/NEES consistency, computational cost). Justify filter selection for this specific problem. *The comparison and justification are the primary deliverables, not the implementation itself.*

### P4: Victoria Park SLAM
Implement a SLAM algorithm (any variant: EKF-SLAM, FastSLAM, or pose-graph) on the Victoria Park dataset. Evaluate performance against the GPS ground truth trajectory. Analyze the effect of data association decisions on map quality and trajectory accuracy. *The capstone: combines all course elements — estimation, observation models, data association, and performance evaluation.*
