# Advanced Autonomous Navigation: Expanded Open Course Outline

## Vision and Philosophy

This course is intended as a public-domain resource in the tradition of Roger Labbe's
*Kalman and Bayesian Filters in Python* — grounded in implementation, freely available,
and organized so that the theory and the code are inseparable. Every concept is accompanied
by a runnable Jupyter notebook. The mathematics explains the code; the code validates
the mathematics.

**The unifying mission:** unite the robotics localization world and the traditional aerospace
and marine navigation world. These communities solve the same problem with different vocabulary,
different sensors, and different conventions. This course treats them as one field.

**The guiding philosophy:** *Code is cheap; software is not.* The ability to generate a
working Kalman filter is no longer a differentiator. Understanding why you chose that filter,
what its failure modes are, how it fits into a larger system, and how to evaluate whether it
is working — that is the expertise this course develops.

---

## Audience

| Background | Entry Point | Primary Goal |
|---|---|---|
| Roboticist (knows Python, basic KF, probabilistic robotics) | Pre-Course or direct to Core | Gain nonlinear estimation, SLAM, and domain extension to aerospace/marine |
| Aerospace/Marine GNC engineer (knows INS/GNSS, Kalman filters, possibly MATLAB) | Pre-Course Unit 1–2 or direct to Core | Gain probabilistic framing, SLAM, modern implementation practices |
| Software engineer / physicist with strong math | Pre-Course Unit 0 onward | Full pathway from foundations to advanced topics |
| Graduate student (navigation, controls, or robotics) | Core directly (Pre-Course as reference) | Complete treatment of nonlinear navigation estimation |

---

## Delivery Format

- **Platform:** GitHub repository of Jupyter notebooks (primary), YouTube video series (companion)
- **Notebooks:** Each chapter is a self-contained notebook with theory, derivation, implementation, and exercises
- **Videos:** ~15–25 minutes per chapter, conceptual walkthrough with notebook as companion
- **Library:** A lightweight navigation utilities library (analogous to `filterpy`) providing reusable filter, sensor model, and visualization components
- **Datasets:** Real datasets used throughout — drone/AprilTag (in-house), Victoria Park (public), EuRoC MAV (public), and domain-specific datasets for post-course tracks

---

## Overall Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PRE-COURSE: Foundations                             │
│              (~4–5 weeks, ~5 notebooks)  — the on-ramp                      │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CORE: 16-Week Course                                  │
│   Block 1: Foundations (Wks 1–3)   Block 2: Estimation (Wks 4–9)           │
│   Block 3: Integration (Wks 10–13) Block 4: Advanced Topics (Wks 14–16)    │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                        ┌───────────┼───────────┬──────────────┐
                        ▼           ▼           ▼              ▼
              ┌──────────────┐ ┌─────────┐ ┌────────┐ ┌────────────────┐
              │   Track A    │ │ Track B │ │Track C │ │    Track D     │
              │  Inertial    │ │ Domain- │ │Modern  │ │  Nav System    │
              │ Nav & GNSS   │ │Specific │ │ SLAM & │ │   Design &     │
              │  Deep Dive   │ │   Nav   │ │  VIO   │ │   Practice     │
              └──────────────┘ └─────────┘ └────────┘ └────────────────┘
                              POST-COURSE TRACKS
                     (~3–4 weeks each, pick ≥1 based on domain)
```

---

# PART I: PRE-COURSE — Foundations

*Target audience: anyone who needs to reach the Core course starting line.*
*Can be skipped by students with the WPI prerequisites (RBE 3002 + RBE 500) or equivalent.*
*The aerospace/marine engineer may skim Unit 1–2 and read Unit 3 carefully.*

---

## Unit 0 — The Navigation Problem (~1 week, 1 notebook)

### Chapter 0.1: What Are We Trying to Do?

**Learning objectives**
- Define navigation as state estimation relative to an external reference
- Describe the observer/state/measurement triad
- Identify the three components of the GNC stack and their interfaces
- Explain why navigation is hard: error accumulation, noise, nonlinearity, unknown environments

**Content**
- Navigation defined: knowing where you are, where you are going, and how to get there — the GNC decomposition
- A tour of the problem space: indoor robot, autonomous car, aircraft, ship, submarine, spacecraft. Same mathematical structure, different physical constraints, different vocabulary.
- The dead reckoning thought experiment: starting from a known position, integrate velocity over time. Add a little noise. Watch it grow. This single experiment motivates the entire course.
- A brief, engaging history: from celestial navigation and the longitude problem to inertial navigation to probabilistic robotics — navigation as a continuous thread of the same fundamental problem
- What the course will teach: the toolkit for maintaining a principled state estimate in the face of all of the above

**Notebook:** `00_the_navigation_problem.ipynb`
- Implement 1D dead reckoning. Vary noise level. Plot position error vs. time. Show how a correction (a position fix) resets the error. End with: "how do we make this rigorous?" — which the course then answers.

---

## Unit 1 — Mathematical Toolkit (~1 week, 1 notebook)

### Chapter 1.1: Probability and Estimation

**Learning objectives**
- Compute with Gaussian distributions: PDF, mean, variance, covariance
- Apply Bayes theorem to update a belief given evidence
- Explain conditional independence and why the Markov assumption matters
- Implement basic probabilistic operations in NumPy

**Content**
- Probability as a model of uncertainty: frequentist vs. Bayesian interpretation (brief), why Bayesian is natural for navigation
- The Gaussian distribution: parameterization by mean and covariance, the covariance matrix as an uncertainty ellipsoid
- Bayes theorem: prior, likelihood, posterior. A navigation example: you believe you are at position x (prior), you measure something (likelihood), you update (posterior).
- Conditional probability and the Markov assumption: why the current state is all you need
- The sum rule and product rule: marginalization and conditioning in multivariate distributions

### Chapter 1.2: Linear Algebra for Estimation

**Learning objectives**
- Perform matrix operations relevant to filter implementation: transpose, inverse, determinant
- Interpret eigenvalues and eigenvectors of a covariance matrix geometrically
- Recognize positive definiteness and its role in covariance matrices
- Implement covariance ellipse visualization in matplotlib

**Content**
- Matrix multiplication review: emphasis on shapes and the meaning of each operation in filter equations
- The covariance matrix: symmetry, positive definiteness, eigendecomposition, the ellipse interpretation
- Matrix inverse: when it exists, numerical stability, the Woodbury identity (preview — used in filter derivations)
- Least squares as a preview of the Kalman filter: the connection between weighted least squares and optimal estimation

**Notebook:** `01_math_toolkit.ipynb`
- Implement and visualize Gaussian distributions and Bayes updates in 1D and 2D. Plot covariance ellipses. Show how multiplying covariances corresponds to fusing information. Short exercises with solutions.

---

## Unit 2 — State Estimation Basics (~1–2 weeks, 2 notebooks)

### Chapter 2.1: The Bayes Filter

**Learning objectives**
- Implement the discrete Bayes filter
- Explain the predict and update steps in terms of probability operations
- Identify when the Bayes filter is tractable and when it is not
- Connect the Bayes filter to the Kalman filter as a special case

**Content**
- The belief as a probability distribution over state space
- The predict step: propagate the belief through the motion model
- The update step: weight the belief by the likelihood of the measurement
- The Markov chain and why the filter only needs the previous state
- The intractability problem: why computing the full belief over a continuous state space is impossible in practice
- Preview: two families of solution — parametric (Gaussian) and non-parametric (particles)

**Notebook:** `02_bayes_filter.ipynb`
- Implement discrete Bayes filter for a 1D robot localizing on a grid. Animate belief propagation. Show localization converging. Deliberately keep simple — this is a pedagogical tool, not a practical filter.

### Chapter 2.2: The Kalman Filter

**Learning objectives**
- Derive the Kalman filter as the optimal Gaussian Bayes filter for linear systems
- Implement a Kalman filter for a tracking problem
- Tune Q and R through physical reasoning
- Diagnose filter performance using normalized innovation statistics

**Content**
- The Gaussian assumption: parameterize the belief as a mean vector and covariance matrix
- Linear systems: process model xₜ = Axₜ₋₁ + Buₜ + noise, measurement model zₜ = Cxₜ + noise
- The prediction equations: propagate mean and covariance through the linear model
- The update equations: Kalman gain as the optimal weighting between prediction and measurement
- Physical intuition: when should the filter trust the sensor vs. the model? The gain tells you.
- Tuning Q and R: physical meaning, starting heuristics, and what wrong tuning looks like
- A first worked example: 1D constant-velocity system with position measurements
- Introduction to consistency checking: NIS as a sanity test

**Notebook:** `03_kalman_filter.ipynb`
- Full KF implementation from scratch. Track a simulated 2D object (position + velocity). Experiment with Q and R. Plot covariance ellipses. Compute NIS and show what inconsistency looks like.

---

## Unit 3 — Simulation as a Tool (~1 week, 1 notebook)

### Chapter 3.1: Building Navigation Simulations

**Learning objectives**
- Build a reusable simulation framework for testing navigation algorithms
- Generate synthetic sensor data from a known ground truth trajectory
- Implement standard evaluation metrics: RMSE, NEES, NIS
- Run Monte Carlo simulations and interpret aggregate statistics

**Content**
- Why simulation: you need to know the truth to evaluate the filter. Real data gives you performance; simulation gives you understanding.
- The simulation framework: ground truth trajectory generator, sensor model (add noise per the specified model), filter, and evaluator
- Evaluation metrics:
  - RMSE (Root Mean Squared Error): how accurate is the estimate?
  - NIS (Normalized Innovation Squared): is the filter's innovation covariance consistent with the actual innovations?
  - NEES (Normalized Estimation Error Squared): is the filter's state covariance consistent with the actual errors?
- Monte Carlo runs: why a single run is not enough, how to aggregate statistics across runs
- The simulation sandbox that will be used throughout the core course

**Notebook:** `04_simulation_framework.ipynb`
- Build the complete simulation + filter + evaluation pipeline. Simulate a 2D robot with a KF. Run 100 Monte Carlo trials. Plot NEES and NIS over time. Show what a well-tuned filter looks like vs. an overconfident or underconfident one.

---

# PART II: CORE 16-WEEK COURSE

*See `course_outline_16week.md` for the full week-by-week breakdown.*

## Summary Structure

| Block | Weeks | Theme |
|---|---|---|
| Block 1: Foundations | 1–3 | What is navigation? Coordinate frames. Sensors. |
| Block 2: Estimation | 4–9 | Bayes filter → KF → EKF → UKF → PF → Filter selection |
| Block 3: Integration | 10–13 | Mapping → EKF-SLAM → GraphSLAM → FastSLAM |
| Block 4: Advanced Topics | 14–16 | Multi-sensor fusion architectures → GNC interfaces → Field frontiers |

## Core Projects

| Project | Topic | Weight |
|---|---|---|
| P1 | Dead Reckoning and Sensor Characterization | 15% |
| P2 | Drone Observation Model (PnP) | 20% |
| P3 | Nonlinear Navigation Filters (comparison + selection) | 35% |
| P4 | Victoria Park SLAM | 30% |

---

# PART III: POST-COURSE TRACKS

*Each track is modular and independent. A student picks the track(s) relevant to their domain.*
*Each track is ~3–4 weeks of material (~4–6 notebooks + capstone exercise).*
*Tracks can be taken in any order after completing the Core.*

---

## Track A — Inertial Navigation and GNSS Integration

*Target audience: roboticists who need to work with real IMUs and GPS; aerospace/marine engineers who want a probabilistic and implementation-grounded treatment of what they already know theoretically.*

*This is the Groves book made accessible. It is the most important post-course track for bridging the aerospace/marine and robotics worlds.*

---

### Chapter A.1: IMU Mechanization — Strapdown Inertial Navigation

**Learning objectives**
- Explain what an accelerometer and gyroscope actually measure (specific force and angular rate — not acceleration and orientation)
- Implement the strapdown inertial navigation algorithm in 3D
- Identify the sources of error in dead reckoning from IMU data
- Explain why high-grade IMUs can navigate for hours without external aiding

**Content**
- The strapdown algorithm: integrating angular rate to attitude, rotating specific force to navigation frame, integrating to velocity and position
- Earth rate and transport rate corrections: why navigation-grade INS must account for the rotation of the Earth and the motion of the navigation frame across the Earth's surface
- The coning and sculling problem: why naive integration of high-frequency motion introduces errors, and the corrections that address it
- IMU grade classification: MEMS (consumer/tactical), tactical-grade, navigation-grade, strategic-grade — performance specs, cost, and application domains
- Error propagation in free-inertial navigation: the Schuler oscillation, the runaway error modes

**Notebook:** `A1_imu_mechanization.ipynb`
- Implement the full 3D strapdown algorithm. Simulate a trajectory with a navigation-grade and a MEMS IMU. Show how position error grows differently. Visualize the Schuler oscillation.

---

### Chapter A.2: IMU Error Modeling and Characterization

**Learning objectives**
- Identify the primary IMU error terms: bias, scale factor, misalignment, noise
- Implement Allan variance/deviation analysis from static IMU data
- Use the Allan deviation plot to extract noise model parameters
- Design a Kalman filter that estimates IMU bias online

**Content**
- IMU error model: bias (constant and temperature-dependent), scale factor, axis misalignment, quantization noise, g-sensitivity
- Stochastic error models: angle random walk, rate random walk, bias instability — the three regimes visible on an Allan deviation plot
- Allan variance analysis: the statistical technique for characterizing time-correlated noise from a static dataset
- Reading an Allan deviation plot: extracting ARW, RRW, and bias instability parameters
- The IMU error state: augmenting the navigation state with bias estimates, modeling bias as a random walk process
- Calibration vs. estimation: what can be calibrated in the factory vs. what must be estimated online

**Notebook:** `A2_imu_characterization.ipynb`
- Compute Allan deviation from a real public IMU dataset. Identify noise parameters. Implement an error-state EKF that estimates position, velocity, attitude, and IMU biases simultaneously.

---

### Chapter A.3: GNSS In Depth

**Learning objectives**
- Explain how a GPS receiver computes a pseudorange measurement
- Calculate position from pseudoranges using least squares
- Interpret DOP values and explain how satellite geometry affects accuracy
- Identify the primary GNSS error sources and their magnitudes

**Content**
- How GPS works: satellite orbits, signal structure, pseudorange as a time-of-flight measurement contaminated by receiver clock error
- The position fix: solving the pseudorange equations, the four-satellite minimum (x, y, z, clock), least squares for overdetermined systems
- Dilution of Precision (DOP): HDOP, VDOP, PDOP — how satellite geometry scales the ranging errors into position errors. Why you want satellites spread across the sky.
- Error sources: ionospheric delay, tropospheric delay, multipath, satellite clock error, ephemeris error — typical magnitudes and mitigation approaches
- Differential GNSS and RTK: using a reference receiver to cancel common-mode errors, achieving centimeter-level accuracy
- GNSS constellations: GPS, GLONASS, Galileo, BeiDou — why multi-constellation receivers are more robust

**Notebook:** `A3_gnss_fundamentals.ipynb`
- Compute a position fix from simulated pseudoranges. Visualize DOP as a function of satellite geometry. Simulate the effect of adding and removing satellites. Demonstrate the multipath error model.

---

### Chapter A.4: GNSS-INS Integration

**Learning objectives**
- Implement a loosely-coupled INS/GNSS integration filter
- Explain the architectural difference between loosely-coupled and tightly-coupled integration
- Design an error-state Kalman filter for GNSS-aided INS
- Demonstrate the performance difference during GNSS outages

**Content**
- The error-state (indirect) Kalman filter: filtering the deviation from the INS solution rather than the full navigation state. Why navigation engineers prefer this architecture.
- Loosely-coupled integration: GPS position+velocity as measurements, INS as the process model. Simple, modular, works when ≥4 satellites are available.
- Tightly-coupled integration: individual pseudorange and Doppler measurements as inputs to the filter. Works with fewer than 4 satellites. More implementation complexity, more robustness.
- Deeply-coupled (ultra-tight) integration: the filter feeds back into GPS receiver tracking loops. Maximum anti-jamming performance. Beyond the scope of this chapter but explained conceptually.
- GNSS outage performance: how long can the INS coast? Depends on IMU grade. How does the error grow? What happens when GPS re-acquires?
- Practical implementation: handling reference frame transformations, lever arm corrections (the GPS antenna is not at the IMU)

**Notebook:** `A4_gnss_ins_integration.ipynb`
- Implement loosely-coupled GNSS-INS. Simulate GNSS outages of varying duration. Compare LC and TC on the same trajectory. Animate the navigation solution during an outage.

---

### Chapter A.5: Navigation Integrity

**Learning objectives**
- Define navigation integrity and distinguish it from accuracy
- Implement a chi-squared consistency test for sensor fault detection
- Explain the concept of RAIM and its role in aviation navigation
- Design a fault detection and exclusion scheme for a multi-sensor system

**Content**
- Integrity vs. accuracy: a system can be accurate on average but have no way of knowing when it is wrong. Integrity is the ability to detect and bound errors.
- The chi-squared test on filter innovations: when the innovation is larger than the filter predicts, something may be wrong
- Receiver Autonomous Integrity Monitoring (RAIM): using redundant satellites to detect and isolate faulty pseudoranges. The snapshot algorithm.
- Protection levels: not just an estimate of error, but a guaranteed bound on the error that the filter's integrity monitor will catch
- Fault detection and exclusion in a multi-sensor system: how to identify which sensor has failed when multiple sensors are fused
- Alert limits and time-to-alert in operational systems: the aviation context where integrity monitoring is a regulatory requirement

**Notebook:** `A5_integrity.ipynb`
- Implement chi-squared innovation monitoring on an INS/GNSS filter. Inject a simulated GPS fault. Show detection, identify the faulty sensor, exclude it, and show recovery.

**Track A Capstone Project**
Take a public IMU + GPS dataset (e.g., from the KITTI benchmark or a public UAV dataset). Implement strapdown INS. Implement loosely-coupled and tightly-coupled GNSS-INS. Remove GPS for a defined period. Compare accuracy during outage. Characterize the IMU using Allan deviation. Document the full sensor error budget.

---

## Track B — Domain-Specific Navigation

*Target audience: engineers or researchers working in a specific vehicle domain who want a deep treatment of the navigation challenges unique to that domain. Each sub-track is independent.*

*This is the most direct embodiment of the "preserving expertise before it exits the workforce" mission.*

---

### Track B.1 — Underwater Navigation (~2 notebooks)

**Chapter B.1.1: The Underwater Navigation Problem**
- Why GPS fails underwater: electromagnetic wave attenuation in seawater, the consequence for autonomous underwater vehicles (AUVs)
- The Doppler Velocity Log (DVL): how acoustic beams measure velocity over the seabed or through the water column. The DVL as a velocity sensor — its error model, beam geometry, altitude limitations.
- Acoustic positioning systems: long baseline (LBL), short baseline (SBL), ultra-short baseline (USBL) — how each works, accuracy vs. infrastructure tradeoff, and the observation model for each
- Depth as a free measurement: a pressure sensor gives nearly free depth information — using it to constrain the vertical channel
- The specific challenges: salinity and temperature gradients affecting acoustic propagation, ocean currents (DVL gives velocity relative to water, not ground), biofouling

**Chapter B.1.2: AUV Navigation Filter Design**
- Dead reckoning with DVL+IMU: the standard AUV approach — integrate DVL velocity, use IMU for attitude, accumulate position error over time
- DVL/INS integration filter: implementing EKF with DVL velocity measurements
- Acoustic aiding: incorporating USBL position fixes as infrequent, high-noise measurements
- Terrain-aided navigation for AUVs: using a bathymetric map as a natural landmark field — the underwater analog of the occupancy grid SLAM problem
- SLAM for AUVs: the challenges of feature-sparse underwater environments, using artificial landmarks or ship hulls

**Notebooks:** `B1_underwater_navigation.ipynb`, `B2_auv_filter_design.ipynb`

---

### Track B.2 — Aerial and UAV Navigation (~2 notebooks)

**Chapter B.2.1: Aerial Navigation Sensors**
- Barometric altimetry: pressure as a proxy for altitude, the ISA model, altimetry in a Kalman filter. Density altitude vs. pressure altitude vs. geometric altitude — why these differ.
- Airspeed measurement: pitot-static system, indicated vs. true vs. calibrated airspeed, the role of wind in navigation. Airspeed is not groundspeed.
- Magnetometer: measuring Earth's magnetic field for heading. Magnetic declination, inclination, hard and soft iron calibration. Why the magnetometer is unreliable near metal structures and at high latitudes.
- Radar altimeter: measuring height above terrain, not above sea level. Its role in terrain-following flight and landing.

**Chapter B.2.2: UAV Navigation System Design**
- The standard MAV navigation filter: IMU + GPS + barometer + magnetometer fusion. State definition, sensor models, the observability of attitude from accelerometers in quasi-static conditions.
- GPS-denied environments: indoor flight, urban canyons, under-canopy. The transition from GPS-aided to visual/inertial-only navigation.
- Fixed-wing vs. rotorcraft: different dynamics, different observability conditions, different sensor suite priorities
- Connecting to VIO (Track C): how the MAV context motivated the development of visual-inertial odometry

**Notebooks:** `B3_aerial_navigation_sensors.ipynb`, `B4_uav_filter_design.ipynb`

---

### Track B.3 — Marine Surface Navigation (~2 notebooks)

**Chapter B.3.1: Marine Navigation Sensors and Environment**
- Ship dynamics and why they complicate dead reckoning: currents, wind, wave-induced motion
- Radar as a navigation sensor: rotating radar for obstacle detection and landmark identification, ARPA (Automatic Radar Plotting Aid) for tracking contacts
- AIS (Automatic Identification System): other ships as collaborative beacons with known positions — a non-traditional aiding source
- ECDIS and electronic charts: how navigation charts are used as maps in the maritime context
- The difference between course over ground (COG) and heading: the effect of current on the navigation solution

**Chapter B.3.2: Marine Navigation Filter Design**
- Integrating GPS, heading, and speed log: the standard ship navigation architecture
- Using radar for relative positioning: extracting bearing/range to known landmarks (buoys, headlands) and incorporating them as observations
- Port approach navigation: the transition from open-ocean to harbor requires higher accuracy and different sensor emphasis
- Historical context: the role of dead reckoning and celestial navigation before GPS. How modern systems degraded gracefully — and what happens if GPS disappears.

**Notebooks:** `B5_marine_navigation.ipynb`, `B6_marine_filter_design.ipynb`

---

### Track B.4 — Ground Vehicle Navigation (~2 notebooks)

**Chapter B.4.1: Ground Vehicle Kinematics and Odometry**
- Wheel odometry as a velocity sensor: the encoder model, differential drive and Ackermann steering kinematics, odometry error model (scale factor, bearing error, slippage)
- ZUPT (Zero-Velocity UPdaTe): using the knowledge that the vehicle is stationary to reset velocity error and calibrate IMU bias. Extremely powerful for pedestrian and slow-vehicle navigation.
- The non-holonomic constraint: wheeled vehicles cannot move sideways. This kinematic constraint is a free measurement in the filter.
- Terrain effects: wheel slippage on loose terrain, vehicle pitch and roll affecting the accelerometer measurements, the limits of odometry off-road

**Chapter B.4.2: Map-Aided Ground Navigation**
- Map matching: using a road network as a prior to constrain the navigation solution to known paths. The observation model for map matching as a probabilistic constraint.
- HD maps for autonomous vehicles: lane-level maps as dense landmark fields, how LIDAR and camera-based features are matched to the map
- GNSS in urban environments: multipath, signal blockage by buildings, the urban canyon problem. Techniques to mitigate: 3D map-aided GNSS, shadow matching.
- The transition from GPS-primary to odometry-primary in tunnels, parking structures, and urban canyons

**Notebooks:** `B7_ground_vehicle_odometry.ipynb`, `B8_map_aided_navigation.ipynb`

---

## Track C — Modern SLAM and Visual-Inertial Navigation

*Target audience: roboticists and researchers who want to connect the Core course SLAM material to the state-of-the-art systems used in practice.*

---

### Chapter C.1: Factor Graphs and Modern SLAM Backends

**Learning objectives**
- Represent a SLAM problem as a factor graph
- Implement pose-graph SLAM using a factor graph library (GTSAM or g2o)
- Explain how loop closure detection and verification work in practice
- Connect the factor graph formulation to the EKF-SLAM and GraphSLAM taught in the Core

**Content**
- Factor graphs as a unifying framework: variables, factors, and sparse structure
- GTSAM/g2o as practical implementations: how to define variables, add factors, and solve
- Robust cost functions: the M-estimator and why it is needed for real-world loop closures that may be wrong
- Incremental smoothing: iSAM2 for online operation — updating only the affected portion of the graph when new measurements arrive
- The full SLAM pipeline: front-end (sensor processing, feature extraction, data association) and back-end (graph optimization) as a separation of concerns

**Notebook:** `C1_factor_graphs.ipynb`
- Build a pose-graph SLAM system using GTSAM. Include loop closure. Add incorrect loop closures and show the effect of robust cost functions vs. standard least squares.

---

### Chapter C.2: Visual Odometry

**Learning objectives**
- Implement monocular visual odometry from a sequence of images
- Explain the scale ambiguity in monocular VO and how stereo or IMU resolves it
- Identify the failure modes of feature-based VO
- Connect the VO output to a filter or factor graph backend

**Content**
- Feature-based VO pipeline: feature detection (ORB, SIFT, FAST), feature matching/tracking (optical flow), essential matrix estimation, pose recovery
- Stereo VO: how a stereo rig recovers metric scale, the stereo matching problem
- Direct methods: dense optical flow, LSD-SLAM, DSO — minimizing photometric error rather than feature reprojection error
- Semi-direct methods: SVO as a practical hybrid
- VO drift: why even good VO accumulates error over long trajectories, and why loop closure is essential
- Connecting VO to a SLAM backend: VO as a relative pose factor in a factor graph

**Notebook:** `C2_visual_odometry.ipynb`
- Implement feature-based monocular VO on a public dataset (KITTI or TUM). Visualize the estimated trajectory vs. ground truth. Show error growth and how it compares to GPS-aided navigation.

---

### Chapter C.3: Visual-Inertial Odometry

**Learning objectives**
- Explain why combining camera and IMU is more powerful than either alone
- Describe the IMU pre-integration approach and its computational advantages
- Identify the states estimated in a VIO system and their observability conditions
- Run an open-source VIO system on a benchmark dataset

**Content**
- The complementary nature of camera and IMU: camera provides orientation and structure, IMU provides gravity direction and scale
- IMU pre-integration: summarizing IMU measurements between keyframes into a single pre-integrated factor, decoupling the IMU integration from the optimization
- VIO state definition: position, velocity, attitude, IMU biases — and why all of these are observable in a well-designed VIO system
- Tightly-coupled VIO: MSCKF, VINS-Mono, OpenVINS. The architecture of each. Why tightly-coupled outperforms loosely-coupled for MAV navigation.
- Initialization: how does a VIO system start from rest? Estimating gravity direction, velocity, and biases from early motion.
- Failure modes: motion blur, texture-less environments, pure rotation (no translation means no triangulation)

**Notebook:** `C3_visual_inertial.ipynb`
- Run OpenVINS on the EuRoC MAV dataset. Visualize the factor graph. Instrument the output to show bias estimation over time. Compare VIO accuracy against GPS ground truth.

---

### Chapter C.4: LiDAR Odometry and Mapping

**Learning objectives**
- Explain the LiDAR scan matching problem and the ICP algorithm
- Implement a simplified LiDAR odometry pipeline in 2D
- Describe the LOAM architecture and why it achieves high accuracy
- Connect LiDAR odometry to the occupancy grid mapping from the Core course

**Content**
- The LiDAR scan as a point cloud: 2D vs. 3D lidar, the density and range characteristics of different sensors
- Iterative Closest Point (ICP): the scan matching algorithm, convergence conditions, sensitivity to initialization
- Normal Distributions Transform (NDT): representing a scan as a set of Gaussians, faster and more robust than ICP
- LOAM (LiDAR Odometry and Mapping): separating the high-frequency odometry estimation from the lower-frequency mapping update. Why this achieves accuracy close to VIO with a single sensor.
- LiDAR SLAM: integrating scan matching with loop closure detection in 3D (LeGO-LOAM, Cartographer, LIO-SAM)

**Notebook:** `C4_lidar_odometry.ipynb`
- Implement 2D lidar odometry using ICP. Test on a simulated or public dataset. Show drift accumulation. Add loop closure correction and show the improvement.

---

### Chapter C.5: Learned Navigation and Hybrid Approaches

**Learning objectives**
- Identify where learned components are genuinely useful in navigation pipelines
- Explain calibrated uncertainty output from neural networks and why it matters for filter integration
- Describe place recognition as a learned component for loop closure detection
- Critically evaluate claims about end-to-end learned navigation vs. hybrid approaches

**Content**
- What neural networks are good at in navigation: feature extraction, place recognition (image retrieval for loop closure), outlier rejection, learning sensor error models from data
- What classical estimation is still better at: interpretability, guaranteed behavior, operation outside the training distribution, sample efficiency
- Learned uncertainty: why a neural network that outputs a covariance matrix (or implicitly calibrated uncertainty) is more useful for filter integration than one that outputs only a point estimate
- End-to-end learned odometry: DeepVO, learning to estimate relative pose from image sequences. Performance comparison against classical VO.
- Hybrid approaches: using learned features or descriptors inside a classical SLAM pipeline. ORB-SLAM with learned features. NetVLAD for loop closure in GraphSLAM.

**Notebook:** `C5_learned_navigation.ipynb`
- Compare classical ORB features vs. learned SuperPoint features for VO on a challenging dataset segment (low texture, lighting change). Discuss implications for real-world deployment.

---

## Track D — Navigation System Design and Practice

*Target audience: engineers moving from "I can implement a filter" to "I can design and evaluate a navigation system." Connects the theoretical course content to engineering practice.*

---

### Chapter D.1: Sensor Selection and Error Budgeting

**Learning objectives**
- Read and interpret navigation sensor datasheets for filter design
- Construct a navigation error budget from component sensor specifications
- Select a sensor suite appropriate for a given application and performance requirement
- Identify the performance-limiting component in a multi-sensor navigation system

**Content**
- Navigation requirements: defining the problem before choosing sensors. Accuracy, update rate, operating environment, size/weight/power constraints, cost.
- Reading sensor datasheets for navigation: what specifications matter (noise density, bias instability, ARW, VRW) and what is marketing
- The error budget: allocating the total allowable navigation error across sensor components. Which sensor limits performance and why.
- Worked example: design a navigation system for a UAV operating indoors for 10 minutes with ±0.5 m position accuracy. Choose sensors, build the budget, identify the binding constraint.
- Cost-performance tradeoffs: consumer MEMS vs. tactical-grade IMU. When is the expensive sensor worth it?

**Notebook:** `D1_sensor_selection.ipynb`
- Reproduce the UAV navigation error budget example. Substitute different IMU grades and show the effect on navigation performance during a GPS outage. Parameterize over outage duration.

---

### Chapter D.2: Observability Analysis in Practice

**Learning objectives**
- Compute the observability Gramian for a linearized navigation system
- Identify unobservable state directions and their physical meaning
- Explain how motion excitation improves observability for IMU bias estimation
- Design a sensor suite that makes all navigation states observable

**Content**
- Observability review: the formal definition, the observability matrix for discrete linear systems
- Nonlinear observability: the extended observability analysis, practical indicators when states are not observable
- Navigation observability examples:
  - Yaw is not observable from accelerometers during stationary flight
  - IMU bias is not observable without external position reference
  - Velocity is not observable from GPS alone when stationary
  - Z-axis magnetometer heading is unreliable near the magnetic poles
- The role of motion: why a maneuvering vehicle provides better observability than a straight-line trajectory. Persistent excitation.
- Sensor redundancy and fault tolerance: how duplicate sensors affect observability and robustness

**Notebook:** `D2_observability.ipynb`
- Compute observability Gramian for an INS/GPS system. Remove sensors one at a time and show which states become unobservable. Simulate different trajectory shapes and show how maneuvers improve observability.

---

### Chapter D.3: Real-Time Implementation Considerations

**Learning objectives**
- Profile the computational cost of EKF, UKF, and particle filter as a function of state dimension and particle count
- Explain the numerical stability issues in covariance propagation and apply standard mitigations
- Describe the asynchronous measurement problem and implement a solution
- Identify the software architecture patterns used in deployed navigation systems

**Content**
- Computational cost analysis: EKF O(n²), UKF O(n³), PF O(Np × n). When do the constants matter?
- Numerical stability: covariance matrices can lose positive definiteness through floating-point accumulation. The Joseph form update, the square-root filter, and symmetry enforcement.
- Asynchronous measurements: real sensors run at different rates (IMU at 200 Hz, GPS at 1 Hz, camera at 30 Hz, lidar at 10 Hz). How to handle measurements that arrive out of order or are delayed.
- The predict-on-demand architecture: re-propagating IMU to the time of a late measurement
- Software patterns: the navigation filter as a software component, interface definitions, thread safety in real-time systems
- Fixed-point arithmetic: when floating-point is not available, how navigation filters have been implemented in constrained hardware

**Notebook:** `D3_realtime_implementation.ipynb`
- Profile EKF vs. UKF vs. PF (100/500/2000 particles) as a function of state dimension. Implement asynchronous measurement handling for a combined IMU/GPS/camera system. Demonstrate the Joseph form update for numerical stability.

---

### Chapter D.4: Evaluation Frameworks and Benchmarks

**Learning objectives**
- Apply standard navigation evaluation metrics: RMSE, ATE, RPE, CEP
- Use public benchmark datasets for algorithm evaluation
- Design a Monte Carlo evaluation framework for a navigation system
- Interpret benchmark results critically — what the numbers mean and what they hide

**Content**
- Navigation accuracy metrics:
  - RMSE: aggregate accuracy, sensitive to outliers
  - ATE (Absolute Trajectory Error): the standard SLAM/VO metric for global consistency
  - RPE (Relative Pose Error): local accuracy, how much error accumulates over a fixed window
  - CEP/2DRMS: the navigation system specification metrics used in practice
- Public benchmark datasets: KITTI (autonomous driving), EuRoC MAV (visual-inertial), TUM RGB-D (indoor), Victoria Park (outdoor SLAM), Oxford RobotCar (long-term autonomy). What each tests and what it does not.
- The Monte Carlo framework: why a single run is not enough for performance characterization. Designing parameter sweeps. Computing confidence intervals.
- What benchmarks hide: the dataset was collected under favorable conditions, the ground truth has its own errors, the algorithm may have been tuned to the benchmark, real-world performance may differ
- Failure mode testing: deliberately stress-testing navigation systems — high dynamics, sensor failures, GPS jamming, lighting changes. The tests that reveal real robustness.

**Notebook:** `D4_evaluation.ipynb`
- Implement ATE, RPE, and NEES metrics. Evaluate the Victoria Park SLAM result from the Core course using the full suite of metrics. Run a Monte Carlo parameter sweep over data association thresholds. Plot the tradeoff between accuracy and robustness.

---

## Appendix: Running Scenarios

*Three vehicles appear throughout the course as consistent implementation targets. Every major algorithm is demonstrated on at least one scenario. Students develop intuition across scales and domains.*

### Scenario 1: Ground Robot (2D)
A differential-drive ground robot navigating indoors. Wheel odometry + IMU + lidar + intermittent GPS.
Appears in: Pre-course (dead reckoning demo), Core Blocks 1–3, Track B.4, Track D

### Scenario 2: Quadrotor (3D)
The Nano+ quadrotor from the existing course projects. Camera (AprilTag observation model) + IMU. Real data available.
Appears in: Core P2 and P3, Track B.2, Track C.3

### Scenario 3: Marine/Aerospace Vehicle (global scale)
A simulated marine surface vessel or UAV operating at scales where Earth curvature matters. IMU + GPS + domain-specific sensors.
Appears in: Core Week 2 (coordinate frames), Track A (INS/GNSS), Track B.1–B.3

---

## Suggested Study Paths

### Path 1: Roboticist extending into aerospace/marine
Pre-Course (skim) → Core → Track A → Track B (choose domain) → Track D

### Path 2: Aerospace/Marine GNC engineer
Pre-Course Units 1–2 (review) → Core → Track C → Track D

### Path 3: SLAM researcher
Pre-Course (skim) → Core → Track C → Track D.4

### Path 4: Self-learner, complete course
Pre-Course (all) → Core → Track A → Track C → Track D

### Path 5: WPI RBE 595 student (semester course)
Core only, use Pre-Course as reference material and Post-Course tracks as further reading

---

## Relationship to Labbe's *Kalman and Bayesian Filters in Python*

Labbe's work is the closest analog to this course in the public domain and is referenced throughout.
This course differs in the following deliberate ways:

| Dimension | Labbe (KBFP) | This Course |
|---|---|---|
| Scope | Estimation theory with robotics examples | Navigation as a discipline (GNC framing, multi-domain) |
| Applications | Primarily tracking and localization | Navigation, SLAM, inertial systems, domain-specific |
| SLAM coverage | Limited | Central to the course (3+ weeks in core, full tracks post-course) |
| Aerospace/Marine | Not covered | Core Track A and B |
| Prerequisites assumed | Minimal | Undergraduate navigation (on-ramp provided) |
| Project structure | None (reference book) | Scaffolded projects with real datasets |

Labbe is recommended as a companion reference throughout, particularly for the estimation fundamentals (Pre-Course and Core Block 2). This course builds on that foundation rather than replacing it.
