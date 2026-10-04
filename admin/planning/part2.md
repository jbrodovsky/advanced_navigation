A strong Part 2 pitch would be:

**Integrated Navigation for Autonomous Field Robots**

The clean arc is not “more filters,” but “how real autonomous aircraft, surface vessels, UUVs, and ground robots keep navigating when sensors are asynchronous, biased, intermittent, delayed, and occasionally lying.”

Part 1 gets students from EKF/UKF/PF through loose strapdown INS. Part 2 should build toward production-style navigation architectures.

**Possible Course Arc**

1. **Navigation System Architecture**
   - Multi-rate IMU propagation
   - Lower-rate GNSS, vision, magnetometer, barometer, DVL, acoustic, wheel, or radar updates
   - Buffering, timestamp alignment, latency, out-of-sequence measurements
   - Nominal state vs error state vs factor graph formulations

2. **Practical Strapdown INS Beyond the Textbook**
   - IMU preintegration
   - Lever arms
   - Misalignment
   - Sensor calibration states
   - Coning/sculling awareness
   - Real-time health monitoring and innovation gating

3. **Tightly Coupled INS/GNSS**
   - Pseudorange/Doppler updates instead of GNSS position fixes
   - Receiver clock states
   - Satellite geometry and observability
   - Degraded GNSS, partial satellite visibility, spoofing/jamming awareness
   - Compare loose vs tight coupling on the same trajectory

4. **Factor Graphs and Smoothing**
   - Batch least squares
   - Pose graphs
   - IMU factors
   - GNSS factors
   - Marginalization and fixed-lag smoothing
   - Where factor graphs beat recursive filters, and where they are overkill

5. **Rao-Blackwellized Particle Filters / FastSLAM**
   - Why RBPF works: sample robot trajectory, analytically estimate landmarks
   - FastSLAM 1.0 vs 2.0
   - Data association as the actual hard part
   - Particle depletion, resampling, landmark covariance management
   - Good bridge from Part 1 PFs into real SLAM

6. **Advanced SLAM for Field Robotics**
   - Loop closure
   - Scan matching / ICP / NDT
   - Visual-inertial odometry concepts
   - LiDAR-inertial odometry concepts
   - Pose graph optimization
   - Robust costs and bad loop closures

7. **Maritime / Aerospace Drone Extensions**
   - UUV navigation with DVL + depth + acoustic fixes
   - USV navigation with GNSS outages and compass issues
   - UAV navigation with vision/GNSS/IMU/baro
   - High-dynamic aerospace cases: IMU quality, timing, and observability matter more than filter branding

8. **Assurance, Failure, and Deployment**
   - Consistency testing
   - NEES/NIS
   - Fault detection and exclusion
   - Replay logs
   - Tuning workflows
   - “What would make this fail in the field?”

**Project Spine**

I would anchor the course around one cumulative project, not many disconnected mini-labs:

**Build a field-robot navigation stack that supports asynchronous sensor fusion and degraded aiding.**

Possible staged deliverables:

1. **Project 1: Multi-rate ESKF architecture**
   - IMU propagation at high rate
   - Delayed GNSS/vision/baro/DVL-style updates
   - Logging and innovation checks

2. **Project 2: Tight INS/GNSS or fixed-lag smoother**
   - Students choose recursive tight coupling or graph-based smoothing
   - Compare against loose coupling from Part 1

3. **Project 3: SLAM / RBPF module**
   - Implement FastSLAM or pose-graph SLAM on a robotics dataset
   - Focus grading on data association, consistency, and evaluation, not just plots

4. **Final integration report**
   - Compare architectures under outages, bias growth, timing errors, and bad measurements

**My honest take**

I would not try to deeply cover RBPF, advanced SLAM, tightly coupled GNSS, visual-inertial odometry, LiDAR-inertial odometry, and maritime acoustic navigation all as implementation-heavy topics in one semester. That becomes a survey plus three impossible projects.

Best shape: make **multi-rate integrated navigation** the backbone, then use **tight INS/GNSS** and **RBPF/SLAM** as the two major advanced implementations. Treat VIO/LIO/acoustic/UUV material as applied case studies unless you want a very heavy course.

A good pitch title/subtitle could be:

**Advanced Integrated Navigation: Multi-Sensor Fusion, SLAM, and Resilient Autonomy**

That connects naturally to robotics, aerospace, and maritime drones without becoming just “Kalman Filters II.”