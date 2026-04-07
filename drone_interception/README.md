# Counter-UAS: Autonomous Drone Interception via Reinforcement Learning

<p align="center">
  <strong>A $6,400 AI system that solves the $3,000,000 problem</strong><br>
  <em>Training a $300 interceptor drone to replace a $3M Patriot missile</em>
</p>

<p align="center">
  <code>PPO</code>&nbsp;&bull;&nbsp;<code>Domain Randomization</code>&nbsp;&bull;&nbsp;<code>Sim2Real</code>&nbsp;&bull;&nbsp;<code>Edge AI</code>&nbsp;&bull;&nbsp;<code>YOLOv8</code>
</p>

---

## Table of Contents

1. [Operational Context: The Modern Drone Threat](#operational-context-the-modern-drone-threat)
2. [The Problem: Asymmetric Cost](#the-problem-asymmetric-cost)
3. [Our Solution](#our-solution)
4. [System Architecture](#system-architecture)
5. [Environment Design](#environment-design)
6. [Reward Structure](#reward-structure)
7. [RL Algorithm: PPO](#rl-algorithm-ppo)
8. [Domain Randomization & Sim2Real](#domain-randomization--sim2real)
9. [Training & Results](#training--results)
10. [Live Demo: Command Center](#live-demo-command-center)
11. [Quick Start](#quick-start)
12. [Project Structure](#project-structure)
13. [Cost Analysis](#cost-analysis)
14. [Future Work](#future-work)
15. [Tech Stack](#tech-stack)
16. [References](#references)

---

## Operational Context: The Modern Drone Threat

Low-cost drone warfare has become the **defining military challenge of the 2020s**. Adversary UAS swarms now threaten forward-operating bases globally, and defenders are hemorrhaging money on legacy interceptors.

### The Escalation

| Date | Event | Impact |
|------|-------|--------|
| **Sep 2019** | Abqaiq-Khurais oil facility attack | 18 low-cost drones cause **$2B damage**, bypass Patriot batteries |
| **Jan 2024** | Tower 22, Jordan | **3 soldiers killed** by a $2,000 drone that mimicked a returning friendly drone's flight path |
| **Apr 2024** | Large-scale drone + missile barrage | 170 drones + 150 missiles; defenders intercept at a cost of **$1.35B in one night** |
| **2024-25** | Red Sea drone campaign | Navy fires **$2M SM-2 missiles** at $2K hobby-grade drones |
| **2025-26** | Mass-produced swarm attacks | Cheap loitering munitions target forward bases; **hundreds produced per month** |
| **Present** | Drone attrition crisis | Billions spent on drone defense; interceptor stockpiles depleting faster than manufacturers can produce |

### Threatened Forward-Operating Bases

| Base | Region | Threat Level |
|------|--------|-------------|
| Al Dhafra AB | Persian Gulf | **CRITICAL** |
| Camp Arifjan | Persian Gulf | **CRITICAL** |
| Al Udeid AB | Persian Gulf | HIGH |
| Al Asad AB | Middle East | **CRITICAL** |
| Camp Lemonnier | East Africa | HIGH |
| Prince Sultan AB | Middle East | HIGH |

> *"We cannot afford to defend against cheap drones with expensive missiles."*
> — Defense officials

---

## The Problem: Asymmetric Cost

Counter-drone defense costs are **1,000-10,000x** the cost of the threats they neutralize. This creates an unsustainable cost exchange ratio that favors the attacker:

| System | Cost per Intercept | Target Type | Autonomous | Reusable |
|--------|-------------------|-------------|-----------|---------|
| **Patriot PAC-3** | **$3,000,000** | Ballistic missiles, aircraft | No (human operator) | No |
| **SM-2 (Navy)** | **$2,100,000** | Aircraft, cruise missiles | No | No |
| **Iron Dome (Tamir)** | **$50,000** | Rockets, mortars | Semi | No |
| **COYOTE (Raytheon)** | **$80,000** | Small UAS | Semi | No |
| **Our System** | **$6,400 total / $300 per intercept** | Small UAS, hobby drones | **Yes (fully)** | **Yes** |

**The Cost Crisis:**
- Mass-produced loitering munitions: **$20,000** per drone
- Patriot PAC-3 interceptor: **$3,000,000** per intercept
- Daily swarm attacks: **15-40 drones** per wave
- Defender burn rate: **$45M-$120M per day** on drone defense

**Our interceptor costs less than the drone it's killing — the cost asymmetry is inverted.**

---

## Our Solution

A **4-stage fully autonomous** detection and interception pipeline with **zero human in the loop**:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  01 RF       │     │  02 YOLO     │     │  03 RL       │     │  04 KINETIC  │
│  DETECTION   │────>│  VISUAL ID   │────>│  POLICY      │────>│  INTERCEPT   │
│              │     │              │     │              │     │              │
│  Passive     │     │  PTZ camera  │     │  PPO neural  │     │  Pursuit     │
│  radio       │     │  slews to    │     │  net computes│     │  drone       │
│  sensor      │     │  bearing     │     │  pursuit     │     │  launches    │
│  Range: 80km │     │  YOLOv8      │     │  thrust      │     │  autonomous  │
│  $5,000      │     │  $1,000      │     │  <10ms       │     │  kinetic kill│
│              │     │              │     │  $100 Jetson  │     │  $300        │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

**Total system cost: $6,400** | **Per-intercept cost: $300** (reusable airframe) | **469x cheaper than Patriot**

---

## System Architecture

### Edge Deployment Stack

| Component | Cost | Function |
|-----------|------|----------|
| RF Sensor | $5,000 | Passive radio detection, 80km range, bearing estimation |
| PTZ Camera + YOLOv8 | $1,000 | Visual classification: UAV / bird / aircraft |
| NVIDIA Jetson Nano | $100 | Cheapest NVIDIA GPU board; runs PPO (<10ms) + YOLOv8 (~30ms) simultaneously. Ground-mounted, not on-drone. |
| Interceptor Drone | $300 | COTS quadcopter, kinetic intercept, reusable airframe |

### Software Stack

| Component | Technology |
|-----------|-----------|
| RL Training | `Stable-Baselines3` (PPO) |
| Environment | `Gymnasium` + NumPy physics / PyBullet |
| Sim2Real | `DomainRandomizationWrapper` (8 randomized params) |
| Visual Detection | `YOLOv8` |
| Edge Inference | `TensorRT` on Jetson Nano |
| Backend API | `FastAPI` + Python |
| Frontend Demo | `React` + `MapLibre GL` + `deck.gl` |
| Training Monitoring | `TensorBoard` |

### Architecture Diagram

```
                   ┌─────────────────────────────────────────────────────┐
                   │                PyBullet / NumPy Physics              │
                   │   [Interceptor]  <-->  [Target]  +  [Obstacles]     │
                   └───────────────┬─────────────────────────────────────┘
                                   │  obs (21-dim) / reward / done
                                   v
                   ┌─────────────────────────────────────────────────────┐
                   │          DroneInterceptionEnv (Gymnasium)            │
                   │   Observation -> MLP [256,256] -> Thrust Actions    │
                   │              + DomainRandomizationWrapper            │
                   └───────────────┬─────────────────────────────────────┘
                                   │
                                   v
                   ┌─────────────────────────────────────────────────────┐
                   │            Stable-Baselines3 (PPO)                  │
                   │   1M timesteps, 4 parallel envs, ~30 min training   │
                   └───────────────┬─────────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          v                        v                        v
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐
│  Evaluation      │  │  FastAPI Backend  │  │  Edge Deployment         │
│  evaluate.py     │  │  server.py        │  │  .zip -> Jetson Nano     │
│  compare_algo.py │  │  PPO inference    │  │  <10ms inference         │
│  visualize_3d.py │  │  geo mapping      │  │  100Hz control loop      │
└──────────────────┘  └────────┬─────────┘  └──────────────────────────┘
                               │
                               v
                   ┌─────────────────────────────────────────────────────┐
                   │  React Command Center (MapLibre + deck.gl)          │
                   │  Real-time visualization over regional map            │
                   │  Live PPO inference, domain randomization display    │
                   └─────────────────────────────────────────────────────┘
```

---

## Environment Design

### `DroneInterceptionEnv` — Custom Gymnasium Environment

A 3D pursuit-evasion environment where a low-cost interceptor drone (**RL agent**) must chase and capture an evading target drone. **Only the interceptor is controlled by the PPO policy** — the adversary drone uses a scripted evasion policy (see below).

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Arena size | 20m x 20m x 8m | Typical urban engagement zone |
| Capture distance | 1.0m | Real net-capture systems activate ~1m |
| Max steps | 500 | ~8.3s at 60Hz; timeout = mission failure |
| Drone mass | 1.0 kg | DJI Tello class with payload |
| Max thrust | 5.0 N | Max per-axis beyond hover |
| Drag coefficient | 0.3 | Linear approximation of air resistance |
| Obstacles | 5 static boxes | Urban clutter simulation |
| Physics timestep | 1/60s | Matches real flight controllers |

### Observation Space (21-dimensional)

```
[0:3]   Interceptor position (x, y, z)
[3:6]   Interceptor velocity (vx, vy, vz)
[6:9]   Target position (x, y, z)
[9:12]  Target velocity (vx, vy, vz)
[12:15] Relative vector (target - interceptor)
[15]    Euclidean distance to target
[16:21] Obstacle proximity (5 raycasts: forward, back, left, right, down)
```

Real-world sensor mapping: IMU + barometer -> own state; camera tracking -> target state; ultrasonic/IR -> obstacle proximity.

### Action Space (3-dimensional continuous)

The PPO agent outputs a **3D thrust vector** every timestep (1/60s):

```
Action = [thrust_x, thrust_y, thrust_z]    each in [-1, 1]
           │          │          │
           │          │          └── push up/down
           │          └──────────── push forward/backward
           └─────────────────────── push left/right
```

Each value is scaled by `MAX_FORCE` (5.0 N). Gravity compensation is auto-added to the z-axis so the drone hovers at zero thrust. In real deployment, a flight controller translates these force commands into individual rotor speeds.

### Target Evasion Policy (4 Layers)

The adversary drone is **not an RL agent** — it uses a **scripted multi-layered evasive policy** that produces challenging, unpredictable behavior:

1. **Figure-8 base pattern** — sinusoidal motion in x/y/z with phase offsets
2. **Reactive evasion** — when interceptor is within 4m, target accelerates away from the pursuit vector
3. **Stochastic noise** — Gaussian perturbations (sigma=0.3) on velocity
4. **Boundary clamping** — keeps target inside arena with minimum altitude

### Physics Backend

- **PyBullet** (if installed): Full collision detection, raycasting, GUI visualization
- **NumPy fallback** (if PyBullet unavailable): Pure-Python physics with AABB collision detection, **10x faster** for headless training. Ensures compatibility with Python 3.13+ where PyBullet wheels may not exist.

---

## Reward Structure

The reward function is the most critical design element. It encodes our operational objective: **intercept fast, avoid obstacles, minimize energy**.

### Positive Rewards

| Component | Signal | Purpose |
|-----------|--------|---------|
| **Progress reward** | `+15 * delta_distance` per step | Main gradient signal; rewards closing distance. Zero for orbiting. |
| **Interception bonus** | `+500 + (max_steps - steps) * 0.5` | Massive terminal reward. Faster kill = bigger bonus. Dominates all per-step rewards. |
| **Proximity bonus** | `+1/(d + 0.3)` when `d < 3m` | Exponential reward within 3m. ~0.5 at 3m, ~3.0 at 0.5m. Capped to prevent orbit-farming. |

### Negative Rewards

| Component | Signal | Purpose |
|-----------|--------|---------|
| **Time penalty** | `-0.5` per step | **KEY anti-orbiting mechanism.** 500-step orbit = -250. Quick intercept clearly optimal. |
| **Collision penalty** | `-75` | Crash = drone replacement cost. Combined with continuous obstacle proximity warning. |
| **Energy penalty** | `-0.02 * \|\|action\|\|^2` | Penalizes excessive thrust. Energy = battery = mission cost. |
| **Obstacle warning** | `-3.0 * proximity` when within 30% ray range | Continuous negative signal near obstacles. Teaches avoidance before collision. |
| **Out of bounds** | `-75` | Leaving arena = mission failure (lost drone). |

### Complete Reward Equation

```
R(s,a) = 15*delta_d + [500 + speed_bonus]_intercept + proximity_bonus
         - 0.5 - 0.02*||a||^2 - 75_collision - 75_OOB - obstacle_warning
```

### Design Constraints

- **Interception bonus (500) MUST dominate cumulative per-step rewards.** Max per-step proximity ~3.0 over 500 steps = 1500, but time penalty (-0.5 * 500 = -250) makes early interception clearly optimal.
- **Energy penalty kept small** so it doesn't discourage aggressive pursuit — just prevents wasteful thrashing.

### Cost-Aware Reward Variant

`core/cost_aware_reward.py` provides an alternative formulation that maps reward components to **approximate real-world dollar costs** ($300 hardware, $5 battery/mission, $0.50/step opportunity cost, $50K threat value). Trains policy to optimize **$/interception** directly.

---

## RL Algorithm: PPO

### PPO (Proximal Policy Optimization)

```bash
python -m training.train_ppo --timesteps 1000000 --domain-rand
```

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Learning rate | 3e-4 | Standard for continuous control |
| Rollout steps | 2048 per env | Full episodes in each batch |
| Batch size | 64 | Mini-batch for SGD |
| Epochs per rollout | 10 | PPO clipping prevents overshoot |
| Discount (gamma) | 0.99 | High: interception bonus propagates backward |
| GAE lambda | 0.95 | Bias-variance balance |
| Clip range | 0.2 | Conservative policy updates |
| Entropy coeff | 0.01 | Prevents premature convergence |
| Network | MLP [256, 256] | Runs on Jetson Nano at <10ms |
| Parallel envs | 4 | More diverse rollouts per update |

**Why PPO:** The clipped objective provides stable training with our shaped reward that has sharp discontinuities (+500 for interception, -75 for collision). PPO's conservative policy updates prevent catastrophic forgetting and ensure monotonic improvement.

---

## Domain Randomization & Sim2Real

Training in simulation is cheap and safe, but the real world doesn't follow simulator assumptions. Our key strategy: **domain randomization** — train across randomized physics so the real world is "just another sample."

### Randomized Parameters (per episode)

| Parameter | Range | Real-World Justification |
|-----------|-------|------------------------|
| **Drone mass** | 0.7 – 1.5 kg | Manufacturing tolerance, payload variation |
| **Max thrust** | 3.5 – 7.0 N | Motor degradation, battery voltage drop (12.6V → 10.5V) |
| **Drag coefficient** | 0.1 – 0.6 | Wind (0-4 m/s), altitude-dependent air density |
| **Evader speed** | 1.0 – 3.5 m/s | DJI Mini (~1 m/s) to racing drones (~10 m/s) |
| **Obstacles** | 2 – 8 | Open field to dense urban |
| **Sensor noise** | sigma 0 – 0.05 | IMU drift, barometer error, optical flow noise |
| **Action delay** | 0 – 3 steps | ESC response (1-5ms), flight controller lag, Jetson compute |
| **Gravity** | 9.75 – 9.85 m/s^2 | Altitude/latitude variation |

### Sim2Real Challenges & Mitigations

| Challenge | Description | Mitigation |
|-----------|-------------|-----------|
| Unmodeled aerodynamics | Rotor wash, ground effect, vortex ring state | Mass/thrust/drag randomization covers 2x range |
| Sensor latency & noise | GPS 100ms lag, IMU drift, dropped frames | Observation noise + action delay randomization |
| Wind & turbulence | Spatially varying, gusty, building effects | Drag coeff randomized 0.1-0.6 |
| Target behavior | Real drones: GPS waypoints, swarm coordination | Scripted multi-layer evasion + speed randomization |

> *"If you train your policy across a wide range of simulated conditions, the real world becomes just another sample from that distribution."*
> — Tobin et al., 2017 (OpenAI)

See [docs/sim2real_analysis.md](docs/sim2real_analysis.md) for the full 1800-word analysis.

---

## Training & Results

### Pre-trained Model

A trained PPO model is included at `models/ppo_interceptor.zip` (1.7 MB):

| Metric | Value |
|--------|-------|
| Average reward | **582.7** |
| Average steps to intercept | **132** (of 500 max) |
| Model size | **1.7 MB** |
| Training steps | 1,000,000 |
| Training time | ~30 min (laptop CPU, no GPU) |
| Inference time | **<10ms** (Jetson Nano) |


---

## Live Demo: Command Center

A real-time React-based command center that visualizes PPO model inference over a **regional map** with actual geographic coordinates.

### Backend (FastAPI + PPO)

```bash
cd drone_interception
python -m uvicorn backend.server:app --reload --port 8000
```

- Runs actual PPO model inference per request
- Maps arena coordinates to adversary origin → defender base geographic corridor
- Returns adversary zig-zag flight path + interceptor Bezier pursuit trajectory
- Domain randomization parameters included in response
- Endpoints: `POST /api/scenario` (baseName, seed, drEnabled)
- Supported bases: Al Dhafra AB (UAE), Camp Arifjan (Kuwait), Prince Sultan AB (KSA)

### Frontend (React + MapLibre GL)

```bash
cd react-demo
npm install
npm run dev    # http://localhost:3000
```

**Components:**
| Component | File | Description |
|-----------|------|-------------|
| `MapView` | `src/components/MapView.jsx` | MapLibre GL map with animated drone flight paths |
| `Sidebar` | `src/components/Sidebar.jsx` | Base selection, seed, speed, domain randomization toggle |
| `Telemetry` | `src/components/Telemetry.jsx` | Real-time distance, altitude, speed readout |
| `Pipeline` | `src/components/Pipeline.jsx` | 4-stage detection pipeline status (RF → YOLO → RL → Intercept) |
| `YoloPanel` | `src/components/YoloPanel.jsx` | Simulated YOLO detection visualization |
| `DomainRand` | `src/components/DomainRand.jsx` | Live display of randomized physics parameters |
| `DetectionLog` | `src/components/DetectionLog.jsx` | Scrolling event log of detection phases |

---

## Quick Start

### Prerequisites

- Python 3.10+ (3.13 compatible — falls back to NumPy physics if PyBullet unavailable)
- Node.js 18+ (for React demo only)

### 1. Install Python Dependencies

```bash
cd drone_interception
pip install -r requirements.txt
```

**requirements.txt:**
```
pybullet>=3.2.5
gymnasium>=0.29.0
stable-baselines3>=2.1.0
numpy>=1.24.0
matplotlib>=3.7.0
tensorboard>=2.14.0
plotly>=5.17.0
```

### 2. Verify Environment

```bash
python -m core.drone_env
python -m core.domain_randomization
```

### 3. Train PPO (~30 min, no GPU needed)

```bash
python -m training.train_ppo --timesteps 1000000 --domain-rand --seed 42
```

### 4. Evaluate

```bash
# Evaluate model (100 episodes)
python -m evaluation.evaluate --model models/ppo_interceptor.zip --episodes 100

# 3D flight path visualization
python -m evaluation.visualize_3d --episodes 5
```

### 5. Launch Live Demo

```bash
# Terminal 1: Backend
python -m uvicorn backend.server:app --reload --port 8000

# Terminal 2: Frontend
cd react-demo && npm install && npm run dev
```

### 6. Monitor Training

```bash
tensorboard --logdir logs/
```

---

## Project Structure

```
drone_interception/
│
├── core/                               # Environment & reward design
│   ├── __init__.py                     # Exports DroneInterceptionEnv, DomainRandomizationWrapper
│   ├── drone_env.py                    # Custom Gymnasium env (43KB, 996 lines)
│   │                                   #   - PyBullet or NumPy physics backend
│   │                                   #   - 21-dim observation, 3-dim continuous action
│   │                                   #   - 8-component shaped reward function
│   │                                   #   - 4-layer scripted evader (figure-8 + reactive + noise)
│   │                                   #   - 5-ray obstacle proximity sensing
│   ├── cost_aware_reward.py            # Dollar-cost reward mapping ($300 hardware model)
│   └── domain_randomization.py         # Gymnasium Wrapper: 8 randomized physics parameters
│
├── training/                           # Algorithm training scripts
│   ├── __init__.py
│   ├── train_ppo.py                    # PPO: 4 parallel envs, 2048 rollout, clip=0.2
│   └── callbacks.py                    # InterceptionTrackerCallback: per-episode metrics
│                                       #   - Tracks intercept/collision/timeout/OOB rates
│                                       #   - Estimates $/interception
│                                       #   - Saves to JSON for analysis
│
├── evaluation/                         # Model evaluation & visualization
│   ├── __init__.py
│   ├── evaluate.py                     # Run N episodes, compute all metrics, save JSON
│   ├── compare_algorithms.py           # Algorithm comparison tables/plots
│   └── visualize_3d.py                 # Publication-quality 3D flight path trajectories
│
├── models/                             # Trained models
│   ├── ppo_interceptor.zip             # Pre-trained PPO model (1.7MB, 1M steps)
│   ├── model_metadata.json             # Training metadata (reward, steps, env config)
│   └── save_model.py                   # Model save utility with metadata
│
├── backend/                            # FastAPI backend for live demo
│   └── server.py                       # PPO inference + geographic coordinate mapping
│                                       #   - Adversary origin -> defender base corridors
│                                       #   - Bezier pursuit trajectories
│                                       #   - Domain randomization params in response
│
├── react-demo/                         # React Command Center frontend
│   ├── package.json                    # React 18, MapLibre GL, deck.gl, Vite
│   ├── src/
│   │   ├── App.jsx                     # Main app: fetches from FastAPI backend
│   │   ├── scenario.js                 # Fallback scenario generation (JS)
│   │   ├── styles.css                  # Military-style dark theme
│   │   └── components/
│   │       ├── MapView.jsx             # MapLibre GL map with animated drone paths
│   │       ├── Sidebar.jsx             # Controls: base, seed, speed, DR toggle
│   │       ├── Telemetry.jsx           # Real-time distance/altitude/speed
│   │       ├── Pipeline.jsx            # RF -> YOLO -> RL -> Intercept status
│   │       ├── YoloPanel.jsx           # Simulated YOLO detection display
│   │       ├── DomainRand.jsx          # Live DR parameter display
│   │       └── DetectionLog.jsx        # Scrolling event log
│   └── vite.config.js
│
├── demo/                               # Terminal-based demo
│   └── command_center.py               # ASCII command center visualization
│
├── presentation/                       # Slide deck
│   └── slides.html                     # 17-slide HTML presentation (arrow keys to navigate)
│
├── docs/                               # Technical documentation
│   ├── sim2real_analysis.md            # 1800-word sim2real transfer analysis
│   └── cost_analysis.md                # Detailed cost-effectiveness breakdown
│
├── results/                            # Evaluation outputs
│   ├── ppo_metrics.json                # Training metrics (per-interval)
│   └── ppo_eval.json                   # Evaluation results (100 episodes)
│
├── logs/                               # TensorBoard logs
│   └── ppo/                            # PPO training runs
│
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## Cost Analysis

### System Bill of Materials

**Fixed costs (one-time infrastructure):**

| Component | Cost | Notes |
|-----------|------|-------|
| RF Sensor (passive radio) | $5,000 | Permanent ground installation; detects control signals at 80km |
| PTZ Camera + YOLOv8 | $1,000 | Permanent ground mount; visual classification |
| NVIDIA Jetson Nano | $100 | Permanent ground station compute; runs PPO + YOLO |
| **Infrastructure Total** | **$6,100** | **Buy once, use indefinitely** |

**Variable costs (per intercept):**

| Component | Cost | Notes |
|-----------|------|-------|
| Interceptor Drone | $300 | COTS quadcopter (frame, motors, battery, GPS). Replaced only if destroyed. |
| Battery recharge | ~$5 | If drone survives and is recovered |
| **Per-Intercept** | **$300** (worst case) | **$5 if drone is recovered and reused** |

**Total first intercept: $6,400** — every subsequent intercept: **$300 or less**. The more you use it, the cheaper it gets.

### Cost Comparison

| System | $/Intercept | Ratio vs Ours |
|--------|------------|---------------|
| Patriot PAC-3 | $3,000,000 | **10,000x** more expensive |
| SM-2 Interceptor | $2,100,000 | 7,000x more expensive |
| Stinger | $120,000 | 400x more expensive |
| Coyote (Raytheon) | $80,000 | 267x more expensive |
| **Our System** | **$300** | **Baseline** |

**Key finding:** Our system is cost-competitive with the Coyote (cheapest kinetic interceptor) at just **0.3% success rate**. At a realistic 70-85% success rate, it's **200-300x cheaper**.

See [docs/cost_analysis.md](docs/cost_analysis.md) for the full breakdown.

---

## Future Work

| Direction | Description |
|-----------|-------------|
| **Multi-Agent Swarm** | Extend PPO to MAPPO for coordinated interception of drone swarms |
| **ONNX Edge Deploy** | Export to ONNX + TensorRT for <5ms inference at 100Hz on Jetson |
| **Real Hardware** | Sim2real transfer to Crazyflie 2.1 micro-drones |
| **Curriculum Learning** | Progressive difficulty: stationary → linear → figure-8 → adversarial RL evader |
| **Sensor Fusion** | RF bearing + YOLO bounding box + LIDAR depth → unified RL observation |
| **Mesh Network** | Multiple counter-UAS nodes share detections with handoff |
| **Vision-Based Policy** | Replace state vector with camera images; CNN + RL pipeline |
| **Adversarial Self-Play** | Train evader and interceptor against each other for emergent strategies |
| **Safety Constraints** | Constrained MDP for no-fly zones, altitude limits, return-to-home |

---

## Key Design Decisions

1. **Why PPO**: Clipped objective provides stable training with our shaped reward that has sharp discontinuities (+500 interception, -75 collision). Conservative policy updates ensure monotonic improvement.

2. **Why shaped reward**: Sparse reward (+1 for interception only) would require millions of timesteps. Our 8-component shaped reward provides continuous gradient: progress, proximity, energy, obstacles — while ensuring the interception bonus dominates to prevent orbiting.

3. **Why scripted evader**: Training against a fixed multi-layer scripted policy is simpler and more reproducible than self-play. The figure-8 + reactive evasion + noise combination is challenging enough to produce robust pursuit policies.

4. **Why MLP (not CNN)**: 21-dim state vector allows MLP [256,256] policies that run at 100Hz on edge hardware (1.7MB model). Vision-based policies need CNNs and more compute.

5. **Why cost-aware reward**: Energy penalty on thrust trains efficiency, not just speed. Less battery = cheaper real-world missions. Direct $/interception optimization.

6. **Why NumPy fallback**: PyBullet doesn't always compile on Python 3.13. Pure-NumPy physics is 10x faster for headless training and ensures the project runs everywhere.

---

## Tech Stack

| Category | Technology | Version |
|----------|-----------|---------|
| Language | Python | 3.10+ |
| Physics | PyBullet / NumPy | >=3.2.5 |
| RL Framework | Stable-Baselines3 | >=2.1.0 |
| Environment | Gymnasium | >=0.29.0 |
| Frontend | React + MapLibre GL + deck.gl | 18.3 / 5.22 / 9.0 |
| Backend | FastAPI | latest |
| Build Tool | Vite | 5.4 |
| Visualization | Matplotlib + Plotly | >=3.7 / >=5.17 |
| Monitoring | TensorBoard | >=2.14.0 |

---

## References

1. Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." *arXiv:1707.06347*.
2. Tobin, J., et al. (2017). "Domain Randomization for Transferring DNNs from Simulation to the Real World." *IROS 2017*.
3. OpenAI et al. (2019). "Solving Rubik's Cube with a Robot Hand." *arXiv:1910.07113*.
4. Baker, B., et al. (2020). "Emergent Tool Use From Multi-Agent Autocurricula." *ICLR 2020*.
5. Kaufmann, E., et al. (2023). "Champion-level drone racing using deep reinforcement learning." *Nature*.
6. Loquercio, A., et al. (2021). "Learning High-Speed Flight in the Wild." *Science Robotics*.
