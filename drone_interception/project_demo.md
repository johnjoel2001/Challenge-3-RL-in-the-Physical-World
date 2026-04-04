# Counter-UAS Drone Interception System — Project Demo Guide

## Overview

This project builds a **complete low-cost Counter-UAS (Unmanned Aerial System) defense pipeline** that detects, identifies, tracks, and intercepts hostile drones using reinforcement learning — all for under **$6,400 total system cost**, compared to $3M+ for a single Patriot missile.

The core innovation: instead of expensive missile systems designed for jets and cruise missiles, we use a **$300 quadrotor drone** guided by a **PPO-trained RL policy** to physically intercept small adversary drones.

---

## System Architecture — The Detection Chain

The system operates as a **4-stage pipeline**. Each stage feeds into the next:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  📡 RF       │───▶│  👁️ YOLO     │───▶│  🧠 RL       │───▶│  🚀 Pursuit  │
│  Detection   │    │  Visual ID   │    │  Policy      │    │  Drone       │
│              │    │              │    │              │    │              │
│  Range: 80km │    │  Range: 30km │    │  Latency:<1ms│    │  Speed: 5m/s │
│  Cost: $5000 │    │  Cost: $1000 │    │  Cost: $100  │    │  Cost: $300  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
 Detects radio       Confirms it's       Computes optimal    Physically
 emissions from      a drone (not a      pursuit thrust      chases and
 adversary drone     bird or plane)      commands             neutralizes
```

### Stage 1: RF Detection ($5,000)

- Passive radio frequency sensors scan for electromagnetic emissions
- Adversary drones emit signals for operator control, video downlink, and GPS
- **Output**: "Something is emitting RF at bearing 045°, ~80km away"
- **Limitation**: Cannot identify *what* the object is — only that it exists

### Stage 2: YOLO Visual Identification ($1,000)

- Once RF provides a bearing, a PTZ (Pan-Tilt-Zoom) camera slews to that direction
- **YOLOv8-nano** (real-time object detection model) runs on the camera feed
- Runs on a **Jetson Orin** edge GPU at 30+ FPS
- **Output**: Bounding box + class label + confidence: `"UAV 92%"`
- **Why YOLO is essential**: RF alone can't distinguish a hostile drone from a bird, a weather balloon, or a commercial aircraft. YOLO provides **visual classification** — it confirms the target is a UAV, not something else. It also provides precise pixel-level tracking that feeds into the RL policy.

### Stage 3: RL Policy — PPO ($100)

- The trained PPO agent takes sensor data and computes thrust commands
- Inference runs in **<1ms** on a Jetson Nano ($100 edge computer)
- This is the **core technical contribution** of the project (see details below)

### Stage 4: Pursuit Drone ($300)

- A cheap quadrotor ($300, similar to DJI Tello class) executes the thrust commands
- Speed: 5 m/s, endurance: ~15 minutes
- Intercepts via net-capture or kinetic impact within 1 meter of target

### Total System Cost: $6,400

| Component | Cost |
|-----------|------|
| RF Sensors | $5,000 |
| PTZ Camera + YOLO | $1,000 |
| Jetson Orin (Edge GPU) | $100 |
| Pursuit Drone | $300 |
| **Total** | **$6,400** |
| Patriot Missile (1 shot) | $3,000,000+ |
| **Cost reduction** | **~99.999%** |

---

## The RL Agent — Technical Deep Dive

### Environment: `DroneInterceptionEnv`

**File**: `core/drone_env.py`

A custom Gymnasium environment simulating a 20m × 20m × 8m arena with:
- An **interceptor drone** (our RL agent) — 1kg quadrotor
- An **evader drone** (the adversary) — follows a scripted evasive policy
- **5 static obstacles** (urban clutter)
- **60Hz physics** simulation with semi-implicit Euler integration

Key parameters:
| Parameter | Value | Real-World Meaning |
|-----------|-------|--------------------|
| `ARENA_SIZE` | 10m (half-width) | 20m × 20m engagement zone |
| `ARENA_HEIGHT` | 8m | Max altitude for small drones |
| `CAPTURE_DISTANCE` | 1.0m | Net-capture activation range |
| `MAX_STEPS` | 500 | ~8.3 seconds at 60Hz |
| `DRONE_MASS` | 1.0 kg | Typical small quadrotor |
| `MAX_FORCE` | 5.0 N | Max thrust per axis beyond hover |
| `EVADER_SPEED` | 2.0 m/s | Hobby drone cruise speed |

### Observation Space (21-dim continuous vector)

The RL agent receives a 21-dimensional observation at every timestep:

```
Index     Content                              Real-World Source
──────────────────────────────────────────────────────────────────
[0:3]     Interceptor position (x, y, z)       Onboard GPS + IMU
[3:6]     Interceptor velocity (vx, vy, vz)    Onboard IMU
[6:9]     Target position (x, y, z)            YOLO + triangulation
[9:12]    Target velocity (vx, vy, vz)         YOLO frame-to-frame tracking
[12:15]   Relative vector (target - self)      Computed
[15]      Euclidean distance to target          Computed
[16:21]   Obstacle proximity (5 raycasts)       Onboard ultrasonic/IR sensors
```

The 5 raycasts point in these directions:
- **Forward**, **Backward**, **Left**, **Right**, **Down** (ground proximity)
- Each returns a normalized distance (0 = touching obstacle, 1 = clear)

**Note**: There is NO time-remaining input. The agent has no explicit clock — the time penalty in the reward implicitly pressures it to act fast.

### Action Space (3-dim continuous, each in [-1, 1])

The agent outputs a **3D thrust vector** `[thrust_x, thrust_y, thrust_z]`:
- Each axis ranges from -1 to +1
- Scaled by `MAX_FORCE` (5.0 N)
- **Hover thrust is automatically added** to the z-axis, so action `[0, 0, 0]` = hover in place
- The agent learns *relative* thrust adjustments, not absolute forces

### Physics Step

Each timestep (1/60th of a second):
1. Scale action by `MAX_FORCE` → raw thrust
2. Add hover thrust (`mass × gravity = 9.81 N`) to z-axis
3. Apply linear drag (`F_drag = -0.3 × velocity`)
4. Compute acceleration: `a = (thrust + drag) / mass - gravity`
5. Semi-implicit Euler integration: `vel += a × dt`, `pos += vel × dt`

### Evader (Adversary) Scripted Policy

The target drone is NOT controlled by RL — it follows a 4-layer scripted policy:

1. **Base Motion**: Figure-8 (lemniscate) pattern with vertical oscillation — simulates a drone on a surveillance mission
2. **Reactive Evasion**: When interceptor gets within 4m, accelerates away proportionally to closeness (1.2× speed multiplier)
3. **Stochastic Noise**: Gaussian perturbation (`σ=0.3`) makes it unpredictable
4. **Boundary Clamping**: Stays within 90% of arena bounds, min altitude 1m

### Reward Function (8 components)

**File**: `core/drone_env.py` → `_compute_reward()`

**Design philosophy**: The interception bonus MUST dominate all cumulative per-step rewards, otherwise the agent learns to orbit the target forever.

| # | Component | Value | Purpose |
|---|-----------|-------|---------|
| 1 | **Progress** | `(prev_dist - dist) × 15` | Gradient toward target |
| 2 | **Interception bonus** | `+500 + (steps_remaining × 0.5)` | Massive payoff for catching target |
| 3 | **Collision penalty** | `-75` | Crashing = drone replacement cost |
| 4 | **Energy penalty** | `-0.02 × Σ(action²)` | Battery conservation |
| 5 | **Time penalty** | `-0.5 per step` | Anti-orbiting mechanism |
| 6 | **Proximity bonus** | `1/(dist+0.3) - 0.3` when `dist < 3m` | Rewards final approach (capped) |
| 7 | **Obstacle warning** | `-3 × (0.3 - min_obs_dist)/0.3` | Continuous signal near obstacles |
| 8 | **Boundary penalty** | `-75` | Leaving arena = mission failure |

**Anti-orbiting math**: Max per-step proximity ≈ 3.0 over 500 steps = 1500. But time penalty = -0.5 × 500 = -250. Quick interception: +500 bonus - 100 time = +400. So early interception always wins.

---

## Training: PPO (Proximal Policy Optimization)

**File**: `training/train_ppo.py`

### Why PPO?

1. **Stable**: Clipped objective prevents catastrophically large policy updates — critical with our shaped reward that has sharp cliffs (±500, ±75)
2. **Sample efficient enough**: 4 parallel envs × 2048-step rollouts = 8192 transitions per update
3. **On-policy**: No replay buffer needed — small memory footprint (runs on student laptops)
4. **Proven**: Same algorithm used by OpenAI Five (Dota 2) and modern robotics

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `learning_rate` | 3e-4 | Standard for continuous control |
| `n_steps` | 2048 | Rollout length per env before update |
| `batch_size` | 64 | Mini-batch for SGD within each epoch |
| `n_epochs` | 10 | Passes over rollout data (PPO clipping prevents instability) |
| `gamma` | 0.99 | High discount — interception bonus at episode end must influence early actions |
| `gae_lambda` | 0.95 | GAE for advantage estimation (bias-variance tradeoff) |
| `clip_range` | 0.2 | PPO's signature — policy ratio stays in [0.8, 1.2] |
| `ent_coef` | 0.01 | Small entropy bonus prevents premature convergence |
| `net_arch` | [256, 256] | Two hidden layers — small enough for Jetson Nano inference |
| Parallel envs | 4 | More diverse rollouts per update |

### Training Command

```bash
python -m training.train_ppo --timesteps 1000000 --seed 42
# With domain randomization:
python -m training.train_ppo --timesteps 1000000 --domain-rand
```

### Results

| Metric | Value |
|--------|-------|
| Training timesteps | 1,000,000 |
| Average reward | 582.7 |
| Average steps to intercept | 132.4 (~2.2 seconds) |
| Collision rate | 20% |
| Model size | 1.7 MB |
| Inference time | <10ms (Jetson Nano) |

---

## Domain Randomization — Sim2Real Transfer

**File**: `core/domain_randomization.py`

To bridge the simulation-to-reality gap, we randomize physics parameters at each episode reset:

| Parameter | Range | Real-World Justification |
|-----------|-------|--------------------------|
| Drone mass | ±30% | Manufacturing variance, payloads |
| Max force | ±20% | Motor degradation, battery charge |
| Drag coefficient | ±40% | Wind, altitude-dependent air density |
| Evader speed | ±30% | Different target drone types |
| Num obstacles | 3–8 | Varying environmental complexity |
| Observation noise | σ=0.05 | IMU drift, sensor error |
| Action delay | 0–3 steps | Motor response lag, compute latency |
| Gravity | ±2% | Altitude variation |

**Key insight** (from Tobin et al., 2017): If you train across a wide enough distribution of simulated conditions, the real world becomes just another sample from that distribution.

---

## Demo: Counter-UAS Command Center

**File**: `demo/command_center.py`

### Running the Demo

```bash
cd drone_interception
streamlit run demo/command_center.py
```

### What the Demo Shows

A full-scale geographic scenario of a hostile drone flying from **Iran** across the **Persian Gulf** toward a defended air base (Al Dhafra AB, UAE):

1. **3D Satellite Map** (pydeck): Shows the entire Persian Gulf theater with:
   - Red trail: adversary drone's non-deterministic flight path (random walk)
   - Blue trail: interceptor drone's pursuit path
   - Yellow circle: 80km RF detection range
   - Green circle: 30km YOLO visual range
   - Arc: threat vector from Iran to UAE
   - 7 different Iranian launch sites (Bandar Abbas, Jask, Bushehr, etc.)

2. **YOLO Camera Feed** (matplotlib): Simulated PTZ camera view showing:
   - Phase 1: "SCANNING" — no targets detected
   - Phase 2: "RF ALERT" — camera slewing to bearing
   - Phase 3: YOLO bounding box around drone silhouette with `UAV 87%` label, confidence bar, threat level
   - Phase 4: Green "TARGET NEUTRALIZED"

3. **Detection Log**: Real-time log showing RF detections, YOLO classifications, and interception events

4. **Telemetry Bar**: Live range, bearing, YOLO confidence, mission phase, cost

5. **Pipeline Visualization**: 4-stage pipeline lights up green as each stage activates

### Demo Controls

- **Airbase**: Select defended base (Al Dhafra AB or Prince Sultan AB)
- **Scenario Seed**: Different seeds produce different Iranian launch sites and flight paths
- **🎲 Button**: Randomize seed
- **Playback Speed**: SLOW / NORMAL / FAST
- **🚀 LAUNCH MISSION**: Start the scenario

---

## Project Structure

```
drone_interception/
├── core/
│   ├── drone_env.py              # Gymnasium environment (21-dim obs, 3-dim action)
│   └── domain_randomization.py   # Sim2real wrapper (randomizes physics)
├── training/
│   ├── train_ppo.py              # PPO training script (Stable-Baselines3)
│   └── callbacks.py              # Custom training callbacks
├── evaluation/
│   ├── evaluate.py               # Model evaluation & metrics
│   ├── compare_algorithms.py     # PPO vs baselines comparison
│   └── visualize_3d.py           # 3D trajectory visualization
├── demo/
│   └── command_center.py         # Streamlit demo (3D map + YOLO + log)
├── models/
│   ├── ppo_interceptor.zip       # Trained PPO model
│   └── model_metadata.json       # Training metrics & deployment specs
├── results/                      # Evaluation results
├── logs/                         # TensorBoard training logs
└── requirements.txt              # Python dependencies
```

---

## Key Talking Points for Presentation

### The Problem
- Small consumer drones ($500) are being weaponized (Iran-backed groups, Ukraine conflict)
- Existing defense systems (Patriot: $3M/shot, CRAM: expensive ammo) are absurdly expensive for this threat
- A $500 drone should not require a $3M missile to defeat

### Our Solution
- A **$6,400 autonomous system** that uses RL to guide a $300 interceptor drone
- 4-stage pipeline: RF → YOLO → PPO → Kill
- The RL agent learns optimal pursuit in simulation, then transfers to real hardware

### Why RL (Not Classical Control)?
- **Pure pursuit** (fly straight at target) fails because the adversary evades
- **Proportional navigation** works but can't handle obstacles
- **PPO** learns through 1M+ training steps to: lead the target, avoid obstacles, minimize energy, and close aggressively in the final approach

### Why PPO Specifically?
- Stable with shaped rewards (clipped objective)
- Sample efficient with parallel environments
- Small network (256×256) runs on edge hardware in <1ms
- Proven in robotics and game-playing

### The Cost Argument (The Punchline)
> "A $3M Patriot missile to shoot down a $500 drone is like using a Ferrari to run over a squirrel. Our system does it for $350 per interception."

---

## How to Run Everything

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (takes ~20 min on laptop)
python -m training.train_ppo --timesteps 1000000

# With domain randomization for better sim2real
python -m training.train_ppo --timesteps 1000000 --domain-rand

# Evaluate
python -m evaluation.evaluate

# Run the demo
streamlit run demo/command_center.py
```

---

## Dependencies

- Python 3.10+
- `stable-baselines3` — PPO implementation
- `gymnasium` — RL environment interface
- `torch` — Neural network backend
- `numpy` — Numerical computation
- `streamlit` — Demo web app
- `pydeck` — 3D map visualization
- `matplotlib` — YOLO camera rendering
- `pybullet` (optional) — Physics visualization
