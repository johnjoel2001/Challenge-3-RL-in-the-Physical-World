"""
Custom Gymnasium Environment for Low-Cost Drone Interception via RL.

Scenario: A low-cost interceptor drone (blue, RL agent) must chase and reach
within 0.8m of an evading target drone (red, scripted evasive policy) inside
a 20m x 20m x 8m arena with random obstacles.

The interceptor is modeled as a cheap commodity drone (~$300 hardware) with
limited battery and limited thrust, while still achieving reliable interception.

Physics Backend:
- If PyBullet is installed: uses PyBullet for collision detection, raycasting,
  and optional GUI visualization. Best fidelity.
- If PyBullet is NOT installed: uses a lightweight pure-NumPy physics backend
  with AABB collision detection. Same dynamics, slightly simplified collisions.
  This fallback ensures the project runs on any Python version (including 3.13+
  where PyBullet wheels may not be available).

Why Gymnasium (not old gym)?
- Gymnasium is the maintained fork of OpenAI Gym
- Required by modern Stable-Baselines3 versions
- Better API (reset returns obs + info, etc.)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List

# =============================================================================
# OPTIONAL PYBULLET IMPORT — graceful fallback to pure-NumPy physics
# =============================================================================
try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False

# =============================================================================
# ENVIRONMENT CONSTANTS — Tuned for realistic low-cost drone interception
# =============================================================================

# Arena dimensions (meters) — represents a typical urban engagement zone
ARENA_SIZE = 10.0        # Half-width: full arena is 20m x 20m
ARENA_HEIGHT = 8.0       # Max altitude — cheap drones rarely exceed this

# Interception parameters
CAPTURE_DISTANCE = 1.0   # meters — proximity to count as "caught"
                          # Real net-capture systems activate within ~1m
MAX_STEPS = 500          # Max timesteps per episode (~8.3 seconds at 60Hz)
                          # Enough for most pursuits; timeout = mission failure

# Drone physics — modeled after DJI Tello / generic $300 quadrotor
DRONE_MASS = 1.0         # kg — typical for small quadrotor with payload
MAX_FORCE = 5.0          # Newtons — max thrust per axis beyond hover
                          # Real small drones: ~15N total, minus ~10N for hover
DRAG_COEFFICIENT = 0.3   # Simplified linear drag — approximates air resistance
                          # Real drag is quadratic, but linear is stable for training
DRONE_RADIUS = 0.2       # meters — visual/collision sphere radius

# Environment complexity
NUM_OBSTACLES = 5         # Static obstacles in arena (urban clutter)
EVADER_SPEED = 2.0        # m/s — target drone's base movement speed
                          # Most hobby drones cruise at 2-5 m/s

# Simulation parameters
DT = 1.0 / 60.0          # Timestep — 60Hz physics, matching real flight controllers
GRAVITY = 9.81            # m/s^2 — standard gravity (can be randomized for sim2real)

# Raycast directions for obstacle detection — 5 rays from interceptor
# These simulate obstacle proximity sensing — in deployment, estimated from
# base-station camera depth perception or pre-loaded 3D terrain maps
RAY_DIRECTIONS = [
    np.array([1, 0, 0], dtype=np.float64),    # Forward
    np.array([-1, 0, 0], dtype=np.float64),   # Backward
    np.array([0, 1, 0], dtype=np.float64),     # Left
    np.array([0, -1, 0], dtype=np.float64),    # Right
    np.array([0, 0, -1], dtype=np.float64),    # Down (ground proximity)
]
RAY_LENGTH = 5.0  # meters — max detection range for obstacle rays

# Visual colors (RGBA) for PyBullet rendering
INTERCEPTOR_COLOR = [0.1, 0.3, 1.0, 1.0]  # Blue — our RL agent
TARGET_COLOR = [1.0, 0.1, 0.1, 1.0]        # Red — the enemy drone
OBSTACLE_COLOR = [0.5, 0.5, 0.5, 0.7]      # Gray, semi-transparent


class DroneInterceptionEnv(gym.Env):
    """
    Gymnasium environment for training an RL agent to intercept a target drone.

    The agent controls a low-cost interceptor drone and must navigate through
    obstacles to reach and "capture" (get within CAPTURE_DISTANCE of) an
    evading target drone. The target follows a scripted evasive policy that
    combines figure-8 patterns with reactive avoidance.

    Observation Space (21-dim continuous):
        [0:3]   Interceptor position (x, y, z)
        [3:6]   Interceptor velocity (vx, vy, vz)
        [6:9]   Target position (x, y, z)
        [9:12]  Target velocity (vx, vy, vz)
        [12:15] Relative vector (target_pos - interceptor_pos)
        [15]    Euclidean distance to target
        [16:21] Obstacle proximity (5 raycasts)

    Action Space (3-dim continuous, each in [-1, 1]):
        Normalized thrust in x, y, z directions.
        Scaled by MAX_FORCE. Hover thrust auto-added to z-axis.

    Reward: Shaped reward encouraging fast, efficient, safe interception.
        See _compute_reward() for full breakdown.
    """

    metadata = {"render_modes": ["human", "direct"], "render_fps": 60}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        arena_size: float = ARENA_SIZE,
        arena_height: float = ARENA_HEIGHT,
        capture_distance: float = CAPTURE_DISTANCE,
        max_steps: int = MAX_STEPS,
        drone_mass: float = DRONE_MASS,
        max_force: float = MAX_FORCE,
        drag_coeff: float = DRAG_COEFFICIENT,
        num_obstacles: int = NUM_OBSTACLES,
        evader_speed: float = EVADER_SPEED,
        gravity: float = GRAVITY,
    ) -> None:
        """
        Initialize the drone interception environment.

        Args:
            render_mode: "human" for GUI visualization, None/"direct" for headless training.
            arena_size: Half-width of the square arena in meters.
            arena_height: Maximum altitude in meters.
            capture_distance: Distance threshold for successful interception.
            max_steps: Maximum timesteps before episode timeout.
            drone_mass: Mass of the interceptor drone in kg.
            max_force: Maximum thrust force per axis in Newtons.
            drag_coeff: Linear drag coefficient for air resistance.
            num_obstacles: Number of static obstacles to place in arena.
            evader_speed: Base speed of the evading target drone in m/s.
            gravity: Gravitational acceleration (can vary for sim2real).
        """
        super().__init__()

        # Store configurable parameters (allows domain randomization to override)
        self.arena_size = arena_size
        self.arena_height = arena_height
        self.capture_distance = capture_distance
        self.max_steps = max_steps
        self.drone_mass = drone_mass
        self.max_force = max_force
        self.drag_coeff = drag_coeff
        self.num_obstacles = int(num_obstacles)
        self.evader_speed = evader_speed
        self.gravity = gravity
        self.render_mode = render_mode

        # Decide which physics backend to use
        # PyBullet provides better collision detection and GUI rendering
        # NumPy fallback uses AABB checks — good enough for training
        self._use_pybullet = PYBULLET_AVAILABLE and (render_mode == "human")
        # For headless training, pure-NumPy is faster and has no dependencies
        if not PYBULLET_AVAILABLE:
            self._use_pybullet = False

        # =====================================================================
        # Action space: 3D thrust vector, normalized to [-1, 1]
        # The RL agent outputs a 3-element vector; we scale by MAX_FORCE.
        # Hover thrust (mass * g) is automatically added to z-component,
        # so action [0, 0, 0] = hover in place.
        # =====================================================================
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # =====================================================================
        # Observation space: 21-dimensional continuous vector
        # Bounds are generous to avoid clipping during domain randomization.
        # Real values stay well within these bounds during normal operation.
        # =====================================================================
        obs_low = -np.ones(21, dtype=np.float32) * 50.0
        obs_high = np.ones(21, dtype=np.float32) * 50.0
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # PyBullet connection state (only used if _use_pybullet)
        self.physics_client = None
        self._is_connected = False

        # State tracking (initialized in reset)
        self.interceptor_pos = np.zeros(3, dtype=np.float64)
        self.interceptor_vel = np.zeros(3, dtype=np.float64)
        self.target_pos = np.zeros(3, dtype=np.float64)
        self.target_vel = np.zeros(3, dtype=np.float64)
        self.step_count = 0
        self.prev_distance = 0.0
        self.cumulative_energy = 0.0  # Tracks total thrust used (proxy for battery)

        # Obstacle data: list of dicts with 'pos' (x,y,z center), 'half_extents' (dx,dy,dz)
        # Used by both PyBullet and NumPy backends for collision checks
        self.obstacles: List[Dict[str, np.ndarray]] = []

        # PyBullet body IDs (only used if _use_pybullet)
        self.interceptor_id = None
        self.target_id = None
        self.obstacle_ids: List[int] = []
        self.ground_id = None

        # Evader state for figure-8 pattern
        self._evader_phase = 0.0

    # =========================================================================
    # PYBULLET CONNECTION MANAGEMENT
    # =========================================================================

    def _connect_pybullet(self) -> None:
        """
        Connect to PyBullet physics engine.

        Uses GUI mode for visualization (render_mode="human") or DIRECT mode
        for fast headless training. DIRECT mode is ~10x faster because it
        skips rendering — essential for training throughput.
        """
        if not self._use_pybullet or self._is_connected:
            return

        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(
                cameraDistance=15,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 3],
                physicsClientId=self.physics_client,
            )
        else:
            self.physics_client = p.connect(p.DIRECT)

        self._is_connected = True

    def _disconnect_pybullet(self) -> None:
        """Safely disconnect from PyBullet, handling edge cases."""
        if self._is_connected and self._use_pybullet:
            try:
                p.disconnect(physicsClientId=self.physics_client)
            except Exception:
                pass
            self._is_connected = False

    # =========================================================================
    # ENVIRONMENT RESET
    # =========================================================================

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to a new random initial state.

        This is called at the start of each episode. It:
        1. Resets physics (PyBullet or internal state)
        2. Spawns interceptor at random position near arena edge
        3. Spawns target at random position on opposite side
        4. Places random obstacles (avoiding center)
        5. Returns initial observation

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options (unused currently).

        Returns:
            observation: 21-dim numpy array of initial state.
            info: Dict with initial distance and other metadata.
        """
        super().reset(seed=seed)

        # Reset episode tracking
        self.step_count = 0
        self.cumulative_energy = 0.0
        self._evader_phase = self.np_random.uniform(0, 2 * np.pi)

        # Spawn drones at random positions on opposite sides of the arena
        self._spawn_drones()

        # Place random obstacles
        self._place_obstacles()

        # Setup PyBullet scene if using it
        if self._use_pybullet:
            self._setup_pybullet_scene()

        # Compute initial state
        self.prev_distance = np.linalg.norm(self.target_pos - self.interceptor_pos)

        obs = self._get_observation()
        info = {
            "distance": self.prev_distance,
            "intercepted": False,
            "collision": False,
            "timeout": False,
            "out_of_bounds": False,
        }

        return obs, info

    def _spawn_drones(self) -> None:
        """
        Spawn interceptor and target drones at random positions.

        Interceptor starts near one edge, target near the opposite edge.
        Both start at random altitudes between 2m and 6m.
        This guarantees a meaningful pursuit distance at episode start.
        """
        # Interceptor: random position on one side of the arena
        int_x = self.np_random.uniform(-self.arena_size * 0.8, -self.arena_size * 0.3)
        int_y = self.np_random.uniform(-self.arena_size * 0.5, self.arena_size * 0.5)
        int_z = self.np_random.uniform(2.0, 6.0)
        self.interceptor_pos = np.array([int_x, int_y, int_z], dtype=np.float64)
        self.interceptor_vel = np.zeros(3, dtype=np.float64)

        # Target: random position on the opposite side
        tgt_x = self.np_random.uniform(self.arena_size * 0.3, self.arena_size * 0.8)
        tgt_y = self.np_random.uniform(-self.arena_size * 0.5, self.arena_size * 0.5)
        tgt_z = self.np_random.uniform(2.0, 6.0)
        self.target_pos = np.array([tgt_x, tgt_y, tgt_z], dtype=np.float64)
        self.target_vel = np.zeros(3, dtype=np.float64)

    def _place_obstacles(self) -> None:
        """
        Place random static obstacles in the arena.

        Obstacles are tall boxes (like buildings/poles) at random positions.
        They avoid the center 3m radius so drones don't spawn inside them.
        Heights vary from 2-6m, widths from 0.5-1.5m — urban clutter.

        Stores obstacle data as dicts for backend-agnostic collision checks.
        """
        self.obstacles = []
        for _ in range(self.num_obstacles):
            # Random position — at least 3m from center to avoid spawn overlap
            while True:
                ox = self.np_random.uniform(-self.arena_size * 0.8, self.arena_size * 0.8)
                oy = self.np_random.uniform(-self.arena_size * 0.8, self.arena_size * 0.8)
                if np.sqrt(ox**2 + oy**2) > 3.0:
                    break

            # Random obstacle dimensions
            width = self.np_random.uniform(0.5, 1.5)
            height = self.np_random.uniform(2.0, 6.0)

            self.obstacles.append({
                "pos": np.array([ox, oy, height / 2.0], dtype=np.float64),
                "half_extents": np.array([width / 2.0, width / 2.0, height / 2.0], dtype=np.float64),
            })

    def _setup_pybullet_scene(self) -> None:
        """
        Initialize PyBullet scene with all objects (only called if PyBullet available).

        Creates visual/collision shapes for drones, obstacles, walls, and ground.
        """
        self._connect_pybullet()

        p.resetSimulation(physicsClientId=self.physics_client)
        p.setGravity(0, 0, -self.gravity, physicsClientId=self.physics_client)
        p.setTimeStep(DT, physicsClientId=self.physics_client)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.physics_client
        )

        # Load ground plane
        self.ground_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)

        # Create interceptor sphere
        int_col = p.createCollisionShape(p.GEOM_SPHERE, radius=DRONE_RADIUS,
                                          physicsClientId=self.physics_client)
        int_vis = p.createVisualShape(p.GEOM_SPHERE, radius=DRONE_RADIUS,
                                       rgbaColor=INTERCEPTOR_COLOR,
                                       physicsClientId=self.physics_client)
        self.interceptor_id = p.createMultiBody(
            baseMass=self.drone_mass, baseCollisionShapeIndex=int_col,
            baseVisualShapeIndex=int_vis, basePosition=self.interceptor_pos.tolist(),
            physicsClientId=self.physics_client)

        # Create target sphere
        tgt_col = p.createCollisionShape(p.GEOM_SPHERE, radius=DRONE_RADIUS,
                                          physicsClientId=self.physics_client)
        tgt_vis = p.createVisualShape(p.GEOM_SPHERE, radius=DRONE_RADIUS,
                                       rgbaColor=TARGET_COLOR,
                                       physicsClientId=self.physics_client)
        self.target_id = p.createMultiBody(
            baseMass=0.01, baseCollisionShapeIndex=tgt_col,
            baseVisualShapeIndex=tgt_vis, basePosition=self.target_pos.tolist(),
            physicsClientId=self.physics_client)
        p.changeDynamics(self.target_id, -1, linearDamping=0, angularDamping=0,
                          physicsClientId=self.physics_client)

        # Create obstacle boxes
        self.obstacle_ids = []
        for obs_data in self.obstacles:
            pos = obs_data["pos"]
            he = obs_data["half_extents"]
            obs_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=he.tolist(),
                                              physicsClientId=self.physics_client)
            obs_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=he.tolist(),
                                           rgbaColor=OBSTACLE_COLOR,
                                           physicsClientId=self.physics_client)
            obs_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=obs_col,
                                        baseVisualShapeIndex=obs_vis,
                                        basePosition=pos.tolist(),
                                        physicsClientId=self.physics_client)
            self.obstacle_ids.append(obs_id)

        # Arena walls (visual only)
        wall_height = self.arena_height
        wall_thickness = 0.1
        wall_color = [0.8, 0.8, 0.9, 0.2]
        walls = [
            ([self.arena_size, 0, wall_height/2], [wall_thickness, self.arena_size, wall_height/2]),
            ([-self.arena_size, 0, wall_height/2], [wall_thickness, self.arena_size, wall_height/2]),
            ([0, self.arena_size, wall_height/2], [self.arena_size, wall_thickness, wall_height/2]),
            ([0, -self.arena_size, wall_height/2], [self.arena_size, wall_thickness, wall_height/2]),
        ]
        for pos, extents in walls:
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=extents, rgbaColor=wall_color,
                                       physicsClientId=self.physics_client)
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=pos,
                               physicsClientId=self.physics_client)

    # =========================================================================
    # ENVIRONMENT STEP
    # =========================================================================

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.

        Process:
        1. Apply thrust forces to interceptor (action * MAX_FORCE + hover)
        2. Apply drag forces (simulates air resistance)
        3. Update interceptor position via semi-implicit Euler integration
        4. Update target drone via scripted evasive policy
        5. Check termination conditions (capture, collision, timeout, OOB)
        6. Compute shaped reward
        7. Return (obs, reward, terminated, truncated, info)

        Args:
            action: 3-dim array in [-1, 1] — normalized thrust in x, y, z.

        Returns:
            observation: 21-dim state vector.
            reward: Scalar shaped reward.
            terminated: True if episode ended (capture, collision, OOB).
            truncated: True if episode timed out (MAX_STEPS reached).
            info: Dict with episode outcome details.
        """
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float64)

        # =====================================================================
        # 1. APPLY FORCES TO INTERCEPTOR
        # =====================================================================
        # Convert normalized action to real force (Newtons)
        thrust = action * self.max_force

        # Add hover thrust: exactly counteracts gravity so action=[0,0,0] = hover
        # This is a common trick — the agent learns RELATIVE thrust, not absolute
        hover_thrust = self.drone_mass * self.gravity
        thrust[2] += hover_thrust

        # Apply linear drag: F_drag = -drag_coeff * velocity
        # This prevents infinite acceleration and adds realism
        drag_force = -self.drag_coeff * self.interceptor_vel

        # Total force on interceptor
        total_force = thrust + drag_force

        # Update interceptor velocity and position using semi-implicit Euler
        # (More stable than explicit Euler for this application)
        acceleration = total_force / self.drone_mass - np.array([0, 0, self.gravity])
        self.interceptor_vel += acceleration * DT
        self.interceptor_pos += self.interceptor_vel * DT

        # Track energy expenditure (proxy for battery consumption / cost)
        self.cumulative_energy += np.sum(action**2)

        # =====================================================================
        # 2. UPDATE TARGET DRONE (scripted evasive policy)
        # =====================================================================
        self._update_evader()

        # =====================================================================
        # 3. SYNC PYBULLET VISUALS (if using PyBullet backend)
        # =====================================================================
        if self._use_pybullet:
            p.resetBasePositionAndOrientation(
                self.interceptor_id, self.interceptor_pos.tolist(), [0, 0, 0, 1],
                physicsClientId=self.physics_client)
            p.resetBasePositionAndOrientation(
                self.target_id, self.target_pos.tolist(), [0, 0, 0, 1],
                physicsClientId=self.physics_client)
            p.stepSimulation(physicsClientId=self.physics_client)

        # =====================================================================
        # 4. CHECK TERMINATION CONDITIONS
        # =====================================================================
        distance = np.linalg.norm(self.target_pos - self.interceptor_pos)
        terminated = False
        truncated = False
        info = {
            "intercepted": False,
            "collision": False,
            "timeout": False,
            "out_of_bounds": False,
            "distance": distance,
            "energy_used": self.cumulative_energy,
            "steps": self.step_count,
        }

        # Check interception (SUCCESS — the whole point!)
        if distance < self.capture_distance:
            terminated = True
            info["intercepted"] = True

        # Check obstacle collision (FAILURE — drone crashed)
        elif self._check_obstacle_collision():
            terminated = True
            info["collision"] = True

        # Check out of bounds (FAILURE — drone left the arena)
        elif self._check_out_of_bounds(self.interceptor_pos):
            terminated = True
            info["out_of_bounds"] = True

        # Check timeout (FAILURE — took too long, target escaped)
        elif self.step_count >= self.max_steps:
            truncated = True
            info["timeout"] = True

        # =====================================================================
        # 5. COMPUTE REWARD
        # =====================================================================
        reward = self._compute_reward(action, distance, info)

        # Update previous distance for next step's progress calculation
        self.prev_distance = distance

        obs = self._get_observation()
        return obs, reward, terminated, truncated, info

    # =========================================================================
    # EVADER (TARGET DRONE) SCRIPTED POLICY
    # =========================================================================

    def _update_evader(self) -> None:
        """
        Update the target drone using a scripted evasive policy.

        The evader uses a multi-layered strategy:
        1. BASE MOTION: Figure-8 pattern with vertical oscillation
           (simulates a drone on a mission, not just hovering)
        2. REACTIVE EVASION: When interceptor gets within 4m, accelerate away
           proportional to closeness (simulates threat detection + response)
        3. STOCHASTIC NOISE: Gaussian perturbation for unpredictability
           (makes the RL agent robust to imperfect predictions)
        4. BOUNDARY CLAMPING: Stay within arena, maintain min altitude
           (prevents trivially easy escapes by leaving the arena)
        """
        # Advance the figure-8 phase
        self._evader_phase += DT * 0.5  # Controls figure-8 speed

        # --- Layer 1: Base figure-8 motion ---
        # Classic lemniscate (figure-8) provides smooth, continuous movement
        base_vx = self.evader_speed * np.cos(self._evader_phase)
        base_vy = self.evader_speed * np.sin(2 * self._evader_phase) * 0.5
        base_vz = self.evader_speed * 0.3 * np.sin(self._evader_phase * 1.5)

        # --- Layer 2: Reactive evasion ---
        # When the interceptor gets close, the target "panics" and accelerates away
        # This is realistic — real drones have proximity detection
        rel_vec = self.target_pos - self.interceptor_pos
        dist = np.linalg.norm(rel_vec)
        evasion = np.zeros(3)

        if dist < 4.0 and dist > 0.01:
            # Evasion force inversely proportional to distance
            # Closer interceptor → stronger evasion response
            evasion_strength = (4.0 - dist) / 4.0 * self.evader_speed * 1.2
            evasion = (rel_vec / dist) * evasion_strength

        # --- Layer 3: Stochastic noise ---
        # Makes the evader unpredictable — crucial for RL generalization
        noise = self.np_random.normal(0, 0.3, size=3)

        # Combine all velocity components
        target_vel = np.array([base_vx, base_vy, base_vz]) + evasion + noise

        # Clamp velocity magnitude to prevent unrealistic speeds
        speed = np.linalg.norm(target_vel)
        max_evader_speed = self.evader_speed * 2.5
        if speed > max_evader_speed:
            target_vel = target_vel / speed * max_evader_speed

        self.target_vel = target_vel

        # Update position
        self.target_pos += self.target_vel * DT

        # --- Layer 4: Boundary clamping ---
        # Keep target inside arena with min altitude of 1.0m
        self.target_pos[0] = np.clip(
            self.target_pos[0], -self.arena_size * 0.9, self.arena_size * 0.9
        )
        self.target_pos[1] = np.clip(
            self.target_pos[1], -self.arena_size * 0.9, self.arena_size * 0.9
        )
        self.target_pos[2] = np.clip(self.target_pos[2], 1.0, self.arena_height - 0.5)

    # =========================================================================
    # REWARD FUNCTION
    # =========================================================================

    def _compute_reward(
        self, action: np.ndarray, distance: float, info: Dict[str, Any]
    ) -> float:
        """
        Compute the shaped reward for this timestep.

        Reward design philosophy: We want the agent to learn EFFICIENT interception.
        Not just "catch the target" but "catch it quickly with minimal energy."
        This directly maps to real-world cost — less battery = cheaper mission.

        CRITICAL DESIGN CONSTRAINT: The interception bonus MUST dominate all
        cumulative per-step rewards. Otherwise the agent learns to orbit the
        target (collecting proximity rewards forever) instead of closing in.

        Math check: max per-step proximity ~ 3.0, over 500 steps = 1500.
        Interception bonus = 500 >> 1500 when combined with time penalty
        (-0.5 * 200 steps = -100), so early interception is clearly optimal.

        Components:
        1. PROGRESS: Reward for closing distance (gradient toward target)
        2. INTERCEPTION BONUS: Massive reward for catching target (500)
        3. COLLISION PENALTY: Punishment for crashing (-75)
        4. ENERGY PENALTY: Cost for excessive thrust (battery = money)
        5. TIME PENALTY: Significant per-step cost (-0.5) to discourage orbiting
        6. PROXIMITY BONUS: Modest reward within 3m (capped to prevent orbit)
        7. OBSTACLE PROXIMITY WARNING: Continuous penalty near obstacles
        8. BOUNDARY PENALTY: Punishment for leaving the arena

        Args:
            action: The action taken this step (for energy calculation).
            distance: Current distance to target.
            info: Episode info dict with termination flags.

        Returns:
            Total shaped reward (float).
        """
        reward = 0.0

        # 1. PROGRESS REWARD: Encourage closing distance to target
        # This is the main gradient signal that guides the agent toward the target.
        # Without this, the agent would have to discover interception purely by chance.
        # Note: progress is ZERO for orbiting (distance doesn't change), so this
        # only rewards approach, not station-keeping.
        progress = (self.prev_distance - distance) * 15.0
        reward += progress

        # 2. INTERCEPTION BONUS: The massive payoff
        # This MUST be large enough that a quick interception beats any amount
        # of orbiting. 500 >> max cumulative proximity (~1500 over 500 steps)
        # because time penalty makes long episodes costly.
        # Also give speed bonus: faster interception = bigger reward
        if info["intercepted"]:
            # Base bonus + time bonus (fewer steps = more reward)
            steps_bonus = max(0, (self.max_steps - self.step_count)) * 0.5
            reward += 500.0 + steps_bonus

        # 3. COLLISION PENALTY: Crashing is expensive (drone replacement cost)
        if info["collision"]:
            reward -= 75.0

        # 4. ENERGY PENALTY: Penalize excessive thrust
        # KEY INSIGHT: This trains the agent to be efficient, not just fast.
        # In real deployment, energy = battery life = mission cost.
        # Kept small so it doesn't discourage aggressive pursuit.
        energy_penalty = -0.02 * np.sum(action**2)
        reward += energy_penalty

        # 5. TIME PENALTY: Significant per-step cost
        # This is the key anti-orbiting mechanism: every step costs -0.5,
        # so a 500-step orbit costs -250, making quick interception
        # (500 bonus - 100 time cost = +400) clearly superior.
        reward -= 0.5

        # 6. PROXIMITY BONUS: Modest reward when closing in (within 3m)
        # Kept small (max 3.0/step) so it can't compete with interception bonus.
        # Uses exponential scaling to reward the final approach strongly.
        if distance < 3.0:
            # Exponential: ~0.5 at 3m, ~3.0 at 0.5m — rewards commitment
            proximity_bonus = 1.0 / (distance + 0.3) - 0.3
            reward += max(0.0, proximity_bonus)

        # 7. OBSTACLE PROXIMITY WARNING: Penalize getting dangerously close
        # Instead of only punishing collisions (-75 at death), we give a
        # continuous negative signal when the drone flies near obstacles.
        # This teaches avoidance BEFORE crashing — much better learning signal.
        obs_distances = self._raycast_obstacles()
        min_obs_dist = np.min(obs_distances)  # 0=touching, 1=clear
        if min_obs_dist < 0.3:  # Within 30% of ray length (~1.5m)
            obstacle_warning = -3.0 * (0.3 - min_obs_dist) / 0.3
            reward += obstacle_warning

        # 8. BOUNDARY PENALTY: Leaving the arena is mission failure
        if info["out_of_bounds"]:
            reward -= 75.0

        return float(reward)

    # =========================================================================
    # OBSERVATION
    # =========================================================================

    def _get_observation(self) -> np.ndarray:
        """
        Construct the 21-dimensional observation vector.

        The observation contains everything the agent needs to make decisions:
        - Its own state (position + velocity)
        - Target state (position + velocity)
        - Relative geometry (direction and distance to target)
        - Obstacle proximity (5 raycasts for spatial awareness)

        In real deployment, all sensing and computation happens at the base station:
        - Drone telemetry (position/velocity) sent to base via radio link
        - PTZ camera + YOLOv8 at the base → target position/velocity
        - Obstacle proximity estimated from base camera depth or pre-loaded 3D terrain maps
        - PPO policy runs on Jetson at the base, outputs thrust commands back to the drone
        - The drone itself is a reusable dumb airframe — motors, flight controller, radio, battery

        Returns:
            21-dim numpy array (float32).
        """
        # Relative vector from interceptor to target
        relative_pos = self.target_pos - self.interceptor_pos
        distance = np.linalg.norm(relative_pos)

        # Obstacle proximity via raycasting
        obstacle_distances = self._raycast_obstacles()

        obs = np.concatenate([
            self.interceptor_pos,        # [0:3]   Own position
            self.interceptor_vel,        # [3:6]   Own velocity
            self.target_pos,             # [6:9]   Target position
            self.target_vel,             # [9:12]  Target velocity
            relative_pos,                # [12:15] Relative vector
            [distance],                  # [15]    Euclidean distance
            obstacle_distances,          # [16:21] Obstacle proximity (5 rays)
        ]).astype(np.float32)

        return obs

    # =========================================================================
    # COLLISION DETECTION & RAYCASTING (backend-agnostic)
    # =========================================================================

    def _raycast_obstacles(self) -> np.ndarray:
        """
        Cast 5 rays from the interceptor to detect nearby obstacles.

        Returns normalized distances (0 = obstacle at drone, 1 = no obstacle within range).
        This simulates cheap proximity sensors that a real $300 drone would have.

        Uses PyBullet raycasting if available, otherwise uses analytic
        ray-AABB intersection (same math, just computed in Python).

        Returns:
            5-element array of normalized obstacle distances.
        """
        if self._use_pybullet:
            return self._raycast_pybullet()
        else:
            return self._raycast_numpy()

    def _raycast_pybullet(self) -> np.ndarray:
        """PyBullet-based raycasting for obstacle detection."""
        distances = np.ones(5, dtype=np.float64)
        start = self.interceptor_pos.tolist()

        for i, direction in enumerate(RAY_DIRECTIONS):
            end = (self.interceptor_pos + direction * RAY_LENGTH).tolist()
            try:
                result = p.rayTest(start, end, physicsClientId=self.physics_client)
                if result and len(result) > 0:
                    hit_fraction = result[0][2]
                    hit_object = result[0][0]
                    if hit_object != self.target_id and hit_object != self.interceptor_id:
                        distances[i] = hit_fraction
            except Exception:
                pass
        return distances

    def _raycast_numpy(self) -> np.ndarray:
        """
        Pure-NumPy raycasting using ray-AABB (Axis-Aligned Bounding Box) intersection.

        For each of the 5 ray directions, we test against all obstacles and
        the ground plane, returning the closest hit fraction (0-1).

        This is the analytical solution — same result as PyBullet's raycasting
        but computed without the C library dependency.
        """
        distances = np.ones(5, dtype=np.float64)
        origin = self.interceptor_pos

        for i, direction in enumerate(RAY_DIRECTIONS):
            closest_frac = 1.0

            # Test against each obstacle (AABB intersection)
            for obs_data in self.obstacles:
                obs_min = obs_data["pos"] - obs_data["half_extents"]
                obs_max = obs_data["pos"] + obs_data["half_extents"]
                frac = self._ray_aabb_intersect(origin, direction, obs_min, obs_max)
                if frac is not None and frac < closest_frac:
                    closest_frac = frac

            # Test against ground plane (z = 0)
            if direction[2] < -1e-6:  # Ray pointing downward
                t = -origin[2] / (direction[2] * RAY_LENGTH)
                # Normalize: t is fraction of unit direction, we need fraction of RAY_LENGTH
                ground_frac = origin[2] / (abs(direction[2]) * RAY_LENGTH)
                if 0 <= ground_frac < closest_frac:
                    closest_frac = ground_frac

            distances[i] = closest_frac

        return distances

    @staticmethod
    def _ray_aabb_intersect(
        origin: np.ndarray,
        direction: np.ndarray,
        box_min: np.ndarray,
        box_max: np.ndarray,
    ) -> Optional[float]:
        """
        Ray-AABB intersection test (slab method).

        Returns the hit fraction (0 to 1, as fraction of RAY_LENGTH) or None if no hit.

        Args:
            origin: Ray origin (3D position).
            direction: Ray direction (unit vector).
            box_min: AABB minimum corner.
            box_max: AABB maximum corner.

        Returns:
            Hit fraction in [0, 1] or None if ray misses the box.
        """
        ray_end = origin + direction * RAY_LENGTH
        scaled_dir = direction * RAY_LENGTH
        # Avoid division by zero for axis-aligned rays (e.g., direction=[1,0,0])
        safe_dir = np.where(np.abs(scaled_dir) > 1e-10, scaled_dir, 1e-10)
        inv_dir = 1.0 / safe_dir

        t1 = (box_min - origin) * inv_dir
        t2 = (box_max - origin) * inv_dir

        t_near = np.minimum(t1, t2)
        t_far = np.maximum(t1, t2)

        t_enter = np.max(t_near)
        t_exit = np.min(t_far)

        if t_enter > t_exit or t_exit < 0:
            return None

        t_hit = t_enter if t_enter >= 0 else t_exit
        if 0 <= t_hit <= 1.0:
            return float(t_hit)
        return None

    def _check_obstacle_collision(self) -> bool:
        """
        Check if the interceptor has collided with any obstacle.

        Uses AABB (Axis-Aligned Bounding Box) collision detection:
        the drone's sphere (simplified to a point + radius) is tested
        against each obstacle's expanded AABB (expanded by drone radius).

        Also checks ground collision (altitude < drone radius).

        Returns:
            True if interceptor is colliding with an obstacle or ground.
        """
        # Ground collision check
        if self.interceptor_pos[2] < DRONE_RADIUS:
            return True

        # AABB collision against each obstacle (expand box by drone radius)
        pos = self.interceptor_pos
        for obs_data in self.obstacles:
            obs_min = obs_data["pos"] - obs_data["half_extents"] - DRONE_RADIUS
            obs_max = obs_data["pos"] + obs_data["half_extents"] + DRONE_RADIUS

            if (obs_min[0] <= pos[0] <= obs_max[0] and
                obs_min[1] <= pos[1] <= obs_max[1] and
                obs_min[2] <= pos[2] <= obs_max[2]):
                return True

        return False

    def _check_out_of_bounds(self, pos: np.ndarray) -> bool:
        """
        Check if a position is outside the arena boundaries.

        Args:
            pos: 3D position to check.

        Returns:
            True if position is outside arena bounds.
        """
        if abs(pos[0]) > self.arena_size:
            return True
        if abs(pos[1]) > self.arena_size:
            return True
        if pos[2] > self.arena_height or pos[2] < 0.0:
            return True
        return False

    # =========================================================================
    # RENDERING & CLEANUP
    # =========================================================================

    def render(self) -> None:
        """
        Render the environment.

        In "human" mode with PyBullet, GUI handles rendering automatically.
        This method exists for Gymnasium API compliance.
        """
        if self.render_mode == "human" and self._use_pybullet:
            import time
            time.sleep(DT)

    def close(self) -> None:
        """Clean up PyBullet connection when environment is destroyed."""
        self._disconnect_pybullet()


# =============================================================================
# SANITY TEST — Run this file directly to verify the environment works
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DroneInterceptionEnv — Sanity Test")
    print("=" * 60)
    print(f"  PyBullet available: {PYBULLET_AVAILABLE}")
    print(f"  Physics backend:    {'PyBullet' if PYBULLET_AVAILABLE else 'Pure NumPy'}")

    env = DroneInterceptionEnv(render_mode=None)  # Headless mode
    obs, info = env.reset(seed=42)

    print(f"\nObservation shape: {obs.shape}")
    print(f"Observation range: [{obs.min():.2f}, {obs.max():.2f}]")
    print(f"Action space: {env.action_space}")
    print(f"Initial distance: {info['distance']:.2f}m")

    total_reward = 0.0
    for step in range(100):
        # Random actions — the untrained "monkey test"
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"\nEpisode ended at step {step + 1}")
            print(f"  Intercepted: {info['intercepted']}")
            print(f"  Collision:   {info['collision']}")
            print(f"  Timeout:     {info['timeout']}")
            print(f"  OOB:         {info['out_of_bounds']}")
            break

    print(f"\nTotal reward over {min(step + 1, 100)} steps: {total_reward:.2f}")
    print(f"Final distance: {info['distance']:.2f}m")
    print(f"Energy used: {info.get('energy_used', 0):.2f}")
    print(f"\n✓ Environment sanity test passed!")

    env.close()
