"""Gymnasium environment for RL-based drone interception.

Agent (blue) intercepts evading target (red) in obstacle-filled 20x20x8m arena.
Physics: PyBullet (if available) + AABB collisions; fallback pure-NumPy backend.
Gymnasium required for Stable-Baselines3 compatibility.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List
try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False

# Arena dimensions (meters)
ARENA_SIZE = 10.0        # Half-width: full arena is 20m x 20m
ARENA_HEIGHT = 8.0       # Max altitude for cheap drones

# Interception parameters
CAPTURE_DISTANCE = 1.0   # Meters; net-capture systems activate within ~1m
MAX_STEPS = 500          # Max steps per episode (~8.3s at 60Hz); timeout = failure

# Drone physics: DJI Tello-like $300 quadrotor
DRONE_MASS = 1.0         # kg; typical with payload
MAX_FORCE = 5.0          # Newtons per axis beyond hover thrust
DRAG_COEFFICIENT = 0.3   # Damping coefficient
DRONE_RADIUS = 0.2       # Meters; collision radius

# Environment
NUM_OBSTACLES = 5        # Number of obstacles
EVADER_SPEED = 2.0       # m/s; typical drone cruising speed

# Simulation
DT = 1.0 / 60.0          # 60Hz physics loop (standard for flight controllers)
GRAVITY = 9.81           # m/s^2; can be randomized for sim2real

# Obstacle sensors: 5 rays for proximity detection
RAY_DIRECTIONS = [
    np.array([1, 0, 0], dtype=np.float64),    # Forward
    np.array([-1, 0, 0], dtype=np.float64),   # Backward
    np.array([0, 1, 0], dtype=np.float64),     # Left
    np.array([0, -1, 0], dtype=np.float64),    # Right
    np.array([0, 0, -1], dtype=np.float64),    # Down (ground proximity)
]
RAY_LENGTH = 5.0  # Meters; max detection range

# Visual colors (RGBA)
INTERCEPTOR_COLOR = [0.1, 0.3, 1.0, 1.0]  # Blue
TARGET_COLOR = [1.0, 0.1, 0.1, 1.0]        # Red
OBSTACLE_COLOR = [0.5, 0.5, 0.5, 0.7]      # Gray


class DroneInterceptionEnv(gym.Env):
    """Intercept an evading target while dodging obstacles.
    
    21-dim observation (own state + target state + relative geometry + obstacle proximity).
    3-dim action (thrust per axis, -1 to 1; hover thrust auto-applied to z-axis).
    Shaped reward favoring fast, efficient interception.
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
        """Initialize environment with configurable physics and arena parameters."""
        super().__init__()

        # Store all parameters so domain randomization wrapper can override them
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

        # PyBullet for GUI rendering, NumPy backend for fast training
        self._use_pybullet = PYBULLET_AVAILABLE and (render_mode == "human")
        if not PYBULLET_AVAILABLE:
            self._use_pybullet = False

        # Action space: [0,0,0] = hover; x,y,z thrust normalized to [-1,1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Observation: 21 dims with wide bounds to prevent clipping
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

        # Obstacles: shared between PyBullet and NumPy backends
        self.obstacles: List[Dict[str, np.ndarray]] = []

        # PyBullet body IDs (only used if _use_pybullet)
        self.interceptor_id = None
        self.target_id = None
        self.obstacle_ids: List[int] = []
        self.ground_id = None

        # Evader state for figure-8 pattern
        self._evader_phase = 0.0

    def _connect_pybullet(self) -> None:
        """Connect to PyBullet; GUI mode for visualization, DIRECT for fast headless training."""
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

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset episode: spawn drones randomly, place obstacles, return initial observation."""
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
        """Spawn drones on opposite sides for guaranteed pursuit distance."""
        # Randomize height and lateral position slightly so each episode is different
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
        """Place random tall boxes; avoid center to prevent spawn interference."""
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
        """Set up PyBullet scene once per episode (only if available)."""
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

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step: apply forces, update target, check terminations, compute reward."""
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float64)

        # Thrust = action * max_force + hover thrust (so [0,0,0] maintains altitude)
        thrust = action * self.max_force
        hover_thrust = self.drone_mass * self.gravity
        thrust[2] += hover_thrust

        # Drag prevents infinite acceleration
        drag_force = -self.drag_coeff * self.interceptor_vel
        total_force = thrust + drag_force

        # Semi-implicit Euler integration (more stable)
        acceleration = total_force / self.drone_mass - np.array([0, 0, self.gravity])
        self.interceptor_vel += acceleration * DT
        self.interceptor_pos += self.interceptor_vel * DT

        # Track energy (proxy for battery consumption / mission cost)
        self.cumulative_energy += np.sum(action**2)

        # Update target drone
        self._update_evader()

        # Sync PyBullet visuals if available
        if self._use_pybullet:
            p.resetBasePositionAndOrientation(
                self.interceptor_id, self.interceptor_pos.tolist(), [0, 0, 0, 1],
                physicsClientId=self.physics_client)
            p.resetBasePositionAndOrientation(
                self.target_id, self.target_pos.tolist(), [0, 0, 0, 1],
                physicsClientId=self.physics_client)
            p.stepSimulation(physicsClientId=self.physics_client)

        # Check termination conditions
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

        # Success: caught the target
        if distance < self.capture_distance:
            terminated = True
            info["intercepted"] = True

        # Failure: crashed into obstacle
        elif self._check_obstacle_collision():
            terminated = True
            info["collision"] = True

        # Failure: left arena
        elif self._check_out_of_bounds(self.interceptor_pos):
            terminated = True
            info["out_of_bounds"] = True

        # Failure: timeout
        elif self.step_count >= self.max_steps:
            truncated = True
            info["timeout"] = True

        # Update previous distance for next step's progress calculation
        self.prev_distance = distance

        obs = self._get_observation()
        return obs, reward, terminated, truncated, info

    def _update_evader(self) -> None:
        """Update target: figure-8 base motion + reactive evasion + noise + boundary clamp."""
        # Advance figure-8 phase
        self._evader_phase += DT * 0.5

        # Base figure-8 motion — smooth continuous movement
        base_vx = self.evader_speed * np.cos(self._evader_phase)
        base_vy = self.evader_speed * np.sin(2 * self._evader_phase) * 0.5
        base_vz = self.evader_speed * 0.3 * np.sin(self._evader_phase * 1.5)

        # Reactive evasion: when interceptor closes in, dodge proportionally (simple proximity logic)
        rel_vec = self.target_pos - self.interceptor_pos
        dist = np.linalg.norm(rel_vec)
        evasion = np.zeros(3)

        if dist < 4.0 and dist > 0.01:
            evasion_strength = (4.0 - dist) / 4.0 * self.evader_speed * 1.2
            evasion = (rel_vec / dist) * evasion_strength

        # Add noise to break determinism
        noise = self.np_random.normal(0, 0.3, size=3)

        # Combine all velocity components
        target_vel = np.array([base_vx, base_vy, base_vz]) + evasion + noise
        speed = np.linalg.norm(target_vel)
        max_evader_speed = self.evader_speed * 2.5
        if speed > max_evader_speed:
            target_vel = target_vel / speed * max_evader_speed

        self.target_vel = target_vel
        self.target_pos += self.target_vel * DT

        # Keep target in bounds
        self.target_pos[0] = np.clip(
            self.target_pos[0], -self.arena_size * 0.9, self.arena_size * 0.9
        )
        self.target_pos[1] = np.clip(
            self.target_pos[1], -self.arena_size * 0.9, self.arena_size * 0.9
        )
        self.target_pos[2] = np.clip(self.target_pos[2], 1.0, self.arena_height - 0.5)

    def _compute_reward(
        self, action: np.ndarray, distance: float, info: Dict[str, Any]
    ) -> float:
        """Compute shaped reward for efficient interception (progress + bonus + penalties)."""
        reward = 0.0

        # Progress: gradient toward target (zero if orbiting at same distance)
        progress = (self.prev_distance - distance) * 15.0
        reward += progress

        # Interception bonus: big enough to beat any orbit strategy (time penalty -0.5/step adds up)
        if info["intercepted"]:
            steps_bonus = max(0, (self.max_steps - self.step_count)) * 0.5
            reward += 500.0 + steps_bonus

        # Crash = drone destroyed
        if info["collision"]:
            reward -= 75.0

        # Penalize excessive thrust (battery cost)
        energy_penalty = -0.02 * np.sum(action**2)
        reward += energy_penalty

        # Time penalty: -0.5/step discourages hovering (500 steps = -250 penalty)
        reward -= 0.5

        # Proximity bonus: mild signal to get closer (but won't compete with interception)
        if distance < 3.0:
            proximity_bonus = 1.0 / (distance + 0.3) - 0.3
            reward += max(0.0, proximity_bonus)

        # Warn before collision (gradual signal, not just -75 at impact)
        obs_distances = self._raycast_obstacles()
        min_obs_dist = np.min(obs_distances)
        if min_obs_dist < 0.3:
            obstacle_warning = -3.0 * (0.3 - min_obs_dist) / 0.3
            reward += obstacle_warning

        # Left the arena
        if info["out_of_bounds"]:
            reward -= 75.0

        return float(reward)

    def _get_observation(self) -> np.ndarray:
        """Build 21-dim observation vector."""
        # Relative vector
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

    def _raycast_obstacles(self) -> np.ndarray:
        """Cast 5 rays to detect obstacles; PyBullet if available, else NumPy."""
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
        """Pure-NumPy raycasting via ray-AABB intersection (no PyBullet dependency)."""
        distances = np.ones(5, dtype=np.float64)
        origin = self.interceptor_pos

        for i, direction in enumerate(RAY_DIRECTIONS):
            closest_frac = 1.0

            for obs_data in self.obstacles:
                obs_min = obs_data["pos"] - obs_data["half_extents"]
                obs_max = obs_data["pos"] + obs_data["half_extents"]
                frac = self._ray_aabb_intersect(origin, direction, obs_min, obs_max)
                if frac is not None and frac < closest_frac:
                    closest_frac = frac

            # Ground collision
            if direction[2] < -1e-6:
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
        """Slab method ray-AABB intersection; returns hit fraction [0,1] or None."""
        scaled_dir = direction * RAY_LENGTH
        # Handle axis-aligned rays (avoid dividing by near-zero)
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
        """Check if drone hit anything."""
        if self.interceptor_pos[2] < DRONE_RADIUS:
            return True

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
        """Return True if position exceeds arena boundaries."""
        if abs(pos[0]) > self.arena_size:
            return True
        if abs(pos[1]) > self.arena_size:
            return True
        if pos[2] > self.arena_height or pos[2] < 0.0:
            return True
        return False

    # =========================================================================
    # RENDERING & CLEANUP
    def render(self) -> None:
        """Render environment (PyBullet GUI handles this automatically)."""
        if self.render_mode == "human" and self._use_pybullet:
            import time
            time.sleep(DT)

    def close(self) -> None:
        """Clean up PyBullet connection."""
        self._disconnect_pybullet()

if __name__ == "__main__":
    env = DroneInterceptionEnv(render_mode=None)
    obs, info = env.reset(seed=42)

    print(f"Backend: {'PyBullet' if PYBULLET_AVAILABLE else 'NumPy'}")
    print(f"Observation shape: {obs.shape}, range: [{obs.min():.2f}, {obs.max():.2f}]")
    print(f"Initial distance: {info['distance']:.2f}m\n")

    total_reward = 0.0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            status = "intercepted" if info["intercepted"] else ("collision" if info["collision"] else ("timeout" if info["timeout"] else "oob"))
            print(f"Episode ended at step {step + 1} ({status})")
            print(f"Final distance: {info['distance']:.2f}m, reward: {total_reward:.2f}")
            break

    env.close()
