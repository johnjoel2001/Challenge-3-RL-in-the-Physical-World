"""
Domain Randomization Wrapper for Sim2Real Transfer.

Domain randomization is one of the most effective techniques for bridging
the gap between simulation and reality. The idea is simple but powerful:
if you train your policy across a WIDE RANGE of simulated conditions,
the real world becomes just another sample from that distribution.

Key reference: Tobin et al. (2017) "Domain Randomization for Transferring
Deep Neural Networks from Simulation to the Real World"

Also: OpenAI's Rubik's Cube work showed that aggressive domain randomization
enabled zero-shot sim2real transfer for dexterous manipulation.

For our drone interception task, we randomize parameters that would vary
in real-world deployment: mass, thrust, drag, wind, sensor noise, etc.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple
from collections import deque


class DomainRandomizationWrapper(gym.Wrapper):
    """
    Gymnasium Wrapper that randomizes physics parameters at each episode reset.

    This forces the RL policy to be robust to a wide range of conditions,
    which is essential for sim2real transfer. A policy trained with only
    nominal parameters will fail when real-world conditions don't match
    the simulation exactly (and they never do).

    Randomized parameters and their real-world justification:
    - Drone mass: manufacturing variance, different payloads
    - Max force: motor degradation, battery charge state
    - Drag coefficient: wind conditions, altitude-dependent air density
    - Evader speed: different target drone types/capabilities
    - Num obstacles: varying environmental complexity
    - Observation noise: IMU drift, sensor measurement error
    - Action delay: motor response lag, onboard compute latency
    - Gravity: altitude variation (negligible but included for completeness)

    Usage:
        base_env = DroneInterceptionEnv()
        env = DomainRandomizationWrapper(base_env)
        obs, info = env.reset()  # Parameters randomized here
    """

    def __init__(
        self,
        env: gym.Env,
        randomize_mass: bool = True,
        randomize_force: bool = True,
        randomize_drag: bool = True,
        randomize_evader: bool = True,
        randomize_obstacles: bool = True,
        randomize_obs_noise: bool = True,
        randomize_action_delay: bool = True,
        randomize_gravity: bool = True,
    ) -> None:
        """
        Initialize the domain randomization wrapper.

        Args:
            env: The base DroneInterceptionEnv to wrap.
            randomize_mass: Whether to randomize drone mass.
            randomize_force: Whether to randomize max thrust force.
            randomize_drag: Whether to randomize drag coefficient.
            randomize_evader: Whether to randomize evader speed.
            randomize_obstacles: Whether to randomize obstacle count.
            randomize_obs_noise: Whether to add observation noise.
            randomize_action_delay: Whether to add action delay.
            randomize_gravity: Whether to randomize gravity.
        """
        super().__init__(env)

        # Store which randomizations are active
        # Each can be toggled independently for ablation studies
        self.randomize_mass = randomize_mass
        self.randomize_force = randomize_force
        self.randomize_drag = randomize_drag
        self.randomize_evader = randomize_evader
        self.randomize_obstacles = randomize_obstacles
        self.randomize_obs_noise = randomize_obs_noise
        self.randomize_action_delay = randomize_action_delay
        self.randomize_gravity = randomize_gravity

        # Current observation noise level (sampled each episode)
        self._obs_noise_std = 0.0

        # Action delay buffer — stores recent actions and replays them delayed
        # Simulates motor response lag and onboard compute latency
        self._action_delay_steps = 0
        self._action_buffer: deque = deque(maxlen=4)

        # Log what was sampled each episode (for analysis and debugging)
        self.randomization_log: Dict[str, Any] = {}

        # History of all episode randomizations (for statistical analysis)
        self.randomization_history: list = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment with randomized physics parameters.

        At the start of each episode, we sample new values for all active
        randomization parameters. This means every episode is slightly
        different, forcing the policy to generalize.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.

        Returns:
            observation: Initial observation (potentially with added noise).
            info: Dict with randomization log included.
        """
        # Use numpy random for parameter sampling
        rng = np.random.RandomState(seed)

        self.randomization_log = {}

        # =====================================================================
        # MASS RANDOMIZATION: [0.7, 1.5] kg
        # Why: Manufacturing tolerance on frame/motors is ±10-20%.
        #      Payload variation (camera, net launcher, etc.) adds more.
        #      A heavier drone has more inertia — harder to maneuver.
        # =====================================================================
        if self.randomize_mass:
            mass = rng.uniform(0.7, 1.5)
            self.env.unwrapped.drone_mass = mass
            self.randomization_log["drone_mass"] = mass

        # =====================================================================
        # MAX FORCE RANDOMIZATION: [3.5, 7.0] N
        # Why: Motor performance degrades with use (bearing wear, magnet weakening).
        #      Battery voltage drops during flight (12.6V full → 10.5V empty),
        #      reducing available thrust by ~15-20%.
        #      Cold weather reduces battery output further.
        # =====================================================================
        if self.randomize_force:
            force = rng.uniform(3.5, 7.0)
            self.env.unwrapped.max_force = force
            self.randomization_log["max_force"] = force

        # =====================================================================
        # DRAG COEFFICIENT RANDOMIZATION: [0.1, 0.6]
        # Why: Wind is the #1 environmental factor for small drones.
        #      Effective drag varies with wind speed and direction.
        #      Also: air density changes ~12% between sea level and 2000m altitude.
        #      Low drag (0.1) = calm indoor conditions.
        #      High drag (0.6) = moderate outdoor wind (3-4 m/s).
        # =====================================================================
        if self.randomize_drag:
            drag = rng.uniform(0.1, 0.6)
            self.env.unwrapped.drag_coeff = drag
            self.randomization_log["drag_coefficient"] = drag

        # =====================================================================
        # EVADER SPEED RANDOMIZATION: [1.0, 3.5] m/s
        # Why: Target drones vary wildly in capability.
        #      DJI Mini (~1 m/s cruise) vs racing drone (~10 m/s).
        #      We focus on the "hobby/commercial drone" threat category.
        #      The policy must handle slow loitering drones and fast ones.
        # =====================================================================
        if self.randomize_evader:
            speed = rng.uniform(1.0, 3.5)
            self.env.unwrapped.evader_speed = speed
            self.randomization_log["evader_speed"] = speed

        # =====================================================================
        # OBSTACLE COUNT RANDOMIZATION: [2, 8]
        # Why: Operating environments vary from open fields (few obstacles)
        #      to dense urban areas (many obstacles).
        #      The policy must navigate both.
        # =====================================================================
        if self.randomize_obstacles:
            num_obs = rng.randint(2, 9)  # [2, 8] inclusive
            self.env.unwrapped.num_obstacles = int(num_obs)
            self.randomization_log["num_obstacles"] = int(num_obs)

        # =====================================================================
        # OBSERVATION NOISE: σ ∈ [0, 0.05]
        # Why: Real sensors are noisy.
        #      IMU gyroscope drift: ~1°/hour (contributes to position error)
        #      Barometer altitude: ±0.5m precision
        #      Optical flow velocity: ±0.1 m/s error
        #      Camera-based target tracking: ±0.2m at 10m range
        #      Training with noise makes the policy robust to sensor imperfections.
        # =====================================================================
        if self.randomize_obs_noise:
            self._obs_noise_std = rng.uniform(0, 0.05)
            self.randomization_log["obs_noise_std"] = self._obs_noise_std
        else:
            self._obs_noise_std = 0.0

        # =====================================================================
        # ACTION DELAY: [0, 3] steps
        # Why: Real motor controllers have latency.
        #      ESC (Electronic Speed Controller): 1-5ms response time
        #      Flight controller loop: 1-20ms depending on hardware
        #      Onboard inference (Jetson Nano): ~10ms per forward pass
        #      At 60Hz sim, 3 steps = 50ms delay — realistic for cheap hardware.
        # =====================================================================
        if self.randomize_action_delay:
            self._action_delay_steps = rng.randint(0, 4)  # [0, 3] inclusive
            self._action_buffer = deque(maxlen=max(self._action_delay_steps + 1, 1))
            # Pre-fill buffer with zero actions (hover)
            for _ in range(self._action_delay_steps):
                self._action_buffer.append(np.zeros(3, dtype=np.float32))
            self.randomization_log["action_delay_steps"] = self._action_delay_steps
        else:
            self._action_delay_steps = 0
            self._action_buffer = deque(maxlen=1)

        # =====================================================================
        # GRAVITY RANDOMIZATION: [9.75, 9.85] m/s^2
        # Why: Gravity varies with altitude and latitude.
        #      Sea level equator: 9.780 m/s^2
        #      Sea level poles: 9.832 m/s^2
        #      At 1000m altitude: ~9.807 m/s^2
        #      Small effect, but included for completeness.
        #      Also helps the policy be robust to systematic force biases.
        # =====================================================================
        if self.randomize_gravity:
            gravity = rng.uniform(9.75, 9.85)
            self.env.unwrapped.gravity = gravity
            self.randomization_log["gravity"] = gravity

        # Save to history for later analysis
        self.randomization_history.append(self.randomization_log.copy())

        # Reset the base environment with the new parameters
        obs, info = self.env.reset(seed=seed, options=options)

        # Add noise to initial observation
        if self._obs_noise_std > 0:
            noise = np.random.normal(0, self._obs_noise_std, size=obs.shape)
            obs = obs + noise.astype(np.float32)

        # Include randomization log in info for debugging/analysis
        info["domain_randomization"] = self.randomization_log

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step with domain randomization effects applied.

        Applies action delay and observation noise on top of the base
        environment's step function.

        Args:
            action: The action from the RL policy.

        Returns:
            Tuple of (obs, reward, terminated, truncated, info) with
            randomization effects applied.
        """
        # Apply action delay: buffer the current action and use a delayed one
        # This simulates the real-world lag between deciding and executing
        if self._action_delay_steps > 0:
            self._action_buffer.append(action.copy())
            # Use the oldest action in the buffer (the delayed one)
            delayed_action = self._action_buffer[0]
        else:
            delayed_action = action

        # Step the base environment with the (possibly delayed) action
        obs, reward, terminated, truncated, info = self.env.step(delayed_action)

        # Add observation noise
        # This simulates sensor measurement error on every reading
        if self._obs_noise_std > 0:
            noise = np.random.normal(0, self._obs_noise_std, size=obs.shape)
            obs = obs + noise.astype(np.float32)

        return obs, reward, terminated, truncated, info

    def get_randomization_summary(self) -> Dict[str, Any]:
        """
        Get statistics over all episodes' randomization parameters.

        Useful for verifying that the randomization is covering the
        intended range and for analyzing which parameter settings
        correlate with success/failure.

        Returns:
            Dict with mean, std, min, max for each randomized parameter.
        """
        if not self.randomization_history:
            return {}

        summary = {}
        # Get all parameter names from the first entry
        param_names = self.randomization_history[0].keys()

        for param in param_names:
            values = [
                ep[param] for ep in self.randomization_history
                if param in ep and isinstance(ep[param], (int, float))
            ]
            if values:
                summary[param] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "n_episodes": len(values),
                }

        return summary


if __name__ == "__main__":
    from core.drone_env import DroneInterceptionEnv

    print("=" * 60)
    print("Domain Randomization Wrapper — Sanity Test")
    print("=" * 60)

    base_env = DroneInterceptionEnv(render_mode=None)
    env = DomainRandomizationWrapper(base_env)

    # Run 5 episodes with randomization
    for ep in range(5):
        obs, info = env.reset(seed=ep)
        rand_log = info.get("domain_randomization", {})
        print(f"\nEpisode {ep + 1} randomization:")
        for param, value in rand_log.items():
            print(f"  {param:25s}: {value:.4f}" if isinstance(value, float)
                  else f"  {param:25s}: {value}")

        # Run a few steps
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

    # Print summary statistics
    summary = env.get_randomization_summary()
    print("\n\nRandomization Summary (over 5 episodes):")
    for param, stats in summary.items():
        print(f"  {param:25s}: mean={stats['mean']:.3f}, "
              f"std={stats['std']:.3f}, "
              f"range=[{stats['min']:.3f}, {stats['max']:.3f}]")

    print("\n✓ Domain randomization test passed!")
    env.close()
