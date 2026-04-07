"""Domain randomization for sim2real transfer.

Randomizes physics parameters across episodes so trained policies work on
real hardware. Reference: Tobin et al. (2017); OpenAI's Rubik's Cube work.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple
from collections import deque
from core.drone_env import DroneInterceptionEnv


class DomainRandomizationWrapper(gym.Wrapper):
    """Randomize physics parameters at episode reset for sim2real robustness.
    
    Variability: manufacturing tolerance, battery degradation, wind, sensor noise, latency.
    Training across this range helps policies transfer to real hardware.
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
        """Initialize wrapper. Each randomization can be toggled independently for ablations."""
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
        """Sample new parameters and reset environment; every episode is different."""
        # Use numpy random for parameter sampling
        rng = np.random.RandomState(seed)

        self.randomization_log = {}

        # Mass: manufacturing tolerance, payload variation
        if self.randomize_mass:
            mass = rng.uniform(0.7, 1.5)
            self.env.unwrapped.drone_mass = mass
            self.randomization_log["drone_mass"] = mass

        # Max force: motor wear, battery voltage sag (~15-20% across flight)
        if self.randomize_force:
            force = rng.uniform(3.5, 7.0)
            self.env.unwrapped.max_force = force
            self.randomization_log["max_force"] = force

        # Drag: wind, air density changes
        if self.randomize_drag:
            drag = rng.uniform(0.1, 0.6)
            self.env.unwrapped.drag_coeff = drag
            self.randomization_log["drag_coefficient"] = drag

        # Evader speed: different drone types have different speeds
        if self.randomize_evader:
            speed = rng.uniform(1.0, 3.5)
            self.env.unwrapped.evader_speed = speed
            self.randomization_log["evader_speed"] = speed

        # Obstacle count: environmental complexity varies
        if self.randomize_obstacles:
            num_obs = rng.randint(2, 9)  # [2, 8] inclusive
            self.env.unwrapped.num_obstacles = int(num_obs)
            self.randomization_log["num_obstacles"] = int(num_obs)

        # Sensor noise: measurement error
        if self.randomize_obs_noise:
            self._obs_noise_std = rng.uniform(0, 0.05)
            self.randomization_log["obs_noise_std"] = self._obs_noise_std
        else:
            self._obs_noise_std = 0.0

        # Action delay: motor lag, compute latency
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

        # Gravity: variation with altitude/latitude
        if self.randomize_gravity:
            gravity = rng.uniform(9.75, 9.85)
            self.env.unwrapped.gravity = gravity
            self.randomization_log["gravity"] = gravity

        # Save to history for later analysis
        self.randomization_history.append(self.randomization_log.copy())

        # Reset the base environment with the new parameters
        obs, info = self.env.reset(seed=seed, options=options)

        # Add noise to initial observation to model sensor error upfront
        if self._obs_noise_std > 0:
            noise = np.random.normal(0, self._obs_noise_std, size=obs.shape)
            obs = obs + noise.astype(np.float32)

        # Include randomization log in info for debugging/analysis
        info["domain_randomization"] = self.randomization_log

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Apply action delay and observation noise on every step."""
        # Action delay: buffer incoming action and use a delayed one (simulates hardware latency)
        if self._action_delay_steps > 0:
            self._action_buffer.append(action.copy())
            # Use the oldest action in the buffer
            delayed_action = self._action_buffer[0]
        else:
            delayed_action = action

        # Execute step with the (possibly delayed) action
        obs, reward, terminated, truncated, info = self.env.step(delayed_action)

        # Observation noise: on every reading to model persistent sensor error
        if self._obs_noise_std > 0:
            noise = np.random.normal(0, self._obs_noise_std, size=obs.shape)
            obs = obs + noise.astype(np.float32)

        return obs, reward, terminated, truncated, info

    def get_randomization_summary(self) -> Dict[str, Any]:
        """Return statistics over all episodes; useful for verifying coverage and correlating with outcomes."""
        if not self.randomization_history:
            return {}

        summary = {}
        # Extract all parameter names from the first episode
        param_names = self.randomization_history[0].keys()

        for param in param_names:
            # Collect numeric values across all episodes
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
    base_env = DroneInterceptionEnv(render_mode=None)
    env = DomainRandomizationWrapper(base_env)

    for ep in range(5):
        obs, info = env.reset(seed=ep)
        rand_log = info.get("domain_randomization", {})
        print(f"Episode {ep + 1}:")
        for param, value in rand_log.items():
            fmt = f"{value:.4f}" if isinstance(value, float) else f"{value}"
            print(f"  {param}: {fmt}")

        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

    summary = env.get_randomization_summary()
    print("\nSummary (5 episodes):")
    for param, stats in summary.items():
        print(f"  {param}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
              f"range=[{stats['min']:.3f}, {stats['max']:.3f}]")

    env.close()
