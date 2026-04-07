"""Custom Stable-Baselines3 callbacks for tracking training metrics."""

import os
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional
from stable_baselines3.common.callbacks import BaseCallback


class InterceptionTrackerCallback(BaseCallback):
    """Track per-episode metrics during training.
    
    Logs: interception rate, collision rate, timeout rate, energy, reward.
    Saves metrics every log_interval steps. Computes cost per interception
    for training economics ($300 base + $5 energy / success rate).
    """

    def __init__(self, log_interval: int = 10000, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.log_interval = log_interval

        # Per-episode tracking
        self.episode_outcomes: List[Dict[str, Any]] = []
        self.current_window: List[Dict[str, Any]] = []

        # Aggregate metrics over training
        self.metrics_history: List[Dict[str, Any]] = []
        self.training_start_time: float = 0.0

        # Running counters since last log
        self._episodes_since_log = 0
        self._last_log_step = 0

    def _on_training_start(self) -> None:
        """Called at the very beginning of training."""
        self.training_start_time = time.time()
        if self.verbose:
            print(f"Interception Tracker started")

    def _on_step(self) -> bool:
        """Check for episode completions and track metrics."""
        # Check if any environments finished an episode this step
        infos = self.locals.get("infos", [])

        for info in infos:
            # SB3's Monitor wrapper stores final info in 'terminal_observation'
            if "intercepted" in info:
                # Extract episode metrics from Monitor wrapper
                ep_info = info.get("episode", None)
                if ep_info is not None:
                    outcome = {
                        "intercepted": info.get("intercepted", False),
                        "collision": info.get("collision", False),
                        "timeout": info.get("timeout", False),
                        "out_of_bounds": info.get("out_of_bounds", False),
                        "reward": ep_info.get("r", 0.0),
                        "length": ep_info.get("l", 0),
                        "energy": info.get("energy_used", 0.0),
                        "distance": info.get("distance", 0.0),
                        "timestep": self.num_timesteps,
                    }
                    self.episode_outcomes.append(outcome)
                    self.current_window.append(outcome)
                    self._episodes_since_log += 1

        # Print summary at regular intervals
        if (self.num_timesteps - self._last_log_step) >= self.log_interval:
            self._print_summary()
            self._last_log_step = self.num_timesteps

        return True

    def _print_summary(self) -> None:
        """Print metrics from the recent episode window."""
        if not self.current_window:
            return

        window = self.current_window
        n = len(window)

        # Compute episode outcome statistics
        intercept_rate = sum(1 for e in window if e["intercepted"]) / n * 100
        collision_rate = sum(1 for e in window if e["collision"]) / n * 100
        timeout_rate = sum(1 for e in window if e["timeout"]) / n * 100
        oob_rate = sum(1 for e in window if e["out_of_bounds"]) / n * 100
        avg_reward = np.mean([e["reward"] for e in window])
        avg_length = np.mean([e["length"] for e in window])
        avg_energy = np.mean([e["energy"] for e in window])

        # Cost per interception: total training expense / success rate
        success_rate = max(intercept_rate / 100, 0.01)
        cost_per_intercept = (300 + 5) / success_rate

        # Accumulate metrics for monitoring
        metrics = {
            "timestep": self.num_timesteps,
            "episodes": n,
            "intercept_rate": intercept_rate,
            "collision_rate": collision_rate,
            "timeout_rate": timeout_rate,
            "oob_rate": oob_rate,
            "avg_reward": float(avg_reward),
            "avg_length": float(avg_length),
            "avg_energy": float(avg_energy),
            "cost_per_interception": cost_per_intercept,
            "elapsed_time": time.time() - self.training_start_time,
        }
        self.metrics_history.append(metrics)

        if self.verbose:
            print(
                f"[Step {self.num_timesteps:>7d}] "
                f"Intercept: {intercept_rate:5.1f}% | Collision: {collision_rate:5.1f}% | "
                f"Timeout: {timeout_rate:5.1f}% | Reward: {avg_reward:7.1f} | "
                f"Length: {avg_length:5.0f} | $/Intercept: ${cost_per_intercept:,.0f}"
            )

        # Reset for next window
        self.current_window = []

    def _on_training_end(self) -> None:
        """Print final training summary."""
        # Print any remaining episodes
        if self.current_window:
            self._print_summary()

        elapsed = time.time() - self.training_start_time
        total_episodes = len(self.episode_outcomes)

        if self.verbose and total_episodes > 0:
            print(f"Training complete: {self.num_timesteps:,} steps, {total_episodes:,} episodes, {elapsed/60:.1f} min")

            # Episode outcome breakdown
            intercept = sum(1 for e in self.episode_outcomes if e["intercepted"])
            collision = sum(1 for e in self.episode_outcomes if e["collision"])
            timeout = sum(1 for e in self.episode_outcomes if e["timeout"])

            print(f"  Intercept: {intercept/total_episodes*100:.1f}% | "
                  f"Collision: {collision/total_episodes*100:.1f}% | "
                  f"Timeout: {timeout/total_episodes*100:.1f}%")

            recent = self.episode_outcomes[-min(100, total_episodes):]
            recent_intercept = sum(1 for e in recent if e["intercepted"]) / len(recent)
            print(f"  Last 100: {recent_intercept*100:.1f}% | "
                  f"Est. $/Intercept: ${305/max(recent_intercept, 0.01):,.0f}")

    def save_metrics(self, filepath: str) -> None:
        """Write metrics to disk for analysis."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

        data = {
            "metrics_history": self.metrics_history,
            "episode_outcomes": self.episode_outcomes,
            "total_episodes": len(self.episode_outcomes),
            "total_timesteps": self.num_timesteps,
            "training_time": time.time() - self.training_start_time,
        }

        try:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)
            if self.verbose:
                print(f"\n  Metrics saved to: {filepath}")
        except IOError as e:
            print(f"  Warning: Could not save metrics to {filepath}: {e}")

    def get_final_metrics(self) -> Dict[str, Any]:
        """Extract final metrics from last 100 episodes."""
        if not self.episode_outcomes:
            return {}

        recent = self.episode_outcomes[-min(100, len(self.episode_outcomes)):]
        n = len(recent)

        intercept_rate = sum(1 for e in recent if e["intercepted"]) / n
        collision_rate = sum(1 for e in recent if e["collision"]) / n
        timeout_rate = sum(1 for e in recent if e["timeout"]) / n
        avg_reward = np.mean([e["reward"] for e in recent])
        avg_length = np.mean([e["length"] for e in recent])
        avg_energy = np.mean([e["energy"] for e in recent])

        return {
            "intercept_rate": intercept_rate,
            "collision_rate": collision_rate,
            "timeout_rate": timeout_rate,
            "avg_reward": float(avg_reward),
            "avg_length": float(avg_length),
            "avg_energy": float(avg_energy),
            "cost_per_interception": 305 / max(intercept_rate, 0.01),
            "total_episodes": len(self.episode_outcomes),
            "total_timesteps": self.num_timesteps,
        }
