"""
Custom Stable-Baselines3 Callbacks for Drone Interception Training.

Callbacks hook into the SB3 training loop to:
1. Track per-episode metrics (interception rate, collision rate, etc.)
2. Print periodic summaries during training
3. Save metrics to JSON for later analysis and visualization

These are essential for monitoring training progress and comparing algorithms.
Without them, you'd only see the raw reward curve — which doesn't tell you
whether the agent is actually catching targets or just avoiding penalties.
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional
from stable_baselines3.common.callbacks import BaseCallback


class InterceptionTrackerCallback(BaseCallback):
    """
    Tracks interception-specific metrics during SB3 training.

    Standard SB3 logging only tracks reward and episode length.
    For our drone interception task, we need to know:
    - What % of episodes end in interception (the goal)?
    - What % end in collision (drone destroyed = $300 lost)?
    - What % time out (mission failed, target escaped)?
    - How much energy was used (proxy for battery cost)?
    - What's the estimated $/interception?

    These metrics are logged every `log_interval` steps and saved
    to a JSON file after training for visualization.

    Usage:
        callback = InterceptionTrackerCallback(log_interval=10000)
        model.learn(total_timesteps=500000, callback=callback)
        callback.save_metrics("results/ppo_metrics.json")
    """

    def __init__(
        self,
        log_interval: int = 10000,
        verbose: int = 1,
    ) -> None:
        """
        Initialize the interception tracker.

        Args:
            log_interval: Print summary every N timesteps.
            verbose: Verbosity level (0=silent, 1=summaries).
        """
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
            print("\n" + "=" * 70)
            print("  Interception Tracker — Training Started")
            print("=" * 70)

    def _on_step(self) -> bool:
        """
        Called at every training step. Check for episode completions.

        We look at the 'infos' from the vectorized environment to detect
        episode endings and extract our custom metrics.

        Returns:
            True (always continue training).
        """
        # Check if any environments finished an episode this step
        # In vectorized envs, 'dones' tells us which sub-envs reset
        infos = self.locals.get("infos", [])

        for info in infos:
            # SB3's Monitor wrapper stores final info in 'terminal_observation'
            # but our custom info keys are passed through
            if "intercepted" in info:
                # Only log if this is a terminal step (episode just ended)
                # Check for episode info from Monitor wrapper
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
        """Print a formatted summary of recent training performance."""
        if not self.current_window:
            return

        window = self.current_window
        n = len(window)

        # Compute rates
        intercept_rate = sum(1 for e in window if e["intercepted"]) / n * 100
        collision_rate = sum(1 for e in window if e["collision"]) / n * 100
        timeout_rate = sum(1 for e in window if e["timeout"]) / n * 100
        oob_rate = sum(1 for e in window if e["out_of_bounds"]) / n * 100
        avg_reward = np.mean([e["reward"] for e in window])
        avg_length = np.mean([e["length"] for e in window])
        avg_energy = np.mean([e["energy"] for e in window])

        # Estimate $/interception
        # Approximate: $300 base + $5 energy, divided by success rate
        success_rate = max(intercept_rate / 100, 0.01)
        cost_per_intercept = (300 + 5) / success_rate

        # Save to history
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
                f"Intercept: {intercept_rate:5.1f}% | "
                f"Collision: {collision_rate:5.1f}% | "
                f"Timeout: {timeout_rate:5.1f}% | "
                f"Avg Reward: {avg_reward:7.1f} | "
                f"Avg Length: {avg_length:5.0f} | "
                f"$/Intercept: ${cost_per_intercept:,.0f}"
            )

        # Reset window for next interval
        self.current_window = []

    def _on_training_end(self) -> None:
        """Called at the end of training — print final summary."""
        # Print any remaining episodes
        if self.current_window:
            self._print_summary()

        elapsed = time.time() - self.training_start_time
        total_episodes = len(self.episode_outcomes)

        if self.verbose and total_episodes > 0:
            print("\n" + "=" * 70)
            print("  Training Complete — Final Summary")
            print("=" * 70)
            print(f"  Total timesteps:  {self.num_timesteps:,}")
            print(f"  Total episodes:   {total_episodes:,}")
            print(f"  Training time:    {elapsed:.1f}s ({elapsed/60:.1f} min)")

            # Overall rates
            intercept = sum(1 for e in self.episode_outcomes if e["intercepted"])
            collision = sum(1 for e in self.episode_outcomes if e["collision"])
            timeout = sum(1 for e in self.episode_outcomes if e["timeout"])
            oob = sum(1 for e in self.episode_outcomes if e["out_of_bounds"])

            print(f"\n  Overall Interception Rate: {intercept/total_episodes*100:.1f}%")
            print(f"  Overall Collision Rate:    {collision/total_episodes*100:.1f}%")
            print(f"  Overall Timeout Rate:      {timeout/total_episodes*100:.1f}%")
            print(f"  Overall OOB Rate:          {oob/total_episodes*100:.1f}%")

            # Final 100 episodes (most representative of trained policy)
            recent = self.episode_outcomes[-min(100, total_episodes):]
            recent_intercept = sum(1 for e in recent if e["intercepted"]) / len(recent)
            print(f"\n  Last {len(recent)} Episodes Interception Rate: "
                  f"{recent_intercept*100:.1f}%")
            print(f"  Estimated $/Interception: "
                  f"${305/max(recent_intercept, 0.01):,.0f}")
            print("=" * 70)

    def save_metrics(self, filepath: str) -> None:
        """
        Save all collected metrics to a JSON file.

        Args:
            filepath: Path to save the JSON metrics file.
        """
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
        """
        Get a summary dict of final training metrics.

        Returns:
            Dict with key performance metrics from the last 100 episodes.
        """
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
