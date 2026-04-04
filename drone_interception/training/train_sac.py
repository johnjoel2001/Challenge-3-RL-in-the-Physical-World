"""
Train a SAC (Soft Actor-Critic) agent for drone interception.

SAC is an off-policy algorithm that maximizes both expected reward AND
entropy (randomness) of the policy. This has two key advantages:

1. EXPLORATION: The entropy bonus prevents the policy from collapsing to
   a single deterministic strategy too early. For drone interception, this
   means SAC explores diverse pursuit trajectories before settling on the best.

2. ROBUSTNESS: A stochastic policy is inherently more robust to perturbations.
   When transferred to a real drone (sim2real), small sensor errors or physics
   mismatches are less likely to cause catastrophic failures.

SAC vs PPO tradeoffs for this task:
- SAC is more sample-efficient (off-policy replay buffer reuses old experience)
- SAC can be less stable with shaped rewards (value function can overfit to buffer)
- SAC trains with 1 env (off-policy doesn't benefit as much from parallel envs)
- SAC's automatic entropy tuning is convenient (no manual ent_coef tuning)

Usage:
    python -m training.train_sac --timesteps 500000 --seed 42
    python -m training.train_sac --timesteps 500000 --domain-rand
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.drone_env import DroneInterceptionEnv
from core.domain_randomization import DomainRandomizationWrapper
from training.callbacks import InterceptionTrackerCallback


def make_env(
    seed: int = 0,
    use_domain_rand: bool = False,
):
    """
    Create a single environment instance for SAC training.

    SAC uses 1 environment (off-policy algorithms don't need parallel envs
    because they can replay old transitions from the buffer).

    Args:
        seed: Random seed.
        use_domain_rand: Whether to wrap with domain randomization.

    Returns:
        Wrapped Gymnasium environment.
    """
    env = DroneInterceptionEnv(render_mode=None)
    if use_domain_rand:
        env = DomainRandomizationWrapper(env)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def train_sac(
    total_timesteps: int = 500_000,
    seed: int = 42,
    use_domain_rand: bool = False,
) -> None:
    """
    Train a SAC agent on the drone interception environment.

    SAC Hyperparameters:
    - learning_rate=3e-4: Standard for continuous control with SAC.
    - buffer_size=100000: Replay buffer holds 100K transitions.
      At ~200 steps/episode, this is ~500 episodes of experience.
      Larger buffers are more stable but use more RAM.
    - batch_size=256: Larger batches than PPO because off-policy updates
      can use more data per gradient step without going stale.
    - gamma=0.99: Same discount as PPO — high to propagate interception
      bonus backward through the episode.
    - tau=0.005: Soft target network update rate. Low = stable but slow.
    - ent_coef="auto": SAC's killer feature — automatically tunes the
      entropy coefficient during training. Starts high (more exploration)
      and decreases as the policy improves.
    - net_arch=[256, 256]: Same architecture as PPO for fair comparison.
      Also ensures the trained policy can run on edge hardware.

    Args:
        total_timesteps: Total environment steps for training.
        seed: Random seed for reproducibility.
        use_domain_rand: Whether to use domain randomization.
    """
    print("\n" + "=" * 70)
    print("  SAC Training — Low-Cost Drone Interception")
    print("=" * 70)
    print(f"  Timesteps:          {total_timesteps:,}")
    print(f"  Seed:               {seed}")
    print(f"  Domain Rand:        {use_domain_rand}")
    print("=" * 70)

    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create single environment (SAC is off-policy — 1 env is sufficient)
    env = make_env(seed, use_domain_rand)

    # Create SAC model
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100_000,     # Replay buffer size
        batch_size=256,          # Mini-batch size for updates
        gamma=0.99,              # Discount factor
        tau=0.005,               # Soft target update coefficient
        ent_coef="auto",         # Automatic entropy tuning
        learning_starts=1000,    # Random actions for first 1K steps (exploration)
        policy_kwargs=dict(
            net_arch=[256, 256],  # Same architecture as PPO
        ),
        verbose=1,
        tensorboard_log="./logs/sac/",
        seed=seed,
    )

    # Create callback
    callback = InterceptionTrackerCallback(log_interval=10000, verbose=1)

    # Train!
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )
    elapsed = time.time() - start_time

    # Save model and metrics
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./results", exist_ok=True)

    model_path = "./models/sac_interceptor"
    model.save(model_path)
    print(f"\n  Model saved to: {model_path}.zip")

    metrics_path = "./results/sac_metrics.json"
    callback.save_metrics(metrics_path)

    print(f"\n  Total training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    final = callback.get_final_metrics()
    if final:
        print(f"  Final interception rate: {final['intercept_rate']*100:.1f}%")
        print(f"  Estimated $/interception: ${final['cost_per_interception']:,.0f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SAC agent for drone interception"
    )
    parser.add_argument(
        "--timesteps", type=int, default=500_000,
        help="Total training timesteps (default: 500000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--domain-rand", action="store_true",
        help="Enable domain randomization for sim2real transfer"
    )
    args = parser.parse_args()

    train_sac(
        total_timesteps=args.timesteps,
        seed=args.seed,
        use_domain_rand=args.domain_rand,
    )
