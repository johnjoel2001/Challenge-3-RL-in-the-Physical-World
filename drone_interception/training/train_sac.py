"""Train SAC agent for drone interception."""

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


def make_env(seed: int = 0, use_domain_rand: bool = False):
    """Create environment for SAC training."""
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
    """Train SAC agent.
    
    Off-policy algorithm with entropy bonus for exploration. Single environment
    with 100K replay buffer. Automatic entropy tuning. Hyperparameters:
    lr=3e-4, batch_size=256, tau=0.005. Network: [256, 256].
    Cost model: $300 drone + $5 energy per episode.
    """
    print(f"SAC Training — {total_timesteps:,} steps, seed={seed}, domain_rand={use_domain_rand}")

    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create single environment
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
        ent_coef="auto",
        learning_starts=1000,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log="./logs/sac/",
        seed=seed,
    )

    # Create callback
    callback = InterceptionTrackerCallback(log_interval=10000, verbose=1)

    # Train
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    elapsed = time.time() - start_time

    # Save model and metrics
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./results", exist_ok=True)

    model_path = "./models/sac_interceptor"
    model.save(model_path)
    print(f"Model saved: {model_path}.zip")

    metrics_path = "./results/sac_metrics.json"
    callback.save_metrics(metrics_path)
    print(f"Training time: {elapsed/60:.1f} min")
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
