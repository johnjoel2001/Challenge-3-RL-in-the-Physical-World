"""Train TD3 agent for drone interception."""

import os
import sys
import time
import argparse
import numpy as np
import torch

from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.drone_env import DroneInterceptionEnv
from core.domain_randomization import DomainRandomizationWrapper
from training.callbacks import InterceptionTrackerCallback


def make_env(seed: int = 0, use_domain_rand: bool = False):
    """Create environment for TD3 training."""
    env = DroneInterceptionEnv(render_mode=None)
    if use_domain_rand:
        env = DomainRandomizationWrapper(env)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def train_td3(
    total_timesteps: int = 500_000,
    seed: int = 42,
    use_domain_rand: bool = False,
) -> None:
    """Train TD3 agent.  
    
    Deterministic off-policy algorithm with twin critics and delayed policy
    updates. Single environment with 100K replay buffer and Gaussian exploration
    noise (σ=0.2). Policy updates every 2 critic updates. Hyperparameters:
    lr=3e-4, batch_size=256, tau=0.005, policy_delay=2. Network: [256, 256].
    Cost model: $300 drone + $5 energy per episode.
    """
    print(f"TD3 Training — {total_timesteps:,} steps, seed={seed}, domain_rand={use_domain_rand}")

    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create single environment
    env = make_env(seed, use_domain_rand)

    # TD3 needs explicit exploration noise because its policy is deterministic
    # Without this, the agent would always take the same action in the same state
    # σ=0.1 provides moderate exploration without being too disruptive
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions),
    )

    # Create TD3 model
    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100_000,       # Replay buffer size
        batch_size=256,            # Mini-batch size
        gamma=0.99,                # Discount factor
        tau=0.005,                 # Soft target update
        policy_noise=0.2,          # Target policy smoothing noise
        noise_clip=0.5,            # Clip target noise
        policy_delay=2,            # Delayed policy updates (TD3's key trick)
        action_noise=action_noise, # Exploration noise
        learning_starts=1000,      # Random actions for first 1K steps
        policy_kwargs=dict(
            net_arch=[256, 256],    # Same architecture as PPO/SAC
        ),
        verbose=1,
        tensorboard_log="./logs/td3/",
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

    model_path = "./models/td3_interceptor"
    model.save(model_path)
    print(f"Model saved: {model_path}.zip")

    metrics_path = "./results/td3_metrics.json"
    callback.save_metrics(metrics_path)
    print(f"Training time: {elapsed/60:.1f} min")
    final = callback.get_final_metrics()
    if final:
        print(f"  Final interception rate: {final['intercept_rate']*100:.1f}%")
        print(f"  Estimated $/interception: ${final['cost_per_interception']:,.0f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TD3 agent for drone interception"
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

    train_td3(
        total_timesteps=args.timesteps,
        seed=args.seed,
        use_domain_rand=args.domain_rand,
    )
