"""Train PPO agent for drone interception."""

import os
import sys
import time
import argparse
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.drone_env import DroneInterceptionEnv
from core.domain_randomization import DomainRandomizationWrapper
from training.callbacks import InterceptionTrackerCallback


def make_env(seed: int = 0, use_domain_rand: bool = False, rank: int = 0):
    def _init():
        env = DroneInterceptionEnv(render_mode=None)
        if use_domain_rand:
            env = DomainRandomizationWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_ppo(
    total_timesteps: int = 500_000,
    seed: int = 42,
    use_domain_rand: bool = False,
    n_envs: int = 4,
) -> None:
    """Train PPO agent.
    
    PPO chosen for stability with curved reward landscape (interception bonus,
    collision penalty). Uses 4 parallel envs (2048 steps each = 8192 transitions
    per update). Hyperparameters: lr=3e-4, clip_range=0.2, entropy=0.01.
    Network: [256, 256] for edge deployment (Jetson Nano, ~10ms inference).
    Cost model: $300 drone + $5 energy per episode.
    """
    print(f"PPO Training — {total_timesteps:,} steps, seed={seed}, domain_rand={use_domain_rand}")

    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create vectorized environment
    env_fns = [make_env(seed, use_domain_rand, rank=i) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,           # Steps per env before update
        batch_size=64,          # Mini-batch size for SGD
        n_epochs=10,            # Epochs per rollout
        gamma=0.99,             # Discount factor
        gae_lambda=0.95,        # GAE lambda
        clip_range=0.2,         # PPO clipping parameter
        ent_coef=0.01,          # Entropy coefficient (exploration)
        vf_coef=0.5,            # Value function coefficient
        max_grad_norm=0.5,      # Gradient clipping
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log="./logs/ppo/",
        seed=seed,
    )

    # Create custom callback for interception tracking
    callback = InterceptionTrackerCallback(log_interval=10000, verbose=1)

    # Train
    start_time = time.time()  
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    elapsed = time.time() - start_time

    # Save model and metrics
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./results", exist_ok=True)

    model_path = "./models/ppo_interceptor"
    model.save(model_path)
    print(f"Model saved: {model_path}.zip")

    metrics_path = "./results/ppo_metrics.json"
    callback.save_metrics(metrics_path)
    print(f"Training time: {elapsed/60:.1f} min")
    final = callback.get_final_metrics()
    if final:
        print(f"  Final interception rate: {final['intercept_rate']*100:.1f}%")
        print(f"  Estimated $/interception: ${final['cost_per_interception']:,.0f}")

    vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO agent for drone interception"
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
    parser.add_argument(
        "--n-envs", type=int, default=4,
        help="Number of parallel environments (default: 4)"
    )
    args = parser.parse_args()

    train_ppo(
        total_timesteps=args.timesteps,
        seed=args.seed,
        use_domain_rand=args.domain_rand,
        n_envs=args.n_envs,
    )
