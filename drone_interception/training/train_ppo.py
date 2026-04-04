"""
Train a PPO (Proximal Policy Optimization) agent for drone interception.

PPO is our primary algorithm choice because:
1. STABLE: Clipped objective prevents catastrophically large policy updates.
   This is crucial — we can't afford training instability when each training
   run takes 20+ minutes.
2. SAMPLE EFFICIENT ENOUGH: With 4 parallel envs and 2048-step rollouts,
   PPO gets good signal from each batch of experience.
3. ON-POLICY: Doesn't need a replay buffer, so memory footprint is small.
   Important for running on student laptops.
4. PROVEN: PPO is the workhorse of modern RL — OpenAI Five (Dota 2),
   OpenAI's robotics work, and most Gymnasium benchmarks use PPO.

For drone interception specifically, PPO's conservative updates help because
the reward landscape has sharp cliffs (interception bonus, collision penalty).
Off-policy methods like SAC/TD3 can be destabilized by these.

Usage:
    python -m training.train_ppo --timesteps 500000 --seed 42
    python -m training.train_ppo --timesteps 500000 --domain-rand
"""

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


def make_env(
    seed: int = 0,
    use_domain_rand: bool = False,
    rank: int = 0,
):
    """
    Factory function for creating a single environment instance.

    Used with DummyVecEnv to create multiple parallel environments.
    Each env gets a unique seed (base_seed + rank) for diversity.

    Args:
        seed: Base random seed.
        use_domain_rand: Whether to wrap with domain randomization.
        rank: Sub-environment index (for seed offsetting).

    Returns:
        Callable that creates and returns an environment.
    """
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
    """
    Train a PPO agent on the drone interception environment.

    PPO Hyperparameters (tuned for this task):
    - learning_rate=3e-4: Standard for continuous control. Lower might be more
      stable but slower; higher risks instability with our shaped reward.
    - n_steps=2048: Rollout length per env. 2048 * 4 envs = 8192 transitions
      per update — enough to see several full episodes per batch.
    - batch_size=64: Mini-batch size for SGD updates within each epoch.
    - n_epochs=10: Number of passes over the rollout data. PPO's clipping
      prevents the policy from changing too much even with 10 passes.
    - gamma=0.99: Discount factor. High because interception bonus at episode
      end should influence actions taken many steps earlier.
    - gae_lambda=0.95: GAE parameter for advantage estimation. Balances
      bias vs variance in the advantage function.
    - clip_range=0.2: PPO's signature — prevents the policy ratio from
      going beyond [0.8, 1.2], ensuring small, stable updates.
    - ent_coef=0.01: Entropy bonus encourages exploration. Small but nonzero
      to prevent premature convergence to suboptimal strategies.
    - vf_coef=0.5: Value function loss weight. Standard value.
    - net_arch=[256, 256]: Two hidden layers of 256 units each.
      Small enough to run inference on a Jetson Nano (~10ms forward pass).

    Args:
        total_timesteps: Total environment steps for training.
        seed: Random seed for reproducibility.
        use_domain_rand: Whether to use domain randomization.
        n_envs: Number of parallel environments.
    """
    print("\n" + "=" * 70)
    print("  PPO Training — Low-Cost Drone Interception")
    print("=" * 70)
    print(f"  Timesteps:          {total_timesteps:,}")
    print(f"  Seed:               {seed}")
    print(f"  Domain Rand:        {use_domain_rand}")
    print(f"  Parallel Envs:      {n_envs}")
    print("=" * 70)

    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create vectorized environment (4 parallel envs for PPO)
    # PPO benefits from parallel envs because it's on-policy:
    # more envs = more diverse rollouts per update = better gradient estimates
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
        policy_kwargs=dict(
            net_arch=[256, 256],  # Two 256-unit hidden layers
        ),
        verbose=1,
        tensorboard_log="./logs/ppo/",
        seed=seed,
    )

    # Create our custom callback for tracking interception metrics
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

    model_path = "./models/ppo_interceptor"
    model.save(model_path)
    print(f"\n  Model saved to: {model_path}.zip")

    metrics_path = "./results/ppo_metrics.json"
    callback.save_metrics(metrics_path)

    # Print final summary
    print(f"\n  Total training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
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
