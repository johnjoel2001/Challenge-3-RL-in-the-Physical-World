"""
Train a TD3 (Twin Delayed DDPG) agent for drone interception.

TD3 improves on DDPG with three key tricks:
1. TWIN CRITICS: Two Q-networks; take the minimum to reduce overestimation bias.
   This is critical for our task because the interception bonus (+100) can cause
   value overestimation that destabilizes training.
2. DELAYED POLICY UPDATES: Update the policy network less frequently than the
   critics (every 2 critic updates). This lets the value estimates stabilize
   before using them to improve the policy.
3. TARGET POLICY SMOOTHING: Add noise to target actions during critic updates.
   This prevents the policy from exploiting narrow peaks in the Q-function.

TD3 vs PPO vs SAC for drone interception:
- TD3 is deterministic (no entropy bonus) → potentially more precise trajectories
- TD3 needs careful exploration noise setup (policy_noise, noise_clip)
- TD3 can be the most sample-efficient but also the most brittle
- TD3's deterministic policy is ideal for deployment (no sampling at inference time)

Usage:
    python -m training.train_td3 --timesteps 500000 --seed 42
    python -m training.train_td3 --timesteps 500000 --domain-rand
"""

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


def make_env(
    seed: int = 0,
    use_domain_rand: bool = False,
):
    """
    Create a single environment instance for TD3 training.

    Like SAC, TD3 is off-policy and uses 1 environment with a replay buffer.

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


def train_td3(
    total_timesteps: int = 500_000,
    seed: int = 42,
    use_domain_rand: bool = False,
) -> None:
    """
    Train a TD3 agent on the drone interception environment.

    TD3 Hyperparameters:
    - learning_rate=3e-4: Same as PPO/SAC for fair comparison.
    - buffer_size=100000: Same replay buffer as SAC.
    - batch_size=256: Same as SAC for consistency.
    - gamma=0.99: High discount for long-horizon interception.
    - tau=0.005: Soft target update rate (same as SAC).
    - policy_noise=0.2: Gaussian noise added to target policy during critic
      updates. Smooths the Q-function landscape.
    - noise_clip=0.5: Clips the target policy noise to prevent extreme values.
    - policy_delay=2: Update policy every 2 critic updates.
      This is TD3's signature — lets Q-values stabilize first.
    - action_noise: Gaussian exploration noise (σ=0.1) added during data
      collection. TD3's policy is deterministic, so without this noise,
      it would never explore.

    Args:
        total_timesteps: Total environment steps for training.
        seed: Random seed for reproducibility.
        use_domain_rand: Whether to use domain randomization.
    """
    print("\n" + "=" * 70)
    print("  TD3 Training — Low-Cost Drone Interception")
    print("=" * 70)
    print(f"  Timesteps:          {total_timesteps:,}")
    print(f"  Seed:               {seed}")
    print(f"  Domain Rand:        {use_domain_rand}")
    print("=" * 70)

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

    model_path = "./models/td3_interceptor"
    model.save(model_path)
    print(f"\n  Model saved to: {model_path}.zip")

    metrics_path = "./results/td3_metrics.json"
    callback.save_metrics(metrics_path)

    print(f"\n  Total training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
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
