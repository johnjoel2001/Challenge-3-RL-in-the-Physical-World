"""
Evaluate a trained drone interception model over multiple episodes.

This script loads a trained model and runs it deterministically for N episodes,
computing comprehensive metrics including interception rate, collision rate,
average reward, episode length, energy usage, and estimated cost per interception.

These metrics are essential for:
1. Validating that the trained policy actually works (not just high reward)
2. Computing real-world cost estimates for the cost-effectiveness argument
3. Comparing different algorithms on the same evaluation protocol

Usage:
    python -m evaluation.evaluate --model models/ppo_interceptor.zip --episodes 100
    python -m evaluation.evaluate --model models/sac_interceptor.zip --episodes 100 --seed 123
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO, SAC, TD3
from core.drone_env import DroneInterceptionEnv
from core.cost_aware_reward import CostAwareReward


# Map algorithm names to their SB3 classes
ALGO_MAP = {
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
}


def detect_algorithm(model_path: str) -> str:
    """
    Detect which algorithm a saved model uses from its filename.

    Args:
        model_path: Path to the saved model .zip file.

    Returns:
        Algorithm name string ("ppo", "sac", or "td3").
    """
    basename = os.path.basename(model_path).lower()
    for algo_name in ALGO_MAP:
        if algo_name in basename:
            return algo_name
    # Default to PPO if we can't detect
    return "ppo"


def evaluate_model(
    model_path: str,
    n_episodes: int = 100,
    seed: int = 42,
    deterministic: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a trained model over multiple episodes.

    Runs the policy deterministically (no exploration noise) to get a clean
    assessment of its learned behavior. Collects per-episode data for
    statistical analysis.

    Args:
        model_path: Path to the saved SB3 model (.zip).
        n_episodes: Number of evaluation episodes.
        seed: Random seed for reproducible evaluation.
        deterministic: Whether to use deterministic actions (recommended for eval).
        verbose: Whether to print progress.

    Returns:
        Dict with aggregate metrics and per-episode data.
    """
    # Load the trained model
    algo_name = detect_algorithm(model_path)
    AlgoClass = ALGO_MAP[algo_name]

    if verbose:
        print(f"\n  Loading {algo_name.upper()} model from: {model_path}")

    model = AlgoClass.load(model_path)

    # Create evaluation environment (separate from training env)
    env = DroneInterceptionEnv(render_mode=None)
    cost_tracker = CostAwareReward()

    # Storage for per-episode results
    episodes: List[Dict[str, Any]] = []

    if verbose:
        print(f"  Running {n_episodes} evaluation episodes (deterministic={deterministic})...")

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        cost_tracker.reset()

        episode_reward = 0.0
        episode_steps = 0
        done = False
        prev_distance = info["distance"]

        while not done:
            # Get action from trained policy
            action, _ = model.predict(obs, deterministic=deterministic)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Track cost
            distance = info.get("distance", 0.0)
            cost_tracker.compute(action, distance, prev_distance, info)
            prev_distance = distance

            episode_reward += reward
            episode_steps += 1

        # Get cost summary for this episode
        cost_summary = cost_tracker.get_episode_summary()

        episode_data = {
            "episode": ep,
            "intercepted": info.get("intercepted", False),
            "collision": info.get("collision", False),
            "timeout": info.get("timeout", False),
            "out_of_bounds": info.get("out_of_bounds", False),
            "reward": episode_reward,
            "steps": episode_steps,
            "final_distance": info.get("distance", 0.0),
            "energy_used": info.get("energy_used", 0.0),
            "mission_cost": cost_summary["total_mission_cost"],
        }
        episodes.append(episode_data)

        # Print progress every 25 episodes
        if verbose and (ep + 1) % 25 == 0:
            recent_intercepts = sum(
                1 for e in episodes[-25:] if e["intercepted"]
            )
            print(f"    Episode {ep + 1:>4d}/{n_episodes}: "
                  f"Recent intercept rate: {recent_intercepts/25*100:.0f}%")

    env.close()

    # Compute aggregate metrics
    n = len(episodes)
    intercept_count = sum(1 for e in episodes if e["intercepted"])
    collision_count = sum(1 for e in episodes if e["collision"])
    timeout_count = sum(1 for e in episodes if e["timeout"])
    oob_count = sum(1 for e in episodes if e["out_of_bounds"])

    intercept_rate = intercept_count / n
    collision_rate = collision_count / n
    timeout_rate = timeout_count / n
    oob_rate = oob_count / n

    avg_reward = np.mean([e["reward"] for e in episodes])
    std_reward = np.std([e["reward"] for e in episodes])
    avg_steps = np.mean([e["steps"] for e in episodes])
    avg_energy = np.mean([e["energy_used"] for e in episodes])

    # Cost per interception: total cost across all missions / number of successes
    total_cost = sum(e["mission_cost"] for e in episodes)
    cost_per_interception = total_cost / max(intercept_count, 1)

    # Successful episode metrics (only episodes that intercepted)
    successful = [e for e in episodes if e["intercepted"]]
    avg_success_steps = np.mean([e["steps"] for e in successful]) if successful else 0
    avg_success_reward = np.mean([e["reward"] for e in successful]) if successful else 0

    results = {
        "algorithm": algo_name.upper(),
        "model_path": model_path,
        "n_episodes": n,
        "deterministic": deterministic,
        "seed": seed,
        "intercept_rate": intercept_rate,
        "collision_rate": collision_rate,
        "timeout_rate": timeout_rate,
        "oob_rate": oob_rate,
        "avg_reward": float(avg_reward),
        "std_reward": float(std_reward),
        "avg_steps": float(avg_steps),
        "avg_energy": float(avg_energy),
        "cost_per_interception": float(cost_per_interception),
        "avg_success_steps": float(avg_success_steps),
        "avg_success_reward": float(avg_success_reward),
        "episodes": episodes,
    }

    if verbose:
        print_results(results)

    return results


def print_results(results: Dict[str, Any]) -> None:
    """Print a formatted summary of evaluation results."""
    print("\n" + "=" * 60)
    print(f"  Evaluation Results — {results['algorithm']}")
    print("=" * 60)
    print(f"  Episodes:              {results['n_episodes']}")
    print(f"  Interception Rate:     {results['intercept_rate']*100:.1f}%")
    print(f"  Collision Rate:        {results['collision_rate']*100:.1f}%")
    print(f"  Timeout Rate:          {results['timeout_rate']*100:.1f}%")
    print(f"  Out of Bounds Rate:    {results['oob_rate']*100:.1f}%")
    print(f"  Avg Reward:            {results['avg_reward']:.1f} ± {results['std_reward']:.1f}")
    print(f"  Avg Steps:             {results['avg_steps']:.0f}")
    print(f"  Avg Energy Used:       {results['avg_energy']:.1f}")
    print(f"  Est. $/Interception:   ${results['cost_per_interception']:,.0f}")

    if results["avg_success_steps"] > 0:
        print(f"\n  Successful Episodes Only:")
        print(f"    Avg Steps to Intercept: {results['avg_success_steps']:.0f}")
        print(f"    Avg Reward:             {results['avg_success_reward']:.1f}")
    print("=" * 60)


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """
    Save evaluation results to a JSON file.

    Args:
        results: Evaluation results dict.
        filepath: Output file path.
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

    # Make a copy without per-episode data for cleaner output
    summary = {k: v for k, v in results.items() if k != "episodes"}
    summary["episode_count"] = len(results.get("episodes", []))

    try:
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to: {filepath}")
    except IOError as e:
        print(f"  Warning: Could not save results to {filepath}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained drone interception model"
    )
    parser.add_argument(
        "--model", type=str, default="./models/ppo_interceptor.zip",
        help="Path to trained model .zip file"
    )
    parser.add_argument(
        "--episodes", type=int, default=100,
        help="Number of evaluation episodes (default: 100)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path (default: ./results/<algo>_eval.json)"
    )
    args = parser.parse_args()

    results = evaluate_model(
        model_path=args.model,
        n_episodes=args.episodes,
        seed=args.seed,
    )

    # Auto-generate output path if not specified
    if args.output is None:
        algo = detect_algorithm(args.model)
        output_path = f"./results/{algo}_eval.json"
    else:
        output_path = args.output

    save_results(results, output_path)
