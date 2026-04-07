"""Compare PPO vs SAC vs TD3 on the same evaluation episodes.

Usage:
    python -m evaluation.compare_algorithms
    python -m evaluation.compare_algorithms --episodes 100 --seed 42
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO, SAC, TD3
from core.drone_env import DroneInterceptionEnv
from core.cost_aware_reward import COUNTER_UAS_COSTS
from evaluation.evaluate import evaluate_model, detect_algorithm


# Algorithm configurations for comparison
ALGORITHMS = {
    "PPO": {"model_path": "./models/ppo_interceptor.zip", "color": "#2196F3", "class": PPO},
    "SAC": {"model_path": "./models/sac_interceptor.zip", "color": "#4CAF50", "class": SAC},
    "TD3": {"model_path": "./models/td3_interceptor.zip", "color": "#FF9800", "class": TD3},
}


def load_training_metrics(algo_name: str) -> Optional[Dict[str, Any]]:
    """Load training metrics from JSON file."""
    filepath = f"./results/{algo_name}_metrics.json"
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def compare_algorithms(
    n_episodes: int = 100,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate and compare all three algorithms."""
    print("\nAlgorithm Comparison")

    results = {}

    for algo_name, config in ALGORITHMS.items():
        model_path = config["model_path"]
        if not os.path.exists(model_path):
            print(f"  ⚠ {algo_name} model not found: {model_path}")
            continue

        print(f"\n  Evaluating {algo_name}...")

        algo_results = evaluate_model(
            model_path=model_path,
            n_episodes=n_episodes,
            seed=seed,
            deterministic=True,
            verbose=True,
        )
        results[algo_name] = algo_results

    return results


def print_comparison_table(results: Dict[str, Dict[str, Any]]) -> None:
    """Print formatted comparison table."""
    if not results:
        print("  No results to compare.")
        return

    print("\n" + "=" * 70)
    print("  Comparison")

    header = (
        f"  {'Algorithm':<12s} │ {'Intercept%':>10s} │ {'Collision%':>10s} │ "
        f"{'Avg Rew':>8s} │ {'Avg Steps':>10s} │ {'$/Intercept':>13s}"
    )
    sep = f"  {'─' * 12}─┼─{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 8}─┼─{'─' * 10}─┼─{'─' * 13}"

    print(header)
    print(sep)

    for algo_name, r in results.items():
        print(
            f"  {algo_name:<12s} │ {r['intercept_rate']*100:>9.1f}% │ "
            f"{r['collision_rate']*100:>9.1f}% │ {r['avg_reward']:>+8.1f} │ "
            f"{r['avg_steps']:>10.0f} │ ${r['cost_per_interception']:>11,.0f}"
        )

    # Counter-UAS comparison table
    print("\n")
    print("=" * 50)
    print("  vs. Current Counter-UAS Methods")
    print("=" * 50)
    print(f"  {'Method':<25s} │ {'$/Interception':>15s}")
    print(f"  {'─' * 25}─┼─{'─' * 15}")

    for method, data in COUNTER_UAS_COSTS.items():
        cost = data["cost_per_interception"]
        marker = " ← OURS" if "RL" in method else ""
        print(f"  {method:<25s} │ ${cost:>13,.0f}{marker}")

    print("=" * 50)


def plot_learning_curves(results: Dict[str, Dict[str, Any]]) -> None:
    """Plot learning curves for all algorithms."""
    os.makedirs("./plots", exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Algorithm Comparison — Drone Interception Training", fontsize=14, fontweight="bold")

    has_data = False

    for algo_name, config in ALGORITHMS.items():
        color = config["color"]
        metrics = load_training_metrics(algo_name.lower())
        if metrics is None or "metrics_history" not in metrics:
            continue
        has_data = True

        history = metrics["metrics_history"]
        timesteps = [m["timestep"] for m in history]
        rewards = [m["avg_reward"] for m in history]
        intercept_rates = [m["intercept_rate"] for m in history]
        episode_lengths = [m["avg_length"] for m in history]
        costs = [m.get("cost_per_interception", 0) for m in history]

        # Plot 1: Learning curves (reward vs timesteps)
        axes[0, 0].plot(timesteps, rewards, color=color, label=algo_name, linewidth=2)

        # Plot 2: Interception rate over training
        axes[0, 1].plot(timesteps, intercept_rates, color=color, label=algo_name, linewidth=2)

        # Plot 4: Episode length over training
        axes[1, 1].plot(timesteps, episode_lengths, color=color, label=algo_name, linewidth=2)

    if has_data:
        axes[0, 0].set_xlabel("Timesteps")
        axes[0, 0].set_ylabel("Average Reward")
        axes[0, 0].set_title("Learning Curves")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_xlabel("Timesteps")
        axes[0, 1].set_ylabel("Interception Rate (%)")
        axes[0, 1].set_title("Interception Rate Over Training")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 1].set_xlabel("Timesteps")
        axes[1, 1].set_ylabel("Average Episode Length")
        axes[1, 1].set_title("Episode Length Over Training")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    # Plot 3: Cost comparison bar chart (log scale)
    ax_cost = axes[1, 0]
    methods = []
    costs_list = []
    colors_list = []

    for method, data in COUNTER_UAS_COSTS.items():
        methods.append(method.replace(" (Raytheon)", "\n(Raytheon)").replace(" (Ours)", "\n(Ours)"))
        costs_list.append(data["cost_per_interception"])
        if "RL" in method:
            colors_list.append("#4CAF50")  # Green for ours
        else:
            colors_list.append("#E53935")  # Red for expensive methods

    bars = ax_cost.bar(range(len(methods)), costs_list, color=colors_list, edgecolor="white")
    ax_cost.set_xticks(range(len(methods)))
    ax_cost.set_xticklabels(methods, rotation=45, ha="right", fontsize=7)
    ax_cost.set_yscale("log")
    ax_cost.set_ylabel("$ per Interception (log scale)")
    ax_cost.set_title("Cost per Interception — Our Approach vs Alternatives")
    ax_cost.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, cost in zip(bars, costs_list):
        ax_cost.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.3,
            f"${cost:,.0f}", ha="center", va="bottom", fontsize=7, fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig("./plots/algorithm_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Saved: ./plots/algorithm_comparison.png")


def plot_episode_length_distribution(results: Dict[str, Dict[str, Any]]) -> None:
    """
    Plot episode length distribution (histogram) for each algorithm.

    Shorter episodes = faster interceptions = cheaper missions.
    The distribution shape reveals whether the policy consistently
    intercepts quickly or has a long tail of slow pursuits.
    """
    os.makedirs("./plots", exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for algo_name, r in results.items():
        color = ALGORITHMS.get(algo_name, {}).get("color", "#666666")
        episodes = r.get("episodes", [])
        if not episodes:
            continue

        lengths = [e["steps"] for e in episodes]
        ax.hist(
            lengths, bins=30, alpha=0.5, color=color, label=algo_name,
            edgecolor="white", linewidth=0.5
        )

    ax.set_xlabel("Episode Length (steps)")
    ax.set_ylabel("Count")
    ax.set_title("Episode Length Distribution by Algorithm")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("./plots/episode_length_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: ./plots/episode_length_distribution.png")


def save_comparison_results(results: Dict[str, Dict[str, Any]]) -> None:
    """Save comparison results to JSON."""
    os.makedirs("./results", exist_ok=True)

    # Create a summary without per-episode data
    summary = {}
    for algo_name, r in results.items():
        summary[algo_name] = {k: v for k, v in r.items() if k != "episodes"}

    filepath = "./results/comparison_results.json"
    try:
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n  Comparison results saved to: {filepath}")
    except IOError as e:
        print(f"  Warning: Could not save to {filepath}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare PPO, SAC, and TD3 for drone interception"
    )
    parser.add_argument(
        "--episodes", type=int, default=100,
        help="Number of evaluation episodes per algorithm (default: 100)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    # Run evaluation comparison
    results = compare_algorithms(n_episodes=args.episodes, seed=args.seed)

    if results:
        # Print comparison table
        print_comparison_table(results)

        # Generate all plots
        plot_learning_curves(results)
        plot_episode_length_distribution(results)

        # Save results
        save_comparison_results(results)

        print("\n  All comparison plots and results saved!")
    else:
        print("\n  No trained models found. Please train at least one model first:")
        print("    python -m training.train_ppo --timesteps 500000")
        print("    python -m training.train_sac --timesteps 500000")
        print("    python -m training.train_td3 --timesteps 500000")
