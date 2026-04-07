"""Save trained PPO model with metadata."""
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def save_model_package():
    """Bundle model + metadata into a versioned deployment package."""
    src = os.path.join(MODEL_DIR, "ppo_interceptor.zip")
    if not os.path.exists(src):
        print("Model not found. Train first: python3 -m training.train_ppo --timesteps 1000000")
        return

    # Load eval results if available
    eval_path = os.path.join(RESULTS_DIR, "ppo_eval.json")
    metrics_path = os.path.join(RESULTS_DIR, "ppo_metrics.json")
    eval_data = {}
    metrics_data = {}
    try:
        with open(eval_path) as f:
            eval_data = json.load(f)
    except FileNotFoundError:
        pass
    try:
        with open(metrics_path) as f:
            metrics_data = json.load(f)
    except FileNotFoundError:
        pass

    metadata = {
        "model_name": "drone_interceptor_ppo_v1",
        "algorithm": "PPO",
        "version": "1.0.0",
        "saved_at": datetime.now().isoformat(),
        "training_timesteps": 1000000,
        "interception_rate": eval_data.get("interception_rate", metrics_data.get("final_metrics", {}).get("intercept_rate", "N/A")),
        "collision_rate": eval_data.get("collision_rate", "N/A"),
        "avg_reward": eval_data.get("avg_reward", "N/A"),
        "cost_per_interception_usd": eval_data.get("est_cost_per_interception", "N/A"),
        "avg_steps_to_intercept": eval_data.get("avg_steps", "N/A"),
        "training_time_min": metrics_data.get("training_time_seconds", 0) / 60,
        "environment": {
            "arena_size": 10.0,
            "capture_distance": 1.0,
            "max_steps": 500,
            "evader_speed": 2.0,
            "drone_mass": 1.0,
            "max_force": 5.0,
        },
        "deployment": {
            "edge_device": "Jetson Nano / RPi 4",
            "inference_ms": "<10",
            "model_size_kb": round(os.path.getsize(src) / 1024, 1),
        },
    }

    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Model: {src}")
    print(f"Metadata: {meta_path}")
    print(f"Size: {metadata['deployment']['model_size_kb']} KB")
    print(f"Intercept Rate: {metadata['interception_rate']}")
    print(f"Cost/Interception: ${metadata['cost_per_interception_usd']}")


if __name__ == "__main__":
    save_model_package()
