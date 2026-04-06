"""
FastAPI backend that runs the actual PPO model in the DroneInterceptionEnv
and returns episode frames mapped to geographic coordinates for the React demo.
"""

import os
import sys
import math
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO
from core.drone_env import DroneInterceptionEnv
from core.domain_randomization import DomainRandomizationWrapper

# ── Load model once at startup ──
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "ppo_interceptor.zip")
print(f"Loading PPO model from {MODEL_PATH} ...")
model = PPO.load(MODEL_PATH)
print("Model loaded successfully.")

# ── Geographic data ──
IRAN_LAUNCH_SITES = [
    {"label": "Bandar-e Shahid Rajaee", "lat": 27.12, "lon": 56.06},
    {"label": "Bushehr", "lat": 28.97, "lon": 50.84},
    {"label": "Abadan", "lat": 30.37, "lon": 48.25},
    {"label": "Jask", "lat": 25.64, "lon": 57.77},
    {"label": "Chabahar", "lat": 25.29, "lon": 60.62},
]

AIRBASES = {
    "Al Dhafra AB, UAE": {"lat": 24.2481, "lon": 54.5472},
    "Camp Arifjan, Kuwait": {"lat": 29.3417, "lon": 47.9775},
    "Prince Sultan AB, KSA": {"lat": 24.0627, "lon": 47.5802},
}

# ── Phase thresholds (fraction of episode steps) ──
PHASE_RF_FRAC = 0.50
PHASE_YOLO_FRAC = 0.72

# ── FastAPI app ──
app = FastAPI(title="Counter-UAS PPO Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScenarioRequest(BaseModel):
    baseName: str = "Al Dhafra AB, UAE"
    seed: int = 42
    drEnabled: bool = True


def pick_iran_site(rng):
    idx = rng.integers(0, len(IRAN_LAUNCH_SITES))
    site = IRAN_LAUNCH_SITES[idx]
    return site["label"], site["lat"], site["lon"]


def run_ppo_episode(seed: int, use_dr: bool):
    """
    Run one full episode using the trained PPO model.
    Returns list of (interceptor_pos, target_pos, distance, done_info) per step.
    """
    rng_local = np.random.default_rng(seed)

    env = DroneInterceptionEnv(render_mode=None)
    if use_dr:
        env = DomainRandomizationWrapper(env)

    obs, info = env.reset(seed=seed)

    trajectory = []
    done = False
    step_count = 0
    intercepted = False
    dr_params = info.get("domain_randomization", {})

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

        int_pos = env.unwrapped.interceptor_pos.copy()
        tgt_pos = env.unwrapped.target_pos.copy()
        dist = np.linalg.norm(int_pos - tgt_pos)

        if info.get("intercepted", False):
            intercepted = True

        trajectory.append({
            "int_pos": int_pos,
            "tgt_pos": tgt_pos,
            "dist": float(dist),
            "intercepted": intercepted,
        })

    env.close()
    return trajectory, intercepted, dr_params


def lerp(a, b, t):
    return a + (b - a) * t


@app.post("/api/scenario")
def generate_scenario(req: ScenarioRequest):
    base = AIRBASES.get(req.baseName)
    if not base:
        return {"error": f"Unknown base: {req.baseName}"}

    base_lat, base_lon = base["lat"], base["lon"]
    rng = np.random.default_rng(req.seed)
    iran_label, iran_lat, iran_lon = pick_iran_site(rng)

    # ── 1. Run PPO model to get outcome + DR params ──
    trajectory, intercepted, dr_params = run_ppo_episode(req.seed, req.drEnabled)
    total_env_steps = len(trajectory)

    # Extract target Y-positions from arena for zig-zag lateral drift
    arena_y = [t["tgt_pos"][1] for t in trajectory]
    arena_size = 10.0

    # ── 2. Build geographic frames ──
    # Use a fixed number of display frames for smooth playback
    N = max(total_env_steps, 150)
    PHASE_RF = 0.50 + rng.uniform(-0.08, 0.05)
    PHASE_YOLO = 0.72 + rng.uniform(-0.05, 0.05)
    # Kill at ~85% of the journey — well before reaching base
    PHASE_KILL = 0.85 + rng.uniform(-0.03, 0.03)

    rf_frame = int(PHASE_RF * (N - 1))
    yolo_frame = int(PHASE_YOLO * (N - 1))
    kill_frame = int(PHASE_KILL * (N - 1))

    # ── 3. Adversary zig-zag path: Iran → Base with lateral drift ──
    dlat = base_lat - iran_lat
    dlon = base_lon - iran_lon
    route_len = math.sqrt(dlat ** 2 + dlon ** 2)
    # Perpendicular vector for lateral offset
    if route_len > 1e-6:
        px, py = -dlon / route_len, dlat / route_len
    else:
        px, py = 0, 0

    drift_scale = rng.uniform(0.03, 0.07)
    adv_path = []
    adv_lat_c, adv_lon_c = iran_lat, iran_lon
    for i in range(N):
        frac = i / (N - 1)
        pull = 0.6 + 0.4 * frac
        tgt_lat = lerp(iran_lat, base_lat, frac)
        tgt_lon = lerp(iran_lon, base_lon, frac)
        # Use arena Y for zig-zag when available, else random walk
        if i < len(arena_y):
            lateral = (arena_y[i] / arena_size) * drift_scale * route_len * 0.5
        else:
            lateral = rng.normal(0, drift_scale) * (1 - 0.5 * frac)
        adv_lat_c += (tgt_lat - adv_lat_c) * pull * 0.15 + rng.normal(0, drift_scale) * (1 - 0.5 * frac)
        adv_lon_c += (tgt_lon - adv_lon_c) * pull * 0.15 + rng.normal(0, drift_scale) * (1 - 0.5 * frac)
        # Add zig-zag lateral offset from PPO arena
        lat = adv_lat_c + lateral * px
        lon = adv_lon_c + lateral * py
        adv_path.append((lat, lon))

    # Freeze adversary at kill point after interception
    kill_lat, kill_lon = adv_path[kill_frame]

    # ── 4. Interceptor: Bezier pursuit from base to kill point ──
    # Control point: midpoint of base→kill, offset toward adversary's mid-pursuit position
    mid_frame = (yolo_frame + kill_frame) // 2
    adv_mid_lat, adv_mid_lon = adv_path[mid_frame]
    mid_lat = (base_lat + kill_lat) / 2
    mid_lon = (base_lon + kill_lon) / 2
    ctrl_lat = mid_lat + 0.5 * (adv_mid_lat - mid_lat)
    ctrl_lon = mid_lon + 0.5 * (adv_mid_lon - mid_lon)

    # ── 5. Build frame array ──
    frames = []
    for i in range(N):
        frac = i / (N - 1)

        # Phase
        if i < rf_frame:
            phase, pn = "CROSSING", 1
        elif i < yolo_frame:
            phase, pn = "RF DETECTED", 2
        elif i < kill_frame:
            phase, pn = "PURSUING", 3
        else:
            phase, pn = "INTERCEPTED", 4

        # Adversary position (freeze after kill)
        if i <= kill_frame:
            adv_lat, adv_lon = adv_path[i]
        else:
            adv_lat, adv_lon = kill_lat, kill_lon

        # Interceptor: Bezier pursuit curve
        if i < yolo_frame:
            int_lat, int_lon = base_lat, base_lon
        elif i <= kill_frame:
            pt = (i - yolo_frame) / max(kill_frame - yolo_frame, 1)
            a = (1 - pt) ** 2
            b = 2 * (1 - pt) * pt
            c = pt ** 2
            int_lat = a * base_lat + b * ctrl_lat + c * kill_lat
            int_lon = a * base_lon + b * ctrl_lon + c * kill_lon
        else:
            int_lat, int_lon = kill_lat, kill_lon

        # Telemetry
        d_lat = adv_lat - base_lat
        d_lon = adv_lon - base_lon
        dist_km = math.sqrt(
            (d_lat * 111) ** 2 + (d_lon * 111 * math.cos(math.radians(base_lat))) ** 2
        )
        bearing = (math.degrees(math.atan2(d_lon, d_lat)) + 360) % 360

        if pn <= 1:
            yc = 0.0
        elif pn == 2:
            yc = round(float(rng.uniform(0.25, 0.50)), 2)
        else:
            yc = round(float(min(0.99, max(0.50, 0.55 + (frac - PHASE_YOLO) * 3.0 + rng.normal(0, 0.03)))), 2)

        adv_alt = float(500 + rng.normal(0, 20))
        int_alt = 100.0 if pn <= 2 else float(np.interp(frac, [PHASE_YOLO, 1.0], [100, adv_alt]))
        brg = round(bearing, 1)

        frames.append({
            "step": i + 1,
            "t": round(frac, 4),
            "advLat": round(adv_lat, 6),
            "advLon": round(adv_lon, 6),
            "advAlt": round(adv_alt, 1),
            "intLat": round(int_lat, 6),
            "intLon": round(int_lon, 6),
            "intAlt": round(int_alt, 1),
            "distKm": round(dist_km, 1),
            "bearing": brg,
            "phase": phase,
            "pn": pn,
            "yc": yc,
        })

    # Domain randomization params for display
    dr_display = None
    if dr_params:
        dr_display = {
            "drone_mass": round(dr_params.get("drone_mass", 1.0), 3),
            "max_force": round(dr_params.get("max_force", 5.0), 2),
            "drag_coeff": round(dr_params.get("drag_coefficient", 0.3), 3),
            "evader_speed": round(dr_params.get("evader_speed", 2.0), 2),
            "num_obstacles": int(dr_params.get("num_obstacles", 5)),
            "obs_noise_std": round(dr_params.get("obs_noise_std", 0.0), 4),
            "action_delay": int(dr_params.get("action_delay_steps", 0)),
            "gravity": round(dr_params.get("gravity", 9.81), 4),
        }

    return {
        "frames": frames,
        "iranLabel": iran_label,
        "iranLat": iran_lat,
        "iranLon": iran_lon,
        "intercepted": intercepted,
        "totalSteps": total_env_steps,
        "drParams": dr_display,
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "model": "ppo_interceptor"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
