"""Backend for geospatial scenario generation using trained PPO model."""

import os
import sys
import math
from typing import Tuple

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from stable_baselines3 import PPO

from core.drone_env import DroneInterceptionEnv
from core.domain_randomization import DomainRandomizationWrapper

# Set up path for local imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Load PPO model at startup
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "ppo_interceptor.zip")
print(f"Loading PPO model from {MODEL_PATH} ...")
model = PPO.load(MODEL_PATH)
print("Model loaded successfully.")

# Geographic origins (launch/detection sites)
IRAN_LAUNCH_SITES = [
    {"label": "Bandar-e Shahid Rajaee", "lat": 27.12, "lon": 56.06},
    {"label": "Bushehr", "lat": 28.97, "lon": 50.84},
    {"label": "Abadan", "lat": 30.37, "lon": 48.25},
    {"label": "Jask", "lat": 25.64, "lon": 57.77},
    {"label": "Chabahar", "lat": 25.29, "lon": 60.62},
]

# Defender airbases
AIRBASES = {
    "Al Dhafra AB, UAE": {"lat": 24.2481, "lon": 54.5472},
    "Camp Arifjan, Kuwait": {"lat": 29.3417, "lon": 47.9775},
    "Prince Sultan AB, KSA": {"lat": 24.0627, "lon": 47.5802},
}

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


def pick_iran_site(rng: np.random.Generator) -> Tuple[str, float, float]:
    """Select random launch site for scenario variety."""
    site = IRAN_LAUNCH_SITES[rng.integers(0, len(IRAN_LAUNCH_SITES))]
    return site["label"], site["lat"], site["lon"]


def run_ppo_episode(seed: int, use_domain_randomization: bool) -> Tuple[list, bool, dict]:
    """Execute one episode to capture both the decision trajectory and physical parameters.
    
    Trajectory shapes the geographic path; DR params calibrate visualization realism.
    """
    env = DroneInterceptionEnv(render_mode=None)
    if use_domain_randomization:
        env = DomainRandomizationWrapper(env)

    obs, info = env.reset(seed=seed)
    trajectory = []
    intercepted = False
    dr_params = info.get("domain_randomization", {})

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

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


def lerp(start: float, end: float, t: float) -> float:
    """Linear interpolation; fast and predictable for trajectory blending."""
    return start + (end - start) * t


def compute_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Bearing for flight displays; front-end needs cardinal direction."""
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    bearing = (math.degrees(math.atan2(d_lon, d_lat)) + 360) % 360
    return bearing


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Fast distance approximation for telemetry; sufficient at regional scales."""
    # 111 km per degree is accurate enough; great-circle overkill for dashboard
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    return math.sqrt(
        (d_lat * 111) ** 2 + (d_lon * 111 * math.cos(math.radians(lat1))) ** 2
    )


def build_adversary_path(
    rng: np.random.Generator,
    start_lat: float,
    start_lon: float,
    target_lat: float,
    target_lon: float,
    arena_y: list,
    num_frames: int,
) -> list:
    """Blend PPO evasion patterns into realistic geographic travel.
    
    Uses arena trajectories to inform lateral drift; maintains smooth motion for visuals.
    """
    dlat = target_lat - start_lat
    dlon = target_lon - start_lon
    route_len = math.sqrt(dlat ** 2 + dlon ** 2)

    # Compute perpendicular to route for left-right drift (evasive maneuvers)
    if route_len > 1e-6:
        perp_x = -dlon / route_len
        perp_y = dlat / route_len
    else:
        perp_x, perp_y = 0, 0

    drift_scale = rng.uniform(0.03, 0.07)
    arena_size = 10.0
    path = []

    # Track smoothed position to apply momentum; avoids jittery paths
    curr_lat, curr_lon = start_lat, start_lon

    for i in range(num_frames):
        frac = i / (num_frames - 1)
        momentum = 0.6 + 0.4 * frac

        # Intended waypoint along straight line
        target = lerp(start_lat, target_lat, frac)
        target_lon_val = lerp(start_lon, target_lon, frac)

        # Translate PPO's arena Y (evasion axis) to geographic lateral drift
        # Falls back to random walk if trajectory runs short
        if i < len(arena_y):
            lateral_offset = (arena_y[i] / arena_size) * drift_scale * route_len * 0.5
        else:
            lateral_offset = rng.normal(0, drift_scale) * (1 - 0.5 * frac)

        # Gradual approach to target reduces discontinuities in geospatial playback
        noise_mag = drift_scale * (1 - 0.5 * frac)
        curr_lat += (target - curr_lat) * momentum * 0.15 + rng.normal(0, noise_mag)
        curr_lon += (target_lon_val - curr_lon) * momentum * 0.15 + rng.normal(0, noise_mag)

        # Apply lateral offset perpendicular to route
        lat = curr_lat + lateral_offset * perp_x
        lon = curr_lon + lateral_offset * perp_y
        path.append((lat, lon))

    return path


@app.post("/api/scenario")
def generate_scenario(req: ScenarioRequest):
    """Build playable scenario; blend PPO decisions with geographic realism for frontend."""
    base = AIRBASES.get(req.baseName)
    if not base:
        return {"error": f"Unknown base: {req.baseName}"}

    base_lat, base_lon = base["lat"], base["lon"]
    rng = np.random.default_rng(req.seed)
    iran_label, iran_lat, iran_lon = pick_iran_site(rng)

    # Run PPO once to extract arena trajectory and learned intercept dynamics
    trajectory, intercepted, dr_params = run_ppo_episode(req.seed, req.drEnabled)
    arena_y = [t["tgt_pos"][1] for t in trajectory]
    total_env_steps = len(trajectory)

    # Smooth playback; 150-frame minimum ensures responsive visualization
    num_frames = max(total_env_steps, 150)

    # Randomize phase timings slightly; avoids unrealistic patterns across scenarios
    phase_rf = 0.50 + rng.uniform(-0.08, 0.05)
    phase_yolo = 0.55 + rng.uniform(-0.03, 0.03)
    phase_kill = 0.90 + rng.uniform(-0.03, 0.03)  # Kill before base to show interception

    rf_frame = int(phase_rf * (num_frames - 1))
    yolo_frame = int(phase_yolo * (num_frames - 1))
    kill_frame = int(phase_kill * (num_frames - 1))

    # Generate adversary path
    adv_path = build_adversary_path(
        rng, iran_lat, iran_lon, base_lat, base_lon, arena_y, num_frames
    )
    kill_lat, kill_lon = adv_path[kill_frame]

    # Bezier curve follows adversary's mid-pursuit position; looks natural for intercept animation
    mid_frame = (yolo_frame + kill_frame) // 2
    adv_mid_lat, adv_mid_lon = adv_path[mid_frame]
    mid_lat = (base_lat + kill_lat) / 2
    mid_lon = (base_lon + kill_lon) / 2
    # Pull control point toward adversary's actual position—more realistic pursuit
    ctrl_lat = mid_lat + 0.5 * (adv_mid_lat - mid_lat)
    ctrl_lon = mid_lon + 0.5 * (adv_mid_lon - mid_lon)

    # Build frame-by-frame data
    frames = []
    for i in range(num_frames):
        frac = i / (num_frames - 1)

        # Phase progression mirrors sensor capability and intercept timeline
        if i < rf_frame:
            phase_name, phase_num = "CROSSING", 1
        elif i < yolo_frame:
            phase_name, phase_num = "RF DETECTED", 2
        elif i < kill_frame:
            phase_name, phase_num = "PURSUING", 3
        else:
            phase_name, phase_num = "INTERCEPTED", 4

        # Adversary position (stays frozen after interception)
        if i <= kill_frame:
            adv_lat, adv_lon = adv_path[i]
        else:
            adv_lat, adv_lon = kill_lat, kill_lon

        # Interceptor launches only after detection; smooth arc is visually superior to straight lines
        if i < yolo_frame:
            int_lat, int_lon = base_lat, base_lon
        elif i <= kill_frame:
            # Quadratic Bezier smoothly transitions via control point
            t = (i - yolo_frame) / max(kill_frame - yolo_frame, 1)
            a = (1 - t) ** 2
            b = 2 * (1 - t) * t
            c = t ** 2
            int_lat = a * base_lat + b * ctrl_lat + c * kill_lat
            int_lon = a * base_lon + b * ctrl_lon + c * kill_lon
        else:
            int_lat, int_lon = kill_lat, kill_lon

        # Compute telemetry
        dist_km_val = distance_km(base_lat, base_lon, adv_lat, adv_lon)
        bearing_val = compute_bearing(base_lat, base_lon, adv_lat, adv_lon)

        # Confidence reflects system certainty; zero until RF detection, then builds
        if phase_num <= 1:
            confidence = 0.0
        elif phase_num == 2:
            # Early RF lock carries uncertainty; wide variance
            confidence = round(float(rng.uniform(0.25, 0.50)), 2)
        else:
            # Track quality improves over pursuit window
            confidence = min(
                0.99,
                max(
                    0.50,
                    0.55 + (frac - phase_yolo) * 3.0 + rng.normal(0, 0.03),
                ),
            )
            confidence = round(float(confidence), 2)

        # Altitude: adversary cruises high, interceptor launches and climbs
        adv_alt = 500 + rng.normal(0, 20)
        if phase_num <= 2:
            int_alt = 100.0
        else:
            # Interceptor altitude ramps toward adversary during pursuit
            int_alt = np.interp(frac, [phase_yolo, 1.0], [100, adv_alt])

        frames.append({
            "step": i + 1,
            "t": round(frac, 4),
            "advLat": round(adv_lat, 6),
            "advLon": round(adv_lon, 6),
            "advAlt": round(adv_alt, 1),
            "intLat": round(int_lat, 6),
            "intLon": round(int_lon, 6),
            "intAlt": round(int_alt, 1),
            "distKm": round(dist_km_val, 1),
            "bearing": round(bearing_val, 1),
            "phase": phase_name,
            "pn": phase_num,
            "yc": confidence,
        })

    # Package DR params for transparency; shows what physical variations were in play
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
    """Quick liveness check for orchestration."""
    return {"status": "ok", "model": "ppo_interceptor"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
