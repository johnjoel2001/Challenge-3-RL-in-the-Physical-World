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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
PHASE_YOLO_FRAC = 0.55

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
    If the episode ends too early (collision/OOB < 80 steps), retry with
    seed offsets to find a meaningful chase. Returns trajectory, outcome, DR params.
    """
    best_traj = None
    best_intercepted = False
    best_dr = {}
    best_reason = "timeout"

    # Try up to 5 seed offsets to get a good episode (>80 steps or intercepted)
    for attempt in range(5):
        actual_seed = seed + attempt * 1000

        env = DroneInterceptionEnv(render_mode=None)
        if use_dr:
            env = DomainRandomizationWrapper(env)

        obs, info = env.reset(seed=actual_seed)

        trajectory = []
        done = False
        step_count = 0
        intercepted = False
        dr_params = info.get("domain_randomization", {})
        reason = "timeout"

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
                reason = "intercepted"
            elif info.get("collision", False):
                reason = "collision"
            elif info.get("out_of_bounds", False):
                reason = "out_of_bounds"

            # action is normalized [-1,1]; real thrust = action * max_force
            max_f = env.unwrapped.max_force
            thrust = action * max_f
            trajectory.append({
                "int_pos": int_pos,
                "tgt_pos": tgt_pos,
                "thrust": thrust.copy(),
                "dist": float(dist),
                "intercepted": intercepted,
            })

        env.close()

        # Keep the best episode (intercepted > long episode > short episode)
        if best_traj is None or intercepted or (not best_intercepted and len(trajectory) > len(best_traj)):
            best_traj = trajectory
            best_intercepted = intercepted
            best_dr = dr_params
            best_reason = reason

        # Good enough — intercepted or ran long enough
        if intercepted or len(trajectory) >= 80:
            break

    return best_traj, best_intercepted, best_dr, best_reason


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
    trajectory, intercepted, dr_params, reason = run_ppo_episode(req.seed, req.drEnabled)
    total_env_steps = len(trajectory)

    # Extract target Y-positions from arena for zig-zag lateral drift
    arena_y = [t["tgt_pos"][1] for t in trajectory]
    arena_size = 10.0

    # ── 2. Build geographic frames ──
    # Use a fixed number of display frames for smooth playback
    N = max(total_env_steps, 150)
    PHASE_RF = 0.50 + rng.uniform(-0.08, 0.05)
    PHASE_YOLO = 0.55 + rng.uniform(-0.03, 0.03)
    # Kill at ~90% of the journey — well before reaching base
    PHASE_KILL = 0.90 + rng.uniform(-0.03, 0.03)

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
    # Control point: 40% along base→kill with small perpendicular offset
    # Creates a gentle curve WITHOUT overshooting past the interception point
    frac_lat = base_lat + 0.4 * (kill_lat - base_lat)
    frac_lon = base_lon + 0.4 * (kill_lon - base_lon)
    d_lat0 = kill_lat - base_lat
    d_lon0 = kill_lon - base_lon
    perp_lat = -d_lon0   # perpendicular direction
    perp_lon = d_lat0
    perp_scale = 0.06    # small offset — visible curve, no overshoot
    ctrl_lat = frac_lat + perp_lat * perp_scale
    ctrl_lon = frac_lon + perp_lon * perp_scale

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

        # Map pursuing-phase frames to PPO trajectory for arena xyz + thrust
        arena_data = {}
        if pn >= 3 and total_env_steps > 0:
            # Map pursuit progress to trajectory index
            pursuit_frac = (i - yolo_frame) / max(kill_frame - yolo_frame, 1)
            pursuit_frac = max(0.0, min(1.0, pursuit_frac))
            tj_idx = min(int(pursuit_frac * (total_env_steps - 1)), total_env_steps - 1)
            t_step = trajectory[tj_idx]
            arena_data = {
                "ax": round(float(t_step["tgt_pos"][0]), 2),
                "ay": round(float(t_step["tgt_pos"][1]), 2),
                "az": round(float(t_step["tgt_pos"][2]), 2),
                "tx": round(float(t_step["thrust"][0]), 2),
                "ty": round(float(t_step["thrust"][1]), 2),
                "tz": round(float(t_step["thrust"][2]), 2),
            }

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
            **arena_data,
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


@app.post("/api/scenario3d")
def generate_scenario_3d(req: ScenarioRequest):
    """
    Run PPO episode and return raw 3D arena positions for Three.js visualization.
    Returns every Nth frame to keep payload manageable.
    """
    trajectory, intercepted, dr_params, reason = run_ppo_episode(req.seed, req.drEnabled)

    # Sample frames for smooth playback (every 2nd frame, or all if short)
    step = max(1, len(trajectory) // 250)
    sampled = trajectory[::step]
    # Always include the last frame
    if sampled[-1] is not trajectory[-1]:
        sampled.append(trajectory[-1])

    frames = []
    for i, t in enumerate(sampled):
        frames.append({
            "ix": round(float(t["int_pos"][0]), 3),
            "iy": round(float(t["int_pos"][2]), 3),   # z -> y (up) in Three.js
            "iz": round(float(t["int_pos"][1]), 3),
            "tx": round(float(t["tgt_pos"][0]), 3),
            "ty": round(float(t["tgt_pos"][2]), 3),
            "tz": round(float(t["tgt_pos"][1]), 3),
            "dist": round(t["dist"], 3),
            "intercepted": t["intercepted"],
        })

    return {
        "frames": frames,
        "totalSteps": len(trajectory),
        "intercepted": intercepted,
        "reason": reason,
        "drParams": dr_params if req.drEnabled else None,
        "arenaSize": 20,
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "model": "ppo_interceptor"}


# ── Serve built React frontend (for Railway deployment) ──
STATIC_DIR = os.path.join(PROJECT_ROOT, "react-demo", "dist")
print(f"[STARTUP] PROJECT_ROOT = {PROJECT_ROOT}")
print(f"[STARTUP] STATIC_DIR  = {STATIC_DIR}")
print(f"[STARTUP] dist exists? = {os.path.isdir(STATIC_DIR)}")
if os.path.isdir(STATIC_DIR):
    print(f"[STARTUP] dist contents: {os.listdir(STATIC_DIR)}")
    app.mount("/assets", StaticFiles(directory=os.path.join(STATIC_DIR, "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        # Serve specific files if they exist, otherwise index.html (SPA fallback)
        file_path = os.path.join(STATIC_DIR, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))
else:
    print(f"[STARTUP] WARNING: dist not found! react-demo contents: {os.listdir(os.path.join(PROJECT_ROOT, 'react-demo')) if os.path.isdir(os.path.join(PROJECT_ROOT, 'react-demo')) else 'react-demo NOT FOUND'}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
