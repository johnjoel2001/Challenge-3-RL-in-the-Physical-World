"""
Counter-UAS Command Center — 3D Globe + YOLO Detection Demo

Scenario: Adversary drone launches from Iran, crosses the Persian Gulf
with evasive maneuvers, gets detected by RF sensors, visually identified
by YOLO, and intercepted by our RL-trained pursuit drone.

Detection chain:
  RF Sensor (80km) → YOLO Camera (30km) → RL Policy → Interceptor Kill

Run:
    cd drone_interception
    streamlit run demo/command_center.py
"""

import os, sys, math, time
import numpy as np
import streamlit as st
import pydeck as pdk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ═════════════════════════════════════════════════════════════════════════
# GEOGRAPHIC CONSTANTS
# ═════════════════════════════════════════════════════════════════════════
IRAN_LAUNCH_SITES = [
    {"lat": 27.18, "lon": 56.27, "label": "Bandar Abbas"},
    {"lat": 26.55, "lon": 54.35, "label": "Bandar Lengeh"},
    {"lat": 27.20, "lon": 52.60, "label": "Bushehr"},
    {"lat": 25.44, "lon": 57.08, "label": "Jask"},
    {"lat": 26.95, "lon": 55.05, "label": "Qeshm Island"},
    {"lat": 27.50, "lon": 56.90, "label": "Minab"},
    {"lat": 26.32, "lon": 54.20, "label": "Sirri Island"},
]

AIRBASES = {
    "Al Dhafra AB, UAE": {"lat": 24.2481, "lon": 54.5472},
    "Prince Sultan AB, KSA": {"lat": 24.0627, "lon": 47.5802},
}

PHASE_RF   = 0.50
PHASE_YOLO = 0.72
PHASE_KILL = 0.93
TOTAL_FRAMES = 150
SPEED_DELAY = {"SLOW": 0.22, "NORMAL": 0.08, "FAST": 0.02}

# ═════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ═════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Counter-UAS Command Center", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.stApp{background:#0a0e17}
section[data-testid="stSidebar"]{background:#0d1117;border-right:1px solid #1a3a2a}
.mt{font-family:'Courier New',monospace;color:#ff4444;font-size:1.35em;text-align:center;letter-spacing:3px;padding:5px 0}
.st2{font-family:'Courier New',monospace;color:#888;font-size:0.78em;text-align:center;letter-spacing:2px}
.bn{font-family:'Courier New',monospace;text-align:center;padding:9px;border-radius:5px;font-size:1em;letter-spacing:2px;margin:5px 0}
.bs{background:linear-gradient(90deg,#0a2e0a,#1a5a1a,#0a2e0a);color:#4cff4c;border:1px solid #2a6a2a}
.ba{background:linear-gradient(90deg,#2e1a0a,#5a3a1a,#2e1a0a);color:#ffaa44;border:1px solid #6a4a2a}
.bw{background:linear-gradient(90deg,#0a0e17,#1a2030,#0a0e17);color:#6688aa;border:1px solid #2a3a4a}
.mb{background:#111822;border:1px solid #1a2a3a;border-radius:4px;padding:5px 8px;text-align:center;font-family:'Courier New',monospace}
.ml2{color:#557788;font-size:0.62em;letter-spacing:1px;text-transform:uppercase}
.mv{color:#44ddff;font-size:1.2em;font-weight:bold}
.mr2{color:#ff4444;font-size:1.2em;font-weight:bold}
.mg{color:#44ff44;font-size:1.2em;font-weight:bold}
.my{color:#ffcc44;font-size:1.2em;font-weight:bold}
.sh{font-family:'Courier New',monospace;color:#557788;font-size:0.72em;letter-spacing:2px;border-bottom:1px solid #1a2a3a;padding-bottom:3px;margin-bottom:5px}
.lg2{font-family:'Courier New',monospace;font-size:0.7em;color:#88aacc;line-height:1.65}
.gc{color:#44ff44}.bc{color:#44ddff}.wc{color:#ffcc44}.rc{color:#ff4444}.sc{color:#44ff44;font-weight:bold}
.pb{background:#111822;border:1px solid #1a2a3a;border-radius:5px;padding:9px;text-align:center;font-family:'Courier New',monospace}
.ptc{color:#44ddff;font-size:0.82em;font-weight:bold;margin-bottom:3px}
.pd{color:#668899;font-size:0.63em;line-height:1.5}
.sl{font-family:'Courier New',monospace;color:#557788;font-size:0.68em;letter-spacing:1px;text-transform:uppercase}
#MainMenu{visibility:hidden}footer{visibility:hidden}header{visibility:hidden}
</style>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
# SCENARIO GENERATION — non-deterministic random-walk path
# ═════════════════════════════════════════════════════════════════════════
def lerp(a, b, t):
    return a + (b - a) * t


def generate_domain_rand(seed=42):
    """Sample domain randomization parameters (mirrors core/domain_randomization.py)."""
    rng = np.random.RandomState(seed + 7777)
    return {
        "drone_mass": round(rng.uniform(0.7, 1.5), 3),
        "max_force": round(rng.uniform(3.5, 7.0), 2),
        "drag_coeff": round(rng.uniform(0.1, 0.6), 3),
        "evader_speed": round(rng.uniform(1.0, 3.5), 2),
        "num_obstacles": int(rng.randint(2, 9)),
        "obs_noise_std": round(rng.uniform(0, 0.05), 4),
        "action_delay": int(rng.randint(0, 4)),
        "gravity": round(rng.uniform(9.75, 9.85), 3),
    }


def generate_scenario(base_name, seed=42):
    rng = np.random.RandomState(seed)
    b = AIRBASES[base_name]
    base_lat, base_lon = b["lat"], b["lon"]

    site = IRAN_LAUNCH_SITES[rng.randint(0, len(IRAN_LAUNCH_SITES))]
    iran_lat, iran_lon = site["lat"], site["lon"]
    iran_label = site["label"]

    rf_t   = PHASE_RF   + rng.uniform(-0.08, 0.05)
    yolo_t = PHASE_YOLO + rng.uniform(-0.05, 0.05)
    kill_t = PHASE_KILL + rng.uniform(-0.04, 0.03)

    drift_scale = rng.uniform(0.02, 0.06)

    # Random walk path for adversary
    adv_lat_c, adv_lon_c = iran_lat, iran_lon
    adv_path = []
    for i in range(TOTAL_FRAMES):
        t = i / (TOTAL_FRAMES - 1)
        pull = 0.6 + 0.4 * t
        tgt_lat = lerp(iran_lat, base_lat, t)
        tgt_lon = lerp(iran_lon, base_lon, t)
        adv_lat_c += (tgt_lat - adv_lat_c) * pull * 0.15 + rng.normal(0, drift_scale) * (1 - 0.5 * t)
        adv_lon_c += (tgt_lon - adv_lon_c) * pull * 0.15 + rng.normal(0, drift_scale) * (1 - 0.5 * t)
        adv_path.append((adv_lat_c, adv_lon_c))

    # Converge last frames near base
    for i in range(max(0, TOTAL_FRAMES - 8), TOTAL_FRAMES):
        bl = (i - (TOTAL_FRAMES - 8)) / 8.0
        la, lo = adv_path[i]
        adv_path[i] = (lerp(la, base_lat + 0.015, bl), lerp(lo, base_lon + 0.015, bl))

    frames = []
    for i in range(TOTAL_FRAMES):
        t = i / (TOTAL_FRAMES - 1)
        adv_lat, adv_lon = adv_path[i]

        dlat = adv_lat - base_lat
        dlon = adv_lon - base_lon
        dist_km = math.sqrt((dlat * 111)**2 + (dlon * 111 * math.cos(math.radians(base_lat)))**2)

        if t < rf_t:
            phase, pn = "CROSSING", 1
        elif t < yolo_t:
            phase, pn = "RF DETECTED", 2
        elif t < kill_t:
            phase, pn = "PURSUING", 3
        else:
            phase, pn = "INTERCEPTED", 4

        if t < yolo_t:
            int_lat, int_lon = base_lat, base_lon
        else:
            pt = min(1.0, (t - yolo_t) / (kill_t - yolo_t))
            int_lat = lerp(base_lat, adv_lat, pt * 0.97) + rng.normal(0, 0.003)
            int_lon = lerp(base_lon, adv_lon, pt * 0.97) + rng.normal(0, 0.003)

        if pn <= 1:
            yc = 0.0
        elif pn == 2:
            yc = round(rng.uniform(0.25, 0.50), 2)
        else:
            yc = round(float(np.clip(0.55 + (t - yolo_t) * 3.0 + rng.normal(0, 0.03), 0.50, 0.99)), 2)

        # Altitude (meters) — adversary cruises ~500m, interceptor climbs to match
        adv_alt = 500 + rng.normal(0, 20)
        int_alt = 100 if t < yolo_t else lerp(100, adv_alt, min(1, (t - yolo_t) / (kill_t - yolo_t)))

        brg = math.degrees(math.atan2(dlon, dlat)) % 360

        frames.append({
            "step": i + 1, "t": t,
            "adv_lat": adv_lat, "adv_lon": adv_lon, "adv_alt": adv_alt,
            "int_lat": int_lat, "int_lon": int_lon, "int_alt": int_alt,
            "dist_km": round(dist_km, 1),
            "bearing": round(brg, 1),
            "phase": phase, "pn": pn, "yc": yc,
        })

    return frames, iran_label, iran_lat, iran_lon


# ═════════════════════════════════════════════════════════════════════════
# 3D PYDECK MAP
# ═════════════════════════════════════════════════════════════════════════
def build_3d_map(base_name, frames, up_to, iran_label, iran_lat, iran_lon):
    b = AIRBASES[base_name]
    blat, blon = b["lat"], b["lon"]
    n = min(up_to, len(frames))

    # Center between Iran and base
    mid_lat = (blat + iran_lat) / 2
    mid_lon = (blon + iran_lon) / 2

    layers = []

    # --- Adversary trail (red) ---
    if n >= 2:
        adv_path = [[f["adv_lon"], f["adv_lat"], f["adv_alt"]] for f in frames[:n]]
        layers.append(pdk.Layer(
            "PathLayer",
            data=[{"path": adv_path, "color": [255, 50, 50]}],
            get_path="path",
            get_color="color",
            width_min_pixels=4,
            get_width=5,
        ))

    # --- Interceptor trail (blue, only if launched) ---
    if n >= 1 and frames[min(n - 1, len(frames) - 1)]["pn"] >= 3:
        int_path = [[f["int_lon"], f["int_lat"], f["int_alt"]]
                     for f in frames[:n] if f["t"] >= PHASE_YOLO]
        if len(int_path) >= 2:
            layers.append(pdk.Layer(
                "PathLayer",
                data=[{"path": int_path, "color": [50, 130, 255]}],
                get_path="path",
                get_color="color",
                width_min_pixels=4,
                get_width=5,
            ))

    # --- Current drone positions (scatterplot) ---
    points = []
    if n >= 1:
        cur = frames[n - 1]
        # Adversary
        points.append({
            "lon": cur["adv_lon"], "lat": cur["adv_lat"],
            "alt": cur["adv_alt"],
            "color": [255, 50, 50, 230], "radius": 1200,
            "label": "HOSTILE UAV",
        })
        # Interceptor (only if launched)
        if cur["pn"] >= 3:
            points.append({
                "lon": cur["int_lon"], "lat": cur["int_lat"],
                "alt": cur["int_alt"],
                "color": [50, 130, 255, 230], "radius": 1200,
                "label": "INTERCEPTOR",
            })

    if points:
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=points,
            get_position=["lon", "lat"],
            get_radius="radius",
            get_fill_color="color",
            pickable=True,
        ))

    # --- Base marker (large green) ---
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=[{"lon": blon, "lat": blat, "label": base_name}],
        get_position=["lon", "lat"],
        get_radius=2500,
        get_fill_color=[50, 200, 255, 180],
        pickable=True,
    ))

    # --- Iran launch site (red) ---
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=[{"lon": iran_lon, "lat": iran_lat, "label": f"🇮🇷 {iran_label}"}],
        get_position=["lon", "lat"],
        get_radius=2500,
        get_fill_color=[255, 80, 80, 180],
        pickable=True,
    ))

    # --- Arc from Iran to base (threat vector) ---
    layers.append(pdk.Layer(
        "ArcLayer",
        data=[{
            "source_lon": iran_lon, "source_lat": iran_lat,
            "target_lon": blon, "target_lat": blat,
        }],
        get_source_position=["source_lon", "source_lat"],
        get_target_position=["target_lon", "target_lat"],
        get_source_color=[255, 80, 80, 120],
        get_target_color=[50, 200, 255, 120],
        get_width=2,
        great_circle=True,
    ))

    # --- RF detection ring (80km ≈ 0.72°) ---
    rf_ring = [[blon + 0.72 * math.cos(a), blat + 0.72 * math.sin(a)]
               for a in np.linspace(0, 2 * math.pi, 64)]
    rf_ring.append(rf_ring[0])
    layers.append(pdk.Layer(
        "PathLayer",
        data=[{"path": rf_ring, "color": [255, 200, 50, 100]}],
        get_path="path", get_color="color", width_min_pixels=2,
    ))

    # --- YOLO detection ring (30km ≈ 0.27°) ---
    yolo_ring = [[blon + 0.27 * math.cos(a), blat + 0.27 * math.sin(a)]
                 for a in np.linspace(0, 2 * math.pi, 64)]
    yolo_ring.append(yolo_ring[0])
    layers.append(pdk.Layer(
        "PathLayer",
        data=[{"path": yolo_ring, "color": [50, 255, 50, 100]}],
        get_path="path", get_color="color", width_min_pixels=2,
    ))

    # --- Kill marker ---
    if n >= 1 and frames[n - 1]["pn"] == 4:
        cur = frames[n - 1]
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=[{"lon": cur["adv_lon"], "lat": cur["adv_lat"], "label": "KILL CONFIRMED"}],
            get_position=["lon", "lat"],
            get_radius=3000,
            get_fill_color=[50, 255, 50, 200],
            pickable=True,
        ))

    view = pdk.ViewState(
        latitude=mid_lat,
        longitude=mid_lon,
        zoom=6.5,
        pitch=45,
        bearing=10,
    )

    return pdk.Deck(
        layers=layers,
        initial_view_state=view,
        map_style="mapbox://styles/mapbox/satellite-streets-v12",
        tooltip={"text": "{label}"},
    )


# ═════════════════════════════════════════════════════════════════════════
# YOLO CAMERA — Visual Identification & Classification
# Purpose: RF says "something is out there". YOLO CONFIRMS it's a drone
# and provides precise tracking data for the RL interception policy.
# ═════════════════════════════════════════════════════════════════════════
def render_yolo(f):
    """Render security-camera-style YOLO detection overlay."""
    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=100)
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#0a0a0a")
    pn, conf, dist = f["pn"], f["yc"], f["dist_km"]

    ax.set_xlim(0, 640); ax.set_ylim(0, 480); ax.set_aspect("equal")
    ax.invert_yaxis()

    # Grainy sky
    noise = np.random.randint(8, 22, (48, 64))
    ax.imshow(noise, extent=[0, 640, 480, 0], cmap="gray", alpha=0.3, aspect="auto")

    # Scan lines
    for y in range(0, 480, 3):
        ax.axhline(y, color="#111111", lw=0.2, alpha=0.3)

    # Camera header
    ax.text(5, 12, "CAM-01  PTZ AUTO-TRACK", color="white", fontsize=7,
            fontfamily="monospace", fontweight="bold")
    ax.text(450, 12, f"FRAME {f['step']:04d}", color="white", fontsize=7,
            fontfamily="monospace")
    ax.text(5, 28, time.strftime("%Y-%m-%d %H:%M:%S"), color="#aaaaaa",
            fontsize=6, fontfamily="monospace")
    ax.plot(610, 12, "o", color="red", markersize=5)
    ax.text(620, 12, "REC", color="red", fontsize=6, fontfamily="monospace",
            fontweight="bold", va="center")

    if pn <= 1:
        # ── NO DETECTION ──
        ax.text(320, 200, "SCANNING", color="#333333", fontsize=22,
                fontfamily="monospace", ha="center", fontweight="bold")
        ax.text(320, 240, "No targets in view", color="#222222", fontsize=10,
                fontfamily="monospace", ha="center")
        ax.text(320, 280, "Awaiting RF handoff...", color="#1a1a1a", fontsize=9,
                fontfamily="monospace", ha="center")
        # Role explanation
        ax.text(320, 440, "ROLE: Visually identify & classify aerial targets",
                color="#222222", fontsize=7, fontfamily="monospace", ha="center")

    elif pn == 2:
        # ── RF DETECTED, CAMERA SLEWING ──
        ax.text(320, 160, "⚠ RF SIGNAL RECEIVED", color="#ffcc00", fontsize=16,
                fontfamily="monospace", ha="center", fontweight="bold")
        ax.text(320, 200, f"Bearing {f['bearing']:.0f}°  |  Range {dist:.0f} km",
                color="#ccaa00", fontsize=10, fontfamily="monospace", ha="center")
        ax.text(320, 240, "PTZ slewing to bearing...", color="#888800",
                fontsize=9, fontfamily="monospace", ha="center")
        ax.text(320, 280, "Waiting for visual acquisition", color="#666600",
                fontsize=8, fontfamily="monospace", ha="center")
        # Crosshair
        ax.axhline(240, color="#ffcc00", lw=0.8, ls="--", alpha=0.4)
        ax.axvline(320, color="#ffcc00", lw=0.8, ls="--", alpha=0.4)
        # Role
        ax.text(320, 440, "ROLE: Camera acquired RF bearing, seeking visual lock",
                color="#555500", fontsize=7, fontfamily="monospace", ha="center")

    else:
        # ── YOLO ACTIVE — VISUAL IDENTIFICATION CONFIRMED ──
        sz = max(30, min(140, 800 / max(dist, 2.0)))
        cx = 320 + np.random.normal(0, 12)
        cy = 220 + np.random.normal(0, 8)

        # Drone silhouette
        dc = "#222222"
        body = plt.Circle((cx, cy), sz * 0.15, color=dc, zorder=5)
        ax.add_patch(body)
        arm = sz * 0.4
        ax.plot([cx - arm, cx + arm], [cy, cy], color=dc, lw=3, zorder=5)
        ax.plot([cx, cx], [cy - arm, cy + arm], color=dc, lw=3, zorder=5)
        for ddx, ddy in [(-arm, 0), (arm, 0), (0, -arm), (0, arm)]:
            ax.add_patch(plt.Circle((cx + ddx, cy + ddy), sz * 0.12,
                                    fill=False, color=dc, lw=2, zorder=5))

        # YOLO bounding box
        bw, bh = sz * 1.1, sz * 0.9
        bx, by = cx - bw, cy - bh
        bbox_c = "#00ff00" if pn == 4 else "#ff0000"

        rect = patches.Rectangle((bx, by), bw * 2, bh * 2,
                                  lw=2.5, edgecolor=bbox_c, facecolor="none", zorder=6)
        ax.add_patch(rect)

        # Class label + confidence
        ax.text(bx, by - 4, f"UAV  {conf:.0%}", color="#000",
                fontsize=11, fontfamily="monospace", fontweight="bold", va="bottom",
                bbox=dict(boxstyle="square,pad=0.3", facecolor=bbox_c, alpha=0.9), zorder=7)

        # Confidence bar
        bar_y = by + bh * 2 + 10
        bar_w = bw * 2
        ax.add_patch(patches.Rectangle((bx, bar_y), bar_w, 8,
                     facecolor="#333333", edgecolor="none", zorder=6))
        ax.add_patch(patches.Rectangle((bx, bar_y), bar_w * conf, 8,
                     facecolor=bbox_c, edgecolor="none", zorder=6))
        ax.text(bx + bar_w + 5, bar_y + 4, f"{conf:.0%}", color=bbox_c,
                fontsize=8, fontfamily="monospace", fontweight="bold", va="center", zorder=7)

        # Distance
        ax.text(bx + bw * 2 + 5, by + 10, f"{dist:.0f}km", color="#88ccff",
                fontsize=9, fontfamily="monospace", fontweight="bold", zorder=7)

        # Threat classification
        if pn == 4:
            thr, tc = "TARGET NEUTRALIZED", "#00ff00"
        elif dist < 5:
            thr, tc = "THREAT: CRITICAL", "#ff0000"
        elif dist < 20:
            thr, tc = "THREAT: HIGH", "#ff8800"
        else:
            thr, tc = "THREAT: MEDIUM", "#ffcc00"
        ax.text(5, 450, thr, color=tc, fontsize=11,
                fontfamily="monospace", fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.3", facecolor="#000", alpha=0.8))

        # Role explanation
        role = "CONFIRMED: Object classified as hostile UAV by YOLOv8"
        if pn == 4:
            role = "TARGET NEUTRALIZED — visual confirmation complete"
        ax.text(5, 470, role, color="#448844" if pn == 4 else "#884444",
                fontsize=6.5, fontfamily="monospace")

    # Bottom bar
    ax.text(5, 475, f"BRG: {f['bearing']:.1f}°  |  RNG: {dist:.0f}km  |  YOLOv8-nano  |  Jetson Orin",
            color="#444444", fontsize=5.5, fontfamily="monospace")

    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    plt.tight_layout(pad=0.1)
    buf = BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ═════════════════════════════════════════════════════════════════════════
# LOG ENTRY
# ═════════════════════════════════════════════════════════════════════════
def log_entry(f):
    d, c, pn = f["dist_km"], f["yc"], f["pn"]
    if pn == 4:
        return '<span class="sc">✅ [KILL] TARGET INTERCEPTED — visual confirm via YOLO</span>'
    elif pn == 3 and d < 5:
        return f'<span class="rc">[YOLO] UAV conf:{c:.0%} | {d:.0f}km | CLOSING — RL policy active</span>'
    elif pn == 3:
        return f'<span class="wc">[YOLO] UAV conf:{c:.0%} | {d:.0f}km — interceptor pursuing</span>'
    elif pn == 2:
        return f'<span class="bc">[RF] Signal BRG:{f["bearing"]:.0f}° | {d:.0f}km — camera slewing</span>'
    else:
        return '<span style="color:#334455;">[SCAN] No contacts — monitoring</span>'


# ═════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═════════════════════════════════════════════════════════════════════════
def render_pipeline(active=0):
    p1, p2, p3, p4 = st.columns(4)
    def _c(n):
        return ("#44ff44", "#44ff44") if active >= n else ("#44ddff", "#1a2a3a")
    c1, b1 = _c(1); c2, b2 = _c(2); c3, b3 = _c(3); c4, b4 = _c(4)
    with p1:
        st.markdown(f'<div class="pb" style="border-color:{b1}"><div class="ptc" style="color:{c1}">📡 RF Detection</div>'
                    '<div class="pd">Detects radio signal<br>Range: 80km<br>Cost: $5,000</div></div>', unsafe_allow_html=True)
    with p2:
        st.markdown(f'<div class="pb" style="border-color:{b2}"><div class="ptc" style="color:{c2}">👁️ YOLO ID</div>'
                    '<div class="pd">Confirms it\'s a UAV<br>Visual classification<br>Cost: $1,000</div></div>', unsafe_allow_html=True)
    with p3:
        st.markdown(f'<div class="pb" style="border-color:{b3}"><div class="ptc" style="color:{c3}">🧠 RL Policy</div>'
                    '<div class="pd">PPO computes thrust<br>Inference &lt;1ms<br>Cost: $100</div></div>', unsafe_allow_html=True)
    with p4:
        st.markdown(f'<div class="pb" style="border-color:{b4}"><div class="ptc" style="color:{c4}">🚀 Intercept</div>'
                    '<div class="pd">Drone pursues &amp; kills<br>Speed: 5 m/s<br>Cost: $300</div></div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div style="color:#ff4444;font-family:Courier New;font-size:1.1em;'
                'letter-spacing:2px;">🛡️ MISSION CONTROL</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="sl">AIRBASE</div>', unsafe_allow_html=True)
    selected_base = st.selectbox("Airbase", list(AIRBASES.keys()), label_visibility="collapsed")

    st.markdown('<div class="sl">RL ALGORITHM</div>', unsafe_allow_html=True)
    st.selectbox("Algorithm", ["PPO (78% intercept)"], label_visibility="collapsed")

    st.markdown('<div class="sl">SCENARIO SEED</div>', unsafe_allow_html=True)
    if "seed_val" not in st.session_state:
        st.session_state.seed_val = 60
    sc, rc = st.columns([3, 1])
    with rc:
        if st.button("🎲", help="Random seed", use_container_width=True):
            st.session_state.seed_val = int(np.random.randint(0, 9999))
            st.rerun()
    with sc:
        seed = st.number_input("Seed", min_value=0, max_value=9999,
                               label_visibility="collapsed", key="seed_val")

    st.markdown('<div class="sl">PLAYBACK SPEED</div>', unsafe_allow_html=True)
    speed = st.radio("Speed", ["SLOW", "NORMAL", "FAST"], index=2,
                     horizontal=True, label_visibility="collapsed")

    st.markdown('<div class="sl" style="margin-top:8px;">SIM2REAL</div>', unsafe_allow_html=True)
    dr_enabled = st.toggle("Domain Randomization", value=True,
                           help="Randomize physics params each episode for sim2real robustness")

    st.markdown("")
    launch = st.button("🚀 LAUNCH MISSION", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="sl">DETECTION CHAIN</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:Courier New;font-size:0.68em;color:#88aacc;line-height:2.2;">'
        '1. 📡 <b>RF Sensor</b> → detects signal<br>'
        '2. 👁️ <b>YOLO Camera</b> → confirms UAV<br>'
        '3. 🧠 <b>RL Policy</b> → computes pursuit<br>'
        '4. 🚀 <b>Interceptor</b> → kills target</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sl">$ SYSTEM COST</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:Courier New;font-size:0.73em;line-height:1.8;">'
        '<span style="color:#88aacc;">RF Sensors</span><span style="color:#44ddff;float:right;">$5,000</span><br>'
        '<span style="color:#88aacc;">PTZ + YOLO</span><span style="color:#44ddff;float:right;">$1,000</span><br>'
        '<span style="color:#88aacc;">Jetson Orin</span><span style="color:#44ddff;float:right;">$100</span><br>'
        '<span style="color:#88aacc;">Pursuit Drone</span><span style="color:#44ddff;float:right;">$300</span><br>'
        '<div style="border-top:1px solid #1a3a2a;margin-top:3px;padding-top:3px;">'
        '<span style="color:#44ff44;font-weight:bold;">Total</span>'
        '<span style="color:#44ff44;font-weight:bold;float:right;">$6,400</span></div>'
        '<div style="color:#557788;font-size:0.85em;margin-top:3px;">vs Patriot: −99.999%</div>'
        '</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════
st.markdown('<div class="mt">🛡️ COUNTER-UAS COMMAND CENTER</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
# MISSION EXECUTION
# ═════════════════════════════════════════════════════════════════════════
if launch:
    frames, iran_label, iran_lat, iran_lon = generate_scenario(selected_base, seed=int(seed))
    dr_params = generate_domain_rand(seed=int(seed)) if dr_enabled else None
    total = len(frames)

    st.markdown(f'<div class="st2">SCENARIO: 🇮🇷 {iran_label.upper()}, IRAN → 🛡️ {selected_base.upper()}</div>',
                unsafe_allow_html=True)

    banner_ph = st.empty()
    banner_ph.markdown('<div class="bn ba">🛫 ADVERSARY DRONE LAUNCHED FROM IRAN</div>',
                       unsafe_allow_html=True)
    met_ph = st.empty()

    # Layout: 3D map (large) | YOLO camera + log
    map_col, right_col = st.columns([6, 4])
    with map_col:
        st.markdown('<div class="sh">🌍 3D TACTICAL MAP — PERSIAN GULF THEATER</div>',
                    unsafe_allow_html=True)
    with right_col:
        yolo_lbl = st.empty()
        yolo_lbl.markdown(
            '<div class="sh">👁️ YOLO VISUAL ID — <span style="color:#ff8844;">Why? RF detects signal, '
            'YOLO confirms it\'s a drone</span></div>', unsafe_allow_html=True)

    map_ph = map_col.empty()
    cam_ph = right_col.empty()

    # Log below YOLO
    with right_col:
        st.markdown('<div class="sh" style="margin-top:8px;">📋 DETECTION LOG</div>',
                    unsafe_allow_html=True)
    log_ph = right_col.empty()

    # Pipeline
    st.markdown("---")
    pipe_ph = st.empty()

    # Domain Randomization panel
    dr_ph = st.empty()

    # --- Determine map update schedule ---
    phase_frames = [0]
    last_pn = frames[0]["pn"]
    for idx, f in enumerate(frames):
        if f["pn"] != last_pn:
            phase_frames.append(idx)
            last_pn = f["pn"]
    phase_frames.append(total - 1)
    mids = [total // 5, 2 * total // 5, 3 * total // 5, 4 * total // 5]
    map_updates = sorted(set(phase_frames + mids))

    delay = SPEED_DELAY.get(speed, 0.08)
    log_lines = []

    for i in range(total):
        f = frames[i]
        pc = "mg" if f["pn"] == 4 else ("my" if f["pn"] >= 2 else "mv")

        # Telemetry
        met_ph.markdown(
            f'<div style="display:flex;gap:6px;">'
            f'<div class="mb" style="flex:1"><div class="ml2">STEP</div><div class="mv">{f["step"]}</div></div>'
            f'<div class="mb" style="flex:1"><div class="ml2">RANGE</div><div class="mr2">{f["dist_km"]:.0f}km</div></div>'
            f'<div class="mb" style="flex:1"><div class="ml2">BEARING</div><div class="mv">{f["bearing"]:.1f}°</div></div>'
            f'<div class="mb" style="flex:1"><div class="ml2">YOLO</div><div class="{pc}">{f["yc"]*100:.0f}%</div></div>'
            f'<div class="mb" style="flex:1"><div class="ml2">PHASE</div><div class="{pc}">{f["phase"]}</div></div>'
            f'<div class="mb" style="flex:1"><div class="ml2">COST</div><div class="mg">$350</div></div>'
            f'</div>', unsafe_allow_html=True)

        # Banner
        banners = {
            1: '<div class="bn bw">🛫 ADVERSARY CROSSING PERSIAN GULF — UNDETECTED</div>',
            2: '<div class="bn ba">📡 RF SIGNAL DETECTED — CAMERA SLEWING TO BEARING</div>',
            3: '<div class="bn ba">🚀 YOLO CONFIRMED UAV — INTERCEPTOR PURSUING</div>',
            4: '<div class="bn bs">✅ KILL CONFIRMED — TARGET NEUTRALIZED</div>',
        }
        banner_ph.markdown(banners[f["pn"]], unsafe_allow_html=True)

        # 3D Map
        if i in map_updates:
            deck = build_3d_map(selected_base, frames, i + 1, iran_label, iran_lat, iran_lon)
            map_ph.pydeck_chart(deck)

        # YOLO camera
        cam_img = render_yolo(f)
        cam_ph.image(cam_img, use_container_width=True)

        # Log
        log_lines.append(log_entry(f))
        log_ph.markdown(f'<div class="lg2">{"<br>".join(log_lines[-15:])}</div>',
                        unsafe_allow_html=True)

        # Pipeline
        with pipe_ph.container():
            render_pipeline(f["pn"])

        # Domain Randomization panel
        if dr_params:
            dr_ph.markdown(
                '<div style="margin-top:8px;">' +
                '<div class="sh">🔀 DOMAIN RANDOMIZATION — Sim2Real Transfer</div>' +
                '<div style="color:#557788;font-family:Courier New;font-size:0.65em;margin-bottom:6px;">' +
                'Physics parameters randomized each episode so the RL policy generalizes to the real world.</div>' +
                '<div style="display:flex;gap:6px;flex-wrap:wrap;">' +
                f'<div class="mb" style="flex:1;min-width:100px"><div class="ml2">Mass</div><div class="mv">{dr_params["drone_mass"]} kg</div><div style="color:#334455;font-size:0.6em">nominal: 1.0</div></div>' +
                f'<div class="mb" style="flex:1;min-width:100px"><div class="ml2">Thrust</div><div class="mv">{dr_params["max_force"]} N</div><div style="color:#334455;font-size:0.6em">nominal: 5.0</div></div>' +
                f'<div class="mb" style="flex:1;min-width:100px"><div class="ml2">Drag</div><div class="mv">{dr_params["drag_coeff"]}</div><div style="color:#334455;font-size:0.6em">nominal: 0.3</div></div>' +
                f'<div class="mb" style="flex:1;min-width:100px"><div class="ml2">Evader Spd</div><div class="mv">{dr_params["evader_speed"]} m/s</div><div style="color:#334455;font-size:0.6em">nominal: 2.0</div></div>' +
                f'<div class="mb" style="flex:1;min-width:100px"><div class="ml2">Obstacles</div><div class="mv">{dr_params["num_obstacles"]}</div><div style="color:#334455;font-size:0.6em">nominal: 5</div></div>' +
                f'<div class="mb" style="flex:1;min-width:100px"><div class="ml2">Sensor Noise</div><div class="mv">σ={dr_params["obs_noise_std"]}</div><div style="color:#334455;font-size:0.6em">nominal: 0</div></div>' +
                f'<div class="mb" style="flex:1;min-width:100px"><div class="ml2">Action Delay</div><div class="mv">{dr_params["action_delay"]} steps</div><div style="color:#334455;font-size:0.6em">nominal: 0</div></div>' +
                f'<div class="mb" style="flex:1;min-width:100px"><div class="ml2">Gravity</div><div class="mv">{dr_params["gravity"]} m/s²</div><div style="color:#334455;font-size:0.6em">nominal: 9.81</div></div>' +
                '</div></div>', unsafe_allow_html=True)

        time.sleep(delay)

else:
    # ═════════════════════════════════════════════════════════════════
    # STANDBY
    # ═════════════════════════════════════════════════════════════════
    st.markdown(f'<div class="st2">[ {selected_base.upper()} ] — MONITORING IRANIAN AIRSPACE</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="bn bw">SECTOR DEFENSE STANDBY — AWAITING LAUNCH COMMAND</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<div style="display:flex;gap:6px;">'
        '<div class="mb" style="flex:1"><div class="ml2">STEP</div><div class="mv">0</div></div>'
        '<div class="mb" style="flex:1"><div class="ml2">RANGE</div><div class="mr2">—</div></div>'
        '<div class="mb" style="flex:1"><div class="ml2">BEARING</div><div class="mv">—</div></div>'
        '<div class="mb" style="flex:1"><div class="ml2">YOLO</div><div class="mv">0%</div></div>'
        '<div class="mb" style="flex:1"><div class="ml2">PHASE</div><div class="my">STANDBY</div></div>'
        '<div class="mb" style="flex:1"><div class="ml2">COST</div><div class="mg">$350</div></div>'
        '</div>', unsafe_allow_html=True)

    preview_site = IRAN_LAUNCH_SITES[0]
    mc, rc2 = st.columns([6, 4])
    with mc:
        st.markdown('<div class="sh">🌍 3D TACTICAL MAP — PERSIAN GULF THEATER</div>',
                    unsafe_allow_html=True)
        deck = build_3d_map(selected_base, [], 0, preview_site["label"],
                            preview_site["lat"], preview_site["lon"])
        st.pydeck_chart(deck)
    with rc2:
        st.markdown(
            '<div class="sh">👁️ YOLO VISUAL ID — <span style="color:#ff8844;">'
            'Why? RF detects signal, YOLO confirms it\'s a drone</span></div>',
            unsafe_allow_html=True)
        st.markdown(
            '<div style="background:#050505;border:1px solid #1a2a3a;border-radius:4px;'
            'padding:25px;text-align:center;">'
            '<div style="color:#222;font-size:2.5em;">📷</div>'
            '<div style="color:#334455;font-family:Courier New;font-size:0.9em;margin-top:10px;">'
            'CAMERA OFFLINE</div>'
            '<div style="color:#223344;font-family:Courier New;font-size:0.7em;margin-top:8px;">'
            'PTZ camera activates when RF detects a signal.<br>'
            'YOLO then classifies the object as UAV / bird / aircraft.</div></div>',
            unsafe_allow_html=True)

        st.markdown('<div class="sh" style="margin-top:12px;">📋 DETECTION LOG</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="lg2" style="color:#334455;">Awaiting RF signal...</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    render_pipeline(0)

    if dr_enabled:
        dr_params_standby = generate_domain_rand(seed=int(seed))
        st.markdown(
            '<div style="margin-top:8px;">' +
            '<div class="sh">🔀 DOMAIN RANDOMIZATION — Sim2Real Transfer</div>' +
            '<div style="color:#557788;font-family:Courier New;font-size:0.65em;margin-bottom:6px;">' +
            'Physics parameters randomized each episode so the RL policy generalizes to the real world. '
            'Each seed samples different conditions. Toggle off in sidebar to use nominal values.</div>' +
            '<div style="display:flex;gap:6px;flex-wrap:wrap;">' +
            f'<div class="mb" style="flex:1;min-width:100px"><div class="ml2">Mass</div><div class="mv">{dr_params_standby["drone_mass"]} kg</div><div style="color:#334455;font-size:0.6em">nominal: 1.0</div></div>' +
            f'<div class="mb" style="flex:1;min-width:100px"><div class="ml2">Thrust</div><div class="mv">{dr_params_standby["max_force"]} N</div><div style="color:#334455;font-size:0.6em">nominal: 5.0</div></div>' +
            f'<div class="mb" style="flex:1;min-width:100px"><div class="ml2">Drag</div><div class="mv">{dr_params_standby["drag_coeff"]}</div><div style="color:#334455;font-size:0.6em">nominal: 0.3</div></div>' +
            f'<div class="mb" style="flex:1;min-width:100px"><div class="ml2">Evader Spd</div><div class="mv">{dr_params_standby["evader_speed"]} m/s</div><div style="color:#334455;font-size:0.6em">nominal: 2.0</div></div>' +
            f'<div class="mb" style="flex:1;min-width:100px"><div class="ml2">Obstacles</div><div class="mv">{dr_params_standby["num_obstacles"]}</div><div style="color:#334455;font-size:0.6em">nominal: 5</div></div>' +
            f'<div class="mb" style="flex:1;min-width:100px"><div class="ml2">Sensor Noise</div><div class="mv">σ={dr_params_standby["obs_noise_std"]}</div><div style="color:#334455;font-size:0.6em">nominal: 0</div></div>' +
            f'<div class="mb" style="flex:1;min-width:100px"><div class="ml2">Action Delay</div><div class="mv">{dr_params_standby["action_delay"]} steps</div><div style="color:#334455;font-size:0.6em">nominal: 0</div></div>' +
            f'<div class="mb" style="flex:1;min-width:100px"><div class="ml2">Gravity</div><div class="mv">{dr_params_standby["gravity"]} m/s²</div><div style="color:#334455;font-size:0.6em">nominal: 9.81</div></div>' +
            '</div></div>', unsafe_allow_html=True)
