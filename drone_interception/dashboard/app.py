"""Interactive dashboard for RL-trained drone interception agents.

Run: streamlit run dashboard/app.py
"""

import os
import sys
import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.drone_env import DroneInterceptionEnv, ARENA_SIZE, ARENA_HEIGHT, CAPTURE_DISTANCE
from core.cost_aware_reward import CostAwareReward, COUNTER_UAS_COSTS

# Configuration
st.set_page_config(
    page_title="RL Drone Interception",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS for polished appearance
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.2em;
        font-weight: bold;
        color: #1565C0;
        text-align: center;
        padding: 0.5em 0;
    }
    .sub-header {
        font-size: 1.1em;
        color: #666;
        text-align: center;
        margin-bottom: 1.5em;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1em;
        color: white;
        text-align: center;
    }
    .cost-highlight {
        font-size: 2em;
        font-weight: bold;
        color: #4CAF50;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================
st.markdown('<div class="main-header">Low-Cost Drone Interception via RL</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Training a $300 drone to do the job of a $3M missile</div>', unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_json_safe(filepath: str) -> Optional[Dict]:
    """Load JSON file with error handling."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def try_load_model(algo_name: str):
    """Try to load a trained SB3 model."""
    model_path = f"./models/{algo_name}_interceptor.zip"
    if not os.path.exists(model_path):
        return None
    try:
        from stable_baselines3 import PPO, SAC, TD3
        algo_map = {"ppo": PPO, "sac": SAC, "td3": TD3}
        return algo_map[algo_name].load(model_path)
    except Exception:
        return None


def run_demo_episode(
    model,
    evader_speed: float = 2.0,
    num_obstacles: int = 5,
    arena_size: float = 10.0,
    capture_distance: float = 0.8,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run a single episode and record trajectory data for visualization."""
    env = DroneInterceptionEnv(
        render_mode=None,
        evader_speed=evader_speed,
        num_obstacles=num_obstacles,
        arena_size=arena_size,
        capture_distance=capture_distance,
    )
    cost_tracker = CostAwareReward()
    cost_tracker.reset()

    obs, info = env.reset(seed=seed)

    int_positions = [env.interceptor_pos.copy()]
    tgt_positions = [env.target_pos.copy()]
    rewards_log = []
    prev_distance = info["distance"]

    # Reward component tracking
    progress_rewards = []
    energy_penalties = []
    time_penalties = []
    proximity_bonuses = []

    done = False
    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        distance = info.get("distance", 0.0)
        cost_tracker.compute(action, distance, prev_distance, info)

        # Track reward components
        progress_rewards.append((prev_distance - distance) * 10.0)
        energy_penalties.append(-0.05 * np.sum(action ** 2))
        time_penalties.append(-0.1)
        prox = max(0, (3.0 - distance) * 2.0) if distance < 3.0 else 0.0
        proximity_bonuses.append(prox)

        prev_distance = distance
        int_positions.append(env.interceptor_pos.copy())
        tgt_positions.append(env.target_pos.copy())
        rewards_log.append(reward)

    cost_summary = cost_tracker.get_episode_summary()
    env.close()

    return {
        "interceptor_trajectory": np.array(int_positions),
        "target_trajectory": np.array(tgt_positions),
        "rewards": rewards_log,
        "intercepted": info.get("intercepted", False),
        "collision": info.get("collision", False),
        "timeout": info.get("timeout", False),
        "steps": len(rewards_log),
        "total_reward": sum(rewards_log),
        "final_distance": info.get("distance", 0.0),
        "cost_summary": cost_summary,
        "reward_components": {
            "Progress": sum(progress_rewards),
            "Energy Penalty": sum(energy_penalties),
            "Time Penalty": sum(time_penalties),
            "Proximity Bonus": sum(proximity_bonuses),
            "Interception Bonus": 100.0 if info.get("intercepted", False) else 0.0,
            "Collision Penalty": -50.0 if info.get("collision", False) else 0.0,
        },
    }


# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Live Demo",
    "Algorithm Comparison",
    "Cost Analysis",
    "Sim2Real Challenges",
])


# =============================================================================
# TAB 1: LIVE DEMO
# =============================================================================
with tab1:
    st.header("Live Interception Demo")
    st.markdown("Adjust parameters and watch the trained agent pursue the target drone.")

    # Sidebar controls
    with st.sidebar:
        st.header("Episode Parameters")
        evader_speed = st.slider("Evader Speed (m/s)", 0.5, 4.0, 2.0, 0.1,
                                  help="Higher = harder to catch")
        num_obstacles = st.slider("Number of Obstacles", 0, 10, 5, 1,
                                   help="More obstacles = more complex navigation")
        arena_size_param = st.slider("Arena Size (half-width, m)", 5.0, 15.0, 10.0, 0.5,
                                      help="Larger arena = longer chases")
        capture_dist = st.slider("Capture Distance (m)", 0.3, 2.0, 0.8, 0.1,
                                  help="How close the interceptor must get")
        demo_seed = st.number_input("Random Seed", 0, 10000, 42,
                                     help="Change for different scenarios")
        algo_choice = st.selectbox("Algorithm", ["PPO", "SAC", "TD3"])

    # Run episode button
    col_btn, col_status = st.columns([1, 2])
    with col_btn:
        run_btn = st.button("Run Episode", type="primary", use_container_width=True)

    if run_btn:
        model = try_load_model(algo_choice.lower())
        if model is None:
            st.warning(
                f"No trained {algo_choice} model found at `./models/{algo_choice.lower()}_interceptor.zip`. "
                f"Using random actions. Train a model first:\n"
                f"```\npython -m training.train_{algo_choice.lower()} --timesteps 500000\n```"
            )

        with st.spinner(f"Running episode with {algo_choice}..."):
            ep_data = run_demo_episode(
                model=model,
                evader_speed=evader_speed,
                num_obstacles=num_obstacles,
                arena_size=arena_size_param,
                capture_distance=capture_dist,
                seed=int(demo_seed),
            )

        # Display outcome
        with col_status:
            if ep_data["intercepted"]:
                st.success(f"TARGET INTERCEPTED in {ep_data['steps']} steps!")
            elif ep_data["collision"]:
                st.error(f"COLLISION after {ep_data['steps']} steps")
            elif ep_data["timeout"]:
                st.warning(f"TIMEOUT after {ep_data['steps']} steps (distance: {ep_data['final_distance']:.1f}m)")

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Steps", ep_data["steps"])
        m2.metric("Total Reward", f"{ep_data['total_reward']:.1f}")
        m3.metric("Final Distance", f"{ep_data['final_distance']:.2f}m")
        m4.metric("Est. Mission Cost", f"${ep_data['cost_summary']['total_mission_cost']:.2f}")

        # Animated 2D top-down view
        st.subheader("Chase Trajectory (Top-Down View)")
        int_traj = ep_data["interceptor_trajectory"]
        tgt_traj = ep_data["target_trajectory"]

        # Create animated plotly figure
        n_frames = len(int_traj)
        frame_step = max(1, n_frames // 50)  # Limit to ~50 frames for performance
        frames = []
        for i in range(1, n_frames, frame_step):
            frames.append(go.Frame(
                data=[
                    go.Scatter(
                        x=int_traj[:i, 0], y=int_traj[:i, 1],
                        mode="lines+markers",
                        name="Interceptor",
                        line=dict(color="#2196F3", width=2),
                        marker=dict(size=3),
                    ),
                    go.Scatter(
                        x=tgt_traj[:i, 0], y=tgt_traj[:i, 1],
                        mode="lines+markers",
                        name="Target",
                        line=dict(color="#F44336", width=2),
                        marker=dict(size=3),
                    ),
                    go.Scatter(
                        x=[int_traj[i-1, 0]], y=[int_traj[i-1, 1]],
                        mode="markers",
                        name="Interceptor (current)",
                        marker=dict(color="#0D47A1", size=12, symbol="circle"),
                        showlegend=False,
                    ),
                    go.Scatter(
                        x=[tgt_traj[i-1, 0]], y=[tgt_traj[i-1, 1]],
                        mode="markers",
                        name="Target (current)",
                        marker=dict(color="#B71C1C", size=12, symbol="diamond"),
                        showlegend=False,
                    ),
                ],
                name=str(i),
            ))

        fig_chase = go.Figure(
            data=[
                go.Scatter(x=int_traj[:1, 0], y=int_traj[:1, 1],
                           mode="markers", name="Interceptor",
                           marker=dict(color="#2196F3", size=10)),
                go.Scatter(x=tgt_traj[:1, 0], y=tgt_traj[:1, 1],
                           mode="markers", name="Target",
                           marker=dict(color="#F44336", size=10)),
            ],
            frames=frames,
            layout=go.Layout(
                xaxis=dict(range=[-arena_size_param * 1.1, arena_size_param * 1.1],
                           title="X (m)", scaleanchor="y"),
                yaxis=dict(range=[-arena_size_param * 1.1, arena_size_param * 1.1],
                           title="Y (m)"),
                title="Drone Chase Animation",
                height=500,
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(label="Play",
                             method="animate",
                             args=[None, {"frame": {"duration": 50}, "fromcurrent": True}]),
                        dict(label="Pause",
                             method="animate",
                             args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
                    ],
                )],
                shapes=[
                    dict(type="rect",
                         x0=-arena_size_param, y0=-arena_size_param,
                         x1=arena_size_param, y1=arena_size_param,
                         line=dict(color="gray", dash="dash")),
                ],
            ),
        )

        # Add interception marker if applicable
        if ep_data["intercepted"]:
            fig_chase.add_trace(go.Scatter(
                x=[int_traj[-1, 0]], y=[int_traj[-1, 1]],
                mode="markers",
                name="INTERCEPTION",
                marker=dict(color="#4CAF50", size=20, symbol="star"),
            ))

        st.plotly_chart(fig_chase, use_container_width=True)

        # Reward breakdown bar chart
        st.subheader("Reward Component Breakdown")
        components = ep_data["reward_components"]
        fig_reward = go.Figure(data=[
            go.Bar(
                x=list(components.keys()),
                y=list(components.values()),
                marker_color=["#4CAF50" if v > 0 else "#F44336" for v in components.values()],
                text=[f"{v:+.1f}" for v in components.values()],
                textposition="outside",
            )
        ])
        fig_reward.update_layout(
            title="Reward Components This Episode",
            yaxis_title="Reward Value",
            height=350,
        )
        st.plotly_chart(fig_reward, use_container_width=True)


# =============================================================================
# TAB 2: ALGORITHM COMPARISON
# =============================================================================
with tab2:
    st.header("Algorithm Comparison: PPO vs SAC vs TD3")

    # Try to load pre-computed comparison results
    comparison_data = load_json_safe("./results/comparison_results.json")

    # Also try to load individual training metrics
    algo_metrics = {}
    for algo in ["ppo", "sac", "td3"]:
        data = load_json_safe(f"./results/{algo}_metrics.json")
        if data:
            algo_metrics[algo.upper()] = data

    if comparison_data:
        st.subheader("Evaluation Results (100 Episodes Each)")

        # Create comparison dataframe
        table_data = []
        for algo_name, r in comparison_data.items():
            table_data.append({
                "Algorithm": algo_name,
                "Intercept %": f"{r.get('intercept_rate', 0) * 100:.1f}%",
                "Collision %": f"{r.get('collision_rate', 0) * 100:.1f}%",
                "Avg Reward": f"{r.get('avg_reward', 0):+.1f}",
                "Avg Steps": f"{r.get('avg_steps', 0):.0f}",
                "$/Interception": f"${r.get('cost_per_interception', 0):,.0f}",
            })

        st.table(table_data)

        # Bar charts
        algos = list(comparison_data.keys())
        intercept_rates = [comparison_data[a].get("intercept_rate", 0) * 100 for a in algos]
        avg_rewards = [comparison_data[a].get("avg_reward", 0) for a in algos]
        costs = [comparison_data[a].get("cost_per_interception", 0) for a in algos]

        col1, col2 = st.columns(2)

        with col1:
            fig_rates = go.Figure(data=[go.Bar(
                x=algos, y=intercept_rates,
                marker_color=["#2196F3", "#4CAF50", "#FF9800"][:len(algos)],
                text=[f"{r:.1f}%" for r in intercept_rates],
                textposition="outside",
            )])
            fig_rates.update_layout(title="Interception Rate by Algorithm",
                                     yaxis_title="Interception Rate (%)", height=400)
            st.plotly_chart(fig_rates, use_container_width=True)

        with col2:
            fig_costs = go.Figure(data=[go.Bar(
                x=algos, y=costs,
                marker_color=["#2196F3", "#4CAF50", "#FF9800"][:len(algos)],
                text=[f"${c:,.0f}" for c in costs],
                textposition="outside",
            )])
            fig_costs.update_layout(title="Cost per Interception by Algorithm",
                                     yaxis_title="$ per Interception", height=400)
            st.plotly_chart(fig_costs, use_container_width=True)

    # Learning curves from training metrics
    if algo_metrics:
        st.subheader("Learning Curves During Training")

        # Algorithm selection
        selected_algos = st.multiselect(
            "Select algorithms to compare:",
            list(algo_metrics.keys()),
            default=list(algo_metrics.keys()),
        )

        colors_map = {"PPO": "#2196F3", "SAC": "#4CAF50", "TD3": "#FF9800"}

        if selected_algos:
            fig_learn = make_subplots(rows=1, cols=2,
                                       subplot_titles=["Average Reward", "Interception Rate (%)"])

            for algo_name in selected_algos:
                data = algo_metrics[algo_name]
                history = data.get("metrics_history", [])
                if not history:
                    continue

                timesteps = [m["timestep"] for m in history]
                rewards = [m["avg_reward"] for m in history]
                rates = [m["intercept_rate"] for m in history]
                color = colors_map.get(algo_name, "#666666")

                fig_learn.add_trace(
                    go.Scatter(x=timesteps, y=rewards, mode="lines",
                               name=f"{algo_name} Reward", line=dict(color=color, width=2)),
                    row=1, col=1,
                )
                fig_learn.add_trace(
                    go.Scatter(x=timesteps, y=rates, mode="lines",
                               name=f"{algo_name} Intercept %", line=dict(color=color, width=2, dash="dash")),
                    row=1, col=2,
                )

            fig_learn.update_layout(height=400, title_text="Training Progress")
            fig_learn.update_xaxes(title_text="Timesteps")
            st.plotly_chart(fig_learn, use_container_width=True)

    if not comparison_data and not algo_metrics:
        st.info(
            "No training/evaluation data found. Train models first:\n"
            "```\n"
            "python -m training.train_ppo --timesteps 500000\n"
            "python -m training.train_sac --timesteps 500000\n"
            "python -m training.train_td3 --timesteps 500000\n"
            "python -m evaluation.compare_algorithms\n"
            "```"
        )


# =============================================================================
# TAB 3: COST ANALYSIS
# =============================================================================
with tab3:
    st.header("Cost-Effectiveness Analysis")
    st.markdown("**Our thesis**: RL-trained pursuit drones achieve a **1000x cost reduction** "
                "over current counter-UAS methods.")

    # Cost comparison table
    st.subheader("Counter-UAS Cost Comparison")

    cost_table_data = []
    for method, data in COUNTER_UAS_COSTS.items():
        cost_table_data.append({
            "Method": method,
            "Cost per Shot": f"${data['cost_per_shot']:,.0f}",
            "Success Rate": f"{data['success_rate']*100:.0f}%",
            "$ / Interception": f"${data['cost_per_interception']:,.0f}",
            "Notes": data["notes"],
        })

    st.table(cost_table_data)

    # Log-scale cost comparison chart
    methods = list(COUNTER_UAS_COSTS.keys())
    costs = [COUNTER_UAS_COSTS[m]["cost_per_interception"] for m in methods]
    colors = ["#4CAF50" if "RL" in m else "#E53935" for m in methods]

    fig_cost = go.Figure(data=[go.Bar(
        x=[m.replace(" (Raytheon)", "<br>(Raytheon)").replace(" (Ours)", "<br>(Ours)") for m in methods],
        y=costs,
        marker_color=colors,
        text=[f"${c:,.0f}" for c in costs],
        textposition="outside",
    )])
    fig_cost.update_layout(
        title="Cost per Interception (Log Scale)",
        yaxis_title="$ per Interception",
        yaxis_type="log",
        height=500,
        yaxis=dict(range=[1, 7.5]),
    )
    st.plotly_chart(fig_cost, use_container_width=True)

    # Interactive cost calculator
    st.subheader("Annual Cost Calculator")
    st.markdown("How much would it cost to protect a site from drone threats?")

    col_input, col_result = st.columns(2)

    with col_input:
        monthly_threats = st.slider("Drone threats per month", 1, 50, 10)
        rl_success_rate = st.slider("RL drone success rate (%)", 50, 95, 80) / 100.0

    annual_threats = monthly_threats * 12

    with col_result:
        st.markdown("**Annual Cost Comparison:**")

        annual_costs = {}
        for method, data in COUNTER_UAS_COSTS.items():
            annual_cost = annual_threats * data["cost_per_interception"]
            annual_costs[method] = annual_cost

        # Sort by cost (descending)
        sorted_methods = sorted(annual_costs.items(), key=lambda x: x[1], reverse=True)

        for method, annual_cost in sorted_methods:
            if "RL" in method:
                # Recalculate with user's success rate
                rl_annual = annual_threats * (350 / rl_success_rate)
                st.markdown(f"- **{method}**: **${rl_annual:,.0f}/year**")
            else:
                st.markdown(f"- **{method}**: ${annual_cost:,.0f}/year")

    # Break-even analysis
    st.subheader("Break-Even Analysis vs. Coyote Drone")
    st.markdown(
        "The Coyote (Raytheon) is the cheapest current kinetic interceptor at $80K/unit. "
        "At what success rate does our RL drone break even?"
    )

    success_rates = np.linspace(0.01, 1.0, 100)
    rl_costs = 350 / success_rates
    coyote_cost = 80000 / 0.80  # Coyote's estimated $/interception

    fig_breakeven = go.Figure()
    fig_breakeven.add_trace(go.Scatter(
        x=success_rates * 100, y=rl_costs,
        mode="lines", name="RL Pursuit Drone",
        line=dict(color="#4CAF50", width=3),
    ))
    fig_breakeven.add_trace(go.Scatter(
        x=[0, 100], y=[coyote_cost, coyote_cost],
        mode="lines", name="Coyote Drone (Raytheon)",
        line=dict(color="#E53935", width=2, dash="dash"),
    ))

    # Find break-even point
    breakeven_rate = 350 / coyote_cost * 100
    fig_breakeven.add_trace(go.Scatter(
        x=[breakeven_rate], y=[coyote_cost],
        mode="markers+text", name=f"Break-even: {breakeven_rate:.2f}%",
        marker=dict(color="gold", size=15, symbol="star"),
        text=[f"Break-even: {breakeven_rate:.2f}%"],
        textposition="top right",
    ))

    fig_breakeven.update_layout(
        title="Break-Even Analysis: RL Drone vs Coyote",
        xaxis_title="RL Drone Success Rate (%)",
        yaxis_title="$ per Interception",
        yaxis_type="log",
        height=450,
    )
    st.plotly_chart(fig_breakeven, use_container_width=True)

    st.markdown(
        f"**Result**: Our RL drone is cost-competitive with the Coyote at just "
        f"**{breakeven_rate:.2f}%** success rate. Even a barely-working prototype beats "
        f"an $80K guided munition on cost."
    )


# =============================================================================
# TAB 4: SIM2REAL CHALLENGES
# =============================================================================
with tab4:
    st.header("Sim-to-Real Transfer Challenges")

    # Load and render the sim2real analysis document
    sim2real_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs", "sim2real_analysis.md"
    )

    try:
        with open(sim2real_path, "r") as f:
            sim2real_content = f.read()
        st.markdown(sim2real_content)
    except FileNotFoundError:
        st.info("Sim2Real analysis document not found at docs/sim2real_analysis.md")

    # Interactive domain randomization demo
    st.subheader("Domain Randomization Impact")
    st.markdown(
        "Toggle randomization parameters to see how they affect training robustness. "
        "Each parameter simulates a real-world source of uncertainty."
    )

    dr_cols = st.columns(4)
    with dr_cols[0]:
        dr_mass = st.checkbox("Mass Randomization", value=True,
                               help="Simulate manufacturing variance [0.7-1.5 kg]")
        dr_force = st.checkbox("Force Randomization", value=True,
                                help="Simulate motor degradation [3.5-7.0 N]")
    with dr_cols[1]:
        dr_drag = st.checkbox("Drag Randomization", value=True,
                               help="Simulate wind conditions [0.1-0.6]")
        dr_evader = st.checkbox("Evader Speed Randomization", value=True,
                                 help="Different target types [1.0-3.5 m/s]")
    with dr_cols[2]:
        dr_noise = st.checkbox("Observation Noise", value=True,
                                help="Simulate sensor error [0-0.05 std]")
        dr_delay = st.checkbox("Action Delay", value=True,
                                help="Simulate motor lag [0-3 steps]")
    with dr_cols[3]:
        dr_obstacles = st.checkbox("Obstacle Randomization", value=True,
                                    help="Vary environment complexity [2-8]")
        dr_gravity = st.checkbox("Gravity Variation", value=True,
                                  help="Altitude variation [9.75-9.85]")

    active_count = sum([dr_mass, dr_force, dr_drag, dr_evader,
                        dr_noise, dr_delay, dr_obstacles, dr_gravity])

    st.markdown(f"**Active randomizations: {active_count}/8**")

    if active_count == 0:
        st.warning("No randomization active — the policy will be brittle in the real world!")
    elif active_count < 4:
        st.info("Partial randomization — some sim2real gap remains.")
    else:
        st.success("Strong randomization — policy should generalize well to real conditions.")

    # Parameter ranges table
    st.subheader("Randomization Parameter Ranges")
    param_table = [
        {"Parameter": "Drone Mass", "Base": "1.0 kg", "Range": "[0.7, 1.5] kg",
         "Real-World Source": "Manufacturing variance, payload differences"},
        {"Parameter": "Max Force", "Base": "5.0 N", "Range": "[3.5, 7.0] N",
         "Real-World Source": "Motor degradation, battery charge state"},
        {"Parameter": "Drag Coefficient", "Base": "0.3", "Range": "[0.1, 0.6]",
         "Real-World Source": "Wind conditions, air density at altitude"},
        {"Parameter": "Evader Speed", "Base": "2.0 m/s", "Range": "[1.0, 3.5] m/s",
         "Real-World Source": "Different target drone types"},
        {"Parameter": "Num Obstacles", "Base": "5", "Range": "[2, 8]",
         "Real-World Source": "Environmental complexity variation"},
        {"Parameter": "Obs. Noise", "Base": "0.0", "Range": "std [0, 0.05]",
         "Real-World Source": "IMU drift, sensor measurement error"},
        {"Parameter": "Action Delay", "Base": "0 steps", "Range": "[0, 3] steps",
         "Real-World Source": "Motor response lag, compute latency"},
        {"Parameter": "Gravity", "Base": "9.81 m/s^2", "Range": "[9.75, 9.85]",
         "Real-World Source": "Altitude and latitude variation"},
    ]
    st.table(param_table)


# =============================================================================
# SIDEBAR INFO
# =============================================================================
with st.sidebar:
    st.markdown("---")
    st.markdown("### About This Project")
    st.markdown(
        "This project trains RL agents to autonomously intercept "
        "enemy drones using commodity hardware (~$300), achieving "
        "a **1000x cost reduction** over traditional counter-UAS systems."
    )
    st.markdown("**Duke University — RL Course Project**")
    st.markdown("---")
    st.markdown("### Quick Commands")
    st.code("python -m training.train_ppo --timesteps 500000", language="bash")
    st.code("python -m training.train_sac --timesteps 500000", language="bash")
    st.code("python -m training.train_td3 --timesteps 500000", language="bash")
    st.code("python -m evaluation.compare_algorithms", language="bash")
