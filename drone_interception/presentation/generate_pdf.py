"""
Generate a professional PDF slide deck for Counter-UAS presentation.
Uses fpdf2 for PDF generation with dark theme matching the HTML slides.
"""

from fpdf import FPDF
import os

OUTPUT = os.path.join(os.path.dirname(__file__), "Counter_UAS_Presentation.pdf")

# Colors
BG = (10, 13, 20)
BG2 = (15, 19, 32)
TEXT = (226, 232, 240)
MUTED = (100, 116, 139)
CYAN = (0, 212, 255)
RED = (255, 59, 59)
GREEN = (0, 230, 118)
AMBER = (255, 171, 0)
BORDER = (30, 41, 59)


class SlidePDF(FPDF):
    def __init__(self):
        super().__init__(orientation="L", unit="mm", format="A4")
        self.set_auto_page_break(auto=False)
        self.slide_num = 0
        self.total_slides = 17

    def new_slide(self):
        self.slide_num += 1
        self.add_page()
        self.set_fill_color(*BG)
        self.rect(0, 0, 297, 210, "F")
        # Slide number
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*MUTED)
        self.set_xy(270, 198)
        self.cell(20, 5, f"{self.slide_num} / {self.total_slides}", align="R")

    def title_text(self, text, y=None, size=28, color=TEXT):
        self.set_font("Helvetica", "B", size)
        self.set_text_color(*color)
        if y:
            self.set_y(y)
        self.multi_cell(0, size * 0.45, text, align="L")

    def body_text(self, text, size=10, color=TEXT, style="", w=0):
        self.set_font("Helvetica", style, size)
        self.set_text_color(*color)
        self.multi_cell(w or (297 - 2 * self.l_margin), size * 0.45, text)

    def bullet(self, text, color=TEXT, indent=0, bold_prefix=""):
        x = self.get_x() + indent
        y = self.get_y()
        # Bullet dot
        self.set_fill_color(*CYAN)
        self.ellipse(x, y + 1.2, 2, 2, "F")
        self.set_xy(x + 5, y)
        if bold_prefix:
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(*color)
            self.cell(self.get_string_width(bold_prefix) + 1, 4.5, bold_prefix, ln=0)
            self.set_font("Helvetica", "", 9)
            rest = text[len(bold_prefix):] if text.startswith(bold_prefix) else text
            self.multi_cell(0, 4.5, rest)
        else:
            self.set_font("Helvetica", "", 9)
            self.set_text_color(*color)
            self.multi_cell(0, 4.5, text)

    def section_header(self, text, color=CYAN, size=12):
        self.set_font("Helvetica", "B", size)
        self.set_text_color(*color)
        self.cell(0, 7, text, ln=True)
        self.ln(1)

    def card_box(self, x, y, w, h):
        self.set_fill_color(*BG2)
        self.set_draw_color(*BORDER)
        self.rect(x, y, w, h, "DF")

    def overline(self, text, y=None):
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*CYAN)
        if y:
            self.set_y(y)
        self.cell(0, 4, text, ln=True, align="L")

    def quote_block(self, text, source=""):
        y = self.get_y()
        self.set_draw_color(*RED)
        self.line(self.l_margin, y, self.l_margin, y + 20)
        self.set_xy(self.l_margin + 4, y)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(176, 190, 197)
        self.multi_cell(120, 4.2, text)
        if source:
            self.set_font("Helvetica", "", 7)
            self.set_text_color(*MUTED)
            self.cell(120, 4, source)
            self.ln(2)

    def tag(self, text, x, y, color=CYAN):
        self.set_xy(x, y)
        w = self.get_string_width(text) + 8
        self.set_draw_color(*color)
        self.set_fill_color(color[0] // 10, color[1] // 10, color[2] // 10)
        self.rect(x, y, w, 6, "DF")
        self.set_font("Helvetica", "B", 6)
        self.set_text_color(*color)
        self.cell(w, 6, text, align="C")
        return x + w + 3

    def timeline_item(self, date, title, desc, title_color=TEXT):
        self.set_font("Helvetica", "B", 7)
        self.set_text_color(*RED)
        y = self.get_y()
        self.cell(22, 4, date)
        self.set_draw_color(*BORDER)
        x = self.get_x()
        self.line(x, y, x, y + 12)
        self.set_xy(x + 3, y)
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*title_color)
        self.cell(0, 4, title, ln=True)
        self.set_x(x + 3 + 22)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*MUTED)
        self.multi_cell(100, 3.5, desc)
        self.ln(1)


def build_pdf():
    pdf = SlidePDF()

    # ═══════════════════ SLIDE 1: TITLE ═══════════════════
    pdf.new_slide()
    pdf.overline("CHALLENGE III - AUTONOMOUS DEFENSE SYSTEMS", y=55)
    pdf.ln(3)
    pdf.title_text("Counter-UAS:", size=30, color=TEXT)
    pdf.title_text("Autonomous Drone Interception", size=30, color=CYAN)
    pdf.title_text("via Reinforcement Learning", size=30, color=CYAN)
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(200, 5.5, "A $6,400 AI-powered system that detects, classifies, and intercepts hostile drones autonomously - at 0.2% the cost of conventional missile defense.")
    pdf.ln(5)
    x = pdf.l_margin
    for t in ["PPO", "YOLO", "SIM2REAL", "EDGE AI", "DOMAIN RANDOMIZATION"]:
        x = pdf.tag(t, x, pdf.get_y())

    # ═══════════════════ SLIDE 2: 2026 US-IRAN WAR ═══════════════════
    pdf.new_slide()
    pdf.title_text("2026: The US-Iran War", y=12, size=22)
    pdf.title_text("Drones Are the Weapon", size=22, color=RED)
    pdf.ln(3)

    pdf.quote_block(
        '"The ongoing US-Iran conflict of 2026 has become the world\'s first full-scale drone war. Iranian UAS swarms are striking US bases across the Gulf daily - and we are spending $3 million per intercept to shoot down $2,000 drones."',
        "- CENTCOM operational briefing, 2026"
    )
    pdf.ln(3)
    pdf.body_text("Following the escalation in early 2026, Iran has deployed mass-produced Shahed-136 and Mohajer-6 drones in coordinated swarms against US forward operating bases. With ~45,000 US troops across the Persian Gulf, every base faces daily drone incursions.", size=9)
    pdf.ln(2)
    pdf.body_text("The US has expended over $4.2 billion on drone defense in 2026 alone. Patriot batteries are running low on interceptors. The Pentagon has called the cost equation \"unsustainable.\"", size=9, color=MUTED)

    # Right column - bases table
    pdf.set_xy(155, 15)
    pdf.section_header("US Bases Under Daily Drone Attack (2026)")
    bases = [
        ("Al Dhafra AB", "UAE", "CRITICAL", RED),
        ("Camp Arifjan", "Kuwait", "CRITICAL", RED),
        ("Al Udeid AB", "Qatar", "HIGH", AMBER),
        ("Al Asad AB", "Iraq", "CRITICAL", RED),
        ("Camp Lemonnier", "Djibouti", "HIGH", AMBER),
        ("Prince Sultan AB", "KSA", "HIGH", AMBER),
    ]
    for name, country, threat, color in bases:
        pdf.set_x(155)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*TEXT)
        pdf.cell(45, 5, name)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*MUTED)
        pdf.cell(25, 5, country)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*color)
        pdf.cell(25, 5, threat, ln=True)

    pdf.ln(3)
    pdf.set_x(155)
    pdf.section_header("The 2026 Cost Crisis", color=RED)
    cost_items = [
        ("Iran's Shahed-136:", " $20,000 per drone (mass-produced)"),
        ("US Patriot PAC-3:", " $3,000,000 per intercept"),
        ("Daily attacks:", " 15-40 drones per wave"),
        ("US burn rate:", " $45M-$120M per day on drone defense"),
    ]
    for bold, rest in cost_items:
        pdf.set_x(155)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*RED)
        pdf.cell(pdf.get_string_width(bold) + 1, 4.5, bold, ln=0)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*MUTED)
        pdf.cell(0, 4.5, rest, ln=True)

    # ═══════════════════ SLIDE 3: TIMELINE ═══════════════════
    pdf.new_slide()
    pdf.title_text("How We Got Here:", y=12, size=20)
    pdf.title_text("The Road to the 2026 Drone War", size=20, color=RED)
    pdf.ln(4)

    # Left column
    pdf.set_x(pdf.l_margin)
    pdf.timeline_item("SEP 2019", "Abqaiq Attack (Saudi Aramco)", "18 Iranian drones bypass US Patriot batteries. $2B damage. First proof cheap drones defeat billion-dollar air defense.")
    pdf.timeline_item("JAN 2024", "Tower 22 - 3 US Soldiers Killed", "$2,000 drone mimics returning US drone's flight path. Sgt. Rivers, Sanders, Moffett killed. First US combat deaths from Iranian drone.", title_color=RED)
    pdf.timeline_item("APR 2024", "Iran's 300+ Missile/Drone Strike on Israel", "US Navy, USAF intercept most at a cost of $1.35 billion in one night. Victory was more expensive than the attack.")

    # Right column
    pdf.set_xy(155, 40)
    pdf.timeline_item("2024-25", "Houthi Red Sea Drone War", "Iranian-backed Houthis attack shipping + US Navy destroyers with $2K drones. USS Carney fires $2M SM-2 missiles at hobby-grade UAS.")
    pdf.set_x(155)
    pdf.timeline_item("EARLY 2026", "Full-Scale US-Iran Hostilities", "Iran retaliates with mass drone swarms. Shahed-136 production: hundreds/month. Patriot stockpiles depleting faster than Lockheed can manufacture.", title_color=RED)
    pdf.set_x(155)
    pdf.timeline_item("NOW 2026", "The Drone Attrition Crisis", "Pentagon: \"We cannot afford to defend against cheap drones with expensive missiles.\" Urgent need for autonomous, low-cost counter-UAS.", title_color=RED)

    pdf.set_xy(pdf.l_margin, 170)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*RED)
    pdf.multi_cell(0, 5, "The 2026 Iran conflict has proven that cheap drones are the defining weapon of modern warfare.\nThe US needs a new answer - now.", align="C")

    # ═══════════════════ SLIDE 4: PROBLEM STATEMENT ═══════════════════
    pdf.new_slide()
    pdf.overline("PROBLEM STATEMENT", y=65)
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(*TEXT)
    pdf.cell(0, 14, "How do you stop a ", ln=False)
    pdf.set_text_color(*RED)
    pdf.cell(0, 14, "", ln=True)
    pdf.set_text_color(*TEXT)
    pdf.set_font("Helvetica", "B", 28)
    txt = "How do you stop a $500 drone"
    pdf.cell(0, 14, txt, align="C", ln=True)
    pdf.set_text_color(*RED)
    pdf.cell(0, 14, "without spending $3,000,000?", align="C", ln=True)
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 6, "Current defense systems create an unsustainable cost asymmetry.", align="C", ln=True)
    pdf.cell(0, 6, "The attacker always wins the economics.", align="C", ln=True)

    # ═══════════════════ SLIDE 5: COST MISMATCH ═══════════════════
    pdf.new_slide()
    pdf.title_text("The $3,000,000 Problem", y=12, size=24, color=RED)
    pdf.ln(5)

    # Patriot side
    pdf.card_box(15, 40, 120, 80)
    pdf.set_xy(20, 45)
    pdf.set_font("Helvetica", "B", 36)
    pdf.set_text_color(*RED)
    pdf.cell(110, 18, "$3M", align="C", ln=True)
    pdf.set_x(20)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*MUTED)
    pdf.cell(110, 5, "PER PATRIOT INTERCEPT", align="C", ln=True)
    pdf.set_xy(25, 75)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(176, 190, 197)
    pdf.multi_cell(105, 4, "MIM-104 Patriot PAC-3 MSE\nPhased-array radar + human operator\nDesigned for ballistic missiles\nBattalion-level logistics required", align="C")

    # VS
    pdf.set_xy(135, 65)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(*MUTED)
    pdf.cell(25, 14, "vs", align="C")

    # Our system side
    pdf.card_box(162, 40, 120, 80)
    pdf.set_xy(167, 45)
    pdf.set_font("Helvetica", "B", 36)
    pdf.set_text_color(*GREEN)
    pdf.cell(110, 18, "$6.4K", align="C", ln=True)
    pdf.set_x(167)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*MUTED)
    pdf.cell(110, 5, "OUR TOTAL SYSTEM COST", align="C", ln=True)
    pdf.set_xy(172, 75)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(176, 190, 197)
    pdf.multi_cell(105, 4, "RF sensor + PTZ camera + Jetson Nano\nAutonomous RL pursuit policy\n$300 interceptor drone (reusable)\nZero human in the loop", align="C")

    # Equation
    pdf.card_box(15, 130, 267, 15)
    pdf.set_xy(15, 132)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*CYAN)
    pdf.cell(267, 10, "COST RATIO:   $3,000,000 / $6,400 = 469x cheaper", align="C")

    # ═══════════════════ SLIDE 6: OUR SOLUTION ═══════════════════
    pdf.new_slide()
    pdf.title_text("Our Solution: Autonomous Counter-UAS", y=12, size=22, color=CYAN)
    pdf.ln(2)
    pdf.body_text("A 4-stage detection and interception pipeline that operates without human intervention.", size=10, color=MUTED)
    pdf.ln(5)

    stages = [
        ("01", "RF Detection", "Passive radio sensor\ndetects control signal\nRange: 80km\n$5,000"),
        ("02", "YOLO Visual ID", "PTZ camera slews to\nbearing. YOLOv8 classifies:\nUAV/bird/aircraft\n$1,000"),
        ("03", "RL Policy", "PPO neural network\ncomputes pursuit thrust\nInference: <10ms\n$100 (Jetson Nano)"),
        ("04", "Intercept", "Pursuit drone launches\nAutonomous kinetic kill\nReusable airframe\n$300"),
    ]
    x_start = 15
    box_w = 62
    gap = 6
    for i, (num, title, desc) in enumerate(stages):
        x = x_start + i * (box_w + gap)
        pdf.card_box(x, 48, box_w, 55)
        pdf.set_xy(x, 50)
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_text_color(*CYAN)
        pdf.cell(box_w, 4, num, align="C", ln=True)
        pdf.set_x(x)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*TEXT)
        pdf.cell(box_w, 6, title, align="C", ln=True)
        pdf.set_x(x + 3)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(*MUTED)
        pdf.multi_cell(box_w - 6, 3.8, desc, align="C")
        # Arrow
        if i < 3:
            ax = x + box_w + 1
            pdf.set_xy(ax, 72)
            pdf.set_font("Helvetica", "", 14)
            pdf.set_text_color(*MUTED)
            pdf.cell(4, 5, ">")

    pdf.ln(15)
    pdf.set_y(115)
    x = pdf.l_margin
    for t in ["FULLY AUTONOMOUS", "NO HUMAN IN LOOP", "EDGE INFERENCE", "REUSABLE"]:
        x = pdf.tag(t, x, pdf.get_y(), color=GREEN)

    # ═══════════════════ SLIDE 7: WHY RL ═══════════════════
    pdf.new_slide()
    pdf.title_text("Why Reinforcement Learning?", y=12, size=24, color=CYAN)
    pdf.ln(4)

    pdf.set_x(pdf.l_margin)
    pdf.section_header("Traditional Pursuit Algorithms")
    for item in [
        ("Proportional Navigation:", " works for missiles, fails with obstacles and wind"),
        ("PID Controllers:", " requires manual tuning per drone, brittle to perturbations"),
        ("Path Planning (A*, RRT):", " too slow for real-time 3D pursuit at 5+ m/s"),
    ]:
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*TEXT)
        pdf.cell(pdf.get_string_width(item[0]) + 1, 5, item[0], ln=0)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*MUTED)
        pdf.cell(0, 5, item[1], ln=True)
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*RED)
    pdf.cell(0, 5, "All break down when conditions deviate from design assumptions.", ln=True)

    pdf.set_xy(155, 30)
    pdf.section_header("RL Advantage")
    rl_items = [
        "Learns optimal pursuit from millions of simulated encounters",
        "Handles uncertainty: wind, sensor noise, motor degradation",
        "Adapts to evasion: trained against figure-8, reactive avoidance",
        "Real-time: single forward pass = action in <10ms",
        "Generalizes: domain randomization makes policy robust",
    ]
    for item in rl_items:
        pdf.set_x(155)
        pdf.bullet(item, indent=0)

    pdf.card_box(15, 130, 267, 12)
    pdf.set_xy(15, 132)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*CYAN)
    pdf.cell(267, 8, "RL Agent:  observation (21-dim)  -->  MLP [256, 256]  -->  thrust (3-dim)  @ <10ms", align="C")

    # ═══════════════════ SLIDE 8: PPO DETAILS ═══════════════════
    pdf.new_slide()
    pdf.title_text("PPO: Proximal Policy Optimization", y=12, size=22, color=CYAN)
    pdf.ln(3)

    pdf.section_header("Why PPO Specifically?")
    for item in [
        "Stable training: clipped objective prevents catastrophic updates",
        "Sample efficient: 4 parallel envs, 2048-step rollouts",
        "Small memory: on-policy, no replay buffer - runs on student laptops",
        "Battle-tested: OpenAI Five (Dota), robotics, Gymnasium benchmarks",
    ]:
        pdf.set_x(pdf.l_margin)
        pdf.bullet(item)

    pdf.ln(2)
    pdf.section_header("Observation Space (21-dim)")
    obs_items = [
        ("[0:3]", "Interceptor position (x, y, z)"),
        ("[3:6]", "Interceptor velocity"),
        ("[6:9]", "Target position"),
        ("[9:12]", "Target velocity"),
        ("[12:15]", "Relative vector to target"),
        ("[15]", "Euclidean distance"),
        ("[16:21]", "Obstacle proximity (5 rays)"),
    ]
    for code, desc in obs_items:
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*CYAN)
        pdf.cell(18, 4, code)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*MUTED)
        pdf.cell(0, 4, desc, ln=True)

    # Right column - hyperparams
    pdf.set_xy(155, 30)
    pdf.section_header("Hyperparameters")
    params = [
        ("Learning Rate", "3e-4"),
        ("Rollout Steps", "2048 per env"),
        ("Batch Size", "64"),
        ("Epochs/Rollout", "10"),
        ("Discount (gamma)", "0.99"),
        ("GAE Lambda", "0.95"),
        ("Clip Range", "0.2"),
        ("Entropy Coeff", "0.01"),
        ("Network Arch", "[256, 256]"),
        ("Training Steps", "1,000,000"),
        ("Parallel Envs", "4"),
    ]
    for name, val in params:
        pdf.set_x(155)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*MUTED)
        pdf.cell(40, 4.2, name)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*CYAN)
        pdf.cell(0, 4.2, val, ln=True)

    pdf.set_xy(155, 135)
    pdf.section_header("Action Space (3-dim)")
    pdf.set_x(155)
    pdf.body_text("Continuous thrust [-1, 1] in x, y, z.\nScaled by MAX_FORCE. Hover auto-added to z.", size=8, color=MUTED, w=120)

    # ═══════════════════ SLIDE 9: REWARD STRUCTURE ═══════════════════
    pdf.new_slide()
    pdf.title_text("Reward Structure:", y=10, size=22, color=CYAN)
    pdf.title_text("Teaching the Agent to Kill Efficiently", size=16, color=TEXT)
    pdf.ln(2)
    pdf.body_text("The reward function encodes our operational objective: intercept fast, avoid obstacles, minimize energy (battery = cost).", size=9, color=MUTED)
    pdf.ln(3)

    # Left - positive rewards
    pdf.section_header("Positive Rewards (Do This)", color=GREEN, size=10)
    rewards_pos = [
        ("1. Progress Reward  [+15 x delta_d]", "Main gradient signal. Rewards closing distance. Zero for orbiting, so agent MUST approach."),
        ("2. Interception Bonus  [+500 + speed]", "Massive terminal reward. Speed bonus = (max_steps - steps) x 0.5. Faster kill = bigger reward."),
        ("3. Proximity Bonus  [+1/(d+0.3)]", "Exponential reward within 3m. ~0.5 at 3m, ~3.0 at 0.5m. Capped to prevent orbit-farming."),
    ]
    for title, desc in rewards_pos:
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*GREEN)
        pdf.cell(130, 4.5, title, ln=True)
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(*MUTED)
        pdf.cell(130, 3.5, desc, ln=True)
        pdf.ln(1.5)

    # Right - negative rewards
    pdf.set_xy(155, 55)
    pdf.section_header("Negative Rewards (Don't Do This)", color=RED, size=10)
    rewards_neg = [
        ("4. Time Penalty  [-0.5 per step]", "KEY anti-orbiting. 500-step orbit costs -250. Quick intercept clearly optimal."),
        ("5. Collision Penalty  [-75]", "Crashing = drone replacement cost. Combined with proximity warning near obstacles."),
        ("6. Energy Penalty  [-0.02 x ||a||^2]", "Penalizes excessive thrust. Energy = battery = mission cost."),
        ("7. Out of Bounds  [-75]", "Leaving arena = mission failure (lost drone). Same magnitude as collision."),
    ]
    for title, desc in rewards_neg:
        pdf.set_x(155)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*RED)
        pdf.cell(130, 4.5, title, ln=True)
        pdf.set_x(155)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(*MUTED)
        pdf.cell(130, 3.5, desc, ln=True)
        pdf.ln(1.5)

    # Equation
    pdf.card_box(15, 155, 267, 12)
    pdf.set_xy(15, 157)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*CYAN)
    pdf.cell(267, 8, "R(s,a) = 15*delta_d + [500 + speed_bonus] + prox_bonus - 0.5 - 0.02||a||^2 - 75_collision - 75_OOB", align="C")

    # ═══════════════════ SLIDE 10: DOMAIN RANDOMIZATION ═══════════════════
    pdf.new_slide()
    pdf.title_text("Domain Randomization for Sim2Real Transfer", y=12, size=22, color=CYAN)
    pdf.ln(1)
    pdf.body_text("Train across a wide distribution of physics so the real world is just another sample.", size=10, color=MUTED)
    pdf.ln(3)

    dr_params = [
        ("Drone Mass", "0.7 - 1.5 kg", "Manufacturing variance, payload variation"),
        ("Max Thrust", "3.5 - 7.0 N", "Motor degradation, battery voltage drop"),
        ("Drag Coeff", "0.1 - 0.6", "Wind speed/direction, air density"),
        ("Evader Speed", "1.0 - 3.5 m/s", "DJI Mini to racing drones"),
        ("Obstacles", "2 - 8", "Open field to dense urban"),
        ("Sensor Noise", "0 - 0.05", "IMU drift, barometer error"),
        ("Action Delay", "0 - 3 steps", "ESC response, flight controller lag"),
        ("Gravity", "9.75 - 9.85", "Altitude & latitude variation"),
    ]
    box_w = 62
    gap = 6
    for i, (name, val, desc) in enumerate(dr_params):
        col = i % 4
        row = i // 4
        x = 15 + col * (box_w + gap)
        y = 48 + row * 42
        pdf.card_box(x, y, box_w, 36)
        pdf.set_xy(x, y + 3)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*CYAN)
        pdf.cell(box_w, 5, name, align="C", ln=True)
        pdf.set_x(x)
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(*CYAN)
        pdf.cell(box_w, 8, val, align="C", ln=True)
        pdf.set_x(x + 3)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(*MUTED)
        pdf.multi_cell(box_w - 6, 3.5, desc, align="C")

    pdf.set_xy(pdf.l_margin, 145)
    pdf.quote_block('"If you train your policy across a wide range of simulated conditions, the real world becomes just another sample from that distribution."', "- Tobin et al., 2017 (OpenAI)")

    # ═══════════════════ SLIDE 11: SIM2REAL CHALLENGES ═══════════════════
    pdf.new_slide()
    pdf.title_text("The Sim2Real Gap: What Can Go Wrong", y=12, size=22, color=RED)
    pdf.ln(1)
    pdf.body_text("Training in simulation is cheap and safe. But the real world doesn't follow your simulator's assumptions.", size=9, color=MUTED)
    pdf.ln(3)

    pdf.section_header("Challenges", color=RED, size=10)
    challenges = [
        ("1. Unmodeled Aerodynamics", "Sim uses point-mass + linear drag. Real: rotor wash, ground effect, vortex ring state."),
        ("2. Sensor Latency & Noise", "Sim observation is instant. Real GPS: 100ms latency, IMU drift, dropped frames."),
        ("3. Wind & Turbulence", "Sim drag is constant. Real wind: spatially varying, gusty, building turbulence."),
        ("4. Target Behavior", "Sim: scripted figure-8. Real: GPS waypoints, swarm coordination, adversarial RL."),
    ]
    for title, desc in challenges:
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*RED)
        pdf.cell(130, 4.5, title, ln=True)
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(*MUTED)
        pdf.cell(130, 3.5, desc, ln=True)
        pdf.ln(1.5)

    pdf.set_xy(155, 40)
    pdf.section_header("Our Mitigations", color=GREEN, size=10)
    mitigations = [
        ("Domain Randomization", "8 physics params randomized per episode. Real world = just another sample."),
        ("Observation Noise Injection", "Gaussian noise added to observations during training."),
        ("Action Delay Randomization", "0-3 step delay simulates ESC, flight controller, compute latency."),
        ("Conservative Reward Tuning", "Energy penalty discourages maneuvers that only work in sim."),
    ]
    for title, desc in mitigations:
        pdf.set_x(155)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*GREEN)
        pdf.cell(130, 4.5, title, ln=True)
        pdf.set_x(155)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(*MUTED)
        pdf.cell(130, 3.5, desc, ln=True)
        pdf.ln(1.5)

    # ═══════════════════ SLIDE 12: ENVIRONMENT ═══════════════════
    pdf.new_slide()
    pdf.title_text("Training Environment: DroneInterceptionEnv", y=12, size=22, color=CYAN)
    pdf.ln(3)

    pdf.section_header("Arena")
    for item in ["20m x 20m x 8m 3D arena", "NumPy physics (10x faster than PyBullet)", "500 max steps per episode", "Capture distance: 1.0m"]:
        pdf.set_x(pdf.l_margin)
        pdf.bullet(item)
    pdf.ln(2)
    pdf.section_header("Target Behavior")
    for item in ["Figure-8 evasive pattern with reactive avoidance", "Reactive evasion when interceptor within 4m", "Stochastic noise for unpredictability"]:
        pdf.set_x(pdf.l_margin)
        pdf.bullet(item)

    # Right - results
    pdf.set_xy(155, 30)
    pdf.section_header("Training Results")
    results = [("582.7", "AVG REWARD", GREEN), ("132", "AVG STEPS TO KILL", CYAN), ("1.7 MB", "MODEL SIZE", AMBER)]
    for val, label, color in results:
        pdf.set_x(155)
        pdf.set_font("Helvetica", "B", 24)
        pdf.set_text_color(*color)
        pdf.cell(60, 12, val, align="C", ln=True)
        pdf.set_x(155)
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_text_color(*MUTED)
        pdf.cell(60, 4, label, align="C", ln=True)
        pdf.ln(3)

    # ═══════════════════ SLIDE 13: ARCHITECTURE ═══════════════════
    pdf.new_slide()
    pdf.title_text("System Architecture", y=12, size=24)
    pdf.ln(3)

    pdf.section_header("Edge Deployment Stack")
    stack = [
        ("RF Sensor", "$5,000", "Passive radio detection, 80km range"),
        ("PTZ Camera + YOLOv8", "$1,000", "Visual classification: UAV/bird/aircraft"),
        ("NVIDIA Jetson Nano", "$100", "PPO + YOLO inference, <10ms"),
        ("Interceptor Drone", "$300", "COTS quadcopter, kinetic intercept, reusable"),
    ]
    for name, cost, desc in stack:
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*TEXT)
        pdf.cell(55, 5, name)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*CYAN)
        pdf.cell(20, 5, cost)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*MUTED)
        pdf.cell(0, 5, desc, ln=True)

    # Right - software stack
    pdf.set_xy(155, 30)
    pdf.section_header("Software Stack")
    sw = [
        ("RL Training", "Stable-Baselines3 PPO"),
        ("Environment", "Gymnasium + NumPy/PyBullet"),
        ("Sim2Real", "Domain Randomization"),
        ("Visual Detection", "YOLOv8"),
        ("Edge Inference", "TensorRT on Jetson"),
        ("Backend", "FastAPI + Python"),
        ("Frontend", "React + MapLibre GL"),
    ]
    for comp, tech in sw:
        pdf.set_x(155)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*MUTED)
        pdf.cell(40, 4.5, comp)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*CYAN)
        pdf.cell(0, 4.5, tech, ln=True)

    # ═══════════════════ SLIDE 14: COST ANALYSIS ═══════════════════
    pdf.new_slide()
    pdf.title_text("Cost Analysis: Inverting the Asymmetry", y=12, size=22, color=GREEN)
    pdf.ln(5)

    headers = ["System", "Cost/Intercept", "Target", "Autonomous", "Reusable"]
    col_w = [55, 45, 50, 40, 35]
    # Header
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_text_color(*MUTED)
    x = pdf.l_margin
    for h, w in zip(headers, col_w):
        pdf.set_x(x)
        pdf.cell(w, 5, h.upper())
        x += w
    pdf.ln(6)

    rows = [
        ("Patriot PAC-3", "$3,000,000", "Ballistic missiles", "No", "No", RED),
        ("Iron Dome (Tamir)", "$50,000", "Rockets, mortars", "Semi", "No", RED),
        ("COYOTE (Raytheon)", "$80,000", "Small UAS", "Semi", "No", AMBER),
        ("OUR SYSTEM", "$6,400 total / $300 per", "Small UAS", "YES", "YES", GREEN),
    ]
    for row in rows:
        *vals, color = row
        x = pdf.l_margin
        for i, (v, w) in enumerate(zip(vals, col_w)):
            pdf.set_x(x)
            if i == 0:
                pdf.set_font("Helvetica", "B", 8)
                pdf.set_text_color(*color if row[0] == "OUR SYSTEM" else TEXT)
            elif i == 1:
                pdf.set_font("Helvetica", "B", 8)
                pdf.set_text_color(*color)
            else:
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(*MUTED)
            pdf.cell(w, 6, v)
            x += w
        pdf.ln(7)

    # Stats
    pdf.ln(5)
    stats = [("469x", "CHEAPER THAN PATRIOT"), ("267x", "CHEAPER THAN COYOTE"), ("$300", "MARGINAL COST/KILL"), ("<$500", "LESS THAN TARGET")]
    x = 15
    for val, label in stats:
        pdf.set_xy(x, pdf.get_y())
        pdf.set_font("Helvetica", "B", 28)
        pdf.set_text_color(*GREEN)
        pdf.cell(62, 14, val, align="C")
        pdf.set_xy(x, pdf.get_y() + 14)
        pdf.set_font("Helvetica", "B", 6)
        pdf.set_text_color(*MUTED)
        pdf.cell(62, 4, label, align="C")
        x += 68

    pdf.set_xy(pdf.l_margin, 175)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*GREEN)
    pdf.cell(0, 5, "Our interceptor costs less than the drone it's killing - the cost asymmetry is INVERTED.", align="C")

    # ═══════════════════ SLIDE 15: LIVE DEMO ═══════════════════
    pdf.new_slide()
    pdf.overline("LIVE DEMONSTRATION", y=65)
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(*CYAN)
    pdf.cell(0, 14, "Counter-UAS Command Center", align="C", ln=True)
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 6, "Real-time visualization of the PPO model intercepting hostile drones over the Persian Gulf.", align="C", ln=True)
    pdf.cell(0, 6, "PPO model runs live - each seed generates a unique episode with domain-randomized physics.", align="C", ln=True)
    pdf.ln(8)
    x = 80
    for t in ["REAL PPO INFERENCE", "DOMAIN RANDOMIZATION", "MAPLIBRE GL", "FASTAPI"]:
        x = pdf.tag(t, x, pdf.get_y())

    # ═══════════════════ SLIDE 16: FUTURE WORK ═══════════════════
    pdf.new_slide()
    pdf.title_text("Future Work & Scalability", y=12, size=24, color=CYAN)
    pdf.ln(5)

    future = [
        ("Multi-Agent Swarm", "Extend PPO to MAPPO for coordinated swarm interception. Multiple interceptors collaborate."),
        ("ONNX Edge Deploy", "Export to ONNX + TensorRT on Jetson. Target: <5ms inference at 100Hz."),
        ("Real Hardware", "Sim2real transfer to Crazyflie 2.1. Validate domain randomization bridges the gap."),
        ("Curriculum Learning", "Progressive: stationary target -> linear -> figure-8 -> adversarial RL evader."),
        ("Sensor Fusion", "RF bearing + YOLO bounding box + LIDAR depth into unified RL observation."),
        ("Mesh Network", "Multiple counter-UAS nodes share detections. Sector coverage with handoff."),
    ]
    box_w = 85
    gap = 8
    for i, (title, desc) in enumerate(future):
        col = i % 3
        row = i // 3
        x = 15 + col * (box_w + gap)
        y = 45 + row * 50
        pdf.card_box(x, y, box_w, 42)
        pdf.set_xy(x + 4, y + 4)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*CYAN)
        pdf.cell(box_w - 8, 5, title, ln=True)
        pdf.set_x(x + 4)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*MUTED)
        pdf.multi_cell(box_w - 8, 3.8, desc)

    # ═══════════════════ SLIDE 17: CLOSING ═══════════════════
    pdf.new_slide()
    pdf.overline("SUMMARY", y=50)
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*TEXT)
    pdf.cell(0, 10, "A ", align="C", ln=False)
    # Build the closing text
    pdf.set_font("Helvetica", "B", 30)
    pdf.set_text_color(*GREEN)
    pdf.set_xy(pdf.l_margin, 60)
    pdf.cell(0, 14, "A $6,400 AI system", align="C", ln=True)
    pdf.set_font("Helvetica", "B", 30)
    pdf.set_text_color(*TEXT)
    pdf.cell(0, 14, "that solves the", align="C", ln=True)
    pdf.set_text_color(*RED)
    pdf.cell(0, 14, "$3,000,000 problem.", align="C", ln=True)
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(0, 5.5, "Reinforcement learning enables autonomous, low-cost drone interception -\ninverting the cost asymmetry that makes conventional defense unsustainable\nagainst the emerging drone threat.", align="C")
    pdf.ln(8)
    x = 60
    for t in ["PPO", "DOMAIN RANDOMIZATION", "YOLO", "EDGE AI", "469x COST REDUCTION"]:
        c = GREEN if "469" in t else CYAN
        x = pdf.tag(t, x, pdf.get_y(), color=c)

    pdf.set_xy(pdf.l_margin, 170)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 6, "Questions?", align="C")

    # Save
    pdf.output(OUTPUT)
    print(f"PDF saved to: {OUTPUT}")


if __name__ == "__main__":
    build_pdf()
