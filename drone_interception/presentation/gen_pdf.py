"""
Generate professional PDF slide deck for Counter-UAS presentation.
Uses fpdf2. Landscape A4 with dark theme.
"""
from fpdf import FPDF, XPos, YPos
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Counter_UAS_Presentation.pdf")

# Colors
BG  = (10, 13, 20)
BG2 = (15, 19, 32)
TXT = (226, 232, 240)
MUT = (100, 116, 139)
CYA = (0, 212, 255)
RED = (255, 59, 59)
GRN = (0, 230, 118)
AMB = (255, 171, 0)
BDR = (30, 41, 59)

W = 297  # page width
H = 210  # page height
M = 15   # margin
TOTAL = 17


class P(FPDF):
    def __init__(self):
        super().__init__(orientation="L", unit="mm", format="A4")
        self.set_auto_page_break(auto=False)
        self.sn = 0

    def slide(self):
        self.sn += 1
        self.add_page()
        self.set_fill_color(*BG)
        self.rect(0, 0, W, H, "F")
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*MUT)
        self.text(W - 30, H - 6, f"{self.sn} / {TOTAL}")

    def t(self, txt, x=M, y=None, sz=24, c=TXT, st="B", w=260):
        if y is not None: self.set_y(y)
        self.set_x(x)
        self.set_font("Helvetica", st, sz)
        self.set_text_color(*c)
        self.cell(w, sz * 0.5, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def p(self, txt, x=M, sz=9, c=MUT, st="", w=130):
        self.set_x(x)
        self.set_font("Helvetica", st, sz)
        self.set_text_color(*c)
        self.multi_cell(w, sz * 0.48, txt)

    def b(self, txt, x=M, c=TXT, sz=8):
        y0 = self.get_y()
        self.set_fill_color(*CYA)
        self.ellipse(x, y0 + 1, 1.8, 1.8, "F")
        self.set_xy(x + 4, y0)
        self.set_font("Helvetica", "", sz)
        self.set_text_color(*c)
        self.cell(120, sz * 0.55, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def h(self, txt, x=M, c=CYA, sz=11):
        self.set_x(x)
        self.set_font("Helvetica", "B", sz)
        self.set_text_color(*c)
        self.cell(120, 6, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def box(self, x, y, w, h):
        self.set_fill_color(*BG2)
        self.set_draw_color(*BDR)
        self.rect(x, y, w, h, "DF")

    def tag(self, txt, x, y, c=CYA):
        tw = self.get_string_width(txt) + 8
        self.set_draw_color(*c)
        r, g, bl = c
        self.set_fill_color(r // 8, g // 8, bl // 8)
        self.rect(x, y, tw, 6, "DF")
        self.set_font("Helvetica", "B", 6)
        self.set_text_color(*c)
        self.set_xy(x, y)
        self.cell(tw, 6, txt, align="C")
        return x + tw + 3

    def kv(self, k, v, x=M, kc=RED, vc=MUT):
        self.set_x(x)
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*kc)
        kw = self.get_string_width(k) + 2
        self.cell(kw, 4.5, k)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*vc)
        self.cell(100, 4.5, v, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def tl(self, date, title, desc, x=M, tc=TXT):
        self.set_x(x)
        self.set_font("Helvetica", "B", 7)
        self.set_text_color(*RED)
        self.cell(20, 4, date)
        dx = self.get_x()
        self.set_draw_color(*BDR)
        self.line(dx, self.get_y(), dx, self.get_y() + 10)
        self.set_x(dx + 3)
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*tc)
        self.cell(100, 4, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_x(dx + 3)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*MUT)
        self.multi_cell(100, 3.3, desc)
        self.ln(2)

    def stat(self, val, lbl, x, y, c=GRN):
        self.set_xy(x, y)
        self.set_font("Helvetica", "B", 28)
        self.set_text_color(*c)
        self.cell(60, 14, val, align="C")
        self.set_xy(x, y + 14)
        self.set_font("Helvetica", "B", 6)
        self.set_text_color(*MUT)
        self.cell(60, 4, lbl, align="C")


def build():
    pdf = P()

    # ─── S1: TITLE ───
    pdf.slide()
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(*CYA)
    pdf.text(M, 58, "CHALLENGE III  -  AUTONOMOUS DEFENSE SYSTEMS")
    pdf.t("Counter-UAS:", y=62, sz=28)
    pdf.t("Autonomous Drone Interception", sz=28, c=CYA)
    pdf.t("via Reinforcement Learning", sz=28, c=CYA)
    pdf.ln(4)
    pdf.p("A $6,400 AI-powered system that detects, classifies, and intercepts hostile drones autonomously - at 0.2% the cost of conventional missile defense.", w=220, sz=10)
    pdf.ln(4)
    x = M
    for tg in ["PPO", "YOLO", "SIM2REAL", "EDGE AI", "DOMAIN RANDOMIZATION"]:
        x = pdf.tag(tg, x, pdf.get_y())

    # ─── S2: 2026 US-IRAN WAR ───
    pdf.slide()
    pdf.t("2026: The US-Iran War -", y=12, sz=22)
    pdf.t("Drones Are the Weapon", sz=22, c=RED)
    pdf.ln(2)
    # Quote
    y0 = pdf.get_y()
    pdf.set_draw_color(*RED)
    pdf.line(M, y0, M, y0 + 22)
    pdf.set_xy(M + 4, y0)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(176, 190, 197)
    pdf.multi_cell(125, 3.8, '"The ongoing US-Iran conflict of 2026 has become the world\'s first full-scale drone war. Iranian UAS swarms are striking US bases across the Gulf daily - and we are spending $3 million per intercept to shoot down $2,000 drones."')
    pdf.set_x(M + 4)
    pdf.set_font("Helvetica", "", 6)
    pdf.set_text_color(*MUT)
    pdf.cell(100, 3.5, "- CENTCOM operational briefing, 2026", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.p("Following the escalation in early 2026, Iran deployed mass-produced Shahed-136 and Mohajer-6 drones in coordinated swarms against US forward operating bases. ~45,000 US troops across the Gulf face daily drone incursions.", w=125, sz=8)
    pdf.ln(2)
    pdf.p("The US has expended over $4.2 billion on drone defense in 2026 alone. Patriot batteries running low. Pentagon calls the cost equation \"unsustainable.\"", w=125, sz=8, c=(176, 190, 197))

    # Right col - bases
    pdf.set_xy(155, 18)
    pdf.h("US Bases Under Daily Drone Attack (2026)", x=155)
    bases = [("Al Dhafra AB", "UAE", "CRITICAL", RED), ("Camp Arifjan", "Kuwait", "CRITICAL", RED),
             ("Al Udeid AB", "Qatar", "HIGH", AMB), ("Al Asad AB", "Iraq", "CRITICAL", RED),
             ("Camp Lemonnier", "Djibouti", "HIGH", AMB), ("Prince Sultan AB", "KSA", "HIGH", AMB)]
    for nm, co, th, cl in bases:
        pdf.set_x(155)
        pdf.set_font("Helvetica", "B", 8); pdf.set_text_color(*TXT); pdf.cell(42, 4.5, nm)
        pdf.set_font("Helvetica", "", 8); pdf.set_text_color(*MUT); pdf.cell(22, 4.5, co)
        pdf.set_font("Helvetica", "B", 8); pdf.set_text_color(*cl); pdf.cell(25, 4.5, th, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(3)
    pdf.h("The 2026 Cost Crisis", x=155, c=RED)
    pdf.kv("Iran's Shahed-136:", " $20,000 per drone (mass-produced)", x=155)
    pdf.kv("US Patriot PAC-3:", " $3,000,000 per intercept", x=155)
    pdf.kv("Daily attacks:", " 15-40 drones per wave", x=155, kc=AMB)
    pdf.kv("US burn rate:", " $45M-$120M per day on drone defense", x=155)

    # ─── S3: TIMELINE ───
    pdf.slide()
    pdf.t("How We Got Here:", y=10, sz=20)
    pdf.t("The Road to the 2026 Drone War", sz=20, c=RED)
    pdf.ln(4)
    pdf.tl("SEP 2019", "Abqaiq Attack (Saudi Aramco)", "18 Iranian drones bypass US Patriot batteries. $2B damage. First proof cheap drones defeat billion-dollar air defense.")
    pdf.tl("JAN 2024", "Tower 22 - 3 US Soldiers Killed", "$2,000 drone mimics returning US drone flight path. Sgt. Rivers, Sanders, Moffett killed.", tc=RED)
    pdf.tl("APR 2024", "Iran's 300+ Missile/Drone Strike", "US/allies intercept most at cost of $1.35 billion in one night.")

    pdf.set_xy(155, 35)
    pdf.tl("2024-25", "Houthi Red Sea Drone War", "Houthis attack shipping + US destroyers. USS Carney fires $2M SM-2 at $2K drones.", x=155)
    pdf.tl("EARLY 2026", "Full-Scale US-Iran Hostilities", "Mass drone swarms against all US Gulf bases. Shahed-136: hundreds/month. Patriot stockpiles depleting.", x=155, tc=RED)
    pdf.tl("NOW 2026", "The Drone Attrition Crisis", 'Pentagon: "We cannot afford to defend against cheap drones with expensive missiles." This is our solution.', x=155, tc=RED)

    pdf.set_xy(M, 168)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*RED)
    pdf.multi_cell(W - 2 * M, 5, "The 2026 Iran conflict has proven that cheap drones are the defining weapon of modern warfare.\nThe US needs a new answer - now.", align="C")

    # ─── S4: PROBLEM STATEMENT ───
    pdf.slide()
    pdf.set_font("Helvetica", "", 7); pdf.set_text_color(*CYA)
    pdf.text(M, 68, "PROBLEM STATEMENT")
    pdf.t("How do you stop a $500 drone", y=75, sz=26, c=TXT, w=W - 2 * M)
    pdf.t("without spending $3,000,000?", sz=26, c=RED, w=W - 2 * M)
    pdf.ln(6)
    pdf.p("Current defense systems create an unsustainable cost asymmetry. The attacker always wins the economics.", sz=11, c=MUT, w=W - 2 * M)

    # ─── S5: COST MISMATCH ───
    pdf.slide()
    pdf.t("The $3,000,000 Problem", y=10, sz=24, c=RED)
    pdf.ln(4)
    # Patriot
    pdf.box(M, 35, 118, 75)
    pdf.set_xy(M, 40); pdf.set_font("Helvetica", "B", 36); pdf.set_text_color(*RED)
    pdf.cell(118, 16, "$3M", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(M); pdf.set_font("Helvetica", "B", 7); pdf.set_text_color(*MUT)
    pdf.cell(118, 4, "PER PATRIOT INTERCEPT", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_xy(M + 5, 68); pdf.p("MIM-104 Patriot PAC-3 MSE\nPhased-array radar + human operator\nDesigned for ballistic missiles\nBattalion-level logistics required", sz=8, c=(176, 190, 197), w=108)
    # VS
    pdf.set_xy(133, 60); pdf.set_font("Helvetica", "B", 22); pdf.set_text_color(*MUT); pdf.cell(25, 14, "vs", align="C")
    # Ours
    pdf.box(160, 35, 118, 75)
    pdf.set_xy(160, 40); pdf.set_font("Helvetica", "B", 36); pdf.set_text_color(*GRN)
    pdf.cell(118, 16, "$6.4K", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(160); pdf.set_font("Helvetica", "B", 7); pdf.set_text_color(*MUT)
    pdf.cell(118, 4, "OUR TOTAL SYSTEM COST", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_xy(165, 68); pdf.p("RF sensor + PTZ camera + Jetson Nano\nAutonomous RL pursuit policy\n$300 interceptor drone (reusable)\nZero human in the loop", sz=8, c=(176, 190, 197), w=108)
    # Equation bar
    pdf.box(M, 120, W - 2 * M, 14)
    pdf.set_xy(M, 122); pdf.set_font("Helvetica", "B", 11); pdf.set_text_color(*CYA)
    pdf.cell(W - 2 * M, 10, "COST RATIO:   $3,000,000 / $6,400  =  469x cheaper", align="C")

    # ─── S6: SOLUTION PIPELINE ───
    pdf.slide()
    pdf.t("Our Solution: Autonomous Counter-UAS", y=10, sz=22, c=CYA)
    pdf.ln(1)
    pdf.p("A 4-stage detection and interception pipeline that operates without human intervention.", sz=9, w=250)
    pdf.ln(3)
    stages = [("01", "RF Detection", "Passive radio sensor\ndetects control signal\nRange: 80km | $5,000"),
              ("02", "YOLO Visual ID", "PTZ camera slews to\nbearing. YOLOv8 classifies\nUAV/bird/aircraft | $1,000"),
              ("03", "RL Policy", "PPO neural network\ncomputes pursuit thrust\n<10ms | $100 Jetson Nano"),
              ("04", "Intercept", "Pursuit drone launches\nAutonomous kinetic kill\nReusable | $300")]
    bw = 60
    for i, (num, title, desc) in enumerate(stages):
        x = M + i * (bw + 7)
        pdf.box(x, 42, bw, 50)
        pdf.set_xy(x, 44); pdf.set_font("Helvetica", "B", 7); pdf.set_text_color(*CYA)
        pdf.cell(bw, 4, num, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(x); pdf.set_font("Helvetica", "B", 10); pdf.set_text_color(*TXT)
        pdf.cell(bw, 6, title, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_xy(x + 3, pdf.get_y() + 2); pdf.set_font("Helvetica", "", 7); pdf.set_text_color(*MUT)
        pdf.multi_cell(bw - 6, 3.5, desc, align="C")
        if i < 3:
            pdf.set_xy(x + bw + 1, 63); pdf.set_font("Helvetica", "", 12); pdf.set_text_color(*MUT); pdf.cell(5, 5, ">")
    x = M
    pdf.set_y(100)
    for tg in ["FULLY AUTONOMOUS", "NO HUMAN IN LOOP", "EDGE INFERENCE", "REUSABLE"]:
        x = pdf.tag(tg, x, pdf.get_y(), c=GRN)

    # ─── S7: WHY RL ───
    pdf.slide()
    pdf.t("Why Reinforcement Learning?", y=10, sz=24, c=CYA)
    pdf.ln(3)
    pdf.h("Traditional Pursuit Algorithms")
    for it in ["Proportional Navigation: works for missiles, fails with obstacles and wind",
               "PID Controllers: manual tuning per drone, brittle to perturbations",
               "Path Planning (A*, RRT): too slow for real-time 3D pursuit at 5+ m/s"]:
        pdf.b(it, sz=8)
    pdf.ln(1)
    pdf.set_x(M); pdf.set_font("Helvetica", "B", 9); pdf.set_text_color(*RED)
    pdf.cell(130, 5, "All break down when conditions deviate from design assumptions.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_xy(155, 25); pdf.h("RL Advantage", x=155)
    for it in ["Learns optimal pursuit from millions of simulated encounters",
               "Handles uncertainty: wind, sensor noise, motor degradation",
               "Adapts to evasion: figure-8, reactive avoidance",
               "Real-time: single forward pass = action in <10ms",
               "Generalizes via domain randomization"]:
        pdf.set_y(pdf.get_y()); pdf.b(it, x=155, sz=8)
    pdf.box(M, 125, W - 2 * M, 12)
    pdf.set_xy(M, 127); pdf.set_font("Helvetica", "B", 10); pdf.set_text_color(*CYA)
    pdf.cell(W - 2 * M, 8, "observation (21-dim)  -->  MLP [256, 256]  -->  thrust (3-dim)  @ <10ms", align="C")

    # ─── S8: PPO ───
    pdf.slide()
    pdf.t("PPO: Proximal Policy Optimization", y=10, sz=22, c=CYA)
    pdf.ln(3)
    pdf.h("Why PPO?")
    for it in ["Stable training: clipped objective prevents catastrophic updates",
               "Sample efficient: 4 parallel envs, 2048-step rollouts",
               "Small memory: on-policy, no replay buffer",
               "Battle-tested: OpenAI Five, robotics, Gymnasium"]:
        pdf.b(it, sz=8)
    pdf.ln(2)
    pdf.h("Observation Space (21-dim)")
    obs = [("[0:3] Interceptor position", "[6:9] Target position", "[12:15] Relative vector"),
           ("[3:6] Interceptor velocity", "[9:12] Target velocity", "[15] Euclidean distance"),
           ("", "", "[16:21] Obstacle proximity (5 rays)")]
    for row in obs:
        pdf.set_x(M); pdf.set_font("Helvetica", "", 7.5); pdf.set_text_color(*CYA)
        for j, item in enumerate(row):
            pdf.set_x(M + j * 45)
            pdf.cell(44, 3.8, item)
        pdf.ln(4)

    # Right - hyperparams
    pdf.set_xy(160, 25); pdf.h("Hyperparameters", x=160)
    params = [("Learning Rate", "3e-4"), ("Rollout Steps", "2048/env"), ("Batch Size", "64"),
              ("Epochs", "10"), ("Gamma", "0.99"), ("GAE Lambda", "0.95"), ("Clip", "0.2"),
              ("Entropy", "0.01"), ("Network", "[256,256]"), ("Total Steps", "1,000,000"), ("Envs", "4")]
    for k, v in params:
        pdf.set_x(160); pdf.set_font("Helvetica", "", 8); pdf.set_text_color(*MUT); pdf.cell(35, 4, k)
        pdf.set_font("Helvetica", "B", 8); pdf.set_text_color(*CYA); pdf.cell(30, 4, v, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_xy(160, 145); pdf.h("Action Space (3-dim)", x=160)
    pdf.set_x(160); pdf.p("Continuous thrust [-1,1] in x,y,z. Scaled by MAX_FORCE. Hover auto-added to z.", sz=8, w=115)

    # ─── S9: REWARD STRUCTURE ───
    pdf.slide()
    pdf.t("Reward Structure:", y=10, sz=22, c=CYA)
    pdf.t("Teaching the Agent to Kill Efficiently", sz=14, c=TXT)
    pdf.ln(2)
    pdf.p("Reward function encodes operational objective: intercept fast, avoid obstacles, minimize energy (battery = cost).", w=250, sz=8)
    pdf.ln(3)
    pdf.h("Positive Rewards (Do This)", c=GRN, sz=10)
    rp = [("1. Progress  [+15 x delta_d]", "Main gradient. Rewards closing distance. Zero for orbiting."),
          ("2. Intercept Bonus  [+500 + speed]", "Massive terminal reward. Faster kill = bigger reward."),
          ("3. Proximity  [+1/(d+0.3) if d<3m]", "Exponential near target. Capped to prevent orbit-farming.")]
    for title, desc in rp:
        pdf.set_x(M); pdf.set_font("Helvetica", "B", 8); pdf.set_text_color(*GRN); pdf.cell(130, 4.5, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(M); pdf.set_font("Helvetica", "", 7); pdf.set_text_color(*MUT); pdf.cell(130, 3.5, desc, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

    pdf.set_xy(155, 52); pdf.h("Negative Rewards (Don't Do This)", x=155, c=RED, sz=10)
    rn = [("4. Time Penalty  [-0.5/step]", "KEY anti-orbiting. 500-step orbit = -250. Quick kill optimal."),
          ("5. Collision  [-75]", "Crash = drone replacement cost + obstacle proximity warning."),
          ("6. Energy  [-0.02 x ||a||^2]", "Penalizes thrust. Energy = battery = mission cost."),
          ("7. Out of Bounds  [-75]", "Leaving arena = lost drone. Same magnitude as collision.")]
    for title, desc in rn:
        pdf.set_x(155); pdf.set_font("Helvetica", "B", 8); pdf.set_text_color(*RED); pdf.cell(120, 4.5, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(155); pdf.set_font("Helvetica", "", 7); pdf.set_text_color(*MUT); pdf.cell(120, 3.5, desc, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

    pdf.box(M, 150, W - 2 * M, 12)
    pdf.set_xy(M, 152); pdf.set_font("Helvetica", "B", 8); pdf.set_text_color(*CYA)
    pdf.cell(W - 2 * M, 8, "R(s,a) = 15*d + [500+speed] + prox - 0.5 - 0.02||a||^2 - 75_collision - 75_OOB", align="C")

    # ─── S10: DOMAIN RANDOMIZATION ───
    pdf.slide()
    pdf.t("Domain Randomization for Sim2Real", y=10, sz=22, c=CYA)
    pdf.ln(1)
    pdf.p("Train across a wide distribution of physics so the real world is just another sample.", sz=9, w=250)
    pdf.ln(3)
    dr = [("Drone Mass", "0.7-1.5 kg", "Payload variation"), ("Max Thrust", "3.5-7.0 N", "Motor degradation"),
          ("Drag Coeff", "0.1-0.6", "Wind/air density"), ("Evader Speed", "1.0-3.5 m/s", "DJI to racing"),
          ("Obstacles", "2-8", "Open to urban"), ("Sensor Noise", "0-0.05", "IMU/baro drift"),
          ("Action Delay", "0-3 steps", "ESC/controller lag"), ("Gravity", "9.75-9.85", "Alt/lat variation")]
    bw = 60
    for i, (nm, vl, ds) in enumerate(dr):
        col, row = i % 4, i // 4
        x = M + col * (bw + 7); y = 42 + row * 38
        pdf.box(x, y, bw, 32)
        pdf.set_xy(x, y + 3); pdf.set_font("Helvetica", "B", 8); pdf.set_text_color(*CYA); pdf.cell(bw, 4, nm, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(x); pdf.set_font("Helvetica", "B", 13); pdf.set_text_color(*CYA); pdf.cell(bw, 7, vl, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(x); pdf.set_font("Helvetica", "", 7); pdf.set_text_color(*MUT); pdf.cell(bw, 4, ds, align="C")

    y0 = 125
    pdf.set_draw_color(*RED); pdf.line(M, y0, M, y0 + 15)
    pdf.set_xy(M + 4, y0); pdf.set_font("Helvetica", "I", 8); pdf.set_text_color(176, 190, 197)
    pdf.multi_cell(200, 3.8, '"If you train your policy across a wide range of simulated conditions, the real world becomes just another sample from that distribution." - Tobin et al., 2017')

    # ─── S11: SIM2REAL CHALLENGES ───
    pdf.slide()
    pdf.t("The Sim2Real Gap: What Can Go Wrong", y=10, sz=22, c=RED)
    pdf.ln(2)
    pdf.p("Training in sim is cheap and safe. But the real world doesn't follow your simulator's assumptions.", sz=9, w=250)
    pdf.ln(3)
    pdf.h("Challenges", c=RED, sz=10)
    ch = [("1. Unmodeled Aerodynamics", "Point-mass sim vs rotor wash, ground effect, vortex ring state."),
          ("2. Sensor Latency", "Instant sim obs vs 100ms GPS latency, IMU drift, dropped frames."),
          ("3. Wind & Turbulence", "Constant drag vs spatially varying, gusty, building turbulence."),
          ("4. Target Behavior", "Scripted figure-8 vs GPS waypoints, swarm, adversarial RL.")]
    for title, desc in ch:
        pdf.set_x(M); pdf.set_font("Helvetica", "B", 8); pdf.set_text_color(*RED); pdf.cell(130, 4.5, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(M); pdf.set_font("Helvetica", "", 7); pdf.set_text_color(*MUT); pdf.cell(130, 3.5, desc, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

    pdf.set_xy(155, 38); pdf.h("Our Mitigations", x=155, c=GRN, sz=10)
    mt = [("Domain Randomization", "8 params randomized/episode. Real world = another sample."),
          ("Observation Noise", "Gaussian noise in training makes policy robust to bad sensors."),
          ("Action Delay (0-3 steps)", "Simulates ESC, flight controller, compute latency."),
          ("Conservative Reward", "Energy penalty discourages maneuvers that only work in sim.")]
    for title, desc in mt:
        pdf.set_x(155); pdf.set_font("Helvetica", "B", 8); pdf.set_text_color(*GRN); pdf.cell(120, 4.5, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(155); pdf.set_font("Helvetica", "", 7); pdf.set_text_color(*MUT); pdf.cell(120, 3.5, desc, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

    # ─── S12: ENVIRONMENT ───
    pdf.slide()
    pdf.t("Training Environment: DroneInterceptionEnv", y=10, sz=20, c=CYA)
    pdf.ln(3)
    pdf.h("Arena")
    for it in ["20m x 20m x 8m 3D arena", "NumPy physics (10x faster than PyBullet)", "500 max steps, capture dist: 1.0m"]:
        pdf.b(it, sz=8)
    pdf.ln(2)
    pdf.h("Target Behavior")
    for it in ["Figure-8 evasive pattern + reactive avoidance within 4m", "Stochastic noise for unpredictability", "Velocity clamped to 2.5x evader speed"]:
        pdf.b(it, sz=8)
    pdf.set_xy(165, 25); pdf.h("Training Results", x=165)
    pdf.stat("582.7", "AVG REWARD", 165, 40, GRN)
    pdf.stat("132", "AVG STEPS TO KILL", 165, 70, CYA)
    pdf.stat("1.7 MB", "MODEL SIZE", 165, 100, AMB)

    # ─── S13: ARCHITECTURE ───
    pdf.slide()
    pdf.t("System Architecture", y=10, sz=24)
    pdf.ln(3)
    pdf.h("Edge Deployment Stack")
    hw = [("RF Sensor", "$5,000", "Passive radio, 80km range"), ("PTZ Camera + YOLOv8", "$1,000", "Visual classification"),
          ("Jetson Nano", "$100", "PPO+YOLO inference <10ms"), ("Interceptor Drone", "$300", "COTS quad, reusable")]
    for nm, co, ds in hw:
        pdf.set_x(M); pdf.set_font("Helvetica", "B", 9); pdf.set_text_color(*TXT); pdf.cell(50, 5, nm)
        pdf.set_font("Helvetica", "B", 9); pdf.set_text_color(*CYA); pdf.cell(18, 5, co)
        pdf.set_font("Helvetica", "", 8); pdf.set_text_color(*MUT); pdf.cell(60, 5, ds, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_xy(160, 25); pdf.h("Software Stack", x=160)
    swst = [("RL Training", "SB3 PPO"), ("Environment", "Gymnasium+NumPy"), ("Sim2Real", "Domain Rand"),
            ("Detection", "YOLOv8"), ("Edge", "TensorRT/Jetson"), ("Backend", "FastAPI"), ("Frontend", "React+MapLibre")]
    for k, v in swst:
        pdf.set_x(160); pdf.set_font("Helvetica", "", 8); pdf.set_text_color(*MUT); pdf.cell(32, 4.2, k)
        pdf.set_font("Helvetica", "B", 8); pdf.set_text_color(*CYA); pdf.cell(40, 4.2, v, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ─── S14: COST ANALYSIS TABLE ───
    pdf.slide()
    pdf.t("Cost Analysis: Inverting the Asymmetry", y=10, sz=22, c=GRN)
    pdf.ln(5)
    # Header
    cols = [("System", 55), ("Cost/Intercept", 42), ("Target", 48), ("Autonomous", 35), ("Reusable", 30)]
    pdf.set_font("Helvetica", "B", 7); pdf.set_text_color(*MUT)
    for nm, w in cols:
        pdf.cell(w, 5, nm.upper())
    pdf.ln(6)
    rows = [("Patriot PAC-3", "$3,000,000", "Ballistic missiles", "No", "No", RED),
            ("Iron Dome", "$50,000", "Rockets/mortars", "Semi", "No", RED),
            ("COYOTE", "$80,000", "Small UAS", "Semi", "No", AMB),
            ("OUR SYSTEM", "$300/intercept", "Small UAS", "YES (fully)", "YES", GRN)]
    for *vals, c in rows:
        pdf.set_x(M)
        for i, ((_, w), v) in enumerate(zip(cols, vals)):
            if i == 0: pdf.set_font("Helvetica", "B", 8); pdf.set_text_color(*(c if vals[0] == "OUR SYSTEM" else TXT))
            elif i == 1: pdf.set_font("Helvetica", "B", 8); pdf.set_text_color(*c)
            else: pdf.set_font("Helvetica", "", 8); pdf.set_text_color(*MUT)
            pdf.cell(w, 6, v)
        pdf.ln(7)
    pdf.ln(5)
    pdf.stat("469x", "CHEAPER THAN PATRIOT", M, pdf.get_y(), GRN)
    pdf.stat("267x", "CHEAPER THAN COYOTE", M + 68, pdf.get_y(), GRN)
    pdf.stat("$300", "MARGINAL COST/KILL", M + 136, pdf.get_y(), GRN)
    pdf.stat("<$500", "LESS THAN TARGET", M + 204, pdf.get_y(), GRN)
    pdf.set_xy(M, 170); pdf.set_font("Helvetica", "B", 10); pdf.set_text_color(*GRN)
    pdf.cell(W - 2 * M, 5, "Our interceptor costs less than the drone it kills - the asymmetry is INVERTED.", align="C")

    # ─── S15: LIVE DEMO ───
    pdf.slide()
    pdf.set_font("Helvetica", "", 7); pdf.set_text_color(*CYA); pdf.text(M, 68, "LIVE DEMONSTRATION")
    pdf.t("Counter-UAS Command Center", y=73, sz=26, c=CYA, w=W - 2 * M)
    pdf.ln(5)
    pdf.p("Real-time visualization of PPO model intercepting hostile drones over the Persian Gulf.", sz=10, c=MUT, w=W - 2 * M)
    pdf.p("PPO model runs live - each seed generates a unique episode with domain-randomized physics.", sz=10, c=MUT, w=W - 2 * M)
    pdf.ln(5)
    x = 75
    for tg in ["REAL PPO INFERENCE", "DOMAIN RANDOMIZATION", "MAPLIBRE GL", "FASTAPI"]:
        x = pdf.tag(tg, x, pdf.get_y())

    # ─── S16: FUTURE WORK ───
    pdf.slide()
    pdf.t("Future Work & Scalability", y=10, sz=24, c=CYA)
    pdf.ln(3)
    fut = [("Multi-Agent Swarm", "MAPPO for coordinated swarm interception."),
           ("ONNX Edge Deploy", "TensorRT on Jetson. <5ms at 100Hz."),
           ("Real Hardware", "Sim2real to Crazyflie 2.1 micro-drones."),
           ("Curriculum Learning", "Stationary -> linear -> figure-8 -> adversarial."),
           ("Sensor Fusion", "RF + YOLO + LIDAR into unified observation."),
           ("Mesh Network", "Multi-node detection sharing with handoff.")]
    bw = 82
    for i, (title, desc) in enumerate(fut):
        col, row = i % 3, i // 3
        x = M + col * (bw + 8); y = 38 + row * 45
        pdf.box(x, y, bw, 38)
        pdf.set_xy(x + 4, y + 4); pdf.set_font("Helvetica", "B", 10); pdf.set_text_color(*CYA); pdf.cell(bw - 8, 5, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(x + 4); pdf.set_font("Helvetica", "", 8); pdf.set_text_color(*MUT); pdf.multi_cell(bw - 8, 3.8, desc)

    # ─── S17: CLOSING ───
    pdf.slide()
    pdf.set_font("Helvetica", "", 7); pdf.set_text_color(*CYA); pdf.text(M, 52, "SUMMARY")
    pdf.t("A $6,400 AI system", y=58, sz=28, c=GRN, w=W - 2 * M)
    pdf.t("that solves the", sz=28, c=TXT, w=W - 2 * M)
    pdf.t("$3,000,000 problem.", sz=28, c=RED, w=W - 2 * M)
    pdf.ln(5)
    pdf.p("Reinforcement learning enables autonomous, low-cost drone interception - inverting the cost asymmetry that makes conventional defense unsustainable against the 2026 drone threat.", sz=10, c=MUT, w=W - 2 * M)
    pdf.ln(6)
    x = 55
    for tg in ["PPO", "DOMAIN RANDOMIZATION", "YOLO", "EDGE AI"]:
        x = pdf.tag(tg, x, pdf.get_y())
    x = pdf.tag("469x COST REDUCTION", x, pdf.get_y(), c=GRN)
    pdf.set_xy(M, 160); pdf.set_font("Helvetica", "", 12); pdf.set_text_color(*MUT)
    pdf.cell(W - 2 * M, 6, "Questions?", align="C")

    pdf.output(OUT)
    print(f"PDF saved: {OUT}")


if __name__ == "__main__":
    build()
