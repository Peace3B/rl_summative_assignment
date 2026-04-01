"""
Custom Gymnasium Environment: NGO Sponsorship Case Manager
=========================================================
Based on: Africa New Life Ministries (ANLM) - Post-Secondary Sponsorship Program
Mission:  An RL agent learns to optimally manage a caseload of sponsored students,
          deciding each week which students to prioritize for follow-up actions so
          that compliance deadlines are met, tuition is tracked, and donor engagement
          stays high — all under limited staff-time resources.

Environment ID: SponsorshipCaseManager-v0
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_STUDENTS   = 20          # students visible in one episode
MAX_WEEKS      = 26          # one 6-month programme cycle
N_TASKS        = 5           # tasks per student that the agent can trigger
STAFF_BUDGET   = 6           # staff-hours available per week (resource cap)

TASK_COST = [1, 1, 2, 1, 1]  # hours consumed per task type
# Task indices:
#  0 – Send reminder SMS/email
#  1 – Collect sponsor letter
#  2 – Process tuition payment
#  3 – Upload compliance photo
#  4 – Log progress report

# Student compliance deadline windows (weeks from enrolment)
DEADLINES = {
    "letter":  4,
    "payment": 6,
    "photo":   8,
    "report":  12,
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class SponsorshipCaseManagerEnv(gym.Env):
    """
    Observation space (per student × MAX_STUDENTS flattened):
        [weeks_until_letter_deadline,
         weeks_until_payment_deadline,
         weeks_until_photo_deadline,
         weeks_until_report_deadline,
         letter_done,
         payment_done,
         photo_done,
         report_done,
         donor_engagement_score,   # 0-1 float
         sponsorship_risk_flag]     # 0 or 1

    Shape: (MAX_STUDENTS * 10,) + 2 global features = 202

    Action space:  MultiDiscrete – for each student pick one of N_TASKS+1 actions
                   (0 = skip/idle, 1-5 = task index 0-4)
                   Shape: (MAX_STUDENTS,)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # ---- Observation space ----
        obs_dim = MAX_STUDENTS * 10 + 2          # per-student features + globals
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # ---- Action space ----
        # For each student: 0 = idle, 1-5 = task 0-4
        self.action_space = spaces.MultiDiscrete([N_TASKS + 1] * MAX_STUDENTS)

        # Internal state
        self.students = None
        self.week = 0
        self.np_random = np.random.default_rng()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.week = 0
        self.students = [self._new_student(i) for i in range(MAX_STUDENTS)]
        return self._get_obs(), {}

    def _new_student(self, idx):
        enrol_week = self.np_random.integers(0, 4)   # staggered enrolments
        return {
            "id":            idx,
            "enrol_week":    enrol_week,
            "letter_done":   False,
            "payment_done":  False,
            "photo_done":    False,
            "report_done":   False,
            "donor_score":   self.np_random.uniform(0.5, 1.0),
            "at_risk":       False,
            "reminder_sent": False,
        }

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action):
        assert self.students is not None, "Call reset() first."

        total_reward = 0.0
        hours_used   = 0

        for s_idx, act in enumerate(action):
            if act == 0:
                continue                           # idle – no cost, no reward
            task = act - 1                         # convert to 0-based task index
            cost = TASK_COST[task]
            if hours_used + cost > STAFF_BUDGET:
                total_reward -= 0.5               # penalty: over-budget
                continue
            hours_used += cost
            total_reward += self._apply_task(s_idx, task)

        # Time-based penalties for missed deadlines
        for s in self.students:
            age = self.week - s["enrol_week"]
            if age > DEADLINES["letter"]  and not s["letter_done"]:
                total_reward -= 1.0
                s["at_risk"] = True
            if age > DEADLINES["payment"] and not s["payment_done"]:
                total_reward -= 2.0
                s["at_risk"] = True
            if age > DEADLINES["photo"]   and not s["photo_done"]:
                total_reward -= 1.0
            if age > DEADLINES["report"]  and not s["report_done"]:
                total_reward -= 1.5
            # Donor engagement decay
            if s["at_risk"] and not s["reminder_sent"]:
                s["donor_score"] = max(0.0, s["donor_score"] - 0.05)

        self.week += 1
        terminated = self.week >= MAX_WEEKS
        truncated  = False

        # Bonus for full compliance at episode end
        if terminated:
            compliant = sum(
                1 for s in self.students
                if s["letter_done"] and s["payment_done"]
                and s["photo_done"] and s["report_done"]
            )
            total_reward += compliant * 5.0

        return self._get_obs(), total_reward, terminated, truncated, {}

    def _apply_task(self, s_idx, task):
        s = self.students[s_idx]
        reward = 0.0

        if task == 0:   # Send reminder
            if not s["reminder_sent"]:
                s["reminder_sent"] = True
                reward = 0.5
            else:
                reward = -0.2   # redundant reminder penalty

        elif task == 1:  # Collect sponsor letter
            if not s["letter_done"]:
                s["letter_done"] = True
                age = self.week - s["enrol_week"]
                # Earlier = better reward
                reward = 3.0 if age <= DEADLINES["letter"] else 1.0
                s["donor_score"] = min(1.0, s["donor_score"] + 0.1)
            else:
                reward = -0.1  # already done

        elif task == 2:  # Process tuition payment
            if not s["payment_done"]:
                s["payment_done"] = True
                age = self.week - s["enrol_week"]
                reward = 4.0 if age <= DEADLINES["payment"] else 1.5
            else:
                reward = -0.1

        elif task == 3:  # Upload compliance photo
            if not s["photo_done"]:
                s["photo_done"] = True
                reward = 2.0
                s["donor_score"] = min(1.0, s["donor_score"] + 0.05)
            else:
                reward = -0.1

        elif task == 4:  # Log progress report
            if not s["report_done"]:
                s["report_done"] = True
                reward = 2.5
            else:
                reward = -0.1

        return reward

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------
    def _get_obs(self):
        features = []
        for s in self.students:
            age = max(0, self.week - s["enrol_week"])
            features.extend([
                max(0, DEADLINES["letter"]  - age) / DEADLINES["letter"],
                max(0, DEADLINES["payment"] - age) / DEADLINES["payment"],
                max(0, DEADLINES["photo"]   - age) / DEADLINES["photo"],
                max(0, DEADLINES["report"]  - age) / DEADLINES["report"],
                float(s["letter_done"]),
                float(s["payment_done"]),
                float(s["photo_done"]),
                float(s["report_done"]),
                s["donor_score"],
                float(s["at_risk"]),
            ])
        # Global features
        features.append(self.week / MAX_WEEKS)             # time progress
        compliant = sum(
            1 for s in self.students
            if s["letter_done"] and s["payment_done"]
            and s["photo_done"] and s["report_done"]
        )
        features.append(compliant / MAX_STUDENTS)          # overall compliance rate
        return np.array(features, dtype=np.float32)

    # ------------------------------------------------------------------
    # Render (pygame)
    # ------------------------------------------------------------------
    def render(self):
        try:
            import pygame
            self._pygame_render()
        except ImportError:
            self._text_render()

    def _text_render(self):
        print(f"\n=== Week {self.week}/{MAX_WEEKS} ===")
        compliant = sum(
            1 for s in self.students
            if s["letter_done"] and s["payment_done"]
            and s["photo_done"] and s["report_done"]
        )
        print(f"Fully compliant students: {compliant}/{MAX_STUDENTS}")
        header = f"{'ID':>3} {'Ltr':>4} {'Pay':>4} {'Pho':>4} {'Rep':>4} {'Risk':>5} {'Score':>6}"
        print(header)
        for s in self.students:
            print(
                f"{s['id']:>3} "
                f"{'✓' if s['letter_done'] else '✗':>4} "
                f"{'✓' if s['payment_done'] else '✗':>4} "
                f"{'✓' if s['photo_done'] else '✗':>4} "
                f"{'✓' if s['report_done'] else '✗':>4} "
                f"{'⚠' if s['at_risk'] else ' ':>5} "
                f"{s['donor_score']:>6.2f}"
            )

    def _pygame_render(self):
        import pygame
        if not hasattr(self, "_screen") or self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((900, 560))
            pygame.display.set_caption("ANLM Sponsorship Case Manager – RL Agent")
            self._font_lg = pygame.font.SysFont("monospace", 18, bold=True)
            self._font_sm = pygame.font.SysFont("monospace", 13)
            self._clock  = pygame.time.Clock()

        screen = self._screen
        screen.fill((15, 20, 35))

        # Title bar
        title = self._font_lg.render(
            f"ANLM Sponsorship Manager  |  Week {self.week}/{MAX_WEEKS}", True, (220, 200, 100))
        screen.blit(title, (20, 12))

        # Column headers
        cols = ["ID", "Letter", "Payment", "Photo", "Report", "Risk", "Donor"]
        xs   = [20, 70, 155, 255, 345, 435, 510]
        for col, x in zip(cols, xs):
            lbl = self._font_sm.render(col, True, (150, 180, 220))
            screen.blit(lbl, (x, 45))

        pygame.draw.line(screen, (80, 100, 140), (15, 63), (640, 63), 1)

        GREEN  = (80, 200, 100)
        RED    = (220, 80,  80)
        YELLOW = (240, 200, 60)
        WHITE  = (220, 220, 220)

        for i, s in enumerate(self.students):
            y = 70 + i * 23
            pygame.draw.rect(screen, (25, 35, 55) if i % 2 == 0 else (20, 28, 48),
                             (15, y, 625, 21))

            def tick(done): return ("✓", GREEN) if done else ("✗", RED)

            screen.blit(self._font_sm.render(f"{s['id']:02d}", True, WHITE), (xs[0], y+3))
            for col_i, done_key in enumerate(["letter_done","payment_done","photo_done","report_done"]):
                ch, color = tick(s[done_key])
                screen.blit(self._font_sm.render(ch, True, color), (xs[col_i+1], y+3))
            risk_txt = "⚠ YES" if s["at_risk"] else "  no"
            screen.blit(self._font_sm.render(risk_txt, True, YELLOW if s["at_risk"] else (100,120,150)), (xs[5], y+3))
            score_color = GREEN if s["donor_score"] > 0.7 else (YELLOW if s["donor_score"] > 0.4 else RED)
            screen.blit(self._font_sm.render(f"{s['donor_score']:.2f}", True, score_color), (xs[6], y+3))

        # Stats panel
        compliant = sum(1 for s in self.students if s["letter_done"] and s["payment_done"]
                        and s["photo_done"] and s["report_done"])
        at_risk   = sum(1 for s in self.students if s["at_risk"])
        avg_score = np.mean([s["donor_score"] for s in self.students])

        stats = [
            f"Fully Compliant : {compliant:>3}/{MAX_STUDENTS}",
            f"At Risk         : {at_risk:>3}",
            f"Avg Donor Score : {avg_score:.3f}",
        ]
        for j, txt in enumerate(stats):
            screen.blit(self._font_lg.render(txt, True, (200, 210, 240)), (660, 80 + j*35))

        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def close(self):
        if hasattr(self, "_screen") and self._screen is not None:
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass
            self._screen = None


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------
gym.register(
    id="SponsorshipCaseManager-v0",
    entry_point="environment.custom_env:SponsorshipCaseManagerEnv",
)
