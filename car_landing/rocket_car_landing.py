"""Rocket Car Landing Lab
=======================
A small, self-contained Q-learning demo for a high-school AI talk.

Run:
    python rocket_car_landing.py

Keys:
    ENTER  Start training       F  Toggle fast/normal training
    D      Watch the AI         M  Try manual control
    P      Pause                R  Restart the current flight
    S      Save the Q-table     ESC Quit

Manual flight:
    SPACE  Main thruster (up)   A / D  Side thrusters
    DOWN   Top thruster (down)

Only PyGame is required. There is no hidden autopilot: during training,
every action is selected by epsilon-greedy Q-learning.
"""

from __future__ import annotations

import argparse
import copy
import math
import pickle
import random
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pygame


# ---------------------------------------------------------------------------
# 1. Small, visible constants make the physics easy to discuss and modify.
# ---------------------------------------------------------------------------

SCREEN_W, SCREEN_H = 1100, 700
SCENE_W, GROUND_Y = 800, 575
SCALE = 2.5                         # pixels per metre
WORLD_W = SCENE_W / SCALE
PAD_X = WORLD_W / 2
PAD_HALF_WIDTH = 17.0
CAR_CONTACT_Y = 3.0                 # centre height when the wheels touch down
DT = 0.05                           # one physics step = 50 ms
GRAVITY = 9.8
MAIN_ACCEL = 18.0
SIDE_ACCEL = 3.5
TOP_ACCEL = 7.0
SIDE_TORQUE = 0.65
MAX_STEPS = 800                      # 40 seconds; enough time to descend and brake

# A safe landing needs all five conditions, not merely reaching the ground.
SAFE_VX = 2.0
SAFE_VY = 2.2
SAFE_ANGLE = 0.16                   # radians; about 9 degrees
SAFE_OMEGA = 0.8

State = Tuple[int, int, int, int, int, int]


@dataclass
class FlightResult:
    """Numbers shown after a touchdown or a crash."""

    outcome: str
    reward: float
    steps: int
    vx: float
    vy: float
    angle: float


class RocketCarEnv:
    """The 2-D world: state, actions, physics, reward and termination.

    Coordinate system: x points right, y points up, and angle rotates
    counter-clockwise. Rendering converts y to screen coordinates later.
    """

    # Each action is a small, human-readable combination of thrusters.
    ACTIONS = (
        "COAST",
        "MAIN",
        "LEFT RCS",
        "RIGHT RCS",
        "MAIN + LEFT",
        "MAIN + RIGHT",
        "TOP",
    )

    # Uneven bins put more resolution near the pad and near zero velocity.
    DX_BINS = (-40, -10, 10, 40)
    ALTITUDE_BINS = (0, 10, 30, 70, 140)
    VX_BINS = (-4, -1, 1, 4)
    VY_BINS = (-10, -3, -1, 1, 5)
    ANGLE_BINS = (-0.25, -0.08, 0.08, 0.25)
    OMEGA_BINS = (-0.5, -0.1, 0.1, 0.5)

    def __init__(self, seed: int = 7) -> None:
        self.rng = random.Random(seed)
        self.x = PAD_X
        self.y = CAR_CONTACT_Y
        self.vx = self.vy = self.angle = self.omega = 0.0
        self.steps = 0
        self.last_action = 0
        self.previous_cost = 0.0
        self.total_reward = 0.0
        self.reset()

    def reset(self, difficulty: float = 1.0) -> State:
        """Start a trial; optional curriculum gradually increases difficulty.

        Easy early trials provide successful examples. This changes only the
        starting state, never the action-selection rule or the landing test.
        """
        difficulty = max(0.0, min(1.0, difficulty))
        # First learn attitude/position control near the ground. Only then add
        # a long descent. Asking for both new skills at once is much harder.
        control_progress = min(1.0, difficulty * 2.0)
        height_progress = max(0.0, difficulty * 2.0 - 1.0)
        self.x = PAD_X + self.rng.uniform(-1, 1) * (8 + 27 * control_progress)
        self.y = self.rng.uniform(
            35 + 110 * height_progress, 50 + 135 * height_progress
        )
        self.vx = self.rng.uniform(-1, 1) * (0.2 + 0.6 * control_progress)
        self.vy = self.rng.uniform(-0.5 - 1.5 * control_progress, 0.0)
        self.angle = self.rng.uniform(-1, 1) * (0.03 + 0.09 * control_progress)
        self.omega = self.rng.uniform(-1, 1) * (0.02 + 0.06 * control_progress)
        self.steps = 0
        self.last_action = 0
        self.previous_cost = self.flight_cost()
        self.total_reward = 0.0
        return self.get_state()

    def get_state(self) -> State:
        """Turn six continuous measurements into a discrete Q-table key.

        This is the same central idea as the Flappy Bird Q-table: nearby
        situations share a row, so the agent does not need infinite memory.
        """
        return (
            bisect_right(self.DX_BINS, self.x - PAD_X),
            bisect_right(self.ALTITUDE_BINS, self.y - CAR_CONTACT_Y),
            bisect_right(self.VX_BINS, self.vx),
            bisect_right(self.VY_BINS, self.vy),
            bisect_right(self.ANGLE_BINS, self.angle),
            bisect_right(self.OMEGA_BINS, self.omega),
        )

    def flight_cost(self) -> float:
        """A scalar 'how far from a good landing?' estimate for reward shaping.

        It supplies feedback during flight; it NEVER chooses an action.
        The policy still has to discover useful thruster combinations.
        """
        altitude = max(0.0, self.y - CAR_CONTACT_Y)
        position_cost = abs(self.x - PAD_X) / 55.0
        height_cost = altitude / 160.0
        # A good descent is fast high up and gentle near the ground. Penalizing
        # all downward speed would accidentally teach the agent to hover.
        desired_vy = -min(8.0, 1.0 + 0.07 * altitude)
        speed_cost = abs(self.vx) / 6.0 + abs(self.vy - desired_vy) / 10.0
        attitude_cost = abs(self.angle) / 0.5
        spin_cost = abs(self.omega) / 2.0
        return (
            1.5 * position_cost
            + 1.2 * height_cost
            + 0.6 * speed_cost
            + 0.8 * attitude_cost
            + 0.3 * spin_cost
        )

    def is_safe_landing(self) -> bool:
        return (
            abs(self.x - PAD_X) <= PAD_HALF_WIDTH - 2.0
            and abs(self.vx) < SAFE_VX
            and abs(self.vy) < SAFE_VY
            and abs(self.angle) < SAFE_ANGLE
            and abs(self.omega) < SAFE_OMEGA
        )

    def step(self, action: int) -> Tuple[State, float, bool, Optional[FlightResult]]:
        """Apply one action, advance physics, and return (s', r, done, info)."""
        self.last_action = action
        ax = ay = angular_accel = 0.0

        main_on = action in (1, 4, 5)
        left_on = action in (2, 4)       # exhaust left -> force to the right
        right_on = action in (3, 5)      # exhaust right -> force to the left

        if main_on:
            # The main nozzle rotates with the car, so attitude matters.
            ax += -math.sin(self.angle) * MAIN_ACCEL
            ay += math.cos(self.angle) * MAIN_ACCEL
        if left_on:
            ax += SIDE_ACCEL
            angular_accel -= SIDE_TORQUE
        if right_on:
            ax -= SIDE_ACCEL
            angular_accel += SIDE_TORQUE
        if action == 6:
            ay -= TOP_ACCEL

        ay -= GRAVITY

        # Semi-implicit Euler integration: update velocity, then position.
        self.vx += ax * DT
        self.vy += ay * DT
        self.omega += angular_accel * DT
        self.vx *= 0.998               # tiny drag keeps speeds numerically tame
        self.omega *= 0.992            # small rotational damping, not auto-leveling
        self.x += self.vx * DT
        self.y += self.vy * DT
        self.angle += self.omega * DT
        self.angle = (self.angle + math.pi) % (2 * math.pi) - math.pi
        self.steps += 1

        # Reward improvement, not just survival. The small time cost prevents
        # the agent from learning to hover forever above the landing pad.
        new_cost = self.flight_cost()
        reward = 8.0 * (self.previous_cost - new_cost) - 0.05
        if main_on:
            reward -= 0.003            # fuel is not free
        self.previous_cost = new_cost

        done = False
        result = None
        hit_ground = self.y <= CAR_CONTACT_Y
        out_of_bounds = self.x < -15 or self.x > WORLD_W + 15 or self.y > 220

        if hit_ground or out_of_bounds:
            done = True
            safe = hit_ground and self.is_safe_landing()
            impact_cost = min(
                5.0,
                abs(self.vx) / 4.0
                + abs(self.vy) / 6.0
                + abs(self.angle) / 0.5
                + abs(self.omega) / 2.0
                + max(0.0, abs(self.x - PAD_X) - PAD_HALF_WIDTH) / 40.0,
            )
            # Even a failed touchdown contains useful information: a gentle,
            # upright miss is better than a fast, spinning crash.
            reward += 300.0 if safe else -80.0 - 100.0 * impact_cost
            result = FlightResult(
                "LANDED" if safe else "CRASHED",
                self.total_reward + reward,
                self.steps,
                self.vx,
                self.vy,
                self.angle,
            )
        elif self.steps >= MAX_STEPS:
            done = True
            reward -= 500.0
            result = FlightResult(
                "TIMEOUT", self.total_reward + reward, self.steps, self.vx, self.vy, self.angle
            )

        self.total_reward += reward
        return self.get_state(), reward, done, result


class QLearningAgent:
    """Tabular Q-learning: Q(s,a) estimates future return after action a."""

    def __init__(self, seed: int = 7) -> None:
        self.q: Dict[State, List[float]] = {}
        self.rng = random.Random(seed)
        self.alpha = 0.22              # learning rate
        self.gamma = 1.0               # finite episodes: do not reward delaying a crash
        self.epsilon = 1.0             # initial exploration probability
        self.epsilon_min = 0.05            # keep exploring when curriculum gets harder
        self.epsilon_decay = 0.995
        self.action_count = len(RocketCarEnv.ACTIONS)
        self.best_q: Optional[Dict[State, List[float]]] = None
        self.best_score = -1.0
        self.best_epsilon = self.epsilon
        self.best_alpha = self.alpha

    def values(self, state: State) -> List[float]:
        if state not in self.q:
            self.q[state] = [0.0] * self.action_count
        return self.q[state]

    def choose_action(self, state: State, explore: bool = True) -> int:
        """Explore random actions early; exploit the best known action later."""
        if explore and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.action_count)
        values = self.values(state)
        best = max(values)
        # Random tie-breaking avoids always favouring action zero at the start.
        choices = [a for a, value in enumerate(values) if value == best]
        return self.rng.choice(choices)

    def learn(
        self, state: State, action: int, reward: float, next_state: State, done: bool
    ) -> None:
        """The one-line Q-learning update, split into readable steps."""
        old_value = self.values(state)[action]
        future_value = 0.0 if done else max(self.values(next_state))
        target = reward + self.gamma * future_value
        self.q[state][action] = old_value + self.alpha * (target - old_value)

    def finish_episode(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        # Large late updates can overwrite a good policy. Gradually lowering
        # the learning rate helps the agent consolidate what it has learned.
        self.alpha = max(0.03, self.alpha * 0.9995)

    def consider_checkpoint(self, recent_landing_rate: float) -> None:
        """Remember the best policy instead of assuming the last one is best."""
        if recent_landing_rate > self.best_score:
            self.best_score = recent_landing_rate
            self.best_q = copy.deepcopy(self.q)
            self.best_epsilon = self.epsilon
            self.best_alpha = self.alpha

    def restore_best(self) -> None:
        if self.best_q is not None:
            self.q = self.best_q
            self.epsilon = self.best_epsilon
            self.alpha = self.best_alpha

    def save(self, path: Path) -> None:
        """Save only local training data; no external services are involved."""
        payload = {
            "q": self.q,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "action_names": RocketCarEnv.ACTIONS,
        }
        with path.open("wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: Path) -> None:
        """Load a Q-table previously produced by this program."""
        with path.open("rb") as file:
            payload = pickle.load(file)
        if payload.get("action_names") != RocketCarEnv.ACTIONS:
            raise ValueError("This Q-table belongs to a different action set.")
        self.q = payload["q"]
        self.epsilon = payload.get("epsilon", self.epsilon_min)
        self.alpha = payload.get("alpha", self.alpha)


# ---------------------------------------------------------------------------
# 2. Rendering is separate from learning, so the physics can run headless.
# ---------------------------------------------------------------------------

class Renderer:
    BG = (10, 18, 34)
    PANEL = (18, 28, 46)
    WHITE = (235, 242, 250)
    MUTED = (143, 162, 184)
    CYAN = (58, 211, 222)
    ORANGE = (255, 157, 66)
    GREEN = (81, 220, 151)
    RED = (255, 99, 110)

    def __init__(self) -> None:
        pygame.display.set_caption("Rocket Car Landing Lab - Q-learning")
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.clock = pygame.time.Clock()
        self.title_font = pygame.font.Font(None, 34)
        self.font = pygame.font.Font(None, 24)
        self.small = pygame.font.Font(None, 19)
        self.big = pygame.font.Font(None, 58)
        rng = random.Random(12)
        self.stars = [(rng.randrange(SCENE_W), rng.randrange(70, 470)) for _ in range(90)]

    def text(
        self, value: str, x: int, y: int, color=WHITE, font=None, align: str = "left"
    ) -> None:
        image = (font or self.font).render(value, True, color)
        rect = image.get_rect()
        if align == "center":
            rect.midtop = (x, y)
        elif align == "right":
            rect.topright = (x, y)
        else:
            rect.topleft = (x, y)
        self.screen.blit(image, rect)

    def world_to_screen(self, x: float, y: float) -> Tuple[float, float]:
        return x * SCALE, GROUND_Y - y * SCALE

    def draw_background(self) -> None:
        self.screen.fill(self.BG)
        pygame.draw.rect(self.screen, self.PANEL, (SCENE_W, 0, SCREEN_W - SCENE_W, SCREEN_H))
        for x, y in self.stars:
            pygame.draw.circle(self.screen, (105, 130, 160), (x, y), 1)
        # Distant hills give the scene depth without external image assets.
        pygame.draw.polygon(
            self.screen, (20, 37, 54),
            [(0, 500), (0, 390), (130, 300), (250, 430), (390, 320),
             (550, 455), (680, 340), (800, 410), (800, 575)],
        )
        pygame.draw.rect(self.screen, (31, 47, 53), (0, GROUND_Y, SCENE_W, SCREEN_H - GROUND_Y))
        pygame.draw.line(self.screen, (92, 123, 119), (0, GROUND_Y), (SCENE_W, GROUND_Y), 3)

        # The landing pad is part of the goal, not just decoration.
        left = (PAD_X - PAD_HALF_WIDTH) * SCALE
        right = (PAD_X + PAD_HALF_WIDTH) * SCALE
        pygame.draw.rect(self.screen, (50, 75, 84), (left, GROUND_Y - 5, right - left, 10))
        pygame.draw.line(self.screen, self.CYAN, (left, GROUND_Y - 5), (right, GROUND_Y - 5), 3)
        for px in range(int(left) + 7, int(right), 18):
            pygame.draw.line(self.screen, self.ORANGE, (px, GROUND_Y), (px + 8, GROUND_Y), 3)
        self.text("LANDING ZONE", int(PAD_X * SCALE), GROUND_Y + 20, self.CYAN, self.small, "center")

    def draw_car(self, env: RocketCarEnv) -> None:
        cx, cy = self.world_to_screen(env.x, env.y)

        def point(local_x: float, local_y: float) -> Tuple[float, float]:
            # Local car coordinates also use y-up; rotate then convert to screen.
            c, s = math.cos(env.angle), math.sin(env.angle)
            wx = env.x + local_x * c - local_y * s
            wy = env.y + local_x * s + local_y * c
            return self.world_to_screen(wx, wy)

        def flame(origin, direction, length, color) -> None:
            ox, oy = origin
            dx, dy = direction
            mag = math.hypot(dx, dy)
            dx, dy = dx / mag, dy / mag
            px, py = -dy, dx
            tip = (ox + dx * length, oy + dy * length)
            pygame.draw.polygon(
                self.screen, color,
                [(ox + px * 3, oy + py * 3), tip, (ox - px * 3, oy - py * 3)],
            )

        # Flames make the selected action visible to the audience.
        action = env.last_action
        if action in (1, 4, 5):
            downward = (math.sin(env.angle), math.cos(env.angle))
            for local_x in (-5.0, 5.0):
                flame(point(local_x, -3), downward, 23, self.ORANGE)
        if action in (2, 4):
            flame(point(-9, 1), (-1, 0), 17, self.CYAN)
        if action in (3, 5):
            flame(point(9, 1), (1, 0), 17, self.CYAN)
        if action == 6:
            flame(point(0, 4), (0, -1), 17, (174, 130, 255))

        body = [point(-9, -3), point(9, -3), point(9, 2), point(5, 4),
                point(-5, 4), point(-9, 2)]
        pygame.draw.polygon(self.screen, (220, 230, 237), body)
        pygame.draw.polygon(self.screen, (92, 111, 128), body, 2)
        cockpit = [point(-3, 4), point(4, 4), point(2, 6), point(-2, 6)]
        pygame.draw.polygon(self.screen, self.CYAN, cockpit)
        for wheel_x in (-5.5, 5.5):
            wx, wy = point(wheel_x, -3.2)
            pygame.draw.circle(self.screen, (7, 12, 20), (int(wx), int(wy)), 5)
            pygame.draw.circle(self.screen, (145, 164, 176), (int(wx), int(wy)), 2)

    def draw_dashboard(
        self,
        env: RocketCarEnv,
        agent: QLearningAgent,
        mode: str,
        episode: int,
        target_episodes: int,
        rewards: deque,
        successes: deque,
        result: Optional[FlightResult],
        fast: bool,
        paused: bool,
        message: str,
    ) -> None:
        x = SCENE_W + 25
        self.text("ROCKET CAR", x, 25, self.WHITE, self.title_font)
        self.text("Q-LEARNING LANDING LAB", x, 60, self.CYAN, self.small)
        pygame.draw.line(self.screen, (48, 65, 84), (x, 91), (SCREEN_W - 25, 91))

        mode_label = {"menu": "READY", "training": "TRAINING", "watch": "AI DEMO", "manual": "MANUAL"}[mode]
        self.text(mode_label, x, 110, self.ORANGE if mode == "training" else self.CYAN)
        trial = f"{min(episode, target_episodes)} / {target_episodes}" if mode == "training" else str(episode)
        self.text(f"Trial: {trial}", x, 143)
        self.text(f"Exploration: {agent.epsilon * 100:4.1f}%", x, 174)
        self.text(f"Q states: {len(agent.q):,}", x, 205)
        rate = 100 * sum(successes) / len(successes) if successes else 0.0
        self.text(f"Recent landing rate: {rate:3.0f}%", x, 236,
                  self.GREEN if rate >= 70 else self.WHITE)

        # Reward sparkline: one point per completed trial.
        chart = (x, 280, SCREEN_W - 25 - x, 95)
        pygame.draw.rect(self.screen, (12, 21, 35), chart)
        pygame.draw.rect(self.screen, (48, 65, 84), chart, 1)
        self.text("REWARD PER TRIAL", x + 8, 286, self.MUTED, self.small)
        if len(rewards) > 1:
            values = list(rewards)
            low, high = min(values), max(values)
            span = max(1.0, high - low)
            points = []
            for i, value in enumerate(values):
                px = chart[0] + 6 + i * (chart[2] - 12) / (len(values) - 1)
                py = chart[1] + chart[3] - 7 - (value - low) / span * 62
                points.append((px, py))
            pygame.draw.lines(self.screen, self.CYAN, False, points, 2)

        self.text("FLIGHT TELEMETRY", x, 400, self.MUTED, self.small)
        self.text(f"Pad error   {env.x - PAD_X:+7.1f} m", x, 428)
        self.text(f"Altitude    {max(0, env.y - CAR_CONTACT_Y):7.1f} m", x, 456)
        self.text(f"Vertical v  {env.vy:+7.1f} m/s", x, 484)
        self.text(f"Horizontal v{env.vx:+7.1f} m/s", x, 512)
        self.text(f"Tilt        {math.degrees(env.angle):+7.1f} deg", x, 540)
        self.text(f"Action: {RocketCarEnv.ACTIONS[env.last_action]}", x, 576, self.ORANGE)

        # Safe envelope: students can compare numbers with visible motion.
        self.text("SAFE LANDING ENVELOPE", 25, 615, self.MUTED, self.small)
        self.text(f"on pad   |Vx| < {SAFE_VX}   |Vy| < {SAFE_VY}   "
                  f"|tilt| < {math.degrees(SAFE_ANGLE):.0f} deg", 25, 640, self.WHITE, self.small)
        self.text("ENTER train   D demo   M manual   F speed   P pause   R retry   S save",
                  25, 674, self.MUTED, self.small)

        if fast and mode == "training":
            self.text("FAST TRAINING", SCENE_W - 25, 22, self.ORANGE, self.small, "right")
        if paused:
            self.text("PAUSED", SCENE_W // 2, 95, self.ORANGE, self.big, "center")
        if result is not None:
            color = self.GREEN if result.outcome == "LANDED" else self.RED
            self.text(result.outcome, SCENE_W // 2, 95, color, self.big, "center")
        if message:
            self.text(message, SCENE_W // 2, 540, self.WHITE, self.small, "center")

    def draw_menu(self) -> None:
        overlay = pygame.Surface((SCENE_W, SCREEN_H), pygame.SRCALPHA)
        overlay.fill((5, 12, 24, 215))
        self.screen.blit(overlay, (0, 0))
        self.text("CAN A CAR LEARN TO LAND?", SCENE_W // 2, 150, self.WHITE, self.big, "center")
        self.text("No rules for landing. Only sensors, thrusters and rewards.",
                  SCENE_W // 2, 225, self.MUTED, self.font, "center")
        self.text("1  Observe the state", SCENE_W // 2, 285, self.CYAN, self.font, "center")
        self.text("2  Try an action", SCENE_W // 2, 320, self.CYAN, self.font, "center")
        self.text("3  Receive a reward and update Q", SCENE_W // 2, 355, self.CYAN, self.font, "center")
        self.text("Press ENTER to start learning", SCENE_W // 2, 435, self.ORANGE, self.title_font, "center")
        self.text("D: watch saved AI     M: fly it yourself", SCENE_W // 2, 485,
                  self.MUTED, self.font, "center")

    def render(self, *args, menu: bool = False) -> None:
        env = args[0]
        self.draw_background()
        self.draw_car(env)
        self.draw_dashboard(*args)
        if menu:
            self.draw_menu()
        pygame.display.flip()
        self.clock.tick(60)


# ---------------------------------------------------------------------------
# 3. Training and keyboard controls.
# ---------------------------------------------------------------------------

def train_headless(
    agent: QLearningAgent, env: RocketCarEnv, episodes: int, report_every: int = 100
) -> None:
    """Fast training without graphics; useful for testing or preparing a talk."""
    recent = deque(maxlen=50)
    for episode in range(1, episodes + 1):
        difficulty = min(1.0, episode / max(1.0, episodes * 0.6))
        state = env.reset(difficulty)
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, result = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
        agent.finish_episode()
        recent.append(result.outcome == "LANDED")
        if difficulty >= 1.0 and len(recent) == recent.maxlen and episode % 10 == 0:
            agent.consider_checkpoint(sum(recent) / len(recent))
        if report_every and (episode % report_every == 0 or episode == 1):
            rate = 100 * sum(recent) / len(recent)
            print(
                f"Episode {episode:4d} | epsilon={agent.epsilon:.3f} | "
                f"recent landings={rate:5.1f}% | states={len(agent.q):,}"
            )
    agent.restore_best()
    if agent.best_score >= 0:
        print(f"Restored best checkpoint (training landing rate={agent.best_score:.0%}).")


def evaluate(agent: QLearningAgent, env: RocketCarEnv, episodes: int = 100) -> float:
    """Measure greedy performance with exploration and learning switched off."""
    landed = 0
    outcomes = {}
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state, explore=False)
            state, _, done, result = env.step(action)
        success = result.outcome == "LANDED"
        landed += success
        outcomes[result.outcome] = outcomes.get(result.outcome, 0) + 1
    print(f"Greedy evaluation: {landed}/{episodes} safe landings; {outcomes}")
    return landed / episodes


def run_gui(
    agent: QLearningAgent, env: RocketCarEnv, target_episodes: int, model_path: Path
) -> None:
    renderer = Renderer()
    mode = "menu"
    episode = 1
    state = env.reset()
    rewards = deque(maxlen=80)
    successes = deque(maxlen=50)
    result = None
    fast = True
    paused = False
    message = "Press ENTER: random trials become a landing policy"
    wait_frames = 0

    def restart(new_mode: str) -> None:
        nonlocal mode, state, result, wait_frames, message, paused
        mode = new_mode
        difficulty = min(1.0, episode / max(1.0, target_episodes * 0.6))
        state = env.reset(difficulty if new_mode == "training" else 1.0)
        result = None
        wait_frames = 0
        paused = False
        message = {
            "training": "Learning from rewards - crashes are useful evidence",
            "watch": "Greedy policy: no random exploration",
            "manual": "SPACE main   A/D side jets   DOWN top jet",
        }.get(new_mode, message)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type != pygame.KEYDOWN:
                continue
            key = event.key
            if key == pygame.K_ESCAPE:
                pygame.quit()
                return
            if key == pygame.K_RETURN:
                restart("training")
            elif key == pygame.K_d:
                restart("watch")
            elif key == pygame.K_m:
                restart("manual")
            elif key == pygame.K_f:
                fast = not fast
            elif key == pygame.K_p:
                paused = not paused
            elif key == pygame.K_r:
                restart(mode if mode != "menu" else "training")
            elif key == pygame.K_s:
                agent.save(model_path)
                message = f"Saved {model_path.name} ({len(agent.q):,} states)"

        if not paused and mode in ("training", "watch", "manual") and wait_frames == 0:
            steps_this_frame = 1
            if mode == "training" and fast:
                steps_this_frame = 800
            for _ in range(steps_this_frame):
                if mode == "manual":
                    keys = pygame.key.get_pressed()
                    main = keys[pygame.K_SPACE]
                    left = keys[pygame.K_a]
                    right = keys[pygame.K_d]
                    top = keys[pygame.K_DOWN]
                    if top:
                        action = 6
                    elif main and left:
                        action = 4
                    elif main and right:
                        action = 5
                    elif main:
                        action = 1
                    elif left:
                        action = 2
                    elif right:
                        action = 3
                    else:
                        action = 0
                else:
                    action = agent.choose_action(state, explore=(mode == "training"))

                next_state, reward, done, result = env.step(action)
                if mode == "training":
                    agent.learn(state, action, reward, next_state, done)
                state = next_state

                if done:
                    rewards.append(result.reward)
                    successes.append(result.outcome == "LANDED")
                    if mode == "training":
                        agent.finish_episode()
                        if (
                            episode > target_episodes * 0.6 + 50
                            and len(successes) == successes.maxlen
                            and episode % 10 == 0
                        ):
                            agent.consider_checkpoint(sum(successes) / len(successes))
                        episode += 1
                        if episode > target_episodes:
                            agent.restore_best()
                            agent.save(model_path)
                            episode = target_episodes
                            restart("watch")
                            message = "Training complete - now watching the learned policy"
                            break
                        difficulty = min(1.0, episode / max(1.0, target_episodes * 0.6))
                        state = env.reset(difficulty)
                    else:
                        wait_frames = 80
                    break

        if wait_frames > 0 and not paused:
            wait_frames -= 1
            if wait_frames == 0:
                state = env.reset()
                result = None

        renderer.render(
            env, agent, mode, episode, target_episodes, rewards, successes,
            result, fast, paused, message, menu=(mode == "menu")
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5000, help="training trials")
    parser.add_argument("--seed", type=int, default=7, help="repeatable random seed")
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("lander_q_table.pkl"))
    parser.add_argument("--fresh", action="store_true", help="ignore an existing Q-table")
    parser.add_argument("--headless", action="store_true", help="train without opening a window")
    parser.add_argument("--evaluate", type=int, default=0, metavar="N", help="run N greedy test flights")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1 or args.evaluate < 0:
        raise ValueError("Episodes must be positive and evaluation count non-negative.")

    # A saved model makes a live presentation repeatable. Move it aside or use
    # --fresh when you want the audience to see learning from scratch.
    agent = QLearningAgent(seed=args.seed)
    env = RocketCarEnv(seed=args.seed)
    if not args.fresh and args.model.exists():
        try:
            agent.load(args.model)
            print(f"Loaded {args.model} with {len(agent.q):,} states.")
        except (OSError, EOFError, ValueError, pickle.PickleError) as error:
            print(f"Could not load model: {error}. Starting fresh.")

    if args.headless:
        train_headless(agent, env, args.episodes)
        agent.save(args.model)
        print(f"Saved Q-table to {args.model}")
        if args.evaluate:
            evaluate(agent, env, args.evaluate)
    else:
        pygame.init()
        try:
            run_gui(agent, env, args.episodes, args.model)
        finally:
            pygame.quit()


if __name__ == "__main__":
    main()
