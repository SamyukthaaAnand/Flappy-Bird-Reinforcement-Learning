"""
flappy_env.py
─────────────
Flappy Bird RL Environment — Phase 2
A clean, gym-style environment your DQN agent will talk to.

State  : [bird_y, bird_vel, pipe_x, pipe_top_y, pipe_bot_y]  (5 floats, all normalised 0-1)
Actions: 0 = do nothing  |  1 = flap
Reward :  +0.1  every frame alive
          +1.0  each pipe cleared
          -1.0  on death
"""

import numpy as np

# ── Game constants ────────────────────────────────────────────────────────────
SCREEN_W   = 400
SCREEN_H   = 600
GRAVITY    = 0.5          # pixels / frame²  (added to velocity each frame)
FLAP_VEL   = -8           # pixels / frame   (upward kick on flap)
BIRD_X     = 80           # fixed horizontal position
BIRD_R     = 14           # collision radius
PIPE_W     = 52
PIPE_GAP   = 150          # vertical opening in the pipe pair
PIPE_SPEED = 3            # pixels / frame
MAX_VEL    = 12           # terminal velocity (downward)


class FlappyEnv:
    """
    Minimal gym-style environment for Flappy Bird.

    Usage
    -----
    env   = FlappyEnv()
    state = env.reset()

    done = False
    while not done:
        action         = agent.choose(state)   # 0 or 1
        state, reward, done, info = env.step(action)
    """

    def __init__(self):
        self.score     = 0
        self.frame     = 0
        self._reset_state()

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Start a new episode. Returns the initial state vector."""
        self._reset_state()
        return self._get_state()

    def step(self, action: int):
        """
        Advance the game by one frame.

        Parameters
        ----------
        action : int  — 0 (nothing) or 1 (flap)

        Returns
        -------
        state  : np.ndarray  shape (5,)
        reward : float
        done   : bool
        info   : dict        extra diagnostics
        """
        assert action in (0, 1), "Action must be 0 or 1"

        self.frame += 1
        reward = 0.1  # alive bonus

        # ── Bird physics ──────────────────────────────────────────────────────
        if action == 1:
            self.bird_vel = FLAP_VEL

        self.bird_vel = min(self.bird_vel + GRAVITY, MAX_VEL)
        self.bird_y  += self.bird_vel

        # ── Move pipes ────────────────────────────────────────────────────────
        self.pipe_x -= PIPE_SPEED

        # Respawn pipe when it leaves the screen
        if self.pipe_x < -PIPE_W:
            self.pipe_x     = SCREEN_W
            self.pipe_top_y = self._random_pipe_top()
            self.passed     = False

        # ── Score when bird clears a pipe ─────────────────────────────────────
        if not self.passed and self.pipe_x + PIPE_W < BIRD_X:
            self.passed  = True
            self.score  += 1
            reward       = 1.0

        # ── Collision detection ───────────────────────────────────────────────
        done = self._is_dead()
        if done:
            reward = -1.0

        state = self._get_state()
        info  = {"score": self.score, "frame": self.frame}
        return state, reward, done, info

    def render_data(self) -> dict:
        """
        Returns raw game data so an external Pygame renderer can draw the frame.
        Nothing is drawn here — keeping rendering and logic separate is key.
        """
        return {
            "bird_y":      self.bird_y,
            "bird_x":      BIRD_X,
            "bird_r":      BIRD_R,
            "pipe_x":      self.pipe_x,
            "pipe_w":      PIPE_W,
            "pipe_top_y":  self.pipe_top_y,
            "pipe_bot_y":  self.pipe_top_y + PIPE_GAP,
            "score":       self.score,
            "screen_w":    SCREEN_W,
            "screen_h":    SCREEN_H,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _reset_state(self):
        self.bird_y     = SCREEN_H // 2
        self.bird_vel   = 0.0
        self.pipe_x     = SCREEN_W + 100
        self.pipe_top_y = self._random_pipe_top()
        self.passed     = False
        self.score      = 0
        self.frame      = 0

    def _random_pipe_top(self) -> float:
        """Returns a random y-coordinate for the top pipe's bottom edge."""
        min_top = 80
        max_top = SCREEN_H - PIPE_GAP - 80
        return float(np.random.randint(min_top, max_top))

    def _get_state(self) -> np.ndarray:
        """
        Returns a normalised state vector — values between 0 and 1.
        Normalisation helps the neural network learn faster.

        Indices
        -------
        0  bird_y          normalised by SCREEN_H
        1  bird_vel        normalised to [-1, 1]  (range: FLAP_VEL … MAX_VEL)
        2  pipe_x          normalised by SCREEN_W
        3  pipe_top_y      normalised by SCREEN_H
        4  pipe_bot_y      normalised by SCREEN_H
        """
        vel_range = MAX_VEL - FLAP_VEL  # 12 - (-8) = 20
        return np.array([
            self.bird_y                         / SCREEN_H,
            (self.bird_vel - FLAP_VEL)          / vel_range,
            max(self.pipe_x, 0)                 / SCREEN_W,
            self.pipe_top_y                     / SCREEN_H,
            (self.pipe_top_y + PIPE_GAP)        / SCREEN_H,
        ], dtype=np.float32)

    def _is_dead(self) -> bool:
        """True if the bird hit the ground, ceiling, or a pipe."""
        # Floor / ceiling
        if self.bird_y - BIRD_R < 0 or self.bird_y + BIRD_R > SCREEN_H:
            return True

        # Pipe collision (axis-aligned bounding box vs circle approximation)
        pipe_bot_y = self.pipe_top_y + PIPE_GAP
        bird_left  = BIRD_X - BIRD_R
        bird_right = BIRD_X + BIRD_R
        bird_top   = self.bird_y - BIRD_R
        bird_bot   = self.bird_y + BIRD_R

        in_pipe_x = bird_right > self.pipe_x and bird_left < self.pipe_x + PIPE_W
        in_pipe_y = bird_top < self.pipe_top_y or bird_bot > pipe_bot_y

        return in_pipe_x and in_pipe_y


# ── Quick sanity test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    env   = FlappyEnv()
    state = env.reset()
    print(f"Initial state : {state}")
    print(f"State shape   : {state.shape}  (should be (5,))")

    for _ in range(5):
        action = np.random.randint(0, 2)
        state, reward, done, info = env.step(action)
        print(f"action={action}  reward={reward:+.1f}  done={done}  score={info['score']}")
        if done:
            print("Episode ended — resetting.")
            state = env.reset()
