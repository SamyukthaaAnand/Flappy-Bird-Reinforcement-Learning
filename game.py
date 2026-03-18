"""
game.py
───────
Flappy Bird — Pygame Renderer   (Phase 2)

Run this file directly to play manually with SPACE / UP arrow.
Later, the DQN agent will call env.step() and this file will
just visualise what the agent decides.

Controls (human mode)
---------------------
SPACE or UP  →  flap
ESC          →  quit
"""

import sys
import pygame
from flappy_env import FlappyEnv, SCREEN_W, SCREEN_H

# ── Colour palette ────────────────────────────────────────────────────────────
SKY        = (113, 197, 207)
GROUND_COL = ( 93, 161,  72)
PIPE_COL   = ( 83, 171,  68)
PIPE_EDGE  = ( 55, 130,  50)
BIRD_COL   = (255, 215,   0)
BIRD_EYE   = (255, 255, 255)
BIRD_PUPIL = ( 30,  30,  30)
TEXT_COL   = (255, 255, 255)
SHADOW_COL = ( 30,  30,  30)
CLOUD_COL  = (255, 255, 255)

FPS = 60


def draw_pipe(surface, pipe_x, pipe_w, pipe_top_y, pipe_bot_y, screen_h):
    """Draw a pipe pair (top + bottom)."""
    cap_h = 20
    cap_extra = 6  # cap is slightly wider than the pipe body

    # ── Top pipe ──────────────────────────────────────────────────────────────
    pygame.draw.rect(surface, PIPE_COL,
                     (pipe_x, 0, pipe_w, pipe_top_y - cap_h))
    pygame.draw.rect(surface, PIPE_EDGE,
                     (pipe_x, 0, pipe_w, pipe_top_y - cap_h), 2)
    # Cap
    pygame.draw.rect(surface, PIPE_COL,
                     (pipe_x - cap_extra, pipe_top_y - cap_h,
                      pipe_w + cap_extra * 2, cap_h))
    pygame.draw.rect(surface, PIPE_EDGE,
                     (pipe_x - cap_extra, pipe_top_y - cap_h,
                      pipe_w + cap_extra * 2, cap_h), 2)

    # ── Bottom pipe ───────────────────────────────────────────────────────────
    bot_body_top = pipe_bot_y + cap_h
    pygame.draw.rect(surface, PIPE_COL,
                     (pipe_x, bot_body_top, pipe_w, screen_h - bot_body_top))
    pygame.draw.rect(surface, PIPE_EDGE,
                     (pipe_x, bot_body_top, pipe_w, screen_h - bot_body_top), 2)
    # Cap
    pygame.draw.rect(surface, PIPE_COL,
                     (pipe_x - cap_extra, pipe_bot_y,
                      pipe_w + cap_extra * 2, cap_h))
    pygame.draw.rect(surface, PIPE_EDGE,
                     (pipe_x - cap_extra, pipe_bot_y,
                      pipe_w + cap_extra * 2, cap_h), 2)


def draw_bird(surface, bx, by, radius):
    """Draw a simple cartoon bird."""
    r = radius

    # Body
    pygame.draw.circle(surface, BIRD_COL, (int(bx), int(by)), r)
    pygame.draw.circle(surface, (200, 160, 0), (int(bx), int(by)), r, 2)

    # Eye white
    pygame.draw.circle(surface, BIRD_EYE,
                       (int(bx + r * 0.4), int(by - r * 0.3)), r // 3)
    # Pupil
    pygame.draw.circle(surface, BIRD_PUPIL,
                       (int(bx + r * 0.5), int(by - r * 0.25)), r // 6)

    # Beak (small triangle)
    beak = [
        (int(bx + r * 0.85), int(by)),
        (int(bx + r * 1.3),  int(by - r * 0.2)),
        (int(bx + r * 1.3),  int(by + r * 0.2)),
    ]
    pygame.draw.polygon(surface, (255, 140, 0), beak)


def draw_ground(surface, screen_w, screen_h):
    ground_h = 40
    pygame.draw.rect(surface, GROUND_COL,
                     (0, screen_h - ground_h, screen_w, ground_h))
    pygame.draw.rect(surface, (60, 120, 45),
                     (0, screen_h - ground_h, screen_w, 4))


def draw_clouds(surface, clouds):
    for (cx, cy, cr) in clouds:
        pygame.draw.ellipse(surface, CLOUD_COL,
                            (int(cx - cr), int(cy - cr * 0.6),
                             int(cr * 2), int(cr * 1.2)))
        pygame.draw.ellipse(surface, CLOUD_COL,
                            (int(cx - cr * 0.6), int(cy - cr),
                             int(cr * 1.2), int(cr * 1.4)))


def render_text_shadow(surface, font, text, x, y, col=TEXT_COL):
    shadow = font.render(text, True, SHADOW_COL)
    surface.blit(shadow, (x + 2, y + 2))
    label = font.render(text, True, col)
    surface.blit(label, (x, y))


def run_game(agent=None, episodes=1, render=True):
    """
    Run the Flappy Bird game.

    Parameters
    ----------
    agent    : object with a .choose(state) method, or None for human play
    episodes : number of episodes to run
    render   : whether to show the Pygame window
    """
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Flappy Bird — RL Project")
    clock  = pygame.time.Clock()
    font_big   = pygame.font.SysFont("Arial", 36, bold=True)
    font_small = pygame.font.SysFont("Arial", 18)

    # Static cloud positions (purely decorative)
    clouds = [
        (80,  80,  35),
        (230, 55,  28),
        (340, 100, 22),
        (150, 130, 18),
    ]

    env = FlappyEnv()

    for ep in range(episodes):
        state = env.reset()
        done  = False
        total_reward = 0.0

        while not done:
            # ── Events ────────────────────────────────────────────────────────
            action = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    if agent is None:  # human control
                        if event.key in (pygame.K_SPACE, pygame.K_UP):
                            action = 1

            # ── Agent control ─────────────────────────────────────────────────
            if agent is not None:
                action = agent.choose(state)

            # ── Step ──────────────────────────────────────────────────────────
            state, reward, done, info = env.step(action)
            total_reward += reward

            # ── Draw ──────────────────────────────────────────────────────────
            if render:
                data = env.render_data()

                screen.fill(SKY)
                draw_clouds(screen, clouds)
                draw_pipe(screen,
                          data["pipe_x"], data["pipe_w"],
                          data["pipe_top_y"], data["pipe_bot_y"],
                          SCREEN_H)
                draw_ground(screen, SCREEN_W, SCREEN_H)
                draw_bird(screen, data["bird_x"], data["bird_y"], data["bird_r"])

                # HUD
                render_text_shadow(screen, font_big,
                                   str(data["score"]),
                                   SCREEN_W // 2 - 12, 30)
                mode = "Human" if agent is None else "Agent"
                render_text_shadow(screen, font_small,
                                   f"{mode}  |  reward: {total_reward:+.1f}",
                                   10, 10)

                pygame.display.flip()
                clock.tick(FPS)

        print(f"Episode {ep+1:>3}  score={info['score']:>4}  "
              f"total_reward={total_reward:+.1f}  frames={info['frame']}")

    pygame.quit()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Controls: SPACE or UP to flap  |  ESC to quit")
    run_game(agent=None, episodes=999, render=True)
