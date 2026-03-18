"""
evaluate.py
───────────
Phase 5 — Load a trained model and watch the agent play.

Usage
─────
  python evaluate.py                          # loads best_model.pth by default
  python evaluate.py --model models/best_model.pth --episodes 5
  python evaluate.py --model models/final_model.pth --no-render
"""

import os
import argparse
import numpy as np

from flappy_env import FlappyEnv
from dqn_agent  import DQNAgent


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained Flappy Bird DQN agent")
    p.add_argument("--model",    type=str, default="models/best_model.pth",
                   help="Path to the .pth checkpoint")
    p.add_argument("--episodes", type=int, default=5,
                   help="Number of episodes to run")
    p.add_argument("--no-render", dest="render", action="store_false",
                   help="Disable Pygame window")
    p.set_defaults(render=True)
    return p.parse_args()


def evaluate(args):
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        print("Train first with:  python train.py")
        return

    # Load agent — force epsilon=0 so it always picks the best action
    agent = DQNAgent()
    agent.load(args.model)
    agent.epsilon = 0.0     # no random exploration during evaluation
    print(f"Epsilon forced to 0.0 — agent will act greedily.")

    env    = FlappyEnv()
    scores = []

    # ── Optional Pygame setup ─────────────────────────────────────────────────
    if args.render:
        import pygame
        from game import draw_bird, draw_pipe, draw_ground, draw_clouds, render_text_shadow
        pygame.init()
        screen     = pygame.display.set_mode((400, 600))
        pygame.display.set_caption("Flappy Bird — Agent Evaluation")
        clock      = pygame.time.Clock()
        font_big   = pygame.font.SysFont("Arial", 36, bold=True)
        font_small = pygame.font.SysFont("Arial", 18)
        clouds     = [(80,80,35),(230,55,28),(340,100,22),(150,130,18)]
        SKY        = (113, 197, 207)

    print(f"\nRunning {args.episodes} evaluation episodes...\n")

    for ep in range(1, args.episodes + 1):
        state        = env.reset()
        done         = False
        total_reward = 0.0

        while not done:
            if args.render:
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return

            action = agent.choose(state)
            state, reward, done, info = env.step(action)
            total_reward += reward

            if args.render:
                import pygame
                data = env.render_data()
                screen.fill(SKY)
                draw_clouds(screen, clouds)
                draw_pipe(screen, data["pipe_x"], data["pipe_w"],
                          data["pipe_top_y"], data["pipe_bot_y"], 600)
                draw_ground(screen, 400, 600)
                draw_bird(screen, data["bird_x"], data["bird_y"], data["bird_r"])
                render_text_shadow(screen, font_big,  str(data["score"]), 188, 30)
                render_text_shadow(screen, font_small,
                                   f"Eval ep {ep}/{args.episodes}", 10, 10)
                pygame.display.flip()
                clock.tick(60)

        scores.append(info["score"])
        print(f"Episode {ep:>3} | score: {info['score']:>4} | "
              f"reward: {total_reward:>+7.1f} | frames: {info['frame']}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print(f"Episodes  : {args.episodes}")
    print(f"Avg score : {np.mean(scores):.1f}")
    print(f"Max score : {max(scores)}")
    print(f"Min score : {min(scores)}")
    print(f"{'─'*40}")

    if args.render:
        import pygame
        pygame.quit()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
