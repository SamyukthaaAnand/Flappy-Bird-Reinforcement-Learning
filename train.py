"""
train.py
────────
Phase 4 — The main training loop.

This is where everything comes together:
  FlappyEnv  →  DQNAgent  →  learn()  →  save checkpoint

What happens each episode
─────────────────────────
  1. Reset the environment → get initial state
  2. Loop until done:
       a. Agent picks an action  (epsilon-greedy)
       b. Environment steps      → next_state, reward, done
       c. Store transition in replay buffer
       d. Agent learns from a random batch
  3. Log score, reward, loss, epsilon
  4. Save a checkpoint every SAVE_EVERY episodes

Run
───
  python train.py
  python train.py --episodes 2000 --render
"""

import os
import csv
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt

from flappy_env import FlappyEnv
from dqn_agent  import DQNAgent

# ── Hyperparameters ───────────────────────────────────────────────────────────
DEFAULT_EPISODES  = 1000
SAVE_EVERY        = 100        # save a checkpoint every N episodes
PRINT_EVERY       = 10         # print progress every N episodes
PLOT_EVERY        = 50         # re-draw training curves every N episodes
LOG_DIR           = "logs"
MODEL_DIR         = "models"

# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train DQN on Flappy Bird")
    p.add_argument("--episodes",  type=int,   default=DEFAULT_EPISODES)
    p.add_argument("--render",    action="store_true",
                   help="Show Pygame window while training (slower)")
    p.add_argument("--resume",    type=str,   default=None,
                   help="Path to a .pth checkpoint to resume from")
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--gamma",     type=float, default=0.99)
    p.add_argument("--batch",     type=int,   default=64)
    return p.parse_args()


# ── Plotting helper ───────────────────────────────────────────────────────────
def save_plots(scores, rewards, epsilons, losses, out_dir):
    """Save training curves to PNG files."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Flappy Bird DQN — Training Curves", fontsize=14)

    def smooth(data, window=20):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode="valid").tolist()

    axes[0, 0].plot(smooth(scores),  color="#1D9E75")
    axes[0, 0].set_title("Score per episode")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Pipes cleared")

    axes[0, 1].plot(smooth(rewards), color="#378ADD")
    axes[0, 1].set_title("Total reward per episode")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Reward")

    axes[1, 0].plot(epsilons, color="#D85A30")
    axes[1, 0].set_title("Epsilon (exploration rate)")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Epsilon")

    valid_losses = [l for l in losses if l is not None]
    axes[1, 1].plot(smooth(valid_losses), color="#BA7517")
    axes[1, 1].set_title("Training loss")
    axes[1, 1].set_xlabel("Training step")
    axes[1, 1].set_ylabel("MSE Loss")

    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=120)
    plt.close()
    return path


# ── Main training loop ────────────────────────────────────────────────────────
def train(args):
    os.makedirs(LOG_DIR,   exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Init ──────────────────────────────────────────────────────────────────
    env   = FlappyEnv()
    agent = DQNAgent(
        lr           = args.lr,
        gamma        = args.gamma,
        batch_size   = args.batch,
    )

    if args.resume:
        agent.load(args.resume)

    # Pygame renderer (optional)
    renderer = None
    if args.render:
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((400, 600))
        pygame.display.set_caption("Flappy Bird — Training")
        clock  = pygame.font.SysFont("Arial", 18)
        # import drawing helpers from game.py
        from game import draw_bird, draw_pipe, draw_ground, draw_clouds, render_text_shadow
        font_big   = pygame.font.SysFont("Arial", 36, bold=True)
        font_small = pygame.font.SysFont("Arial", 18)
        clouds = [(80,80,35),(230,55,28),(340,100,22),(150,130,18)]
        SKY = (113, 197, 207)

    # ── Logging ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(LOG_DIR, "training_log.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "score", "total_reward", "epsilon", "avg_loss"])

    all_scores   = []
    all_rewards  = []
    all_epsilons = []
    all_losses   = []

    best_score = 0

    print("=" * 60)
    print("  Flappy Bird DQN — Training")
    print(f"  Episodes : {args.episodes}")
    print(f"  Render   : {args.render}")
    print(f"  Device   : {agent.device}")
    print("=" * 60)

    # ── Episode loop ──────────────────────────────────────────────────────────
    for episode in range(1, args.episodes + 1):
        state        = env.reset()
        done         = False
        total_reward = 0.0
        ep_losses    = []

        while not done:
            # ── Pygame event pump (keeps window responsive) ───────────────────
            if args.render:
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        csv_file.close()
                        return
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        csv_file.close()
                        return

            # ── Agent acts ────────────────────────────────────────────────────
            action = agent.choose(state)
            next_state, reward, done, info = env.step(action)

            # ── Store and learn ───────────────────────────────────────────────
            agent.remember(state, action, reward, next_state, done)
            loss = agent.learn()

            if loss is not None:
                ep_losses.append(loss)
                all_losses.append(loss)

            total_reward += reward
            state         = next_state

            # ── Render frame ──────────────────────────────────────────────────
            if args.render:
                import pygame
                data = env.render_data()
                screen.fill(SKY)
                draw_clouds(screen, clouds)
                draw_pipe(screen, data["pipe_x"], data["pipe_w"],
                          data["pipe_top_y"], data["pipe_bot_y"], 600)
                draw_ground(screen, 400, 600)
                draw_bird(screen, data["bird_x"], data["bird_y"], data["bird_r"])
                render_text_shadow(screen, font_big, str(data["score"]),
                                   188, 30)
                render_text_shadow(screen, font_small,
                                   f"ep:{episode}  ε:{agent.epsilon:.3f}",
                                   10, 10)
                pygame.display.flip()

        # ── End of episode bookkeeping ────────────────────────────────────────
        score     = info["score"]
        avg_loss  = float(np.mean(ep_losses)) if ep_losses else 0.0

        all_scores.append(score)
        all_rewards.append(total_reward)
        all_epsilons.append(agent.epsilon)

        csv_writer.writerow([episode, score, round(total_reward, 2),
                             round(agent.epsilon, 4), round(avg_loss, 6)])
        csv_file.flush()

        # ── Console log ───────────────────────────────────────────────────────
        if episode % PRINT_EVERY == 0:
            recent_avg = np.mean(all_scores[-PRINT_EVERY:])
            print(f"Ep {episode:>5} | "
                  f"score {score:>4} | "
                  f"avg(last {PRINT_EVERY}) {recent_avg:>5.1f} | "
                  f"reward {total_reward:>+7.1f} | "
                  f"ε {agent.epsilon:.4f} | "
                  f"loss {avg_loss:.5f}")

        # ── Save best model ───────────────────────────────────────────────────
        if score > best_score:
            best_score = score
            agent.save(os.path.join(MODEL_DIR, "best_model.pth"))

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if episode % SAVE_EVERY == 0:
            path = os.path.join(MODEL_DIR, f"checkpoint_ep{episode}.pth")
            agent.save(path)

        # ── Periodic plots ────────────────────────────────────────────────────
        if episode % PLOT_EVERY == 0:
            plot_path = save_plots(all_scores, all_rewards,
                                   all_epsilons, all_losses, LOG_DIR)
            print(f"  → Plot saved: {plot_path}")

    # ── Final save ────────────────────────────────────────────────────────────
    agent.save(os.path.join(MODEL_DIR, "final_model.pth"))
    save_plots(all_scores, all_rewards, all_epsilons, all_losses, LOG_DIR)
    csv_file.close()

    print("\n" + "=" * 60)
    print(f"  Training complete!")
    print(f"  Best score   : {best_score}")
    print(f"  Final epsilon: {agent.epsilon:.4f}")
    print(f"  Log saved    : {csv_path}")
    print(f"  Models saved : {MODEL_DIR}/")
    print("=" * 60)

    if args.render:
        import pygame
        pygame.quit()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    train(args)
