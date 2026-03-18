"""
plot.py
───────
Phase 5 — Plot training curves from the CSV log.

Reads  : logs/training_log.csv  (written by train.py)
Outputs: logs/training_curves.png

Usage
─────
  python plot.py
  python plot.py --log logs/training_log.csv --window 30
"""

import os
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def parse_args():
    p = argparse.ArgumentParser(description="Plot Flappy Bird DQN training curves")
    p.add_argument("--log",    type=str, default="logs/training_log.csv")
    p.add_argument("--window", type=int, default=20,
                   help="Smoothing window size")
    p.add_argument("--out",    type=str, default="logs/training_curves.png")
    return p.parse_args()


def smooth(data, window):
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid").tolist()


def load_csv(path):
    episodes, scores, rewards, epsilons, losses = [], [], [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            scores.append(float(row["score"]))
            rewards.append(float(row["total_reward"]))
            epsilons.append(float(row["epsilon"]))
            losses.append(float(row["avg_loss"]))
    return episodes, scores, rewards, epsilons, losses


def plot(args):
    if not os.path.exists(args.log):
        print(f"Log file not found: {args.log}")
        print("Train first with:  python train.py")
        return

    episodes, scores, rewards, epsilons, losses = load_csv(args.log)
    w = args.window

    # ── Compute rolling averages ──────────────────────────────────────────────
    sm_scores  = smooth(scores,  w)
    sm_rewards = smooth(rewards, w)
    ep_offset  = w - 1   # x-axis shift after convolution

    # ── Style ─────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor":   "#16213e",
        "axes.edgecolor":   "#444466",
        "axes.labelcolor":  "#ccccee",
        "xtick.color":      "#888899",
        "ytick.color":      "#888899",
        "text.color":       "#ccccee",
        "grid.color":       "#2a2a4a",
        "grid.linestyle":   "--",
        "grid.alpha":       0.5,
    })

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("Flappy Bird DQN — Training Results", fontsize=16,
                 color="#eeeeff", fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    # ── Score ─────────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(episodes, scores,  color="#1D9E75", alpha=0.25, linewidth=0.8, label="Raw")
    ax1.plot(episodes[ep_offset:], sm_scores, color="#5DCAA5", linewidth=2, label=f"Avg ({w})")
    ax1.set_title("Score per episode", fontsize=11)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Pipes cleared")
    ax1.legend(fontsize=9)
    ax1.grid(True)

    # ── Total reward ──────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(episodes, rewards,  color="#378ADD", alpha=0.25, linewidth=0.8)
    ax2.plot(episodes[ep_offset:], sm_rewards, color="#85B7EB", linewidth=2)
    ax2.set_title("Total reward per episode", fontsize=11)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Reward")
    ax2.grid(True)

    # ── Epsilon ───────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(episodes, epsilons, color="#D85A30", linewidth=2)
    ax3.fill_between(episodes, epsilons, alpha=0.15, color="#D85A30")
    ax3.set_title("Epsilon decay (exploration → exploitation)", fontsize=11)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Epsilon")
    ax3.set_ylim(0, 1.05)
    ax3.grid(True)

    # ── Loss ─────────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    nonzero_losses = [(i+1, l) for i, l in enumerate(losses) if l > 0]
    if nonzero_losses:
        lx, ly = zip(*nonzero_losses)
        sm_loss = smooth(list(ly), min(w, len(ly)//2 or 1))
        ax4.plot(list(lx), list(ly), color="#BA7517", alpha=0.2, linewidth=0.7)
        ax4.plot(list(lx)[len(lx)-len(sm_loss):], sm_loss,
                 color="#EF9F27", linewidth=2)
    ax4.set_title("Training loss (MSE)", fontsize=11)
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Loss")
    ax4.grid(True)

    # ── Stats box ─────────────────────────────────────────────────────────────
    best_ep  = episodes[np.argmax(scores)]
    stats_txt = (
        f"Episodes: {len(episodes)}\n"
        f"Best score: {max(scores):.0f}  (ep {best_ep})\n"
        f"Avg score (last 100): {np.mean(scores[-100:]):.1f}\n"
        f"Final epsilon: {epsilons[-1]:.4f}"
    )
    fig.text(0.5, 0.01, stats_txt, ha="center", va="bottom",
             fontsize=10, color="#aaaacc",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f0f1f",
                       edgecolor="#333355", alpha=0.8))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"Plot saved → {args.out}")
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    plot(args)
