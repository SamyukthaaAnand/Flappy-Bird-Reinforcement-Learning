"""
dqn_agent.py
────────────
The DQN Agent — ties the Q-Network and Replay Buffer together.

Key ideas implemented here
──────────────────────────

1. Epsilon-greedy exploration
   The agent starts completely random (epsilon=1.0) and gradually
   becomes more confident as it learns (epsilon decays to 0.01).
   At each step:
     - With probability epsilon  → pick a RANDOM action (explore)
     - With probability 1-epsilon → pick the BEST action (exploit)

2. Target network
   We maintain TWO copies of the Q-network:
     - online_net  : updated every training step
     - target_net  : updated every N steps by copying online_net
   The target network gives stable Q-value targets so the online
   network isn't chasing a moving target — a major source of
   instability in early DQN implementations.

3. Bellman update (the core of Q-learning)
   For each transition (s, a, r, s', done):
     target = r + gamma * max_a' Q_target(s', a')   if not done
     target = r                                       if done
   Loss = MSE(Q_online(s, a),  target)
   We minimise this loss with gradient descent.

4. Learning starts only after the buffer has enough transitions.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model         import QNetwork
from replay_buffer import ReplayBuffer


class DQNAgent:
    """
    DQN agent for Flappy Bird.

    Parameters
    ----------
    state_dim    : int   — size of state vector (5)
    action_dim   : int   — number of actions    (2)
    lr           : float — learning rate        (default 1e-3)
    gamma        : float — discount factor      (default 0.99)
    epsilon_start: float — starting exploration (default 1.0)
    epsilon_end  : float — minimum exploration  (default 0.01)
    epsilon_decay: float — multiplicative decay per step (default 0.995)
    buffer_size  : int   — replay buffer capacity
    batch_size   : int   — training mini-batch size
    target_update: int   — copy online→target every N steps
    """

    def __init__(
        self,
        state_dim:     int   = 5,
        action_dim:    int   = 2,
        lr:            float = 1e-3,
        gamma:         float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end:   float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size:   int   = 10_000,
        batch_size:    int   = 64,
        target_update: int   = 100,
    ):
        self.action_dim    = action_dim
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.steps_done    = 0

        # Device — use GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQNAgent using device: {self.device}")

        # Networks
        self.online_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()   # target net is never trained directly

        # Optimizer and loss
        self.optimizer  = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn    = nn.MSELoss()

        # Memory
        self.buffer = ReplayBuffer(capacity=buffer_size)

    # ── Action selection ──────────────────────────────────────────────────────

    def choose(self, state) -> int:
        """
        Epsilon-greedy action selection.

        Parameters
        ----------
        state : array-like  shape (5,)

        Returns
        -------
        int  — 0 (do nothing) or 1 (flap)
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)   # explore

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(q_values.argmax().item())                 # exploit

    # ── Store experience ──────────────────────────────────────────────────────

    def remember(self, state, action, reward, next_state, done):
        """Push one transition into the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    # ── Training step ─────────────────────────────────────────────────────────

    def learn(self) -> float | None:
        """
        Sample a mini-batch and perform one gradient step.

        Returns
        -------
        float | None  — the loss value, or None if buffer not ready yet
        """
        if len(self.buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Convert to tensors
        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # Current Q-values: Q_online(s, a)
        # gather picks the Q-value for the action that was actually taken
        current_q = self.online_net(states_t).gather(
            1, actions_t.unsqueeze(1)
        ).squeeze(1)

        # Target Q-values: r + gamma * max Q_target(s', a')
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(dim=1)[0]
            target_q   = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        # Compute loss and backpropagate
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update counters and decay epsilon
        self.steps_done += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Sync target network periodically
        if self.steps_done % self.target_update == 0:
            self._update_target()

        return float(loss.item())

    # ── Save / load ───────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save the online network weights."""
        torch.save({
            "online_net":  self.online_net.state_dict(),
            "target_net":  self.target_net.state_dict(),
            "epsilon":     self.epsilon,
            "steps_done":  self.steps_done,
        }, path)
        print(f"Model saved → {path}")

    def load(self, path: str):
        """Load previously saved weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.epsilon    = checkpoint["epsilon"]
        self.steps_done = checkpoint["steps_done"]
        print(f"Model loaded ← {path}  (epsilon={self.epsilon:.3f})")

    # ── Private ───────────────────────────────────────────────────────────────

    def _update_target(self):
        """Copy online network weights to the target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())


# ── Quick sanity test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    from flappy_env import FlappyEnv

    agent = DQNAgent()
    env   = FlappyEnv()
    state = env.reset()

    print(f"\nRunning 500 random steps to fill the replay buffer...")
    for i in range(500):
        action                    = agent.choose(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()

    print(f"Buffer size: {len(agent.buffer)}")

    loss = agent.learn()
    print(f"First training loss: {loss:.6f}")
    print(f"Epsilon after 500 steps: {agent.epsilon:.4f}")
    print("\nAll good! Ready for the training loop (train.py).")
