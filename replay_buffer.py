"""
replay_buffer.py
────────────────
Experience Replay Buffer — the agent's memory.

Why do we need this?
────────────────────
If we trained the network on every transition the moment it
happened, consecutive frames are highly correlated (frame 42
and frame 43 look almost identical). The network would just
overfit to recent experience and forget everything before it.

Instead we:
  1. Store every transition (s, a, r, s', done) in a big buffer.
  2. At each training step, sample a RANDOM mini-batch from the buffer.
  3. Train the network on that batch.

This breaks temporal correlations and makes training much more stable.

Buffer behaviour
────────────────
  - Fixed capacity (default 10 000 transitions).
  - Once full, the OLDEST transitions are overwritten (circular buffer).
  - We only start sampling once the buffer has at least `batch_size` entries.
"""

import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """
    Circular experience replay buffer.

    Parameters
    ----------
    capacity   : int — maximum number of transitions to store
    """

    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    # ── Public API ────────────────────────────────────────────────────────────

    def push(self, state, action: int, reward: float,
             next_state, done: bool):
        """
        Store one transition.

        Parameters
        ----------
        state      : array-like  shape (5,)
        action     : int         0 or 1
        reward     : float
        next_state : array-like  shape (5,)
        done       : bool        True if the episode ended
        """
        self.buffer.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int = 64):
        """
        Return a random mini-batch as numpy arrays.

        Returns
        -------
        states      : np.ndarray  (batch_size, 5)
        actions     : np.ndarray  (batch_size,)   int
        rewards     : np.ndarray  (batch_size,)   float32
        next_states : np.ndarray  (batch_size, 5)
        dones       : np.ndarray  (batch_size,)   float32  (1.0 = done)
        """
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def is_ready(self, batch_size: int = 64) -> bool:
        """True once we have enough samples to train."""
        return len(self.buffer) >= batch_size


# ── Quick sanity test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    buf = ReplayBuffer(capacity=1000)
    print(f"Buffer size before push : {len(buf)}")

    # Fill with dummy transitions
    for i in range(200):
        s  = np.random.rand(5).astype(np.float32)
        a  = np.random.randint(0, 2)
        r  = np.random.uniform(-1, 1)
        s2 = np.random.rand(5).astype(np.float32)
        d  = bool(np.random.rand() > 0.95)
        buf.push(s, a, r, s2, d)

    print(f"Buffer size after 200 pushes : {len(buf)}")

    states, actions, rewards, next_states, dones = buf.sample(batch_size=32)
    print(f"\nSampled batch shapes:")
    print(f"  states      : {states.shape}")
    print(f"  actions     : {actions.shape}  values: {set(actions.tolist())}")
    print(f"  rewards     : {rewards.shape}  min={rewards.min():.2f} max={rewards.max():.2f}")
    print(f"  next_states : {next_states.shape}")
    print(f"  dones       : {dones.shape}    num_done={dones.sum():.0f}")
