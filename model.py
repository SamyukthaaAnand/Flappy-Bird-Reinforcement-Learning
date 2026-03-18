"""
model.py
────────
The Q-Network — a small fully-connected neural network.

Input  : state vector  (5 numbers)
Output : Q-value for each action  (2 numbers)
         index 0 = Q(s, do nothing)
         index 1 = Q(s, flap)

The agent picks the action with the HIGHER Q-value
(unless it's exploring randomly via epsilon-greedy).

Architecture
------------
  Linear(5  → 64)  + ReLU
  Linear(64 → 64)  + ReLU
  Linear(64 → 2)              ← no activation on output layer

Why no activation at the end?
  Q-values can be any real number (positive or negative),
  so we don't want to squash them with sigmoid/tanh.
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Feedforward Q-Network for Flappy Bird.

    Parameters
    ----------
    state_dim  : int  — number of inputs  (5 for our game)
    action_dim : int  — number of outputs (2: nothing or flap)
    hidden_dim : int  — neurons per hidden layer (default 64)
    """

    def __init__(self, state_dim: int = 5, action_dim: int = 2, hidden_dim: int = 64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim,  hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Initialise weights with Xavier uniform — helps training stability
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (batch_size, state_dim)

        Returns
        -------
        torch.Tensor  shape (batch_size, action_dim)
            Q-value for every action in each state
        """
        return self.net(x)


# ── Quick sanity test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = QNetwork()
    print(model)

    # Single state (shape: 1 × 5)
    dummy_state = torch.rand(1, 5)
    q_values    = model(dummy_state)
    print(f"\nDummy state  : {dummy_state.numpy().round(3)}")
    print(f"Q-values     : {q_values.detach().numpy().round(4)}")
    print(f"Best action  : {q_values.argmax().item()}  (0=nothing, 1=flap)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params}")
