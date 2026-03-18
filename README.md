# 🐦 Flappy Bird — Reinforcement Learning

An AI agent trained to play **Flappy Bird** using **Deep Q-Network (DQN)**. The agent learns purely from gameplay experience — no hard-coded rules, just a neural network approximating optimal actions through trial and reward.

---

## 📌 Overview

This project implements a **DQN agent** that learns to play Flappy Bird by interacting with a custom OpenAI Gym-style environment. A neural network replaces the traditional Q-table, allowing the agent to generalize across continuous state spaces. Core techniques like **experience replay** and a **target network** stabilize training.

---

## 🧠 Architecture

### Agent (`dqn_agent.py`)
The DQN agent decides whether to flap or do nothing at each timestep. It follows an **ε-greedy policy** — exploring randomly early in training and gradually exploiting learned Q-values as epsilon decays.

### Neural Network (`model.py`)
A feedforward neural network that takes the game state as input and outputs Q-values for each action:
- **Input**: Game state vector (bird position, velocity, pipe distances, etc.)
- **Hidden Layers**: Fully connected layers with ReLU activations
- **Output**: Q-values for `[do nothing, flap]`

### Replay Buffer (`replay_buffer.py`)
Stores past `(state, action, reward, next_state, done)` transitions. Random mini-batches are sampled during training to break temporal correlations and stabilize learning.

### Environment (`flappy_env.py`)
A custom Gym-compatible wrapper around the Flappy Bird game that exposes:
- `reset()` — starts a new episode
- `step(action)` — applies an action and returns `(state, reward, done, info)`

### Game (`game.py`)
The core Flappy Bird game logic built with **Pygame** — handles rendering, physics, pipe generation, and collision detection.

---

## 🗂️ Project Structure

```
Flappy-Bird-Reinforcement-Learning/
│
├── train.py             # Train the DQN agent
├── evaluate.py          # Run and evaluate a trained agent
├── dqn_agent.py         # DQN agent (epsilon-greedy policy, training loop)
├── model.py             # Neural network (Q-network)
├── replay_buffer.py     # Experience replay memory
├── flappy_env.py        # Custom Gym-style Flappy Bird environment
├── game.py              # Flappy Bird game engine (Pygame)
├── plot.py              # Plot training rewards/scores over episodes
├── models/              # Saved model checkpoints
├── logs/                # Training logs
├── requirements.txt     # Python dependencies
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/SamyukthaaAnand/Flappy-Bird-Reinforcement-Learning.git
cd Flappy-Bird-Reinforcement-Learning

# 2. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train the Agent

```bash
python train.py
```

Trains the DQN agent from scratch. Model checkpoints are saved to the `models/` directory and training logs to `logs/`.

### Evaluate a Trained Agent

```bash
python evaluate.py
```

Loads a saved model and runs the agent visually so you can watch it play.

### Plot Training Progress

```bash
python plot.py
```

Generates reward/score curves from the training logs.

---

## 📈 Training Progress

The agent's performance improves as the neural network converges over episodes:

| Episodes  | Avg. Score |
|-----------|------------|
| ~100      | < 1        |
| ~500      | ~5–15      |
| ~2000     | ~50–100    |
| ~5000+    | 100+       |

> Results may vary depending on hyperparameter tuning and random seeds.

---

## ⚙️ Hyperparameters

| Parameter       | Description                        |
|-----------------|------------------------------------|
| `learning_rate` | Adam optimizer learning rate       |
| `gamma`         | Discount factor for future rewards |
| `epsilon_start` | Initial exploration rate           |
| `epsilon_end`   | Minimum exploration rate           |
| `epsilon_decay` | Rate at which epsilon decreases    |
| `batch_size`    | Mini-batch size for replay buffer  |
| `buffer_size`   | Max capacity of replay memory      |
| `target_update` | Frequency of target network update |

---

## 🛠️ Built With

- [Python](https://www.python.org/)
- [PyTorch](https://pytorch.org/) — Neural network & training
- [Pygame](https://www.pygame.org/) — Game engine
- [NumPy](https://numpy.org/) — Numerical operations
- [Matplotlib](https://matplotlib.org/) — Training plots

---

## 📚 References

- [Human-level control through deep reinforcement learning — Mnih et al., 2015](https://www.nature.com/articles/nature14236)
- [Deep Q-Network (DQN) — DeepMind](https://deepmind.google/discover/blog/deep-reinforcement-learning/)
- [OpenAI Gym](https://gymnasium.farama.org/)

---

## 🙋‍♀️ Author

**Samyukthaa Anand**
- GitHub: [@SamyukthaaAnand](https://github.com/SamyukthaaAnand)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
