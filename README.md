# Blokus AI Agents

Reinforcement-learning and game-tree-search agents for **Blokus**, built for my
University of Cambridge Computer Science Part II dissertation (**First-Class**),
supervised by **Dr Petar Veličković**.

The project builds a fast, Gymnasium-compatible Blokus environment and a range of
agents — from classical search to deep reinforcement learning — and studies how
they scale from tiny boards (solvable by search) to large ones (where only
learning is feasible).

## Headline results

- ⚡ **~10× faster** move generation than the standard open-source approach on 4×4–10×10 boards (**~6×** on 20×20) — the speedup that made large-scale RL tractable.
- ♟️ **Provably solved small Blokus:** the Alpha-Beta agent found a **guaranteed first-player winning strategy** on boards from 5×5 up to **7×7** (game tree ≈ 8×10¹⁴).
- 🧠 **Tabular Q-Learning (7×7)** beat a minimax-optimised opponent **84%** of 1,000 games (78% vs random) — exceeding the project's 75% target.
- 🤖 **Deep Q-Network (10×10)** reached **≥80%** vs a fixed heuristic opponent — on a board with a game tree **> 10³²**, roughly **10¹⁷× larger** than 7×7 and far beyond exhaustive search.
- 🔬 Two novel contributions: a **corner-cell action-precomputation** scheme and a **dual-Q-value training method** for learning values under optimal opponent play.

📄 **Full dissertation** (source + PDF): [`MarioPariona117/LaTeX`](https://github.com/MarioPariona117/LaTeX) → `Blokus_/2327D.pdf`

## Why it's interesting

Blokus has an enormous branching factor (up to ~300 legal moves per state). Using
**Knuth's method** to estimate tree size, two-player 20×20 Blokus has a state-space
complexity of **≥ 10⁵⁴ — larger than chess**. The project's arc is: *search solves
the small boards optimally; deep RL scales to boards where search is impossible.*

## Move-generation benchmark

Cost per 100,000 environment steps (seconds, lower is better):

| Board | Naïve | Corner-cell (standard) | **Precomputed (this repo)** |
|------:|------:|-----------------------:|----------------------------:|
| 4×4   | 8.12  | 5.43  | **0.57** |
| 7×7   | 36.15 | 13.37 | **1.38** |
| 10×10 | 76.09 | 22.20 | **2.62** |
| 20×20 | 261.23| 48.71 | **8.93** |

The trick: precompute legal placements per corner-cell bitmask, split into 22
**core** + 6 **peripheral** cells to avoid a ~6 GB table (one-time ~102 s precompute,
cached to disk; ~3 s to load thereafter).

## Agents

All agents subclass a common `Agent` interface (`get_action(env, obs) -> action`):

| Agent | Description |
|---|---|
| `RandomAgent` | Uniformly random legal move (baseline / exploration opponent) |
| `HeuristicAgent` | Configurable hand-designed evaluation of state-action pairs |
| `MinimaxAgent` | Negamax search with a disk-persisted action cache |
| `ABPruningAgent` | Minimax + alpha-beta pruning (proves the 5×5–7×7 forced win) |
| `QAgent` | Tabular Q-learning for small boards (≤ 7×7) |
| `QNetworkAgent` | Deep Q-Network with a configurable architecture; trainable via `DQNTrainer` or the custom dual-Q trainer |
| `MixedAgent` | Combines agents, selecting one per move by a probability distribution |

The DQN encodes the board as a 44-channel tensor (21 shape channels/player + 2
corner-cell channels) concatenated with the proposed action, passed through 3 ResNet
blocks; the trainer uses experience replay, a target network, Huber loss, and Adam.

## Repository layout

```
gymnasium_env/       Fast, Gymnasium-compatible Blokus environment
  envs/              BlokusEnv, SingleAgentBlokusEnv, pieces, actions, theme
  envs/auxiliary/    precompute_fast_env.py (the action-table precomputation)
  wrappers/          reward shaping / action-space wrappers
src/
  agents/            random, heuristic, minimax, ab_pruning, q_learning, qnetwork, mixed
  training/          Q-learning and Q-network training notebooks
  play/              interactive UI to play against the agents
  simulator.py       run agents against each other
tests/               unit tests for the environment and agents
```

## Setup

```bash
git clone https://github.com/MarioPariona117/Blokus.git
cd Blokus
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # installs the gymnasium_env as an editable package
```

Requires Python 3.11+, PyTorch, and Gymnasium.

## Usage

- **Train an agent:** the training pipelines live as notebooks under
  `src/training/` — `qnetwork/` for the DQN and the dual-Q method, `q_learning/`
  for the tabular agent.
- **Watch agents play / evaluate:** `src/simulator.py` runs agents head-to-head.
- **Play against an agent yourself:** an interactive UI in `src/play/`.
- **Run the tests:** `pytest` from the repo root.

## Tech

Python · PyTorch · Gymnasium · NumPy · Weights & Biases (experiment tracking)

## Limitations & future work

Stated honestly in the dissertation: the DQN architecture is relatively shallow and
its performance is moderate; the dual-Q method doesn't always converge reliably
(loss drops but action selection isn't consistently optimal). Future directions:
port the environment backend to C++, add Double/Dueling DQN and prioritised replay,
bake in a symmetry inductive bias, and explore transfer learning across board sizes.

---

*University of Cambridge, Trinity College · Computer Science Tripos Part II · May 2025 · supervised by Dr Petar Veličković.*
