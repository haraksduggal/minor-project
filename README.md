# <img src="https://img.icons8.com/3d-fluency/48/air-conditioner.png" width="32" align="top"/> HVAC Placement Optimizer

> **Find the minimum number of HVAC units and their optimal placement to cool an entire floor — powered by Reinforcement Learning.**

<p>
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/RL-Deep%20Q--Network-FF6B6B?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/NumPy-Pure%20Implementation-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Dependencies-1-4CAF50?style=for-the-badge"/>
</p>

---

### 🧊 The Problem

Placing HVAC units by hand leads to either **over-provisioning** (wasted money and energy) or **dead zones** (uncomfortable occupants). This tool uses a DQN agent to learn optimal placements from scratch — no heuristics, no manual tuning.

### ✅ The Solution

A Reinforcement Learning agent that:

- Takes any 2D floor plan as input
- Learns to place the **fewest HVAC units** needed for ≥95% cooling coverage
- Outputs exact **(row, col) coordinates** for each unit
- Visualizes the result as a colored terminal heatmap

---

## ⚡ Quick Start

```bash
pip install numpy
python main.py
```

That's it. The interactive menu lets you pick a preset floor plan or enter your own.

---

## 🏗️ How It Works

### Input: Floor Map

A 2D grid where each cell is:

| Value | Meaning | Visual |
|:-----:|---------|:------:|
| `0` | Wall — impassable, blocks cooling | ⬛ |
| `1` | Walkable floor — needs cooling | ⬜ |
| `2` | Door/opening — walkable, partial cooling pass-through | 🚪 |

### The RL Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  Floor Map   │────▶│ Thermal Model │────▶│  DQN Agent  │────▶│  Optimal     │
│  (2D Grid)   │     │ (Raycasting)  │     │ (Q-Learning) │     │  Placements  │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
```

1. **Thermal Model** — Each HVAC unit emits cooling that decays exponentially with distance. Walls attenuate cooling via Bresenham raycasting.

2. **RL Environment** — Agent places one unit per step. Episodes end when ≥95% coverage is achieved or max units are exhausted.

3. **DQN Agent** — Pure NumPy implementation with:
   - Experience replay buffer
   - Target network with soft updates
   - Adam optimizer with gradient clipping
   - Epsilon-greedy exploration with decay
   - Invalid action masking (no duplicate placements)

### Reward Shaping

| Signal | Value | Purpose |
|--------|------:|---------|
| Coverage gain | `+100 × Δcoverage` | Maximize new area cooled per unit |
| Unit penalty | `-2.0` per unit | Minimize total units placed |
| Target bonus | `+50` | Reward reaching 95% coverage |
| Invalid action | `-1.0` | Discourage duplicate placements |

---

## 🎮 Usage

### Interactive Mode

```bash
python main.py
```

### Preset Floor Plans

```bash
python main.py --preset 1    # L-Shaped Office
python main.py --preset 2    # Multi-Room Office
python main.py --preset 3    # Long Corridor
python main.py --preset 4    # Open Plan with Pillars
python main.py --preset 5    # U-Shaped Floor
```

### Custom Floor Map (inline JSON)

```bash
python main.py --json '[[0,0,0,0,0],[0,1,1,1,0],[0,1,1,1,0],[0,1,1,1,0],[0,0,0,0,0]]'
```

### Load from File

```bash
python main.py --file my_floor.json
```

Where `my_floor.json` contains:

```json
[
  [0,0,0,0,0,0,0,0],
  [0,1,1,1,1,1,1,0],
  [0,1,1,1,1,1,1,0],
  [0,1,1,0,0,1,1,0],
  [0,1,1,0,0,1,1,0],
  [0,1,1,1,1,1,1,0],
  [0,1,1,1,1,1,1,0],
  [0,0,0,0,0,0,0,0]
]
```

### Tuning Parameters

```bash
python main.py --preset 2 --episodes 800 --radius 6
```

| Flag | Default | Description |
|------|:-------:|-------------|
| `--episodes` | `400` | Training episodes (more = better results, slower) |
| `--radius` | `5` | HVAC cooling radius in grid cells |
| `--quiet` | `off` | Suppress banner, minimal output |

---

## 📊 Example Output

```
  Configuration:
    Grid Size:         14 × 14
    Walkable Cells:    79
    Cooling Radius:    5
    Target Coverage:   95%
    Training Episodes: 400

  Training DQN Agent...

  ╔══════════════════════════════════╗
  ║  HVAC Units Required: 2         ║
  ║  Floor Coverage:      97.5%     ║
  ║                                  ║
  ║  Unit #1:  row=4   col=5        ║
  ║  Unit #2:  row=10  col=3        ║
  ╚══════════════════════════════════╝
```

### Output Includes

- ✅ Minimum number of HVAC units required
- 📍 Exact `(row, col)` coordinates for each unit
- 📈 Coverage percentage achieved
- 🎨 Colored terminal heatmap visualization
- 💾 JSON results file (`hvac_result.json`)

---

## 📁 Project Structure

```
├── main.py                  # Entry point & CLI interface
├── requirements.txt         # Dependencies (numpy only)
├── README.md
├── __init__.py
├── environment.py           # Thermal model + RL environment + DQN agent
├── presets.py               # 5 built-in floor plan templates
└── visualizer.py            # Terminal rendering with ANSI colors
```

---

## 🧠 Technical Highlights

The entire DQN, replay buffer, Q-network, and Adam optimizer are implemented **from scratch in pure NumPy** — no PyTorch, no TensorFlow, no external ML libraries.

| Component | Implementation |
|-----------|---------------|
| Q-Network | 2-layer fully connected (NumPy matrices) |
| Optimizer | Adam with gradient clipping |
| Replay Buffer | Circular buffer with uniform sampling |
| Target Network | Soft updates (τ = 0.01) |
| Exploration | ε-greedy with exponential decay |
| Action Masking | Invalid placements are masked before argmax |

---

## 📋 Requirements

- **Python 3.8+**
- **NumPy** — the only dependency

```bash
pip install numpy
```

---

<p align="center">
  <sub>Built as a Minor Project — B.Tech CSE, Amity University Punjab</sub>
</p>
