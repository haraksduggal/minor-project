# HVAC Placement Optimizer

**Find the minimum number of HVAC units and their optimal placement to cool an entire floor — using Reinforcement Learning.**

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![RL](https://img.shields.io/badge/RL-DQN-green)
![NumPy](https://img.shields.io/badge/NumPy-only-orange)

---

## Quick Start

```bash
pip install numpy
python main.py
```

That's it. The interactive menu lets you pick a preset floor plan or enter your own.

---

## How It Works

### Input: Floor Map
A 2D grid where each cell is:
| Value | Meaning |
|-------|---------|
| `0`   | Wall — impassable, blocks cooling |
| `1`   | Walkable floor — needs cooling |
| `2`   | Door/opening — walkable, partial cooling pass-through |

### The RL Pipeline
1. **Thermal Model** — Each HVAC unit emits cooling that decays exponentially with distance. Walls attenuate cooling via Bresenham raycasting.
2. **RL Environment** — Agent places one unit per step. Episodes end when ≥95% coverage is achieved or max units exhausted.
3. **DQN Agent** — Pure NumPy implementation with:
   - Experience replay buffer
   - Target network with soft updates
   - Adam optimizer with gradient clipping
   - Epsilon-greedy exploration with decay
   - Invalid action masking (no duplicate placements)

### Reward Shaping
- `+100 × coverage_gain` — incentivize each unit to maximize new coverage
- `-2.0` per unit — penalize using more units
- `+50` bonus — when target coverage (95%) is reached
- `-1.0` — for attempting duplicate placement

### Output
- Number of HVAC units required
- `(row, col)` coordinates for each unit
- Coverage percentage achieved
- Colored terminal heatmap visualization
- JSON results file (`hvac_result.json`)

---

## Usage

### Interactive Mode
```bash
python main.py
```

### Use a Preset
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
|------|---------|-------------|
| `--episodes` | 400 | Training episodes (more = better results, slower) |
| `--radius` | 5 | HVAC cooling radius in grid cells |
| `--quiet` | off | Suppress banner, minimal output |

---

## Project Structure

```
hvac_optimizer/
├── main.py                  # Entry point (CLI)
├── requirements.txt         # Dependencies (numpy only)
├── README.md
└── hvac_rl/
    ├── __init__.py
    ├── environment.py       # Thermal model + RL env + DQN agent
    ├── presets.py            # 5 built-in floor plans
    └── visualizer.py        # Terminal rendering with ANSI colors
```

---

## Example Output

```
  Configuration:
    Grid Size:        14 × 14
    Walkable Cells:   79
    Cooling Radius:   5
    Target Coverage:  95%
    Training Episodes:400

  Training DQN Agent...

  ╔══════════════════════════════════╗
  │  HVAC Units Required: 2         │
  │  Floor Coverage:      97.5%     │
  │                                  │
  │  Unit #1:  row=4   col=5        │
  │  Unit #2:  row=10  col=3        │
  └──────────────────────────────────┘
```

---

## Dependencies

- **Python 3.8+**
- **NumPy** (the only dependency — no PyTorch/TensorFlow needed)

The entire DQN, replay buffer, Q-network, and Adam optimizer are implemented from scratch in pure NumPy.
