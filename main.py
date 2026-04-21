#!/usr/bin/env python3
"""
HVAC Placement Optimizer — Main Entry Point
=============================================

Usage:
    python main.py                          # Interactive mode (pick a preset)
    python main.py --preset 2               # Use preset #2 directly
    python main.py --file floor.json        # Load floor map from JSON file
    python main.py --json '[[0,0],[0,1]]'   # Inline JSON floor map
    python main.py --episodes 600           # Custom episode count
    python main.py --radius 6               # Custom cooling radius

Floor Map Format (JSON):
    2D array where: 0 = wall, 1 = walkable floor, 2 = door/opening

Example:
    python main.py --preset 1 --episodes 400 --radius 5
"""

import argparse
import json
import sys
import os
import numpy as np

# Add parent to path so imports work from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hvac_rl.environment import ThermalConfig, train_hvac_placement, compute_cooling_map
from hvac_rl.presets import ALL_PRESETS
from hvac_rl.visualizer import (
    render_floor_map, render_result,
    render_training_progress, rgb_fg, RESET, BOLD, DIM
)


CYAN = rgb_fg(0, 229, 255)
GREEN = rgb_fg(0, 255, 160)


def print_banner():
    print(f"""
  {BOLD}{CYAN}╔══════════════════════════════════════════════════════╗
  ║                                                      ║
  ║   ██╗  ██╗██╗   ██╗ █████╗  ██████╗                  ║
  ║   ██║  ██║██║   ██║██╔══██╗██╔════╝                  ║
  ║   ███████║██║   ██║███████║██║                        ║
  ║   ██╔══██║╚██╗ ██╔╝██╔══██║██║                        ║
  ║   ██║  ██║ ╚████╔╝ ██║  ██║╚██████╗                  ║
  ║   ╚═╝  ╚═╝  ╚═══╝  ╚═╝  ╚═╝ ╚═════╝                  ║
  ║                                                      ║
  ║   Placement Optimizer using Reinforcement Learning   ║
  ║   DQN Agent · Thermal Simulation · Raycasting        ║
  ║                                                      ║
  ╚══════════════════════════════════════════════════════╝{RESET}
    """)


def select_preset_interactive():
    """Interactive preset selection menu."""
    print(f"\n  {BOLD}{CYAN}Select a floor plan:{RESET}\n")
    for key, fn in ALL_PRESETS.items():
        floor, name = fn()
        walkable = int(np.sum(floor > 0))
        rows, cols = floor.shape
        print(f"    {BOLD}{CYAN}[{key}]{RESET}  {name}  "
              f"{DIM}({rows}×{cols}, {walkable} walkable cells){RESET}")

    print(f"\n    {BOLD}{CYAN}[C]{RESET}  Custom — enter JSON array")
    print(f"    {BOLD}{CYAN}[Q]{RESET}  Quit\n")

    while True:
        choice = input(f"  {CYAN}▶{RESET} Enter choice: ").strip().upper()
        if choice == 'Q':
            sys.exit(0)
        elif choice == 'C':
            return get_custom_floor()
        elif choice in ALL_PRESETS:
            return ALL_PRESETS[choice]()
        else:
            print(f"  {DIM}Invalid choice, try again.{RESET}")


def get_custom_floor():
    """Prompt user for a custom JSON floor map."""
    print(f"\n  {DIM}Enter your floor map as a JSON 2D array.{RESET}")
    print(f"  {DIM}Values: 0=wall, 1=floor, 2=door{RESET}")
    print(f"  {DIM}Example: [[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]]{RESET}\n")

    while True:
        raw = input(f"  {CYAN}▶{RESET} JSON: ").strip()
        try:
            data = json.loads(raw)
            floor = np.array(data, dtype=int)
            if floor.ndim != 2:
                raise ValueError("Must be a 2D array")
            if not np.all((floor >= 0) & (floor <= 2)):
                raise ValueError("Values must be 0, 1, or 2")
            walkable = int(np.sum(floor > 0))
            if walkable == 0:
                raise ValueError("Floor map has no walkable cells")
            return floor, f"Custom Floor ({floor.shape[0]}×{floor.shape[1]})"
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  {rgb_fg(255,80,80)}Error: {e}. Try again.{RESET}")


def run(floor_map, name, n_episodes, cooling_radius):
    """Run the full optimization pipeline."""
    rows, cols = floor_map.shape
    walkable = int(np.sum(floor_map > 0))

    # Show the input floor map
    render_floor_map(floor_map, title=name)

    # Print config
    config = ThermalConfig(cooling_radius=cooling_radius)
    print(f"\n  {BOLD}{CYAN}Configuration:{RESET}")
    print(f"    Grid Size:        {rows} × {cols}")
    print(f"    Walkable Cells:   {walkable}")
    print(f"    Cooling Radius:   {config.cooling_radius}")
    print(f"    Target Coverage:  {config.target_coverage_pct*100:.0f}%")
    print(f"    Training Episodes:{n_episodes}")

    # Train
    print(f"\n  {BOLD}{CYAN}Training DQN Agent...{RESET}\n")

    result = train_hvac_placement(
        floor_map, config,
        n_episodes=n_episodes,
        verbose=True,
    )

    # Compute final heatmap
    positions = [tuple(int(x) for x in p) for p in result["best_positions"]]
    cooling = compute_cooling_map(floor_map, positions, config)

    # Render result
    render_result(floor_map, positions, cooling, result["best_coverage"], title=f"{name} — Optimized")

    # JSON output for programmatic use
    output = {
        "floor_name": name,
        "floor_shape": [rows, cols],
        "walkable_cells": walkable,
        "hvac_units_required": result["best_n_units"],
        "coverage_percent": result["best_coverage"],
        "unit_positions": [{"row": int(r), "col": int(c)} for r, c in positions],
        "cooling_radius": cooling_radius,
        "episodes_trained": n_episodes,
    }

    # Save to file
    out_path = "hvac_result.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  {DIM}Results saved to {out_path}{RESET}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="HVAC Placement Optimizer using Reinforcement Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        # Interactive mode
  python main.py --preset 2             # Multi-room preset
  python main.py --file my_floor.json   # Load from file
  python main.py --json '[[0,0,0],[0,1,0],[0,0,0]]'
  python main.py --preset 4 --episodes 800 --radius 6

Floor Map JSON Format:
  2D array of integers:
    0 = Wall (impassable, blocks cooling)
    1 = Walkable floor (needs cooling)
    2 = Door/opening (walkable, partial cooling pass-through)
        """,
    )
    parser.add_argument("--preset", type=str, choices=list(ALL_PRESETS.keys()),
                        help="Use a built-in preset floor plan (1-5)")
    parser.add_argument("--file", type=str,
                        help="Path to a JSON file containing the 2D floor map array")
    parser.add_argument("--json", type=str,
                        help="Inline JSON 2D array for the floor map")
    parser.add_argument("--episodes", type=int, default=400,
                        help="Number of training episodes (default: 400)")
    parser.add_argument("--radius", type=int, default=5,
                        help="HVAC cooling radius in grid cells (default: 5)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress banner and use minimal output")

    args = parser.parse_args()

    if not args.quiet:
        print_banner()

    # Determine floor map source
    if args.json:
        try:
            data = json.loads(args.json)
            floor_map = np.array(data, dtype=int)
            name = f"Custom Floor ({floor_map.shape[0]}×{floor_map.shape[1]})"
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            sys.exit(1)

    elif args.file:
        try:
            with open(args.file) as f:
                data = json.load(f)
            floor_map = np.array(data, dtype=int)
            name = f"Loaded from {args.file} ({floor_map.shape[0]}×{floor_map.shape[1]})"
        except Exception as e:
            print(f"Error loading file: {e}")
            sys.exit(1)

    elif args.preset:
        floor_map, name = ALL_PRESETS[args.preset]()

    else:
        # Interactive mode
        floor_map, name = select_preset_interactive()

    run(floor_map, name, args.episodes, args.radius)


if __name__ == "__main__":
    main()
