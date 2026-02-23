"""
Running ``python main.py`` will:
- Solve the shortest-path problems on the NYC graph (if available) or a
  small built-in example using Dijkstra, UCS with energy constraint, and
  A* with an admissible Euclidean heuristic.
- Solve the 5x5 GridWorld MDP using value iteration and policy iteration
  in the deterministic setting.
- Train Monte Carlo control and tabular Q-learning agents on the
  stochastic GridWorld and print the learned policies.
"""
from __future__ import annotations

import argparse
import os
import random
from typing import Optional

from helpers.graph_loader import load_graph_data
from helpers.rcsp_search import (
    astar_shortest_path_with_energy,
    dijkstra_shortest_path,
    ucs_shortest_path_with_energy,
)
from helpers.gridworld import GridWorld
from helpers.dp_planning import (
    value_iteration,
    policy_iteration,
    print_value_table,
    print_policy_table,
)
from helpers.mc_rl import mc_control_every_visit
from helpers.q_learning import q_learning


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CZ3005 Lab Assignment 1")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=".",
        help="Directory containing G/Coord/Dist/Cost dictionaries (JSON or pickle).",
    )
    parser.add_argument(
        "--mc_episodes",
        type=int,
        default=None,
        help="Number of training episodes for Monte Carlo control (override env / default).",
    )
    parser.add_argument(
        "--ql_episodes",
        type=int,
        default=None,
        help="Number of training episodes for Q-learning (override env / default).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility of RL components.",
    )
    return parser.parse_args()


def _resolve_episodes(arg_value: Optional[int], env_var: str, default: int) -> int:
    if arg_value is not None:
        return arg_value
    value = os.environ.get(env_var)
    if value is not None:
        try:
            return int(value)
        except ValueError:
            print(f"Warning: invalid value for {env_var!r}: {value!r}. Using default {default}.")
    return default


def run_part1(data_dir: str) -> None:
    print("=== Part 1: Shortest Path with Energy Budget ===")

    graph = load_graph_data(data_dir)
    G, Coord, Dist, Cost = graph.G, graph.Coord, graph.Dist, graph.Cost
    start, goal = graph.source, graph.target
    energy_budget = graph.energy_budget

    # Task 1: Dijkstra without energy constraint
    print("\nTask 1: Shortest path without energy constraint (Dijkstra)")
    res1 = dijkstra_shortest_path(G, Dist, Cost, start, goal)
    if res1.path is None:
        print("No path found by Dijkstra.")
    else:
        path_str = "->".join(res1.path)
        print(f"Shortest path: {path_str}.")
        print(f"Shortest distance: {res1.distance}.")
        print(f"Total energy cost: {res1.energy}.")

    if energy_budget is None:
        print("No energy budget specified; skipping Tasks 2 and 3.")
        return

    # Task 2: UCS with energy budget
    print("\nTask 2: Uniform Cost Search with energy budget constraint")
    res2 = ucs_shortest_path_with_energy(G, Dist, Cost, start, goal, energy_budget)
    if res2.path is None:
        print("No feasible path within the energy budget using UCS.")
    else:
        path_str = "->".join(res2.path)
        print(f"Shortest path: {path_str}.")
        print(f"Shortest distance: {res2.distance}.")
        print(f"Total energy cost: {res2.energy}.")

    # Task 3: A* with Euclidean heuristic and label-setting RCSP
    print("\nTask 3: A* search with Euclidean heuristic and energy budget constraint")
    res3 = astar_shortest_path_with_energy(G, Coord, Dist, Cost, start, goal, energy_budget)
    if res3.path is None:
        print("No feasible path within the energy budget using A*.")
    else:
        path_str = "->".join(res3.path)
        print(f"Shortest path: {path_str}.")
        print(f"Shortest distance: {res3.distance}.")
        print(f"Total energy cost: {res3.energy}.")


def run_part2(mc_episodes: int, ql_episodes: int) -> None:
    print("\n=== Part 2: GridWorld MDP and Reinforcement Learning ===")

    # Deterministic environment for planning
    det_env = GridWorld(stochastic=False)

    # Task 1: Value Iteration
    V_vi, policy_vi = value_iteration(det_env, gamma=0.9)
    print_value_table(det_env, V_vi, "Value function from Value Iteration (deterministic):")
    print_policy_table(det_env, policy_vi, "Optimal policy from Value Iteration:")

    # Task 1: Policy Iteration
    V_pi, policy_pi = policy_iteration(det_env, gamma=0.9)
    print_value_table(det_env, V_pi, "Value function from Policy Iteration (deterministic):")
    print_policy_table(det_env, policy_pi, "Optimal policy from Policy Iteration:")

    # Stochastic environment for RL
    rl_env = GridWorld(stochastic=True)

    # Task 2: Monte Carlo control
    print(f"Training Monte Carlo control for {mc_episodes} episodes...")
    Q_mc, policy_mc = mc_control_every_visit(rl_env, episodes=mc_episodes, gamma=0.9)
    print_policy_table(rl_env, policy_mc, "Monte Carlo control policy (stochastic):")

    # Task 3: Q-learning
    print(f"Training Q-learning for {ql_episodes} episodes...")
    Q_ql, policy_ql = q_learning(rl_env, episodes=ql_episodes, gamma=0.9)
    print_policy_table(rl_env, policy_ql, "Q-learning policy (stochastic):")

    print(
        "Note: Monte Carlo uses full episodes and tends to be less sample-efficient "
        "than Q-learning, which updates online at each step."
    )


def main() -> None:
    args = _parse_args()

    # Seed RNG for reproducibility of stochastic transitions and RL.
    random.seed(args.seed)

    mc_episodes = _resolve_episodes(args.mc_episodes, "MC_EPISODES", default=5000)
    ql_episodes = _resolve_episodes(args.ql_episodes, "QL_EPISODES", default=5000)

    run_part1(args.data_dir)
    run_part2(mc_episodes, ql_episodes)


if __name__ == "__main__":  # pragma: no cover
    main()
