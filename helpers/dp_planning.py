"""Dynamic programming algorithms for the GridWorld MDP (Part 2, Task 1)."""
from __future__ import annotations

from typing import Dict, Tuple

from .gridworld import ARROW_MAP, GridWorld, State, Action


def value_iteration(
    env: GridWorld,
    gamma: float = 0.9,
    theta: float = 1e-4,
) -> Tuple[Dict[State, float], Dict[State, Action]]:
    """Run value iteration and return (value_function, greedy_policy)."""
    states = env.get_all_states()
    V: Dict[State, float] = {s: 0.0 for s in states}

    while True:
        delta = 0.0
        for s in states:
            if env.is_terminal(s):
                continue
            best_value = None
            for a in env.get_actions(s):
                ns, r, done = env.step_deterministic(s, a)
                v = r + gamma * (0.0 if done else V[ns])
                if best_value is None or v > best_value:
                    best_value = v
            assert best_value is not None
            diff = abs(best_value - V[s])
            delta = max(delta, diff)
            V[s] = best_value
        if delta < theta:
            break

    policy: Dict[State, Action] = {}
    for s in states:
        if env.is_terminal(s):
            continue
        best_action: Action | None = None
        best_value = None
        for a in env.get_actions(s):
            ns, r, done = env.step_deterministic(s, a)
            v = r + gamma * (0.0 if done else V[ns])
            if best_value is None or v > best_value:
                best_value = v
                best_action = a
        assert best_action is not None
        policy[s] = best_action

    return V, policy


def policy_iteration(
    env: GridWorld,
    gamma: float = 0.9,
    theta: float = 1e-4,
) -> Tuple[Dict[State, float], Dict[State, Action]]:
    """Run policy iteration and return (value_function, optimal_policy)."""
    states = env.get_all_states()

    # Initial policy: always move Right (arbitrary choice) where possible.
    policy: Dict[State, Action] = {}
    for s in states:
        if env.is_terminal(s):
            continue
        actions = env.get_actions(s)
        policy[s] = actions[0] if actions else "U"

    V: Dict[State, float] = {s: 0.0 for s in states}

    while True:
        # Policy evaluation
        while True:
            delta = 0.0
            for s in states:
                v_old = V[s]
                if env.is_terminal(s):
                    continue
                a = policy[s]
                ns, r, done = env.step_deterministic(s, a)
                V[s] = r + gamma * (0.0 if done else V[ns])
                delta = max(delta, abs(v_old - V[s]))
            if delta < theta:
                break

        # Policy improvement
        policy_stable = True
        for s in states:
            if env.is_terminal(s):
                continue
            old_action = policy[s]

            best_action: Action | None = None
            best_value = None
            for a in env.get_actions(s):
                ns, r, done = env.step_deterministic(s, a)
                v = r + gamma * (0.0 if done else V[ns])
                if best_value is None or v > best_value:
                    best_value = v
                    best_action = a

            assert best_action is not None
            policy[s] = best_action
            if best_action != old_action:
                policy_stable = False

        if policy_stable:
            break

    return V, policy


def print_value_table(env: GridWorld, V: Dict[State, float], title: str) -> None:
    """Pretty-print a 5x5 table of state values."""
    print(title)
    for y in reversed(range(env.height)):
        row_cells = []
        for x in range(env.width):
            s = (x, y)
            if s in env.obstacles:
                cell = "#####"
            else:
                v = V.get(s, 0.0)
                cell = f"{v:6.2f}"
            row_cells.append(cell)
        print(" ".join(row_cells))
    print()


def print_policy_table(env: GridWorld, policy: Dict[State, Action], title: str) -> None:
    """Pretty-print a 5x5 table of policy arrows."""
    print(title)
    for y in reversed(range(env.height)):
        row_cells = []
        for x in range(env.width):
            s = (x, y)
            if s in env.obstacles:
                cell = "X"
            elif env.is_terminal(s):
                cell = "G"
            else:
                a = policy.get(s)
                cell = ARROW_MAP.get(a, ".") if a is not None else "."
            row_cells.append(f"  {cell}  ")
        print("".join(row_cells))
    print()
