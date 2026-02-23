"""Monte Carlo prediction and control for the stochastic GridWorld (Part 2, Task 2)."""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

from .gridworld import GridWorld, State, Action, ACTIONS


def _epsilon_greedy_action(
    q_for_state: Dict[Action, float],
    available_actions: List[Action],
    epsilon: float,
) -> Action:
    if not available_actions:
        raise ValueError("No available actions for epsilon-greedy selection")

    if random.random() < epsilon:
        return random.choice(available_actions)

    # Greedy action selection (ties broken arbitrarily)
    best_a = available_actions[0]
    best_q = q_for_state.get(best_a, 0.0)
    for a in available_actions[1:]:
        q = q_for_state.get(a, 0.0)
        if q > best_q:
            best_q = q
            best_a = a
    return best_a


def mc_control_every_visit(
    env: GridWorld,
    episodes: int = 5000,
    gamma: float = 0.9,
    epsilon: float = 0.1,
    max_steps_per_episode: int = 200,
) -> Tuple[Dict[State, Dict[Action, float]], Dict[State, Action]]:
    """Every-visit Monte Carlo control with epsilon-greedy policy.

    Returns
    -------
    Q : dict
        State-action value estimates.
    policy : dict
        Greedy policy derived from Q.
    """
    states = env.get_all_states()

    Q: Dict[State, Dict[Action, float]] = {}
    returns_sum: Dict[State, Dict[Action, float]] = {}
    returns_count: Dict[State, Dict[Action, float]] = {}

    for s in states:
        actions = env.get_actions(s)
        Q[s] = {a: 0.0 for a in actions}
        returns_sum[s] = {a: 0.0 for a in actions}
        returns_count[s] = {a: 0.0 for a in actions}

    for _ in range(episodes):
        state = env.start  # type: ignore[attr-defined]
        episode: List[Tuple[State, Action, float]] = []

        for _t in range(max_steps_per_episode):
            actions = env.get_actions(state)
            if not actions:
                break
            action = _epsilon_greedy_action(Q[state], actions, epsilon)
            next_state, reward, done = env.step(state, action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break

        # Every-visit MC update
        G = 0.0
        for t in reversed(range(len(episode))):
            s_t, a_t, r_t = episode[t]
            G = gamma * G + r_t
            returns_sum[s_t][a_t] += G
            returns_count[s_t][a_t] += 1.0
            Q[s_t][a_t] = returns_sum[s_t][a_t] / returns_count[s_t][a_t]

    policy: Dict[State, Action] = {}
    for s in states:
        actions = env.get_actions(s)
        if not actions:
            continue
        # Greedy action from Q
        best_a = actions[0]
        best_q = Q[s].get(best_a, 0.0)
        for a in actions[1:]:
            q = Q[s].get(a, 0.0)
            if q > best_q:
                best_q = q
                best_a = a
        policy[s] = best_a

    return Q, policy
