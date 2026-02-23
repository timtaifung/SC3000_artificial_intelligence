"""Tabular Q-learning for the stochastic GridWorld (Part 2, Task 3)."""
from __future__ import annotations

from typing import Dict, Tuple

from .gridworld import GridWorld, State, Action
from .mc_rl import _epsilon_greedy_action


def q_learning(
    env: GridWorld,
    episodes: int = 5000,
    gamma: float = 0.9,
    epsilon: float = 0.1,
    alpha: float = 0.1,
    max_steps_per_episode: int = 200,
) -> Tuple[Dict[State, Dict[Action, float]], Dict[State, Action]]:
    """Tabular Q-learning with epsilon-greedy exploration.

    Returns
    -------
    Q : dict
        Learned state-action values.
    policy : dict
        Greedy policy derived from Q.
    """
    states = env.get_all_states()

    Q: Dict[State, Dict[Action, float]] = {}
    for s in states:
        actions = env.get_actions(s)
        Q[s] = {a: 0.0 for a in actions}

    for _ in range(episodes):
        state = env.start  # type: ignore[attr-defined]

        for _t in range(max_steps_per_episode):
            actions = env.get_actions(state)
            if not actions:
                break

            action = _epsilon_greedy_action(Q[state], actions, epsilon)
            next_state, reward, done = env.step(state, action)

            next_actions = env.get_actions(next_state)
            if not next_actions:
                target = reward
            else:
                max_next_q = max(Q[next_state].get(a, 0.0) for a in next_actions)
                target = reward + gamma * max_next_q

            old_q = Q[state].get(action, 0.0)
            Q[state][action] = old_q + alpha * (target - old_q)

            state = next_state
            if done:
                break

    policy: Dict[State, Action] = {}
    for s in states:
        actions = env.get_actions(s)
        if not actions:
            continue
        best_a = actions[0]
        best_q = Q[s].get(best_a, 0.0)
        for a in actions[1:]:
            q = Q[s].get(a, 0.0)
            if q > best_q:
                best_q = q
                best_a = a
        policy[s] = best_a

    return Q, policy
