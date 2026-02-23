"""5x5 GridWorld environment for Part 2.

The environment supports:
- Deterministic transitions (for planning with value/policy iteration).
- Stochastic transitions (for Monte Carlo and Q-learning).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


State = Tuple[int, int]
Action = str  # 'U', 'D', 'L', 'R'

ACTIONS: Sequence[Action] = ("U", "D", "L", "R")
ARROW_MAP: Dict[Action, str] = {"U": "↑", "D": "↓", "L": "←", "R": "→"}


@dataclass
class GridWorld:
    width: int = 5
    height: int = 5
    obstacles: Iterable[State] = ((2, 1), (2, 3))
    start: State = (0, 0)
    goal: State = (4, 4)
    step_reward: float = -1.0
    goal_reward: float = 10.0
    stochastic: bool = False

    def __post_init__(self) -> None:
        self.obstacles = set(self.obstacles)

    # --- basic helpers -------------------------------------------------

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_obstacle(self, state: State) -> bool:
        return state in self.obstacles

    def is_terminal(self, state: State) -> bool:
        return state == self.goal

    def get_all_states(self) -> List[State]:
        states: List[State] = []
        for x in range(self.width):
            for y in range(self.height):
                s = (x, y)
                if not self.is_obstacle(s):
                    states.append(s)
        return states

    def get_actions(self, state: State) -> List[Action]:
        if self.is_terminal(state):
            return []
        return list(ACTIONS)

    # --- deterministic dynamics ----------------------------------------

    def _move_deterministic(self, state: State, action: Action) -> State:
        if self.is_terminal(state):
            return state

        x, y = state
        if action == "U":
            new_state = (x, y + 1)
        elif action == "D":
            new_state = (x, y - 1)
        elif action == "L":
            new_state = (x - 1, y)
        elif action == "R":
            new_state = (x + 1, y)
        else:
            raise ValueError(f"Unknown action: {action!r}")

        if not self.in_bounds(*new_state) or self.is_obstacle(new_state):
            return state
        return new_state

    def step_deterministic(self, state: State, action: Action) -> Tuple[State, float, bool]:
        """Deterministic transition model used by planning algorithms."""
        if self.is_terminal(state):
            return state, 0.0, True

        next_state = self._move_deterministic(state, action)
        if next_state == self.goal:
            return next_state, self.goal_reward, True
        return next_state, self.step_reward, False

    # --- stochastic dynamics -------------------------------------------

    def _sample_actual_action(self, intended: Action) -> Action:
        """Sample actual action under the 0.8 / 0.1 / 0.1 model."""
        if intended not in ACTIONS:
            raise ValueError(f"Unknown action: {intended!r}")

        r = random.random()
        if r < 0.8:
            return intended

        # Perpendicular actions
        if intended in ("U", "D"):
            perpendicular = ("L", "R")
        else:  # 'L' or 'R'
            perpendicular = ("U", "D")

        if r < 0.9:
            return perpendicular[0]
        return perpendicular[1]

    def step(self, state: State, action: Action) -> Tuple[State, float, bool]:
        """Environment step for RL algorithms.

        If ``self.stochastic`` is False, this reduces to the deterministic
        model; otherwise the 0.8 / 0.1 / 0.1 transition model is used.
        """
        if not self.stochastic:
            return self.step_deterministic(state, action)

        if self.is_terminal(state):
            return state, 0.0, True

        actual_action = self._sample_actual_action(action)
        next_state = self._move_deterministic(state, actual_action)
        if next_state == self.goal:
            return next_state, self.goal_reward, True
        return next_state, self.step_reward, False
