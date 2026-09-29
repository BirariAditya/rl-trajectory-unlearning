"""An offline empirical MDP with removable per-trajectory sufficient statistics.

The learner is explicitly defined as trajectory counts -> empirical MDP -> planner.
An unseen state-action row has reward zero and terminates. This fixed convention
must be identical in the original and counterfactual retained-data learner.
"""

from collections import Counter
from math import isfinite
from typing import Iterable, Mapping

from .data import Trajectory, Transition

StateAction = tuple[int, int]
Successor = int | None
QTable = Mapping[StateAction, float]


class EmpiricalMDP:
    def __init__(
        self,
        states: Iterable[int],
        actions: Iterable[int],
        gamma: float,
        trajectories: Iterable[Trajectory] = (),
    ) -> None:
        self.states = tuple(states)
        self.actions = tuple(actions)
        if not self.states or len(set(self.states)) != len(self.states):
            raise ValueError("states must be a nonempty sequence of unique integers")
        if not self.actions or len(set(self.actions)) != len(self.actions):
            raise ValueError("actions must be a nonempty sequence of unique integers")
        if not 0 <= gamma < 1:
            raise ValueError("gamma must satisfy 0 <= gamma < 1")
        self.gamma = float(gamma)
        self._visits: Counter[StateAction] = Counter()
        self._successors_by_row: dict[StateAction, Counter[Successor]] = {}
        self._reward_sums: dict[StateAction, float] = {}
        self._trajectories: dict[str, Trajectory] = {}
        for trajectory in trajectories:
            self.add_trajectory(trajectory)

    @property
    def trajectory_ids(self) -> frozenset[str]:
        return frozenset(self._trajectories)

    def active_trajectories(self) -> tuple[Trajectory, ...]:
        return tuple(self._trajectories.values())

    def _validate_transition(self, transition: Transition) -> None:
        if transition.state not in self.states:
            raise ValueError(f"unknown state {transition.state}")
        if transition.action not in self.actions:
            raise ValueError(f"unknown action {transition.action}")
        if transition.next_state is not None and transition.next_state not in self.states:
            raise ValueError(f"unknown successor state {transition.next_state}")
        if not isfinite(transition.reward):
            raise ValueError("rewards must be finite")

    def _apply(self, trajectory: Trajectory, sign: int) -> set[StateAction]:
        changed: set[StateAction] = set()
        for transition in trajectory.transitions:
            row = (transition.state, transition.action)
            self._visits[row] += sign
            row_successors = self._successors_by_row.setdefault(row, Counter())
            row_successors[transition.next_state] += sign
            self._reward_sums[row] = self._reward_sums.get(row, 0.0) + sign * transition.reward
            if self._visits[row] == 0:
                del self._visits[row]
                self._reward_sums.pop(row, None)
            if row_successors[transition.next_state] == 0:
                del row_successors[transition.next_state]
            if not row_successors:
                del self._successors_by_row[row]
            changed.add(row)
        return changed

    def add_trajectory(self, trajectory: Trajectory) -> set[StateAction]:
        if trajectory.trajectory_id in self._trajectories:
            raise ValueError(f"duplicate trajectory id: {trajectory.trajectory_id}")
        for transition in trajectory.transitions:
            self._validate_transition(transition)
        changed = self._apply(trajectory, +1)
        self._trajectories[trajectory.trajectory_id] = trajectory
        return changed

    def delete_trajectories(self, ids: Iterable[str]) -> set[StateAction]:
        """Atomically validate a deletion request, then remove logged experience."""
        requested = tuple(ids)
        if len(requested) != len(set(requested)):
            raise ValueError("a deletion request contains duplicate trajectory ids")
        unknown = set(requested) - self.trajectory_ids
        if unknown:
            raise KeyError(f"unknown or already deleted trajectory ids: {sorted(unknown)}")
        changed: set[StateAction] = set()
        for trajectory_id in requested:
            trajectory = self._trajectories.pop(trajectory_id)
            changed.update(self._apply(trajectory, -1))
        return changed

    def count(self, state: int, action: int) -> int:
        return self._visits[(state, action)]

    def mean_reward(self, state: int, action: int) -> float:
        row = (state, action)
        n = self._visits[row]
        return self._reward_sums[row] / n if n else 0.0

    def transition_probabilities(self, state: int, action: int) -> dict[Successor, float]:
        n = self.count(state, action)
        if not n:
            return {None: 1.0}
        return {
            successor: count / n
            for successor, count in self._successors_by_row[(state, action)].items()
        }

    def backup(self, q: QTable, state: int, action: int) -> float:
        continuation = 0.0
        for successor, probability in self.transition_probabilities(state, action).items():
            if successor is not None:
                continuation += probability * max(q[(successor, a)] for a in self.actions)
        return self.mean_reward(state, action) + self.gamma * continuation

    def predecessor_rows(self) -> dict[int, set[StateAction]]:
        """Every action that can reach a state, including non-greedy actions."""
        reverse: dict[int, set[StateAction]] = {state: set() for state in self.states}
        for row, successors in self._successors_by_row.items():
            for successor, count in successors.items():
                if successor is not None and count:
                    reverse[successor].add(row)
        return reverse
