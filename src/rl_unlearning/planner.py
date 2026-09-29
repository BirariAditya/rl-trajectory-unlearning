"""Deterministic full value iteration and retraining-relative diagnostics."""

from dataclasses import dataclass
from typing import Mapping

from .tabular import EmpiricalMDP, QTable, StateAction


@dataclass(frozen=True, slots=True)
class PlanResult:
    q: dict[StateAction, float]
    policy: dict[int, int]
    residual: float
    sweeps: int
    backups: int
    residual_checks: int


def zero_q(model: EmpiricalMDP) -> dict[StateAction, float]:
    return {(s, a): 0.0 for s in model.states for a in model.actions}


def greedy_policy(model: EmpiricalMDP, q: QTable) -> dict[int, int]:
    # max preserves the first action in model.actions when values tie.
    return {state: max(model.actions, key=lambda action: q[(state, action)]) for state in model.states}


def bellman_residual(model: EmpiricalMDP, q: QTable) -> float:
    return max(
        abs(model.backup(q, state, action) - q[(state, action)])
        for state in model.states
        for action in model.actions
    )


def value_iteration(
    model: EmpiricalMDP,
    *,
    tolerance: float = 1e-10,
    max_sweeps: int = 100_000,
    initial_q: Mapping[StateAction, float] | None = None,
) -> PlanResult:
    if tolerance <= 0 or max_sweeps <= 0:
        raise ValueError("tolerance and max_sweeps must be positive")
    q = zero_q(model) if initial_q is None else {row: float(initial_q[row]) for row in zero_q(model)}
    rows = [(s, a) for s in model.states for a in model.actions]
    for sweep in range(1, max_sweeps + 1):
        updated = {row: model.backup(q, *row) for row in rows}
        q = updated
        residual = bellman_residual(model, q)
        if residual <= tolerance:
            return PlanResult(q, greedy_policy(model, q), residual, sweep, sweep * len(rows), sweep * len(rows))
    raise RuntimeError("value iteration did not reach its Bellman residual tolerance")
