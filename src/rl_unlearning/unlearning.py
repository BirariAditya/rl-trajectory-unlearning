"""Local Bellman repair after trajectory deletion with an action-gap certificate."""

from dataclasses import dataclass
import heapq
from typing import Iterable, Mapping

from .planner import PlanResult, bellman_residual, greedy_policy, value_iteration, zero_q
from .tabular import EmpiricalMDP, StateAction


@dataclass(frozen=True, slots=True)
class RepairResult:
    q: dict[StateAction, float]
    policy: dict[int, int]
    residual: float
    certified_optimal_policy: bool
    used_full_fallback: bool
    local_backups: int
    local_target_evaluations: int
    residual_checks: int
    fallback_plan: PlanResult | None


def _action_gaps_certify(model: EmpiricalMDP, q: Mapping[StateAction, float], residual: float) -> bool:
    if len(model.actions) == 1:
        return True
    error_bound = residual / (1.0 - model.gamma)
    for state in model.states:
        ranked = sorted((q[(state, action)] for action in model.actions), reverse=True)
        if ranked[0] - ranked[1] <= 2.0 * error_bound:
            return False
    return True


def delete_and_repair(
    model: EmpiricalMDP,
    old_q: Mapping[StateAction, float],
    trajectory_ids: Iterable[str],
    *,
    tolerance: float = 1e-10,
    local_backup_budget: int = 10_000,
    fallback_max_sweeps: int = 100_000,
) -> RepairResult:
    """Delete from an offline empirical MDP, locally repair Q, then certify or solve.

    The certificate proves greedy-policy equality to the *optimal policy of this
    retained empirical MDP* under a strict action gap. It does not certify the
    deletion of an arbitrary RL learner's weights or released metadata.
    """
    if tolerance <= 0 or local_backup_budget < 0 or fallback_max_sweeps <= 0:
        raise ValueError("tolerance and fallback_max_sweeps must be positive; local_backup_budget must be nonnegative")
    rows = zero_q(model)
    q = {row: float(old_q[row]) for row in rows}
    changed = model.delete_trajectories(trajectory_ids)
    predecessors = model.predecessor_rows()

    queue: list[tuple[float, int, StateAction]] = []
    serial = 0
    local_target_evaluations = 0

    def schedule(row: StateAction) -> None:
        nonlocal serial, local_target_evaluations
        residual = abs(model.backup(q, *row) - q[row])
        local_target_evaluations += 1
        if residual > tolerance:
            heapq.heappush(queue, (-residual, serial, row))
            serial += 1

    for row in sorted(changed):
        schedule(row)

    local_backups = 0
    while queue and local_backups < local_backup_budget:
        _, _, row = heapq.heappop(queue)
        target = model.backup(q, *row)
        local_target_evaluations += 1
        if abs(target - q[row]) <= tolerance:
            continue
        state, _ = row
        previous_value = max(q[(state, action)] for action in model.actions)
        q[row] = target
        local_backups += 1
        current_value = max(q[(state, action)] for action in model.actions)
        if abs(current_value - previous_value) > tolerance:
            for predecessor in predecessors[state]:
                schedule(predecessor)

    # A full scan checks that the queue did not miss a changed Bellman target.
    # Its cost is measured separately; it is not a free certificate.
    residual = bellman_residual(model, q)
    residual_checks = len(rows)
    certified = residual <= tolerance and _action_gaps_certify(model, q, residual)
    if certified:
        return RepairResult(q, greedy_policy(model, q), residual, True, False,
                            local_backups, local_target_evaluations,
                            residual_checks, None)

    fallback = value_iteration(model, tolerance=tolerance,
                               max_sweeps=fallback_max_sweeps, initial_q=q)
    residual_checks += fallback.residual_checks
    return RepairResult(fallback.q, fallback.policy, fallback.residual,
                        _action_gaps_certify(model, fallback.q, fallback.residual),
                        True, local_backups, local_target_evaluations,
                        residual_checks, fallback)
