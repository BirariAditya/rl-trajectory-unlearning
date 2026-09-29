"""Ordinary prioritized sweeping with an explicit global convergence check."""

import heapq
from dataclasses import dataclass

from .planner import bellman_residual, greedy_policy


@dataclass(frozen=True)
class SweepResult:
    q: dict
    policy: dict
    residual: float
    converged: bool
    backups: int


def prioritized_sweeping(model, initial_q, changed_rows, *, tolerance=1e-10,
                         max_backups=100_000):
    if tolerance <= 0 or max_backups < 0:
        raise ValueError("positive tolerance and nonnegative budget required")
    q = dict(initial_q)
    predecessors = model.predecessor_rows()
    queue = []
    serial = 0

    def schedule(row):
        nonlocal serial
        error = abs(model.backup(q, *row) - q[row])
        if error > tolerance:
            heapq.heappush(queue, (-error, serial, row))
            serial += 1

    for row in sorted(changed_rows):
        schedule(row)
    backups = 0
    while True:
        while queue and backups < max_backups:
            _, _, row = heapq.heappop(queue)
            target = model.backup(q, *row)
            if abs(target - q[row]) <= tolerance:
                continue
            state = row[0]
            before = max(q[(state, a)] for a in model.actions)
            q[row] = target
            backups += 1
            after = max(q[(state, a)] for a in model.actions)
            if after != before:
                for predecessor in sorted(predecessors[state]):
                    schedule(predecessor)
        residual = bellman_residual(model, q)
        if residual <= tolerance or backups >= max_backups:
            return SweepResult(q, greedy_policy(model, q), residual,
                               residual <= tolerance, backups)
        # A stale or sub-tolerance propagation must not silently stop the solver.
        for row in q:
            schedule(row)
