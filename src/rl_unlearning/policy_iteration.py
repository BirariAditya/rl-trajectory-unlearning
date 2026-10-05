"""Exact policy iteration on all states or a block with fixed boundary values.

Rational arithmetic is a correctness-first baseline; dense elimination is costly.
"""

from dataclasses import dataclass
from fractions import Fraction as F

from .reference import _solve


@dataclass(frozen=True)
class ExactPlan:
    policy: dict
    values: dict
    q: dict
    iterations: int
    action_evaluations: int
    linear_system_sizes: tuple
    graph_rows_scanned: int = 0
    graph_successors_scanned: int = 0
    component_sizes: tuple = ()


def policy_iteration(states, actions, gamma, row_lookup, *, initial_policy=None,
                     boundary_values=None, max_iterations=10_000):
    """Solve a block; row_lookup returns rational reward and successor probabilities.

    Boundary states must have fixed values justified by the caller. The solver
    makes no claim that an arbitrary frozen boundary is valid after deletion.
    """
    states, actions = tuple(states), tuple(actions)
    gamma = F(str(gamma))
    if not actions or len(set(actions)) != len(actions) or len(set(states)) != len(states):
        raise ValueError("unique states and nonempty unique actions required")
    if not 0 <= gamma < 1 or max_iterations < 1:
        raise ValueError("invalid discount or iteration budget")
    if not states:
        return ExactPlan({}, {}, {}, 0, 0, ())
    boundary = {} if boundary_values is None else boundary_values
    indices = {s: i for i, s in enumerate(states)}
    policy = {s: initial_policy[s] if initial_policy is not None else actions[0] for s in states}
    if any(a not in actions for a in policy.values()):
        raise ValueError("invalid initial policy")
    action_evaluations = 0
    sizes = []
    for iteration in range(1, max_iterations + 1):
        matrix, rhs = [], []
        for i, state in enumerate(states):
            reward, probabilities = row_lookup(state, policy[state])
            row = [F(i == j) for j in range(len(states))]
            target = reward
            for successor, probability in probabilities.items():
                if successor is None:
                    continue
                if successor in indices:
                    row[indices[successor]] -= gamma * probability
                else:
                    target += gamma * probability * boundary[successor]
            matrix.append(row)
            rhs.append(target)
        values = dict(zip(states, _solve(matrix, rhs)))
        sizes.append(len(states))
        q = {}
        for s in states:
            for a in actions:
                reward, probabilities = row_lookup(s, a)
                q[(s, a)] = reward + gamma * sum(
                    (p * (values[t] if t in indices else boundary[t])
                     for t, p in probabilities.items() if t is not None), F(0))
                action_evaluations += 1
        improved = {s: max(actions, key=lambda a: q[(s, a)]) for s in states}
        if improved == policy:
            return ExactPlan(policy, values, q, iteration, action_evaluations, tuple(sizes))
        policy = improved
    raise RuntimeError("exact policy iteration exceeded its iteration budget")
