"""Independent rational policy-enumeration oracle for tiny logged MDPs.

This deliberately expensive solver is for validation, never a timing baseline.
Decimal strings of logged floats define the exact rational rewards.
"""

from fractions import Fraction as F
from itertools import product

from .data import Trajectory


def _solve(matrix, rhs):
    n = len(rhs)
    augmented = [list(row) + [value] for row, value in zip(matrix, rhs)]
    for col in range(n):
        pivot = next(i for i in range(col, n) if augmented[i][col])
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        augmented[col] = [x / scale for x in augmented[col]]
        for row in range(n):
            if row != col:
                scale = augmented[row][col]
                augmented[row] = [x - scale * y for x, y in
                                  zip(augmented[row], augmented[col])]
    return [row[-1] for row in augmented]


def exact_reference(states, actions, gamma, trajectories: tuple[Trajectory, ...],
                    *, max_policies=4096):
    """Return exact Q and tie-ordered greedy actions, independently aggregating logs."""
    states, actions = tuple(states), tuple(actions)
    if not states or not actions or len(set(states)) != len(states) or len(set(actions)) != len(actions):
        raise ValueError("nonempty unique states and actions required")
    discount = F(str(gamma))
    if not 0 <= discount < 1:
        raise ValueError("gamma must be in [0, 1)")
    if len(actions) ** len(states) > max_policies:
        raise ValueError("exact reference is limited to tiny MDPs")
    samples = {(s, a): [] for s in states for a in actions}
    for trajectory in trajectories:
        for transition in trajectory.transitions:
            if transition.next_state is not None and transition.next_state not in states:
                raise ValueError("unknown successor")
            samples[(transition.state, transition.action)].append(transition)
    rewards, transitions = {}, {}
    for row, items in samples.items():
        rewards[row] = sum((F(str(t.reward)) for t in items), F(0)) / len(items) if items else F(0)
        transitions[row] = {s: F(sum(t.next_state == s for t in items), len(items))
                            for s in states} if items else {}
    best = None
    for choices in product(actions, repeat=len(states)):
        matrix = [[F(i == j) - discount * transitions[(s, choices[i])].get(t, F(0))
                   for j, t in enumerate(states)] for i, s in enumerate(states)]
        values = _solve(matrix, [rewards[(s, a)] for s, a in zip(states, choices)])
        best = values if best is None else [max(x, y) for x, y in zip(best, values)]
    q = {row: rewards[row] + discount * sum(
        (prob * best[states.index(s)] for s, prob in transitions[row].items()), F(0))
         for row in samples}
    policy = {s: max(actions, key=lambda a: q[(s, a)]) for s in states}
    return q, policy
