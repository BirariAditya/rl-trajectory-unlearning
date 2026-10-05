"""Classical SCC decomposition with exact PI inside each component.

This is a graph-aware planning baseline, not a new unlearning algorithm or
a reproduction of TVI/FTVI. All positive-probability action edges are included.
"""

from collections import ChainMap
from fractions import Fraction as F

from .policy_iteration import ExactPlan, policy_iteration


def _sink_components(states, adjacency):
    """Iterative Kosaraju; return SCCs in successor-before-predecessor order."""
    reverse = {s: [] for s in states}
    for state in states:
        for successor in adjacency[state]:
            reverse[successor].append(state)
    seen, finish = set(), []
    for root in states:
        if root in seen:
            continue
        seen.add(root)
        stack = [(root, iter(adjacency[root]))]
        while stack:
            state, successors = stack[-1]
            successor = next(successors, None)
            if successor is None:
                finish.append(state)
                stack.pop()
            elif successor not in seen:
                seen.add(successor)
                stack.append((successor, iter(adjacency[successor])))
    seen, components = set(), []
    for root in reversed(finish):
        if root in seen:
            continue
        seen.add(root)
        pending, component = [root], []
        while pending:
            state = pending.pop()
            component.append(state)
            for predecessor in reverse[state]:
                if predecessor not in seen:
                    seen.add(predecessor)
                    pending.append(predecessor)
        components.append(tuple(component))
    return tuple(reversed(components))


def scc_policy_iteration(states, actions, gamma, row_lookup, *,
                         initial_policy=None, boundary_values=None,
                         max_iterations=10_000):
    """Solve sink SCCs first, using already solved values as exact boundaries.

    Rebuild the all-action support graph on every call. Graph construction and
    all per-component solves are charged to caller time. max_iterations is a
    per-component limit; returned iterations is their sum. External boundary
    validity remains the caller's responsibility, as for policy_iteration.
    """
    states, actions = tuple(states), tuple(actions)
    gamma = F(str(gamma))
    if not actions or len(set(actions)) != len(actions) or len(set(states)) != len(states):
        raise ValueError("unique states and nonempty unique actions required")
    if not 0 <= gamma < 1 or max_iterations < 1:
        raise ValueError("invalid discount or iteration budget")
    state_set = set(states)
    boundary = {} if boundary_values is None else boundary_values
    adjacency, rows, entries = {}, 0, 0
    for state in states:
        successors = {}
        for action in actions:
            _, probabilities = row_lookup(state, action)
            rows += 1
            for successor, probability in probabilities.items():
                entries += 1
                if successor is None or probability == 0:
                    continue
                if successor in state_set:
                    successors[successor] = None
                elif successor not in boundary:
                    raise ValueError("missing external boundary value")
        adjacency[state] = tuple(successors)
    components = _sink_components(states, adjacency)
    policy, values, q = {}, {}, {}
    iterations = evaluations = 0
    sizes = []
    for component in components:
        plan = policy_iteration(component, actions, gamma, row_lookup,
                                initial_policy=initial_policy,
                                boundary_values=ChainMap(values, boundary),
                                max_iterations=max_iterations)
        policy.update(plan.policy)
        values.update(plan.values)
        q.update(plan.q)
        iterations += plan.iterations
        evaluations += plan.action_evaluations
        sizes.extend(plan.linear_system_sizes)
    return ExactPlan(policy, values, q, iterations, evaluations, tuple(sizes),
                     rows, entries, tuple(map(len, components)))
