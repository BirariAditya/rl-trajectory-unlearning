"""Conventional exact inverse-maintenance baseline for finite-schedule LSPI.

This is a comparator, not an unlearning novelty claim. It caches the inverse of
each policy-evaluation matrix encountered in the *previous successful retained
dataset*. After trajectory deletion, Sherman--Morrison rank-one downdates reuse
those inverses where possible. New policies, or an intermediate singular
downdate, use one fresh exact inverse of the final retained matrix.

The declared LSPI schedule always restarts at its data-independent initial
policy and runs all K iterations. Repeated policies within one request reuse an
evaluation but still emit every trace entry and score every state/action.
Candidate caches are discarded after a failed (singular) deletion request;
the last successful cache remains intact. Successful caches retain only policies
encountered in the latest successful schedule.
"""

from fractions import Fraction as F

from .lspi_baselines import LSPICachedState, _change_successor
from .lspi_control import (
    LSPIIterate,
    LSPIResult,
    LSPIWork,
    SingularLSTDMatrix,
    _fraction,
    _new_work,
    _outer_add,
)


def inverse_matrix(matrix):
    """Invert one square matrix by exact Gauss--Jordan elimination on [A | I].

    This is a single multi-right-hand-side elimination, not one solve per
    identity column. A singular square matrix raises SingularLSTDMatrix.
    Non-square or empty input raises ValueError. Float inputs, if supplied,
    follow the repository's decimal-string rational convention.
    """
    rows = tuple(tuple(_fraction(value) for value in row) for row in matrix)
    n = len(rows)
    if not n or any(len(row) != n for row in rows):
        raise ValueError("matrix must be nonempty and square")
    augmented = [
        list(row) + [F(i == j) for j in range(n)]
        for i, row in enumerate(rows)
    ]
    for col in range(n):
        pivot = next((i for i in range(col, n) if augmented[i][col]), None)
        if pivot is None:
            raise SingularLSTDMatrix("LSTD matrix is singular")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        augmented[col] = [value / scale for value in augmented[col]]
        for row in range(n):
            if row != col:
                scale = augmented[row][col]
                augmented[row] = [
                    value - scale * pivot_value
                    for value, pivot_value in zip(augmented[row], augmented[col])
                ]
    return tuple(tuple(row[n:]) for row in augmented)


def _matvec(matrix, vector):
    return tuple(sum((value * vector[j] for j, value in enumerate(row)), F(0))
                 for row in matrix)


def _downdate_inverse(inverse, left, right):
    """Return inverse(A - left*right^T), or None at a singular intermediate."""
    h_left = _matvec(inverse, left)
    right_h = tuple(
        sum((right[i] * inverse[i][j] for i in range(len(right))), F(0))
        for j in range(len(right))
    )
    denominator = F(1) - sum(
        (right[i] * h_left[i] for i in range(len(right))), F(0)
    )
    if denominator == 0:
        return None
    return tuple(
        tuple(inverse[i][j] + h_left[i] * right_h[j] / denominator
              for j in range(len(right)))
        for i in range(len(right))
    )


class LSPIInverseState(LSPICachedState):
    """Cached C,b,G statistics plus policy-specific exact inverse reuse.

    Construction reads and aggregates the full log, as in LSPICachedState.
    Call replay() once to build the original-data inverse cache and charge
    that preparation explicitly. If a caller deletes without replaying first,
    the first request builds fresh inverses rather than hiding a preparation
    run. last_inverse_work describes the most recent call; it counts attempted
    fresh inversions and successful Sherman--Morrison downdates separately.
    It does not model rational bit growth, Python allocations, or memory.
    """

    reuse_greedy_outputs = False

    def __init__(self, problem, trajectories):
        super().__init__(problem, trajectories)
        self._inverse_cache = {}
        self.last_inverse_work = {}

    @property
    def cached_policies(self):
        return frozenset(self._inverse_cache)

    def _assemble(self, covariance, successors, policy, work):
        """Build A(policy) from candidate C and G for a previously unseen policy."""
        matrix = [list(row) for row in covariance]
        choices = dict(zip(self.problem.states, policy))
        for state, group in successors.items():
            work["successor_entries_checked"] += 1
            _outer_add(
                matrix, group, self.problem.features[(state, choices[state])],
                -self.problem.gamma,
            )
            work["target_rank_one_updates"] += 1
        return tuple(tuple(row) for row in matrix)

    def _updated_cached_matrix(self, old_matrix, old_inverse, deleted, policy, work,
                               inverse_work):
        """Downdate a cached policy matrix; rebuild if an intermediate is singular."""
        matrix = [list(row) for row in old_matrix]
        inverse = old_inverse
        choices = dict(zip(self.problem.states, policy))
        intermediate_singular = False
        for transition in deleted:
            phi = self.problem.features[(transition.state, transition.action)]
            successor = (
                (F(0),) * self.problem.dimension if transition.next_state is None
                else self.problem.features[
                    (transition.next_state, choices[transition.next_state])
                ]
            )
            right = tuple(
                value - self.problem.gamma * target
                for value, target in zip(phi, successor)
            )
            _outer_add(matrix, phi, right, F(-1))
            if not intermediate_singular:
                inverse_work["rank_one_attempts"] += 1
                updated = _downdate_inverse(inverse, phi, right)
                if updated is None:
                    intermediate_singular = True
                    inverse_work["intermediate_singular_fallbacks"] += 1
                else:
                    inverse = updated
                    inverse_work["rank_one_downdates"] += 1
        final_matrix = tuple(tuple(row) for row in matrix)
        if intermediate_singular:
            inverse_work["inverse_builds"] += 1
            work["linear_solves"] += 1
            try:
                inverse = inverse_matrix(final_matrix)
            except SingularLSTDMatrix:
                inverse = None
        return final_matrix, inverse

    def _run_inverse(self, covariance, rhs, successors, deleted=()):
        """Evaluate the full fixed LSPI schedule against candidate statistics."""
        problem = self.problem
        work = _new_work(len(deleted))
        inverse_work = {
            "samples_read": len(deleted),
            "successor_entries_checked": 0,
            "target_terms": 0,
            "inverse_builds": 0,
            "rank_one_attempts": 0,
            "rank_one_downdates": 0,
            "intermediate_singular_fallbacks": 0,
            "within_request_cache_hits": 0,
            "action_scores": 0,
        }
        rhs = tuple(rhs)
        policy = tuple(problem.initial_policy[state] for state in problem.states)
        seen = {policy: 0}
        repeat_start = repeat_length = None
        trace = []
        current_cache = {}
        greedy_cache = {}

        for index in range(problem.iterations):
            if policy in current_cache:
                matrix, inverse, weights = current_cache[policy]
                inverse_work["within_request_cache_hits"] += 1
            else:
                try:
                    if policy in self._inverse_cache:
                        old_matrix, old_inverse = self._inverse_cache[policy]
                        matrix, inverse = self._updated_cached_matrix(
                            old_matrix, old_inverse, deleted, policy, work,
                            inverse_work,
                        )
                    else:
                        matrix = self._assemble(covariance, successors, policy, work)
                        inverse_work["inverse_builds"] += 1
                        work["linear_solves"] += 1
                        inverse = inverse_matrix(matrix)
                    if inverse is None:
                        raise SingularLSTDMatrix("LSTD matrix is singular")
                except SingularLSTDMatrix:
                    trace.append(LSPIIterate(index, policy, matrix, rhs, None, None))
                    inverse_work["successor_entries_checked"] = work[
                        "successor_entries_checked"
                    ]
                    inverse_work["target_terms"] = work["target_rank_one_updates"]
                    inverse_work["action_scores"] = work["action_scores"]
                    self.last_inverse_work = inverse_work
                    return (
                        LSPIResult(
                            tuple(trace), policy, None, "singular",
                            repeat_start, repeat_length, LSPIWork(**work),
                        ),
                        None,
                    )
                weights = _matvec(inverse, rhs)
                current_cache[policy] = (matrix, inverse, weights)

            if self.reuse_greedy_outputs and policy in greedy_cache:
                improved = greedy_cache[policy]
            else:
                improved = []
                for state in problem.states:
                    scores = [
                        sum(
                            (feature * weight for feature, weight in
                             zip(problem.features[(state, action)], weights)),
                            F(0),
                        )
                        for action in problem.actions
                    ]
                    work["action_scores"] += len(problem.actions)
                    improved.append(
                        problem.actions[
                            max(range(len(scores)), key=scores.__getitem__)
                        ]
                    )
                improved = tuple(improved)
                if self.reuse_greedy_outputs:
                    greedy_cache[policy] = improved
            trace.append(LSPIIterate(index, policy, matrix, rhs, weights, improved))
            if improved in seen and repeat_start is None:
                repeat_start = seen[improved]
                repeat_length = index + 1 - repeat_start
            seen.setdefault(improved, index + 1)
            policy = improved

        status = (
            "fixed_point" if repeat_length == 1
            else "cycle" if repeat_length is not None
            else "budget_exhausted"
        )
        inverse_work["successor_entries_checked"] = work[
            "successor_entries_checked"
        ]
        inverse_work["target_terms"] = work["target_rank_one_updates"]
        inverse_work["action_scores"] = work["action_scores"]
        self.last_inverse_work = inverse_work
        result = LSPIResult(
            tuple(trace), policy, trace[-1].weights, status,
            repeat_start, repeat_length, LSPIWork(**work),
        )
        return result, {
            candidate_policy: (matrix, inverse)
            for candidate_policy, (matrix, inverse, _) in current_cache.items()
        }

    def replay(self):
        """Run the current retained learner and cache its encountered inverses."""
        result, candidate_cache = self._run_inverse(
            self._covariance, self._rhs, self._successors
        )
        if candidate_cache is not None:
            self._inverse_cache = candidate_cache
        return result


    def delete_and_replay(self, trajectory_ids):
        """Try deletion atomically; cache inverses only after a full valid schedule."""
        requested = tuple(trajectory_ids)
        if len(set(requested)) != len(requested):
            raise ValueError("duplicate deletion ids")
        unknown = {identifier for identifier in requested
                   if identifier not in self._data}
        if unknown:
            raise KeyError(f"unknown or already deleted ids: {sorted(unknown)}")

        covariance = [list(row) for row in self._covariance]
        rhs = list(self._rhs)
        successors = {
            state: list(group) for state, group in self._successors.items()
        }
        deleted = []
        for identifier in requested:
            for transition in self._data[identifier].transitions:
                deleted.append(transition)
                phi = self.problem.features[
                    (transition.state, transition.action)
                ]
                _outer_add(covariance, phi, phi, F(-1))
                reward = _fraction(transition.reward)
                for i, value in enumerate(phi):
                    rhs[i] -= value * reward
                _change_successor(
                    successors, transition.next_state, phi, -1
                )

        result, candidate_cache = self._run_inverse(
            covariance, rhs, successors, tuple(deleted)
        )
        if candidate_cache is not None:
            self._covariance, self._rhs, self._successors = (
                covariance, rhs, successors
            )
            self._inverse_cache = candidate_cache
            for identifier in requested:
                del self._data[identifier]
        return result


class LSPIInversePolicyState(LSPIInverseState):
    """Also reuse greedy outputs for repeated policies within each request.

    This stronger conventional control avoids rescoring actions for evaluations
    already seen on the same retained data. It preserves the full finite trace.
    """

    reuse_greedy_outputs = True
