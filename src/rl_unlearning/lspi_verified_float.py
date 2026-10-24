"""Binary64 LSPI replay with an exact a posteriori policy certificate.

The fast path solves each retained LSTD system in binary64.  It then treats the
computed inverse and weights as exact dyadic rationals and verifies a bound on
the true rational solution.  If ``R`` is the computed inverse candidate and
``eta = ||I - R A||_inf < 1``, then

    ||x_hat - A^{-1} b||_inf
        <= ||R (b - A x_hat)||_inf / (1 - eta).

Every greedy comparison must clear the induced feature-score error.  Otherwise
the complete request falls back to exact retained-only cached replay.  Thus an
accepted fast path returns the exact policy sequence of the declared rational
LSPI learner, while deliberately omitting an exact matrix/weight trace.

This is a numerical research prototype.  Exact verification still performs
O(d^3) rational matrix products, and no wall-time or novelty claim follows from
the certificate alone.
"""

from dataclasses import dataclass
from fractions import Fraction as F
import math

from .lspi_baselines import LSPICachedState, _change_successor
from .lspi_control import LSPIWork, _fraction, _new_work, _outer_add


@dataclass(frozen=True)
class FloatEvaluationCertificate:
    inverse_defect: F
    residual_norm: F
    weight_error_bound: F
    minimum_clearance: F


@dataclass(frozen=True)
class VerifiedFloatResult:
    policy: tuple
    policy_sequence: tuple
    status: str
    repeat_start: int | None
    repeat_length: int | None
    certified: bool
    used_fallback: bool
    certificates: tuple
    work: object


def _dyadic(value):
    if not math.isfinite(value):
        raise ArithmeticError("non-finite binary64 value")
    return F.from_float(value)


def _float_inverse_and_solve(matrix, rhs):
    """Partial-pivoted Gauss-Jordan solve returning binary64 inverse and x."""
    n = len(rhs)
    augmented = [
        [float(value) for value in row]
        + [1.0 if i == j else 0.0 for j in range(n)]
        + [float(rhs[i])]
        for i, row in enumerate(matrix)
    ]
    if any(not math.isfinite(value) for row in augmented for value in row):
        raise ArithmeticError("matrix is outside binary64 range")
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if augmented[pivot][col] == 0.0:
            raise ArithmeticError("binary64 elimination found a zero pivot")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        augmented[col] = [value / scale for value in augmented[col]]
        if any(not math.isfinite(value) for value in augmented[col]):
            raise ArithmeticError("non-finite normalized pivot row")
        for row in range(n):
            if row == col:
                continue
            scale = augmented[row][col]
            augmented[row] = [
                value - scale * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[col])
            ]
            if any(not math.isfinite(value) for value in augmented[row]):
                raise ArithmeticError("non-finite elimination row")
    inverse = tuple(tuple(row[n:2*n]) for row in augmented)
    weights = tuple(row[-1] for row in augmented)
    return inverse, weights


def _verified_weight_bound(matrix, rhs, inverse, weights):
    """Verify the inverse defect and return an exact infinity-norm error bound."""
    n = len(rhs)
    rational_inverse = tuple(tuple(_dyadic(value) for value in row) for row in inverse)
    rational_weights = tuple(_dyadic(value) for value in weights)
    defect_rows = []
    for i in range(n):
        row = []
        for j in range(n):
            product = sum(
                (rational_inverse[i][k] * matrix[k][j] for k in range(n)), F(0)
            )
            row.append(F(i == j) - product)
        defect_rows.append(tuple(row))
    eta = max(sum(map(abs, row), F(0)) for row in defect_rows)
    if eta >= 1:
        raise ArithmeticError("approximate inverse defect cannot certify nonsingularity")
    residual = tuple(
        rhs[i] - sum((matrix[i][j] * rational_weights[j] for j in range(n)), F(0))
        for i in range(n)
    )
    residual_norm = max(map(abs, residual), default=F(0))
    preconditioned = tuple(
        sum((rational_inverse[i][j] * residual[j] for j in range(n)), F(0))
        for i in range(n)
    )
    error = max(map(abs, preconditioned), default=F(0)) / (1 - eta)
    return rational_weights, eta, residual_norm, error


def _certified_greedy(problem, weights, error_bound, work):
    """Return the greedy policy only when every nonidentical comparison clears."""
    improved = []
    minimum_clearance = None
    for state in problem.states:
        rows = tuple(problem.features[(state, action)] for action in problem.actions)
        scores = tuple(
            sum((feature * weight for feature, weight in zip(row, weights)), F(0))
            for row in rows
        )
        work["action_scores"] += len(problem.actions)
        chosen_index = max(range(len(scores)), key=scores.__getitem__)
        chosen = rows[chosen_index]
        for index, row in enumerate(rows):
            if index == chosen_index or row == chosen:
                continue
            difference = tuple(x - y for x, y in zip(chosen, row))
            clearance = scores[chosen_index] - scores[index] - error_bound * sum(
                map(abs, difference), F(0)
            )
            if clearance <= 0:
                raise ArithmeticError("greedy action is not numerically certified")
            if minimum_clearance is None or clearance < minimum_clearance:
                minimum_clearance = clearance
        improved.append(problem.actions[chosen_index])
    return tuple(improved), F(0) if minimum_clearance is None else minimum_clearance


class LSPIVerifiedFloatState(LSPICachedState):
    """Cached retained statistics with verified binary64 replay and exact fallback."""

    def _assemble(self, covariance, successors, policy, work):
        matrix = [list(row) for row in covariance]
        choices = dict(zip(self.problem.states, policy))
        for state, group in successors.items():
            work["successor_entries_checked"] += 1
            _outer_add(
                matrix,
                group,
                self.problem.features[(state, choices[state])],
                -self.problem.gamma,
            )
            work["target_rank_one_updates"] += 1
        return tuple(tuple(row) for row in matrix)

    def _verified_replay(self, covariance, rhs, successors, samples_read=0):
        problem = self.problem
        work = _new_work(samples_read)
        policy = tuple(problem.initial_policy[state] for state in problem.states)
        seen = {policy: 0}
        repeat_start = repeat_length = None
        sequence, certificates, cache = [], [], {}
        try:
            for index in range(problem.iterations):
                if policy in cache:
                    improved, certificate = cache[policy]
                else:
                    matrix = self._assemble(covariance, successors, policy, work)
                    work["linear_solves"] += 1
                    inverse, approximate = _float_inverse_and_solve(matrix, rhs)
                    weights, eta, residual, error = _verified_weight_bound(
                        matrix, tuple(rhs), inverse, approximate
                    )
                    improved, clearance = _certified_greedy(
                        problem, weights, error, work
                    )
                    certificate = FloatEvaluationCertificate(
                        eta, residual, error, clearance
                    )
                    cache[policy] = improved, certificate
                sequence.append(improved)
                certificates.append(certificate)
                if improved in seen and repeat_start is None:
                    repeat_start = seen[improved]
                    repeat_length = index + 1 - repeat_start
                seen.setdefault(improved, index + 1)
                policy = improved
        except (ArithmeticError, OverflowError, ValueError):
            exact = self._reconstruct(covariance, rhs, successors, samples_read)
            policy_sequence = tuple(iterate.policy_out for iterate in exact.iterates)
            return VerifiedFloatResult(
                exact.policy, policy_sequence, exact.status,
                exact.repeat_start, exact.repeat_length, False, True,
                tuple(certificates), exact.work,
            )
        status = (
            "fixed_point" if repeat_length == 1
            else "cycle" if repeat_length is not None
            else "budget_exhausted"
        )
        return VerifiedFloatResult(
            policy, tuple(sequence), status, repeat_start, repeat_length,
            True, False, tuple(certificates), LSPIWork(**work),
        )

    def replay(self):
        return self._verified_replay(self._covariance, self._rhs, self._successors)

    def delete_and_replay(self, trajectory_ids):
        requested = tuple(trajectory_ids)
        if len(set(requested)) != len(requested):
            raise ValueError("duplicate deletion ids")
        unknown = {identifier for identifier in requested if identifier not in self._data}
        if unknown:
            raise KeyError(f"unknown or already deleted ids: {sorted(unknown)}")
        covariance = [list(row) for row in self._covariance]
        rhs = list(self._rhs)
        successors = {state: list(group) for state, group in self._successors.items()}
        samples_read = 0
        for identifier in requested:
            for transition in self._data[identifier].transitions:
                samples_read += 1
                phi = self.problem.features[(transition.state, transition.action)]
                _outer_add(covariance, phi, phi, F(-1))
                reward = _fraction(transition.reward)
                for i, value in enumerate(phi):
                    rhs[i] -= value * reward
                _change_successor(successors, transition.next_state, phi, -1)
        result = self._verified_replay(covariance, rhs, successors, samples_read)
        if result.status != "singular":
            self._covariance, self._rhs, self._successors = covariance, rhs, successors
            for identifier in requested:
                del self._data[identifier]
        return result
