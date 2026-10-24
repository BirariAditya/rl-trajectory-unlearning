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
import sys

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

    def _weight_certificate(self, matrix, rhs, inverse, approximate):
        return _verified_weight_bound(matrix, rhs, inverse, approximate)

    def _greedy_certificate(self, weights, error, work):
        return _certified_greedy(self.problem, weights, error, work)

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
                    weights, eta, residual, error = self._weight_certificate(
                        matrix, tuple(rhs), inverse, approximate
                    )
                    improved, clearance = self._greedy_certificate(weights, error, work)
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


_NEGATIVE_INFINITY = float("-inf")
_POSITIVE_INFINITY = float("inf")


def _down(value):
    return math.nextafter(value, _NEGATIVE_INFINITY)


def _up(value):
    return math.nextafter(value, _POSITIVE_INFINITY)


def _fraction_interval(value):
    """Smallest adjacent-binary64 interval containing an exact Fraction."""
    value = _fraction(value)
    nearest = float(value)
    if not math.isfinite(nearest):
        raise ArithmeticError("exact value is outside binary64 range")
    represented = F.from_float(nearest)
    if represented < value:
        return nearest, _up(nearest)
    if represented > value:
        return _down(nearest), nearest
    return nearest, nearest


def _interval_add(left, right):
    return _down(left[0] + right[0]), _up(left[1] + right[1])


def _interval_subtract(left, right):
    return _down(left[0] - right[1]), _up(left[1] - right[0])


def _interval_multiply(left, right):
    products = (
        left[0] * right[0], left[0] * right[1],
        left[1] * right[0], left[1] * right[1],
    )
    if any(not math.isfinite(value) for value in products):
        raise ArithmeticError("non-finite interval product")
    return _down(min(products)), _up(max(products))


def _interval_sum(terms):
    total = (0.0, 0.0)
    for term in terms:
        total = _interval_add(total, term)
    return total


def _interval_abs_upper(interval):
    return max(abs(interval[0]), abs(interval[1]))


def _interval_weight_bound(matrix, rhs, inverse, weights):
    """Outward-rounded version of the approximate-inverse theorem."""
    if sys.float_info.radix != 2 or sys.float_info.mant_dig != 53:
        raise ArithmeticError("interval verifier requires IEEE-754 binary64")
    n = len(rhs)
    matrix_intervals = tuple(
        tuple(_fraction_interval(value) for value in row) for row in matrix
    )
    rhs_intervals = tuple(_fraction_interval(value) for value in rhs)
    if any(not math.isfinite(value) for row in inverse for value in row):
        raise ArithmeticError("non-finite approximate inverse")
    inverse_intervals = tuple(tuple((value, value) for value in row) for row in inverse)
    weight_intervals = tuple((value, value) for value in weights)

    eta = 0.0
    for i in range(n):
        row_sum = 0.0
        for j in range(n):
            product = _interval_sum(
                _interval_multiply(inverse_intervals[i][k], matrix_intervals[k][j])
                for k in range(n)
            )
            defect = _interval_subtract((float(i == j), float(i == j)), product)
            row_sum = _up(row_sum + _interval_abs_upper(defect))
        eta = max(eta, row_sum)
    if not math.isfinite(eta) or eta >= 1.0:
        raise ArithmeticError("interval inverse defect cannot certify nonsingularity")

    residual = []
    residual_norm = 0.0
    for i in range(n):
        product = _interval_sum(
            _interval_multiply(matrix_intervals[i][j], weight_intervals[j])
            for j in range(n)
        )
        item = _interval_subtract(rhs_intervals[i], product)
        residual.append(item)
        residual_norm = max(residual_norm, _interval_abs_upper(item))
    preconditioned_norm = 0.0
    for i in range(n):
        item = _interval_sum(
            _interval_multiply(inverse_intervals[i][j], residual[j])
            for j in range(n)
        )
        preconditioned_norm = max(preconditioned_norm, _interval_abs_upper(item))
    denominator = _down(1.0 - eta)
    if denominator <= 0.0:
        raise ArithmeticError("interval inverse defect has no positive denominator")
    error = _up(preconditioned_norm / denominator)
    if not math.isfinite(error):
        raise ArithmeticError("non-finite interval error bound")
    return tuple(weights), eta, residual_norm, error


def _interval_certified_greedy(problem, weights, error_bound, work):
    improved = []
    minimum_clearance = _POSITIVE_INFINITY
    for state in problem.states:
        rows = tuple(problem.features[(state, action)] for action in problem.actions)
        approximate_scores = tuple(
            sum(float(feature) * weight for feature, weight in zip(row, weights))
            for row in rows
        )
        if any(not math.isfinite(score) for score in approximate_scores):
            raise ArithmeticError("non-finite approximate action score")
        work["action_scores"] += len(problem.actions)
        chosen_index = max(range(len(rows)), key=approximate_scores.__getitem__)
        chosen = rows[chosen_index]
        for index, row in enumerate(rows):
            if index == chosen_index or row == chosen:
                continue
            difference = tuple(x - y for x, y in zip(chosen, row))
            gap = _interval_sum(
                _interval_multiply(_fraction_interval(value), (weight, weight))
                for value, weight in zip(difference, weights)
            )
            norm_upper = _fraction_interval(sum(map(abs, difference), F(0)))[1]
            uncertainty = _up(norm_upper * error_bound)
            clearance = _down(gap[0] - uncertainty)
            if clearance <= 0.0:
                raise ArithmeticError("greedy action is not interval certified")
            minimum_clearance = min(minimum_clearance, clearance)
        improved.append(problem.actions[chosen_index])
    if minimum_clearance == _POSITIVE_INFINITY:
        minimum_clearance = 0.0
    return tuple(improved), minimum_clearance


class LSPIIntervalVerifiedFloatState(LSPIVerifiedFloatState):
    """Use outward-rounded binary64 intervals for the verification products."""

    def _weight_certificate(self, matrix, rhs, inverse, approximate):
        return _interval_weight_bound(matrix, rhs, inverse, approximate)

    def _greedy_certificate(self, weights, error, work):
        return _interval_certified_greedy(self.problem, weights, error, work)
