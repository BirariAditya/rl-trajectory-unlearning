"""Sufficient policy-sequence stability certificate for finite exact LSPI.

Output is ONLY the policy sequence/status, not retained-only weights or private
internal state. Old anchor inverses/weights remain internal after certification.
Ordinary perturbation bounds and margin tests are not claimed as novel.
"""

from dataclasses import dataclass
from fractions import Fraction as F

from .lspi_control import _data, _fraction, SingularLSTDMatrix
from .lspi_baselines import LSPIMemoizedState
from .lspi_inverse import inverse_matrix


def _dot(a, b):
    return sum((x * y for x, y in zip(a, b)), F(0))


def _matvec(matrix, vector):
    return tuple(_dot(row, vector) for row in matrix)


@dataclass(frozen=True)
class PolicyCertificateResult:
    policy: tuple
    policy_sequence: tuple
    status: str
    repeat_start: int | None
    repeat_length: int | None
    certified: bool
    used_fallback: bool
    work: dict


@dataclass(frozen=True)
class _Anchor:
    policy_in: tuple
    inverse: tuple
    weights: tuple
    radius: F | None


def _result(trace, certified, fallback, work):
    return PolicyCertificateResult(trace.policy, tuple(t.policy_out for t in trace.iterates),
                                   trace.status, trace.repeat_start, trace.repeat_length,
                                   certified, fallback, work)


class LSPIPolicyCertificateState:
    """Maintain cumulative deletion bounds relative to the last exact anchor.

    For each unique anchor input policy, alpha bounds ||H E||_inf and h equals
    H(f-Ew) for ALL deletions since that anchor. If alpha<1 then retained A is
    nonsingular and ||w_retained-w||_inf <= ||h||_inf/(1-alpha). A precomputed
    global action-margin radius certifies the same greedy output at every state.
    Induction through the finite schedule gives the same policy sequence.

    Inconclusive requests rebuild retained LSPI and all anchor inverses. Singular
    requests leave data, bounds and anchors unchanged. No clipping or numerical
    tolerance is used. This version intentionally uses conservative norm bounds.
    """

    def __init__(self, problem, trajectories):
        self.problem = problem
        self._indices = {s: i for i, s in enumerate(problem.states)}
        self._data = _data(problem, trajectories)
        trace, anchors, work = self._build(self._data.values())
        if trace.status == "singular":
            raise SingularLSTDMatrix("certificate requires a nonsingular initial schedule")
        self._trace, self._anchors = trace, anchors
        self._alpha = tuple(F(0) for _ in anchors)
        self._h = tuple((F(0),) * problem.dimension for _ in anchors)
        self.preparation_work = work
        self.initial_result = _result(trace, False, False, work)

    @property
    def trajectory_ids(self):
        return frozenset(self._data)

    def active_trajectories(self):
        return tuple(self._data.values())

    def _build(self, trajectories):
        state = LSPIMemoizedState(self.problem, trajectories)
        trace = state.replay()
        work = dict(inverse_builds=0, linear_solves=trace.work.linear_solves,
                    action_scores=trace.work.action_scores, margin_comparisons=0,
                    preparation_samples=state.preparation_samples)
        if trace.status == "singular":
            return trace, (), work
        anchors, seen = [], set()
        for step in trace.iterates:
            if step.policy_in in seen:
                continue
            seen.add(step.policy_in)
            inverse = inverse_matrix(step.matrix)
            work["inverse_builds"] += 1
            radii = []
            for s, chosen in zip(self.problem.states, step.policy_out):
                for a in self.problem.actions:
                    if a == chosen:
                        continue
                    work["margin_comparisons"] += 1
                    difference = tuple(x-y for x, y in zip(self.problem.features[s, chosen],
                                                          self.problem.features[s, a]))
                    size = sum(map(abs, difference), F(0))
                    if size:
                        radii.append(_dot(difference, step.weights) / size)
            anchors.append(_Anchor(step.policy_in, inverse, step.weights, min(radii) if radii else None))
        return trace, tuple(anchors), work

    def delete_and_replay(self, trajectory_ids):
        requested = tuple(trajectory_ids)
        if len(set(requested)) != len(requested):
            raise ValueError("duplicate deletion ids")
        unknown = {key for key in requested if key not in self._data}
        if unknown:
            raise KeyError(f"unknown or already deleted ids: {sorted(unknown)}")
        problem = self.problem
        alpha, h = list(self._alpha), [list(row) for row in self._h]
        indices = self._indices
        work = dict(anchors_checked=0, samples_read=0, matrix_vector_products=0,
                    inverse_builds=0, linear_solves=0, action_scores=0,
                    margin_comparisons=0, preparation_samples=0)
        for trajectory_id in requested:
            for t in self._data[trajectory_id].transitions:
                work["samples_read"] += 1
                phi = problem.features[t.state, t.action]
                for k, anchor in enumerate(self._anchors):
                    target = ((F(0),) * problem.dimension if t.next_state is None else
                              problem.features[t.next_state, anchor.policy_in[indices[t.next_state]]])
                    v = tuple(x - problem.gamma*y for x, y in zip(phi, target))
                    g = _matvec(anchor.inverse, phi)
                    work["matrix_vector_products"] += 1
                    alpha[k] += max(map(abs, g)) * sum(map(abs, v), F(0))
                    residual = _dot(v, anchor.weights) - _fraction(t.reward)
                    for j in range(problem.dimension):
                        h[k][j] += g[j] * residual
        accepted = True
        for anchor, a, residual in zip(self._anchors, alpha, h):
            work["anchors_checked"] += 1
            if a >= 1:
                accepted = False
                break
            size = max(map(abs, residual))
            # Zero residual gives exactly unchanged weights, preserving all ties.
            if size and anchor.radius is not None and size / (1-a) >= anchor.radius:
                accepted = False
                break
        if accepted:
            for key in requested:
                del self._data[key]
            self._alpha, self._h = tuple(alpha), tuple(tuple(row) for row in h)
            return _result(self._trace, True, False, work)
        removed = set(requested)
        retained = tuple(t for key, t in self._data.items() if key not in removed)
        trace, anchors, extra = self._build(retained)
        for key, value in extra.items():
            work[key] += value
        if trace.status != "singular":
            for key in requested:
                del self._data[key]
            self._trace, self._anchors = trace, anchors
            self._alpha = tuple(F(0) for _ in anchors)
            self._h = tuple((F(0),) * problem.dimension for _ in anchors)
        return _result(trace, False, True, work)
