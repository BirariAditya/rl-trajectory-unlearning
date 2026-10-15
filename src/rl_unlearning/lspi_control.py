"""Exact-rational validation of a fixed-budget, fixed-feature LSPI learner.

The scratch path rebuilds normal equations from retained transitions at every
iteration. The replay path subtracts deleted data at the declared initial policy
and updates policy-dependent targets through retained successor aggregates.
Both restart the SAME finite schedule; neither warm-starts at a learned policy.

This is an assembly/equivalence prototype, not a novel complexity result or a
claim about neural, online, privacy-preserving, or converged optimal control.
Each evaluation uses a fresh dense rational solve; no fast inverse is claimed.
"""

from dataclasses import dataclass
from fractions import Fraction as F
from types import MappingProxyType


def _fraction(value):
    """Interpret float inputs by their decimal spelling, not binary expansion."""
    return value if isinstance(value, F) else F(str(value))


@dataclass(frozen=True)
class LSPIProblem:
    """Declared learner; policy tuples follow states, ties follow actions.

    Features and initial_policy must be data-independent. Regularization is
    added to unnormalized sample SUMS, so it does not change with sample count.
    A positive regularizer does not guarantee an invertible off-policy matrix.
    """

    states: tuple
    actions: tuple
    features: object
    initial_policy: object
    gamma: F = F(1, 2)
    regularization: F = F(1, 2)
    iterations: int = 3

    def __post_init__(self):
        states, actions = tuple(self.states), tuple(self.actions)
        if not states or len(set(states)) != len(states) or None in states:
            raise ValueError("states must be nonempty, unique, and exclude terminal None")
        if not actions or len(set(actions)) != len(actions):
            raise ValueError("actions must be nonempty and unique")
        gamma, regularization = _fraction(self.gamma), _fraction(self.regularization)
        if not 0 <= gamma < 1 or regularization < 0:
            raise ValueError("invalid discount or regularization")
        if type(self.iterations) is not int or self.iterations < 1:
            raise ValueError("iterations must be a positive integer")
        keys = {(s, a) for s in states for a in actions}
        if set(self.features) != keys:
            raise ValueError("features must cover exactly the declared state-action pairs")
        features = {key: tuple(_fraction(x) for x in self.features[key]) for key in keys}
        dimensions = {len(vector) for vector in features.values()}
        if len(dimensions) != 1 or 0 in dimensions:
            raise ValueError("feature vectors must have one positive common dimension")
        if set(self.initial_policy) != set(states):
            raise ValueError("initial policy must cover exactly the declared states")
        initial_policy = {s: self.initial_policy[s] for s in states}
        if any(a not in actions for a in initial_policy.values()):
            raise ValueError("initial policy contains an unknown action")
        for name, value in (("states", states), ("actions", actions), ("gamma", gamma),
                            ("regularization", regularization),
                            ("features", MappingProxyType(features)),
                            ("initial_policy", MappingProxyType(initial_policy))):
            object.__setattr__(self, name, value)

    @property
    def dimension(self):
        return len(self.features[(self.states[0], self.actions[0])])


@dataclass(frozen=True)
class LSPIIterate:
    index: int
    policy_in: tuple
    matrix: tuple
    rhs: tuple
    weights: tuple | None
    policy_out: tuple | None


@dataclass(frozen=True)
class LSPIWork:
    """Selected work counts, not a complete arithmetic/time complexity model."""

    samples_read: int
    successor_entries_checked: int
    target_rank_one_updates: int
    linear_solves: int
    action_scores: int


@dataclass(frozen=True)
class LSPIResult:
    iterates: tuple[LSPIIterate, ...]
    policy: tuple
    weights: tuple | None
    status: str
    repeat_start: int | None
    repeat_length: int | None
    work: LSPIWork


@dataclass(frozen=True)
class LSPIPolicyResult:
    """Policy-only projection; no matrix/weight trace is constructed for output."""

    policy: tuple
    policy_sequence: tuple
    status: str
    repeat_start: int | None
    repeat_length: int | None
    work: LSPIWork


class SingularLSTDMatrix(ArithmeticError):
    pass


def _solve(matrix, rhs):
    """Deterministic exact Gaussian elimination; reject nonunique solutions."""
    n = len(rhs)
    augmented = [list(row) + [value] for row, value in zip(matrix, rhs)]
    for col in range(n):
        pivot = next((i for i in range(col, n) if augmented[i][col]), None)
        if pivot is None:
            raise SingularLSTDMatrix("LSTD matrix is singular")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        augmented[col] = [x / scale for x in augmented[col]]
        for row in range(n):
            if row != col:
                scale = augmented[row][col]
                augmented[row] = [x - scale * y for x, y in
                                  zip(augmented[row], augmented[col])]
    return tuple(row[-1] for row in augmented)


def _data(problem, trajectories):
    data = {}
    for trajectory in trajectories:
        if trajectory.trajectory_id in data:
            raise ValueError("duplicate trajectory id")
        for t in trajectory.transitions:
            if (t.state, t.action) not in problem.features:
                raise ValueError("unknown state or action in trajectory")
            if t.next_state is not None and t.next_state not in problem.states:
                raise ValueError("unknown successor in trajectory")
            _fraction(t.reward)  # Reject NaN, infinity, and nonnumeric rewards.
        data[trajectory.trajectory_id] = trajectory
    return data


def _initial_matrix(problem):
    return [[problem.regularization if i == j else F(0)
             for j in range(problem.dimension)] for i in range(problem.dimension)]


def _outer_add(matrix, left, right, scale=F(1)):
    for i, x in enumerate(left):
        for j, y in enumerate(right):
            matrix[i][j] += scale * x * y


def _new_work(samples_read=0):
    return dict(samples_read=samples_read, successor_entries_checked=0,
                target_rank_one_updates=0, linear_solves=0, action_scores=0)


def _run(problem, assemble, work, *, reuse_policy_evaluations=False):
    """Run all declared iterations, even after a fixed point or a policy cycle.

    A singular evaluation terminates explicitly; no pseudoinverse or altered
    ridge is silently substituted. The final weights evaluate policy_in of the
    last iterate, and the returned policy is its greedy policy_out.
    """
    policy = tuple(problem.initial_policy[s] for s in problem.states)
    seen, repeat_start, repeat_length = {policy: 0}, None, None
    trace, cache = [], {}
    for index in range(problem.iterations):
        if reuse_policy_evaluations and policy in cache:
            matrix, rhs, weights = cache[policy]
        else:
            matrix, rhs = assemble(policy)
            matrix, rhs = tuple(tuple(row) for row in matrix), tuple(rhs)
            work["linear_solves"] += 1
            try:
                weights = _solve(matrix, rhs)
            except SingularLSTDMatrix:
                trace.append(LSPIIterate(index, policy, matrix, rhs, None, None))
                return LSPIResult(tuple(trace), policy, None, "singular",
                                  repeat_start, repeat_length, LSPIWork(**work))
            if reuse_policy_evaluations:
                cache[policy] = matrix, rhs, weights
        improved = []
        for state in problem.states:
            scores = [sum((x * w for x, w in zip(problem.features[(state, action)], weights)), F(0))
                      for action in problem.actions]
            work["action_scores"] += len(problem.actions)
            improved.append(problem.actions[max(range(len(scores)), key=scores.__getitem__)])
        improved = tuple(improved)
        trace.append(LSPIIterate(index, policy, matrix, rhs, weights, improved))
        if improved in seen and repeat_start is None:
            repeat_start = seen[improved]
            repeat_length = index + 1 - repeat_start
        seen.setdefault(improved, index + 1)
        policy = improved
    status = ("fixed_point" if repeat_length == 1 else
              "cycle" if repeat_length is not None else "budget_exhausted")
    return LSPIResult(tuple(trace), policy, trace[-1].weights, status,
                      repeat_start, repeat_length, LSPIWork(**work))


def scratch_lspi(problem, trajectories):
    """Independent per-transition retained-only assembly at each policy step."""
    data, work = _data(problem, trajectories), _new_work()

    def assemble(policy):
        choices = dict(zip(problem.states, policy))
        matrix, rhs = _initial_matrix(problem), [F(0)] * problem.dimension
        for trajectory in data.values():
            for t in trajectory.transitions:
                work["samples_read"] += 1
                phi = problem.features[(t.state, t.action)]
                successor = ((F(0),) * problem.dimension if t.next_state is None else
                             problem.features[(t.next_state, choices[t.next_state])])
                # Directly assemble every sample; no successor grouping or
                # reuse of the replay path's rank-one-update helper.
                for i in range(problem.dimension):
                    rhs[i] += phi[i] * _fraction(t.reward)
                    for j in range(problem.dimension):
                        matrix[i][j] += phi[i] * (phi[j] - problem.gamma * successor[j])
        return matrix, rhs

    return _run(problem, assemble, work)


class LSPIReplayState:
    """Retained sufficient statistics for the specified finite LSPI schedule.

    Stores A(pi0), b, and G(z)=sum_{i: successor_i=z} phi_i. Every replay
    starts from pi0. Data deletion is committed only if all scheduled solves
    succeed; singular results leave this state unchanged. Cycles and budget
    exhaustion are successful finite-schedule outputs, not convergence claims.
    """

    reuse_policy_evaluations = False

    def __init__(self, problem, trajectories):
        self.problem = problem
        self._data = _data(problem, trajectories)
        self._anchor = _initial_matrix(problem)
        self._rhs = [F(0)] * problem.dimension
        self._successors = {}
        self.preparation_samples = 0
        for trajectory in self._data.values():
            for t in trajectory.transitions:
                self.preparation_samples += 1
                phi = problem.features[(t.state, t.action)]
                _outer_add(self._anchor, phi, phi)
                for i, x in enumerate(phi):
                    self._rhs[i] += x * _fraction(t.reward)
                if t.next_state is not None:
                    group = self._successors.setdefault(t.next_state, [F(0)] * problem.dimension)
                    for i, x in enumerate(phi):
                        group[i] += x
        for state, group in self._successors.items():
            _outer_add(self._anchor, group,
                       problem.features[(state, problem.initial_policy[state])], -problem.gamma)

    @property
    def trajectory_ids(self):
        return frozenset(self._data)

    def active_trajectories(self):
        return tuple(self._data.values())

    def _replay(self, anchor, rhs, successors, samples_read=0):
        problem, work = self.problem, _new_work(samples_read)
        matrix = [list(row) for row in anchor]
        previous = tuple(problem.initial_policy[s] for s in problem.states)
        indices = {s: i for i, s in enumerate(problem.states)}

        def assemble(policy):
            nonlocal previous
            for state, group in successors.items():
                work["successor_entries_checked"] += 1
                old, new = previous[indices[state]], policy[indices[state]]
                if old != new:
                    change = tuple(y - x for x, y in zip(problem.features[(state, old)],
                                                       problem.features[(state, new)]))
                    if any(group) and any(change):
                        _outer_add(matrix, group, change, -problem.gamma)
                        work["target_rank_one_updates"] += 1
            previous = policy
            return matrix, rhs

        return _run(problem, assemble, work, reuse_policy_evaluations=self.reuse_policy_evaluations)

    def replay(self):
        return self._replay(self._anchor, self._rhs, self._successors)

    def delete_and_replay(self, trajectory_ids):
        requested = tuple(trajectory_ids)
        if len(set(requested)) != len(requested):
            raise ValueError("duplicate deletion ids")
        unknown = {identifier for identifier in requested if identifier not in self._data}
        if unknown:
            raise KeyError(f"unknown or already deleted ids: {sorted(unknown)}")
        problem = self.problem
        anchor, rhs = [list(row) for row in self._anchor], list(self._rhs)
        successors = {s: list(group) for s, group in self._successors.items()}
        samples_read = 0
        for trajectory_id in requested:
            for t in self._data[trajectory_id].transitions:
                samples_read += 1
                phi = problem.features[(t.state, t.action)]
                target = ((F(0),) * problem.dimension if t.next_state is None else
                          problem.features[(t.next_state, problem.initial_policy[t.next_state])])
                residual = tuple(x - problem.gamma * y for x, y in zip(phi, target))
                _outer_add(anchor, phi, residual, F(-1))
                for i, x in enumerate(phi):
                    rhs[i] -= x * _fraction(t.reward)
                if t.next_state is not None:
                    group = successors[t.next_state]
                    for i, x in enumerate(phi):
                        group[i] -= x
        result = self._replay(anchor, rhs, successors, samples_read)
        if result.status != "singular":
            self._anchor, self._rhs, self._successors = anchor, rhs, successors
            for trajectory_id in requested:
                del self._data[trajectory_id]
        return result


class LSPIReplayMemoizedState(LSPIReplayState):
    """Target corrections with the same per-request memoization as cached LSPI.

    A cache hit skips assembly and solving, but preserves every scheduled trace
    entry and greedy action scan. The mutable assembly matrix and its `previous`
    policy both remain at the last actual assembly, so they stay consistent.
    Evaluation caches never cross deletion requests. This is a conventional
    optimization, not a new unlearning algorithm or an inverse-update method.
    """

    reuse_policy_evaluations = True
