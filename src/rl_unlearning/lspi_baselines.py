"""Cached-statistic reconstruction baseline for the declared rational LSPI learner.

Store C=lambda I+sum phi phi^T, b=sum phi r, and successor sums G(z).
Rebuild A(pi)=C-gamma sum_z G(z) phi(z,pi(z))^T at EVERY iteration,
including unchanged policies. This is a stronger comparator than rebuilding
from all retained samples, without policy-switch-specific matrix reuse.

The finite schedule, exact solver, terminal mask, ties, and singular outcome
are inherited from lspi_control. This is established sufficient-statistic
algebra, not a new unlearning or complexity claim.
"""

from fractions import Fraction as F

from .lspi_control import _data, _fraction, _initial_matrix, _new_work, _outer_add, _run


def _change_successor(groups, state, phi, sign):
    """Keep only nonzero aggregates; deletion can revive a cancelled group."""
    if state is None:
        return
    old = groups.get(state, (F(0),) * len(phi))
    updated = [x + sign * y for x, y in zip(old, phi)]
    if any(updated):
        groups[state] = updated
    else:
        groups.pop(state, None)


class LSPICachedState:
    """Cached reconstruction with atomic trajectory deletion and full replay.

    API matches LSPIReplayState. Every request starts the declared initial
    policy, not the full-data final policy. A singular evaluation leaves all
    stored data/statistics unchanged. Cycle/budget statuses are valid results
    of the finite learner and commit deletion, without claiming convergence.

    Cost boundary: preparation reads all samples. Deletion copies C, b, and G,
    subtracts requested samples, and replays all scheduled evaluations. Each
    evaluation copies C, reads every nonzero successor aggregate, reconstructs
    all its target terms, scores all actions, and performs a fresh dense solve.
    Selected LSPIWork counters exclude copy/allocation and rational-bit costs;
    benchmark complete wall time and memory, including preparation.

    For this class, target_rank_one_updates counts reconstruction outer-product
    terms, including terms at UNCHANGED policies. In LSPIReplayState it counts
    only nonzero policy-switch corrections. successor_entries_checked counts
    every retained nonzero aggregate visited. Cancelled/empty groups are omitted
    exactly, and are restored if a later deletion makes their sum nonzero.
    """

    reuse_policy_evaluations = False

    def __init__(self, problem, trajectories):
        self.problem = problem
        self._data = _data(problem, trajectories)
        self._covariance = _initial_matrix(problem)
        self._rhs = [F(0)] * problem.dimension
        self._successors = {}
        self.preparation_samples = 0
        for trajectory in self._data.values():
            for t in trajectory.transitions:
                self.preparation_samples += 1
                phi = problem.features[(t.state, t.action)]
                _outer_add(self._covariance, phi, phi)
                reward = _fraction(t.reward)
                for i, value in enumerate(phi):
                    self._rhs[i] += value * reward
                _change_successor(self._successors, t.next_state, phi, 1)

    @property
    def trajectory_ids(self):
        return frozenset(self._data)

    def active_trajectories(self):
        return tuple(self._data.values())

    def _reconstruct(self, covariance, rhs, successors, samples_read=0):
        problem, work = self.problem, _new_work(samples_read)
        indices = {state: i for i, state in enumerate(problem.states)}

        def assemble(policy):
            matrix = [list(row) for row in covariance]
            for state, group in successors.items():
                work["successor_entries_checked"] += 1
                target = problem.features[(state, policy[indices[state]])]
                _outer_add(matrix, group, target, -problem.gamma)
                work["target_rank_one_updates"] += 1
            return matrix, rhs

        return _run(problem, assemble, work, reuse_policy_evaluations=self.reuse_policy_evaluations)

    def replay(self):
        return self._reconstruct(self._covariance, self._rhs, self._successors)

    def delete_and_replay(self, trajectory_ids):
        requested = tuple(trajectory_ids)
        if len(set(requested)) != len(requested):
            raise ValueError("duplicate deletion ids")
        unknown = {trajectory_id for trajectory_id in requested if trajectory_id not in self._data}
        if unknown:
            raise KeyError(f"unknown or already deleted ids: {sorted(unknown)}")
        problem = self.problem
        covariance = [list(row) for row in self._covariance]
        rhs = list(self._rhs)
        successors = {state: list(group) for state, group in self._successors.items()}
        samples_read = 0
        for trajectory_id in requested:
            for t in self._data[trajectory_id].transitions:
                samples_read += 1
                phi = problem.features[(t.state, t.action)]
                _outer_add(covariance, phi, phi, F(-1))
                reward = _fraction(t.reward)
                for i, value in enumerate(phi):
                    rhs[i] -= value * reward
                _change_successor(successors, t.next_state, phi, -1)
        result = self._reconstruct(covariance, rhs, successors, samples_read)
        if result.status != "singular":
            self._covariance, self._rhs, self._successors = covariance, rhs, successors
            for trajectory_id in requested:
                del self._data[trajectory_id]
        return result


class LSPIMemoizedState(LSPICachedState):
    """Classical per-request memoization of identical policy evaluations.

    Reuses exact matrices/weights only within the same retained-data replay;
    every deletion clears this evaluation cache. The declared trace still has
    K steps and all greedy scores are computed. Cache storage is O(K*d*d),
    and there is no cross-deletion inverse update or new algorithmic claim.
    """

    reuse_policy_evaluations = True
