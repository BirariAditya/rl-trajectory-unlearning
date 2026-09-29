"""Exact, restricted no-policy-change certificate using rational statistics.

Preparation is global and expensive. Requests inspect only affected rows, and
are accepted only when every chosen-policy row stays identical. An inconclusive
request leaves this object unchanged; callers must use a general solver.
"""

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction as F
from types import MappingProxyType

from .reference import _solve


@dataclass(frozen=True)
class CertificateResult:
    accepted: bool
    reason: str
    affected_rows: int
    checked_actions: int
    transitions_read: int
    successor_entries_copied: int
    # No Q-table equality or parameter-deletion claim is made.
    policy: object


class NonPolicyCertificate:
    """Own a verified policy and exact sufficient statistics for repeated requests.

Rewards/discount are interpreted through their decimal strings, as in the exact
reference. A binary64 learner is a different numerical model. Do not use this
certificate to claim bitwise equivalence to binary64 training.
"""

    def __init__(self, states, actions, gamma, trajectories, policy):
        self._states, self._actions = tuple(states), tuple(actions)
        if not self._states or not self._actions or len(set(self._states)) != len(self._states) or len(set(self._actions)) != len(self._actions):
            raise ValueError("nonempty unique states/actions required")
        self._gamma = F(str(gamma))
        if not 0 <= self._gamma < 1:
            raise ValueError("gamma must be in [0, 1)")
        self._policy = {s: policy[s] for s in self._states}
        if any(a not in self._actions for a in self._policy.values()):
            raise ValueError("invalid policy action")
        self.policy = MappingProxyType(self._policy)
        self._order = {a: i for i, a in enumerate(self._actions)}
        self._logs, self._rows = {}, {}
        for trajectory in trajectories:
            if trajectory.trajectory_id in self._logs:
                raise ValueError("duplicate trajectory ID")
            self._logs[trajectory.trajectory_id] = trajectory
            for t in trajectory.transitions:
                if t.state not in self._policy or t.action not in self._order or (t.next_state is not None and t.next_state not in self._policy):
                    raise ValueError("unknown state/action/successor")
                count, reward, successors = self._rows.get((t.state, t.action), (0, F(0), Counter()))
                successors[t.next_state] += 1
                self._rows[(t.state, t.action)] = (count + 1, reward + F(str(t.reward)), successors)
        matrix, rhs = [], []
        for i, s in enumerate(self._states):
            reward, probabilities = self._normalized(self._rows.get((s, self._policy[s])))
            rhs.append(reward)
            matrix.append([F(i == j) - self._gamma * probabilities.get(t, F(0))
                           for j, t in enumerate(self._states)])
        values = _solve(matrix, rhs)
        self._values = dict(zip(self._states, values))
        for s in self._states:
            for a in self._actions:
                if not self._action_valid(s, a, self._rows.get((s, a))):
                    raise ValueError("initial policy is not the exact tie-ordered optimum")

    @staticmethod
    def _normalized(row):
        if row is None or row[0] == 0:
            return F(0), {None: F(1)}
        count, reward, successors = row
        return reward / count, {s: F(n, count) for s, n in successors.items() if n}

    def _action_valid(self, state, action, row):
        reward, probabilities = self._normalized(row)
        target = reward + self._gamma * sum(
            (p * self._values[s] for s, p in probabilities.items() if s is not None), F(0))
        value = self._values[state]
        return target < value or (target == value and
               self._order[action] >= self._order[self._policy[state]])

    def try_delete(self, trajectory_ids):
        """Commit a certified request or return inconclusive without mutation."""
        ids, candidates, transitions_read, copied = self._prepare_deletion(trajectory_ids)
        checked = 0

        def result(accepted, reason):
            return CertificateResult(accepted, reason, len(candidates), checked,
                                     transitions_read, copied, self.policy)

        for (s, a), candidate in candidates.items():
            if a == self._policy[s] and self._normalized(candidate) != self._normalized(self._rows[(s, a)]):
                return result(False, "chosen_policy_row_changed")
        for (s, a), candidate in candidates.items():
            checked += 1
            if not self._action_valid(s, a, candidate):
                return result(False, "changed_action_beats_policy_or_tie_order")
        self._commit_deletion(ids, candidates)
        return result(True, "policy_unchanged")

    def _prepare_deletion(self, trajectory_ids):
        """Copy only touched statistics; all validation precedes mutation."""
        ids = tuple(trajectory_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate deletion ID")
        if any(identifier not in self._logs for identifier in ids):
            raise KeyError("unknown or already deleted trajectory ID")
        candidates = {}
        transitions_read = copied = 0
        for identifier in ids:
            for t in self._logs[identifier].transitions:
                row = (t.state, t.action)
                if row not in candidates:
                    count, reward, successors = self._rows[row]
                    candidates[row] = (count, reward, successors.copy())
                    copied += len(successors)
                count, reward, successors = candidates[row]
                successors[t.next_state] -= 1
                if successors[t.next_state] == 0:
                    del successors[t.next_state]
                candidates[row] = (count - 1, reward - F(str(t.reward)), successors)
                transitions_read += 1

        return ids, candidates, transitions_read, copied

    def _commit_deletion(self, ids, candidates):
        for row, candidate in candidates.items():
            if candidate[0]:
                self._rows[row] = candidate
            else:
                self._rows.pop(row, None)
        for identifier in ids:
            del self._logs[identifier]
