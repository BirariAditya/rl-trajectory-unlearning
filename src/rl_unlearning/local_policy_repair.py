"""Exact repair on a backward-closed state block, with counted global fallback.

This is a structural locality baseline, not the proposed margin-pruned theorem.
The old all-action support graph ensures the frozen outside values remain exact.
"""

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction as F
from types import MappingProxyType

from .nonpolicy_certificate import NonPolicyCertificate
from .policy_iteration import policy_iteration
from .reference import _solve


@dataclass(frozen=True)
class LocalRepairResult:
    policy: object
    affected_states: tuple
    changed_actions: tuple
    used_full_fallback: bool
    transitions_read: int
    successor_entries_copied: int
    reverse_edges_visited: int
    index_edges_removed: int
    action_evaluations: int
    linear_system_sizes: tuple
    pruned_action_rows: int
    pruning_disabled: bool
    pruning_rows_checked: int
    index_rows_rebuilt: int
    index_edges_added: int
    exact_policy_certificate: bool = True


class LocalPolicyRepair(NonPolicyCertificate):
    """Own exact rational logs, a policy/value cache, and an all-action reverse index.

    The policy view is live across requests. Copy it outside timed requests if
    historical snapshots are needed, and count that output cost separately.
    """

    def __init__(self, states, actions, gamma, trajectories, initial_policy=None,
                 anchor_policy=None):
        self._states, self._actions = tuple(states), tuple(actions)
        if not self._states or not self._actions or len(set(self._states)) != len(self._states) or len(set(self._actions)) != len(self._actions):
            raise ValueError("nonempty unique states/actions required")
        self._gamma = F(str(gamma))
        if not 0 <= self._gamma < 1:
            raise ValueError("invalid gamma")
        self._order = {a: i for i, a in enumerate(self._actions)}
        self._logs, self._rows = {}, {}
        reward_max = F(0)
        state_set = set(self._states)
        for trajectory in trajectories:
            if trajectory.trajectory_id in self._logs:
                raise ValueError("duplicate trajectory ID")
            self._logs[trajectory.trajectory_id] = trajectory
            for t in trajectory.transitions:
                if t.state not in state_set or t.action not in self._order or (t.next_state is not None and t.next_state not in state_set):
                    raise ValueError("unknown state/action/successor")
                count, reward, successors = self._rows.get((t.state, t.action), (0, F(0), Counter()))
                successors[t.next_state] += 1
                self._rows[(t.state, t.action)] = (count + 1, reward + F(str(t.reward)), successors)
                reward_max = max(reward_max, F(str(t.reward)))
        self._upper = reward_max / (1 - self._gamma)
        self._anchor_policy = None if anchor_policy is None else {s: anchor_policy[s] for s in self._states}
        self._anchor_values = {}
        self._pruned_rows = set()
        if self._anchor_policy is not None:
            if any(a not in self._actions for a in self._anchor_policy.values()):
                raise ValueError("invalid anchor action")
            matrix, rhs = [], []
            for i, s in enumerate(self._states):
                reward, probabilities = self._lookup(s, self._anchor_policy[s])
                matrix.append([F(i == j) - self._gamma * probabilities.get(t, F(0))
                               for j, t in enumerate(self._states)])
                rhs.append(reward)
            self._anchor_values = dict(zip(self._states, _solve(matrix, rhs)))
            self._pruned_rows = {(s, a) for s in self._states for a in self._actions
                                 if self._dominated(s, self._rows.get((s, a)))}
        self._reverse = {s: set() for s in self._states}
        for row, (_, _, successors) in self._rows.items():
            for successor in successors:
                if successor is not None and row not in self._pruned_rows:
                    self._reverse[successor].add(row)
        self.preparation_plan = policy_iteration(self._states, self._actions, self._gamma,
                                                self._lookup, initial_policy=initial_policy)
        self._policy = dict(self.preparation_plan.policy)
        self._values = dict(self.preparation_plan.values)
        self.policy = MappingProxyType(self._policy)
        self.values = MappingProxyType(self._values)

    def _lookup(self, state, action):
        return self._normalized(self._rows.get((state, action)))

    def _dominated(self, state, row):
        reward, probabilities = self._normalized(row)
        upper_q = reward + self._gamma * self._upper * sum(
            (p for s, p in probabilities.items() if s is not None), F(0))
        return upper_q < self._anchor_values[state]

    def delete_and_repair(self, trajectory_ids, *, max_local_fraction=1.0,
                          max_iterations=10_000, force_full=False):
        if not 0 <= max_local_fraction <= 1 or max_iterations < 1:
            raise ValueError("invalid local fraction or solver budget")
        ids, candidates, transitions_read, copied = self._prepare_deletion(trajectory_ids)
        # A touched row can be statistically unchanged after redundant deletion.
        changed = {row for row, candidate in candidates.items()
                   if self._normalized(candidate) != self._normalized(self._rows[row])}
        anchor_invalid = self._anchor_policy is not None and any(
            a == self._anchor_policy[s] for s, a in changed)
        candidate_pruned = ({row: self._dominated(row[0], value) for row, value in candidates.items()}
                            if self._anchor_policy is not None and not anchor_invalid else {})
        affected = {s for s, a in changed
                    if anchor_invalid or (s, a) not in self._pruned_rows or not candidate_pruned.get((s, a), False)}
        pending = list(sorted(affected))
        edges_visited = 0
        while pending and not force_full and not anchor_invalid:
            state = pending.pop()
            for row in self._reverse[state]:
                edges_visited += 1
                if row[0] not in affected:
                    affected.add(row[0])
                    pending.append(row[0])
        fallback = bool(force_full or anchor_invalid or
                        (affected and len(affected) > max_local_fraction * len(self._states)))
        solve_states = self._states if fallback else tuple(sorted(affected))

        def retained_lookup(s, a):
            row = (s, a)
            return self._normalized(candidates[row] if row in candidates else self._rows.get(row))

        plan = policy_iteration(solve_states, self._actions, self._gamma, retained_lookup,
                                initial_policy=self._policy, boundary_values=self._values,
                                max_iterations=max_iterations)
        changed_actions = tuple((s, plan.policy[s]) for s in solve_states
                                if plan.policy[s] != self._policy[s])
        removed_edges = added_edges = rebuilt_rows = 0
        if not anchor_invalid:
            for row, candidate in candidates.items():
                old_edges = {s for s in self._rows[row][2] if s is not None} if row not in self._pruned_rows else set()
                new_edges = {s for s in candidate[2] if s is not None} if not candidate_pruned.get(row, False) else set()
                for successor in old_edges - new_edges:
                    self._reverse[successor].remove(row)
                    removed_edges += 1
                for successor in new_edges - old_edges:
                    self._reverse[successor].add(row)
                    added_edges += 1
                if candidate_pruned.get(row, False):
                    self._pruned_rows.add(row)
                else:
                    self._pruned_rows.discard(row)
        # Commit only after successful exact solution. Deletion cannot create a
        # new nonterminal support edge under this empirical model convention.
        self._commit_deletion(ids, candidates)
        if anchor_invalid:
            # A changed lower-bound policy invalidates previous action pruning.
            # The request used a full exact solve; rebuild the raw support index.
            self._anchor_policy = None
            self._anchor_values = {}
            self._pruned_rows.clear()
            self._reverse = {s: set() for s in self._states}
            for row, (_, _, successors) in self._rows.items():
                rebuilt_rows += 1
                for successor in successors:
                    if successor is not None:
                        self._reverse[successor].add(row)
                        added_edges += 1
        self._policy.update(plan.policy)
        self._values.update(plan.values)
        return LocalRepairResult(self.policy, tuple(sorted(affected)), changed_actions,
                                 fallback, transitions_read, copied, edges_visited,
                                 removed_edges, plan.action_evaluations, plan.linear_system_sizes,
                                 len(self._pruned_rows), anchor_invalid, len(candidate_pruned),
                                 rebuilt_rows, added_edges)

    def try_delete(self, trajectory_ids):
        raise NotImplementedError("use delete_and_repair; it also maintains the reverse index")
