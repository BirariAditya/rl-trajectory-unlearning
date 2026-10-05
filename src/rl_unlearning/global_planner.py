"""Lean retained-model planning, optionally with classical bound elimination.

No reverse dependency index or local closure is constructed. Exact statistics
helpers are shared with the deletion baseline, but the planner owns no local
repair state. These are ordinary planning controls, not new unlearning methods.
"""

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction as F
from types import MappingProxyType

from .nonpolicy_certificate import NonPolicyCertificate
from .policy_iteration import ExactPlan, policy_iteration
from .scc_planner import scc_policy_iteration


@dataclass(frozen=True)
class GlobalPlanResult:
    plan: ExactPlan
    anchor_plan: ExactPlan | None
    transitions_read: int
    successor_entries_copied: int
    pruning_rows_checked: int
    pruned_action_rows: int
    anchor_refreshed: bool


class ExactGlobalPlanner:
    _normalized = staticmethod(NonPolicyCertificate._normalized)
    _prepare_deletion = NonPolicyCertificate._prepare_deletion
    _commit_deletion = NonPolicyCertificate._commit_deletion

    def __init__(self, states, actions, gamma, trajectories, *, solver="scc", anchor_policy=None):
        self._states, self._actions = tuple(states), tuple(actions)
        if not self._states or not self._actions or len(set(self._states)) != len(self._states) or len(set(self._actions)) != len(self._actions):
            raise ValueError("nonempty unique states/actions required")
        self._gamma = F(str(gamma))
        if not 0 <= self._gamma < 1 or solver not in ("dense", "scc"):
            raise ValueError("invalid discount or solver")
        self._solver = policy_iteration if solver == "dense" else scc_policy_iteration
        self._logs, self._rows = {}, {}
        states_set, actions_set = set(self._states), set(self._actions)
        reward_max = F(0)
        for trajectory in trajectories:
            if trajectory.trajectory_id in self._logs:
                raise ValueError("duplicate trajectory ID")
            self._logs[trajectory.trajectory_id] = trajectory
            for t in trajectory.transitions:
                if t.state not in states_set or t.action not in actions_set or (t.next_state is not None and t.next_state not in states_set):
                    raise ValueError("unknown state/action/successor")
                count, total, successors = self._rows.get((t.state, t.action), (0, F(0), Counter()))
                successors[t.next_state] += 1
                reward = F(str(t.reward))
                self._rows[t.state, t.action] = count + 1, total + reward, successors
                reward_max = max(reward_max, reward)
        self._upper = reward_max / (1 - self._gamma)
        self._anchor_policy = None if anchor_policy is None else {s: anchor_policy[s] for s in self._states}
        if self._anchor_policy is not None and any(a not in actions_set for a in self._anchor_policy.values()):
            raise ValueError("invalid anchor policy")
        self._anchor_values = {}
        anchor = self._evaluate_anchor(self._lookup, 10_000) if self._anchor_policy is not None else None
        if anchor is not None:
            self._anchor_values = anchor.values
        allowed, checked, pruned = self._admissible(self._lookup, self._anchor_values)
        self.preparation_plan = self._solver(self._states, self._actions, self._gamma, self._lookup,
                                             allowed_actions=allowed)
        self.preparation_result = GlobalPlanResult(self.preparation_plan, anchor, 0, 0, checked, pruned, anchor is not None)
        self._policy, self._values = dict(self.preparation_plan.policy), dict(self.preparation_plan.values)
        self.policy, self.values = MappingProxyType(self._policy), MappingProxyType(self._values)

    def _lookup(self, state, action):
        return self._normalized(self._rows.get((state, action)))

    def _evaluate_anchor(self, lookup, max_iterations):
        # Single-action optimization is exact fixed-policy evaluation. SCCs can
        # be much smaller than the all-action graph; charge this whole solve.
        return scc_policy_iteration(self._states, self._actions, self._gamma, lookup,
                                    initial_policy=self._anchor_policy, max_iterations=max_iterations,
                                    allowed_actions={s: (a,) for s, a in self._anchor_policy.items()})

    def _admissible(self, lookup, lower):
        if self._anchor_policy is None:
            return None, 0, 0
        allowed, checked, pruned = {}, 0, 0
        for state in self._states:
            keep = []
            for action in self._actions:
                reward, probabilities = lookup(state, action)
                upper_q = reward + self._gamma * self._upper * sum(
                    (p for successor, p in probabilities.items() if successor is not None), F(0))
                checked += 1
                if upper_q < lower[state]:
                    pruned += 1
                else:
                    keep.append(action)
            if not keep:
                raise RuntimeError("sound bounds cannot eliminate every action")
            allowed[state] = tuple(keep)
        return allowed, checked, pruned

    def delete_and_replan(self, trajectory_ids, *, max_iterations=10_000):
        if max_iterations < 1:
            raise ValueError("positive iteration budget required")
        ids, candidates, read, copied = self._prepare_deletion(trajectory_ids)

        def retained_lookup(state, action):
            row = state, action
            return self._normalized(candidates[row] if row in candidates else self._rows.get(row))

        refresh = self._anchor_policy is not None and any(
            a == self._anchor_policy[s] and self._normalized(candidate) != self._normalized(self._rows[s, a])
            for (s, a), candidate in candidates.items())
        anchor = self._evaluate_anchor(retained_lookup, max_iterations) if refresh else None
        lower = anchor.values if anchor is not None else self._anchor_values
        # Rescan all rows: a formerly eliminated action may become admissible.
        allowed, checked, pruned = self._admissible(retained_lookup, lower)
        plan = self._solver(self._states, self._actions, self._gamma, retained_lookup,
                            initial_policy=self._policy, max_iterations=max_iterations,
                            allowed_actions=allowed)
        # Commit statistics, cache and anchor only after all exact solves succeed.
        self._commit_deletion(ids, candidates)
        self._policy.update(plan.policy)
        self._values.update(plan.values)
        self._anchor_values = lower
        return GlobalPlanResult(plan, anchor, read, copied, checked, pruned, refresh)
