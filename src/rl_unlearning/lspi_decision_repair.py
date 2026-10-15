"""Exact LSPI with indexed greedy-decision screening, including policy changes.

This is an experimental combination of established inverse updates and margin
screening, not a claimed new unlearning theorem or privacy mechanism. Full
retained-only matrices, weights and policy traces are returned by default.
Historical decision anchors remain internal; they are not erased.
"""

from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction as F

from .lspi_inverse import LSPIInverseDeduplicatedState


@dataclass(frozen=True)
class DecisionAnchor:
    weights: tuple
    group_choices: tuple
    policy_out: tuple
    radii: tuple
    group_order: tuple


class LSPISelectiveRepairState(LSPIInverseDeduplicatedState):
    """Reuse fixed margin anchors; rescore only groups not certified unchanged.

    For each old greedy action, rho is the minimum action gap divided by the
    feature-difference L1 norm. With delta=||w_new-w_anchor||_inf, groups whose
    rho>delta retain their action. rho<=delta groups are evaluated exactly.
    Identical action features impose no constraint; delta=0 preserves ties.

    Anchors are immutable across deletions, so no uncounted global margin
    refresh is required. An input policy first encountered in a successful
    schedule needs a full index build. Only anchors for the latest successful
    schedule are kept. Candidate anchors are discarded on singular failure.
    Full policy copying/hashing is still O(S); rational bit costs are not
    bounded. Preparation and new-policy indexing must be timed by callers.
    """

    def __init__(self, problem, trajectories, *, policy_only=False):
        super().__init__(problem, trajectories, policy_only=policy_only)
        self._decision_cache = {}
        self.last_decision_work = {}

    def _run_inverse(self, covariance, rhs, successors, deleted=()):
        self._candidate_decisions = dict(self._decision_cache)
        self.last_decision_work = dict(
            groups_scored=0, groups_certified=0, margin_groups_built=0,
            margin_constraints_built=0, index_queries=0,
            policy_entries_materialized=0, decisions_changed_from_anchor=0,
        )
        try:
            result, inverse_cache = super()._run_inverse(covariance, rhs, successors, deleted)
            if inverse_cache is not None:
                self._decision_cache = {
                    policy: self._candidate_decisions[policy] for policy in inverse_cache
                }
            return result, inverse_cache
        finally:
            del self._candidate_decisions

    def _build_anchor(self, weights, work):
        choices, radii = [], []
        for index, group in enumerate(self.feature_groups):
            chosen = self._score_group(group, weights, work)
            choices.append(chosen)
            self.last_decision_work['groups_scored'] += 1
            self.last_decision_work['margin_groups_built'] += 1
            chosen_features = group[self.problem.actions.index(chosen)]
            constraints = []
            for features in group:
                difference = tuple(x-y for x, y in zip(chosen_features, features))
                norm = sum(map(abs, difference), F(0))
                if norm:
                    gap = sum((x*y for x, y in zip(difference, weights)), F(0))
                    constraints.append(gap / norm)
                    self.last_decision_work['margin_constraints_built'] += 1
            if constraints:
                radii.append((min(constraints), index))
        radii.sort()
        policy_out = tuple(choices[group] for group in self.state_groups)
        self.last_decision_work['policy_entries_materialized'] += len(policy_out)
        return DecisionAnchor(tuple(weights), tuple(choices), policy_out,
                              tuple(r for r, _ in radii), tuple(i for _, i in radii))

    def _greedy(self, policy, weights, work):
        anchor = self._candidate_decisions.get(policy)
        if anchor is None:
            anchor = self._build_anchor(weights, work)
            self._candidate_decisions[policy] = anchor
            return anchor.policy_out
        return self._screen(anchor, weights, work)

    def _screen(self, anchor, weights, work, *, delta=None, count=None):
        if delta is None:
            delta = max(abs(x-y) for x, y in zip(weights, anchor.weights))
        if not delta:
            self.last_decision_work['groups_certified'] += len(self.feature_groups)
            return anchor.policy_out
        if count is None:
            self.last_decision_work['index_queries'] += 1
            count = bisect_right(anchor.radii, delta)
        self.last_decision_work['groups_certified'] += len(self.feature_groups) - count
        if count == 0:
            return anchor.policy_out
        choices = list(anchor.group_choices)
        for group_id in anchor.group_order[:count]:
            choices[group_id] = self._score_group(self.feature_groups[group_id], weights, work)
            self.last_decision_work['groups_scored'] += 1
            self.last_decision_work['decisions_changed_from_anchor'] += (
                choices[group_id] != anchor.group_choices[group_id]
            )
        self.last_decision_work['policy_entries_materialized'] += len(self.state_groups)
        return tuple(choices[group] for group in self.state_groups)


class LSPISharedAnchorRepairState(LSPISelectiveRepairState):
    """Screen any new policy's weights using a frozen initial bank of anchors.

    Greedy decisions depend on weights and features, not on which input policy
    produced the weights. Thus the initial schedule's at-most-K anchors remain
    valid on new branches. Choose the anchor with the smallest uncertain prefix,
    charging every distance and binary search. No margin refresh is performed.
    Matrix evaluation on a new branch still requires exact retained assembly.
    """

    def __init__(self, problem, trajectories, *, policy_only=False):
        super().__init__(problem, trajectories, policy_only=policy_only)
        self._anchor_bank = None

    def _run_inverse(self, covariance, rhs, successors, deleted=()):
        result, cache = super()._run_inverse(covariance, rhs, successors, deleted)
        if self._anchor_bank is None and cache is not None:
            self._anchor_bank = tuple(self._decision_cache.values())
        return result, cache

    def _greedy(self, policy, weights, work):
        if self._anchor_bank is None:
            return super()._greedy(policy, weights, work)
        best = None
        for anchor in self._anchor_bank:
            delta = max(abs(x-y) for x, y in zip(weights, anchor.weights))
            count = bisect_right(anchor.radii, delta) if delta else 0
            self.last_decision_work['index_queries'] += 1
            if best is None or count < best[0]:
                best = count, delta, anchor
        count, delta, anchor = best
        self._candidate_decisions[policy] = anchor
        return self._screen(anchor, weights, work, delta=delta, count=count)
