"""Retained-only equivalence and screening boundary regressions for LSPI."""

from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rl_unlearning.lspi_control import LSPIProblem, LSPIPolicyResult, scratch_lspi
from rl_unlearning.lspi_decision_repair import LSPISelectiveRepairState, LSPISharedAnchorRepairState
from rl_unlearning.lspi_inverse import LSPIInverseDeduplicatedState
from test_lspi_control import episode, switch_fixture


class DecisionRepairTests(unittest.TestCase):
    def assertTrace(self, result, reference):
        self.assertEqual(result.iterates, reference.iterates)
        for name in ('policy', 'weights', 'status', 'repeat_start', 'repeat_length'):
            self.assertEqual(getattr(result, name), getattr(reference, name), name)

    def assertPolicy(self, result, reference):
        self.assertIsInstance(result, LSPIPolicyResult)
        self.assertFalse(hasattr(result, 'weights'))
        self.assertFalse(hasattr(result, 'iterates'))
        self.assertEqual(result.policy_sequence, tuple(t.policy_out for t in reference.iterates))
        for name in ('policy', 'status', 'repeat_start', 'repeat_length'):
            self.assertEqual(getattr(result, name), getattr(reference, name), name)

    @staticmethod
    def snapshot(state):
        return deepcopy((state._data, state._covariance, state._rhs,
                         state._successors, state._inverse_cache, state._decision_cache))

    def test_policy_changes_while_stable_feature_group_is_skipped(self):
        problem = LSPIProblem((0, 1), (0, 1),
                              {(0, 0): (1, 0), (0, 1): (-1, 0),
                               (1, 0): (0, 1), (1, 1): (0, -1)},
                              {0: 0, 1: 0}, gamma=0, regularization=1, iterations=5)
        logs = (episode('positive', (0, 0, 3, None)),
                episode('negative', (0, 0, -1, None)),
                episode('stable', (1, 0, 20, None)))
        state = LSPISelectiveRepairState(problem, logs)
        before = state.replay()
        self.assertTrace(before, scratch_lspi(problem, logs))
        result = state.delete_and_replay(('positive',))
        self.assertTrace(result, scratch_lspi(problem, logs[1:]))
        self.assertEqual(before.policy, (0, 0))
        self.assertEqual(result.policy, (1, 0))
        self.assertGreater(state.last_decision_work['groups_certified'], 0)
        self.assertGreater(state.last_decision_work['decisions_changed_from_anchor'], 0)
        baseline = LSPIInverseDeduplicatedState(problem, logs)
        baseline.replay()
        comparison = baseline.delete_and_replay(('positive',))
        self.assertLess(result.work.action_scores, comparison.work.action_scores)

    def test_immutable_anchor_uses_cumulative_drift_and_scores_equality_boundary(self):
        # w starts at 3/4, then moves to 1/2 and exactly zero. Updating the
        # anchor weight after the first safe skip while retaining its old
        # margin would miss the second request's tie-induced action switch.
        problem = LSPIProblem((0,), (0, 1), {(0, 0): (-1,), (0, 1): (1,)},
                              {0: 1}, gamma=0, regularization=1, iterations=4)
        logs = (episode('a', (0, 1, F(3, 2), None)),
                episode('b', (0, 1, F(3, 2), None)),
                episode('zero', (0, 1, 0, None)))
        state = LSPISelectiveRepairState(problem, logs)
        state.replay()
        anchor = state._decision_cache[(1,)]
        self.assertEqual(anchor.weights, (F(3, 4),))
        self.assertEqual(anchor.radii, (F(3, 4),))
        first = state.delete_and_replay(('a',))
        self.assertTrace(first, scratch_lspi(problem, logs[1:]))
        self.assertEqual(first.work.action_scores, 0)
        self.assertIs(state._decision_cache[(1,)], anchor)
        second = state.delete_and_replay(('b',))
        self.assertTrace(second, scratch_lspi(problem, logs[2:]))
        self.assertEqual(second.policy, (0,))
        self.assertGreater(second.work.action_scores, 0)
        self.assertGreater(state.last_decision_work['decisions_changed_from_anchor'], 0)
        self.assertEqual(anchor.weights, (F(3, 4),))

    def test_zero_drift_preserves_ties_and_identical_actions_need_no_margin(self):
        for identical in (False, True):
            with self.subTest(identical=identical):
                problem = LSPIProblem((0,), (1, 0),
                                      {(0, 0): (1,), (0, 1): (1 if identical else -1,)},
                                      {0: 0}, gamma=0, iterations=5)
                reward = 1 if identical else 0
                logs = (episode('a', (0, 0, reward, None)),
                        episode('b', (0, 0, reward, None)))
                state = LSPISelectiveRepairState(problem, logs)
                state.replay()
                result = state.delete_and_replay(('a',))
                self.assertTrace(result, scratch_lspi(problem, logs[1:]))
                self.assertEqual(result.policy, (1,))
                self.assertEqual(result.work.action_scores, 0)
                self.assertGreater(state.last_decision_work['groups_certified'], 0)
                if identical:
                    self.assertTrue(all(not anchor.radii for anchor in state._decision_cache.values()))

    def test_singular_deletion_rolls_back_statistics_inverses_and_decision_anchors(self):
        logs = (episode('loop', (0, 1, 1, 0), (0, 1, 0, None)),
                episode('extra', (0, 1, 0, None)))
        for initial in (0, 1):
            for policy_only in (False, True):
                with self.subTest(initial=initial, policy_only=policy_only):
                    problem = LSPIProblem((0,), (0, 1),
                                          {(0, 0): (5,), (0, 1): (1,)}, {0: initial})
                    state = LSPISelectiveRepairState(problem, logs, policy_only=policy_only)
                    state.replay()
                    before = self.snapshot(state)
                    for _ in range(2):
                        result = state.delete_and_replay(('extra',))
                        reference = scratch_lspi(problem, logs[:1])
                        (self.assertPolicy if policy_only else self.assertTrace)(result, reference)
                        self.assertEqual(result.status, 'singular')
                        self.assertEqual(self.snapshot(state), before)

    def test_invalid_requests_are_atomic_and_empty_request_preserves_trace(self):
        problem, logs, _ = switch_fixture()
        state = LSPISelectiveRepairState(problem, logs)
        state.replay()
        before = self.snapshot(state)
        for ids, error in ((('forget', 'absent'), KeyError),
                           (('forget', 'forget'), ValueError)):
            with self.assertRaises(error):
                state.delete_and_replay(ids)
            self.assertEqual(self.snapshot(state), before)
        self.assertTrace(state.delete_and_replay(()), scratch_lspi(problem, logs))
        self.assertEqual(self.snapshot(state), before)
        self.assertEqual(state.last_inverse_work['action_scores'], 0)

    def test_random_mixed_features_sequential_requests_match_full_scratch_trace(self):
        rng = random.Random(61016)
        accepted = certified = changed = 0
        for trial in range(24):
            d = 1 + trial % 4
            features = {(s, a): tuple(F(rng.randrange(-3, 4), 3) for _ in range(d))
                        for s in range(4) for a in range(3)}
            problem = LSPIProblem(tuple(range(4)), (2, 0, 1), features,
                                  dict.fromkeys(range(4), trial % 3),
                                  gamma=F(2, 3), iterations=6)
            logs = []
            for i in range(12):
                s, z = rng.randrange(4), rng.randrange(4)
                logs.append(episode(str(i), (s, rng.randrange(3), rng.randrange(-3, 4), z),
                                    (z, rng.randrange(3), rng.randrange(-3, 4), None)))
            state = LSPISelectiveRepairState(problem, logs)
            self.assertTrace(state.replay(), scratch_lspi(problem, logs))
            for i in range(4):
                with self.subTest(trial=trial, deletion=i):
                    candidate = tuple(t for t in state.active_trajectories()
                                      if t.trajectory_id != str(i))
                    before = self.snapshot(state)
                    result = state.delete_and_replay((str(i),))
                    self.assertTrace(result, scratch_lspi(problem, candidate))
                    if result.status == 'singular':
                        self.assertEqual(self.snapshot(state), before)
                    else:
                        accepted += 1
                        certified += state.last_decision_work['groups_certified']
                        changed += state.last_decision_work['decisions_changed_from_anchor']
                        self.assertEqual(set(state._decision_cache), set(state.cached_policies))
                        self.assertLessEqual(len(state._decision_cache), problem.iterations)
        self.assertGreater(accepted, 80)
        self.assertGreater(certified, 0)
        self.assertGreater(changed, 0)

    def test_deduplication_scores_distinct_ordered_action_rows_once_per_input_policy(self):
        features = {(s, a): ((1 if a == 1 else -1),)
                    for s in range(8) for a in (1, 0)}
        problem = LSPIProblem(tuple(range(8)), (1, 0), features,
                              dict.fromkeys(range(8), 1), gamma=0, iterations=7)
        logs = (episode('a', (0, 1, 1, None)), episode('b', (7, 1, 1, None)))
        for policy_only in (False, True):
            state = LSPIInverseDeduplicatedState(problem, logs, policy_only=policy_only)
            self.assertEqual(len(state.feature_groups), 1)
            for result, retained in ((state.replay(), logs),
                                     (state.delete_and_replay(('a',)), logs[1:])):
                (self.assertPolicy if policy_only else self.assertTrace)(
                    result, scratch_lspi(problem, retained))
                self.assertEqual(result.work.action_scores, 2)

    def test_policy_only_matches_full_cycle_budget_and_singular_outputs(self):
        cycle = LSPIProblem((0,), (0, 1), {(0, 0): (3,), (0, 1): (1,)},
                            {0: 0}, gamma=F(9, 10), iterations=7)
        cycle_logs = (episode('loop', (0, 1, 1, 0), (0, 1, 0, None)),)
        singular = LSPIProblem((0,), (0,), {(0, 0): (1,)}, {0: 0}, regularization=0)
        budget, _, budget_logs = switch_fixture(iterations=1)
        for problem, logs, expected_status in ((cycle, cycle_logs, 'cycle'),
                                               (singular, (), 'singular'),
                                               (budget, budget_logs, 'budget_exhausted')):
            reference = scratch_lspi(problem, logs)
            self.assertEqual(reference.status, expected_status)
            for cls in (LSPISelectiveRepairState, LSPIInverseDeduplicatedState):
                with self.subTest(method=cls.__name__, status=reference.status):
                    full_state, policy_state = cls(problem, logs), cls(problem, logs, policy_only=True)
                    full, policy = full_state.replay(), policy_state.replay()
                    self.assertTrace(full, reference)
                    self.assertPolicy(policy, reference)
                    self.assertEqual(full.work, policy.work)
                    self.assertEqual(full_state._inverse_cache, policy_state._inverse_cache)
                    if reference.status != 'singular':
                        self.assertPolicy(policy_state.delete_and_replay(()), reference)

    def test_shared_bank_new_policy_branch_never_rebuilds_margin_index(self):
        problem, logs, retained = switch_fixture(iterations=7)
        for policy_only in (False, True):
            with self.subTest(policy_only=policy_only):
                state = LSPISharedAnchorRepairState(problem, logs, policy_only=policy_only)
                state.replay()
                original_policies = state.cached_policies
                bank = state._anchor_bank
                self.assertGreater(len(bank), 0)
                result = state.delete_and_replay(('forget',))
                reference = scratch_lspi(problem, retained)
                (self.assertPolicy if policy_only else self.assertTrace)(result, reference)
                self.assertTrue(state.cached_policies - original_policies)
                self.assertEqual(state.last_decision_work['margin_groups_built'], 0)
                self.assertEqual(state.last_decision_work['margin_constraints_built'], 0)
                self.assertGreater(state.last_decision_work['index_queries'], 0)
                self.assertIs(state._anchor_bank, bank)
                self.assertTrue(all(any(anchor is item for item in bank)
                                    for anchor in state._decision_cache.values()))

    def test_shared_bank_random_sequential_full_traces_and_frozen_anchors(self):
        rng = random.Random(262606)
        accepted = new_branches = 0
        for trial in range(24):
            d = 1 + trial % 4
            features = {(s, a): tuple(F(rng.randrange(-3, 4), 2) for _ in range(d))
                        for s in range(4) for a in range(2)}
            problem = LSPIProblem(tuple(range(4)), (1, 0), features,
                                  dict.fromkeys(range(4), trial % 2),
                                  gamma=F(2, 3), iterations=7)
            logs = []
            for i in range(10):
                s, z = rng.randrange(4), rng.randrange(4)
                logs.append(episode(str(i), (s, rng.randrange(2), rng.randrange(-3, 4), z),
                                    (z, rng.randrange(2), rng.randrange(-3, 4), None)))
            state = LSPISharedAnchorRepairState(problem, logs)
            initial = state.replay()
            self.assertTrace(initial, scratch_lspi(problem, logs))
            if initial.status == 'singular':
                self.assertIsNone(state._anchor_bank)
                continue
            bank = state._anchor_bank
            frozen = deepcopy(bank)
            self.assertLessEqual(len(bank), problem.iterations)
            for i in range(5):
                with self.subTest(trial=trial, deletion=i):
                    candidate = tuple(t for t in state.active_trajectories()
                                      if t.trajectory_id != str(i))
                    old_policies = state.cached_policies
                    before = self.snapshot(state)
                    result = state.delete_and_replay((str(i),))
                    self.assertTrace(result, scratch_lspi(problem, candidate))
                    self.assertIs(state._anchor_bank, bank)
                    self.assertEqual(bank, frozen)
                    self.assertEqual(state.last_decision_work['margin_groups_built'], 0)
                    self.assertEqual(state.last_decision_work['margin_constraints_built'], 0)
                    if result.status == 'singular':
                        self.assertEqual(self.snapshot(state), before)
                    else:
                        accepted += 1
                        new_branches += bool(state.cached_policies - old_policies)
                        self.assertEqual(set(state._decision_cache), set(state.cached_policies))
                        self.assertTrue(all(any(anchor is item for item in bank)
                                            for anchor in state._decision_cache.values()))
        self.assertGreater(accepted, 100)
        self.assertGreater(new_branches, 0)

    def test_shared_bank_singular_rollback_and_failed_initial_schedule_leave_bank_unchanged(self):
        logs = (episode('loop', (0, 1, 1, 0), (0, 1, 0, None)),
                episode('extra', (0, 1, 0, None)))
        for initial in (0, 1):
            problem = LSPIProblem((0,), (0, 1),
                                  {(0, 0): (5,), (0, 1): (1,)}, {0: initial})
            for policy_only in (False, True):
                with self.subTest(initial=initial, policy_only=policy_only):
                    state = LSPISharedAnchorRepairState(problem, logs, policy_only=policy_only)
                    state.replay()
                    bank, before = state._anchor_bank, self.snapshot(state)
                    for _ in range(2):
                        result = state.delete_and_replay(('extra',))
                        (self.assertPolicy if policy_only else self.assertTrace)(
                            result, scratch_lspi(problem, logs[:1]))
                        self.assertEqual(result.status, 'singular')
                        self.assertIs(state._anchor_bank, bank)
                        self.assertEqual(self.snapshot(state), before)
                    cold = LSPISharedAnchorRepairState(problem, logs[:1], policy_only=policy_only)
                    failed = cold.replay()
                    self.assertEqual(failed.status, 'singular')
                    self.assertIsNone(cold._anchor_bank)
                    self.assertEqual(cold._decision_cache, {})
                    self.assertEqual(cold._inverse_cache, {})


if __name__ == '__main__':
    unittest.main()
