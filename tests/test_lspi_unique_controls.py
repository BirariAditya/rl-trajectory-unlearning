"""Check fixture semantics and the benchmark's independent exact comparison."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_lspi_unique_controls import dataset, feature_counts, run, summarize, decision_work_totals, METHODS


class UniqueControlsTests(unittest.TestCase):
    def test_aliases_and_injective_features_are_measured(self):
        for states in (16, 128):
            with self.subTest(states=states):
                aliased, _, _ = dataset('stable', 'aliased', states, 4, 0)
                unique, _, _ = dataset('stable', 'unique', states, 4, 0)
                self.assertLess(feature_counts(aliased)['unique_ordered_action_rows'], states)
                self.assertEqual(feature_counts(unique)['unique_ordered_action_rows'], states)
                self.assertEqual(feature_counts(unique)['unique_feature_vectors'], 2 * states)
                # The teacher is not uniformly biased toward one action.
                winners = {max(unique.actions, key=lambda a: unique.features[s, a][0]) for s in unique.states}
                self.assertEqual(winners, {0, 1})

    def test_paired_generation_and_fixed_feature_map(self):
        for regime in ('stable', 'fragile'):
            pa, la, ba = dataset(regime, 'aliased', 16, 4, 7)
            pu, lu, bu = dataset(regime, 'unique', 16, 4, 7)
            self.assertEqual(ba, bu)
            self.assertEqual([(x.state, x.action, x.next_state) for t in la for x in t.transitions],
                             [(x.state, x.action, x.next_state) for t in lu for x in t.transitions])
            if regime == 'fragile':
                self.assertEqual(la, lu)
            other, _, _ = dataset(regime, 'unique', 16, 4, 99)
            self.assertEqual(pu.features, other.features)
            self.assertNotEqual(pa.features, pu.features)

    def test_tiny_runner_checks_all_contracts_and_switches(self):
        row = run(('switch', 'one_hot', 2, 4), 0, 0, 2)
        self.assertGreater(row['requests'][0]['changed_final_actions'], 0)
        self.assertTrue(row['requests'][0]['policy_sequence_changed'])
        for request in row['requests']:
            self.assertEqual(set(request['methods']), set(METHODS))
            self.assertEqual(len({x['policy_sha256'] for x in request['methods'].values()}), 1)
            for method, result in request['methods'].items():
                self.assertTrue(result['policy_match'])
                if method.endswith('policy_only') or method == 'policy_certificate':
                    self.assertIsNone(result['full_trace_match'])
                else:
                    self.assertTrue(result['full_trace_match'])
        text = summarize(dict(runs=[row], methods=METHODS, git_commit='test', git_dirty=True))
        self.assertIn(f'{2 * len(METHODS)} method executions; 0 mismatches', text)
        self.assertIn('clean at launch: False', text)
        self.assertIn('Request-only decision work', text)
        self.assertIn('previous successfully retained dataset', text)
        shared = decision_work_totals([row], 'shared_anchor_repair')
        self.assertEqual(shared['margin_groups_built'], 0)
        self.assertGreater(shared['decisions_changed_from_anchor'], 0)

    def test_work_totals_exclude_preparation_and_preserve_missing_counters(self):
        def record(scores, decision=None):
            return dict(work=dict(action_scores=scores), decision_work=decision)
        decisions = dict(margin_groups_built=3, groups_certified=5, groups_scored=4,
                         decisions_changed_from_anchor=2)
        row = dict(preparation_results={'inverse_deduplicated': record(999),
                                        'selective_repair': record(999, decisions)},
                   requests=[dict(methods={'inverse_deduplicated': record(18),
                                           'selective_repair': record(8, decisions)})])
        baseline = decision_work_totals([row, row], 'inverse_deduplicated')
        self.assertEqual(baseline['action_scores'], 36)
        self.assertIsNone(baseline['groups_scored'])
        selective = decision_work_totals([row, row], 'selective_repair')
        self.assertEqual(selective['action_scores'], 16)
        self.assertEqual(selective['requests'], 2)
        self.assertEqual(selective['margin_groups_built'], 6)
        self.assertEqual(selective['groups_certified'], 10)
        self.assertEqual(selective['decisions_changed_from_anchor'], 4)

    def test_invalid_fixture_arguments(self):
        for requests in (0, 33):
            with self.assertRaises(ValueError):
                dataset('stable', 'unique', 16, 4, 0, requests)
        with self.assertRaises(ValueError):
            dataset('switch', 'unique', 2, 4, 0)
        with self.assertRaises(ValueError):
            dataset('stable', 'unknown', 16, 4, 0)


if __name__ == '__main__':
    unittest.main()
