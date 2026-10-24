"""Validate the multi-action fixture and exact retained-only comparison."""

from pathlib import Path
from fractions import Fraction as F
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_lspi_multiaction import CASES, METHODS, dataset, feature_counts, run, summarize


class MultiActionTests(unittest.TestCase):
    def test_features_are_unique_nonopposite_and_cover_three_winners(self):
        problem, _, _ = dataset("stable", 32, 8, 3, 0, 3)
        counts = feature_counts(problem)
        self.assertEqual(counts["unique_feature_vectors"], 32 * 3)
        self.assertEqual(counts["unique_ordered_action_rows"], 32)
        self.assertEqual(counts["opposite_action_pairs"], 0)
        teacher = tuple(
            1 if j == 0 else F(1 if j % 2 else -1, 16)
            for j in range(problem.dimension)
        )
        winners = {
            max(
                problem.actions,
                key=lambda action: sum(
                    feature * weight
                    for feature, weight in zip(problem.features[state, action], teacher)
                ),
            )
            for state in problem.states
        }
        self.assertEqual(winners, set(problem.actions))

    def test_smoke_run_matches_every_full_trace(self):
        row = run(CASES[0], 0, 0, 3)
        self.assertEqual(len(row["requests"]), 3)
        for request in row["requests"]:
            self.assertEqual(set(request["methods"]), set(METHODS))
            self.assertTrue(
                all(record["full_trace_match"] for record in request["methods"].values())
            )
        text = summarize(
            {
                "runs": [row],
                "methods": METHODS,
                "git_commit": "test",
                "git_dirty": True,
                "memory": [],
            }
        )
        self.assertIn("12 method executions; 0 full-trace mismatches", text)
        self.assertIn("three actions", text)

    def test_invalid_arguments(self):
        for arguments in (
            ("unknown", 32, 8, 3, 0, 3),
            ("stable", 2, 8, 3, 0, 3),
            ("stable", 32, 3, 3, 0, 3),
            ("stable", 32, 8, 2, 0, 3),
            ("stable", 32, 8, 3, 0, 0),
            ("stable", 32, 8, 3, 0, 41),
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                dataset(*arguments)


if __name__ == "__main__":
    unittest.main()
