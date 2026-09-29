from fractions import Fraction
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rl_unlearning import EmpiricalMDP, Trajectory, Transition, value_iteration
from rl_unlearning.baselines import prioritized_sweeping
from rl_unlearning.experiments import CASES, MeteredMDP, run_case
from rl_unlearning.reference import exact_reference


class ReferenceTests(unittest.TestCase):
    def test_exact_cycle_and_tie_order(self):
        data = (Trajectory("cycle", (Transition(0, 1, 0.5, 0),)),
                Trajectory("exit", (Transition(0, 0, 1.0),)))
        q, policy = exact_reference((0,), (1, 0), 0.5, data)
        self.assertEqual(q[(0, 1)], Fraction(1))
        self.assertEqual(q[(0, 0)], Fraction(1))
        self.assertEqual(policy, {0: 1})

    def test_random_models_against_independent_oracle(self):
        rng = random.Random(91)
        for _ in range(20):
            data = tuple(Trajectory(str(i), (Transition(rng.randrange(3), rng.randrange(2),
                             rng.randrange(-5, 6) / 10, rng.choice((None, 0, 1, 2))),))
                         for i in range(25))
            q, policy = exact_reference(range(3), (0, 1), 0.7, data)
            result = value_iteration(EmpiricalMDP(range(3), (0, 1), 0.7, data), tolerance=1e-12)
            for row in q:
                self.assertAlmostEqual(result.q[row], float(q[row]), places=10)
            # Exact ties need not match a finite-tolerance iterative solver.
            for state in policy:
                if q[(state, 0)] != q[(state, 1)]:
                    self.assertEqual(result.policy[state], policy[state])

    def test_oracle_refuses_large_enumeration(self):
        with self.assertRaises(ValueError):
            exact_reference(range(20), (0, 1), 0.9, ())


class ExperimentTests(unittest.TestCase):
    def test_counters_include_verification(self):
        model = MeteredMDP((0,), (0, 1), 0.9,
                           (Trajectory("a", (Transition(0, 0, 1.0),)),))
        result = value_iteration(model)
        self.assertEqual(model.work.bellman_rows, result.backups + result.residual_checks)
        self.assertEqual(model.work.transition_entries_read, model.work.bellman_rows)
        self.assertEqual(model.work.log_transitions_applied, 1)

    def test_baseline_cases_match_exact_reference(self):
        for case in CASES:
            result = run_case(case, 0)
            for row in result["records"]:
                if row["method"] != "noop":
                    self.assertEqual(row["policy_disagreement"], 0, (case, row))
                    self.assertTrue(row["converged"])
                    self.assertLess(row["max_q_error"], 1e-8)
            if case == "tie":
                self.assertFalse(result["records"][-1]["certified_optimal_policy"])

    def test_sweeping_budget_exhaustion_is_visible(self):
        model = EmpiricalMDP((0,), (0, 1), 0.9,
                             (Trajectory("a", (Transition(0, 0, 1.0),)),))
        result = prioritized_sweeping(model, {(0, 0): 0.0, (0, 1): 0.0},
                                      {(0, 0)}, max_backups=0)
        self.assertFalse(result.converged)
        self.assertEqual(result.residual, 1.0)


if __name__ == "__main__":
    unittest.main()
