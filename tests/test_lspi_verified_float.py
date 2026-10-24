from fractions import Fraction as F
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from rl_unlearning.lspi_control import LSPIProblem, scratch_lspi
from rl_unlearning.lspi_verified_float import (
    LSPIVerifiedFloatState,
    _float_inverse_and_solve,
    _verified_weight_bound,
)
from rl_unlearning.lspi_baselines import LSPICachedState
from rl_unlearning.lspi_control import _outer_add
from test_lspi_control import episode, switch_fixture
from run_lspi_multiaction import CASES
from run_lspi_verified_float import METHODS, run, summarize


class VerifiedFloatTests(unittest.TestCase):
    def compare(self, result, reference):
        self.assertEqual(result.policy, reference.policy)
        self.assertEqual(
            result.policy_sequence,
            tuple(item.policy_out for item in reference.iterates if item.policy_out is not None),
        )
        self.assertEqual(result.status, reference.status)
        self.assertEqual(
            (result.repeat_start, result.repeat_length),
            (reference.repeat_start, reference.repeat_length),
        )

    def test_well_conditioned_fast_path_certifies_policy_sequence(self):
        problem = LSPIProblem(
            (0, 1), (0, 1),
            {(0, 0): (1, 0), (0, 1): (-1, 0),
             (1, 0): (0, 1), (1, 1): (0, -1)},
            {0: 0, 1: 0}, gamma=F(1, 4), regularization=2, iterations=5,
        )
        logs = tuple(
            episode(str(i), (i % 2, 0, 2, None), ((i + 1) % 2, 1, -1, None))
            for i in range(12)
        )
        state = LSPIVerifiedFloatState(problem, logs)
        result = state.delete_and_replay(("0", "1"))
        self.compare(result, scratch_lspi(problem, logs[2:]))
        self.assertTrue(result.certified)
        self.assertFalse(result.used_fallback)
        self.assertEqual(len(result.certificates), problem.iterations)
        self.assertTrue(all(item.inverse_defect < 1 for item in result.certificates))

    def test_inverse_defect_bound_contains_exact_solution(self):
        matrix = ((F(7, 3), F(-2, 5)), (F(1, 7), F(11, 4)))
        rhs = (F(5, 6), F(-9, 8))
        inverse, approximate = _float_inverse_and_solve(matrix, rhs)
        weights, eta, _, error = _verified_weight_bound(
            matrix, rhs, inverse, approximate
        )
        determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        exact = (
            (rhs[0] * matrix[1][1] - matrix[0][1] * rhs[1]) / determinant,
            (matrix[0][0] * rhs[1] - rhs[0] * matrix[1][0]) / determinant,
        )
        self.assertLess(eta, 1)
        self.assertLessEqual(max(abs(x - y) for x, y in zip(weights, exact)), error)

    def test_small_margin_uses_exact_fallback(self):
        tiny = F(1, 10**18)
        problem = LSPIProblem(
            (0,), (0, 1), {(0, 0): (1,), (0, 1): (1 - tiny,)},
            {0: 0}, regularization=1, iterations=4,
        )
        # A true tie between nonidentical feature rows cannot be separated by
        # any positive numerical error bound, so the exact tie-order fallback
        # is mandatory.
        logs = (episode("keep", (0, 0, 0, None)), episode("forget", (0, 0, 0, None)))
        state = LSPIVerifiedFloatState(problem, logs)
        result = state.delete_and_replay(("forget",))
        self.compare(result, scratch_lspi(problem, logs[:1]))
        self.assertTrue(result.used_fallback)
        self.assertFalse(result.certified)

    def test_adversarial_binary64_scoring_error_is_certified_away(self):
        features = {
            (0, 0): (F(-1, 10**6), F(-1000)),
            (0, 1): (F(1, 5 * 10**11), F(-1000)),
            (1, 0): (F(10**9), F(-1, 5 * 10**11)),
            (1, 1): (F(-1, 10**6), F(-1, 5 * 10**11)),
        }
        problem = LSPIProblem(
            (0, 1), (0, 1), features, {0: 0, 1: 0},
            gamma=F(999999, 1000000), regularization=F(1, 10**12), iterations=3,
        )
        raw = (
            ((1, 0, F(1, 10**6), 1), (1, 1, F(1, 1000), None)),
            ((0, 1, F(1, 10**9), 0), (0, 0, F(-1, 500000), None)),
            ((0, 0, F(-2000), 0), (0, 1, F(-1, 10**9), None)),
            ((1, 1, F(1, 10**12), 0), (0, 1, F(1, 10**9), None)),
            ((1, 0, F(1, 10**6), 1), (1, 0, F(-1, 5 * 10**11), None)),
        )
        logs = tuple(episode(str(i), *transitions) for i, transitions in enumerate(raw))
        reference = scratch_lspi(problem, logs)

        # Deliberately use ordinary binary64 dot products without the exact
        # score/error certificate.  Catastrophic scale imbalance silently
        # selects the wrong first action.
        cached = LSPICachedState(problem, logs)
        policy = tuple(problem.initial_policy[s] for s in problem.states)
        naive_sequence = []
        for _ in range(problem.iterations):
            matrix = [list(row) for row in cached._covariance]
            choices = dict(zip(problem.states, policy))
            for state, group in cached._successors.items():
                _outer_add(matrix, group, problem.features[(state, choices[state])],
                           -problem.gamma)
            _, weights = _float_inverse_and_solve(matrix, cached._rhs)
            policy = tuple(
                problem.actions[max(
                    range(len(problem.actions)),
                    key=lambda i: sum(
                        float(x) * y for x, y in
                        zip(problem.features[(state, problem.actions[i])], weights)
                    ),
                )]
                for state in problem.states
            )
            naive_sequence.append(policy)

        exact_sequence = tuple(item.policy_out for item in reference.iterates)
        self.assertNotEqual(tuple(naive_sequence), exact_sequence)
        verified = LSPIVerifiedFloatState(problem, logs).replay()
        self.compare(verified, reference)
        self.assertTrue(verified.certified)
        self.assertFalse(verified.used_fallback)

    def test_policy_switch_and_random_sequential_requests_never_false_certify(self):
        problem, logs, retained = switch_fixture(iterations=6)
        result = LSPIVerifiedFloatState(problem, logs).delete_and_replay(("forget",))
        self.compare(result, scratch_lspi(problem, retained))

        rng = random.Random(4127)
        certified = fallback = 0
        for trial in range(20):
            d = 2 + trial % 3
            features = {
                (state, action): tuple(F(rng.randrange(-4, 5)) for _ in range(d))
                for state in range(3) for action in range(3)
            }
            if any(not any(row) for row in features.values()):
                continue
            candidate_problem = LSPIProblem(
                (0, 1, 2), (0, 1, 2), features, dict.fromkeys(range(3), 0),
                gamma=F(1, 4), regularization=3, iterations=5,
            )
            candidate_logs = [
                episode(
                    str(i),
                    (rng.randrange(3), rng.randrange(3), rng.randrange(-3, 4), rng.randrange(3)),
                    (rng.randrange(3), rng.randrange(3), rng.randrange(-3, 4), None),
                )
                for i in range(18)
            ]
            reference = scratch_lspi(candidate_problem, candidate_logs)
            if reference.status == "singular":
                continue
            state = LSPIVerifiedFloatState(candidate_problem, candidate_logs)
            for identifier in range(3):
                candidate_logs = [
                    trajectory for trajectory in candidate_logs
                    if trajectory.trajectory_id != str(identifier)
                ]
                result = state.delete_and_replay((str(identifier),))
                self.compare(result, scratch_lspi(candidate_problem, candidate_logs))
                certified += result.certified
                fallback += result.used_fallback
        self.assertGreater(certified, 0)
        self.assertGreaterEqual(fallback, 0)

    def test_invalid_and_singular_requests_are_atomic(self):
        problem = LSPIProblem(
            (0,), (0,), {(0, 0): (1,)}, {0: 0},
            gamma=0, regularization=0, iterations=2,
        )
        logs = (episode("keep", (0, 0, 1, None)),)
        state = LSPIVerifiedFloatState(problem, logs)
        for ids, error in ((('missing',), KeyError), (('keep', 'keep'), ValueError)):
            with self.assertRaises(error):
                state.delete_and_replay(ids)
        singular = state.delete_and_replay(("keep",))
        self.assertEqual(singular.status, "singular")
        self.assertEqual(state.trajectory_ids, frozenset(("keep",)))

    def test_runner_matches_policy_contract_and_reports_certification(self):
        row = run(CASES[0], 0, 0, 2)
        self.assertEqual(len(row["requests"]), 2)
        for request in row["requests"]:
            self.assertEqual(set(request["methods"]), set(METHODS))
            self.assertTrue(all(
                item["policy_sequence_match"]
                for item in request["methods"].values()
            ))
            self.assertTrue(request["methods"]["verified_float"]["certified"])
        report = {
            "runs": [row], "git_commit": "test", "git_dirty": True,
            "methods": METHODS,
        }
        text = summarize(report)
        self.assertIn("6 executions; 0 mismatches", text)
        self.assertIn("2 / 2", text)


if __name__ == "__main__":
    unittest.main()
