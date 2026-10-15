"""Exact conventional inverse-update baseline against retained-only LSPI."""

from fractions import Fraction as F
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rl_unlearning import Trajectory, Transition
from rl_unlearning.lspi_control import LSPIProblem, SingularLSTDMatrix, scratch_lspi
from rl_unlearning.lspi_inverse import LSPIInverseState, LSPIInversePolicyState, inverse_matrix


def episode(name, *transitions):
    return Trajectory(name, tuple(Transition(*item) for item in transitions))


def switch_fixture(iterations=3):
    pairs = ((0, 0), (0, 1), (1, 0), (1, 1))
    features = {
        pair: tuple(F(i == j) for j in range(4))
        for i, pair in enumerate(pairs)
    }
    problem = LSPIProblem(
        (0, 1), (0, 1), features, {0: 0, 1: 0}, iterations=iterations
    )
    retained = (
        episode("xa", (0, 0, 0, 1), (1, 0, 0, None)),
        episode("xb", (0, 1, F(1, 16), None)),
        episode("zb", (1, 1, F(1, 4), None)),
    )
    return problem, retained + (episode("forget", (1, 0, 1, None)),), retained


class LSPIInverseTests(unittest.TestCase):
    def assertSameTrace(self, actual, expected):
        self.assertEqual(actual.iterates, expected.iterates)
        for field in ("policy", "weights", "status", "repeat_start", "repeat_length"):
            self.assertEqual(getattr(actual, field), getattr(expected, field), field)

    def test_inverse_matrix_is_one_exact_multi_rhs_elimination(self):
        matrix = ((F(2), F(1)), (F(3), F(2)))
        self.assertEqual(inverse_matrix(matrix), ((F(2), F(-1)), (F(-3), F(2))))
        self.assertEqual(inverse_matrix(((F(3, 2),),)), ((F(2, 3),),))
        with self.assertRaises(SingularLSTDMatrix):
            inverse_matrix(((1, 2), (2, 4)))
        with self.assertRaises(ValueError):
            inverse_matrix(((1, 2),))

    def test_switch_deletion_matches_each_matrix_and_iterate(self):
        problem, data, retained = switch_fixture()
        state = LSPIInverseState(problem, data)
        self.assertSameTrace(state.replay(), scratch_lspi(problem, data))
        self.assertEqual(state.preparation_samples, 5)
        self.assertIn((0, 0), state.cached_policies)

        result = state.delete_and_replay(("forget",))
        self.assertSameTrace(result, scratch_lspi(problem, retained))
        self.assertEqual(result.policy, (0, 1))
        self.assertEqual(result.iterates[0].policy_out, (1, 1))
        self.assertEqual(state.last_inverse_work["samples_read"], 1)
        self.assertGreater(state.last_inverse_work["rank_one_downdates"], 0)
        self.assertEqual(
            state.last_inverse_work["action_scores"],
            problem.iterations * len(problem.states) * len(problem.actions),
        )
        self.assertLessEqual(len(state.cached_policies), problem.iterations)

    def test_intermediate_singularity_can_disappear_after_next_downdate(self):
        # Initial A=1+1-1=1. Deleting the +1 terminal row gives singular
        # A=0; deleting the -1 bootstrap row restores the final A=1.
        problem = LSPIProblem(
            (0, 1), (0,), {(0, 0): (F(1),), (1, 0): (F(4),)},
            {0: 0, 1: 0}, gamma=F(1, 2), regularization=F(1),
            iterations=3,
        )
        data = (
            episode("positive", (0, 0, 1, None)),
            episode("negative", (0, 0, 0, 1)),
        )
        state = LSPIInverseState(problem, data)
        self.assertSameTrace(state.replay(), scratch_lspi(problem, data))
        result = state.delete_and_replay(("positive", "negative"))
        self.assertSameTrace(result, scratch_lspi(problem, ()))
        self.assertEqual(result.status, "fixed_point")
        self.assertEqual(state.last_inverse_work["intermediate_singular_fallbacks"], 1)
        self.assertEqual(state.last_inverse_work["inverse_builds"], 1)
        self.assertEqual(state.trajectory_ids, frozenset())

    def test_initial_and_later_singular_deletions_do_not_commit(self):
        retained = (episode("loop", (0, 1, 1, 0), (0, 1, 0, None)),)
        extra = episode("extra", (0, 1, 0, None))
        for initial in (0, 1):
            with self.subTest(initial=initial):
                problem = LSPIProblem(
                    (0,), (0, 1), {(0, 0): (F(5),), (0, 1): (F(1),)},
                    {0: initial},
                )
                state = LSPIInverseState(problem, retained + (extra,))
                original = state.replay()
                old_cache = dict(state._inverse_cache)
                failed = state.delete_and_replay(("extra",))
                self.assertSameTrace(failed, scratch_lspi(problem, retained))
                self.assertEqual(failed.status, "singular")
                self.assertEqual(state.trajectory_ids, frozenset(("loop", "extra")))
                self.assertEqual(state._inverse_cache, old_cache)
                self.assertSameTrace(state.replay(), original)
                self.assertSameTrace(state.delete_and_replay(("extra",)), failed)

    def test_cycles_keep_full_schedule_and_bound_cache(self):
        problem = LSPIProblem(
            (0,), (0, 1), {(0, 0): (F(3),), (0, 1): (F(1),)},
            {0: 0}, gamma=F(9, 10), iterations=5,
        )
        data = (episode("cycle", (0, 1, 1, 0), (0, 1, 0, None)),)
        state = LSPIInverseState(problem, data)
        result = state.replay()
        self.assertSameTrace(result, scratch_lspi(problem, data))
        self.assertEqual((result.status, result.repeat_length), ("cycle", 2))
        self.assertEqual(len(result.iterates), 5)
        self.assertLessEqual(len(state.cached_policies), 2)
        self.assertGreater(state.last_inverse_work["within_request_cache_hits"], 0)
        self.assertEqual(state.last_inverse_work["action_scores"], 10)

    def test_sequential_union_and_invalid_requests(self):
        problem, data, retained = switch_fixture()
        state = LSPIInverseState(problem, data)
        state.replay()
        before = dict(state._inverse_cache)
        with self.assertRaises(KeyError):
            state.delete_and_replay(("forget", "missing"))
        with self.assertRaises(ValueError):
            state.delete_and_replay(("forget", "forget"))
        self.assertEqual(state.trajectory_ids, frozenset(t.trajectory_id for t in data))
        self.assertEqual(state._inverse_cache, before)

        first = state.delete_and_replay(("forget",))
        self.assertSameTrace(first, scratch_lspi(problem, retained))
        second = state.delete_and_replay(("xb",))
        final = tuple(t for t in retained if t.trajectory_id != "xb")
        self.assertSameTrace(second, scratch_lspi(problem, final))
        union = LSPIInverseState(problem, data)
        union.replay()
        self.assertSameTrace(
            union.delete_and_replay(("forget", "xb")), second
        )
        with self.assertRaises(KeyError):
            state.delete_and_replay(("forget",))
        empty = state.delete_and_replay(t.trajectory_id for t in final)
        self.assertSameTrace(empty, scratch_lspi(problem, ()))
        self.assertEqual(state.cached_policies, frozenset(
            item.policy_in for item in empty.iterates
        ))

    def test_cached_inverses_are_exact_for_latest_retained_matrices(self):
        problem, data, _ = switch_fixture(iterations=4)
        state = LSPIInverseState(problem, data)
        state.replay()
        for ids in (("forget",), ("xb",), ("zb",)):
            result = state.delete_and_replay(ids)
            self.assertNotEqual(result.status, "singular")
            self.assertEqual(
                state.cached_policies,
                frozenset(item.policy_in for item in result.iterates),
            )
            for matrix, inverse in state._inverse_cache.values():
                n = len(matrix)
                for i in range(n):
                    for j in range(n):
                        product = sum(
                            (matrix[i][k] * inverse[k][j] for k in range(n)),
                            F(0),
                        )
                        self.assertEqual(product, F(i == j))

    def test_random_shared_features_sequential_full_traces(self):
        rng = random.Random(1884)
        accepted = downdated = 0
        for trial in range(20):
            dimension = 1 + trial % 4
            states, actions = (0, 1, 2), (0, 1)
            features = {
                (state, action): tuple(
                    F(rng.randrange(-2, 3)) for _ in range(dimension)
                )
                for state in states for action in actions
            }
            problem = LSPIProblem(
                states, actions, features, dict.fromkeys(states, trial % 2),
                gamma=F(2, 3), iterations=4,
            )
            data = []
            for i in range(9):
                state, action = rng.randrange(3), rng.randrange(2)
                successor, next_action = rng.randrange(3), rng.randrange(2)
                data.append(episode(
                    str(i),
                    (state, action, F(rng.randrange(-3, 4), 3), successor),
                    (successor, next_action, F(rng.randrange(-3, 4), 3), None),
                ))
            inverse_state = LSPIInverseState(problem, data)
            self.assertSameTrace(inverse_state.replay(), scratch_lspi(problem, data))
            for deleted in ("0", "1", "2"):
                candidate = tuple(
                    trajectory for trajectory in inverse_state.active_trajectories()
                    if trajectory.trajectory_id != deleted
                )
                result = inverse_state.delete_and_replay((deleted,))
                self.assertSameTrace(result, scratch_lspi(problem, candidate))
                if result.status == "singular":
                    self.assertIn(deleted, inverse_state.trajectory_ids)
                else:
                    accepted += 1
                    downdated += inverse_state.last_inverse_work["rank_one_downdates"]
                    self.assertNotIn(deleted, inverse_state.trajectory_ids)
                    self.assertLessEqual(
                        len(inverse_state.cached_policies), problem.iterations
                    )
        self.assertGreater(accepted, 40)
        self.assertGreater(downdated, 0)


class InversePolicyReuseTests(unittest.TestCase):
    def test_greedy_reuse_preserves_full_trace_and_changing_deletion(self):
        problem, logs, retained = switch_fixture(iterations=8)
        state = LSPIInversePolicyState(problem, logs)
        full = state.replay()
        self.assertEqual(full.iterates, scratch_lspi(problem, logs).iterates)
        self.assertEqual(full.work.action_scores, 4)
        result = state.delete_and_replay(('forget',))
        self.assertEqual(result.iterates, scratch_lspi(problem, retained).iterates)
        self.assertEqual(result.work.action_scores, 12)

    def test_cycle_greedy_reuse_scores_each_distinct_policy_once(self):
        problem = LSPIProblem((0,), (0, 1), {(0, 0): (3,), (0, 1): (1,)}, {0: 0},
                              gamma=F(9, 10), iterations=9)
        logs = (episode('cycle', (0, 1, 1, 0), (0, 1, 0, None)),)
        result = LSPIInversePolicyState(problem, logs).replay()
        self.assertEqual(result.iterates, scratch_lspi(problem, logs).iterates)
        self.assertEqual(result.work.action_scores, 4)
        self.assertEqual(result.status, 'cycle')


if __name__ == "__main__":
    unittest.main()
