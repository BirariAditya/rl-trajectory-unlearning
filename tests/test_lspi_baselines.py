from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rl_unlearning import Trajectory, Transition
from rl_unlearning.lspi_baselines import LSPICachedState
from rl_unlearning.lspi_control import LSPIProblem, LSPIReplayState, scratch_lspi


def episode(name, *transitions):
    return Trajectory(name, tuple(Transition(*t) for t in transitions))


def switch_fixture(iterations=3):
    pairs = ((0, 0), (0, 1), (1, 0), (1, 1))
    features = {pair: tuple(F(i == j) for j in range(4)) for i, pair in enumerate(pairs)}
    problem = LSPIProblem((0, 1), (0, 1), features, {0: 0, 1: 0}, iterations=iterations)
    retained = (episode("xa", (0, 0, 0, 1), (1, 0, 0, None)),
                episode("xb", (0, 1, F(1, 16), None)),
                episode("zb", (1, 1, F(1, 4), None)))
    return problem, retained + (episode("forget", (1, 0, 1, None)),), retained


class LSPICachedBaselineTests(unittest.TestCase):
    def assertSameLearner(self, first, second):
        self.assertEqual(first.iterates, second.iterates)
        for field in ("policy", "weights", "status", "repeat_start", "repeat_length"):
            self.assertEqual(getattr(first, field), getattr(second, field), field)

    def assertThreeWays(self, problem, data, cached):
        self.assertSameLearner(cached, scratch_lspi(problem, data))
        self.assertSameLearner(cached, LSPIReplayState(problem, data).replay())

    def test_reconstructs_all_terms_even_when_policy_is_unchanged(self):
        problem, data, _ = switch_fixture()
        state = LSPICachedState(problem, data)
        result = state.replay()
        self.assertThreeWays(problem, data, result)
        self.assertEqual(state.preparation_samples, 5)
        self.assertEqual(result.policy, (0, 0))
        self.assertEqual(result.work.samples_read, 0)
        self.assertEqual(result.work.target_rank_one_updates, problem.iterations)
        self.assertEqual(result.work.successor_entries_checked, problem.iterations)
        self.assertEqual(LSPIReplayState(problem, data).replay().work.target_rank_one_updates, 0)

    def test_deleted_policy_switch_matches_every_retained_iteration(self):
        problem, data, retained = switch_fixture()
        state = LSPICachedState(problem, data)
        result = state.delete_and_replay(("forget",))
        self.assertThreeWays(problem, retained, result)
        correction = LSPIReplayState(problem, data).delete_and_replay(("forget",))
        self.assertSameLearner(result, correction)
        self.assertEqual(result.iterates[0].policy_out, (1, 1))
        self.assertEqual(result.policy, (0, 1))
        self.assertEqual(result.iterates[1].weights, (F(1, 18), F(1, 24), F(0), F(1, 6)))
        self.assertEqual(result.work.target_rank_one_updates, 3)
        self.assertEqual(correction.work.target_rank_one_updates, 1)
        self.assertEqual(result.work.samples_read, 1)
        self.assertEqual(scratch_lspi(problem, retained).work.samples_read, 12)
        self.assertEqual(result.work.linear_solves, 3)
        self.assertEqual(result.work.action_scores, 12)

    def test_noop_sequential_union_and_all_deleted_match_scratch(self):
        problem, data, _ = switch_fixture()
        state = LSPICachedState(problem, data)
        self.assertThreeWays(problem, data, state.delete_and_replay(()))
        state.delete_and_replay(("forget",))
        result = state.delete_and_replay(("xb",))
        retained = tuple(t for t in data if t.trajectory_id not in {"forget", "xb"})
        self.assertThreeWays(problem, retained, result)
        union = LSPICachedState(problem, data).delete_and_replay(("forget", "xb"))
        self.assertSameLearner(result, union)
        self.assertEqual(state.trajectory_ids, frozenset(t.trajectory_id for t in retained))
        empty = state.delete_and_replay(t.trajectory_id for t in retained)
        self.assertThreeWays(problem, (), empty)
        self.assertEqual(empty.weights, (F(0),) * 4)
        self.assertEqual(empty.work.successor_entries_checked, 0)
        self.assertEqual(state.active_trajectories(), ())

    def test_terminal_mask_and_exact_ties_follow_declared_order(self):
        problem = LSPIProblem((0,), (1, 0), {(0, 0): (1, 0), (0, 1): (0, 1)},
                              {0: 0}, iterations=4)
        data = (episode("a", (0, 0, 1, None)), episode("b", (0, 1, 1, None)))
        result = LSPICachedState(problem, data).replay()
        self.assertThreeWays(problem, data, result)
        self.assertEqual(result.policy, (1,))
        self.assertEqual(result.weights, (F(2, 3), F(2, 3)))
        self.assertEqual(result.work.successor_entries_checked, 0)
        self.assertEqual(len(result.iterates), 4)

    def test_cancelled_successor_group_is_restored_by_deletion(self):
        problem = LSPIProblem((0,), (0, 1), {(0, 0): (1,), (0, 1): (-1,)}, {0: 0})
        data = (episode("positive", (0, 0, 1, 0), (0, 0, 0, None)),
                episode("negative", (0, 1, 1, 0), (0, 1, 0, None)))
        state = LSPICachedState(problem, data)
        full = state.replay()
        self.assertThreeWays(problem, data, full)
        self.assertEqual(full.work.successor_entries_checked, 0)
        result = state.delete_and_replay(("negative",))
        self.assertThreeWays(problem, data[:1], result)
        self.assertEqual(result.work.successor_entries_checked, 3)
        self.assertEqual(result.iterates[0].matrix, ((F(2),),))
        self.assertEqual(result.weights, (F(1, 2),))

    def test_shared_feature_cycle_and_budget_match_reference(self):
        problem = LSPIProblem((0,), (0, 1), {(0, 0): (3,), (0, 1): (1,)}, {0: 0},
                              gamma=F(9, 10), iterations=5)
        data = (episode("cycle", (0, 1, 1, 0), (0, 1, 0, None)),)
        result = LSPICachedState(problem, data).replay()
        self.assertThreeWays(problem, data, result)
        self.assertEqual((result.status, result.repeat_length), ("cycle", 2))
        self.assertEqual(result.policy, (1,))
        self.assertEqual(result.work.target_rank_one_updates, 5)
        shorter = replace(problem, iterations=1)
        single = LSPICachedState(shorter, data).replay()
        self.assertThreeWays(shorter, data, single)
        self.assertEqual(single.status, "budget_exhausted")

    def test_initial_and_later_singular_deletions_are_atomic(self):
        retained = (episode("loop", (0, 1, 1, 0), (0, 1, 0, None)),)
        extra = episode("extra", (0, 1, 0, None))
        for initial in (0, 1):
            with self.subTest(initial=initial):
                problem = LSPIProblem((0,), (0, 1), {(0, 0): (5,), (0, 1): (1,)}, {0: initial})
                state = LSPICachedState(problem, retained + (extra,))
                original = state.replay()
                result = state.delete_and_replay(("extra",))
                self.assertThreeWays(problem, retained, result)
                self.assertEqual(result.status, "singular")
                self.assertEqual(len(result.iterates), initial + 1)
                self.assertEqual(state.trajectory_ids, frozenset(("loop", "extra")))
                self.assertSameLearner(state.replay(), original)
                # Repeating a failed request must not subtract its sample twice.
                self.assertSameLearner(state.delete_and_replay(("extra",)), result)

    def test_zero_ridge_empty_deletion_fails_without_committing(self):
        problem = LSPIProblem((0,), (0,), {(0, 0): (1,)}, {0: 0}, regularization=0)
        data = (episode("only", (0, 0, 1, None)),)
        state = LSPICachedState(problem, data)
        result = state.delete_and_replay(("only",))
        self.assertThreeWays(problem, (), result)
        self.assertEqual(result.status, "singular")
        self.assertEqual(state.trajectory_ids, frozenset(("only",)))

    def test_invalid_requests_are_atomic(self):
        problem, data, _ = switch_fixture()
        state = LSPICachedState(problem, data)
        original = state.replay()
        for ids, error in ((("forget", "unknown"), KeyError), (("forget", "forget"), ValueError)):
            with self.assertRaises(error):
                state.delete_and_replay(ids)
            self.assertSameLearner(state.replay(), original)
        state.delete_and_replay(("forget",))
        with self.assertRaises(KeyError):
            state.delete_and_replay(("forget",))
        with self.assertRaises(ValueError):
            LSPICachedState(problem, (data[0], data[0]))
        with self.assertRaises(ValueError):
            LSPICachedState(problem, (episode("bad", (0, 0, 1, 20)),))

    def test_exact_reconstruction_is_independent_of_log_order(self):
        problem, data, retained = switch_fixture()
        result = LSPICachedState(problem, tuple(reversed(data))).delete_and_replay(("forget",))
        self.assertThreeWays(problem, retained, result)

    def test_random_shared_features_and_sequential_deletions_match_every_iterate(self):
        rng = random.Random(614)
        successes = 0
        for trial in range(24):
            dimension = 1 + trial % 4
            states, actions = (0, 1, 2), (0, 1)
            features = {(s, a): tuple(F(rng.randrange(-2, 3)) for _ in range(dimension))
                        for s in states for a in actions}
            problem = LSPIProblem(states, actions, features, dict.fromkeys(states, trial % 2),
                                  gamma=F(2, 3), iterations=4)
            data = []
            for i in range(10):
                s, a, z, b = rng.randrange(3), rng.randrange(2), rng.randrange(3), rng.randrange(2)
                data.append(episode(str(i), (s, a, F(rng.randrange(-3, 4), 3), z),
                                    (z, b, F(rng.randrange(-3, 4), 3), None)))
            cached, replay = LSPICachedState(problem, data), LSPIReplayState(problem, data)
            self.assertSameLearner(cached.replay(), scratch_lspi(problem, data))
            for deleted in ("0", "1", "2"):
                retained = tuple(t for t in cached.active_trajectories() if t.trajectory_id != deleted)
                result = cached.delete_and_replay((deleted,))
                self.assertSameLearner(result, scratch_lspi(problem, retained))
                self.assertSameLearner(result, replay.delete_and_replay((deleted,)))
                self.assertEqual(cached.trajectory_ids, replay.trajectory_ids)
                successes += result.status != "singular"
        self.assertGreater(successes, 60)


if __name__ == "__main__":
    unittest.main()
