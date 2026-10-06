from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rl_unlearning import Trajectory, Transition
from rl_unlearning.lspi_control import LSPIProblem, LSPIReplayState, scratch_lspi


def episode(name, *transitions):
    return Trajectory(name, tuple(Transition(*t) for t in transitions))


def switch_fixture(iterations=3):
    # Feature coordinates: (x,a), (x,b), (z,a), (z,b).
    pairs = ((0, 0), (0, 1), (1, 0), (1, 1))
    features = {pair: tuple(F(i == j) for j in range(4)) for i, pair in enumerate(pairs)}
    problem = LSPIProblem((0, 1), (0, 1), features, {0: 0, 1: 0}, iterations=iterations)
    retained = (episode("xa", (0, 0, 0, 1), (1, 0, 0, None)),
                episode("xb", (0, 1, F(1, 16), None)),
                episode("zb", (1, 1, F(1, 4), None)))
    forgotten = episode("forget", (1, 0, 1, None))
    return problem, retained + (forgotten,), retained


class LSPIControlTests(unittest.TestCase):
    def assertReplayEqual(self, first, second):
        self.assertEqual(first.iterates, second.iterates)
        for field in ("policy", "weights", "status", "repeat_start", "repeat_length"):
            self.assertEqual(getattr(first, field), getattr(second, field), field)

    def test_one_shot_downdate_is_not_control_replay(self):
        problem, data, retained = switch_fixture()
        full = scratch_lspi(problem, data)
        self.assertEqual(full.iterates[0].weights, (F(2, 15), F(1, 24), F(2, 5), F(1, 6)))
        self.assertEqual(full.policy, (0, 0))
        state = LSPIReplayState(problem, data)
        replay = state.delete_and_replay(("forget",))
        self.assertReplayEqual(replay, scratch_lspi(problem, retained))
        first, second = replay.iterates[:2]
        self.assertEqual(first.weights, (F(0), F(1, 24), F(0), F(1, 6)))
        self.assertEqual(first.policy_out, (1, 1))
        self.assertEqual(second.weights, (F(1, 18), F(1, 24), F(0), F(1, 6)))
        self.assertEqual(replay.policy, (0, 1))
        self.assertNotEqual(first.policy_out, replay.policy)
        self.assertEqual(second.matrix[0], (F(3, 2), F(0), F(0), F(-1, 2)))
        self.assertEqual(replay.work.target_rank_one_updates, 1)
        self.assertEqual(replay.work.samples_read, 1)
        self.assertEqual(state.preparation_samples, 5)

    def test_finite_budget_returns_declared_iteration_not_arbitrary_fixed_point(self):
        problem, data, retained = switch_fixture(iterations=1)
        replay = LSPIReplayState(problem, data).delete_and_replay(("forget",))
        self.assertReplayEqual(replay, scratch_lspi(problem, retained))
        self.assertEqual(replay.policy, (1, 1))
        self.assertEqual(replay.status, "budget_exhausted")
        self.assertEqual(len(replay.iterates), 1)
        longer = scratch_lspi(replace(problem, iterations=3), retained)
        self.assertNotEqual(replay.policy, longer.policy)

    def test_exact_ties_follow_declared_action_order_and_continue_budget(self):
        problem = LSPIProblem((0,), (1, 0), {(0, 0): (1, 0), (0, 1): (0, 1)}, {0: 0},
                              iterations=4)
        data = (episode("a", (0, 0, 1, None)), episode("b", (0, 1, 1, None)))
        result = LSPIReplayState(problem, data).replay()
        self.assertReplayEqual(result, scratch_lspi(problem, data))
        self.assertEqual(result.iterates[0].weights, (F(2, 3), F(2, 3)))
        self.assertEqual(result.policy, (1,))
        self.assertEqual(result.status, "fixed_point")
        self.assertEqual((result.repeat_start, result.repeat_length), (1, 1))
        self.assertEqual(len(result.iterates), 4)

    def test_terminal_mask_removes_all_policy_dependence_from_target(self):
        problem = LSPIProblem((0,), (0, 1), {(0, 0): (1,), (0, 1): (100,)}, {0: 1})
        data = (episode("terminal", (0, 0, 1, None)),)
        result = LSPIReplayState(problem, data).replay()
        self.assertReplayEqual(result, scratch_lspi(problem, data))
        self.assertEqual(result.weights, (F(2, 3),))
        self.assertEqual(result.iterates[0].matrix, ((F(3, 2),),))
        self.assertEqual(result.work.target_rank_one_updates, 0)
        self.assertEqual(result.work.successor_entries_checked, 0)

    def test_shared_feature_policy_cycle_is_reported_but_schedule_finishes(self):
        problem = LSPIProblem((0,), (0, 1), {(0, 0): (3,), (0, 1): (1,)}, {0: 0},
                              gamma=F(9, 10), iterations=5)
        data = (episode("cycle", (0, 1, 1, 0), (0, 1, 0, None)),)
        result = LSPIReplayState(problem, data).replay()
        self.assertReplayEqual(result, scratch_lspi(problem, data))
        self.assertEqual([x.weights for x in result.iterates],
                         [(F(-5),), (F(5, 8),), (F(-5),), (F(5, 8),), (F(-5),)])
        self.assertEqual((result.status, result.repeat_start, result.repeat_length), ("cycle", 0, 2))
        self.assertEqual(result.policy, (1,))
        self.assertEqual(len(result.iterates), 5)

    def test_positive_regularization_does_not_prevent_singularity(self):
        problem = LSPIProblem((0,), (0, 1), {(0, 0): (5,), (0, 1): (1,)}, {0: 0})
        retained = (episode("singular", (0, 1, 1, 0), (0, 1, 0, None)),)
        extra = episode("extra", (0, 1, 0, None))
        state = LSPIReplayState(problem, retained + (extra,))
        original = state.replay()
        result = state.delete_and_replay(("extra",))
        self.assertReplayEqual(result, scratch_lspi(problem, retained))
        self.assertEqual(result.status, "singular")
        self.assertEqual(result.iterates[0].matrix, ((F(0),),))
        self.assertIsNone(result.weights)
        self.assertIsNone(result.iterates[0].policy_out)
        self.assertEqual(state.trajectory_ids, frozenset(("singular", "extra")))
        self.assertReplayEqual(state.replay(), original)

    def test_noop_sequential_and_union_deletions_match_retained_replay(self):
        problem, data, retained = switch_fixture()
        state = LSPIReplayState(problem, data)
        self.assertReplayEqual(state.delete_and_replay(()), scratch_lspi(problem, data))
        state.delete_and_replay(("forget",))
        result = state.delete_and_replay(("xb",))
        final = tuple(t for t in retained if t.trajectory_id != "xb")
        self.assertReplayEqual(result, scratch_lspi(problem, final))
        union = LSPIReplayState(problem, data).delete_and_replay(("forget", "xb"))
        self.assertReplayEqual(result, union)
        self.assertEqual(result.iterates[0].policy_in, (0, 0))
        self.assertEqual(state.trajectory_ids, frozenset(t.trajectory_id for t in final))

    def test_all_deleted_has_exact_zero_critic_and_tie_policy(self):
        problem, data, _ = switch_fixture()
        result = LSPIReplayState(problem, data).delete_and_replay(t.trajectory_id for t in data)
        self.assertReplayEqual(result, scratch_lspi(problem, ()))
        self.assertEqual(result.weights, (F(0),) * 4)
        self.assertEqual(result.policy, (0, 0))

    def test_invalid_requests_and_feature_mutation_cannot_change_state(self):
        problem, data, _ = switch_fixture()
        state = LSPIReplayState(problem, data)
        before = state.replay()
        for ids, error in ((("forget", "missing"), KeyError), (("forget", "forget"), ValueError)):
            with self.assertRaises(error):
                state.delete_and_replay(ids)
            self.assertReplayEqual(state.replay(), before)
        with self.assertRaises(TypeError):
            problem.features[(0, 0)] = (1,)
        state.delete_and_replay(("forget",))
        with self.assertRaises(KeyError):
            state.delete_and_replay(("forget",))
        with self.assertRaises(ValueError):
            scratch_lspi(problem, (data[0], data[0]))
        with self.assertRaises(ValueError):
            scratch_lspi(problem, (episode("bad", (0, 0, float("nan"), None)),))

    def test_random_shared_feature_deletions_match_every_matrix_and_iterate(self):
        rng = random.Random(431)
        accepted = 0
        for trial in range(25):
            states, actions = (0, 1, 2), (0, 1)
            features = {(s, a): tuple(F(rng.randrange(-2, 3)) for _ in range(3))
                        for s in states for a in actions}
            problem = LSPIProblem(states, actions, features, dict.fromkeys(states, 0),
                                  gamma=F(2, 3), iterations=4)
            data = []
            for i in range(9):
                s, a, z, b = rng.randrange(3), rng.randrange(2), rng.randrange(3), rng.randrange(2)
                data.append(episode(str(i), (s, a, F(rng.randrange(-3, 4), 3), z),
                                    (z, b, F(rng.randrange(-3, 4), 3), None)))
            state = LSPIReplayState(problem, data)
            self.assertReplayEqual(state.replay(), scratch_lspi(problem, data))
            for deleted in ("0", "1", "2"):
                candidate = tuple(t for t in state.active_trajectories() if t.trajectory_id != deleted)
                result = state.delete_and_replay((deleted,))
                self.assertReplayEqual(result, scratch_lspi(problem, candidate))
                if result.status != "singular":
                    accepted += 1
        self.assertGreater(accepted, 60)


if __name__ == "__main__":
    unittest.main()
