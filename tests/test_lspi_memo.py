from pathlib import Path
import sys
import unittest
import random

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rl_unlearning.lspi_control import LSPIProblem, LSPIReplayMemoizedState, scratch_lspi
from rl_unlearning.lspi_baselines import LSPIMemoizedState
from test_lspi_control import episode, switch_fixture
from fractions import Fraction as F


class MemoTests(unittest.TestCase):
    state_class = LSPIMemoizedState

    def compare(self, a, b):
        self.assertEqual(a.iterates, b.iterates)
        self.assertEqual((a.policy, a.weights, a.status), (b.policy, b.weights, b.status))
        self.assertEqual((a.repeat_start, a.repeat_length), (b.repeat_start, b.repeat_length))

    def test_deletion_resets_memo_and_fixed_point_reuses_solves(self):
        problem, logs, retained = switch_fixture(iterations=6)
        state = self.state_class(problem, logs)
        full = state.replay()
        self.compare(full, scratch_lspi(problem, logs))
        self.assertEqual(full.work.linear_solves, 1)
        after = state.delete_and_replay(["forget"])
        self.compare(after, scratch_lspi(problem, retained))
        self.assertEqual(after.work.linear_solves, 3)
        self.assertEqual(len(after.iterates), 6)

    def test_cycle_reuses_each_distinct_policy(self):
        problem = LSPIProblem((0,), (0, 1), {(0, 0): (3,), (0, 1): (1,)}, {0: 0},
                              gamma=F(9, 10), iterations=7)
        logs = (episode("cycle", (0, 1, 1, 0), (0, 1, 0, None)),)
        result = self.state_class(problem, logs).replay()
        self.compare(result, scratch_lspi(problem, logs))
        self.assertEqual(result.status, "cycle")
        self.assertEqual(result.work.linear_solves, 2)

    def test_singularity_after_switch_remains_atomic(self):
        problem = LSPIProblem((0,), (0, 1), {(0, 0): (5,), (0, 1): (1,)}, {0: 1})
        logs = (episode("loop", (0, 1, 1, 0), (0, 1, 0, None)),
                episode("extra", (0, 1, 0, None)))
        state = self.state_class(problem, logs)
        before = state.replay()
        result = state.delete_and_replay(["extra"])
        self.compare(result, scratch_lspi(problem, logs[:1]))
        self.assertEqual(result.status, "singular")
        self.assertEqual(state.trajectory_ids, frozenset(("loop", "extra")))
        self.compare(state.replay(), before)


class ReplayMemoTests(MemoTests):
    state_class = LSPIReplayMemoizedState

    def test_random_sequential_requests_match_both_scratch_and_cached_memo(self):
        rng = random.Random(814)
        successes = 0
        for trial in range(32):
            dimension = 1 + trial % 4
            features = {(s, a): tuple(F(rng.randrange(-2, 3)) for _ in range(dimension))
                        for s in range(3) for a in range(2)}
            problem = LSPIProblem((0, 1, 2), (0, 1), features, dict.fromkeys(range(3), trial % 2),
                                  gamma=F(2, 3), iterations=8)
            logs = []
            for i in range(10):
                s, z = rng.randrange(3), rng.randrange(3)
                logs.append(episode(str(i), (s, rng.randrange(2), F(rng.randrange(-3, 4), 3), z),
                                    (z, rng.randrange(2), F(rng.randrange(-3, 4), 3), None)))
            replay = self.state_class(problem, logs)
            cached = LSPIMemoizedState(problem, logs)
            for request in ((), ("0", "1"), ("2",), ("3", "4")):
                with self.subTest(trial=trial, request=request):
                    before = replay.replay()
                    retained = tuple(t for t in logs if t.trajectory_id not in request)
                    actual = replay.delete_and_replay(request)
                    control = cached.delete_and_replay(request)
                    self.compare(actual, scratch_lspi(problem, retained))
                    self.compare(actual, control)
                    self.assertEqual(actual.work.linear_solves, control.work.linear_solves)
                    self.assertEqual(actual.work.action_scores, control.work.action_scores)
                    self.assertEqual(actual.work.linear_solves,
                                     len({step.policy_in for step in actual.iterates}))
                    if actual.status == "singular":
                        self.compare(replay.replay(), before)
                    else:
                        successes += 1
                        logs = list(retained)
                    self.assertEqual(replay.trajectory_ids, frozenset(t.trajectory_id for t in logs))
        self.assertGreater(successes, 80)

    def test_cancelled_group_and_delete_all_preserve_trace(self):
        problem = LSPIProblem((0,), (0, 1), {(0, 0): (1,), (0, 1): (-1,)}, {0: 0}, iterations=6)
        logs = (episode("positive", (0, 0, 1, 0), (0, 0, 0, None)),
                episode("negative", (0, 1, 1, 0), (0, 1, 0, None)))
        state = self.state_class(problem, logs)
        self.compare(state.replay(), scratch_lspi(problem, logs))
        self.compare(state.delete_and_replay(("negative",)), scratch_lspi(problem, logs[:1]))
        self.compare(state.delete_and_replay(("positive",)), scratch_lspi(problem, ()))
        self.assertEqual(state.active_trajectories(), ())

    def test_invalid_request_does_not_change_replay(self):
        problem, logs, _ = switch_fixture(iterations=6)
        state = self.state_class(problem, logs)
        original = state.replay()
        for request, error in ((("forget", "missing"), KeyError), (("forget", "forget"), ValueError)):
            with self.assertRaises(error):
                state.delete_and_replay(request)
            self.compare(state.replay(), original)
