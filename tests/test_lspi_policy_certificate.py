from fractions import Fraction as F
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rl_unlearning.lspi_control import LSPIProblem, scratch_lspi, SingularLSTDMatrix
from rl_unlearning.lspi_policy_certificate import LSPIPolicyCertificateState
from test_lspi_control import episode, switch_fixture


class PolicyCertificateTests(unittest.TestCase):
    def compare(self, result, reference):
        self.assertEqual(result.policy, reference.policy)
        self.assertEqual(result.policy_sequence, tuple(t.policy_out for t in reference.iterates))
        self.assertEqual(result.status, reference.status)
        self.assertEqual((result.repeat_start, result.repeat_length),
                         (reference.repeat_start, reference.repeat_length))

    def test_small_deletion_certifies_without_solves_or_action_scan(self):
        problem = LSPIProblem((0,), (0, 1), {(0, 0): (1,), (0, 1): (-1,)}, {0: 0}, iterations=6)
        logs = [episode(str(i), (0, 0, 1, None)) for i in range(50)]
        state = LSPIPolicyCertificateState(problem, logs)
        after = state.delete_and_replay(('0',))
        self.compare(after, scratch_lspi(problem, logs[1:]))
        self.assertTrue(after.certified)
        self.assertEqual(after.work['linear_solves'], 0)
        self.assertEqual(after.work['action_scores'], 0)
        # The returned object deliberately exposes no stale weights or matrices.
        self.assertFalse(hasattr(after, 'weights'))

    def test_policy_switch_rejects_certificate_and_rebuilds(self):
        problem, logs, retained = switch_fixture(iterations=6)
        state = LSPIPolicyCertificateState(problem, logs)
        after = state.delete_and_replay(('forget',))
        self.compare(after, scratch_lspi(problem, retained))
        self.assertFalse(after.certified)
        self.assertTrue(after.used_fallback)
        self.assertEqual(after.policy, (0, 1))

    def test_sequential_bounds_accumulate_then_fallback_resets_anchor(self):
        problem = LSPIProblem((0,), (0, 1), {(0, 0): (1,), (0, 1): (-1,)}, {0: 0})
        logs = [episode(str(i), (0, 0, 1 if i < 12 else -1, None)) for i in range(20)]
        state = LSPIPolicyCertificateState(problem, logs)
        certified = fallbacks = 0
        for i in range(12):
            logs = [t for t in logs if t.trajectory_id != str(i)]
            after = state.delete_and_replay((str(i),))
            self.compare(after, scratch_lspi(problem, logs))
            certified += after.certified
            fallbacks += after.used_fallback
        self.assertGreater(certified, 0)
        self.assertGreater(fallbacks, 0)
        self.assertEqual(after.policy, (1,))

    def test_tie_zero_residual_and_identical_features(self):
        for same in (False, True):
            with self.subTest(identical_features=same):
                problem = LSPIProblem((0,), (1, 0), {(0, 0): (1,), (0, 1): (1 if same else -1,)},
                                      {0: 0}, iterations=5)
                logs = (episode('a', (0, 0, 0, None)), episode('b', (0, 0, 0, None)))
                state = LSPIPolicyCertificateState(problem, logs)
                after = state.delete_and_replay(('a',))
                self.compare(after, scratch_lspi(problem, logs[1:]))
                self.assertTrue(after.certified)
                self.assertEqual(after.policy, (1,))

    def test_singular_fallback_is_atomic_including_bounds(self):
        problem = LSPIProblem((0,), (0, 1), {(0, 0): (5,), (0, 1): (1,)}, {0: 1})
        logs = (episode('loop', (0, 1, 1, 0), (0, 1, 0, None)), episode('extra', (0, 1, 0, None)))
        state = LSPIPolicyCertificateState(problem, logs)
        original_alpha, original_h = state._alpha, state._h
        for _ in range(2):
            after = state.delete_and_replay(('extra',))
            self.compare(after, scratch_lspi(problem, logs[:1]))
            self.assertEqual(after.status, 'singular')
            self.assertEqual(state.trajectory_ids, frozenset(('loop', 'extra')))
            self.assertEqual((state._alpha, state._h), (original_alpha, original_h))

    def test_invalid_requests_and_singular_initial_schedule(self):
        problem, logs, _ = switch_fixture()
        state = LSPIPolicyCertificateState(problem, logs)
        for ids, error in ((('forget', 'absent'), KeyError), (('forget', 'forget'), ValueError)):
            with self.assertRaises(error):
                state.delete_and_replay(ids)
            self.assertEqual(state.trajectory_ids, frozenset(t.trajectory_id for t in logs))
        singular = LSPIProblem((0,), (0,), {(0, 0): (1,)}, {0: 0}, regularization=0)
        with self.assertRaises(SingularLSTDMatrix):
            LSPIPolicyCertificateState(singular, ())

    def test_residual_cancellation_does_not_reset_accumulated_perturbation(self):
        problem = LSPIProblem((0,), (0, 1), {(0, 0): (1,), (0, 1): (0,)}, {0: 0}, regularization=1)
        rewards = [0] * 4 + [F(2, 3)] * 4 + [0, 1]
        logs = tuple(episode(str(i), (0, 0, r, None)) for i, r in enumerate(rewards))
        state = LSPIPolicyCertificateState(problem, logs)
        first = state.delete_and_replay(tuple(str(i) for i in range(8)))
        self.assertTrue(first.certified)
        self.assertEqual(state._h, ((F(0),),))
        self.assertEqual(state._alpha, (F(8, 11),))
        second = state.delete_and_replay(('8',))
        self.compare(second, scratch_lspi(problem, logs[9:]))
        self.assertEqual(state._alpha, (F(9, 11),))
        self.assertEqual(state._h, ((F(1, 33),),))

    def test_randomized_policy_sequences_never_false_certify(self):
        rng = random.Random(953)
        certified = 0
        for trial in range(40):
            d = 1 + trial % 3
            features = {(s, a): tuple(F(rng.randrange(-2, 3)) for _ in range(d))
                        for s in range(3) for a in range(2)}
            problem = LSPIProblem((0, 1, 2), (0, 1), features, dict.fromkeys(range(3), 0),
                                  gamma=F(1, 3), iterations=6)
            logs = []
            for i in range(16):
                s, z = rng.randrange(3), rng.randrange(3)
                logs.append(episode(str(i), (s, rng.randrange(2), rng.randrange(-3, 4), z),
                                    (z, rng.randrange(2), rng.randrange(-3, 4), None)))
            if scratch_lspi(problem, logs).status == 'singular':
                continue
            state = LSPIPolicyCertificateState(problem, logs)
            for i in range(5):
                with self.subTest(trial=trial, deletion=i):
                    candidate = tuple(t for t in logs if t.trajectory_id != str(i))
                    result = state.delete_and_replay((str(i),))
                    self.compare(result, scratch_lspi(problem, candidate))
                    certified += result.certified
                    if result.status != 'singular':
                        logs = list(candidate)
        self.assertGreater(certified, 0)
