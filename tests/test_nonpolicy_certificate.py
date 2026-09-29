from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rl_unlearning import Trajectory, Transition
from rl_unlearning.nonpolicy_certificate import NonPolicyCertificate
from rl_unlearning.reference import exact_reference


def episode(name, reward, action=1):
    return Trajectory(name, (Transition(0, action, reward),))


class NonPolicyTests(unittest.TestCase):
    def test_request_does_not_scan_unaffected_states(self):
        states = tuple(range(32))
        data = tuple(Trajectory(f"{s}-{a}", (Transition(s, a, 1.0 if a == 0 else .2),))
                     for s in states for a in (0, 1))
        cert = NonPolicyCertificate(states, (0, 1), .9, data, dict.fromkeys(states, 0))
        result = cert.try_delete(["3-1"])
        self.assertTrue(result.accepted)
        self.assertEqual((result.affected_rows, result.checked_actions, result.transitions_read,
                          result.successor_entries_copied), (1, 1, 1, 1))

    def test_repeated_requests_share_policy_and_match_retraining(self):
        data = (episode("best", 1, 0), episode("a", .2), episode("b", .3), episode("c", .4))
        cert = NonPolicyCertificate((0,), (0, 1), .9, data, {0: 0})
        for deleted in ("a", "b", "c"):
            result = cert.try_delete([deleted])
            self.assertTrue(result.accepted)
            self.assertEqual(result.checked_actions, 1)
            self.assertIs(result.policy, cert.policy)
            data = tuple(t for t in data if t.trajectory_id != deleted)
            self.assertEqual(dict(result.policy), exact_reference((0,), (0, 1), .9, data)[1])
        with self.assertRaises(KeyError):
            cert.try_delete(["c"])

    def test_failed_request_is_atomic_and_can_be_retried(self):
        data = (episode("best", 1, 0), episode("low", -2), episode("high", 2))
        cert = NonPolicyCertificate((0,), (0, 1), .9, data, {0: 0})
        for _ in range(2):
            result = cert.try_delete(["low"])
            self.assertFalse(result.accepted)
            self.assertEqual(result.reason, "changed_action_beats_policy_or_tie_order")
        self.assertTrue(cert.try_delete(["low", "high"]).accepted)

    def test_chosen_row_change_rejected_but_redundant_sample_allowed(self):
        data = (episode("a", 1, 0), episode("b", 1, 0), episode("other", .5))
        cert = NonPolicyCertificate((0,), (0, 1), .9, data, {0: 0})
        self.assertTrue(cert.try_delete(["a"]).accepted)
        self.assertEqual(cert.try_delete(["b"]).reason, "chosen_policy_row_changed")

    def test_tie_order_is_checked(self):
        data = (episode("chosen", 1, 1), episode("low", 0, 0), episode("tie", 1, 0))
        cert = NonPolicyCertificate((0,), (0, 1), .9, data, {0: 1})
        self.assertFalse(cert.try_delete(["low"]).accepted)
        with self.assertRaises(ValueError):
            NonPolicyCertificate((0,), (0, 1), .9, (episode("a", 1, 0), episode("b", 1, 1)), {0: 1})

    def test_invalid_requests_do_not_mutate(self):
        cert = NonPolicyCertificate((0,), (0, 1), .9, (episode("a", 1, 0), episode("b", .1)), {0: 0})
        with self.assertRaises(KeyError):
            cert.try_delete(["b", "missing"])
        with self.assertRaises(ValueError):
            cert.try_delete(["b", "b"])
        self.assertTrue(cert.try_delete(["b"]).accepted)
        self.assertTrue(cert.try_delete([]).accepted)

    def test_random_accepted_deletions_match_exact_oracle(self):
        rng = random.Random(18)
        accepted = 0
        for trial in range(50):
            data = tuple(Trajectory(str(i), (Transition(rng.randrange(3), rng.randrange(2),
                         rng.randrange(-10, 11) / 10, rng.choice((None, 0, 1, 2))),)) for i in range(20))
            _, policy = exact_reference(range(3), (0, 1), .7, data)
            cert = NonPolicyCertificate(range(3), (0, 1), .7, data, policy)
            for t in data:
                if t.transitions[0].action == policy[t.transitions[0].state]:
                    continue
                result = cert.try_delete([t.trajectory_id])
                if result.accepted:
                    accepted += 1
                    data = tuple(x for x in data if x.trajectory_id != t.trajectory_id)
                    self.assertEqual(dict(result.policy), exact_reference(range(3), (0, 1), .7, data)[1])
        self.assertGreater(accepted, 100)


if __name__ == "__main__":
    unittest.main()
