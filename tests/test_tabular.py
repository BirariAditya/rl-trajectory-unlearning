"""Counterfactual tests: deletion must match a fresh retained-data learner."""

from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rl_unlearning import EmpiricalMDP, Trajectory, Transition, delete_and_repair, value_iteration


def trajectory(trajectory_id: str, *transitions: Transition) -> Trajectory:
    return Trajectory(trajectory_id, transitions)


class TabularDeletionTests(unittest.TestCase):
    def compare_to_retraining(self, data, forgotten, *, states, actions=(0, 1), gamma=0.9,
                              budget=10_000):
        original = EmpiricalMDP(states, actions, gamma, data)
        old_plan = value_iteration(original)
        result = delete_and_repair(original, old_plan.q, forgotten,
                                   local_backup_budget=budget)
        retained = [item for item in data if item.trajectory_id not in forgotten]
        reference = EmpiricalMDP(states, actions, gamma, retained)
        retrained = value_iteration(reference)
        self.assertEqual(result.policy, retrained.policy)
        for row in retrained.q:
            self.assertAlmostEqual(result.q[row], retrained.q[row], places=8)
            self.assertAlmostEqual(original.mean_reward(*row), reference.mean_reward(*row))
            self.assertEqual(original.transition_probabilities(*row),
                             reference.transition_probabilities(*row))
        return original, result, retrained

    def test_shared_state_action_retains_its_supported_reward(self):
        data = [
            trajectory("retain-low-1", Transition(0, 0, 0.2)),
            trajectory("retain-low-2", Transition(0, 0, 0.2)),
            trajectory("forget-high", Transition(0, 0, 1.0)),
            trajectory("retain-other", Transition(0, 1, 0.4)),
        ]
        model, result, retrained = self.compare_to_retraining(
            data, {"forget-high"}, states=(0,))
        self.assertEqual(model.count(0, 0), 2)
        self.assertAlmostEqual(model.mean_reward(0, 0), 0.2)
        self.assertEqual(result.policy[0], 1)
        self.assertTrue(result.certified_optimal_policy)

    def test_non_greedy_predecessor_becomes_optimal(self):
        data = [
            trajectory("direct", Transition(0, 0, 1.0)),
            trajectory("branch", Transition(0, 1, 0.0, 1)),
            trajectory("weak", Transition(1, 0, 0.0)),
            trajectory("strong", Transition(1, 0, 2.0)),
        ]
        original = EmpiricalMDP((0, 1), (0, 1), 0.9, data)
        self.assertEqual(value_iteration(original).policy[0], 0)
        _, result, _ = self.compare_to_retraining(data, {"weak"}, states=(0, 1))
        self.assertEqual(result.policy[0], 1)
        self.assertFalse(result.used_full_fallback)

    def test_transition_probability_changes_and_propagates(self):
        data = [
            trajectory("low-route", Transition(0, 0, 0.0, 1)),
            trajectory("high-route", Transition(0, 0, 0.0, 2)),
            trajectory("alternative", Transition(0, 1, 0.6)),
            trajectory("low-value", Transition(1, 0, 0.0)),
            trajectory("high-value", Transition(2, 0, 2.0)),
        ]
        _, result, _ = self.compare_to_retraining(
            data, {"high-route"}, states=(0, 1, 2))
        self.assertEqual(result.policy[0], 1)

    def test_empty_state_action_row_uses_fixed_terminal_zero_rule(self):
        data = [
            trajectory("delete-only", Transition(0, 0, 3.0)),
            trajectory("retain", Transition(0, 1, 0.5)),
        ]
        model, result, _ = self.compare_to_retraining(
            data, {"delete-only"}, states=(0,))
        self.assertEqual(model.count(0, 0), 0)
        self.assertEqual(model.transition_probabilities(0, 0), {None: 1.0})
        self.assertEqual(result.policy[0], 1)

    def test_tie_forces_fallback_and_deterministic_tiebreak(self):
        data = [
            trajectory("a0", Transition(0, 0, 1.0)),
            trajectory("a1", Transition(0, 1, 1.0)),
            trajectory("to-delete", Transition(0, 0, 1.0)),
        ]
        _, result, _ = self.compare_to_retraining(
            data, {"to-delete"}, states=(0,))
        self.assertEqual(result.policy[0], 0)
        self.assertTrue(result.used_full_fallback)
        self.assertFalse(result.certified_optimal_policy)

    def test_repeated_deletions_and_request_atomicity(self):
        data = [
            trajectory("first", Transition(0, 0, 2.0)),
            trajectory("second", Transition(0, 0, 0.0)),
            trajectory("third", Transition(0, 1, 0.5)),
        ]
        model = EmpiricalMDP((0,), (0, 1), 0.9, data)
        first = delete_and_repair(model, value_iteration(model).q, ["first"])
        second = delete_and_repair(model, first.q, ["second"])
        reference = value_iteration(EmpiricalMDP((0,), (0, 1), 0.9, [data[2]]))
        self.assertEqual(second.policy, reference.policy)
        self.assertEqual(model.trajectory_ids, frozenset({"third"}))
        before = model.count(0, 1)
        with self.assertRaises(KeyError):
            model.delete_trajectories(["third", "missing"])
        self.assertEqual(model.count(0, 1), before)
        with self.assertRaises(ValueError):
            model.delete_trajectories(["third", "third"])

    def test_zero_local_budget_uses_full_fallback(self):
        data = [
            trajectory("delete", Transition(0, 0, 2.0)),
            trajectory("retain", Transition(0, 1, 1.0)),
        ]
        _, result, _ = self.compare_to_retraining(
            data, {"delete"}, states=(0,), budget=0)
        self.assertTrue(result.used_full_fallback)
        self.assertEqual(result.local_backups, 0)

    def test_invalid_fallback_limit_does_not_delete(self):
        data = [trajectory("keep", Transition(0, 0, 1.0))]
        model = EmpiricalMDP((0,), (0, 1), 0.9, data)
        with self.assertRaises(ValueError):
            delete_and_repair(model, value_iteration(model).q, ["keep"],
                              fallback_max_sweeps=0)
        self.assertEqual(model.trajectory_ids, frozenset({"keep"}))

    def test_loose_tolerance_does_not_imply_scratch_policy_equality(self):
        data = [
            trajectory("direct", Transition(0, 0, 1.0)),
            trajectory("branch", Transition(0, 1, 0.0, 1)),
            trajectory("low", Transition(1, 0, 0.1112, 1)),
            trajectory("high", Transition(1, 0, 0.1488, 1)),
        ]
        model = EmpiricalMDP((0, 1), (0, 1), 0.9, data)
        original = value_iteration(model, tolerance=0.01)
        repaired = delete_and_repair(model, original.q, ["high"], tolerance=0.01)
        scratch_model = EmpiricalMDP((0, 1), (0, 1), 0.9, data[:3])
        scratch = value_iteration(scratch_model, tolerance=0.01)
        self.assertNotEqual(repaired.policy[0], scratch.policy[0])
        self.assertFalse(repaired.certified_optimal_policy)
        self.assertTrue(repaired.used_full_fallback)
        # A strict solve settles the retained empirical MDP's actual preference.
        strict = value_iteration(scratch_model, tolerance=1e-12)
        self.assertEqual(strict.policy[0], 1)

    def test_redundant_deleted_experience_need_not_reduce_behavior(self):
        data = [
            trajectory("retain-a", Transition(0, 0, 1.0)),
            trajectory("delete-a", Transition(0, 0, 1.0)),
            trajectory("retain-b", Transition(0, 1, 0.5)),
        ]
        _, result, _ = self.compare_to_retraining(
            data, {"delete-a"}, states=(0,))
        self.assertEqual(result.policy[0], 0)
        self.assertTrue(result.certified_optimal_policy)

    def test_random_counterfactual_models_and_certificates(self):
        rng = random.Random(20260929)
        for trial in range(100):
            data = []
            for index in range(15):
                steps = tuple(
                    Transition(
                        state=rng.randrange(3),
                        action=rng.randrange(2),
                        reward=rng.uniform(-1, 1),
                        next_state=rng.choice((None, 0, 1, 2)),
                    )
                    for _ in range(rng.randint(1, 3))
                )
                data.append(Trajectory(f"{trial}-{index}", steps))
            forgotten = {item.trajectory_id for item in rng.sample(data, rng.randint(1, 6))}
            original = EmpiricalMDP((0, 1, 2), (0, 1), 0.8, data)
            repaired = delete_and_repair(
                original, value_iteration(original).q, forgotten,
                local_backup_budget=rng.choice((0, 1, 10, 100)))
            reference = EmpiricalMDP(
                (0, 1, 2), (0, 1), 0.8,
                (item for item in data if item.trajectory_id not in forgotten))
            retrained = value_iteration(reference)
            for state in reference.states:
                for action in reference.actions:
                    row = (state, action)
                    self.assertEqual(original.count(*row), reference.count(*row))
                    self.assertAlmostEqual(original.mean_reward(*row),
                                           reference.mean_reward(*row), places=12)
                    self.assertEqual(original.transition_probabilities(*row),
                                     reference.transition_probabilities(*row))
                    self.assertAlmostEqual(repaired.q[row], retrained.q[row], places=7)
            if repaired.certified_optimal_policy:
                self.assertEqual(repaired.policy, retrained.policy)


if __name__ == "__main__":
    unittest.main()
