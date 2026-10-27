from dataclasses import dataclass
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from rl_unlearning.frozen_bridge import (
    build_minigrid_frozen_lspi,
    canonical_minigrid_observation,
    hashed_state_action_features,
)
from rl_unlearning.lspi_control import scratch_lspi
from rl_unlearning.lspi_verified_float import LSPIIntervalVerifiedFloatState
from run_minari_frozen_bridge import METHODS, run_window, summarize


def image(value=0):
    return [[[value, (row + col) % 6, 0] for col in range(7)] for row in range(7)]


@dataclass
class Episode:
    id: int
    observations: dict
    actions: tuple
    rewards: tuple
    terminations: tuple
    truncations: tuple


def episode(identifier, direction, actions=(2, 0)):
    count = len(actions)
    return Episode(
        identifier,
        {
            "image": tuple(image(identifier + index) for index in range(count + 1)),
            "direction": tuple((direction + index) % 4 for index in range(count + 1)),
        },
        tuple(actions),
        tuple(0.5 + index / 10 for index in range(count)),
        tuple(index == count - 1 for index in range(count)),
        tuple(False for _ in range(count)),
    )


class FrozenBridgeTests(unittest.TestCase):
    def test_canonical_encoding_and_features_are_deterministic(self):
        payload = canonical_minigrid_observation(image(2), 3)
        self.assertEqual(len(payload), 148)
        first = hashed_state_action_features(payload, 1, (0, 1, 2), 12, "seed")
        second = hashed_state_action_features(payload, 1, (0, 1, 2), 12, "seed")
        self.assertEqual(first, second)
        self.assertEqual(first[1], 8)
        self.assertNotEqual(
            first, hashed_state_action_features(payload, 2, (0, 1, 2), 12, "seed")
        )

    def test_adapter_preserves_complete_trajectories_and_terminal_mask(self):
        bridge = build_minigrid_frozen_lspi(
            (episode(0, 0), episode(1, 2)), dimension=12, iterations=3
        )
        self.assertEqual(bridge.transitions, 4)
        self.assertEqual(bridge.episode_ids, ("0", "1"))
        self.assertIsNotNone(bridge.trajectories[0].transitions[0].next_state)
        self.assertIsNone(bridge.trajectories[0].transitions[-1].next_state)
        self.assertEqual(
            set(bridge.problem.features),
            {(state, action) for state in bridge.problem.states for action in (0, 1, 2)},
        )

    def test_retained_only_verified_sequence_matches_scratch(self):
        bridge = build_minigrid_frozen_lspi(
            (episode(0, 0), episode(1, 1), episode(2, 2)),
            dimension=12,
            iterations=3,
        )
        state = LSPIIntervalVerifiedFloatState(bridge.problem, bridge.trajectories)
        result = state.delete_and_replay(("1",))
        reference = scratch_lspi(
            bridge.problem,
            tuple(item for item in bridge.trajectories if item.trajectory_id != "1"),
        )
        self.assertEqual(result.policy, reference.policy)
        self.assertEqual(
            result.policy_sequence,
            tuple(item.policy_out for item in reference.iterates),
        )
        self.assertEqual(result.status, reference.status)

    def test_rejects_partial_episode_and_unsupported_action(self):
        partial = episode(0, 0)
        partial.terminations = (False, False)
        with self.assertRaisesRegex(ValueError, "terminate or truncate"):
            build_minigrid_frozen_lspi((partial,), dimension=12)
        with self.assertRaisesRegex(ValueError, "outside the declared subset"):
            build_minigrid_frozen_lspi((episode(1, 0, actions=(6,)),), dimension=12)

    def test_runner_charges_preprocessing_and_matches_every_method(self):
        row = run_window(
            tuple(episode(index, index % 4) for index in range(4)),
            window=0,
            requests=1,
            dimension=12,
            feature_seed="test-seed",
        )
        self.assertGreater(row["preprocessing_seconds"], 0)
        self.assertEqual(set(row["preparation"]), set(METHODS))
        self.assertTrue(all(
            item["policy_sequence_match"]
            for item in row["requests"][0]["methods"].values()
        ))
        report = {
            "git_commit": "test", "git_dirty": True, "runs": [row],
            "dataset": {
                "id": "test", "total_episodes": 4, "total_steps": 8,
                "load_seconds": 0.1,
            },
        }
        text = summarize(report)
        self.assertIn("Preprocessing is charged", text)
        self.assertIn("0 policy-sequence mismatches", text)


if __name__ == "__main__":
    unittest.main()
