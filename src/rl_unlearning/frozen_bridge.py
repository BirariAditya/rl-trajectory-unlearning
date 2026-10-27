"""Deterministic frozen-feature bridge for MiniGrid-style offline episodes.

The adapter has no Minari or NumPy dependency.  It accepts episode objects with
the public ``EpisodeData`` fields and converts their image/direction observations
to a declared finite LSPI problem.  The feature map is fixed by its dimension,
action set, and seed; it is never fitted to rewards or to the retain/forget split.
"""

from dataclasses import dataclass
from fractions import Fraction as F
import hashlib

from .data import Trajectory, Transition
from .lspi_control import LSPIProblem


@dataclass(frozen=True)
class FrozenBridgeResult:
    problem: LSPIProblem
    trajectories: tuple[Trajectory, ...]
    episode_ids: tuple[str, ...]
    transitions: int
    unique_states: int
    feature_seed: str


def _length(value, name):
    try:
        return len(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a sized sequence") from error


def canonical_minigrid_observation(image, direction):
    """Return a platform-independent byte encoding of a 7x7x3 observation."""
    direction = int(direction)
    if not 0 <= direction < 4:
        raise ValueError("direction must be in 0..3")
    if _length(image, "image") != 7:
        raise ValueError("image must have shape 7x7x3")
    values = [direction]
    for row in image:
        if _length(row, "image row") != 7:
            raise ValueError("image must have shape 7x7x3")
        for cell in row:
            if _length(cell, "image cell") != 3:
                raise ValueError("image must have shape 7x7x3")
            for value in cell:
                value = int(value)
                if not 0 <= value <= 255:
                    raise ValueError("image values must be bytes")
                values.append(value)
    return bytes(values)


def hashed_state_action_features(payload, action, actions, dimension, seed):
    """CountSketch-style categorical features with explicit action indicators."""
    actions = tuple(actions)
    if action not in actions:
        raise ValueError("unknown action")
    if type(dimension) is not int or dimension < len(actions) + 4:
        raise ValueError("dimension must leave at least four hashed coordinates")
    if not isinstance(seed, str) or not seed:
        raise ValueError("feature seed must be a nonempty string")
    hashed = dimension - len(actions)
    vector = [0] * dimension
    vector[actions.index(action)] = 8
    prefix = seed.encode("utf-8") + b"\0" + int(action).to_bytes(4, "big", signed=True)
    # The bias token avoids a zero hashed block. Every observation byte is a
    # categorical (position, value) token, so numeric MiniGrid object codes are
    # not incorrectly treated as ordinal magnitudes.
    tokens = ((-1, 0),) + tuple(enumerate(payload))
    for position, value in tokens:
        token = (
            prefix
            + int(position).to_bytes(4, "big", signed=True)
            + int(value).to_bytes(2, "big")
        )
        digest = hashlib.blake2b(token, digest_size=8).digest()
        number = int.from_bytes(digest, "big")
        bucket = len(actions) + number % hashed
        vector[bucket] += 1 if number & (1 << 63) else -1
    return tuple(vector)


def build_minigrid_frozen_lspi(
    episodes,
    *,
    actions=(0, 1, 2),
    dimension=24,
    feature_seed="fourrooms-countsketch-v1",
    gamma=F(99, 100),
    regularization=64,
    iterations=5,
):
    """Convert complete episodes to one immutable fixed-feature LSPI problem."""
    actions = tuple(actions)
    if not actions or len(set(actions)) != len(actions):
        raise ValueError("actions must be unique and nonempty")
    episodes = tuple(episodes)
    if not episodes:
        raise ValueError("episodes must be nonempty")

    payload_to_state = {}
    payloads = []

    def state_for(observations, index):
        payload = canonical_minigrid_observation(
            observations["image"][index], observations["direction"][index]
        )
        if payload not in payload_to_state:
            payload_to_state[payload] = len(payloads)
            payloads.append(payload)
        return payload_to_state[payload]

    logs = []
    episode_ids = []
    for episode in episodes:
        identifier = str(episode.id)
        if not identifier or identifier in episode_ids:
            raise ValueError("episode ids must be nonempty and unique")
        episode_ids.append(identifier)
        observations = episode.observations
        if not isinstance(observations, dict) or not {"image", "direction"} <= set(observations):
            raise ValueError("observations must contain image and direction")
        count = _length(episode.actions, "actions")
        if count < 1:
            raise ValueError("episodes must contain at least one action")
        for name, values, expected in (
            ("images", observations["image"], count + 1),
            ("directions", observations["direction"], count + 1),
            ("rewards", episode.rewards, count),
            ("terminations", episode.terminations, count),
            ("truncations", episode.truncations, count),
        ):
            if _length(values, name) != expected:
                raise ValueError(f"{name} has inconsistent episode length")
        transitions = []
        finished = False
        for index in range(count):
            if finished:
                raise ValueError("episode contains steps after termination")
            action = int(episode.actions[index])
            if action not in actions:
                raise ValueError("episode contains an action outside the declared subset")
            state = state_for(observations, index)
            finished = bool(episode.terminations[index]) or bool(episode.truncations[index])
            next_state = None if finished else state_for(observations, index + 1)
            transitions.append(
                Transition(state, action, F(str(float(episode.rewards[index]))), next_state)
            )
        if not finished:
            # Minari episodes are complete by contract. Requiring a declared end
            # prevents silently treating a cut trajectory as continuing data.
            raise ValueError("episode must terminate or truncate on its final step")
        logs.append(Trajectory(identifier, tuple(transitions)))

    state_space = tuple(range(len(payloads)))
    features = {
        (state, action): hashed_state_action_features(
            payloads[state], action, actions, dimension, feature_seed
        )
        for state in state_space
        for action in actions
    }
    problem = LSPIProblem(
        state_space,
        actions,
        features,
        dict.fromkeys(state_space, actions[0]),
        gamma=gamma,
        regularization=regularization,
        iterations=iterations,
    )
    trajectories = tuple(logs)
    return FrozenBridgeResult(
        problem=problem,
        trajectories=trajectories,
        episode_ids=tuple(episode_ids),
        transitions=sum(len(item.transitions) for item in trajectories),
        unique_states=len(state_space),
        feature_seed=feature_seed,
    )
