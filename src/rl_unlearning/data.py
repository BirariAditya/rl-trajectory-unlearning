"""Immutable logged experience used by the tabular experiments."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Transition:
    state: int
    action: int
    reward: float
    next_state: int | None = None


@dataclass(frozen=True, slots=True)
class Trajectory:
    trajectory_id: str
    transitions: tuple[Transition, ...]

    def __post_init__(self) -> None:
        if not self.trajectory_id:
            raise ValueError("trajectory_id must be nonempty")
        if not self.transitions:
            raise ValueError("a trajectory must contain at least one transition")
