# Minari FourRooms frozen-feature bridge: contract and limits

## External source

The bridge uses Farama's `D4RL/minigrid/fourrooms-v0` dataset. The official
[dataset page](https://minari.farama.org/datasets/D4RL/minigrid/fourrooms-v0/)
describes 590 expert-policy episodes and 10,010 transitions from
`MiniGrid-FourRooms-v0`. The official
[Minari basic-usage documentation](https://minari.farama.org/main/content/basic_usage/)
documents remote download, `load_dataset`, and episode iteration. The
[FourRooms environment source](https://github.com/Farama-Foundation/Minigrid/blob/main/minigrid/envs/fourrooms.py)
defines actions 0, 1, and 2 as left, right, and forward; pickup, drop, toggle,
and done are unused in this environment. The adapter therefore declares the
supported action subset `(0, 1, 2)` before any retain/forget split.

## Frozen representation

For each observation, the adapter serializes direction followed by the 7x7x3
MiniGrid image as 148 bytes. Mission text is constant ("reach the goal") and is
excluded. Each state-action row contains an explicit action indicator plus a
deterministic CountSketch-style sum of categorical `(position, byte value)`
tokens. BLAKE2b fixes bucket and sign from the declared feature seed, action,
position, and value.

No parameter is fitted to rewards, behavior returns, a deletion request, or
the retained subset. The original selected window determines only the finite
observed-state dictionary. That dictionary and every feature row remain fixed
through sequential deletion. A state that loses all retained support is not
removed from the problem.

## Deletion and comparison contract

- The deletion unit is one complete Minari episode identified by episode ID.
- Terminated or truncated final steps use the existing LSPI terminal mask.
- Direct retained-only scratch LSPI is the correctness reference.
- Exact policy-only inverse maintenance is the strong conventional control.
- Interval-verified binary64 replay must match the scratch learner's complete
  declared finite policy sequence or fall back atomically to exact replay.
- Feature construction is timed once per original window and charged to every
  method's end-to-end total. Local dataset loading is reported separately;
  network download is an acquisition cost, not a per-deletion learner cost.
- Source and local Minari data files are SHA-256 recorded in the raw report.

## What this bridge does not establish

The hash representation is deliberately frozen and CPU-testable. It is not a
learned encoder, a claim of good FourRooms return, or a substitute for IQL.
Policy-sequence equality does not imply exact critic weights, removal of every
historical intermediate, membership privacy, or distributional unlearning.
The benchmark uses disjoint episode windows from one dataset, not independent
tasks. Standard feature hashing, LSPI, verified linear algebra, and retraining
comparisons are prior components; connecting them here is external-validity
engineering rather than a publication-novelty claim.
