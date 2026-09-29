# Checkpoint 01: exact local trajectory deletion in tabular control

Reached 2026-09-29. Source revision: `70f13773b8e467e99cb429115cc2b437c008708e`.

## What now works

The repository can delete complete trajectories, change the agent's chosen actions, and return the exact retained-only policy and values for a specified rational empirical-MDP learner. It supports cyclic transitions, sequential requests, exact action ties, and updates to the dependency index. It either solves a provably sufficient block or uses full exact planning.

An optional anchor-policy bound identifies actions that cannot be optimal and removes their edges from the dependency graph. When deletion changes the anchor policy's model rows, the method performs full repair and disables pruning. When a deleted sample makes a previously pruned action competitive, its edges are restored. The [proof/contract](../audits/local_repair_proof.md) explains these conditions and counts preparation and verification work.

This goes beyond the earlier no-policy-change prototype: chosen actions can now change, and exact retained values can be reconstructed locally despite cycles. It is a **major implementation and correctness checkpoint**. It does not establish newness relative to decades of incremental planning and action elimination.

## Evidence

- **36 tests pass**, including 40 random cyclic datasets with four sequential deletions each, checked against independent rational policy enumeration.
- [GitHub CI](https://github.com/BirariAditya/rl-trajectory-unlearning/actions/runs/36603869847) passed the tests and all three smoke runners on the source revision above.
- [Checkpoint benchmark](../output/benchmarks/local_checkpoint_summary.md): 54 repeated runs, 162 sequential deletion requests, and **810 method executions with zero failures**. Rational local and warm methods matched scratch exact policies and values; warm binary64 VI matched the policy in these instances.
- The [compressed report](../output/benchmarks/local_checkpoint.json.gz) stores source hashes, full first-repeat datasets, deletion sequences, timings, preparation costs, action counts, reverse-index work, and linear-system sizes.

## Main observation

In the constructed 16-state cyclic model with highly suboptimal bridge actions, ordinary reverse closure selected all 16 states. Bound-certified pruning reduced the solved block to four states even though the optimal action could change inside it. Median request latency was **0.448 ms** for pruned exact repair and **9.847 ms** for warm full exact policy iteration. Including preparation and all four requests, median sequence time was **12.502 ms versus 46.365 ms**.

This finding has a clear limit. In the dense admissible-influence model, both methods solved all 16 states; their request medians were **6.837 ms versus 6.807 ms**. Including preparation, pruned repair was slower: **47.962 ms versus 42.633 ms**. The method does not obtain universal local repair. In already disconnected modular graphs, ordinary local repair also avoids the anchor's extra preparation cost.

These are small, designed examples with three dataset seeds and three repeats, not a statistically established result on practical RL tasks. Dense rational elimination is a correctness-oriented implementation; a mature sparse or SCC-aware global planner may narrow the apparent gains substantially.

## Research decision

Continue from this verified substrate, but keep the **novelty gate open**. Upper/lower-bound action elimination dates at least to [1973](https://pubsonline.informs.org/doi/10.1287/opre.21.3.848); faster local planning alone is insufficient as the paper's contribution.

The next decision gate is whether deletion-specific structure yields an advantage beyond existing incremental/SCC/action-elimination planners. Test less extreme bridge rewards, anchor-changing deletions, larger state spaces, and user-sized trajectory blocks. Seek an instance-dependent complexity result that predicts these outcomes without free global preprocessing. If that gap cannot be established, shift the headline contribution to policy-target-aware frozen-feature LSPI control and keep this implementation as an exact tabular baseline.

## Remaining work

1. Add mature graph/SCC-aware planning and action-elimination baselines under matched arithmetic and solver choices.
2. Measure pruning failure and anchor-rebuild cost under realistic deletion shifts; the current timed matrix keeps the anchor unchanged.
3. Replace or supplement dense rational solves with a numerically certified practical solver before scaling conclusions.
4. Develop the general margin-sensitive locality theorem or execute the fixed-feature control pivot.
5. Begin GPU experiments only after the restricted-learner and novelty gates in the implementation plan are justified.
