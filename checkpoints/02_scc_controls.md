# Checkpoint 02: stronger graph controls and deletion stress tests

Reached 5 October 2026. Experiment source: `94e7f45107346afb0b9a89d00122b007941e7abc`, clean at launch. [Raw results](../output/benchmarks/scc_checkpoint.json.gz), [full tables](../output/benchmarks/scc_checkpoint_summary.md), [solver contract](../audits/scc_planning_checkpoint.md).

## What was completed

Added an exact strongly-connected-component (SCC) planner using retained all-action support, sink-first solves, and the same rational policy iteration within each component. Local repair, pruned repair, warm full planning, and scratch full planning each have dense and SCC variants. This controls for ordinary graph decomposition instead of attributing all improvement over dense solves to unlearning.

The experiment now covers six regimes: disconnected modules, strongly dominated bridges, weaker bridges, anchor-invalidating deletion followed by further requests, batches of trajectories across modules, and dense influence. Sizes are 8, 16, and 32 states, with three dataset seeds and three timing repetitions per seed. Every method sees the same deletion sequence. Preparation and request order are randomized within a run; all preparation, SCC reconstruction, and fallback work is timed.

## Evidence

- **42 tests pass**, including 96 new randomized sequential deletion checks against independent rational policy enumeration. Regressions cover non-greedy dependencies, updated external boundaries, deletion splitting a component, exact ties, a 1,100-state chain, and failure without partial mutation.
- [Source-revision GitHub CI](https://github.com/BirariAditya/rl-trajectory-unlearning/actions/runs/37332934561) passed the test suite and all four smoke runners.
- **162 runs, 711 sequential requests, 5,688 method executions; zero failures.** Every method's policy and values matched scratch dense exact planning. Large-run agreement is not an independent formal proof because methods share the inner solver; independent enumeration covers tiny tests.

## What the stronger comparison changes

The following full-sequence medians include preparation and all requests, in milliseconds. Each row is the 32-state setting.

| Regime | Local dense | Warm dense | Local SCC | Pruned SCC | Warm SCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Modular | 44.971 | 650.733 | 7.778 | 47.027 | 21.679 |
| Dominated bridges | 706.774 | 693.037 | 703.403 | 91.129 | 695.543 |
| Weaker bridges | 971.128 | 957.462 | 966.303 | 1026.023 | 965.084 |
| Anchor invalidation | 757.568 | 764.781 | 770.761 | 812.678 | 766.089 |
| Batched deletion | 65.788 | 218.010 | 7.734 | 54.713 | 10.256 |
| Dense | 559.263 | 561.488 | 572.765 | 627.802 | 567.670 |

1. **Graph-aware controls substantially reduce the apparent advantage.** On modular data the dense local-versus-warm sequence ratio is about 14.5x, while the matched SCC ratio is about 2.8x. Warm SCC even beats the original local dense implementation on full-sequence cost. A large dense-solver speedup is therefore not sufficient evidence of a new unlearning contribution.
2. **Pruning still helps the deliberately favorable bridge case.** It solves four states instead of 32, with median request time 0.570 ms versus 80.369 ms for warm SCC. Including preparation reduces that difference to about 7.6x for the full sequence. This is a designed locality example, not a real-task superiority result.
3. **Its failure regimes are now timed.** Weak bridges and dense influence select the full state space. Every anchor-invalidation run disables pruning on its first request and continues safely; the 32-state first-request medians are 42.764 ms for pruned SCC versus 41.567 ms for warm SCC. Extra anchor preparation makes pruned full-sequence cost worse in those settings.
4. **Preparation matters even when requests are cheap.** On modular data pruned SCC has cheap updates but costs 47.027 ms per full sequence versus 7.778 ms for unpruned local SCC, chiefly because the extra anchor evaluation remains a dense solve.

These are descriptive synthetic measurements. Timing repetitions share datasets, request observations are correlated, and no significance or practical-scale speed claim is made. The controls share a reverse-index scaffold that an optimized standalone global planner could avoid. Our SCC method is an internal classical baseline, not an external mature planner or an FTVI reproduction. Full action elimination inside SCCs and dynamic SCC maintenance remain untested.

## Research decision

[FTVI already combines provable action elimination with smaller graph components](https://arxiv.org/abs/1401.3910). This directly overlaps the mechanism behind the dominated-bridge demonstration. Checkpoint 02 strengthens correctness and measurement, while weakening any claim that the current mechanism is itself novel.

The immediate gate is to compare with an action-eliminating graph planner and a lean global baseline without local-index overhead. Track A then needs a deletion-specific theorem about complete request work beyond standard decomposition and bound pruning. If that gate fails, prioritize the fixed-feature LSPI control target, accounting for policy switches on retained successor transitions. Neither a new theorem nor the LSPI/GPU implementation has been completed. Do not scale neural experiments or describe the current baseline as a publishable breakthrough yet.
