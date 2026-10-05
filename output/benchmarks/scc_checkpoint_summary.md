# Matched SCC planning checkpoint

Source: `94e7f45107346afb0b9a89d00122b007941e7abc`; clean at execution: `True`.

**162 runs, 711 sequential requests, 5688 method executions, 0 failures.**
All methods checked for exact policy and value equality against scratch dense rational PI.

## Request cost (milliseconds)

Median across runs of each run's median request time. Scratch includes rebuilding retained statistics and its index. Whole-method timings include graph construction and index maintenance.

| Regime | States | local_dense | pruned_dense | warm_dense | scratch_dense | local_scc | pruned_scc | warm_scc | scratch_scc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| anchor_invalidation | 8 | 1.718 | 1.738 | 1.737 | 1.972 | 1.798 | 1.838 | 1.867 | 2.065 |
| anchor_invalidation | 16 | 11.167 | 11.959 | 11.715 | 11.214 | 11.143 | 11.876 | 11.217 | 11.983 |
| anchor_invalidation | 32 | 82.243 | 82.796 | 83.511 | 84.948 | 84.302 | 83.745 | 83.038 | 84.907 |
| block_deletion | 8 | 0.454 | 0.438 | 2.022 | 1.968 | 0.513 | 0.535 | 0.762 | 1.223 |
| block_deletion | 16 | 1.871 | 1.967 | 12.820 | 12.977 | 1.038 | 1.045 | 1.755 | 2.111 |
| block_deletion | 32 | 11.660 | 12.482 | 87.199 | 88.987 | 2.110 | 2.126 | 3.198 | 4.787 |
| dense | 8 | 1.735 | 2.210 | 1.771 | 3.479 | 2.107 | 1.911 | 1.976 | 3.425 |
| dense | 16 | 7.834 | 8.381 | 8.158 | 12.709 | 8.409 | 8.948 | 8.004 | 13.188 |
| dense | 32 | 54.682 | 54.984 | 55.374 | 72.872 | 57.915 | 56.512 | 57.504 | 75.704 |
| irrelevant_bridges | 8 | 1.925 | 0.686 | 1.823 | 2.163 | 1.925 | 0.513 | 2.413 | 2.010 |
| irrelevant_bridges | 16 | 11.530 | 0.479 | 10.732 | 11.779 | 11.101 | 0.540 | 10.868 | 11.487 |
| irrelevant_bridges | 32 | 81.921 | 0.528 | 81.566 | 81.717 | 81.939 | 0.570 | 80.369 | 83.523 |
| modular | 8 | 0.438 | 0.438 | 1.751 | 1.777 | 0.454 | 0.451 | 0.679 | 0.958 |
| modular | 16 | 0.405 | 0.421 | 9.758 | 10.933 | 0.455 | 0.481 | 1.162 | 1.935 |
| modular | 32 | 0.425 | 0.450 | 76.024 | 76.082 | 0.492 | 0.500 | 2.157 | 3.750 |
| weak_bridges | 8 | 1.768 | 1.750 | 1.814 | 3.221 | 2.222 | 1.866 | 2.177 | 2.943 |
| weak_bridges | 16 | 12.641 | 12.037 | 13.278 | 20.180 | 12.553 | 11.953 | 12.355 | 20.588 |
| weak_bridges | 32 | 121.489 | 123.524 | 120.426 | 139.094 | 120.164 | 121.974 | 120.598 | 138.724 |

## Preparation plus full deletion sequence (milliseconds)

Median of preparation plus every request, including the anchor-invalidating request where applicable. Preparation includes original training and retained indexes; pruned variants also include a dense exact anchor evaluation. Scratch is reported only as a per-request rebuild above.

| Regime | States | Requests | local_dense | pruned_dense | warm_dense | local_scc | pruned_scc | warm_scc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| anchor_invalidation | 8 | 3 | 6.822 | 7.501 | 5.969 | 6.088 | 7.393 | 6.569 |
| anchor_invalidation | 16 | 5 | 59.794 | 67.681 | 59.088 | 60.715 | 69.640 | 58.966 |
| anchor_invalidation | 32 | 9 | 757.568 | 805.471 | 764.781 | 770.761 | 812.678 | 766.089 |
| block_deletion | 8 | 2 | 1.947 | 3.032 | 5.096 | 2.042 | 3.347 | 2.358 |
| block_deletion | 16 | 2 | 11.700 | 17.429 | 33.410 | 3.722 | 10.159 | 5.273 |
| block_deletion | 32 | 2 | 65.788 | 112.430 | 218.010 | 7.734 | 54.713 | 10.256 |
| dense | 8 | 2 | 6.908 | 8.200 | 6.774 | 6.936 | 8.688 | 6.687 |
| dense | 16 | 4 | 49.435 | 55.689 | 50.353 | 52.860 | 63.311 | 52.185 |
| dense | 32 | 8 | 559.263 | 608.693 | 561.488 | 572.765 | 627.802 | 567.670 |
| irrelevant_bridges | 8 | 2 | 4.965 | 3.878 | 5.117 | 5.239 | 3.299 | 6.021 |
| irrelevant_bridges | 16 | 4 | 52.095 | 15.106 | 49.529 | 53.951 | 14.809 | 50.183 |
| irrelevant_bridges | 32 | 8 | 706.774 | 89.508 | 693.037 | 703.403 | 91.129 | 695.543 |
| modular | 8 | 2 | 1.951 | 3.019 | 4.494 | 1.765 | 2.507 | 2.057 |
| modular | 16 | 4 | 7.994 | 13.345 | 46.266 | 3.238 | 8.660 | 6.996 |
| modular | 32 | 8 | 44.971 | 81.721 | 650.733 | 7.778 | 47.027 | 21.679 |
| weak_bridges | 8 | 2 | 6.161 | 8.518 | 5.707 | 6.416 | 6.667 | 6.496 |
| weak_bridges | 16 | 4 | 60.484 | 69.553 | 66.415 | 62.104 | 69.688 | 68.566 |
| weak_bridges | 32 | 8 | 971.128 | 1019.887 | 957.462 | 966.303 | 1026.023 | 965.084 |

## Mechanism checks

Affected sizes are reported closure sizes before any forced global fallback; an anchor-invalidating request can report one affected source while solving the full model. Largest SCC describes actual component solves. Pruned methods retain every action inside the solve block; they do not apply FTVI action elimination to its internal SCC graph.

| Regime | States | Local SCC affected sizes | Pruned SCC affected sizes | Warm SCC largest components | Pruned SCC largest components | Pruning invalidations |
| --- | ---: | --- | --- | --- | --- | ---: |
| anchor_invalidation | 8 | [8] | [1, 8] | [8] | [8] | 9 |
| anchor_invalidation | 16 | [16] | [1, 16] | [16] | [16] | 9 |
| anchor_invalidation | 32 | [32] | [1, 32] | [32] | [32] | 9 |
| block_deletion | 8 | [4] | [4] | [4] | [4] | 0 |
| block_deletion | 16 | [8] | [8] | [4] | [4] | 0 |
| block_deletion | 32 | [16] | [16] | [4] | [4] | 0 |
| dense | 8 | [8] | [8] | [8] | [8] | 0 |
| dense | 16 | [16] | [16] | [16] | [16] | 0 |
| dense | 32 | [32] | [32] | [32] | [32] | 0 |
| irrelevant_bridges | 8 | [8] | [4] | [8] | [4] | 0 |
| irrelevant_bridges | 16 | [16] | [4] | [16] | [4] | 0 |
| irrelevant_bridges | 32 | [32] | [4] | [32] | [4] | 0 |
| modular | 8 | [4] | [4] | [4] | [4] | 0 |
| modular | 16 | [4] | [4] | [4] | [4] | 0 |
| modular | 32 | [4] | [4] | [4] | [4] | 0 |
| weak_bridges | 8 | [8] | [8] | [8] | [8] | 0 |
| weak_bridges | 16 | [16] | [16] | [16] | [16] | 0 |
| weak_bridges | 32 | [32] | [32] | [32] | [32] | 0 |

## Timed anchor invalidation

Median first-request time in anchor_invalidation. Later requests are included in the full-sequence table and run with pruning disabled.

| States | Pruned dense | Warm dense | Pruned SCC | Warm SCC |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.939 | 0.928 | 1.022 | 1.025 |
| 16 | 5.571 | 5.467 | 6.054 | 5.792 |
| 32 | 43.068 | 42.686 | 42.764 | 41.567 |

## Scope

- Seeds: [0, 1, 2]; repeats: [0, 1, 2]. Repeats reuse each seed's dataset and sequence, so they are timing replicates. Synthetic models only; medians are descriptive, without statistical significance claims.
- Matched arithmetic, inner exact solver, data, and tie order isolate a graph-decomposition ablation. This is our implementation of classical SCC planning, not an optimized external solver or a TVI/FTVI reproduction.
- All warm/scratch methods share the reverse-index scaffolding used by local repair. A standalone global planner could avoid that extra overhead; this limits speedup interpretation.
- Whole-call wall times include preparation, rational arithmetic, SCC reconstruction, invalidation, and index work as applicable. Work counters are partial diagnostics; anchor evaluation and every Python operation are not individually counted.
- Policy/value snapshots and equality checks are common offline evaluation overhead, excluded from request latency. No neural, privacy, universal locality, or novelty claim is made.
