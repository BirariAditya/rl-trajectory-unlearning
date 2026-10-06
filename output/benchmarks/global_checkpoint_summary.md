# Lean global planning and elimination checkpoint

Source: `51a999a2519ada3b95e08de9d380a1233a109bbf`; clean at launch: `True`.

**108 runs, 594 sequential requests, 4158 method executions, 0 failures.**
Every method checked for exact policy and value agreement with lean scratch rational SCC planning.

## Request cost (milliseconds)

Median across runs of per-run median request time. Scratch reconstructs retained statistics; other methods subtract deleted records. All graph reconstruction, pruning scans and anchor refresh are charged. Snapshots/equality checks are excluded equally as offline evaluation overhead.

| Regime | States | local_scc | pruned_local_scc | lean_warm_dense | lean_warm_scc | eliminate_scc | lean_scratch_scc | eliminate_scratch_scc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| anchor_invalidation | 16 | 12.255 | 12.226 | 11.720 | 11.969 | 1.623 | 12.532 | 2.958 |
| anchor_invalidation | 32 | 85.580 | 85.263 | 86.044 | 85.052 | 3.095 | 86.486 | 5.864 |
| block_deletion | 16 | 1.064 | 1.136 | 11.827 | 1.755 | 1.720 | 2.072 | 3.002 |
| block_deletion | 32 | 2.213 | 2.170 | 87.583 | 3.453 | 3.620 | 4.456 | 5.564 |
| dense | 16 | 8.509 | 8.274 | 9.129 | 8.314 | 9.106 | 11.082 | 13.178 |
| dense | 32 | 57.137 | 58.739 | 56.049 | 56.666 | 58.860 | 69.661 | 73.305 |
| irrelevant_bridges | 16 | 11.861 | 0.636 | 11.737 | 12.417 | 1.630 | 12.869 | 2.875 |
| irrelevant_bridges | 32 | 89.259 | 0.674 | 88.249 | 90.411 | 3.257 | 91.019 | 5.967 |
| modular | 16 | 0.553 | 0.565 | 10.481 | 1.319 | 1.457 | 2.004 | 2.678 |
| modular | 32 | 0.558 | 0.575 | 79.249 | 2.401 | 2.820 | 3.944 | 5.243 |
| weak_bridges | 16 | 14.937 | 13.478 | 13.147 | 11.909 | 13.011 | 21.688 | 22.981 |
| weak_bridges | 32 | 128.251 | 125.335 | 127.116 | 128.362 | 128.468 | 148.164 | 146.279 |

## Preparation plus all requests (milliseconds)

Median full-run cost. Every method is charged original full-data training once; scratch controls additionally retrain on retained logs at every request. Includes preparation, every request and anchor invalidation. Legacy pruned local and global elimination use different anchor-evaluation/preparation algorithms; this is a method comparison, not a pure locality ablation.

| Regime | States | Requests | local_scc | pruned_local_scc | lean_warm_dense | lean_warm_scc | eliminate_scc | lean_scratch_scc | eliminate_scratch_scc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| anchor_invalidation | 16 | 5 | 63.009 | 71.643 | 61.628 | 62.200 | 10.909 | 63.873 | 18.473 |
| anchor_invalidation | 32 | 9 | 790.315 | 835.267 | 798.816 | 783.694 | 39.642 | 795.262 | 58.788 |
| block_deletion | 16 | 2 | 3.888 | 10.387 | 29.569 | 5.124 | 6.135 | 5.999 | 8.457 |
| block_deletion | 32 | 2 | 7.774 | 56.832 | 220.050 | 11.044 | 12.272 | 12.192 | 15.983 |
| dense | 16 | 4 | 53.995 | 59.265 | 49.991 | 48.794 | 54.690 | 62.719 | 68.043 |
| dense | 32 | 8 | 566.784 | 620.498 | 561.876 | 571.815 | 581.265 | 663.059 | 702.797 |
| irrelevant_bridges | 16 | 4 | 54.893 | 16.011 | 53.078 | 56.639 | 9.315 | 58.420 | 15.077 |
| irrelevant_bridges | 32 | 8 | 785.555 | 96.257 | 787.787 | 769.006 | 32.969 | 785.993 | 57.706 |
| modular | 16 | 4 | 4.288 | 10.251 | 49.298 | 7.160 | 8.994 | 10.664 | 14.006 |
| modular | 32 | 8 | 7.930 | 51.400 | 682.585 | 24.282 | 27.497 | 34.429 | 48.336 |
| weak_bridges | 16 | 4 | 74.258 | 81.636 | 68.468 | 64.464 | 70.002 | 96.098 | 90.020 |
| weak_bridges | 32 | 8 | 1017.567 | 1046.041 | 1019.583 | 1000.213 | 1017.147 | 1536.023 | 1500.832 |

## Elimination mechanism

Sizes are the largest SCC in each actual request solve. Anchor refreshes count warm eliminate_scc only (initial preparation excluded).

| Regime | States | Lean warm SCC sizes | Eliminated SCC sizes | Pruned action counts | Anchor refreshes |
| --- | ---: | --- | --- | --- | ---: |
| anchor_invalidation | 16 | [16] | [4] | [16] | 9 |
| anchor_invalidation | 32 | [32] | [4] | [32] | 9 |
| block_deletion | 16 | [4] | [4] | [16] | 0 |
| block_deletion | 32 | [4] | [4] | [32] | 0 |
| dense | 16 | [16] | [16] | [16] | 0 |
| dense | 32 | [32] | [32] | [32] | 0 |
| irrelevant_bridges | 16 | [16] | [4] | [16] | 0 |
| irrelevant_bridges | 32 | [32] | [4] | [32] | 0 |
| modular | 16 | [4] | [4] | [16] | 0 |
| modular | 32 | [4] | [4] | [32] | 0 |
| weak_bridges | 16 | [16] | [16] | [0] | 0 |
| weak_bridges | 32 | [32] | [32] | [0] | 0 |

## Interpretation limits

- Dataset seeds: [0, 1, 2]; timing repeats: [0, 1, 2]. Repeats reuse datasets. Synthetic medians only, no significance or universal-speedup claim.
- Lean global controls have no repair reverse index. Global elimination is classical strict-bound pruning plus SCC planning, not a new algorithm or a reproduction of FTVI's full heuristic search.
- Global pruning rescans all action rows after every request; its initial reward upper bound stays conservative after deletion. Scratch elimination can get a tighter bound from retained rewards. Anchor evaluation uses a single-action SCC graph; refreshed anchor values are charged.
- Legacy local pruning uses dense anchor preparation, keeps all actions inside the selected solve block, and disables pruning on anchor changes. Whole-method differences combine these choices with locality.
- Dense/SCC solvers and statistics helpers share code. Tiny enumeration tests supply an independent reference; this benchmark's large-model equality is not an independent theorem.
