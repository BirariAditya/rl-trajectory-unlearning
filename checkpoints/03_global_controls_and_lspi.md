# Checkpoint 03: classical controls change the research priority

Reached 7 October 2026. Global experiment source: `51a999a2519ada3b95e08de9d380a1233a109bbf`, clean at launch. [Full tables](../output/benchmarks/global_checkpoint_summary.md), [raw archive](../output/benchmarks/global_checkpoint.json.gz), [global contract](../audits/global_elimination_contract.md), [LSPI example](lspi_control_example.md).

## Completed work

The new lean global planner maintains empirical statistics without local-repair reverse indexes. A classical action-elimination variant scans all rows, removes only provably suboptimal actions, decomposes the admissible graph, and refreshes its anchor bound when required. It restores newly competitive actions and solves from candidate retained statistics before committing deletion. Both warm and scratch controls are now charged appropriately; full lifecycle totals include original full-data training once for every method.

Separately, fixed-feature LSPI now has two exact rational paths: direct per-transition retained-only matrix assembly and aggregate replay with deletion and policy-target corrections. Both run the same finite schedule from a data-independent initial policy. This provides a correctness substrate for the next research direction, not a new complexity result.

## Validation

- **60 tests pass.** This includes seven global-planner tests and eleven LSPI tests. Randomized global variants cover 288 deletion/method checks against independent policy enumeration. LSPI random shared-feature tests compare every matrix, RHS, weight and policy on 25 datasets with three sequential requests each.
- **108 global benchmark runs, 594 requests, 4,158 method executions; zero policy/value mismatches.** Six regimes, 16/32 states, three dataset seeds and three timing repeats. All seven methods match lean scratch rational SCC planning.
- [CI on the experiment source](https://github.com/BirariAditya/rl-trajectory-unlearning/actions/runs/37545210432) passed 59 tests and all smoke/demo runners. The final 60th test additionally checks a singular LSPI matrix after a successful first policy switch, with no partial deletion commit.
- Independent code review found no algebraic/soundness defect in global elimination or LSPI replay. Shared solvers mean large-run agreement is not an independent proof; tiny enumeration and hand-calculated fixtures supply additional checks.

## Result that changes our priority

These are 32-state median lifecycle times in milliseconds, including original training, preparation, and every deletion request.

| Regime | Local SCC | Current pruned-local SCC | Lean warm SCC | Classical elimination SCC | Lean scratch SCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Modular | 7.930 | 51.400 | 24.282 | 27.497 | 34.429 |
| Dominated bridges | 785.555 | 96.257 | 769.006 | 32.969 | 785.993 |
| Anchor invalidation | 790.315 | 835.267 | 783.694 | 39.642 | 795.262 |
| Batched deletion | 7.774 | 56.832 | 11.044 | 12.272 | 12.192 |
| Weaker bridges | 1017.567 | 1046.041 | 1000.213 | 1017.147 | 1536.023 |
| Dense | 566.784 | 620.498 | 571.815 | 581.265 | 663.059 |

The classical elimination control is about 2.9x faster than current pruned-local repair over the dominated-bridge lifecycle, and about 21x faster in the anchor-invalidation regime. It preserves four-state components after refreshing the bound instead of disabling pruning. Our current local implementation still has cheaper individual bridge requests (0.674 ms versus 3.257 ms), but its preparation cost outweighs that advantage in these eight-request sequences.

Local repair retains a useful modular-case benefit: 7.930 ms versus 24.282 ms for lean warm SCC, about 3.1x. This is consistent with reusing unaffected components and is not, by itself, evidence of a new algorithmic contribution. Dense and weak-bridge cases remain global and show no convincing locality benefit.

These are comparisons of implemented methods, not a pure locality ablation: the global elimination implementation also uses more efficient anchor evaluation and prunes inside SCC construction; the current pruned-local method does neither. The former is not a complete FTVI reproduction or an optimized external library. Synthetic timings, correlated requests, and repeated datasets do not establish broad performance or statistical significance.

## LSPI result and boundary

In the [two-state example](lspi_control_example.md), full-data policy is `(a,a)`. A one-shot sample downdate gives `(b,b)`, while the declared retained-only three-step LSPI learner gives `(a,b)`. Correcting the retained successor-policy target reproduces `(a,b)` and all intermediate matrices and weights exactly. The JSON [reproduction](../output/benchmarks/lspi_demo.json) stores the dataset, feature map, hyperparameters, traces, and source hashes.

Tests also cover shared features, policy cycles, terminal masking, ties, sequential/union deletion, and singular evaluations. A singular result is a failed request with no data commit. Equality concerns the finite learner's trace/policy/weights, not its work metadata, optimality, privacy, or neural checkpoints. The implementation still uses dense rational refactorization each step and scans successor groups; no measured LSPI speedup is claimed.

## Updated research phases

1. **Completed: strengthen and stress the tabular controls.** Keep them as exact references and a failure taxonomy. Stop treating the current local/pruning mechanism as the intended paper contribution. [FTVI](https://arxiv.org/abs/1401.3910) already combines action elimination and graph decomposition.
2. **Completed initial gate: fixed-feature LSPI correctness.** The new prototype handles policy switches and matches a specified retained-only finite schedule. The matrix identity alone is established algebra; original [LSPI](https://www.jmlr.org/papers/v4/lagoudakis03a.html) already studies incremental policy evaluation.
3. **Next: LSPI complete-cost and prior-art gate.** Compare direct reconstruction, cached sufficient-statistic reconstruction, target-correction replay, and incremental/refactorized solves. Vary retained log size, feature dimension, successor-group count, deletion size, and policy switches. Include all preparation, copies, global policy scoring, conditioning failures, and matrix solve costs. A win over per-sample scratch alone is insufficient.
4. **Then: theory or reframe.** Seek a substantive deletion-dependent or policy-switch-dependent bound beyond standard incremental LSPI. If the advantage reduces to old caching/matrix updates, narrow the contribution to a rigorous evaluation/failure result or select a different learner before more scale-up.
5. **Conditional: neural/GPU transfer.** Begin only after the restricted-learner novelty/efficiency gate. Learned encoders, full-model deletion and privacy claims require separate evidence.

This checkpoint improves the project by rejecting an overstated tabular story and establishing a concrete next implementation. A paper-level novelty result remains open.
