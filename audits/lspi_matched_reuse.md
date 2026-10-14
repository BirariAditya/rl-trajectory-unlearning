# Matched policy-evaluation reuse in exact LSPI

14 October 2026. This is a conventional implementation control, not a new algorithm or theorem. The learner is the same fixed-feature, exact-rational, finite-schedule learner defined in [the LSPI contract](lspi_control_contract.md).

## Why add this control?

[Checkpoint 04](../checkpoints/04_lspi_cost.md) compares memoized cached reconstruction with target replay that performs every solve. This cannot isolate the advantage of correcting targets: ordinary memoization may explain part of the cost difference. `LSPIReplayMemoizedState` adds the identical per-request evaluation cache to replay. The two memoized methods must now have equal linear-solve and greedy-action-score counts for each request.

## Correctness invariant

For a fixed retained dataset and input policy, the evaluation matrix and right-hand side are deterministic. An invertible exact solve therefore has the same weights whenever that policy repeats. Store immutable matrix, RHS and weights under the full policy tuple. A hit skips assembly/solve but still computes greedy action scores and records the declared trace step. Reset the cache for each replay/request: deletion changes the dataset.

Replay assembly also has a mutable matrix and a `previous` policy. Both refer to the **last actual assembly**, not the last scheduled trace step. On a cache hit, neither changes. On a subsequent assembly, successor-target corrections transform that matrix from its associated policy to the requested policy. This keeps the assembly state consistent even though cached trace entries need not describe the mutable matrix. Under the deterministic learner, once a policy repeats the future policy sequence repeats as well; retaining the invariant avoids relying on an implicit cycle shortcut.

The cache exists only within `_run`; a failed singular evaluation does not commit copied deletion statistics or trajectory removals. Cycle and budget statuses retain their original finite-schedule meaning. A hit does not modify the iteration count, initialization, tie order or learner's output.

## Validation and remaining differences

The memoized replay tests compare every matrix, RHS, weight, policy and repeat status against per-transition scratch and memoized cached reconstruction. Cases cover fixed points, cycles, a later singular solve, deletion reset, restored successor aggregates, deleting all data, invalid IDs, and 32 randomized shared-feature datasets with four requests each (including a no-op). Solve/action-score equality is also asserted in the five-method experiment runner.

The methods still differ in preparation: cached reconstruction stores C/b/G, while replay stores A(pi0)/b/G. Cached reconstruction removes exact-zero successor groups; replay retains them. Both use the same dense rational solver and full action scoring. These facts prevent interpreting wall-time differences as a pure matrix-assembly microbenchmark. All preparation and initial learning remain charged. Memoization can store up to K matrices/weights; current timings do not profile memory or rational-number bit complexity.

Use `--matched-reuse` in [the runner](../scripts/run_lspi_checkpoint.py). Default four-method runs remain available. The summary reads the method list from each report, with a fallback for the historical Checkpoint 04 archive. Report the new results separately; do not replace historical timings or claim the earlier source hashes describe the new implementation.
