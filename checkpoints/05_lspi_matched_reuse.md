# Checkpoint 05: LSPI comparison with equal policy-evaluation reuse

Completed 14 October 2026. **80 tests pass. 63 runs / 162 deletion requests / 810 method executions have zero exact trace mismatches.** Both memoized methods perform equal numbers of linear solves and action evaluations on every request. Memoized target replay has the lowest request and full-lifecycle medians in **five of seven** settings; memoized cached reconstruction wins the two larger switch fixtures. This is a conditional engineering result on small synthetic rational problems, not established novelty or a general practical speedup.

## What changed from Checkpoint 04?

[Checkpoint 04](04_lspi_cost.md) gave cached reconstruction an optimization that target replay did not have: reuse of repeated policy evaluations within a request. We now add `LSPIReplayMemoizedState` with exactly that cache policy. It preserves all four scheduled trace entries, resets its cache after every deletion, and still scores every action each iteration. [The contract](../audits/lspi_matched_reuse.md) explains the mutable assembly invariant, cycles and atomic failures.

The five-method experiment retains scratch, cached reconstruction, memoized cached reconstruction, and nonmemoized target replay, adding memoized target replay. Original training and preparation are charged to every method. Datasets, deletion schedules and iteration budgets follow the same seven settings as Checkpoint 04. Method order is randomized. Compare methods **within this new run**: absolute timings changed between sessions and should not be treated as a before/after speedup.

## Results

All 63 initial training comparisons and all 162 deletion comparisons match in exact matrices, RHS, weights, policy iterates and final/repeat statuses. Every timed request reaches a fixed point. Cycles, singularity, cancellation, invalid requests and deleting all data are covered by tests, not adverse timing regimes in this matrix.

Median milliseconds, using the same aggregation convention as [the complete generated table](../output/benchmarks/lspi_matched_checkpoint_summary.md):

| Setting | Cached memo request | Replay memo request | Cached memo lifecycle | Replay memo lifecycle |
| --- | ---: | ---: | ---: | ---: |
| Shared, 32 transitions, 4 features | 1.467 | **1.226** | 8.735 | **7.855** |
| Shared, 128 transitions, 4 features | 1.957 | **1.648** | 15.463 | **14.234** |
| Shared, 128 transitions, 8 features | 7.766 | **6.468** | 58.306 | **50.185** |
| Shared, 512 transitions, 4 features | 3.879 | **3.231** | 43.021 | **40.027** |
| Switch, 5 transitions, 4 features | 0.640 | **0.601** | 2.694 | **2.121** |
| Switch, 160 transitions, 4 features | **1.702** | 1.983 | **12.653** | 14.325 |
| Switch, 640 transitions, 4 features | **5.005** | 6.320 | **45.685** | 52.715 |

For the four shared-feature settings, request medians are roughly 16–17% lower with memoized replay. The smallest switch setting has a smaller request difference. At the individual-run median level, replay is faster in 7/9, 9/9, 9/9, 9/9, 7/9, 2/9 and 0/9 runs in the table's order. These are descriptive counts: timing repeats reuse datasets, and switch seeds only shuffle order inside the same deletion batch. They are not independent scientific replications or a significance test.

In shared settings, the median request has **two dense solves for both methods**, while cached reconstruction performs 12 successor outer-product terms and replay performs 3–4 corrections. Thus the solve-count imbalance from Checkpoint 04 has been removed. In the two larger switch settings, both perform three solves; replay performs one correction versus three reconstruction terms, yet its complete cost is higher. Fewer correction terms do not guarantee lower total cost. Profiling is needed to attribute the remaining difference to deletion preparation, matrix operations or other overhead.

## Boundaries and remaining gates

- Preparation still differs: C/b/G versus A(pi0)/b/G. Cached reconstruction removes exact-zero successor groups; replay retains them. This is a matched reuse comparison, not a pure instruction-level assembly ablation.
- Request time includes deletion validation, copied statistics, updates, replay, full action scoring and trace creation. Lifecycle adds original full-data training/preparation. Equality/hash/JSON work is excluded equally.
- Work counters omit allocation/copies and rational bit complexity. Peak memory is not measured; evaluation caches retain matrices and weights for up to the declared iteration budget.
- Only 4/8-dimensional rational features and tiny state spaces were tested. No stable numerical factorization/inverse-update baseline, floating-point conditioning study, long deletion stream, real-environment return or neural/GPU result exists yet.
- The matrix identity and memoization are conventional. These results justify another controlled experiment, not a novelty claim or an ICML/NeurIPS acceptance estimate.

## Provenance and reproduction

Source `b576ee4101cecc23369cfda5848e1c2253ef59d8` was **clean at launch**. All 28 recorded source/script hashes were verified after execution. [Source CI](https://github.com/BirariAditya/rl-trajectory-unlearning/actions/runs/38596853065) passes tests and all smoke runners, including matched reuse. The [67,985-byte compressed archive](../output/benchmarks/lspi_matched_checkpoint.json.gz) records datasets/features, request IDs, times, work counts, trace hashes and environment. Full traces are directly compared during execution; hashes, rather than every matrix, are persisted.

```powershell
Set-Location C:\Users\adity\Research
python -m unittest discover -s tests -v
python scripts/run_lspi_checkpoint.py --matched-reuse --seeds 0 1 2 --repeats 3 --output results/lspi_matched_checkpoint.json.gz
python scripts/summarize_lspi_checkpoint.py results/lspi_matched_checkpoint.json.gz
```

To regenerate the table without rerunning timings:

```powershell
python scripts/summarize_lspi_checkpoint.py output/benchmarks/lspi_matched_checkpoint.json.gz --output results/lspi_matched_archived_summary.md
```

## Next decision

Keep the matched memoized controls. Next measure memory separately from timing, add numerical factorization/conditioning comparisons, and expand deletion-sequence length and feature dimension. A useful algorithmic claim needs a deletion-specific work or stability result beyond incremental LSPI. For the 18 October presentation, preserve Checkpoint 04 as the reason this ablation was necessary, then show this conditional improvement and both losing regimes.
