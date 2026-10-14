# Checkpoint 04: exact LSPI against cached controls

Completed 13 October 2026. **74 tests pass; 63 runs cover 162 deletion requests and 648 method executions, with zero exact trace mismatches.** The stronger conventional control has the lowest request-time median in all seven settings. Target-correction replay has the lowest measured full-lifecycle median in two settings. This is a completed correctness and comparison checkpoint, not a novel algorithm or a general speedup.

## Question and methods

Can removing selected complete trajectories avoid repeatedly reading retained transitions while reproducing a specified retained-only fixed-feature LSPI learner? Every method uses the same rational features, initial policy, action tie order, regularization, discount, and four-iteration schedule. Each request restarts from that initial policy. Terminal successors contribute zero continuation features.

| Method | What is reused | Work still required |
| --- | --- | --- |
| `scratch` | Nothing across requests | Filter retained logs; assemble each iteration from all retained transitions; solve and score all actions |
| `cached` | Retained sufficient statistics C, b, G | Subtract deletion contributions; reconstruct each policy-dependent matrix from successor groups; solve and score all actions |
| `cached_memo` | C, b, G plus evaluations of policies already encountered **within this request** | Same deletion update; reconstruct/solve distinct policies; still score and record every declared iteration |
| `target_replay` | Matrix at the initial policy, b, G | Subtract deletion contributions; correct changed successor targets; copy matrices, scan groups, solve and score each iteration |

Cached reconstruction and memoization are conventional optimizations. For fixed retained data, a repeated policy has exactly the same matrix, right-hand side, and solution. Reusing its evaluation preserves all scheduled trace entries. That cache is reset after every deletion; a matrix from the previous dataset cannot be silently reused. The replay implementation currently does not memoize repeated policy evaluations, so this is a comparison of implemented methods, not an isolated ablation of matrix assembly.

The [contract](../audits/lspi_control_contract.md) and [worked two-state example](lspi_control_example.md) explain why one-shot sample subtraction can return the wrong final policy. The present experiment asks whether correcting that issue gives an advantage beyond ordinary cached learning.

## Results

All 63 initial training comparisons and all 162 deletion comparisons have identical trace hashes across the four methods. During execution, the runner directly compares every exact matrix, right-hand side, weight, input/output policy, and final status. All timed requests reached a fixed point within the declared schedule. Cycles and singular rejection are covered by unit tests, **not** timed adverse cases in this matrix. A rejected singular request is not successful deletion.

Selected medians in milliseconds are below. Full seven-setting results and work counters are in the [generated summary](../output/benchmarks/lspi_checkpoint_summary.md).

| Setting | Scope | Scratch | Cached | Cached + memo | Target replay |
| --- | --- | ---: | ---: | ---: | ---: |
| Shared, 128 transitions, 4 features | Request | 20.955 | 1.852 | **1.286** | 1.309 |
| Shared, 128 transitions, 4 features | Training + 3 requests | 88.661 | 13.031 | 12.052 | **10.711** |
| Shared, 128 transitions, 8 features | Request | 75.110 | 8.735 | **5.027** | 6.174 |
| Shared, 128 transitions, 8 features | Training + 3 requests | 316.650 | 49.538 | **39.948** | 40.955 |
| Switch, 640 transitions, 4 features | Request | 90.776 | 3.249 | **3.116** | 3.935 |
| Switch, 640 transitions, 4 features | Training + 2 requests | 303.938 | 29.215 | **28.239** | 35.111 |

Memoized cached evaluation has the lowest request median in **7/7** settings. Replay has the lowest lifecycle median in **2/7**: shared 128-transition/4-feature and shared 512-transition/4-feature cases. Those lifecycle results include preparation and original training, so their order need not match request-only order. They do not demonstrate statistical significance.

Replay reduces dense outer-product terms: shared cases typically use 3–4 corrections per request against 24 target terms for cached reconstruction. However, cached memoization typically performs two dense solves versus four for replay. Fewer matrix-update terms alone therefore do not establish lower total latency. Terms themselves scale with feature dimension; they are not constant-time operations.

## Experimental scope and cost accounting

- Seven settings, three seeds, three timing repeats. Shared-feature cases have six states, 4 or 8 features, and 16/64/256 two-transition trajectories. Switch cases repeat the four-trajectory reversal fixture 1/32/128 times.
- Shared-feature seeds produce different logs with a fixed feature map. Switch seeds only change deletion order within an otherwise identical batch; they are not independent datasets. Timing repeats reuse data.
- Method order is randomized. Original full-data training is charged to every method, including scratch. Cached statistics, deletion validation/updates, matrix copies, policy scans, solves, and common trace creation are inside measured calls. Common equality checks, hashing, and serialization are outside them.
- Request numbers are medians across runs of each run's median request latency. Lifecycle numbers are medians of each run's original training plus all deletion requests.
- Exact rational arithmetic, small dimensions, and shared schedule/solver code limit interpretation. This is not a practical-scale numerical LSPI benchmark or an independent proof for all inputs.
- Cached methods omit exact-zero successor groups and restore them if deletion breaks cancellation. Replay retains zero groups. This engineering difference is charged but not isolated.
- Memory is not measured. Memoization may store up to K matrices and weights within a request. Incremental inverses/refactorization, numerical conditioning, large dimensions, neural representations, GPU throughput, real-environment return, and privacy attacks remain untested here.

## Reproduction and provenance

Source revision: `475330cc3795e016d55babe089d398dded542aee`. [Source CI](https://github.com/BirariAditya/rl-trajectory-unlearning/actions/runs/38338893080) passed tests and smoke runners. The report records `git_dirty=true`: untracked learning-guide/presentation Markdown was being prepared at launch. All **28 recorded source/script SHA-256 hashes** were checked against the source files after execution with no mismatch. Preserve the dirty flag; do not relabel this as a clean launch.

The lossless [raw archive](../output/benchmarks/lspi_checkpoint.json.gz) is 62,114 bytes. It records environment, arguments, source hashes, full repeat-zero datasets/features, deletion IDs, work counters, times, statuses, and trace digests. It stores trace digests rather than every matrix; rerunning the recorded learner/data reconstructs the traces.

```powershell
Set-Location C:\Users\adity\Research
python -m unittest discover -s tests -v
python scripts/run_lspi_checkpoint.py --seeds 0 1 2 --repeats 3 --output results/lspi_checkpoint.json.gz
python scripts/summarize_lspi_checkpoint.py results/lspi_checkpoint.json.gz
```

To regenerate the archived table without rerunning timings:

```powershell
python scripts/summarize_lspi_checkpoint.py output/benchmarks/lspi_checkpoint.json.gz --output results/lspi_archived_summary.md
```

## Research decision

Do not promote target corrections alone as a paper contribution. Keep exact replay, cached reconstruction, and memoization as the baseline suite. Any continuation should first isolate the assembly mechanism under equal repeated-policy reuse, add a stable numerical factorization comparator, and measure preparation, memory, failures, and long deletion sequences. Such implementation improvements are still not novelty by themselves.

A paper-level next claim needs a deletion-specific guarantee or demonstrable advantage beyond established incremental LSPI, with a stated learner/output and conditioning assumptions. Stop or reframe that claim if it does not survive these controls. For 18 October, present the complete research journey, exact counterexample, verified implementation, and negative comparisons; GPU scaling remains a subsequent, separately justified phase.
