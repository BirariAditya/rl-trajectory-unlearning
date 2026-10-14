# Exact LSPI cost checkpoint

Source: `b576ee4101cecc23369cfda5848e1c2253ef59d8`; clean at launch: `True`.

**63 runs, 162 requests, 810 method executions, 0 trace mismatches.**
Reference request statuses: {'fixed_point': 162}. Singular outcomes are rejected requests, not successful unlearning.

## Request cost (milliseconds)

Median across runs of each run's median request time. Exact matrix/RHS/weight/policy traces and failure statuses are compared; costs deliberately differ. Trace creation is timed; equality/hashing/JSON are common evaluation overhead.

| Regime | Count parameter | Initial transitions | Features | scratch | cached | cached_memo | target_replay | target_replay_memo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shared | 16 | 32 | 4 | 9.295 | 2.335 | 1.467 | 1.571 | 1.226 |
| shared | 64 | 128 | 4 | 30.612 | 2.825 | 1.957 | 2.037 | 1.648 |
| shared | 64 | 128 | 8 | 114.778 | 12.900 | 7.766 | 9.832 | 6.468 |
| shared | 256 | 512 | 4 | 125.308 | 4.226 | 3.879 | 3.607 | 3.231 |
| switch | 1 | 5 | 4 | 1.793 | 1.029 | 0.640 | 0.993 | 0.601 |
| switch | 32 | 160 | 4 | 36.182 | 1.868 | 1.702 | 2.089 | 1.983 |
| switch | 128 | 640 | 4 | 148.961 | 5.078 | 5.005 | 6.352 | 6.320 |

## Preparation plus every request (milliseconds)

Includes initial full-data learning for every method, plus statistic preparation for cached/replay methods. All requests start from the same initial policy and run four declared iterations. Cached memoization preserves those four trace entries while reusing evaluations of repeated policies within a request.

| Regime | Count parameter | Features | Requests/run | scratch | cached | cached_memo | target_replay | target_replay_memo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shared | 16 | 4 | 3 | 37.576 | 13.721 | 8.735 | 8.692 | 7.855 |
| shared | 64 | 4 | 3 | 131.244 | 19.816 | 15.463 | 15.783 | 14.234 |
| shared | 64 | 8 | 3 | 496.946 | 74.757 | 58.306 | 63.945 | 50.185 |
| shared | 256 | 4 | 3 | 514.165 | 47.609 | 43.021 | 42.942 | 40.027 |
| switch | 1 | 4 | 2 | 6.211 | 3.248 | 2.694 | 3.040 | 2.121 |
| switch | 32 | 4 | 2 | 125.718 | 14.231 | 12.653 | 14.966 | 14.325 |
| switch | 128 | 4 | 2 | 491.323 | 52.291 | 45.685 | 54.358 | 52.715 |

## Work diagnostics: median per request

Terms are dense outer-product operations, not constant-time work. These counters omit copies, validation, bit complexity and cache allocation. Cached memoization stores up to K matrices/weights per request; memory is not measured here.

| Regime | Count | Features | Cached target terms | Replay corrections | Memo solves | Replay solves |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| shared | 16 | 4 | 24 | 4 | 2 | 4 |
| shared | 64 | 4 | 24 | 3 | 2 | 4 |
| shared | 64 | 8 | 24 | 3 | 2 | 4 |
| shared | 256 | 4 | 24 | 4 | 2 | 4 |
| switch | 1 | 4 | 4 | 1 | 2.5 | 4 |
| switch | 32 | 4 | 4 | 1 | 3 | 4 |
| switch | 128 | 4 | 4 | 1 | 3 | 4 |

## Matched reuse diagnostics

Both memoized methods use the same evaluation cache policy, dense solver and full action scoring. Caches reset for each deletion. All scheduled trace entries remain. Preparation/anchor assembly and zero-group handling still differ; this is not a pure instruction-level assembly ablation.

| Regime | Count | Features | Cached memo terms | Replay memo corrections | Cached memo solves | Replay memo solves |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| shared | 16 | 4 | 12 | 4 | 2 | 2 |
| shared | 64 | 4 | 12 | 3 | 2 | 2 |
| shared | 64 | 8 | 12 | 3 | 2 | 2 |
| shared | 256 | 4 | 12 | 4 | 2 | 2 |
| switch | 1 | 4 | 2.5 | 1 | 2.5 | 2.5 |
| switch | 32 | 4 | 3 | 1 | 3 | 3 |
| switch | 128 | 4 | 3 | 1 | 3 | 3 |

Requests with unequal memoized solve/action-score counts: **0**.

## Interpretation boundary

- Seeds [0, 1, 2]; repeats [0, 1, 2]. Shared-feature seeds generate different logs. Switch seeds only shuffle deletion order inside an otherwise identical batch; these are not independent datasets.
- Count is repetitions of the four-trajectory switch fixture, or the number of two-transition shared-feature trajectories. Shared cases use six states and a fixed feature map independent of log seed.
- Cached reconstruction is established sufficient-statistic algebra. Cached memoization is ordinary reuse of an identical policy's evaluation within a request. Neither is a new algorithm. No incremental-inverse, measured-memory, large-dimensional floating-point or neural comparison is included.
- Cached methods omit exact zero successor groups, restoring them on deletion when needed; target replay retains zero groups. This small engineering difference is charged in total time.
- Synthetic exact-rational timings and partial work counts do not establish practical-scale speedups, an asymptotic theorem, privacy or novelty. No statistical significance claim is made.
