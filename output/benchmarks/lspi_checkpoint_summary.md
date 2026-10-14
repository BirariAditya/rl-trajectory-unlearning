# Exact LSPI cost checkpoint

Source: `475330cc3795e016d55babe089d398dded542aee`; clean at launch: `False`.

**63 runs, 162 requests, 648 method executions, 0 trace mismatches.**
Reference request statuses: {'fixed_point': 162}. Singular outcomes are rejected requests, not successful unlearning.

## Request cost (milliseconds)

Median across runs of each run's median request time. Exact matrix/RHS/weight/policy traces and failure statuses are compared; costs deliberately differ. Trace creation is timed; equality/hashing/JSON are common evaluation overhead.

| Regime | Count parameter | Initial transitions | Features | scratch | cached | cached_memo | target_replay |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shared | 16 | 32 | 4 | 5.742 | 1.476 | 0.959 | 0.986 |
| shared | 64 | 128 | 4 | 20.955 | 1.852 | 1.286 | 1.309 |
| shared | 64 | 128 | 8 | 75.110 | 8.735 | 5.027 | 6.174 |
| shared | 256 | 512 | 4 | 79.951 | 2.826 | 2.206 | 2.455 |
| switch | 1 | 5 | 4 | 1.126 | 0.570 | 0.406 | 0.510 |
| switch | 32 | 160 | 4 | 24.066 | 1.181 | 1.073 | 1.337 |
| switch | 128 | 640 | 4 | 90.776 | 3.249 | 3.116 | 3.935 |

## Preparation plus every request (milliseconds)

Includes initial full-data learning for every method, plus statistic preparation for cached/replay methods. All requests start from the same initial policy and run four declared iterations. Cached memoization preserves those four trace entries while reusing evaluations of repeated policies within a request.

| Regime | Count parameter | Features | Requests/run | scratch | cached | cached_memo | target_replay |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shared | 16 | 4 | 3 | 24.165 | 7.937 | 5.305 | 5.752 |
| shared | 64 | 4 | 3 | 88.661 | 13.031 | 12.052 | 10.711 |
| shared | 64 | 8 | 3 | 316.650 | 49.538 | 39.948 | 40.955 |
| shared | 256 | 4 | 3 | 325.874 | 31.074 | 27.589 | 25.858 |
| switch | 1 | 4 | 2 | 3.738 | 1.999 | 1.208 | 1.698 |
| switch | 32 | 4 | 2 | 78.316 | 8.084 | 8.397 | 9.344 |
| switch | 128 | 4 | 2 | 303.938 | 29.215 | 28.239 | 35.111 |

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

## Interpretation boundary

- Seeds [0, 1, 2]; repeats [0, 1, 2]. Shared-feature seeds generate different logs. Switch seeds only shuffle deletion order inside an otherwise identical batch; these are not independent datasets.
- Count is repetitions of the four-trajectory switch fixture, or the number of two-transition shared-feature trajectories. Shared cases use six states and a fixed feature map independent of log seed.
- Cached reconstruction is established sufficient-statistic algebra. Cached memoization is ordinary reuse of an identical policy's evaluation within a request. Neither is a new algorithm. No incremental-inverse, measured-memory, large-dimensional floating-point or neural comparison is included.
- Cached methods omit exact zero successor groups, restoring them on deletion when needed; target replay retains zero groups. This small engineering difference is charged in total time.
- Synthetic exact-rational timings and partial work counts do not establish practical-scale speedups, an asymptotic theorem, privacy or novelty. No statistical significance claim is made.
