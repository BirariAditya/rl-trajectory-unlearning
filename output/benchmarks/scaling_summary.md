# Sparse/dense scaling pilot

Source revision: `c6ce7331a94a6f9484de0305bf1c2099dd058909`; dirty at execution: `False`.

12 settings; 180 timed method runs and 60 separate memory-pass runs. Non-noop policy/convergence failures: **0**. Maximum updated Q error versus the recorded reference: `1.67e-10`.

Timings below are milliseconds: median of per-seed medians. Each seed's repeat distribution is preserved in the report. These instrumented CPU results use acyclic synthetic logs and do not establish general speedups or statistical significance.

| Graph | States | Scratch | Warm start | Prioritized sweeping | Global certified repair |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 16 | 5.665 | 4.112 | 5.852 | 6.065 |
| dense | 64 | 36.095 | 29.086 | 120.660 | 120.520 |
| sparse | 16 | 4.517 | 3.395 | 1.506 | 1.620 |
| sparse | 64 | 59.466 | 42.199 | 28.419 | 32.378 |

## Interpretation and scope

- Compare local queue methods with warm-started planning in every graph regime. Dense successor support can make queue propagation slower even when the final policy is correct.
- `certified_repair` is the existing global-residual method. The new rational restricted certificate is not part of this timing matrix; no performance claim for that prototype follows from this report.
- Models beyond the exact enumeration limit use tightly converged binary64 value iteration; inspect recorded residuals and action gaps. This is numerical fidelity, not a rational proof for the larger MDPs.
- Memory passes measure peak Python allocation during the request, excluding the resident model, original training, and native allocations. They are separate from the timing medians. Each timing run records preparation and original-training costs separately.
- Further gates: cyclic MDPs, sequential requests, more seeds/repetitions, warm-start policy iteration, and restricted-certificate preparation amortization.
