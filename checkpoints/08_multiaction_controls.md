# Checkpoint 08: multi-action fixed-feature LSPI controls

## Research question

Do exact selective and shared-anchor LSPI deletion repairs remain correct and
useful after removing the two-action, opposite-feature, and low-dimensional
shortcuts in the earlier synthetic controls?

This checkpoint uses three actions, unique ordered action-feature rows, no
within-state action pair related by exact sign reversal, dimensions 8 and 16,
and 10 sequential deletion requests. Every method is checked against direct
retained-only LSPI at every scheduled iteration. Equality covers the matrices,
right-hand sides, weights, policies, and terminal status of the declared
five-iteration learner.

## Reproduction

Source revision: `720e0a3b206e19583b6542f9223489c0d48ff432`, clean at experiment launch.

```powershell
python -m unittest discover -s tests -v
python scripts/run_lspi_multiaction.py --seeds 0 1 --repeats 1 --requests 10 --output results/lspi_multiaction.json.gz --summary results/lspi_multiaction_summary.md
```

The suite has 117 passing tests. The experiment contains 8 independently
generated dataset runs, 80 sequential requests, and 320 method executions. All
320 executions match the direct retained-only full trace. Lifecycle timing
charges original learning, method preparation, and every request. Reference
comparison and serialization are outside competitor timings.

## Lifecycle results

Median lifecycle milliseconds across the two dataset seeds:

| Regime | States | Dimension | Scratch retrain | Deduplicated inverse | Selective repair | Shared-anchor repair |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fragile | 32 | 8 | 8113.064 | 278.523 | 269.881 | 246.763 |
| fragile | 128 | 16 | 70241.789 | 7627.397 | 9508.872 | 7971.463 |
| stable | 32 | 8 | 27094.652 | 724.735 | 447.700 | 448.851 |
| stable | 128 | 16 | 106470.888 | 4347.826 | 3189.980 | 3178.098 |

On stable data, neither final policy changes across the 20 requests in each
size. Both repair methods certify every feature group without action scoring.
Selective repair is 1.62x faster than the matched deduplicated inverse control
at 32 states and 1.36x faster at 128 states. Shared-anchor repair gives 1.61x
and 1.37x respectively.

Fragile data expose the cost boundary. At 32 states, 5 of 20 final policies
change; selective repair reduces action scores from 4,608 to 2,325 and is only
1.03x faster, while shared-anchor repair scores 1,896 actions and is 1.13x
faster. At 128 states, every final policy changes. Selective repair reduces
scores from 23,040 to 21,369 but is 24.7% slower than the inverse control.
Shared-anchor repair reduces them to 19,557 and is 4.5% slower. Exact rational
matrix work, bound queries, and branch bookkeeping dominate the saved greedy
scores in this regime.

Every deletion changes the complete numerical trace even in stable cases,
because retained matrices and weights change while the final greedy policy can
remain fixed. Therefore the full-trace change count is distinct from the final
policy-change count.

## What this establishes

- Exact selective repair continues to match retained-only LSPI after adding a
  third action, unique non-opposite features, larger dimensions, and longer
  deletion sequences.
- Decision certificates can remove all greedy action scoring when margins are
  stable, and that reduction can improve full lifecycle time.
- Reduced action-score counts do not imply lower lifecycle time. The fragile
  128-state result is a concrete negative control against such a claim.
- Shared initial anchors can reduce rescoring on changed policy branches, but
  their index and bound-query costs must be charged.

## Limits and next gate

There is one timing execution for each of two independent seeds, so the medians
are descriptive and do not support a significance claim. The tasks use
synthetic fixed features and exact rational arithmetic. They do not establish
floating-point conditioning, neural-policy unlearning, privacy, historical
state erasure, a universal speedup, or publication novelty.

The next technical gate is a verified floating-point implementation with
conditioning diagnostics and residual/error bounds. Larger multi-action tasks
and additional independent timing repetitions should follow only after that
numerical contract is explicit.

Artifacts: [generated tables](../output/benchmarks/lspi_multiaction_summary.md)
and [compressed raw report](../output/benchmarks/lspi_multiaction.json.gz).
