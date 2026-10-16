# Checkpoint 07: exact decision repair with stronger feature controls

16 October 2026. **114 tests pass; 81 runs, 972 deletion requests and 6,804
method executions have zero counterfactual mismatches.** This is a measured
algorithmic extension, not confirmed publication novelty. No GPU or real-task
return experiment has run.

## What changed

Checkpoint 06 certified that the entire policy sequence remained unchanged;
an inconclusive bound triggered retained-only retraining and new anchors. The
new methods first compute exact retained weights through the classical inverse
engine, then use an indexed margin test to skip provably unchanged decisions.
Uncertain decisions are scored exactly, including decisions that really change.
The default output matches every retained-only matrix, RHS, weight and policy
iterate, not just the final actions.

`LSPISelectiveRepairState` keeps an immutable margin anchor per input policy.
`LSPISharedAnchorRepairState` freezes the initial schedule's bank of anchors and
can use any of them on a new policy branch. Greedy choices depend on the current
weights and fixed features, not on which input policy produced those weights.
The bank therefore avoids building new margin indexes on subsequent branches.
It does not avoid evaluating those branches' retained matrices and weights.

For an anchor weight w0, action-feature difference v and its greedy action,
store the radius `rho = min v^T w0 / ||v||_1`. At exact new weights w, let
`delta = ||w-w0||_infinity`. A sorted index selects only groups with
`rho <= delta` for rescoring. Outside that set, Holder's inequality proves the
old action still wins. Equality is rescored because ties may change the chosen
action; zero drift preserves the entire old greedy map. Full proof, cost
accounting and prior art: [audit](../audits/decision_repair_review.md).

Historical anchors remain internal. Full released trace equality is **not**
complete-state erasure or a privacy guarantee. Full policy copies and hashes
still cost O(number of states); fewer action dot products do not imply a
state-independent request cost.

## Fairness controls and protocol

Seven methods share the same exact retained-data reference: inverse reuse,
feature-deduplicated inverse reuse, its policy-only version, the old cumulative
certificate, selective repair, its policy-only version, and shared-anchor
repair. The deduplicated control scores identical ordered action-feature rows
once. Policy-only controls genuinely omit construction of matrix/weight output
trace entries, while keeping exact internal evaluations.

Nine settings cover stable/fragile rewards, aliased/unique features, 16/128
states, plus the two-state switch fixture. The larger fixtures have 256
two-transition trajectories, four features, two actions and four scheduled
iterations. Features have mixed signs and state-dependent preferred actions.
The unique 128-state fixtures have 256 distinct state-action vectors and 128
distinct ordered action rows. Their construction remains deliberately simple;
uniqueness alone does not make a realistic benchmark.

Three dataset seeds, three timing repeats and twelve sequential deletion
requests give 81 runs. Every method pays original training, feature indexes,
margin preparation, request copies, matrix work and fallbacks. Method order is
randomized. Scratch reference evaluation and result serialization are outside
timing. A separate seed-0 allocation pass covers all 63 case/method pairs,
including preparation, while excluding pre-existing input objects.

Stable paired fixtures share state/action draws and deletion IDs, but teacher
rewards change slightly with features. Fragile paired fixtures share rewards
as well. Timing repeats are not independent scientific datasets. Switch seeds
only reorder a batch of the same deterministic fixture.

## Results that survive the controls

Milliseconds below are median full lifecycle costs, including twelve requests.
The matched full-trace comparator is deduplicated inverse reuse.

| Setting | Matched inverse | Selective repair | Shared-anchor repair | Policy-only certificate |
| --- | ---: | ---: | ---: | ---: |
| Stable, unique, 128 states | 126.722 | 62.653 | 62.033 | 55.017 |
| Fragile, unique, 128 states | 139.231 | 123.163 | 103.918 | 569.484 |
| Stable, aliased, 128 states | 50.134 | 46.175 | 46.553 | 51.000 |
| Fragile, aliased, 16 states | 41.553 | 43.453 | 42.334 | 137.353 |
| Two-state switch | 16.440 | 17.319 | 17.385 | 14.799 |

The stable unique case supports a **2.04x** shared-anchor lifecycle improvement
over the matched full-trace control. The fragile unique case supports **1.34x**;
the slightly faster ordinary inverse control there costs 137.671 ms, giving
**1.32x** against the fastest full-trace conventional control. This second result
matters because some retained policies actually change. It is not evidence that
every individual changed-policy request is faster.

In fragile unique 128-state runs, final policies change in 15/108 repeated
requests and full policy sequences in 18/108. With timing repeats removed, these
are 5/36 and 6/36 dataset-seed/request instances. Shared-anchor action scores
total 21,966 versus 61,440 for deduplicated inverse reuse, **64.2% fewer**.
It builds zero new margin groups after preparation; per-policy repair builds
2,688 across these repeated requests. Stable unique repair scores no actions
during requests while still producing exact retained weights and matrices.

Across the entire experiment, 42 repeated requests change final policies and
57 change sequences. All seven methods agree on policy sequence/status; all
four full-trace methods additionally agree on every trace entry. The archive
records hashes and match indicators, not a complete dump of all reference
matrices. Unit tests separately exercise cycles, singular rejection, equality
ties, zero drift, immutable-anchor cumulative drift and randomized sequential
deletions.

## What weakens the headline

The old certificate's apparent advantage is sensitive to aliases. In stable
aliased 128-state data, deduplicating the conventional baseline reduces lifecycle
cost from 125.980 to 50.134 ms; the certificate costs 51.000 ms. Thus part of the
earlier pattern was avoidable duplicate scoring. In stable unique data, the
certificate remains useful under its narrower output contract.

Shared-anchor repair loses on the tiny switch fixture and fragile aliased
16-state fixture. It can rescore every group if all margins are fragile or its
fixed anchors become stale. No adaptive refresh or universal advantage is
claimed. On fragile unique 128-state data, median run-maximum request latency
is 7.302 ms for the shared bank, 13.782 ms for deduplicated inverse reuse and
14.020 ms for per-policy repair; these are descriptive tails, not guarantees.

Memory increases: stable unique 128-state peak traced allocation is **171.0 KiB
versus 121.6 KiB** for deduplicated inverse reuse. Fragile unique peaks are
**170.8 versus 121.1 KiB**. These exclude shared inputs and are not process RSS.
Exact rational arithmetic and four-dimensional synthetic features remain major
external-validity limitations. There is no statistical-significance claim.

## Research decision

Keep shared-anchor repair as the strongest current **full-trace** candidate in
the larger unique-feature tests. It fixes the old certificate's restriction to
unchanged sequences and avoids new margin indexes on policy branches. However,
incremental LSPI, inverse updates and prediction sensitivity after removal all
have prior art, including [Okumura et al., 2015](https://arxiv.org/abs/1504.02870).
The margin lemma is elementary. This checkpoint establishes usefulness in a
restricted test, not a substantially novel algorithm or an ICML/NeurIPS claim.

Next gates: (1) freeze and explain this evidence for 18 October; (2) test larger
non-aliased feature dimensions, non-opposite multi-action features, longer
sequences and conditioning under verified floating-point computations; (3)
seek a deletion-specific work theorem or a clear separation from optimized
incremental-control methods; (4) only then spend on real-task/GPU experiments.
If these gates fail, report the evaluation/failure findings rather than relabel
classical machinery as new.

## Reproduction and provenance

Clean source: `1fb3e300aaee97786be1b115c45b03606df042d2`. All **34** recorded
source/script SHA-256 values were verified against the run's implementation.
The source review found no actionable defect and all 114 local tests passed.
Remote CI status was not confirmed at archive time because the GitHub API
request timed out; do not infer CI success from local success.

```powershell
python -m unittest discover -s tests -v
python scripts/run_lspi_unique_controls.py --seeds 0 1 2 --repeats 3 --requests 12 --memory --output results/lspi_unique_controls.json.gz
python scripts/run_lspi_unique_controls.py --summarize output/benchmarks/lspi_unique_controls.json.gz --summary results/regenerated_checkpoint07.md
```

[Complete tables](../output/benchmarks/lspi_unique_controls_summary.md),
[compressed archive](../output/benchmarks/lspi_unique_controls.json.gz),
[implementation](../src/rl_unlearning/lspi_decision_repair.py),
[tests](../tests/test_lspi_decision_repair.py).
