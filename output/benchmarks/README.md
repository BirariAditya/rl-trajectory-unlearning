# Baseline correctness pilot

The subsequent [scaling summary](scaling_summary.md) and [compressed report](scaling.json.gz) cover repeated 16/64-state sparse/dense experiments. Rebuild that summary with `python scripts/summarize_scaling.py output/benchmarks/scaling.json.gz`. It includes the observed dense-case slowdown; the table below describes only the earlier tiny correctness pilot.

Run date: 2026-09-29. Source revision: `bd37c7927c98494187ed9f20debadc8c2d6e76d4`.

[phase1.json](phase1.json) contains five scenarios × three seeds × five methods = 75 runs. All 60 updating runs converged and matched the independent exact retained-policy reference. Maximum Q error was below `5.56e-17`. No-op is an intentional negative control and is not expected to match after behavior-changing deletions.

| Scenario | No-op policy disagreement, seed 0 | Updating baseline disagreement | Global repair certificate |
| --- | --- | --- | --- |
| Shared-state reward deletion | 100% | 0% | Passed |
| Non-policy-row deletion | 0% | 0% | Passed |
| Bottleneck chain | 100% | 0% | Passed |
| Lost row coverage | 100% | 0% | Passed |
| Exact action tie | 100% | 0% | Inconclusive; fallback |

Timing uses one instrumented execution per case/method on tiny synthetic MDPs. These numbers are **not evidence of algorithmic speedup**. They validate the measurement plumbing and record global verification overhead. Dataset selection/aggregation is included for scratch retraining; full-model preparation is reported separately for deletion methods; original training cost is also recorded. No-op intentionally performs no model repair or deletion. Required solver checks are charged inside request time, whereas the independent exact oracle and accuracy comparisons are common offline evaluation overhead. Memory, repeated timing uncertainty, amortized preprocessing, larger sparse/dense cases, and a warm-start policy-iteration baseline remain future work.

Reproduce from the repository root with Python 3.10+:

```powershell
python -m unittest discover -s tests -v
python scripts/run_baselines.py --output results/phase1.json
```

Timings and machine metadata vary. Policy agreement and declared tolerances should reproduce. New runs record their own Git commit, dirty status, and source hashes rather than pretending to be the archived run.
