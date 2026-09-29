# Research progress

Last updated: 2026-09-29. Repository: `BirariAditya/rl-trajectory-unlearning`.

## Current stage

**Phase 0–1 baseline validation is in progress.** The initial empirical-MDP learner and deletion repair are implemented. The new experiment harness and independent exact reference are implemented and tested. The sparse-certificate theorem, LSPI-control bridge, and neural experiments remain planned research.

## Stage tracker

| Stage | Status | Evidence / next gate |
| --- | --- | --- |
| Literature and novelty audit | Initial review complete | `literature_review.md`, `novelty_reassessment.md`, three `audits/` notes; CURe full text remains unresolved |
| Tabular data subtraction and planning | Implemented and tested | Retained-only rebuild comparisons, empty rows, repeated requests, ties, non-greedy predecessor regression |
| Independent retained-data reference | Implemented and tested | Exact rational policy enumeration for tiny MDPs, independently aggregated from logs; 20 random-model comparisons |
| Baselines and cost ledger | Implemented; pilot pending | No-op, scratch, warm-start, ordinary prioritized sweeping, global certified repair; total request time and separate preparation cost |
| Sparse decision certificate | Not started | Prove sound local verification with no hidden global scan; include hard cases and cost bounds |
| Frozen-feature LSPI control | Not started | Validate policy-dependent target corrections against retained-only replay |
| GPU IQL pilot | Not started | Dataset/learner identity, multiple retrains, strong fine-tune, matched controls, multi-attack audit |
| Full experiments and manuscript | Not started | Pass prior gates before expanding |

## Latest validation

- `python -m unittest discover -s tests -v`: **17 tests passed**.
- Added five complete-episode scenarios: shared-state reward deletion, non-policy row deletion, bottleneck propagation, coverage loss, and exact tie.
- Work counters include every Bellman target evaluation (including convergence/certificate scans), transition entries read, predecessor edges indexed, and log transitions applied. Wall time includes remaining Python work and fallback. These counters are not a complete machine-instruction or memory profiler.
- Rational reference uses decimal-string interpretations of logged rewards and discount; the production planner uses binary64. Comparisons therefore use declared tolerances, with exact ties checked separately.

## Next implementation tasks

1. Run and archive the three-seed correctness pilot; inspect reference agreement and certificate failures.
2. Add larger sparse/dense generated MDPs, repeat timings, and memory measurement before claiming speedups. Expand sequential-request scenarios and policy-iteration comparisons.
3. Formalize the changed-non-policy-row certificate and its numerical assumptions; prototype against the global and rational oracles.
4. Attempt the general sparse influence envelope and search for counterexamples before implementing the full proposed method.

## Limitations and open decisions

- Tiny synthetic results establish correctness checks, not benchmark superiority, novelty, or a deletion-time complexity theorem.
- The current repair still builds the predecessor index and scans all rows for its certificate. It has no sublinear request-time claim.
- Policy equality does not certify deletion of released critic, metadata, optimizer, or neural encoder state.
- The exact oracle enumerates policies and intentionally refuses more than 4,096 policies. It is validation overhead, not a competitive planning baseline.
- GPU experiments and the new sparse algorithm have not run. Conference acceptance is not predicted from prototype tests.

## Update convention

At the end of each implementation session, update this file with the current stage, actual commands/results, output paths, completed tasks, next tasks, and blockers. Mark a gate complete only when its evidence exists. Keep failed experiments and scope limitations visible. Do not silently upgrade a hypothesis to a result.

## Activity log

- **2026-09-29:** completed three-agent novelty reassessment and revised implementation phases.
- **2026-09-29:** implemented independent rational reference, ordinary prioritized-sweeping baseline, metered experiment harness, reproducible JSON manifest runner, and GitHub test workflow. All 17 tests passed; pilot and repository publication pending.
