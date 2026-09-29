# Research progress

Last updated: 2026-09-29. Repository: [BirariAditya/rl-trajectory-unlearning](https://github.com/BirariAditya/rl-trajectory-unlearning) (private).

## Current stage

**Broader baseline validation and Phase 3A special-case prototype.** A restricted rational certificate for unchanged chosen-policy rows is implemented, with proof notes and randomized oracle checks. The scaling runner now supports larger acyclic sparse/dense datasets, repeated timings, and separate memory passes. The general sparse-influence theorem, LSPI-control bridge, and neural experiments remain planned research.

## Stage tracker

| Stage | Status | Evidence / next gate |
| --- | --- | --- |
| Literature and novelty audit | Initial review complete | `literature_review.md`, `novelty_reassessment.md`, three `audits/` notes; CURe full text remains unresolved |
| Tabular data subtraction and planning | Implemented and tested | Retained-only rebuild comparisons, empty rows, repeated requests, ties, non-greedy predecessor regression |
| Independent retained-data reference | Implemented and tested | Exact rational policy enumeration for tiny MDPs, independently aggregated from logs; 20 random-model comparisons |
| Baselines and cost ledger | Initial correctness pilot complete | 15 case/seed combinations, 75 method runs; all 60 updating runs matched retained-only exact policy; larger timing and memory study remains |
| Sparse decision certificate | Restricted special case implemented | Exact unchanged-policy-row test, tie order, repeated requests, rejection atomicity; general influence envelope remains open |
| Frozen-feature LSPI control | Not started | Validate policy-dependent target corrections against retained-only replay |
| GPU IQL pilot | Not started | Dataset/learner identity, multiple retrains, strong fine-tune, matched controls, multi-attack audit |
| Full experiments and manuscript | Not started | Pass prior gates before expanding |

## Latest validation

- Current suite: **26 tests passed**, including more than 100 accepted random deletion requests checked against exact retained-policy enumeration, generated-episode validation, memory-pass checks, and a 32-state case inspecting one affected action. [Special-case proof and contract](audits/nonpolicy_certificate_proof.md).
- Larger scaling pilot is ready to run; final measurements will be recorded after execution. It includes explicit reference type, source hashes, randomized method order, repeat distributions, and separate request-allocation passes.

- `python -m unittest discover -s tests -v`: **17 tests passed**.
- [GitHub CI run 36522820517](https://github.com/BirariAditya/rl-trajectory-unlearning/actions/runs/36522820517) **passed** on revision `c229e7c`: all tests plus a one-seed baseline smoke run on Ubuntu/Python 3.11.
- `python scripts/run_baselines.py --output output/benchmarks/phase1.json`: **15 cases / 75 method runs**, zero non-noop convergence or policy-agreement failures. [Raw report](output/benchmarks/phase1.json), [interpretation](output/benchmarks/README.md).
- Pilot source revision: `bd37c7927c98494187ed9f20debadc8c2d6e76d4`, clean working tree at execution; the JSON also records source SHA-256 hashes, full datasets, deletion IDs, seeds, solver tolerance, and device.
- Maximum Q error for updating methods versus the rational reference: `5.56e-17` (rounded upward). At seed 0, no-op disagreed in shared-state, bottleneck, coverage-loss, and tie cases, and agreed in the non-policy-row case. The strict gap certificate rejected all three exact-tie requests and used fallback, as intended.
- Added five complete-episode scenarios: shared-state reward deletion, non-policy row deletion, bottleneck propagation, coverage loss, and exact tie.
- Work counters include every Bellman target evaluation (including convergence/certificate scans), transition entries read, predecessor edges indexed, and log transitions applied. Wall time includes remaining Python work and fallback. These counters are not a complete machine-instruction or memory profiler.
- Rational reference uses decimal-string interpretations of logged rewards and discount; the production planner uses binary64. Comparisons therefore use declared tolerances, with exact ties checked separately.

## Next implementation tasks

1. Execute and inspect the repeated sparse/dense scaling matrix. Add cyclic cases, sequential-request benchmark runs, and warm-start policy iteration before publication-level comparisons.
2. Measure restricted certificate preparation amortization and integrate its inconclusive result with a general retained-data solver.
3. Attempt the general sparse influence envelope and search for counterexamples before claiming the proposed theorem.

## Limitations and open decisions

- Tiny synthetic results establish correctness checks, not benchmark superiority, novelty, or a deletion-time complexity theorem.
- The current repair still builds the predecessor index and scans all rows for its certificate. It has no sublinear request-time claim.
- Policy equality does not certify deletion of released critic, metadata, optimizer, or neural encoder state.
- The exact oracle enumerates policies and intentionally refuses more than 4,096 policies. It is validation overhead, not a competitive planning baseline.
- The restricted certificate uses a rational empirical model and dense global preparation. It proves only unchanged-policy behavior for accepted requests; the general sparse algorithm and GPU experiments have not run. Conference acceptance is not predicted from prototype tests.

## Update convention

At the end of each implementation session, update this file with the current stage, actual commands/results, output paths, completed tasks, next tasks, and blockers. Mark a gate complete only when its evidence exists. Keep failed experiments and scope limitations visible. Do not silently upgrade a hypothesis to a result.

## Activity log

- **2026-09-29:** implemented restricted rational certificate and its proof/contract, seven certificate tests, acyclic sparse/dense log generation, repeated timing runner and separate memory passes. All 26 tests passed. Scaling measurements pending execution.

- **2026-09-29:** completed three-agent novelty reassessment and revised implementation phases.
- **2026-09-29:** implemented independent rational reference, ordinary prioritized-sweeping baseline, metered experiment harness, reproducible JSON manifest runner, and GitHub test workflow. All 17 tests passed.
- **2026-09-29:** created private GitHub repository and completed the three-seed correctness pilot. Archived 75 runs with source provenance and complete small datasets. GitHub workflow repeats the tests and a one-seed baseline smoke run on pushes and pull requests.
