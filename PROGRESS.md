# Research progress

Last updated: 2026-10-05. Repository: [BirariAditya/rl-trajectory-unlearning](https://github.com/BirariAditya/rl-trajectory-unlearning) (private).

## Current stage

**Checkpoint 02 reached: matched SCC controls and timed deletion stress tests complete.** Read [the new checkpoint](checkpoints/02_scc_controls.md). Ordinary graph decomposition narrows the apparent speedup; pruned repair helps a constructed bridge regime but loses its benefit on weaker bridges and anchor-invalidating requests. FTVI directly overlaps the pruning/decomposition mechanism, so the scientific novelty gate remains open. The general deletion-sensitive theorem, LSPI-control bridge, and GPU experiments remain future work.

## Stage tracker

| Stage | Status | Evidence / next gate |
| --- | --- | --- |
| Literature and novelty audit | Initial review complete | `literature_review.md`, `novelty_reassessment.md`, three `audits/` notes; CURe full text remains unresolved |
| Tabular data subtraction and planning | Implemented and tested | Retained-only rebuild comparisons, empty rows, repeated requests, ties, non-greedy predecessor regression |
| Independent retained-data reference | Implemented and tested | Exact rational policy enumeration for tiny MDPs, independently aggregated from logs; 20 random-model comparisons |
| Baselines and cost ledger | Matched SCC and deletion stress checkpoint complete | 5,688 additional executions over 162 runs; classical SCC baseline implemented; optimized/action-eliminating planner comparisons remain |
| Sparse decision certificate | Exact block repair and safe pruning implemented | Chosen-action changes, cyclic/sequential requests, exact ties, index repair, anchor invalidation; mature-planner novelty and general margin envelope remain open |
| Frozen-feature LSPI control | Not started | Validate policy-dependent target corrections against retained-only replay |
| GPU IQL pilot | Not started | Dataset/learner identity, multiple retrains, strong fine-tune, matched controls, multi-attack audit |
| Full experiments and manuscript | Not started | Pass prior gates before expanding |

## Latest validation

- **Next checkpoint underway:** lean global planning without local reverse indexes and global strict-bound action elimination are implemented. The 48-test suite passed before adding one further redundant-anchor regression; all seven global-planner tests now pass. The initial six-regime smoke completed 91 executions with zero disagreements; full comparison is pending. A fixed-feature LSPI replay prototype is being implemented separately. [Global baseline contract](audits/global_elimination_contract.md).

- **SCC checkpoint: 42 tests passed; 162 runs / 711 requests / 5,688 method executions, zero failures.** All eight matched dense/SCC methods agree exactly with scratch dense policy/value output. Six regimes, 8/16/32 states, three dataset seeds and three timing repeats. [Report](checkpoints/02_scc_controls.md), [full tables](output/benchmarks/scc_checkpoint_summary.md), [compressed raw results](output/benchmarks/scc_checkpoint.json.gz).
- Source `94e7f45107346afb0b9a89d00122b007941e7abc` was clean at launch. [CI 37332934561](https://github.com/BirariAditya/rl-trajectory-unlearning/actions/runs/37332934561) passed 42 tests and all four smoke runners. An independent code review found no correctness defect; benchmark limitations remain documented.
- At 32 states, modular preparation-plus-sequence cost is 7.778 ms for local SCC versus 21.679 ms for warm SCC. Pruned SCC costs 47.027 ms there. Dominated bridges favor pruning (91.129 ms versus 695.543 ms warm SCC); weaker bridges (1026.023 versus 965.084), anchor invalidation (812.678 versus 766.089), and dense influence (627.802 versus 567.670) do not. These are synthetic descriptive medians, not a universal speedup or novelty result.

- **36 tests passed** after adding exact block policy iteration and certified action pruning. Forty random cyclic datasets × four deletions each match exact retained policies and values under enumeration. Deterministic tests cover restored admissible edges, anchor invalidation, and solver failure without mutation. [Proof and publication boundary](audits/local_repair_proof.md).
- **Full cyclic checkpoint completed:** 54 runs, 162 sequential deletion requests, **810 method executions**, zero failures. Exact local/pruned and warm policy iteration match scratch rational policies and values; binary64 warm VI matches policies in these cases. [Results](output/benchmarks/local_checkpoint_summary.md), [raw report](output/benchmarks/local_checkpoint.json.gz).
- Source `70f13773b8e467e99cb429115cc2b437c008708e` was clean at execution. [CI run 36603869847](https://github.com/BirariAditya/rl-trajectory-unlearning/actions/runs/36603869847) passed 36 tests and tiny/scaling/cyclic smoke experiments.
- Constructed 16-state dominated-bridge case: pruned block size **4 versus 16**; median request **0.448 ms versus 9.847 ms** for warm exact PI. Preparation plus four requests: **12.502 ms versus 46.365 ms**. Dense influence still selects 16 states; full-sequence pruned cost **47.962 ms versus 42.633 ms**, so no universal benefit is claimed.

- Earlier scaling checkpoint: **26 tests passed**, including more than 100 accepted random deletion requests checked against exact retained-policy enumeration, generated-episode validation, memory-pass checks, and a 32-state case inspecting one affected action. [Special-case proof and contract](audits/nonpolicy_certificate_proof.md).
- [Scaling summary](output/benchmarks/scaling_summary.md) and [compressed raw report](output/benchmarks/scaling.json.gz): **12 settings**, sparse/dense × 16/64 states × 3 seeds, with 3 randomized-order timing repetitions and a separate memory pass. All 192 updating-method runs across timing/memory passes converged and matched the retained numerical reference. Maximum Q error: `1.67e-10` rounded upward. The reference minimum action gap across these runs was approximately `3.74e-4`.
- Scaling source: clean revision `c6ce7331a94a6f9484de0305bf1c2099dd058909`. [GitHub CI run 36600961815](https://github.com/BirariAditya/rl-trajectory-unlearning/actions/runs/36600961815) passed all 26 tests plus tiny and scaling smoke runs.
- **Negative result retained:** median of per-seed median request times on the 64-state dense logs was 120.520 ms for global certified repair versus 29.086 ms for warm-started planning. On the 64-state sparse logs it was 32.378 ms versus 42.199 ms. These synthetic, instrumented results motivate testing graph dependence; they do not establish a general speedup or significance.

- Initial milestone: **17 tests passed**; superseded by the current 26-test suite above.
- [GitHub CI run 36522820517](https://github.com/BirariAditya/rl-trajectory-unlearning/actions/runs/36522820517) **passed** on revision `c229e7c`: all tests plus a one-seed baseline smoke run on Ubuntu/Python 3.11.
- `python scripts/run_baselines.py --output output/benchmarks/phase1.json`: **15 cases / 75 method runs**, zero non-noop convergence or policy-agreement failures. [Raw report](output/benchmarks/phase1.json), [interpretation](output/benchmarks/README.md).
- Pilot source revision: `bd37c7927c98494187ed9f20debadc8c2d6e76d4`, clean working tree at execution; the JSON also records source SHA-256 hashes, full datasets, deletion IDs, seeds, solver tolerance, and device.
- Maximum Q error for updating methods versus the rational reference: `5.56e-17` (rounded upward). At seed 0, no-op disagreed in shared-state, bottleneck, coverage-loss, and tie cases, and agreed in the non-policy-row case. The strict gap certificate rejected all three exact-tie requests and used fallback, as intended.
- Added five complete-episode scenarios: shared-state reward deletion, non-policy row deletion, bottleneck propagation, coverage loss, and exact tie.
- Work counters include every Bellman target evaluation (including convergence/certificate scans), transition entries read, predecessor edges indexed, and log transitions applied. Wall time includes remaining Python work and fallback. These counters are not a complete machine-instruction or memory profiler.
- Rational reference uses decimal-string interpretations of logged rewards and discount; the production planner uses binary64. Comparisons therefore use declared tolerances, with exact ties checked separately.

## Next implementation tasks

1. Add a lean global retained-data planner without repair-index overhead and a matched action-eliminating graph planner. Internal SCC controls are complete; mature-planner/FTVI comparison is not.
2. Seek a deletion-specific sensitivity/work theorem beyond classical FTVI, with verification, invalidation, and indexing included. Pivot to fixed-feature LSPI control if it reduces to standard planning.
3. Extend beyond 32-state synthetic cases and trajectory batches only after the comparison gate; batches currently have no user-identity schema. GPU experiments remain staged.

## Limitations and open decisions

- Tiny synthetic results establish correctness checks, not benchmark superiority, novelty, or a deletion-time complexity theorem.
- The legacy `delete_and_repair` baseline rebuilds its predecessor index and scans all rows. The new exact block repair maintains an index and can avoid global verification under its structural/pruning conditions. Its worst case is global, and preparation/anchor invalidation are expensive.
- Policy equality does not certify deletion of released critic, metadata, optimizer, or neural encoder state.
- The exact oracle enumerates policies and intentionally refuses more than 4,096 policies. It is validation overhead, not a competitive planning baseline.
- The restricted no-change certificate remains available. The new block repair handles changed policies but uses rational arithmetic and potentially expensive global preparation. Action elimination is established prior art; this implementation is not a novelty claim. GPU experiments have not run.

## Update convention

At the end of each implementation session, update this file with the current stage, actual commands/results, output paths, completed tasks, next tasks, and blockers. Mark a gate complete only when its evidence exists. Keep failed experiments and scope limitations visible. Do not silently upgrade a hypothesis to a result.

## Activity log

- **2026-10-05:** reached Checkpoint 02. Added exact SCC planner, matched dense/SCC runner, six correctness regressions, CI smoke coverage, and SCC/prior-art contract. Forty-two tests and CI pass; full matrix completed with 5,688 executions and zero failures. Archived full costs and timed failure regimes. FTVI already combines action elimination with recovered graph locality; that combination is not our novelty.

- **2026-09-29:** implemented and pushed exact block repair, warm full policy iteration, optional bound-certified action pruning, and atomic sequential requests. Added proof notes, 10 new tests, and cyclic checkpoint runner. All 36 tests and GitHub CI passed.
- **2026-09-29:** reached Checkpoint 01 after 810 failure-free cyclic method executions. Archived full preparation-inclusive results, retained dense failure regime, and recorded action-elimination prior art. Novelty remains unproven; checkpoint report specifies next gates.

- **2026-09-29:** implemented and pushed restricted rational certificate and its proof/contract, seven certificate tests, acyclic sparse/dense log generation, repeated timing runner and separate memory passes. All 26 tests passed locally and in GitHub CI.
- **2026-09-29:** completed scaling pilot, archived verified lossless gzip report (170,750 bytes), and added a reproducible summary script. Kept the raw 4 MB report locally under ignored `results/`. Recorded dense-case slowdown and next gates.

- **2026-09-29:** completed three-agent novelty reassessment and revised implementation phases.
- **2026-09-29:** implemented independent rational reference, ordinary prioritized-sweeping baseline, metered experiment harness, reproducible JSON manifest runner, and GitHub test workflow. All 17 tests passed.
- **2026-09-29:** created private GitHub repository and completed the three-seed correctness pilot. Archived 75 runs with source provenance and complete small datasets. GitHub workflow repeats the tests and a one-seed baseline smoke run on pushes and pull requests.
