# Presentation talk track — trajectory unlearning in offline RL

**Updated 15 October for 18 October 2026; evidence through Checkpoint 06.** This comprehensive, modular 30-slide Markdown outline covers all work so far, with speaker notes and evidence links. The policy-sequence certificate is included with its narrower guarantee and adverse cases. The companion [learning guide](../../docs/LEARNING_GUIDE.md) explains the equations more slowly. This is presentation source material, not a rendered slide deck.

## Presentation contract

Open with the research question and define the retained-only learner. Show the implementation and how stronger classical controls changed its interpretation. Close on the policy-sequence certificate, its narrower output and the next feature/output/numerical controls. The strongest demonstrated claims are **correctness relative to declared restricted learners** and conditional synthetic lifecycle savings. We have not measured continuous-control or real-environment retained-task return, nor run a GPU experiment. A policy matching a retained empirical model is not thereby useful in the true environment.

Use these terms consistently:

- “Retained-only reference”: the same declared learner run on the trajectories remaining after deletion.
- “Exact tabular policy/value”: rational equality for that finite empirical-MDP learner; no claim about arbitrary RL systems.
- “Exact LSPI replay”: equality of matrices, weights, and policies at every step of a declared finite schedule; no convergence or optimal-control claim.
- “Full lifecycle cost”: original training, preparation, and all deletion requests. Request latency is a separate number.
- “Checkpoint 03”: the dated [report](../../checkpoints/03_global_controls_and_lspi.md) and [summary](../../output/benchmarks/global_checkpoint_summary.md).

## Module A — The question and its target

### Slide 1 — Title and thesis

**On slide:** “Deleting trajectories from an offline RL agent: matching retained-only retraining while measuring complete update cost.” Subtitle: “Exact restricted-learner checkpoints; algorithmic novelty remains open.”

**Speaker notes:** A trained agent may later receive a request to remove complete interaction histories. We seek the policy the same learner would have produced without those histories. The project has built correctness references and tested several planning strategies. A stronger baseline overturned the initial local-repair speed story, leading to fixed-feature LSPI control as the next research focus.

**Evidence:** [Progress](../../PROGRESS.md).

### Slide 2 — Formal counterfactual

**On slide:** \(D=D_R\cup D_F\), \(\pi_{\rm ref}=A(D_R)\), \(\pi_U=U(A(D),D_F)\). Seek fidelity to \(\pi_{\rm ref}\) with lower complete cost.

**Speaker notes:** Define each symbol aloud: full log, retained log, forgotten log, declared learning algorithm, update algorithm. For a deterministic finite learner, we can compare policies exactly. For a stochastic or neural learner, one retrain is only one draw from a distribution. The training procedure must be named precisely, including tie-breaking, stopping rule, and input conventions.

**Evidence:** [Target definition](../../literature_review.md#1-define-the-target-before-choosing-an-algorithm).

### Slide 3 — Forgetting does not mean failing on forgotten states

**On slide:** Didactic one-step log: action \(a\) has rewards 1 and 1; action \(b\) has reward 0.5. Delete one \(a\) record; retained-only still chooses \(a\).

**Speaker notes:** This is an illustrative calculation, not an experiment. Retained examples can independently support the same behavior. Artificially suppressing value on deleted-state actions may move the policy away from retraining. This motivates a retained-only reference rather than a deletion score alone.

**Evidence:** [Learning guide](../../docs/LEARNING_GUIDE.md#1-the-question-in-one-minute).

### Slide 4 — Four different claims

**On slide:** Retained return; policy/value fidelity; attack resistance; distributional exactness. “One does not imply the next.”

**Speaker notes:** A good return does not show deletion. A matching deployed policy does not erase a critic, replay buffer, optimizer, or metadata. A failed membership attack is evidence against that attack. Distributional exactness concerns the entire released output. Our current results are policy/value or finite LSPI trace fidelity in restricted learners. No real-environment retained-return or neural privacy result exists yet.

**Evidence:** [Claim table](../../literature_review.md#1-define-the-target-before-choosing-an-algorithm).

## Module B — Literature and novelty boundary

### Slide 5 — Direct offline trajectory baseline

**On slide:** TrajDeleter: suppress forgotten-trajectory values, then restore retained behavior. Deep offline RL comparator.

**Speaker notes:** TrajDeleter is direct prior art for trajectory deletion. It reports approximate forgetting and return under its benchmarks and audit. Its value-suppression objective can differ from matching the retained-only learner when states overlap. If we later test neural policies, compare it under matched dataset, architecture, and evaluation conditions; do not dismiss it as unrelated.

**Evidence:** [TrajDeleter review](../../literature_review.md#trajdeleter-the-required-offline-trajectory-baseline).

### Slide 6 — TOUR raises the evaluation standard

**On slide:** Retained retraining; matched non-members; several attacks; retained utility. “One audit score can mislead.”

**Speaker notes:** TOUR is principally a benchmark. Low membership detection may coexist with policy collapse or fail under a different attack. Its logic argues for reporting reference-relative behavior, utility, and audits together. Our current tabular benchmarks do not yet measure deep RL utility or run TOUR-style privacy attacks.

**Evidence:** [TOUR review](../../literature_review.md#tour-the-required-evaluation-comparator).

### Slide 7 — Strong formal unlearning, different data model

**On slide:** Nguyen-Tang & Arora: online user-level RL; environment oracle; distributional equality of complete output.

**Speaker notes:** Their setting can replay counterfactual online interactions because the user/environment response model supplies trajectories for changed policies. A fixed offline log generally lacks those counterfactual interactions. Their guarantee is stronger but its assumptions differ. Our tabular policy equality must not be presented as an extension of their distributional theorem.

**Evidence:** [Exact Unlearning review](../../literature_review.md#exact-unlearning-in-reinforcement-learning-strongest-formal-benchmark-different-data-model).

### Slide 8 — The five supplied papers: what transfers

**On slide:** RUM → heterogeneous deletion difficulty. RULE, PURGE, Yao et al. → LLM behavioral objectives and retain anchors. “Adjacent ideas, distinct problem.”

**Speaker notes:** RUM is supervised unlearning; it suggests stratifying deletions by overlap, rare coverage, and leverage. RULE and PURGE use RL to suppress specified LLM outputs. Yao et al. motivate negative-gradient and retention baselines. Most do not delete experience from an RL agent or establish equality to its retained-data retraining. This distinction prevented a misleading literature claim.

**Evidence:** [Supplied-paper review](../../literature_review.md#3-the-five-supplied-papers-and-what-transfers).

### Slide 9 — Classical planning prior art

**On slide:** Prioritized sweeping; incremental policy iteration; SCC planning; action elimination; incremental LSPI.

**Speaker notes:** Predecessor propagation, graph decomposition, and upper/lower bound pruning were well studied before RL unlearning. FTVI already combines action elimination and graph components. A fast constructed example against a dense solver is therefore not sufficient evidence for a novel algorithm. This realization drove stronger controls in Checkpoints 02 and 03.

**Evidence:** [Planning prior art](../../literature_review.md#4-prior-planning-methods-already-cover-much-of-the-candidate-mechanism), [Checkpoint 02](../../checkpoints/02_scc_controls.md).

## Module C — The tabular implementation

### Slide 10 — Declare the learner before the deletion method

**On slide:** Trajectory records → counts and rewards → empirical MDP → rational optimal policy/value. Empty rows: zero reward and termination. Fixed tie order.

**Speaker notes:** A visit count and reward sum define each observed state-action row. Deletion subtracts a trajectory's contribution. If all observations of a row disappear, the learner's explicit convention applies. The exact path interprets decimal inputs as rationals; the binary64 value-iteration path has a numerical tolerance. “Exact” here belongs to the declared rational learner, not a floating neural model.

**Evidence:** [Learning guide](../../docs/LEARNING_GUIDE.md#3-the-tabular-learner-from-log-to-model-to-policy), [tabular.py](../../src/rl_unlearning/tabular.py).

### Slide 11 — Bellman dependencies

**On slide:** \(Q(s,a)=\bar r(s,a)+\gamma\sum_{s'}P(s'|s,a)\max_b Q(s',b)\). Highlight the successor value term.

**Speaker notes:** Immediate reward is only one part of an action's value. Deleting data at state \(z\) can alter its best continuation; a retained transition from \(x\) into \(z\) can then change value. This influence propagates backward through predecessors, including actions that were previously not chosen. Sparse modules may allow local updates; bottlenecks and dense graphs may not.

**Evidence:** [Learner explanation](../../docs/LEARNING_GUIDE.md#3-the-tabular-learner-from-log-to-model-to-policy).

### Slide 12 — First prototype: residual and action gap

**On slide:** \(\eta=\|T_RQ-Q\|_\infty\). Error bound \(\eta/(1-\gamma)\). Strict best-action gap \(>2\eta/(1-\gamma)\) certifies the action. Full scan visits all state-action rows and their transition entries.

**Speaker notes:** Bellman contraction bounds how far a candidate Q table can be from retained-optimal Q. If the best action wins by more than twice that uncertainty, Q error cannot reverse the decision. This is a standard behavioral certificate. Its global residual checks all state-action rows; a small repair queue alone does not make the complete request local. Exact ties may be inconclusive.

**Evidence:** [README boundary](../../README.md), [unlearning.py](../../src/rl_unlearning/unlearning.py).

### Slide 13 — Exact backward-closed block repair

**On slide:** Changed source states \(C\) → all old-graph predecessors \(B\) → solve \(B\) using exact outside values.

**Speaker notes:** Outside the backward closure, no model row changed and no path reaches a changed row; old values remain valid. Inside, solve the retained Bellman control problem with those outside values as boundary conditions. The joined values satisfy the full Bellman equations, so discounting identifies the retained optimum. This handles cycles, changing actions, and repeated requests. The reverse graph includes edges of non-greedy actions. The algorithm updates the index after successful deletions and does not commit failed ones.

**Evidence:** [Block proof](../../audits/local_repair_proof.md), [Checkpoint 01](../../checkpoints/01_exact_local_repair.md).

### Slide 14 — Pruning and anchor invalidation

**On slide:** Anchor lower bound \(L=V^b\); action upper bound \(\overline Q\). If \(\overline Q<L\), prune action edge. Changed anchor row invalidates the lower bound.

**Speaker notes:** A provably dominated action cannot determine an optimal value or propagate influence through its edge. The inequality is strict to preserve tie behavior. In the current local implementation, an anchor-changing deletion forces a full solve and disables pruning. A later global control refreshes the anchor efficiently. This distinction explains why a global classical method can beat the current local method over a full request sequence. Action elimination itself is established prior art.

**Evidence:** [Block proof](../../audits/local_repair_proof.md), [global control contract](../../audits/global_elimination_contract.md).

### Slide 15 — SCC planning as a stronger control

**On slide:** Solve mutually reachable states together; solve the component graph in dependency order.

**Speaker notes:** A strongly connected component contains states coupled through cycles. Between SCCs, dependencies form an acyclic graph, allowing smaller exact solves. A global planner can exploit this structure too, so an apparent local advantage against dense global policy iteration may shrink. This is an internal matched SCC control, not an optimized external FTVI reproduction.

**Evidence:** [Checkpoint 02](../../checkpoints/02_scc_controls.md), [SCC planning contract](../../audits/scc_planning_checkpoint.md).

## Module D — Checkpoint evidence and negative results

### Slide 16 — Checkpoint 01: an exact correctness substrate

**On slide:** 36 tests; 810 method executions, zero reported failures. Constructed dominated bridge: 4 of 16 states solved; full sequence 12.502 vs 46.365 ms. Dense case loses.

**Speaker notes:** Exact local/pruned repair matched retained-only policy and values in tested cyclic and sequential requests. The designed bridge case made pruning attractive. But dense influence selected all states, and preparation-inclusive pruned repair took 47.962 ms versus 42.633 ms for warm exact policy iteration. Present both numbers. This checkpoint proves implementation capability under tested cases, not universal speed or novelty.

**Evidence:** [Checkpoint 01](../../checkpoints/01_exact_local_repair.md), [local summary](../../output/benchmarks/local_checkpoint_summary.md).

### Slide 17 — Checkpoint 02: SCC controls narrow the story

**On slide:** 42 tests; 5,688 method executions / 711 requests, zero reported policy/value failures. 32-state modular lifecycle: local SCC 7.778 ms; warm SCC 21.679 ms; pruned SCC 47.027 ms.

**Speaker notes:** Once both sides used SCC planning, local repair still helped on modular data but pruning's anchor preparation was costly. Weak bridges, anchor invalidation, and dense influence did not favor pruning. These are descriptive synthetic medians, with repeated timings on the same dataset seeds. The result made a dense-solver-only comparison inadequate.

**Evidence:** [Checkpoint 02](../../checkpoints/02_scc_controls.md), [SCC summary](../../output/benchmarks/scc_checkpoint_summary.md).

### Slide 18 — Checkpoint 03: lean global elimination reverses key comparisons

**On slide:** 60 tests. 108 runs / 594 requests / 4,158 further method executions with zero reported policy/value mismatches. 32-state full lifecycle: dominated bridge 96.257 ms current pruned-local vs 32.969 ms classical global elimination; anchor invalidation 835.267 vs 39.642 ms.

**Speaker notes:** The lean global method avoids a local reverse index, prunes inside its SCC graph, and refreshes invalidated anchor values. This is a **method comparison**, not a pure locality ablation: preparation and implementation details differ. But it decisively weakens the claim that the current pruned-local implementation is the faster paper contribution in these regimes. All compared methods matched retained-only exact policy and values in the reported requests.

**Evidence:** [Checkpoint 03](../../checkpoints/03_global_controls_and_lspi.md), [global summary](../../output/benchmarks/global_checkpoint_summary.md).

### Slide 19 — What remains useful and what did not survive

**On slide:** Useful: exact reference, modular locality, failure taxonomy. Not supported: broad local-pruning speedup or novelty. Dense and weak bridges induce broad work.

**Speaker notes:** Local SCC still has 7.930 ms full lifecycle versus 24.282 ms for lean warm SCC on 32-state modular data. That is real for this synthetic regime, but classical incremental planning can exploit similar structure. The current local method has very cheap individual dominated-bridge requests; its preparation loses over the sequence. We must count original training, index construction, anchor evaluation/refresh, verification, fallback, memory, and output cost.

**Evidence:** [Checkpoint 03 interpretation](../../checkpoints/03_global_controls_and_lspi.md), [full cost table](../../output/benchmarks/global_checkpoint_summary.md).

### Slide 20 — How strong is this evidence?

**On slide:** Tiny independent enumeration; 16/32-state synthetic regimes; three dataset seeds; reused timing repeats; shared implementation pieces. Descriptive, not a significance claim.

**Speaker notes:** Method executions are not independent datasets. Large benchmark methods share solvers and statistics code; tiny independently enumerated models add a separate correctness check. Synthetic timing differences identify counterexamples to a broad claim, but do not estimate practical deployment speed or real-environment return. The global control is classical but not a full FTVI reproduction. This is a candid, scientifically useful checkpoint.

**Evidence:** [Global summary limits](../../output/benchmarks/global_checkpoint_summary.md#interpretation-limits), [Progress](../../PROGRESS.md).

## Module E — Fixed-feature LSPI control

### Slide 21 — Why fixed-feature LSPI?

**On slide:** Features bridge tabular and neural methods. Deletion can change policy-dependent targets on **retained** transitions.

**Speaker notes:** The tabular code remains an exact reference. LSPI creates a different opportunity: removing samples can change the improved policy, which changes the target used to evaluate retained transitions. Fix features and initial policy independently of deletable data, and define a finite iteration budget. Then the counterfactual learner is precise. Original LSPI already discusses incremental evaluation, so correctness alone will not establish novelty.

**Evidence:** [Checkpoint 03 LSPI boundary](../../checkpoints/03_global_controls_and_lspi.md#lspi-result-and-boundary), [contract](../../audits/lspi_control_contract.md).

### Slide 22 — LSPI equation in words

**On slide:** \(A_R(\pi)w=b_R\), where \(A_R(\pi)=\lambda I+\sum_{i\in R}\phi_i[\phi_i-\gamma\phi(s_i',\pi(s_i'))]^\top\), \(b_R=\sum_{i\in R}\phi_i r_i\).

**Speaker notes:** \(\phi_i\) is a fixed feature vector for the observed state-action pair; \(w\) is the linear Q estimate; \(\lambda I\) is fixed regularization. The current policy decides which feature is used at the successor. Solve the linear system, greedily improve the policy, and repeat exactly the declared number of times. Terminal successors contribute zero features. A cycle or exhausted budget is a valid finite-schedule output, not proof of convergence.

**Evidence:** [LSPI contract](../../audits/lspi_control_contract.md), [implementation](../../src/rl_unlearning/lspi_control.py).

### Slide 23 — One-shot sample subtraction misses target changes

**On slide:** \(G_R(z)=\sum_{i\in R:s_i'=z}\phi_i\). If successor action changes, \(\Delta A=-\gamma G_R(z)[\phi(z,\pi_{\rm new}(z))-\phi(z,\pi_{\rm old}(z))]^\top\).

**Speaker notes:** Data deletion changes matrix and RHS. A policy switch at successor \(z\) changes the target of every retained transition into \(z\). Grouping those transitions' features gives the outer-product correction. It is not free work: the current prototype copies matrices, scans successor groups, scores all state-actions, and runs a dense exact solve each iteration. Its equality result is a correctness substrate.

**Evidence:** [LSPI assembly contract](../../audits/lspi_control_contract.md#two-independent-assembly-paths).

### Slide 24 — Two-state worked reversal and current validation

**On slide:** Full-data \((a,a)\). Delete terminal \((z,a,1)\). One-shot downdate \((b,b)\). Correct retained-only three-step LSPI \((a,b)\). Current suite: 80 tests across the repository.

**Speaker notes:** The retained transition \(x\to z\) must change its successor target after \(z\) chooses \(b\). Correcting it gives \(Q(x,a)=1/18>Q(x,b)=1/24\), so \(x\) returns to \(a\). Direct scratch assembly and replay match every matrix, RHS, weight, and policy in tested finite schedules, including switches, cycles, ties, sequential deletion, and singular rejection without partial commit. Both share the schedule and solver. The next slide shows the completed four-method cost comparison. Incremental inverse/refactorization and measured memory remain untested. The teaching example has three iterations; the cost benchmark declares four.

**Evidence:** [Worked example](../../checkpoints/lspi_control_example.md), [Checkpoint 03](../../checkpoints/03_global_controls_and_lspi.md), [LSPI contract](../../audits/lspi_control_contract.md).

### Slide 25 — Checkpoint 04: ordinary caching is a strong competitor

**On slide:** 63 runs / 162 requests / 648 method executions, zero trace mismatches. Memoized caching: lowest request median in 7/7 settings. Target replay: lowest full-lifecycle median in 2/7. No broad speed advantage.

| Setting | Memo request | Replay request | Memo lifecycle | Replay lifecycle |
| --- | ---: | ---: | ---: | ---: |
| Shared: 128 transitions, 4 features | 1.286 ms | 1.309 ms | 12.052 ms | 10.711 ms |
| Shared: 128 transitions, 8 features | 5.027 ms | 6.174 ms | 39.948 ms | 40.955 ms |
| Switch: 640 transitions, 4 features | 3.116 ms | 3.935 ms | 28.239 ms | 35.111 ms |

**Speaker notes:** Cached reconstruction stores \(C=\lambda I+\sum\phi_i\phi_i^\top\), \(b=\sum\phi_i r_i\), and successor sums \(G(z)\); it assembles \(A(\pi)=C-\gamma\sum_zG(z)\phi(z,\pi(z))^\top\). The memoized version also reuses a repeated policy's matrix and weights within the same request. It still emits all four trace steps and scores every action. Its evaluation cache resets after deletion because the data changed. Replay performs fewer target corrections but still solves at every iteration; conventional memoization avoids some solves. This comparison therefore does not isolate assembly alone. Lifecycle includes original learning and preparation, which explains why its ordering differs from request latency. All timed requests reached fixed points; cycles and singular failures were tested separately. These are small rational synthetic problems, with three log seeds and reused timing repeats; switch seeds only shuffle within-batch deletion order. No significance, memory, numerical factorization or GPU claim is supported.

**Evidence:** [Checkpoint 04 and provenance](../../checkpoints/04_lspi_cost.md), [complete tables](../../output/benchmarks/lspi_checkpoint_summary.md), [cached implementation](../../src/rl_unlearning/lspi_baselines.py), [tests and CI](https://github.com/BirariAditya/rl-trajectory-unlearning/actions/runs/38338893080).

### Slide 26 — Checkpoint 05: equalize evaluation reuse

**On slide:** 80 tests. 63 runs / 162 requests / 810 exact trace matches. Both memoized methods use equal solve/action-score counts. Memoized replay has the lowest request and lifecycle medians in 5/7 settings; two larger switch fixtures lose.

| Setting | Cached memo request | Replay memo request | Cached memo lifecycle | Replay memo lifecycle |
| --- | ---: | ---: | ---: | ---: |
| Shared: 128 transitions, 8 features | 7.766 ms | 6.468 ms | 58.306 ms | 50.185 ms |
| Switch: 640 transitions, 4 features | 5.005 ms | 6.320 ms | 45.685 ms | 52.715 ms |

**Speaker notes:** The preceding comparison gave cached reconstruction repeated-policy memoization but not replay. We now give replay the same cache policy, reset per deletion, while preserving every finite-schedule iterate. In shared settings both methods use two solves; cached reconstruction has 12 target terms versus 3–4 replay corrections. Conditional gains survive equal solve counts, but larger switch fixtures still lose despite fewer corrections. Preparation and zero-group handling differ, so this does not isolate every assembly operation. Compare methods within this experiment, not absolute milliseconds against the previous session. Every timed request reaches a fixed point; failure and cycle cases are separately unit-tested. These are descriptive rational synthetic results with no memory, stable numerical factorization, significance or novelty claim.

**Evidence:** [Checkpoint 05](../../checkpoints/05_lspi_matched_reuse.md), [complete tables](../../output/benchmarks/lspi_matched_checkpoint_summary.md), [reuse invariant](../../audits/lspi_matched_reuse.md), [source CI](https://github.com/BirariAditya/rl-trajectory-unlearning/actions/runs/38596853065).

### Slide 27 — A narrower question: must the policy be relearned?

**On slide:** Weights may change while actions stay identical. Certify every policy-improvement output. Return policy sequence/status only. Old internal anchors remain.

**Speaker notes:** This changes the output contract explicitly: earlier checkpoints matched matrices and weights as well. The new fast path proves only the same policy sequence and status as the specified retained-only learner. It does not claim old weights are retained-only weights, or that internal caches have lost deleted influence. Predictions that remain unchanged under removal already have prior art, including Wicker et al. and Abstract Gradient Training. The candidate specialization is to the policy-dependent sequence of LSPI evaluations. A fast check is meaningful only if its preparation and failures are charged.

**Evidence:** [Policy certificate proof and prior art](../../audits/lspi_policy_certificate.md), [learning guide Section 10](../../docs/LEARNING_GUIDE.md#10-new-candidate-certify-the-policy-sequence-without-recomputing-the-weights).

### Slide 28 — Cumulative bound and sequence induction

**On slide:** Anchor inverse H=A^-1. Deletion changes matrix/RHS by E,f. `(I+HE) delta_w = h = H(f-Ew)`. If `alpha >= ||HE||inf` and `alpha < 1`, then `||delta_w||inf <= ||h||inf/(1-alpha)`. Compare with precomputed action-margin radius.

**Speaker notes:** The bound controls how far weights can move for one input policy. Precompute the smallest normalized action gap over the entire state/action domain. If the bound is smaller, every greedy choice stays unchanged. Check each distinct input policy in the old finite schedule, and induction preserves all later inputs and outputs from the fixed initialization. Maintain cumulative h and alpha across deletions. h may cancel without alpha becoming zero; resetting both would be unsound. A failed bound is inconclusive and triggers exact retained replay plus anchor rebuilding. A singular retained solve rejects the request without changing state. This uses established perturbation algebra; its usefulness depends on the complete cost.

**Evidence:** [Proof and cancellation counterexample](../../audits/lspi_policy_certificate.md), [tests](../../tests/test_lspi_policy_certificate.py).

### Slide 29 — Checkpoint 06: strongest conditional result so far

**On slide:** 98 tests; 45 runs / 900 requests / 3,600 method executions, zero mismatches under declared contracts. 819 certificates pass; 81 fallbacks. Full lifecycle includes training/preparation and 20 requests.

| Setting | Classical inverse reuse | Policy certificate | Interpretation |
| --- | ---: | ---: | --- |
| Stable, 128 states, 4 features | 269.630 ms | 78.241 ms | 3.45x lower lifecycle time |
| Stable, 128 states, 8 features | 693.163 ms | 426.610 ms | 1.62x lower lifecycle time |
| Fragile, 16 states, 4 features | 65.925 ms | 83.808 ms | Certificate loses overall |

**Speaker notes:** The inverse control is classical Sherman–Morrison and also reuses repeated greedy outputs, so it avoids redundant action scans within a request. The certificate avoids new scans across accepted deletions. Its bound is arithmetic O(J*m*d^2 + J*d + deleted IDs) after paid inverse/margin preparation, independent of retained sample count and action-domain size on the accepted path; output serialization and rational bit costs are excluded. Show the failure: the eight-feature stable case has a median run-maximum request of 184.620 ms from fallback. Separate allocation peaks there are 210.5 KiB for the certificate and 204.8 KiB for inverse reuse, excluding pre-existing inputs. Some synthetic states share features; prediction deduplication and matched policy-only outputs remain important controls. These are not representative-task, statistical significance, privacy or novelty results.

**Evidence:** [Checkpoint 06](../../checkpoints/06_policy_sequence_certificate.md), [complete timing/allocation tables](../../output/benchmarks/lspi_certificate_summary.md), [source CI](https://github.com/BirariAditya/rl-trajectory-unlearning/actions/runs/38674439858).

## Module F — Research decision

### Slide 30 — What we can defend on 18 October

**On slide:** (1) Precise retained-only targets. (2) Tested exact tabular/full-trace learners. (3) A narrower policy-sequence certificate with conditional lifecycle gains. (4) Feature-alias, output-contract and numerical controls before a new-algorithm claim.

**Speaker notes:** Exact inverse reuse, 20-request sequences and separate Python allocation measurements are now complete. Next test unique features and feature-deduplicated prediction-only baselines, verified floating-point bounds and adverse conditioning. A paper claim must add a substantive deletion-specific guarantee or advantage beyond established incremental LSPI and unchanged-prediction certification. GPU experiments remain conditional; we have not measured retained-task return or privacy. Ask the advisor whether a policy-output guarantee is a useful target and what additional theorem/comparator is required. Do not assert conference acceptance probability or a confirmed novel algorithm.

**Evidence:** [Progress next steps](../../PROGRESS.md#next-implementation-tasks), [Checkpoint 03 updated phases](../../checkpoints/03_global_controls_and_lspi.md#updated-research-phases).

## Claim–evidence table for slide checking

| Claim safe to make | Evidence | Boundary to say aloud |
| --- | --- | --- |
| Exact tabular repair matches the declared retained empirical-MDP policy and values in tested cases | [Checkpoint 01](../../checkpoints/01_exact_local_repair.md), [proof](../../audits/local_repair_proof.md) | Restricted rational learner; not arbitrary neural unlearning |
| SCC controls remove much of the apparent dense-planner advantage | [Checkpoint 02](../../checkpoints/02_scc_controls.md), [tables](../../output/benchmarks/scc_checkpoint_summary.md) | Internal classical SCC comparator; synthetic models |
| Classical global elimination beats current pruned-local full lifecycle in bridge and anchor-invalidating regimes | [Checkpoint 03](../../checkpoints/03_global_controls_and_lspi.md), [tables](../../output/benchmarks/global_checkpoint_summary.md) | Methods differ in preparation and pruning placement |
| LSPI replay matches direct retained-only finite schedules and one-shot deletion can be wrong | [LSPI example](../../checkpoints/lspi_control_example.md), [contract](../../audits/lspi_control_contract.md), [tests](../../tests/test_lspi_control.py) | Correctness, no established efficiency or novelty |
| A real-environment utility or neural privacy result exists | **No evidence yet** | Do not claim |
| Memoized caching has the lowest request median in 7/7 settings; replay has the lowest lifecycle median in 2/7 | [Checkpoint 04](../../checkpoints/04_lspi_cost.md), [tables](../../output/benchmarks/lspi_checkpoint_summary.md) | Descriptive small rational timings; reuse is not matched, memory and factorization remain unmeasured |
| With equal reuse, memoized replay has the lowest request and lifecycle medians in 5/7 settings | [Checkpoint 05](../../checkpoints/05_lspi_matched_reuse.md), [tables](../../output/benchmarks/lspi_matched_checkpoint_summary.md) | Conditional engineering gain; two larger switch fixtures lose; not novelty or significance |
| Policy-sequence certificates give conditional lifecycle savings over classical inverse reuse | [Checkpoint 06](../../checkpoints/06_policy_sequence_certificate.md), [proof](../../audits/lspi_policy_certificate.md) | Policy-only output, retained internal anchors, feature aliases, fragile/fallback failures; not full-state erasure |

## Optional backup slides

**Backup A — Reproduction.** From C:\Users\adity\Research run python -m unittest discover -s tests -v, python scripts/run_tabular_demo.py, and python scripts/run_lspi_demo.py --output results/lspi_demo.json. The archived global report and its source revision are in [Checkpoint 03](../../checkpoints/03_global_controls_and_lspi.md). Use the [README](../../README.md) for larger runner commands.

**Backup B — Block proof.** Outside backward closure, changed rows are unreachable; cached values remain valid. Solve inside with fixed outside values and check Bellman optimality. Pruning additionally requires bounds and anchor preservation. See [proof](../../audits/local_repair_proof.md).

**Backup C — LSPI singularity and cycles.** Positive ridge does not guarantee an invertible off-policy nonsymmetric LSTD matrix. The code rejects a singular solve and does not commit deletion. Cycles and budget exhaustion are valid finite-schedule outcomes, not convergence. See [contract](../../audits/lspi_control_contract.md).

## Likely professor questions

**“What exactly is deleted?”** Complete trajectory records. Tabular repair matches the declared retained empirical-MDP learner's policy and values. LSPI replay matches the declared finite learner's trace. No claim about independently released neural or optimizer state.

**“Why is policy match stronger than a low membership score?”** It tests the named counterfactual learner directly. A low attack score can accompany collapse and is attack-dependent. Policy match remains narrower than distributional exactness or privacy; see [TOUR review](../../literature_review.md#tour-the-required-evaluation-comparator).

**“Could you just subtract counts and rerun planning?”** Yes; that is the retained-only reference and a strong baseline. The research question is whether a complete update costs less while matching it. Checkpoint 03 shows this is nontrivial.

**“Is local repair novel?”** No claim of that kind. Predecessor propagation, SCCs, and action elimination are classical. Strong controls weakened the current local/pruned headline; see [prior art](../../literature_review.md#4-prior-planning-methods-already-cover-much-of-the-candidate-mechanism).

**“Why can deleting at \(z\) change an action at \(x\)?”** A retained \(x\to z\) transition depends on \(z\)'s value. LSPI also uses the feature of the action now selected at \(z\); the [two-state fixture](../../checkpoints/lspi_control_example.md) shows the wrong policy if that retained target is not corrected.

**“What is exact about LSPI?”** Every matrix, RHS, weight, and policy agrees with direct retained-only training under the same fixed iteration schedule, including finite cycles. It does not imply convergence, optimal control, fast computation, privacy, or correctness for a learned feature map.

**“Does the new certificate also match the weights?”** No. The earlier replay/inverse methods match full traces; the certificate only guarantees the policy sequence and status. It keeps old internal weights/inverses as anchors and does not release them as retained-only weights. That distinction is central to Checkpoint 06.

**“Why no GPU result yet?”** The exactness arguments belong to restricted learners. GPUs can support multiple neural retrains, but a neural claim needs its own target, strong baselines, attacks, and return measurements. Current [progress](../../PROGRESS.md) records no GPU experiment.

**“Do the timing ratios prove significance?”** No. Three dataset seeds and timing repeats on those datasets are descriptive. The models are small and synthetic. The ratios are useful to expose failure regimes, not to estimate general deployment performance.

**“What would change the conclusion before the talk?”** A separately archived comparison with unique features and matched prediction-only outputs, followed by verified numerical bounds and conditioning controls. Checkpoint 06 already adds exact inverse reuse and allocation measurements; its gains remain conditional and its output narrower. Further engineering gains still need a focused novelty argument.

## Speaker claim-control card

| Safe wording | Avoid |
| --- | --- |
| “Exact for our declared rational empirical-MDP learner” | “Exact unlearning of RL agents” |
| “Exact finite-schedule fixed-feature LSPI replay” | “Exact deep RL unlearning” |
| “Zero mismatches in tested executions” | “Benchmark proves correctness for all models” |
| “Constructed regime, descriptive median” | “Universal or statistically significant speedup” |
| “Classical elimination beat our current local method over these lifecycles” | Hiding that result behind request-only numbers |
| “With equal reuse, memoized replay wins five settings and loses two in this pilot” | Universal speedup or an acceptance estimate |
| “The certificate matches the retained-only policy sequence under its conditions” | “It erases deleted influence from all weights and caches” |

## Presenter study order

1. Read the [learning guide](../../docs/LEARNING_GUIDE.md), especially the target, block-repair intuition, and LSPI example.
2. Run the [tabular demo](../../scripts/run_tabular_demo.py) and [LSPI demo](../../scripts/run_lspi_demo.py).
3. Read [Checkpoint 01](../../checkpoints/01_exact_local_repair.md), [02](../../checkpoints/02_scc_controls.md), and [03](../../checkpoints/03_global_controls_and_lspi.md) in order; explain how each comparator changed the story.
4. Verify all numeric slides directly in the [global table](../../output/benchmarks/global_checkpoint_summary.md) and [LSPI fixture](../../checkpoints/lspi_control_example.md).
5. Read [Checkpoint 04](../../checkpoints/04_lspi_cost.md) and [05](../../checkpoints/05_lspi_matched_reuse.md), then verify Slides 25–26 in their separate archived tables. Before 18 October, check [PROGRESS.md](../../PROGRESS.md) for any later validated checkpoint and retain the negative findings.
6. Read [Checkpoint 06](../../checkpoints/06_policy_sequence_certificate.md) and the [certificate proof](../../audits/lspi_policy_certificate.md). Explain why the output contract is narrower before discussing the 3.45x lifecycle example and fragile-case loss.
