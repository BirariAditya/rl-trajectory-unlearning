# Implementation plan: trajectory deletion in offline reinforcement learning

Revised 29 September 2026 after the [novelty reassessment](novelty_reassessment.md) and three [specialist audits](audits/). The existing tabular code is a **baseline scaffold**, not yet the proposed paper contribution. The flagship hypothesis is a deletion-sensitive policy certificate whose *complete* request cost can be local under stated graph and action-margin conditions. A fixed-feature LSPI-control update is a second, independently testable hypothesis. GPU IQL experiments test transfer; they do not inherit a tabular certificate. Count subtraction, prioritized sweeping, warm starts, global Bellman residuals, action-gap tests, linear-matrix downdates, and adaptive fallback are established components, not stand-alone novelty claims.

### Research order and claim gates

**16 October active gate:** test selective decision repair using exact retained
weights and immutable margin indexes; include a frozen bank shared across new
input-policy branches. Compare against feature-deduplicated inverse updates and
matched policy-only output on paired aliased/unique mixed-sign features. Require
full trace equality, actual decision changes, setup-inclusive timing, tail costs
and separate allocations. Existing prediction-sensitivity work from 2015 further
narrows novelty; the screen alone is not a contribution. Complete this bounded
checkpoint before changing the presentation story or beginning GPU work.

**15 October completed certificate checkpoint:** [Checkpoint 06](checkpoints/06_policy_sequence_certificate.md) passes 98 tests and 900 requests across four methods. A policy-only certificate has conditional lifecycle savings versus exact inverse reuse (up to 3.45x in the tested stable settings), loses on fragile margins, and exposes expensive fallback tails. Separate allocation measurement is complete. Next remove two confounders—feature aliases and mismatched output richness—then test verified numerical bounds and conditioning. Keep the narrower output contract explicit; no full-state or publication-novelty claim is established. The candidate entry below records the earlier decision to run this experiment.

**15 October candidate gate:** after Checkpoint 05, test a cumulative certificate for the **policy-sequence output**, explicitly narrower than a matrix/weight trace. It combines an anchor inverse perturbation bound with precomputed all-state action margins and inducts over every distinct input policy. [Proof, cost and prior art](audits/lspi_policy_certificate.md). The implementation and exact Sherman–Morrison control pass 98 tests; the full repeated timing and separate allocation experiment is running. No breakthrough claim until total lifecycle cost, fallback regimes and prior art are assessed. If the gain comes only from repeated feature aliases or a weaker output interface, narrow the claim and add matched prediction-only controls before expanding.

**14 October continuation decision:** [Checkpoint 05](checkpoints/05_lspi_matched_reuse.md) completes equal repeated-policy reuse ahead of the planned continuation window. The five-method matrix has 810 exact matches, 80 passing tests, and equal memoized solve/action-score counts. Memoized replay has lower request and lifecycle medians in five of seven settings, losing on the two larger switch fixtures. Keep this conditional finding and both failure regimes. The next experimental gate is separately measured memory, numerical factorization/conditioning and longer deletion sequences; a useful deletion-specific theorem remains unproved. Compare costs within each recorded session, not absolute milliseconds across Checkpoints 04 and 05.

**14 October decision, superseding the earlier next steps:** [Checkpoint 04](checkpoints/04_lspi_cost.md) completes the first four-method LSPI cost matrix: 74 passing tests, 63 runs, 162 requests and 648 matching executions. Cached memoization wins request medians in all seven settings; replay wins lifecycle medians in two. Keep both as baselines. The next controlled implementation is equal repeated-policy reuse on both assembly paths, followed by numerical factorization, memory and adverse-conditioning comparisons. No new algorithm or useful deletion-time theorem is established. The dated preparation phases below govern work through the presentation; the original week-based phases remain the longer research roadmap.

| Date / phase | Deliverable and acceptance gate | Status |
| --- | --- | --- |
| 13–14 October: freeze evidence | Archive all LSPI costs and source manifest; reconcile claims, progress and study notes | Completed, Checkpoint 04 |
| 14–15 October: understand and explain | Work through the tabular and two-state LSPI demos; explain every equation and why stronger baselines changed conclusions | Guide and speaker notes prepared; presenter walkthrough remains |
| 15–16 October: targeted continuation | Match memoization across assembly methods; add only a bounded experiment with exact trace checks and complete cost reporting | Completed early on 14 October: Checkpoint 05; further expansion requires a separate archive |
| 16–17 October: presentation freeze | Reconcile every numeric claim with archived evidence; rehearse comprehensive story and professor questions | Outline prepared; rehearsal remains |
| 18 October: present and choose next gate | Present the full journey, including negative results; agree on the learner, baseline and proof needed for a paper claim | Planned |
| After presentation: numerical / novelty gate | Unique-feature/prediction-only controls, verified floating-point bounds and conditioning; focused novelty audit before neural scale-up | Exact inverse control and Python allocation pilot complete; numerical and novelty gates remain |

See the [presentation preparation plan](presentation/2026-10-18/README.md). These dates are work priorities, not scheduled automatic runs.

**7 October decision:** [Checkpoint 03](checkpoints/03_global_controls_and_lspi.md) completed 4,158 global-control executions with exact agreement and 60 passing tests. Classical elimination is cheaper over full bridge/anchor-invalidation lifecycles than the current pruned-local method. Keep Track A as a reference and prioritize Track B's efficiency/novelty gate. Its prototype exactly replays the declared finite LSPI schedule from a data-independent initial policy, including switches, shared features, cycles and singularity. [The worked example](checkpoints/lspi_control_example.md) shows why one-shot subtraction is insufficient. Next compare scratch, cached-statistic, target-correction, and incremental-solve baselines with full preparation/copy/policy-scan costs; a matrix identity or win over a weak scratch baseline is not enough. GPU work stays conditional on that gate.

**5 October update:** [Checkpoint 02](checkpoints/02_scc_controls.md) is complete: exact SCC planning, six stress regimes, 42 passing tests, and 5,688 method executions with exact policy/value agreement. The matched SCC control substantially narrows the modular-case advantage; weak bridges, anchor invalidation, and dense influence do not favor pruning. [FTVI explicitly combines action elimination and smaller components](https://arxiv.org/abs/1401.3910), so that combination is not a viable novelty claim by itself. Next add a lean global planner without repair-index overhead and matched action elimination, then decide whether a deletion-specific work theorem survives or Track B should become primary. The SCC solver is a controlled internal baseline, not an FTVI reproduction or a mature optimized planner; see [its contract](audits/scc_planning_checkpoint.md).

Live implementation status and validation evidence are maintained in [PROGRESS.md](PROGRESS.md). Implemented: baseline harness, rational oracle, ordinary sweeping, repeated scaling, restricted no-change certificate, exact backward-closed block repair, optional action pruning, and cyclic sequential comparisons with warm policy iteration. The [block-repair proof](audits/local_repair_proof.md) explicitly treats action elimination as prior art. The general margin-sensitive deletion theorem and LSPI-control bridge remain staged work; the novelty gate is still open.

| Track | Proposed claim | Gate before expanding |
| --- | --- | --- |
| A: finite empirical MDP | Equality of the *deployed policy* to a specified retained-only learner, with deletion-dependent certificate/work bound in favorable cases | Prove a sparse verification procedure without an uncounted full scan; beat warm-start planning in total latency; exhibit hard global cases |
| B: frozen-feature LSPI control | Exact replay of a declared retained-only LSPI schedule, including changes to policy-dependent targets | Initial rational replay implemented; next compare against cached/incremental LSPI, account for matrix copies/solves and policy scans, and establish conditioning plus a useful work bound |
| C: neural offline RL | Empirical retraining-relative fidelity and retained utility at a measured compute budget | Beat strong fine-tuning on a nontrivial split, with multiple retrains, attacks, tasks, and fully charged GPU/preprocessing cost |

These are hypotheses, not results. If Track A's certificate still scans the whole state-action graph or matches old sensitivity results, pivot to Track B or a rigorously characterized failure-mode paper by week 4. Do not spend the GPU budget to compensate for a missing algorithmic claim.

**Checkpoint update:** [Checkpoint 01](checkpoints/01_exact_local_repair.md) establishes exact block repair with chosen-action changes, cycles, sequential requests, and preparation-inclusive comparisons. Safe pruning can exclude provably dominated bridge actions; dense admissible influence remains global. This passes a concrete correctness/measurement gate but leaves the novelty gate open because action elimination and incremental planning are established. Before neural scale-up, add mature SCC/action-elimination baselines and timed anchor-invalidation regimes, then decide whether to strengthen Track A or pivot to Track B.

## 1. Define the estimand before coding

Let `D` be a fixed collection of complete trajectories, each with a unique trajectory ID. A later data schema may add a user ID for block deletion. A deletion request names whole trajectory IDs. We initially exclude partial-trajectory deletion, nonstationary data, and changes to the environment itself. Split `D = D_retain ∪ D_forget` before running any unlearning method.

Define a deterministic learner `A` as follows:

1. Build a discounted empirical MDP from `D` using a fixed convention for unseen state-action rows in every run.
2. Solve for the optimal action values with a declared convergence tolerance or an exact small-MDP reference solver.
3. Return a greedy policy with a fixed action order for ties.

The counterfactual target is `π_ref = A(D_retain)`. It is a *specific learner on a specific retained dataset*, not an unknowable environment-optimal policy. The method receives the full-data learner state and deletion IDs and produces `π_U`. The primary questions are whether `π_U` agrees with `π_ref`, retains value, and costs less than rebuilding and solving from scratch.

**Claim boundary:** equality of the deployed deterministic policy to `π_ref` is a behavioral result for this learner. It does not imply that the critic, caches, parameters, or output distribution are indistinguishable from a fresh training run. If these objects are released, evaluate them separately.

## 2. Repository and API contract

Keep manuscripts and code separate. The planned layout is:

```text
Research/
  literature_review.md
  implementation_plan.md
  novelty_reassessment.md
  audits/                 # dated primary-source and theorem-risk notes
  src/
    rl_unlearning/
      __init__.py
      data.py              # Transition, Trajectory, validation and splits
      tabular.py           # additive empirical-MDP statistics and subtraction
      planner.py           # reference tabular planning
      unlearning.py        # local repair, certificate, fallback
      # staged additions: baselines.py, experiments.py, metrics.py,
      # sparse_certificate.py, fixed_features.py, lspi_control.py, deep/
  tests/
    test_tabular.py         # initial end-to-end and model tests
    # staged additions: test_planning.py, test_sparse_certificate.py,
    # test_lspi_control.py, etc.
  scripts/
    run_tabular_demo.py
  # staged additions: configs/, results/, packaging metadata
```

Keep the first tabular implementation standard-library-only, with `unittest` and a runnable demo script. Add editable installation, a pinned `pyproject.toml`, NumPy, `pytest`, and config files only when the prototype needs them. The deep phase can add PyTorch, Minari/D4RL compatibility, and task-specific libraries after the tabular gates pass. Record package versions, Git revision, seed, config, deletion IDs, and device in every experiment result file.

Core interfaces, which may be adjusted as implementation reveals constraints:

```python
data.Transition, data.Trajectory
tabular.EmpiricalMDP(states, actions, gamma, trajectories)
model.delete_trajectories(trajectory_ids) -> changed state-action rows
planner.value_iteration(model, tolerance, max_sweeps, initial_q) -> PlanResult
planner.bellman_residual(model, q_values) -> float
unlearning.delete_and_repair(model, old_q, trajectory_ids, budget) -> RepairResult
# staged: metrics.evaluate(result, retained_reference, environment, counter)
```

The first `EmpiricalMDP` stores the full-data empirical statistics and active trajectories; the planner output is passed separately to deletion and repair. The deletion routine rejects unknown or already-deleted IDs. It mutates its model argument, so experiments must create an independent model for each method. All methods receive exactly the same retained dataset and evaluation states. Mark the current package as **implemented prototype**; every proposed new module above is staged work, not existing code.

## 3. Data and empirical-model rules

A transition stores `(state, action, reward, next_state)`, where `next_state=None` means termination and has zero continuation value. A trajectory stores an ordered sequence of transitions and a unique ID. Time-limit truncation has no representation in this first schema; reject or preprocess truncated episodes with a declared bootstrap convention before importing them. This avoids silently treating a timeout as a zero-value terminal.

For each trajectory `τ`, store or recover additive statistics `N_τ(s,a,s')`, `N_τ(s,a)`, and `R_τ(s,a)`. Rewards are modeled by their mean conditional on `(s,a)`; if a task needs transition-dependent mean rewards, extend the sufficient statistics to `R_τ(s,a,s')` and keep that choice fixed throughout all methods. Use integer counts and Python binary64 floating rewards.

The first learner uses empirical frequencies for observed rows:

```text
P_D(s'|s,a) = N_D(s,a,s') / N_D(s,a)
r_D(s,a)   = R_D(s,a) / N_D(s,a).
```

When `N_D(s,a)=0`, that row has reward zero and terminates with probability one. This convention remains fixed after deletion. Use `0 ≤ γ < 1` and a stable action order. Record `γ`, the empty-row convention, and the planner stopping rule in the experiment manifest. A smoothed model is a possible later alternative learner, but introducing a prior would require rebuilding all references under the same rule. The first implementation retains trajectories in the model so subtraction removes their contributions exactly. Numerical equality should be checked within a declared binary64 tolerance against rebuilding the model from `D_retain`; do not assert bitwise equality for floating reward sums.

## 4. Staged implementation and acceptance gates

**Current status:** Phases 0–3 have implemented tabular learners, an independent rational reference, randomized correctness checks, and a timing ledger. Exact local/pruned repair and the SCC planning ablation are also implemented. Broad adversarial benchmarking, mature external planner comparisons, and the general sparse-certificate theorem remain open; consult `PROGRESS.md` for dated evidence.

### Phase 0 — Baseline scaffold and a tiny known MDP

Create the standard-library package, a runnable demo, and a 3–5-state hand-computable MDP. Generate a fixed logged dataset with trajectory IDs. Define the output object as `policy + optional diagnostics`; decide separately which diagnostics are part of a published release. Add a seed manifest before running sweep experiments.

**Gate:** one command builds the empirical MDP, prints its transition rows, solves it, and reproduces the same policy from the same inputs. The hand-computed action values agree within the stated planner tolerance.

### Phase 1 — Counterfactual retraining and exact data subtraction

Implement data validation, additive contributions, model construction, and rebuild-from-retain. For every test deletion set, compare a fresh model built from `D` and then mutated by `delete_trajectories(F)` with a model built directly from `D_retain`, checking counts, transition probabilities, rewards, and terminal handling. Run a converged reference planner on each retained model. Deleting zero IDs must leave the original model and policy unchanged.

**Gate:** property tests over random small MDP logs and deletion subsets pass; sequential deletion `F₁` then `F₂` matches deleting `F₁ ∪ F₂` once. Check the aggregate count is nonnegative and each transition row sums to one. Save both scratch aggregation time and scratch planning time, because speedup denominators differ.

### Phase 2 — Local Bellman repair

After subtraction, enqueue every changed `(s,a)` row. Maintain predecessor actions for all transition edges with nonzero probability under the **retained** model, including actions that are currently non-greedy. Recompute a row's Bellman target, update its Q entry, and enqueue predecessors when the change can alter their targets. Use a priority queue keyed by residual magnitude, with lazy stale-entry handling. Avoid assuming that a small changed row affects only nearby *states*: discounted influence can travel through cycles and bottlenecks.

For the first version, a global Bellman residual scan after a bounded batch is acceptable and must be counted in runtime. Optimize residual maintenance only after correctness is established. Expose a work budget (`max_backups`, `max_seconds`, or both) and deterministic fallback to a full warm-started planner if repair does not meet the certificate.

**Gate:** on random sparse and dense MDPs, local repair plus fallback returns a policy that agrees with the retained reference whenever the certificate reports success. Add a regression case in which deleting data makes a previously non-greedy action optimal; predecessor propagation through that action is mandatory.

### Phase 3 — Behavioral certificate and adaptive decision

For retained-model Bellman operator `T_R`, compute the global residual `η = ||T_R Q_U - Q_U||∞`. Contraction yields the conservative error radius `ε_Q = η/(1-γ)` around the retained-model optimal Q-function. At state `s`, if the gap between the top two Q values under `Q_U` is **strictly greater** than `2 ε_Q`, the greedy action there matches the unique retained-model optimal action. Report a Boolean mask of certified states, the maximum residual, minimum gap, and every failed state's reason. All states used by the deployed policy must pass for a full-policy equality claim.

Handle zero/tiny action gaps carefully: a failed gap test is *inconclusive*, not evidence that the policy is wrong. For finite tabular MDPs, fallback may use a more accurate planner and, if needed, an independent reference policy-iteration check under fixed tie-breaking. A finite-tolerance planner alone cannot certify equality at exact ties without extra care. The certificate compares against the optimal policy of the retained empirical model; describe any separate numerical tolerance in the reported result.

**Gate:** no false certificates on exhaustive small-MDP tests and randomized adversarial tests within numerical tolerance. Measure the cost of residual scans, graph construction, queue operations, and fallback. Produce at least one example where repair is cheap and one where a one-trajectory deletion forces broad recomputation.

### Phase 3A — The new sparse-certificate research gate

Treat Phase 3's global residual/action-gap certificate as an **independent oracle** for testing, not as the intended main theorem. Let `C` denote changed empirical state-action rows. Develop a sound bound on their effect on action choices that accounts for direct row shocks, discounted reachability, and the pre-deletion action margins. Test the special case where only non-policy rows change: the old policy's value is unchanged, so an exact changed-row advantage check may certify it without rescanning all rows. For changed policy rows, use a rigorously bounded influence envelope or a certified policy-value correction, and prove which states/actions can be skipped. Analyze terminal rows, transition edges added or removed by deletion, exact ties, and finite-precision errors.

Instrument **all** work: reverse index build/update, model subtraction, envelope propagation or linear solves, every checked action, numerical certification, fallback, and output materialization. A per-request complexity claim must specify preprocessing and stored memory; a fresh `O(SA)` scan is not local work. Compare with plain prioritized sweeping plus its check and warm-started complete policy/value iteration. Construct a bottleneck chain and a near-tie family where global propagation is necessary. Search sensitivity and dynamic-MDP prior art before naming the theorem.

**Gate by week 4:** a written theorem statement and proof outline, exhaustive small-MDP counterexample search, and at least one instance class with a valid sublinear **verification** cost after stated preprocessing. If the bound is loose or every request still scans globally, stop making the certified-locality claim and pivot to Phase 5's LSPI-control work or a failure-mode study.

### Phase 4 — Deletion difficulty, adversarial cases, and ablation study

Generate chain, branching chain, grid/bottleneck, and dense random MDPs. Predefine deletion strata: uniform random; rare-state coverage; high reward contribution; bridge/bottleneck; high versus low retain/forget visitation overlap; and sequential 1%, 5%, 10%, 20% deletions. Add shared-state deletion where suppressing forget values would **over-forget**, and camouflage/retained-backdoor deletion inspired by [UBA-ORL](https://arxiv.org/abs/2609.22711). Record the predecessor subgraph size, changed empirical rows, action-gap distribution, coverage loss, and any triggered safety failure before comparing algorithms. Counterfactual fidelity and safety must be reported separately: the retained-only reference itself may reveal a retained backdoor.

Run these baselines:

1. scratch rebuild plus scratch planning (the reference and time denominator);
2. no operation;
3. full planning warm-started from the original Q;
4. local prioritized sweeping without a certificate or adaptive fallback;
5. existing global-residual certified repair with fallback;
6. proposed sparse certificate/repair, with and without its fallback.

Retained-data fine-tuning and negative-gradient/refit are meaningful mainly once a parameterized learner exists; include them in fixed-feature/deep experiments, not as artificial tabular-model comparisons. TrajDeleter is a direct deep/offline trajectory baseline and belongs in the deep phase if its code and training setup can be aligned.

**Gate by week 6:** at least three independent dataset seeds and several deletion requests per stratum in pilots; increase seeds for final figures. Plots show *total* wall-clock and complete work against retraining-relative policy agreement and return. Report fallback frequency, verification overhead, index memory, amortization across repeated requests, and worst-case strata rather than only average speedup. The proposed method must improve on warm-started full planning in at least one preregistered regime and detect its hard cases.

### Phase 5 — Frozen-feature evaluation, then LSPI control

Start with *fixed-policy evaluation* as a correctness check. Given fixed features `φ` and a fixed evaluation policy, accumulate per-trajectory contributions to

```text
A = Σ_i φ_i (φ_i - γ φ'_i)^T + λI,     b = Σ_i φ_i r_i.
```

Delete contributions, solve `A_retain w = b_retain`, and compare to a full retained-data rebuild and solve. Test rank downdates only after conditioning checks; fall back to a fresh factorization when the matrix is ill-conditioned or the update is large. Record condition number, coefficient error, evaluation-value error, time, and memory.

The potential contribution is **control**, where a changed policy changes the next-action feature in retained transitions. For an LSPI policy `π`, define `G_R(z)=Σ_{i∈R:s'_i=z}φ_i`. In its normal matrix, a changed next action at successor state `z` contributes the rank-one target correction

```text
−γ G_R(z) [φ(z,π_new(z)) − φ(z,π_old(z))]ᵀ.
```

This correction is in addition to subtracting deleted samples. Fix an LSPI initialization, iteration count or deterministic stopping rule, feature map, regularizer, solver, and tie order. Compare every post-deletion policy-iteration step against a scratch retained-only implementation, especially after a policy switch. Do not assume regularization makes nonsymmetric LSTD matrices invertible; check conditioning and use a fresh factorization on failure. A frozen random feature map or encoder trained only on public/non-deletable data is required for a head-only exactness claim. Store successor aggregates and index cost; compare against LSPI's own incremental matrix-update baseline, not only naive scratch refitting.

**Gate by week 8:** fixed-policy deleted-data coefficients match retained-only recomputation, and the policy-switch control case matches a full retained-only LSPI replay at every declared iteration. A useful cost theorem must involve the number of deleted samples and changed successor states, include aggregate maintenance and factorization work, and show a regime faster than strong incremental LSPI. If only fixed-policy evaluation works, keep it as a baseline and do not call it the main result.

### Phase 6 — GPU offline-RL transfer pilot

Choose one algorithm family first: IQL with a validated implementation, one original D4RL-v2 snapshot, and an environment/deletion slice where no-op and retained-only retraining are measurably different. [Minari's D4RL page](https://minari.farama.org/datasets/D4RL/index.html) states that some releases are reproductions, so label Minari as a separate dataset track rather than treating it as the same log. HalfCheetah medium-expert is a pilot candidate, subject to our own signal check. Freeze dataset hash, episode parsing and timeout handling, preprocessing, architecture, training steps, checkpoint selection, seeds, and deletion IDs. Split by complete episode before training. Train the original model and multiple independent scratch retained-data references on GPU.

Build the strong baselines first: no-op, scratch retrain, budget-matched uniform retained-only fine-tuning, warm-started full retained update, gradient ascent plus retain refit, and [TrajDeleter](https://www.ndss-symposium.org/wp-content/uploads/2025-126-paper.pdf). Compare our prioritized/adaptive retained-only repair with its own always-local ablation. Add [Relative Fisher Forgetting](https://openreview.net/pdf?id=TidLO0qdp0) only if a comparable diffusion policy is included; do not compare its published numbers with IQL under a different learner. Check the currently inaccessible NeurIPS 2026 **CURe** paper before finalizing deep novelty claims. Use [TOUR](https://arxiv.org/abs/2607.21111)'s matched non-members, multiple attack families, retained-utility anchors, and retraining-relative calibration, adapted explicitly to IQL.

There is no tabular Bellman certificate for a changing neural representation. Train a frozen-feature bridge only with features/normalization that have never seen deletable data; a learned encoder from the full log invalidates a head-only removal guarantee. For full IQL, treat the method as retraining-relative empirical evidence and the stopping rule as a diagnostic, not a proof. Do not call small parameter distance, a membership-attack score, or high retained return proof of exact removal. If the one-task pilot is stable and beats strong fine-tuning at equal compute, expand to HalfCheetah, Hopper, and Walker2D on medium/medium-expert; medium-replay and AntMaze are hard-coverage stress tests. Parallelize independent seeds and retained retrains across available GPUs and keep one source of truth for deletion IDs across methods.

**Gate by week 10:** before scale-up, scratch retrains have reproducible returns and evaluation distinguishes no-op, retraining, and deliberate policy collapse. A three-seed, two-request pilot must demonstrate an advantage over compute-matched fine-tuning on retraining-relative behavior without losing retained return. If not, change the split once under a preregistered rule or narrow the paper to the rigorous learner. Final central deep claims require at least five independent training seeds, multiple fixed deletion requests per stratum, a held-out environment/deletion matrix, and hierarchical uncertainty over seeds and requests. If a baseline cannot be reproduced, state the limitation and preserve its intended objective in the comparison description.

## 5. Metrics and experimental protocol

The primary tabular metric is the fraction of evaluation states on which `π_U(s) != π_ref(s)`, reported both uniformly over all states and weighted by a predefined retained-state visitation distribution. Also report `||Q_U - Q_ref||∞`, discounted value gap under the retained empirical MDP, held-out environment return, certified-state fraction, and **fraction of states/actions actually examined by the sparse certificate**. For stochastic deep policies use action-distribution TV/KL or a task-appropriate action-distance metric. Make state weighting explicit; near-perfect agreement on common states can hide a critical bottleneck error. Report coverage loss, zero-count rows, and unsafe-action/backdoor rate as **separate** outcomes, since retraining fidelity alone does not imply safety.

Efficiency metrics are total wall-clock from deletion request to deployable output, counts of Bellman backups, **all certificate rows/edges/linear-solver operations**, predecessor edges traversed, peak memory, and speedup versus `A(D_retain)` from scratch and warm-started full planning. Include index-building and stored sufficient-statistics costs separately as one-time training overhead, and show the number of deletion requests needed to amortize them. For repeated deletions report cumulative and per-request work. Exclude neither fallback nor required validation from the method's reported time; separately time optional privacy attacks that are common to every method.

Use paired deletion splits: every algorithm runs on the same original dataset and exact retained subset. Include multiple independent dataset-generation seeds and, for neural learners, training seeds. Bootstrap confidence intervals at the **dataset/seed** level rather than treating thousands of correlated states as independent samples. Reserve some environments and deletion strata for final evaluation after algorithm choices are fixed. Report negative cases openly.

Membership inference and trajectory auditing are secondary empirical checks. Use matched non-members and multiple attacks; an attack failing to distinguish membership is not equivalent to a policy matching retraining. Plot privacy/audit outcomes alongside return and reference-relative behavior to expose policy collapse.

## 6. Four-month research phases and decision schedule

| Period | Deliverable | Decision to proceed |
|---|---|---|
| Weeks 1–2 | Freeze learner and output contract; verify existing prototype, independent retained reference, timing ledger, and strong planning baselines | Counts and policy reference reproduce; end-to-end latency is measured; changed/nonchanged-policy-row and tie cases work |
| Weeks 3–4 | Prove or falsify sparse action-certificate theorem; audit dynamic-MDP sensitivity prior art | A valid local verification procedure with no hidden global scan exists in a stated favorable regime, or pivot |
| Weeks 5–6 | Implement sparse prototype, hard global examples, repeated deletion and coverage/safety strata | Beats warm-start planning in total time in at least one preregistered regime and detects hard deletions |
| Weeks 7–8 | Implement frozen-feature LSTD baseline and policy-switch-aware LSPI control; prove exact replay under stated conditions | Matrix identity and each LSPI step match independent retained-only replay; switch-rank cost can be stated honestly |
| Weeks 9–10 | Freeze GPU dataset/learner, multiple scratch retrains, matched controls and attacks; one-task IQL pilot | Nontrivial no-op/retrain separation and advantage over budget-matched fine-tuning, or narrow claims |
| Weeks 11–13 | Five-seed/multi-task deep matrix only if pilot passes; holdout deletion strata, ablations, safety checks | Effects survive across seeds, tasks and attacks; no omitted compute or encoder contamination |
| Weeks 14–16 | Final prior-art refresh, theorem review, plots, paper and reproducibility package | Every headline claim has a proof/test/table; failed gates and scope limits are explicit |

Keep GPU spending staged: a small pilot measures training duration and seed variance before a large sweep. Reserve substantial time for scratch retained-data retrains because they define the target; reduce task breadth before reducing reference seeds. Recalculate the matrix after measuring one base run and one retrain on the available GPU; record actual GPU-hours, GPU-seconds per deletion, and peak VRAM. If deep results are unstable by week 10, finish the tabular/fixed-feature contribution and present neural results as a limit rather than scaling an inconclusive sweep.

## 7. Paper claim checklist

- State exactly what is forgotten: trajectory IDs, user blocks, or reward records.
- State exactly what is returned: policy only, or policy plus critic/metadata.
- Specify the empirical learner, empty-row convention, terminal handling, tolerance, and tie-breaking; include priors if a smoothed learner is added later.
- Distinguish equality to the retained-data learner from privacy indistinguishability and environment-optimality.
- Compare with ordinary prioritized sweeping and warm-started full planning; speedups over scratch alone are insufficient.
- Include TrajDeleter and TOUR-aware evaluation in the deep phase where comparable; check RFF for diffusion policies and CURe once its paper is accessible.
- State whether the sparse certificate truly avoids a full graph scan and charge preprocessing, verification, and fallback to latency. Standard global residual bounds and adaptive switching are not novelty claims.
- Distinguish exact frozen-feature head/control replay from full neural-checkpoint unlearning; document feature and normalization provenance.
- Keep retraining fidelity, privacy auditing, and deployment safety as different endpoints.
- Show an MDP in which a single deletion has global influence. No universal sublinear-work claim follows from local repair.
