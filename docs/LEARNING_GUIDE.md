# Learning guide: trajectory deletion in offline reinforcement learning

**Status: 14 October 2026; evidence through Checkpoint 05.** This is a study companion to the implemented repository, not a claim that a publishable new algorithm has been established. [Checkpoint 04](../checkpoints/04_lspi_cost.md) exposed a strong cached baseline. [Checkpoint 05](../checkpoints/05_lspi_matched_reuse.md) gives both methods equal policy-evaluation reuse; memoized replay then has the lowest request/lifecycle medians in five of seven settings. Start with this guide, run the tiny demos, and use the linked contracts when a proof detail matters.

## 1. The question in one minute

An offline RL learner trains on a fixed log of trajectories. Later, someone requests deletion of one or more trajectories. We want an updated agent that behaves like the **same specified learner** trained on the retained log alone, ideally with less work than starting over.

Let:

- \(D\): all logged trajectories;
- \(F\): trajectory IDs to forget;
- \(D_R=D\setminus F\): retained trajectories;
- \(A\): a fully specified learning procedure;
- \(U\): an update procedure that receives the old learner state and \(F\).

The target is \(A(D_R)\). For a deterministic policy learner, our first question is whether the policy returned by \(U\) equals the policy returned by \(A(D_R)\). This is a **counterfactual** comparison: what would this exact learner have produced if the deleted records had not entered its input?

This does not ask the agent to fail on a forgotten state. Imagine a one-step task with two logged rewards of 1 for action \(a\) and one logged reward of 0.5 for action \(b\). Delete one \(a\) record. The retained log still supports \(a\), so a retained-only learner still chooses \(a\). Artificially lowering \(a\)'s value to “forget” the deleted record would move away from the target. This is why we compare to retraining, not just to a forgetting score. The broader distinction is explained in the [literature review](../literature_review.md).

## 2. Four meanings of “unlearned”

| Statement | What it means | What it leaves open |
| --- | --- | --- |
| Good retained return | The agent still earns reward on retained tasks | Whether the deleted records influenced it |
| Policy/value fidelity | The output agrees with the specified retained-only learner | Hidden critic, optimizer, cache, or metadata state |
| Attack resistance | Chosen audits cannot detect membership | Other attacks and formal exactness |
| Distributional exactness | The entire released output has the same probability distribution as counterfactual retraining | Only holds for the stated data and learner model |

The repository's rational tabular repair can exactly match a **declared finite empirical-MDP learner's policy and values**. Its LSPI prototype can exactly match the **declared finite LSPI schedule's matrices, weights, and policies**. The rational implementations use Python Fraction objects and interpret decimal numeric inputs by their decimal spelling; equality is relative to that declared arithmetic and learner. It is not a claim about an arbitrary binary64 or neural training run. These are narrow and valuable correctness claims. They do not imply privacy indistinguishability or exact removal from a neural policy. [Exact Unlearning in Reinforcement Learning](https://arxiv.org/abs/2606.04182) asks a stronger distributional question in a different online, environment-oracle setting; see the [comparison](../literature_review.md#exact-unlearning-in-reinforcement-learning-strongest-formal-benchmark-different-data-model).

The computed optimal values are values **inside the empirical model built from the log**. They are not measured return in the true environment. Matching retained-model policy/value can faithfully reproduce a poorly covered or unsafe retained-only learner. No continuous-control benchmark, real-environment retained-task return study, or GPU experiment has been completed in the documented checkpoints.

## 3. The tabular learner: from log to model to policy

A trajectory is a sequence of transitions \((s,a,r,s')\). Here \(s\) is the current state, \(a\) the action, \(r\) the reward, and \(s'\) the next state; \(s'=\mathrm{None}\) means termination. The present tabular code stores a unique trajectory ID and these transitions in [data.py](../src/rl_unlearning/data.py). Deletion is by complete trajectory ID. A time limit should not silently be encoded as a true terminal state; this first schema does not represent truncation separately.

For every state-action pair, [tabular.py](../src/rl_unlearning/tabular.py) keeps three additive quantities:

\[
N(s,a)=\text{number of visits},\qquad
N(s,a,s')=\text{number of transitions to }s',\qquad
R(s,a)=\sum r.
\]

When \(N(s,a)>0\), the empirical transition probability is \(P(s'|s,a)=N(s,a,s')/N(s,a)\) and mean reward is \(\bar r(s,a)=R(s,a)/N(s,a)\). When the pair has no data, the declared learner assigns reward zero and termination. This is a modeling convention, not an estimate that an unseen action is harmless in the real environment. It must remain the same in full-data training and retained-only retraining.

The planner solves the Bellman optimality equation:

\[
Q^*(s,a)=\bar r(s,a)+\gamma\sum_{s'}P(s'|s,a)\max_b Q^*(s',b).
\]

\(\gamma\), between 0 and 1, discounts future rewards. The sum excludes terminal continuation. The \(\max_b\) means that after reaching \(s'\), the agent chooses its best available action. The policy selects a maximizing action at each state; the declared action order resolves exact ties. [planner.py](../src/rl_unlearning/planner.py) uses binary64 value iteration, whereas [reference.py](../src/rl_unlearning/reference.py), [policy_iteration.py](../src/rl_unlearning/policy_iteration.py), and [scc_planner.py](../src/rl_unlearning/scc_planner.py) support exact rational validation/planning. “Exact” always means exact relative to this specified finite learner and its rational interpretation of inputs.

Deleting a trajectory subtracts its visit counts, transition counts, and reward sums. The **retained model** should equal one rebuilt independently from \(D_R\). But correct count subtraction alone does not finish control: a changed reward or transition may change a state's best action, and that change can propagate backward through predecessors.

### Tiny influence example

Suppose state \(x\) has an action leading to \(z\), where a reward-bearing action was observed. Another action at \(x\) terminates for a modest reward. If deleting a trajectory changes the preferred action at \(z\), the value of the **retained** transition \(x\to z\) changes even though that transition was not deleted. Thus a deletion at \(z\) can alter the chosen action at \(x\). In a connected or cyclic MDP, this influence can spread far. The [LSPI worked example](../checkpoints/lspi_control_example.md) makes the same issue numerical under shared linear-control targets.

## 4. The first repair route: residuals and action gaps

The initial [unlearning.py](../src/rl_unlearning/unlearning.py) starts with the full-data Q table, subtracts requested trajectories, and schedules Bellman backups for changed rows and their predecessors. It then checks a global Bellman residual:

\[
\eta=\lVert T_RQ-Q\rVert_\infty,
\]

where \(T_R\) is the retained-model Bellman operator, \(Q\) is the candidate action-value table, and \(\lVert\cdot\rVert_\infty\) means the largest absolute error over all state-action pairs. Discounting makes this operator a contraction, giving

\[
\lVert Q-Q_R^*\rVert_\infty\leq\frac{\eta}{1-\gamma}.
\]

Call the right side \(\epsilon_Q\). If the difference between the candidate's best and second-best action at a state is **strictly greater than \(2\epsilon_Q\)**, even the worst allowed Q errors cannot reverse their order. If every state passes, the candidate greedy policy equals the unique retained-model optimal policy. A failed gap test is inconclusive, especially at a tie; it does not prove the candidate policy is wrong. The implementation falls back to a full warm solve when needed.

The important cost fact: computing this global residual reads all state-action rows. A small repair queue does not make the *whole request* local if verification scans the model. This is why the project moved beyond that first certificate. See [README](../README.md) and the [implementation plan](../implementation_plan.md).

## 5. Exact block repair: why the boundary can stay fixed

[LocalPolicyRepair](../src/rl_unlearning/local_policy_repair.py) addresses a precise rational tabular learner. Let \(C\) be the source states of empirical rows whose normalized reward or transition distribution actually changed. Build the **old all-action support graph**: an edge \(s\to t\) exists if some action at \(s\) had a nonzero chance of reaching \(t\). Let \(B\) contain \(C\) and every state that can reach \(C\) along this graph. This is the backward closure of the changed states.

Why solve only \(B\)? No state outside \(B\) has a changed model row or a path into \(B\). Its control problem is unchanged, so its old optimal value remains valid. Solve the retained Bellman equations inside \(B\), using those exact outside values as boundary conditions. The combined vector satisfies the full retained Bellman equations, and discounting gives a unique optimal value. This reasoning handles cycles and changed actions **inside** \(B\). It is not valid to stop at an arbitrary fixed-radius neighborhood; such a cutoff can miss a predecessor. The full argument and deletion/index assumptions are in the [block-repair proof](../audits/local_repair_proof.md).

If the graph is modular, \(B\) can be small. If a bridge or dense graph connects most states to the change, \(B\) may be the whole state space. A request can also be routed to full planning under the method's budget/fallback rule. The reverse index, its maintenance, rational solves, and output copies cost time and memory; none are free preprocessing.

### Bound-based action pruning

An optional **anchor policy** \(b\) gives a known lower value \(L(s)=V^b(s)\). A conservative reward bound gives an upper value \(U=R_{\max}/(1-\gamma)\). For action \(a\),

\[
\overline Q(s,a)=\bar r(s,a)+\gamma U\sum_{t\ne\mathrm{terminal}}P(t|s,a).
\]

If \(\overline Q(s,a)<L(s)\), that action cannot be optimal, so its graph edges can be ignored. Strict inequality matters for exact ties. If a deletion changes an anchor-policy row, the old lower bound may be invalid: the current local implementation does a full solve and disables pruning. Another global comparator refreshes the bound efficiently. This distinction drives the later negative result; [Checkpoint 03](../checkpoints/03_global_controls_and_lspi.md) explains it. Action elimination has decades of prior art.

### SCC planning

A strongly connected component (SCC) is a set of states with paths both ways. Within a component, values are coupled; between components, a directed acyclic component graph can be solved in dependency order. [scc_planner.py](../src/rl_unlearning/scc_planner.py) uses exact policy iteration component by component. SCC planning can make a global planner efficient even when a dense solver looked expensive. It is a strong control, not a new unlearning invention. Read [Checkpoint 02](../checkpoints/02_scc_controls.md).

## 6. What the tabular experiments actually show

The repository has several successive checkpoints. They should be read in order because each stronger comparison changes the interpretation of the earlier speed numbers.

| Checkpoint | Supported result | Important limitation |
| --- | --- | --- |
| [01](../checkpoints/01_exact_local_repair.md) | Exact block repair matches retained-only policy/values in cyclic and sequential tests; a constructed dominated-bridge case selects 4 of 16 states | Dense influence selects all states; classical pruning and incremental planning predate this work |
| [02](../checkpoints/02_scc_controls.md) | 5,688 matched method executions across 711 requests with zero reported policy/value failures; SCC controls narrow a dense-solver speed advantage | Internal classical SCC baseline, not a reproduced optimized FTVI implementation |
| [03](../checkpoints/03_global_controls_and_lspi.md) | 4,158 further method executions across 594 requests with zero reported policy/value mismatches; lean global classical elimination is faster over bridge and anchor-invalidation lifecycles | Synthetic 16/32-state timings; methods differ in preparation and implementation; no broad speed or novelty claim |

The 32-state [global summary](../output/benchmarks/global_checkpoint_summary.md) records **full lifecycle** median time, including original training, preparation, and all requests. In its constructed dominated-bridge regime, current pruned-local SCC takes 96.257 ms, while classical global elimination takes 32.969 ms. Under anchor invalidation, the corresponding times are 835.267 and 39.642 ms. Local SCC is still useful in a modular regime: 7.930 ms versus 24.282 ms for lean warm SCC. Dense and weak-bridge regimes are essentially global. These are descriptive synthetic results with three dataset seeds and reused timing repeats, not significance tests. They establish that a result versus a weak dense baseline can disappear under stronger classical controls.

The independent tiny-MDP rational enumeration checks correctness separately from the methods' shared inner solver. The larger benchmark methods also share code, so “zero mismatch” is verification evidence under the tested cases, not an external proof of all inputs. Checkpoint 03 recorded 60 passing tests. With the later LSPI baseline and memoization regressions, 74 tests pass and [CI run 38338893080](https://github.com/BirariAditya/rl-trajectory-unlearning/actions/runs/38338893080) succeeded. No GPU experiment appears in the completed checkpoints.

## 7. The second route: fixed-feature LSPI control

Least-Squares Policy Iteration (LSPI) uses a fixed feature vector \(\phi(s,a)\) to approximate \(Q(s,a)\approx\phi(s,a)^\top w\). The feature map, initial policy, action order, discount, regularization, and number of iterations are fixed **independently of deletable data**. This last condition matters: deleting records from a trained encoder would require unlearning the encoder too.

For a retained set \(R\) and current policy \(\pi\), one policy-evaluation step solves

\[
A_R(\pi)w=b_R,
\]
\[
A_R(\pi)=\lambda I+\sum_{i\in R}\phi_i\big[\phi_i-\gamma\phi(s_i',\pi(s_i'))\big]^\top,
\qquad
b_R=\sum_{i\in R}\phi_i r_i.
\]

Here \(i\) indexes retained transitions, \(\phi_i=\phi(s_i,a_i)\), \(\lambda I\) is the declared regularizer, and a terminal successor contributes a zero feature vector. Each evaluation solves a linear system for \(w\); then the policy chooses the action with the highest \(\phi(s,a)^\top w\) at each state. Repeat for the **declared finite number** of iterations. The code records cycles or budget exhaustion; it does not silently call them convergence. Final weights evaluate the last iteration's input policy, while the returned final policy is the greedy improvement of those weights. See the [LSPI contract](../audits/lspi_control_contract.md).

Deleting samples changes \(A\) and \(b\). But after a policy improvement, \(\pi(s_i')\) can change for a **retained** transition. So a one-time subtraction at the old policy is generally insufficient. Group retained transitions by successor \(z\):

\[
G_R(z)=\sum_{i\in R:\,s_i'=z}\phi_i.
\]

When the target action at \(z\) changes from \(\pi_{\rm old}(z)\) to \(\pi_{\rm new}(z)\), apply

\[
\Delta A=-\gamma G_R(z)
\left[\phi(z,\pi_{\rm new}(z))-\phi(z,\pi_{\rm old}(z))\right]^\top.
\]

This is an outer-product correction. Its sign comes from the negative \(\gamma\phi(s_i',\pi(s_i'))\) term in \(A\). [lspi_control.py](../src/rl_unlearning/lspi_control.py) implements both direct retained-only matrix assembly at every step and an aggregate replay path with these corrections. Both start from the same data-independent initial policy and execute the same finite schedule.

### Strong cached comparison: reconstruct instead of replaying target switches

The later [cached baseline](../src/rl_unlearning/lspi_baselines.py) stores three retained-data aggregates:

\[
C_R=\lambda I+\sum_{i\in R}\phi_i\phi_i^\top,\qquad
b_R=\sum_{i\in R}\phi_i r_i,\qquad
G_R(z)=\sum_{i\in R:s_i'=z}\phi_i.
\]

At **every** policy-evaluation step, it reconstructs

\[
A_R(\pi)=C_R-\gamma\sum_zG_R(z)\phi(z,\pi(z))^\top.
\]

This is algebraically the same matrix as direct assembly, but it avoids visiting every retained sample at every iteration. After a deletion, subtract its contributions from \(C_R\), \(b_R\), and \(G_R\). A zero successor aggregate is omitted and can reappear if a later deletion uncancels a sum. This is established sufficient-statistic reuse, not a new unlearning algorithm.

An even stronger baseline memoizes the exact matrix, right-hand side, and weights for a policy that repeats **within the same deletion request**. Each request begins with an empty memo because the retained data and matrices may have changed. The learner still emits every scheduled iterate and scores the actions at every step; it merely reuses a repeated policy evaluation. Thus the declared finite trace is preserved. The cache can hold up to the iteration budget's worth of matrices and weights, and its memory cost is not measured by the checkpoint's partial counters.

### A worked two-state control reversal

The [worked fixture](../checkpoints/lspi_control_example.md) uses states \(x,z\), actions \(a,b\), four one-hot features, \(\gamma=\lambda=1/2\), and three LSPI iterations. The deleted trajectory is terminal \((z,a,1)\); the retained log includes \((x,a,0,z)\), terminal \((z,a,0)\), terminal \((x,b,1/16)\), and terminal \((z,b,1/4)\).

| Computation | Policy at \((x,z)\) |
| --- | --- |
| Original full-data learner | \((a,a)\) |
| One-shot downdate, then stop | \((b,b)\) |
| Correct retained-only three-step learner | \((a,b)\) |

Why does \(x\) switch back to \(a\)? After deletion, policy improvement chooses \(b\) at \(z\). The retained \(x\to z\) transition must now use feature \(\phi(z,b)\) as its successor target. Correcting that retained target raises the evaluated \(Q(x,a)\) to \(1/18\), above \(Q(x,b)=1/24\). The one-shot method misses this. Run [run_lspi_demo.py](../scripts/run_lspi_demo.py) to inspect every exact matrix and weight; the saved [JSON](../output/benchmarks/lspi_demo.json) includes the data, traces, settings, and source hashes.

The prototype uses exact rational dense solves and scans successor groups. Even a “rank-one” correction touches a matrix, and the full request also copies statistics, compares policies, scores actions, and solves a linear system. Exact equality to the retained-only finite LSPI trace is a **correctness substrate**. Original LSPI already discussed incremental evaluation and policy-dependent successors; the [contract](../audits/lspi_control_contract.md) is explicit on prior art and singular matrices. A singular evaluation fails without committing the deletion.

### Checkpoint 04: what the complete-cost pilot found

The [archived summary](../output/benchmarks/lspi_checkpoint_summary.md) compares four methods on the same declared four-iteration rational learner: direct scratch assembly, cached \(C,b,G\) reconstruction, cached reconstruction with within-request policy memoization, and target-switch replay. Across **63 runs, 162 deletion requests, and 648 method executions**, all compared exact matrix/RHS/weight/policy traces and failure statuses matched the retained-only reference; the report records zero mismatches. The 74-test suite and [CI run 38338893080](https://github.com/BirariAditya/rl-trajectory-unlearning/actions/runs/38338893080) also pass. Equality of traces does not mean equality of work.

| Synthetic setting | Cached memo request median | Target-replay request median | Cached memo full lifecycle | Target-replay full lifecycle |
| --- | ---: | ---: | ---: | ---: |
| Shared features, 128 initial transitions, 4 features | 1.286 ms | 1.309 ms | 12.052 ms | 10.711 ms |
| Shared features, 512 initial transitions, 4 features | 2.206 ms | 2.455 ms | 27.589 ms | 25.858 ms |
| Switch fixture, 160 initial transitions, 4 features | 1.073 ms | 1.337 ms | 8.397 ms | 9.344 ms |

The memoized cached method has the lowest **request** median in all seven settings. Target replay wins **full lifecycle** cost in two of seven settings, the two shared-feature four-dimensional rows shown above, but has no universal advantage. Lifecycle includes original full-data learning, statistic preparation, and every request. These are descriptive synthetic rational timings, not statistically established or practical-scale speedups. Three seeds generate independent shared-feature logs, whereas the switch-fixture seeds merely shuffle deletion order; timing repeats reuse datasets. The archived report marks the working tree as **not clean at launch**, so cite its recorded source hash and manifest when reproducing it.

The comparisons still omit an incremental inverse/factorization baseline, measured cache memory, large-dimensional floating-point conditioning, and neural or GPU experiments. Within this checkpoint all methods use dense exact rational solves; selected operation counters omit copy/allocation and rational-number bit complexity. The safe conclusion is that target-switch replay is **correct**, but ordinary cached reconstruction and memoization are strong competitors that limit the present efficiency/novelty claim.

### Checkpoint 05: give both methods the same optimization

Checkpoint 04's replay path solved the matrix every iteration while the memoized cached path reused evaluations of repeated policies. That difference could hide the value of target corrections. `LSPIReplayMemoizedState` now uses the same evaluation cache as cached reconstruction: cache entries live for one request, action scores and every trace entry remain, and the cache resets after deletion.

The replay method's mutable assembly matrix stays associated with its last **actually assembled** policy. A cache hit returns an immutable saved result without moving either that matrix or its associated policy. Any subsequent correction is computed from that matching pair. This distinction avoids treating a cached trace entry as if the underlying mutable matrix had also been updated. See [the contract](../audits/lspi_matched_reuse.md).

The new comparison has **80 passing tests, 63 runs, 162 requests and 810 exact trace matches**. Both memoized methods perform equal solve and action-score counts in every request. Replay now has the lowest request and lifecycle medians in five of seven settings. For 128 transitions and eight features, request medians are 6.468 ms for memoized replay and 7.766 ms for memoized cached reconstruction; full lifecycles are 50.185 and 58.306 ms. For the 640-transition switch fixture, replay loses: 6.320 versus 5.005 ms per request and 52.715 versus 45.685 ms per lifecycle. [The report](../checkpoints/05_lspi_matched_reuse.md) includes all settings and provenance.

The shared settings use two solves for both methods, with 3–4 replay corrections versus 12 reconstruction terms. This explains a reduction in one component of work; it does not prove that component causes the entire measured latency difference. Preparation, zero-group handling and other overhead still differ. The larger switch cases lose despite fewer corrections. These are descriptive rational synthetic results, not a novelty, significance or neural result. Compare methods within this run; absolute milliseconds from different sessions cannot establish an improvement.

## 8. How to reproduce and inspect the code

Run these from the repository root, C:\Users\adity\Research, with Python 3.10 or newer. The current package uses only the standard library. The commands below are documented in the [README](../README.md).

~~~powershell
python -m unittest discover -s tests -v
python scripts/run_tabular_demo.py
python scripts/run_lspi_demo.py --output results/lspi_demo.json
python scripts/run_global_checkpoint.py --output results/global_checkpoint.json.gz
python scripts/summarize_global_checkpoint.py results/global_checkpoint.json.gz
python scripts/run_lspi_checkpoint.py --output results/lspi_checkpoint.json.gz
python scripts/summarize_lspi_checkpoint.py results/lspi_checkpoint.json.gz
python scripts/run_lspi_checkpoint.py --matched-reuse --output results/lspi_matched_checkpoint.json.gz
python scripts/summarize_lspi_checkpoint.py results/lspi_matched_checkpoint.json.gz
~~~

The full global and LSPI benchmarks can take time; begin with the two demos and tests. The archived [global summary](../output/benchmarks/global_checkpoint_summary.md), [global raw report](../output/benchmarks/global_checkpoint.json.gz), and [LSPI summary](../output/benchmarks/lspi_checkpoint_summary.md) let you inspect completed checkpoints without rerunning them. The project records source revision/hashes and dataset/deletion sequences in benchmark artifacts. If current code has changed since those archives, do not present a new run as the same frozen checkpoint without checking its manifest.

Read code in this order: [data.py](../src/rl_unlearning/data.py) and [tabular.py](../src/rl_unlearning/tabular.py) for the learner's input/model; [reference.py](../src/rl_unlearning/reference.py) for the independent tiny exact oracle; [local_policy_repair.py](../src/rl_unlearning/local_policy_repair.py) and [global_planner.py](../src/rl_unlearning/global_planner.py) for the strong tabular comparison; [lspi_control.py](../src/rl_unlearning/lspi_control.py) for target-switch replay; and [lspi_baselines.py](../src/rl_unlearning/lspi_baselines.py) for cached reconstruction and memoization. The [tests](../tests/) are best read next to each module. Start with [test_lspi_control.py](../tests/test_lspi_control.py) after the two-state demo, then inspect [test_lspi_baselines.py](../tests/test_lspi_baselines.py) and [test_lspi_memo.py](../tests/test_lspi_memo.py).

## 9. What is still research, and what to study next

Both the first cost comparison and **matched policy-evaluation reuse comparison are complete**. The latter finds a conditional replay benefit but two losing regimes. The next comparison should add a genuine incremental inverse/refactorization baseline, measured cache memory, larger feature dimensions, conditioning/failure sweeps, and floating-point behavior. Charge original preparation, deleted-sample processing, matrix copies, all policy scans, group checks, solves, memory, and failure handling. A win only over naive per-transition scratch is insufficient. The [progress log](../PROGRESS.md) and [Checkpoint 05 summary](../output/benchmarks/lspi_matched_checkpoint_summary.md) are the evidence for the current state.

GPU experiments remain conditional. A later IQL/other deep offline-RL study would need multiple independent retained-data retrains, strong fine-tuning and TrajDeleter baselines, matched deletion/non-member controls, retained return, policy distance, several audits, and GPU-hour accounting. The tabular and fixed-feature exactness arguments do not transfer automatically to a learned neural encoder.

Suggested study sequence:

1. Read Sections 1–3 here and reproduce the tabular demo. Be able to explain \(A(D_R)\) in one sentence.
2. Work through the Bellman residual/action-gap argument in Section 4; identify its global scan.
3. Draw a changed state, its predecessors, and the unaffected outside block; then read the [block proof](../audits/local_repair_proof.md).
4. Read [Checkpoint 01](../checkpoints/01_exact_local_repair.md), then [02](../checkpoints/02_scc_controls.md), then [03](../checkpoints/03_global_controls_and_lspi.md). Observe how the comparator changes the story.
5. Solve the two-state [LSPI example](../checkpoints/lspi_control_example.md) by hand, then run the demo and read the [LSPI contract](../audits/lspi_control_contract.md).
6. Read the [direct RL literature](../literature_review.md#2-direct-work-on-rl-experience-deletion), especially TrajDeleter, TOUR, and Exact Unlearning in RL; only then use the LLM papers for adjacent ideas.

Three checks of understanding: (i) Why can a retained-only policy still visit a forgotten state? (ii) Why does a changed row sometimes alter a predecessor's action? (iii) Why does LSPI need to update targets of retained transitions after policy improvement? If these are clear, you have the core of the project.
