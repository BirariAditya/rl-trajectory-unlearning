# Efficient Trajectory Unlearning in Reinforcement Learning

## Reading synthesis, research question, candidate method, and experimental plan

Date: 20 September 2026

### The central distinction

There are two different research problems in the supplied papers:

1. **Unlearning in an RL agent:** remove the causal influence of selected users, episodes, transitions, rewards, or environments from an agent. The counterfactual target is the learner that would result if the deleted experience had never been included.
2. **Using RL to unlearn an LLM:** optimize an LLM policy so that it refuses or avoids designated content while retaining normal utility. This changes observable behavior, but does not by itself establish that the underlying training data's influence has been removed.

The proposed project is the first problem. The Exact Unlearning paper is directly relevant. RULE, PURGE, and the Yao et al. paper contribute useful objectives and evaluation cautions, but study the second problem. RUM studies image-classification unlearning and contributes a useful way to think about deletion difficulty.

### What “forgetting” should mean

Let an RL learning algorithm be \(A\), the original dataset be

\[
D = D_{\mathrm{retain}} \cup D_{\mathrm{forget}},
\]

and the original learned policy be \(\pi_D=A(D)\). The gold-standard counterfactual is

\[
\pi_{-F}=A(D_{\mathrm{retain}}).
\]

An unlearning algorithm \(U\) receives the original learner state and a deletion request and returns \(\pi_U\). Its primary goal is to make \(\pi_U\) match \(\pi_{-F}\), while costing less than running \(A(D_{\mathrm{retain}})\) from scratch.

This definition has an important consequence: an agent does not have to perform badly on a deleted trajectory. Retained data may independently support the same action. Deliberately lowering return or Q-values on deleted states can overshoot the counterfactual target.

There are several progressively stronger targets:

- **Behavioral approximation:** small action disagreement, policy KL/TV, value difference, and return gap relative to \(\pi_{-F}\).
- **Deterministic behavioral equality:** the deployed policy is exactly the policy returned by the specified retained-data learner.
- **Distributional exact unlearning:** the entire released output distribution, including any released metadata, is identical to retraining with the user replaced or omitted.
- **Empirical privacy:** attacks cannot distinguish deleted members from matched non-members. This is evidence, not a proof of exact unlearning.

These targets must not be conflated. A small Bellman error is a behavioral certificate. It is not automatically an \((\epsilon,\delta)\) or total-variation data-removal certificate.

## The five supplied papers

### 1. What Makes Unlearning Hard and What to Do About It (RUM, NeurIPS 2024)

This is a supervised image-classification paper, not an RL paper. Its key empirical claim is that unlearning becomes harder when:

- forget and retain examples are more entangled in the model's embedding space; and
- the forget examples were more memorized by the original learner.

The paper's Tug-of-War metric multiplies three retraining-relative terms: agreement with retraining on the forget set, retain set, and test set. It then proposes RUM, which partitions a heterogeneous forget request into homogeneous subsets and applies unlearning sequentially, possibly selecting a different method per subset. On CIFAR-10 and CIFAR-100, this improves the evaluated approximate methods.

Transferable lesson for RL: deletion requests should be stratified by difficulty. For trajectories, useful difficulty variables include visitation overlap with retained trajectories, reward uniqueness, successor/predecessor reachability, coverage leverage, and the action gap at affected states. The specific RUM algorithms and image metrics do not transfer directly.

### 2. RULE: Reinforcement UnLEarning Achieves Forget-retain Pareto Optimality (NeurIPS 2025)

RULE treats an LLM as a policy. It first teaches natural refusal responses, then performs on-policy RL on forget queries and semantically nearby boundary queries. Its reward encourages refusal on forget-related prompts and informative responses on permissible prompts, with regularization toward a reference policy.

The “Pareto optimality” result is an empirical frontier under the paper's selected forgetting and utility measurements. It is not a proof that model outputs match retraining without the source data. The supplied file is the June 2025 arXiv v1; a later NeurIPS 2025 version exists.

Transferable lesson for RL: explicitly optimize a forget-retain tradeoff and test nearby cases, rather than only the exact deletion examples. The refusal target itself is unsuitable for trajectory deletion because the counterfactual policy may still choose the same actions.

### 3. Exact Unlearning in Reinforcement Learning (ICML 2026 Spotlight)

This is the most directly relevant theoretical paper. It studies sequential online tabular RL. A “user” is modeled as a fixed depth-\(H\) response tree that supplies the trajectory induced by the current policy, rather than merely a stored realized trajectory.

Its exactness definition requires the distribution of the complete output—model sequence and metadata—after unlearning to equal what the learner would have produced if the target user had been replaced by a fixed dummy from the beginning:

\[
U(A(z_{1:T}),z_t,z') \overset{D}{=}
A(z_1,\ldots,z_{t-1},z',z_{t+1},\ldots,z_T).
\]

The construction adds calibrated noise to prefix statistics using a binary tree. Total-variation stability makes the original and counterfactual executions agree with high probability under maximal coupling; if coupling fails, a suffix is replayed. Here \(\rho\) controls stability and the probability/cost of replay. It does not mean the final unlearning is only approximately correct.

The paper reports expected unlearning cost roughly a \(\rho\sqrt{\log T}\) fraction of retraining, plus tree work, and tabular regret

\[
\widetilde O\!\left(H^2\sqrt{SAT}+H^3S^2A+
\frac{H^{5/2}S^2A}{\rho}\right).
\]

It also gives a lower bound

\[
\Omega\!\left(H\sqrt{SAT}+\frac{SAH}{\rho}\right).
\]

This establishes a real efficiency-utility tradeoff, although polynomial gaps in \(H\) and \(S\) remain. The approach assumes a single finite-horizon tabular MDP, an oblivious user sequence, an environment oracle capable of producing the trajectory that a replacement execution would generate, and large stored tree state. It does not directly solve fixed-log offline trajectory deletion, heterogeneous users, function approximation, or deep RL.

### 4. Reinforcement Unlearning via Group Relative Policy Optimization (PURGE, ICLR 2026)

PURGE also studies LLM behavioral unlearning. It constructs a synthetic forget corpus of descriptive entities, generates several completions per query, and assigns a binary reward:

\[
\phi(y)=\mathbf 1[y\text{ contains no forbidden entity}].
\]

Rewards are normalized within the completion group to produce relative advantages, then a clipped GRPO objective updates the LLM while a KL penalty anchors it to the reference model.

The reported “46x” figure is a reduction in forget-corpus tokens per target per epoch relative to a comparator, not a demonstrated 46x wall-clock speedup. The reported 11% forgetting figure is a change on the paper's benchmark metrics, not 89% residual-data erasure or a distributional deletion guarantee. The paper reports roughly 98% retained utility in its chosen setup. Its theory concerns suppression of forbidden-token probability and an idealized surrogate/regret comparison; it does not prove equality with retraining without the underlying source data.

Transferable lesson for RL: relative normalization may stabilize conflicting objectives, and cheap intrinsic signals can replace learned reward models. It does not provide a trajectory-deletion mechanism.

### 5. Large Language Model Unlearning (NeurIPS 2024)

Yao et al. frame harmfulness, copyright removal, and hallucination reduction as learning from negative examples. Their update combines:

- gradient ascent on the unwanted response likelihood;
- a random-mismatch loss that teaches unrelated outputs for forget prompts; and
- a KL term that keeps normal-prompt outputs near the original LLM.

They evaluate effectiveness/generalization on unwanted prompts and utility/fluency on normal prompts, including PKU-SafeRLHF. The claim that unlearning uses about 2% of RLHF compute is an experimental comparison in their LLM setup, not a general complexity result.

Transferable lesson for RL: a negative-gradient baseline and an explicit retain anchor are necessary comparisons. But forcing wrong behavior is not the counterfactual target for trajectory deletion.

## Missing direct RL literature that must be included

The direct literature is small, but no longer empty:

- **Reinforcement Unlearning (NDSS 2025)** removes entire environments using decremental RL or environment poisoning.
- **TrajDeleter (NDSS 2025)** directly addresses offline trajectory deletion. Its forgetting phase suppresses values on forgotten trajectories and its convergence phase restores behavior elsewhere. It reports about 1.5% of retraining time in its experiments and introduces TrajAuditor.
- **TOUR (arXiv, July 2026)** provides an offline trajectory-unlearning benchmark with retraining references, matched non-members, utility anchors, and multiple attacks. It shows that conclusions can change across attacks and environments; a single membership score may confuse deletion with policy collapse.
- Recent adjacent work studies demonstration unlearning, targeted behavioral regions, and diffusion-policy unlearning. These increase the burden for a novelty claim.

Therefore, “the first trajectory-unlearning algorithm” is not a defensible claim.

## Recommended research question

> When can deleting trajectories from an offline RL dataset be handled by local Bellman repair, and when must the learner fall back to broad recomputation? Can an adaptive algorithm certify retraining-equivalent policy behavior while saving computation on local deletion requests?

This question begins with an empirical tabular MDP so that the counterfactual learner and certificate are precise. Available GPUs can then support a deep offline-RL validation phase. The theoretical claim should still be established in the controlled setting; GPU experiments test whether the underlying deletion-locality idea survives function approximation.

## Candidate method: Certified Adaptive Bellman Repair (working name)

This is a candidate contribution, not a verified novel algorithm. Count subtraction, prioritized sweeping, small backups, Bellman residual bounds, and matrix downdates all have substantial prior art. The potential contribution is their deletion-specific certification, adaptive fallback, repeated-deletion analysis, and coverage/action-gap-dependent complexity.

### Training state

For every trajectory \(\tau\), store additive contributions to:

- transition counts \(N_\tau(s,a,s')\);
- visit counts \(N_\tau(s,a)\); and
- reward sums \(R_\tau(s,a)\).

Aggregate them into an empirical MDP. Specify fixed smoothing priors, behavior for empty state-action cells, discount \(\gamma<1\), planning tolerance, and deterministic tie-breaking. Those details are part of the learning algorithm being unlearned.

### Deletion procedure

1. Subtract the requested trajectories' sufficient statistics exactly.
2. Recompute only changed empirical reward and transition rows.
3. Insert affected state-action pairs into a priority queue using their Bellman residuals.
4. Apply small Bellman backups and propagate changes to **all** predecessor actions. Restricting propagation to currently greedy actions is unsound because a previously non-greedy action can become optimal after deletion.
5. Maintain or periodically compute a retained-model Bellman residual

   \[
   \eta=\|T_RQ_U-Q_U\|_\infty,
   \]

   where \(T_R\) is the optimality operator of the empirical MDP built from exactly the retained data.
6. Use contraction to obtain

   \[
   \|Q_U-Q_R^*\|_\infty\leq \frac{\eta}{1-\gamma}.
   \]

7. If every state's top action under \(Q_U\) has gap larger than

   \[
   \frac{2\eta}{1-\gamma},
   \]

   then the unique greedy action agrees with the retained model's optimal action. If some states remain uncertified, continue repair or fall back to a full warm-started planner.

The returned-policy value loss admits a standard bound such as

\[
\|V_R^*-V_R^{\pi_U}\|_infty
\leq \frac{2\eta}{1-\gamma}
\]

under the corresponding residual construction. State the precise lemma and assumptions in the paper; do not present the contraction fact itself as the novelty.

### What the certificate means

If the released output is only a deterministic policy, fixed tie-breaking and the action-gap condition can certify equality with the policy produced by the defined retained empirical-model learner. If the product releases a critic, counts, caches, optimizer state, or metadata, policy equality does not certify that those objects were unlearned. Distributional exact unlearning is also stronger: two close deterministic parameter vectors can have total-variation distance one.

### Adaptive fallback

Local repair cannot have a universal sublinear guarantee. In a dense or bottleneck MDP, one deleted transition can change values throughout the state space. The algorithm should detect this and fall back. A useful result would bound deletion work in terms of the affected predecessor subgraph, action-gap fragility, discounting, and requested tolerance, while ensuring the fallback is no worse than a warm-started full solve by more than certificate overhead.

### Fixed-feature extension

For fixed-policy evaluation with fixed features, store

\[
A=\sum_i \phi_i(\phi_i-\gamma\phi'_i)^\top+\lambda I,
\qquad
b=\sum_i \phi_i r_i.
\]

Deleting a trajectory subtracts its contributions and solves \(Aw=b\), potentially using rank downdates. This is exact for the specified regularized fixed-policy estimator if the retained matrix remains well-conditioned.

Control is harder. When the greedy policy changes, next-action features and Bellman targets of retained transitions also change. One Woodbury downdate is therefore insufficient for LSPI-style control. A defensible extension would combine action-gap certificates with selective target recomputation and refit when conditioning or policy changes invalidate the local update.

## Hypotheses that can support a paper

- H1: Local repair cost correlates with predecessor reachability and action-gap fragility, not merely forget-set size.
- H2: The certified hybrid matches the retrained empirical-model policy on most sparse/local deletion requests and falls back on dense or high-leverage requests.
- H3: It requires fewer Bellman backups and lower wall-clock time than scratch retraining while preserving retraining-relative behavior.
- H4: Random deletion requests substantially overstate efficiency; rare-coverage and bridge trajectories expose broad influence.
- H5: Repeated deletions cause certificate failures and conditioning problems earlier than single deletions, motivating explicit fallback rules.

## Experiments

### Environments

Start with small controlled MDPs:

- chain and branching-chain MDPs;
- grid worlds with local and bottleneck paths;
- FrozenLake and CliffWalking variants;
- a deliberately dense random MDP as a negative case.

Then add one fixed-feature setting using tile coding or frozen encoder features. Once the tabular method and claims work, add a deep offline-RL phase on D4RL/Minari locomotion tasks such as Hopper, HalfCheetah, and Walker2D, matching the direct trajectory-unlearning literature. AntMaze can be an optional hard coverage case.

For deep offline RL, use one or two representative algorithm families rather than many shallow runs:

- a value-based or actor-critic method such as IQL or CQL;
- a sequence-model policy such as Decision Transformer if TOUR-compatible code is used;
- TrajDeleter and GA+Refit implementations as direct baselines.

Train independent retained-data reference models for every deletion split. Those retrains are expensive but indispensable: they define the target and reveal ordinary seed-to-seed variation. Use GPU parallelism for independent seeds and deletion splits rather than only scaling model size.

### Deletion requests

Use at least:

- uniform random trajectories;
- one user/block of trajectories;
- rare-state trajectories;
- reward outliers;
- bottleneck or bridge trajectories;
- highly overlapping versus weakly overlapping state-visitation distributions;
- repeated sequential deletion requests of 1%, 5%, 10%, and 20% of data.

The overlap and leverage strata are the RL analogue of RUM's entanglement and memorization analysis.

### Baselines

1. retraining from scratch on retained data;
2. no operation;
3. retain-data fine-tuning;
4. negative gradient or gradient-ascent plus refit;
5. TrajDeleter where applicable;
6. warm-started full value iteration;
7. ordinary prioritized sweeping or small backups without the deletion certificate;
8. the certified adaptive method;
9. fixed-feature rank downdate plus full policy iteration for the extension.

### Metrics

Counterfactual fidelity:

- state-weighted action disagreement with retraining;
- policy TV/KL for stochastic policies;
- \(\|Q_U-Q_{-F}\|_\infty\) and weighted \(L_2\) error;
- value and return gap relative to retraining;
- fraction of states whose action is certified.

Retention:

- return from retained-state initial distributions;
- value error on states covered by retained data;
- performance on held-out environment rollouts.

Deletion evidence:

- several membership/auditing attacks with matched non-members, following TOUR;
- behavioral agreement with independent retrains;
- certificate pass/fail and fallback rate.

Efficiency:

- Bellman backups, affected edges visited, and full residual scans;
- wall-clock time and speedup over retraining;
- memory overhead;
- amortized cost under repeated deletions.

Report all metrics together. A membership score can improve because the policy collapsed, and retained return can remain high even when deleted membership is detectable.

### Statistical protocol

- Predefine the learner, smoothing, stopping rule, and tie-breaking.
- Use multiple dataset-generation and algorithm seeds.
- Compare every method to retraining on the exact same retained dataset.
- Report means, confidence intervals, and worst-case deletion strata.
- Measure certificate computation; a global residual scan is not free.
- Include explicit failure examples where repair becomes global.

## Three levels of possible contribution

### Feasible and useful

A careful benchmark and analysis showing that trajectory deletion difficulty is governed by coverage leverage, predecessor influence, and action gaps. Add a certified hybrid that safely chooses local repair or full recomputation.

### Stronger

A theorem bounding backups or affected states under a sparse predecessor graph, contraction, and action-gap assumptions, plus repeated-deletion guarantees and fixed-feature experiments.

### High risk

Exact online unlearning with linear MDPs or learned representations. This requires distributional equality for the whole learner state, policy-dependent future observations, covariance sensitivity, coupling, regret, and memory analysis. The Exact Unlearning paper itself points toward feature/covariance sufficient statistics, so merely suggesting that substitution is not new.

## Paper framing

A cautious working title is:

> Certified Local Repair for Trajectory Deletion in Offline Reinforcement Learning

A defensible claim, if experiments and proofs support it, would be:

> We characterize when a trajectory deletion can be handled by local planning updates. Our adaptive algorithm removes the trajectory's contribution from an explicitly defined empirical-model learner, certifies retraining-equivalent deployed actions when action gaps permit, and otherwise falls back to broad recomputation. Its savings track deletion influence and MDP connectivity rather than forget-set size alone.

Avoid claims of data-removal certification unless the theorem provides statistical indistinguishability. Avoid “first trajectory unlearning,” “exact deep-RL unlearning,” and “sublinear deletion for all MDPs.”

## Suggested three-month schedule

Weeks 1–2: reproduce tabular retraining, no-op, fine-tuning, negative-gradient/refit, and prioritized-sweeping baselines. Implement exact per-trajectory count subtraction.

Weeks 3–4: implement the residual/action-gap certificate and adaptive fallback. Verify exact policy agreement against retained-data retraining on small MDPs.

Weeks 5–6: build deletion strata and repeated-deletion tests. Add dense and bottleneck negative cases.

Weeks 7–8: formulate and prove the narrowest correct locality/action-gap result. Count certificate and fallback costs.

Weeks 9–10: implement the fixed-feature evaluation extension and conditioning fallback.

Weeks 11–12: run deep offline-RL pilots on one locomotion task, including retained-data retrains and TOUR-style audits.

Weeks 13–16: expand successful deep experiments to two or three tasks, run seeds and ablations, create figures, and write. If the deep method does not reproduce retraining-relative behavior, report it as an empirical limit and keep the primary contribution tabular/fixed-feature.

### GPU execution priorities

1. Parallel independent retrains, because the retraining distribution and its variance are the scientific target.
2. Deletion-type sweeps: random, rare-coverage, bridge, and repeated requests.
3. Multi-attack auditing rather than a single membership score.
4. Hyperparameter tuning performed symmetrically for baselines and the proposed method.
5. Larger models only after the counterfactual and audit protocol are correct.

## Immediate reading order

1. Exact Unlearning in Reinforcement Learning: definitions, assumptions, stability/coupling mechanism, and theorem statements.
2. TrajDeleter: direct offline trajectory baseline and its target mismatch with retraining.
3. TOUR: evaluation protocol and failure of single-score auditing.
4. Prioritized Sweeping and Small Backups: established local-planning machinery.
5. RUM: deletion difficulty stratification.
6. RULE, PURGE, and Yao et al.: objective design and behavior/utility tradeoffs, with the LLM-versus-RL distinction kept explicit.

## Novelty checklist before submission

- Search for the exact combination of trajectory deletion, Bellman certificates, adaptive fallback, repeated deletions, and function approximation.
- Compare against prioritized sweeping/small backups, not only full value iteration.
- Compare against TrajDeleter and TOUR protocols.
- State whether the released object is policy-only or includes critic/metadata.
- Define exactness relative to a fully specified learner.
- Separate behavioral equality, parameter proximity, membership resistance, and distributional exactness.
- Show hard deletion requests and fallback behavior.
- Phrase novelty as provisional until a systematic literature search is complete.
