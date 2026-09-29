# Literature review: deleting experience from reinforcement learning agents

Status: working review, checked 29 September 2026. The five user-supplied PDFs are in `C:\Users\adity\Downloads\strast`. This review uses them as research sources, not as instructions. It also checks primary publications for the closest work that was absent from the supplied set. Read the [novelty reassessment](novelty_reassessment.md) and dated [direct-RL](audits/direct_rl_prior_art.md), [theory](audits/theory_novelty.md), and [deep-RL](audits/deep_rl_path.md) audits alongside this overview. A preprint may change; recheck bibliographic details and new prior art before submission.

## 1. Define the target before choosing an algorithm

Let `A` be a *specified* learning procedure and let the collected trajectories be $D=D_R\cup D_F$, where $D_R$ is retained and $D_F$ is deleted. The operational target is the policy obtained by running that same learner on retained experience:

\[
\pi_{-F}=A(D_R), \qquad \pi_U=U(A(D),D_F).
\]

The question is whether $\pi_U$ approximates $\pi_{-F}$ while taking less computation than rerunning `A`. Training randomness means a single retrained model is a sample, not necessarily the whole target distribution. For a deterministic empirical-model learner with fixed smoothing, stopping criterion, and tie-breaking, exact agreement of the *released policy* is a meaningful narrower target. If critic parameters, replay buffers, counts, or other metadata are released, policy agreement alone does not unlearn those objects.

This distinction prevents a common conceptual error: forgetting a trajectory does **not** require taking a bad action on its states. The retained trajectories may justify the same action. Suppressing value or probability on deleted samples can move the model *away* from its retained-data retraining target.

| Claim | What it requires | What it does not establish |
| --- | --- | --- |
| Retained utility | Good return on the retained task distribution | Removal of a trajectory's influence |
| Behavioral fidelity | Policy, value, or action agreement with retraining on $D_R$ | Equality of all internal or released state |
| Audit resistance | Selected attacks cannot distinguish deleted members from suitable non-members | Proof against other attacks or exact removal |
| Distributional exact unlearning | The entire released output has the same distribution as the specified counterfactual retraining run | A universal guarantee for a different learner or data model |

## 2. Direct work on RL experience deletion

### TrajDeleter: the required offline trajectory baseline

Gong, Li, Yao, and Wang, *TrajDeleter: Enabling Trajectory Forgetting in Offline Reinforcement Learning Agents*, NDSS 2025 ([paper](https://www.ndss-symposium.org/wp-content/uploads/2025-126-paper.pdf), [project/code](https://github.com/2019ChenGong/TrajDeleter)).

**Setting and method.** An offline RL agent has already trained on a fixed trajectory dataset. TrajDeleter has a forgetting phase that lowers learned values on states of the requested trajectories while encouraging high values elsewhere, followed by convergence training on retained trajectories. It introduces TrajAuditor, an audit using Q-value vectors, shadow agents, perturbations, and a statistical outlier test.

**Evidence.** The NDSS paper reports results for six offline RL algorithms on Hopper, HalfCheetah, and Walker2D. Its abstract reports about **1.5% of scratch retraining time** and **94.8% of targeted trajectories** judged forgotten by its audit, while retaining task return in those experiments. These are experiment-specific measurements under its audit and hardware, not a worst-case complexity result or exact deletion certificate. The paper describes its method as approximate unlearning.

**Limitation relative to our question.** Its value-suppression objective is not the same as policy matching to (A(D_R)). A state shared by deleted and retained trajectories may still deserve high value after retraining. The audit is useful evidence but depends on its attack family. This is the nearest algorithmic competitor for any deep offline RL experiment; citing it only as background would leave an inadequate comparison.

### TOUR: the required evaluation comparator

Pan et al., *TOUR: A Trajectory-Level Unlearning Benchmark for Offline Reinforcement Learning*, arXiv:2607.21111, July 2026 ([paper](https://arxiv.org/abs/2607.21111), [full HTML](https://arxiv.org/html/2607.21111)). This is a preprint as of this review.

**Setting and method.** TOUR is primarily a benchmark, not a new deletion algorithm. It partitions offline trajectories into forget and retain sets, builds non-members matched on observable trajectory features, trains retained-data reference policies, and evaluates several attacks alongside retained return. Its central experiments cover D4RL HalfCheetah, Hopper, and Walker2D with medium-replay, medium, and medium-expert data. Decision Transformer is the principal policy family; MLP, LSTM, and IQL analyses have more limited, distinct roles. The AntMaze extension is explicitly exploratory.

**Evidence.** Its comparisons include no update, retraining, retained fine-tuning, gradient ascent plus refitting, and an adapted TrajDeleter. A low forget-membership score can coincide with policy collapse, can fail under another attack, or can reflect residual differences between forget and non-member data. TOUR therefore reports retained utility and several attack families; it warns against treating a single area-under-curve score as proof of removal. Its retrained policy is an operational baseline rather than a perfect zero-leakage oracle, because matched non-members can still differ in unmeasured ways.

**Limitation relative to our question.** It does not give an efficient update with a retraining-equivalence certificate. It does make the evaluation standard materially stricter. Adopt its matched non-member construction and multi-attack logic when moving beyond tabular experiments, while keeping policy distance to retraining as a separate primary endpoint.

### Exact Unlearning in Reinforcement Learning: strongest formal benchmark, different data model

Nguyen-Tang and Arora, *Exact Unlearning in Reinforcement Learning*, arXiv:2606.04182, June 2026 ([paper](https://arxiv.org/abs/2606.04182), [full HTML](https://arxiv.org/html/2606.04182)). The authors list it as an ICML 2026 Spotlight. The supplied file is `2606.04182v1.pdf`.

**Setting.** This is online, finite-horizon tabular RL with one stationary MDP, an oblivious sequence of users, and an environment oracle. Each user is modeled as a response object able to generate the trajectory induced by a policy; this is stronger access than merely possessing one logged episode. The learner's output includes the model sequence and metadata. A deleted user is replaced by a fixed dummy interaction. The exactness requirement is

\[
U(A(z_{1:T}),z_t,z')\overset{d}{=}A(z_1,\ldots,z_{t-1},z',z_{t+1},\ldots,z_T).
\]

The equality is between output **distributions**, not merely expected return, a close parameter vector, or a matching greedy action.

**Mechanism and guarantee.** The construction keeps noisy prefix statistics in a binary tree. Noise makes the online learner stable in total variation when one user changes. A maximal coupling can reuse most of the previous execution; when coupling fails, later computation is replayed with the environment oracle. For a stability parameter (\rho>0), the paper states expected unlearning work on the order of a (\rho\sqrt{\ln T}) fraction of retraining, and gives tabular regret of order

\[
\widetilde O\!\left(H^2\sqrt{SAT}+H^3S^2A+\frac{H^{5/2}S^2A}{\rho}\right),
\]

with a corresponding lower bound (\Omega(H\sqrt{SAT}+SAH/\rho)) for stable algorithms. Here (S,A,H,T) are states, actions, horizon, and episodes. (\rho) trades learning performance against the frequency of replay; it is **not** residual approximation error after a successful exact unlearning run.

**Limitation relative to our question.** An offline log generally cannot provide the counterfactual trajectory the learner would have collected after an earlier user was removed. This algorithm and theorem do not transfer unchanged to fixed offline data, a learned representation, or deep actor-critic training. Extending its stability machinery to function approximation would require a new analysis of feature/covariance statistics, policy-dependent observations, output state, and computation. A simple feature substitution would not be a supported theorem.

### Reinforcement Unlearning: environment-scale deletion

Ye et al., *Reinforcement Unlearning*, NDSS 2025 ([paper](https://www.ndss-symposium.org/wp-content/uploads/2025-80-paper.pdf), [publication page](https://www.ndss-symposium.org/ndss-paper/reinforcement-unlearning/), [code](https://github.com/cp-lab-uts/Reinforcement-Unlearning)).

**Setting and method.** The deletion unit is a whole environment or task previously encountered by an RL agent. The paper proposes decremental RL and an environment-poisoning method, plus an environment-inference evaluation. Its environment owner controls or can alter the environment in ways unavailable in a fixed offline trajectory log.

**Limit for this project.** It establishes that RL unlearning predates this project, but it is not a direct trajectory-deletion baseline. Its behavioral erasure and poisoning objective also need not match retraining on retained trajectories. Cite it for scope and history; do not claim a first RL-unlearning formulation.

### ForgetMimic: very recent adjacent motion-level work

Luan et al., *ForgetMimic: Motion Unlearning for Reinforcement Learning Humanoid Control*, arXiv:2609.28378, submitted 23 September 2026 ([paper](https://arxiv.org/abs/2609.28378)). This late preprint concerns a policy trained to imitate a collection of motions on humanoid robots. It degrades performance on designated motions while preserving the others, with experiments on Unitree G1 and H2 over 12 motions. The deletion unit is a *motion skill*, and the stated goal is targeted behavioral degradation, so it is not a direct fixed-log offline trajectory counterfactual baseline. It does narrow any sweeping claim that RL behavior unlearning is unexplored. Its timing also illustrates why the novelty search must be refreshed just before submission.

### Additional 2026 work that narrows the proposed contribution

Ye et al., *Unlearning Offline Stochastic Multi-Armed Bandits*, arXiv:2605.00638 ([paper](https://arxiv.org/abs/2605.00638), [full HTML](https://arxiv.org/html/2605.00638)), formalizes single- and multi-source deletion in offline stochastic bandits under fixed-sample and distributional data models. Its Gaussian and rollback methods are chosen adaptively, with utility guarantees and lower bounds. Bandits have no transition-driven Bellman propagation, so this does not solve fixed-log MDP trajectory deletion; it does preempt a generic claim that adaptive method selection or rollback is new in sequential unlearning.

Jiang, Zhu, and Kong, *Safe-RULE: Safe Reinforcement UnLEarning*, arXiv:2606.09559 ([paper](https://arxiv.org/abs/2606.09559)), targets poisoned data in **offline safe RL**, explicitly considering both task performance and safety constraints without returning to the training environment. Its scope is closer than the LLM RULE paper despite the similar name. It should be included in a safety-related comparison if the research pivots toward constrained control; its abstract alone does not establish retained-retraining equivalence.

*Unlearning Diffusion Policies with Relative Fisher Forgetting* ([OpenReview manuscript](https://openreview.net/pdf?id=TidLO0qdp0), ICLR 2026 submission status not independently verified) combines relative-Fisher-scaled actor updates, critic Q suppression, and retained-data regularization for diffusion-based offline RL. Its trajectory/behavior setting makes it a direct deep-policy neighbor. It blocks a generic novelty claim for Fisher-weighted actor/critic forgetting; compare it only in an architecture-matched diffusion-policy experiment, not by juxtaposing published scores from different learner families.

Wang et al., *UBA-ORL: Unlearning-Activated Backdoor Attacks on Offline Reinforcement Learning*, arXiv:2609.22711 ([paper](https://arxiv.org/abs/2609.22711)), describes a deletion request that removes benign camouflage trajectories and reveals a malicious pattern in **retained** trajectories. The arXiv record says it was accepted at IEEE ICDM 2026. This is an adversarial deletion stratum, not a standard unlearning baseline. It demonstrates that fidelity to retained-only retraining and safe deployment can diverge: the correct counterfactual policy may still learn the retained backdoor.

The official [NeurIPS 2026 downloads listing](https://neurips.cc/Downloads/2026) includes the title *CURe: Conservative Unlearning with Soft-Gating Regularization for Offline Reinforcement Learning*. Its linked poster/paper was inaccessible in this audit, and a primary abstract or technical text could not be verified. Treat it as a **title-only novelty risk**; retrieve the paper before claiming a deep offline-RL method is new or finalizing the baseline matrix. Do not infer its method or guarantee from the title.

## 3. The five supplied papers and what transfers

### RUM: deletion difficulty is heterogeneous

Zhao et al., *What makes unlearning hard and what to do about it*, NeurIPS 2024 ([conference paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/16e18fa3b3add076c30f2a2598f03031-Paper-Conference.pdf), [code](https://github.com/kairanzhao/RUM)); supplied as `2406.01257v2.pdf`.

This is supervised classification, primarily CIFAR, rather than RL. It studies how forget/retain embedding entanglement and memorization affect unlearning difficulty. Its Refined-Unlearning Meta-algorithm partitions a heterogeneous forget set and applies existing unlearning methods to its subsets. The transferable hypothesis is that uniformly random deletion requests obscure important failure modes. For trajectories, test state-action coverage overlap, rare rewards, predecessor reachability, and action-gap sensitivity. These are **proposed RL analogues**, not properties proved by RUM.

### RULE: learn a refusal boundary with on-policy RL

Zhang et al., *RULE: Reinforcement UnLEarning Achieves Forget-retain Pareto Optimality*, NeurIPS 2025 ([conference paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3811007a23d95e7e239b6262337b98b9-Abstract-Conference.html)); supplied as the earlier June 2025 `2506.07171v1.pdf`.

RULE first teaches natural refusals, then uses on-policy RL and a verifiable reward over forget-related prompts and synthesized boundary prompts. Its Pareto statement concerns the measured forget-versus-utility frontier in its LLM task. It is not a guarantee that the resulting LLM has the distribution of one retrained without the source documents. The transferable idea is to test nearby but permissible cases as well as exact forget cases. A refusal objective is the wrong target for trajectory deletion when retained experience supports the same action.

### PURGE: group-relative optimization for LLM content suppression

Zaradoukas, Prenkaj, and Kasneci, *Reinforcement Unlearning via Group Relative Policy Optimization*, ICLR 2026 ([conference paper](https://openreview.net/pdf?id=BjWwqPE7mk)); supplied as `ICLR-2026-reinforcement-unlearning-via-group-relative-policy-optimization-Paper-Conference.pdf`.

PURGE samples groups of LLM completions and assigns a verifiable reward for avoiding a forbidden entity. It computes within-group relative advantages, applies a clipped GRPO update, and regularizes toward a reference LLM. This is behavioral suppression evaluated with LLM unlearning metrics, including RWKU, rather than deletion of an RL agent's experience. The paper's token-use and unlearning-effectiveness figures should be quoted with their denominators; they are not wall-clock deletion guarantees or counterfactual retraining equivalence. Group-relative feedback may inspire an approximate deep-RL baseline, but it does not solve the offline trajectory problem by itself.

### Yao et al.: negative-gradient and retention baselines

Yao, Xu, and Liu, *Large Language Model Unlearning*, NeurIPS 2024 ([conference paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/be52acf6bccf4a8c0a90fe2f5cfcead3-Paper-Conference.pdf)); supplied as `NeurIPS-2024-large-language-model-unlearning-Paper-Conference.pdf`.

The paper targets harmful responses, copyright content, and hallucination in LLMs. Its method uses gradient ascent on undesirable responses, random-mismatch outputs, and a normal-data KL anchor. PKU-SafeRLHF appears in its evaluation. Its “2% of RLHF computational time” is a comparison within the paper's LLM experiments, not an RL trajectory-unlearning cost bound. It motivates a gradient-ascent-plus-retain baseline; the retained-data retraining target still decides whether such a baseline over-forgets.

## 4. Prior planning methods already cover much of the candidate mechanism

Moore and Atkeson's [Prioritized Sweeping](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa15/optreadings/MooreAtkeson-1993-prioritized-sweeping.pdf) (1993) propagates important model/value changes through predecessors. Van Seijen and Sutton's [small-backup version](https://proceedings.mlr.press/v28/vanseijen13.html) (ICML 2013) updates contributions from one successor at a time. The [PRISM incremental-policy-iteration report](https://prismmodelchecker.org/papers/RR-12-05.pdf) (2012) reuses an old policy after MDP parameters change. These are established dynamic-planning ideas. Similarly, subtracting a deleted trajectory's transition counts or reward sums is elementary for an additive empirical model. [LSPI](https://jmlr.csail.mit.edu/papers/volume4/lagoudakis03a/lagoudakis03a.pdf) (2003) already exploits incremental matrix updates, and [Designing to Forget](https://arxiv.org/abs/2603.22870) (CVPR 2026) already uses an explicitly deletion-friendly semiparametric architecture outside RL.

The candidate contribution must therefore be more specific: prove whether the deletion's changed-row support, discounted reachability, and old action margins permit **sparse verification of policy equality** to the retained-data learner; include index, certificate, fallback, and output costs. The current global Bellman-residual certificate evaluates every state-action row and therefore has at least linear-in-model work per deletion, even if its repair queue is tiny. The review supports a *possible* deletion-dependent theorem, not a verified novelty claim. A second hypothesis is exact **control** replay for frozen-feature LSPI that updates retained transition targets when the improved policy changes; a fixed-policy matrix downdate by itself is established. Search again for near-identical sensitivity and LSPI-control results before making a priority claim.

## 5. Gap and defensible research question

The literature currently separates into (i) approximate offline trajectory suppression and auditing, (ii) exact online user removal under a simulator/response oracle, and (iii) established incremental planning. The intersection worth **testing for novelty**, rather than assuming it is empty, is:

> Given a fixed offline trajectory log and a fully specified empirical-model RL learner, when can the changed-row support and action margins certify the retained-only **policy** with genuinely local verification work? Which deletions force global work, and can a method recognize those cases before spending more than retraining?

For a discounted tabular empirical MDP, let (T_R) be the retained-model Bellman optimality operator and (Q_U) the repaired action values. Its residual is (\eta=\lVert T_RQ_U-Q_U\rVert_\infty). Bellman contraction gives (\lVert Q_U-Q_R^*\rVert_\infty\leq\eta/(1-\gamma)). If the top-versus-second action gap in every state exceeds (2\eta/(1-\gamma)), deterministic greedy actions agree with those of the retained empirical model (subject to a specified tie rule). This familiar bound is a useful **behavioral oracle** for experiments. It does not certify distributional deletion, privacy, equality of a released critic, or a sublinear-work algorithm: evaluating its global residual touches the whole model. The intended novelty is a rigorously valid local alternative with an instance-dependent work theorem and hard cases, if the follow-up prior-art audit and proof support it.

Four tests can falsify an overly broad claim early:

1. A deleted trajectory at a bottleneck can change actions far upstream even if the forget set has one item.
2. A trajectory may be redundant because retained trajectories traverse the same states and justify the same behavior; value suppression may then worsen retraining fidelity.
3. A small average Bellman residual or return gap may hide one uncaught state with a changed optimal action; compute a global or provably valid bound before certifying.
4. An accurate retained-only policy can become unsafe after coverage deletion or activation of a retained backdoor; measure safety independently of counterfactual fidelity.

The strongest near-term paper should target a deletion-dependent tabular theorem first, then a fixed-feature **control** bridge, with GPU-backed deep offline RL to test external validity. In deep models, report retraining-relative evidence and multiple attacks without calling the result exact unless the full released output distribution is actually controlled. The [implementation plan](implementation_plan.md) sets explicit week-4, week-8, and week-10 pivot gates so failure of one research hypothesis does not silently become an unsupported claim.

## 6. Reading and comparison priorities

1. Read Exact Unlearning's Definition 2.1, TV-stability definition, oracle model, and theorem to understand what “exact” requires.
2. Reproduce a small TrajDeleter or equivalent value-suppression baseline and show where its target diverges from retained-data retraining.
3. Use TOUR to design matched non-members, independent retraining references, utility anchors, and several attacks for deep experiments.
4. Implement ordinary prioritized sweeping and small backups as planning baselines before claiming a new repair mechanism.
5. Use RUM's insight to stratify deletion requests by overlap and leverage; use RULE/PURGE/Yao only as carefully labeled transfer ideas for reward or optimization baselines.

## 7. Source list and evidence boundaries

The primary sources linked above support the paper descriptions and the reported empirical numbers. Local supplied versions were identified by first-page text and page counts (RUM 34 pages; RULE 24; Exact 24; PURGE 27; Yao et al. 51). The supplied RULE file is an arXiv preprint; the linked NeurIPS record is its later conference version. The exact-RL, TOUR, UBA-ORL, and ForgetMimic arXiv papers are especially recent as of 29 September 2026. Conference status for Exact is stated by its [author's publication list](https://thanhnguyentang.github.io/); the arXiv record is the stable technical source cited here. The CURe mention is deliberately title-only pending a primary abstract or paper.

No paper above proves that a generic deep offline RL policy can be efficiently made distributionally identical to retraining after arbitrary trajectory deletion. No membership attack suite can prove that either. Those are open requirements, not claims of the proposed method.
