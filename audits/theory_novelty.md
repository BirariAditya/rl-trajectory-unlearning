# Theory audit: offline trajectory deletion (29 September 2026)

Scope: deterministic finite discounted empirical-MDP learner and, separately, a fixed-feature LSPI learner. This is a **proposal audit**, not a proof of novelty or a completed theorem. All the cited works below are primary papers, author manuscripts, or original technical reports. The strongest theoretical opportunity is a *decision-sensitive* deletion certificate whose actual verification cost is sublinear in the MDP graph in favorable cases. The present global residual and action-gap test is a sound baseline but is not a research contribution by itself.

## Prior art that narrows the claims

* [Moore and Atkeson, *Prioritized Sweeping* (1993)](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa15/optreadings/MooreAtkeson-1993-prioritized-sweeping.pdf) already store experiences, update a model, and propagate value changes to predecessors. It even discusses forgetting in changing environments. Calling reverse Bellman propagation after deletion novel would be untenable.
* [PRISM incremental MDP model checking, RR-12-05](https://prismmodelchecker.org/papers/RR-12-05.pdf), Section 4.2, reuses a previous optimal policy after transition changes and initializes policy iteration around changed states. Warm-starting or adaptive fallback alone is old.
* The sup-norm Bellman residual inequality `||Q-Q*||∞ ≤ ||TQ-Q||∞/(1−γ)` follows immediately from contraction. A `2ε` greedy action-gap separation is the elementary corollary. The current implementation in `src/rl_unlearning/unlearning.py` scans **all** state-action rows to compute the residual, so its certificate costs `Ω(E)` per request, where `E` is the number of stored transition edges. A paper cannot claim sublinear unlearning time from counting only local backups.
* [Lagoudakis and Parr, *Least-Squares Policy Iteration* (2003)](https://jmlr.csail.mit.edu/papers/volume4/lagoudakis03a/lagoudakis03a.pdf) already use Sherman–Morrison for incremental LSTDQ matrix inversion. A fixed-policy downdate alone is old. The proposed extension below targets changes in **the policy-dependent next-action features** across LSPI control iterations.
* [Ye et al., *Unlearning Offline Stochastic Multi-Armed Bandits* (2026)](https://arxiv.org/abs/2605.00638) already choose adaptively between Gaussian mechanisms and rollback and prove lower bounds in bandits. Do not claim that adaptive switching or rollback is new in sequential unlearning. Their target is bandit privacy/utility, not counterfactual empirical-MDP control.
* [Zheng et al., *Designing to Forget* (CVPR 2026)](https://arxiv.org/abs/2603.22870) already architect a semiparametric model for efficient explicit removal. Storing sufficient statistics or a deletion-friendly architecture is not new in general.
* MDP perturbation and sensitivity analysis are established; e.g. [Cao et al., *A unified approach to Markov decision problems and performance sensitivity analysis* (2004)](https://eecao.home.ece.ust.hk/paper/A%20unified%20approach%20to%20Markov%20decision%20problems%20senitivity%20analysis.pdf), and transition-error loss bounds of [Mastin and Jaillet (2012)](https://dspace.mit.edu/entities/publication/aa633c8d-a627-4f60-b9ff-cae4c6db3239). A generic resolvent identity or discounted occupancy interpretation is not new. I did not find, in this targeted search, the exact *deletion-specific local-work theorem* below; a full bibliography audit is still required before claiming priority.

## Candidate theorem A: certified no-change test with signed deletion influence

This is the best immediately testable tabular result. Let `M=(P,r)` be the empirical MDP from all trajectories, `M'=(P',r')` the model after deletion, `0≤γ<1`, and `π` a **verified** optimal deterministic policy for `M` with a declared tie order. Empty rows follow the same terminal convention in both models. Assume exact arithmetic for the theorem; implementation needs interval or error-certified linear solves. Let `V=V_M^π`. For each state define the policy-row deletion shock

`d_π(s) = r'(s,π(s))−r(s,π(s)) + γ[P'_{s,π(s)}−P_{s,π(s)}]V`.

Only states whose **chosen policy row** changed have nonzero `d_π`. Set

`Δ = (I−γP'_π)^{-1}d_π`  and  `V' = V+Δ`.

For any action `a`, put `e_a(s)=r'(s,a)−r(s,a)+γ[P'_{s,a}−P_{s,a}]V` and let the original advantage gap be `g_a(s)=V(s)−[r(s,a)+γP_{s,a}V]`. The retained-model gap is **exactly**

`g'_a(s)=g_a(s)+Δ(s)−γP'_{s,a}Δ−e_a(s)`.

**Statement sketch.** `π` is optimal for `M'` if and only if every `g'_a(s)≥0`; with strict inequalities for earlier actions in the fixed tie order, it is precisely the counterfactual greedy policy. Proof: the resolvent identity gives `V'=V_{M'}^π`; the inequalities say `T'V' = T'^πV'=V'`; Bellman optimality uniqueness gives `V'=V*_{M'}`. This is a *behavioral* certificate for the declared learner, not distributional exact unlearning of hidden state.

**Useful special case.** If deletion changes no chosen-policy rows, `d_π=0`, hence `Δ=0`: every unchanged action retains its old gap, and one needs to check only changed non-policy rows. This can verify a deletion in `O(k)` after preprocessing, where `k` is the changed-row support. It is especially relevant because many removed trajectories alter behavior-policy actions the deployed optimal policy never chooses.

**General computation.** With `k_π` changed policy rows, `P'_π−P_π` has rank at most `k_π`. A stored factorization or inverse of `I−γP_π` allows a Woodbury solve for `Δ`; dense preprocessing costs `O(S³)` time and `O(S²)` memory, and a naive update costs `O(S² k_π+k_π³)`. The final all-action gap scan costs `O(E)`, which often dominates. There is no sublinear claim until that scan is removed or amortized. A sparse linear solver may use less memory but gives problem-dependent complexity. Compare against both fresh value iteration and warm-started policy iteration, and count factorization/index build cost across repeated deletions.

**Failure mode / counterexample.** One deleted reward observation at a distant bottleneck may make an action switch in every predecessor state of a long chain. Signed influence may reach the whole graph. Even if only one model row changes, materializing a changed policy table is `Ω(S)`. A universal `O(k)` deletion theorem is false.

**Novelty judgment.** The equation and policy-improvement test are standard linear algebra and MDP optimality. Publishability depends on a new **data-dependent bound on the number of gaps examined / inverse work** tied to deletion support and action margins, plus evidence that the regime occurs in offline RL. Do not sell the exact identity alone as a theorem-level contribution.

### A possible truly local-work refinement to prove

Let `C` be the states containing any changed action row and `x(s)=max_a |(T'_{s,a}V*)−(T_{s,a}V*)|`, supported on `C`, with `x_max=||x||∞`. For a state `s`, let `d(s,C)` be the shortest support-graph path from `s` to `C` under **any** action of the retained model, with infinity if unreachable. Monotone Bellman comparison gives a componentwise envelope `h` solving

`h(s)=x(s)+γ max_a P'_{s,a}h`,  so  `|V*_{M'}(s)−V*_M(s)|≤h(s)`.

Because no path can encounter `C` before `d(s,C)` transitions, `h(s)≤γ^{d(s,C)} x_max/(1−γ)`. For each original nonchosen action `a`, compare its original gap with the sum of direct-row shocks and successor envelopes for `a` and `π(s)`. If the old gap exceeds this bound, `a` cannot become optimal after deletion. Under a uniform positive old gap `g_min` and bounded shock, choose radius `L=O(log(x_max/[(1−γ)g_min])/log(1/γ))`. Outside the reverse `L`-neighborhood of `C`, every old action choice remains optimal. A local algorithm may inspect only the reverse neighborhood and changed rows, **provided** it can certify a bound for unvisited states from a precomputed global margin and correctly handle new/removed transition edges.

This gives a plausible instance-dependent work statement, `O(E(B_L(C))+k)` per deletion after `O(E)` indexing, **for classifying where the policy can change**. It does not automatically compute exact new Q-values or values, which can change outside `B_L(C)` even when actions do not. It also does not suffice to solve inside the neighborhood using frozen boundary values; boundary values change. The formal target should be exact action output plus a certified value-error interval or an exact resolvent solve, with its cost included. If exact policy output is required in every state, at least `Ω(S)` is needed to emit a fresh table; an in-place policy representation with delta updates can have local request cost.

Proof tasks: (1) derive the envelope carefully for substochastic terminal rows and changed support; (2) turn per-action envelopes into a no-switch inequality without assuming the new optimal policy; (3) specify a query/index representation that avoids a hidden all-state minimum-gap scan; (4) count the cost of the influence bound itself; (5) produce dense-graph and zero-gap counterexamples. Risk: this may be viewed as a straightforward robust-MDP sensitivity application unless the deletion-specific locality law and data structure are genuinely sharp.

## Candidate theorem B: exact counterfactual **control** for frozen-feature LSPI via low-rank policy-target updates

This is a more substantial change of setting than fixed-policy LSTD downdating, though it carries significant prior-art and numerical risks. Freeze a feature map `φ(s,a)∈R^d`, finite evaluation states, a data-independent initial policy `π_0`, regularization `λ>0`, fixed iteration count `J` or a deterministic stopping rule, and deterministic tie-breaking. For a policy `π`, LSPI's retained-data normal equations are

`A_R(π)=λI+Σ_{i∈R} φ_i[φ_i−γφ(s'_i,π(s'_i))]^T`,  `b_R=Σ_{i∈R}φ_i r_i`.

For any baseline full-data policy `π̄`, let `G_R(z)=Σ_{i∈R:s'_i=z}φ_i` and let `F` be deleted samples. Then the **exact identity** is

`A_R(π)=A_D(π̄)−Σ_{i∈F}φ_i[φ_i−γφ(s'_i,π̄(s'_i))]^T−γΣ_{z:π(z)≠π̄(z)}G_R(z)[φ(z,π(z))−φ(z,π̄(z))]^T`.

`b_R=b_D−Σ_{i∈F}φ_i r_i`. Thus the correction rank is at most `|F|+m`, where `m` is the number of successor states at which `π` differs from `π̄`. The last term is essential: deleting a few samples can change `π(z)` and thereby change the target of **many retained transitions** ending at `z`. A naive matrix downdate misses this.

**Statement sketch.** If every retained normal matrix encountered is invertible and the low-rank correction solve is computed exactly, replaying the same deterministic LSPI policy-improvement sequence with the identity above yields the same `π_j,w_j` as LSPI trained from scratch on `D_R`, by induction on `j`. Hence the final **declared LSPI output** matches retained-data retraining exactly in exact arithmetic, even when policies switch. Cache `A_D(π̄_j)` and its factorization for full-data iterates or build a baseline factorization for each needed reference policy. Each step costs roughly `O(d²(|F|+m_j)+(|F|+m_j)^3+SAd)` plus aggregate maintenance, versus `O(Nd²+Jd³)` for naive recomputation, conditional on low rank, feature access, and efficient baseline construction. State **all** training/index/caching cost and memory `O(Jd²+Sd)`; if `m_j` is large or factors are ill-conditioned, refactor from retained samples.

**Proof tasks.** Confirm signs and terminal-state conventions; use generalized Woodbury for nonsymmetric LSTD matrices; establish invertibility from an explicit condition (regularization alone does *not* guarantee invertibility for an arbitrary nonsymmetric `A`), or use a stable solver and fallback. Bound numerical error so it cannot alter an action across a small margin. Prove induction under a fixed iteration count first, then analyze policy convergence separately; approximate PI can cycle, so avoid an unqualified convergence claim. Compare to original LSPI's already-known Sherman–Morrison implementation.

**Counterexample.** A changed decision at a hub state can alter the next-action feature for nearly every retained transition (`m=1` but `G_R(z)` is dense in samples). The aggregate update remains rank one, yet constructing `G_R(z)` without a maintained index costs `Ω(N)`. With many changed successor states, rank and solve cost approach full recomputation. Frozen features cannot model representation changes in a deep learner; unlearning a trainable encoder is outside this theorem.

**Novelty judgment.** The algebraic identity is a promising exact **control** bridge because it handles target changes, whereas the current plan covers only fixed-policy evaluation. It is not safe to claim originality from the formula alone; LSPI and Woodbury are old. A stronger paper would prove a switch-count or margin condition controlling `m_j`, perhaps paired with a sharp counterexample and a robust implementation on frozen GPU features. It provides the clearest route beyond purely tabular results within a few months.

## Decision and immediate experiments

1. Test theorem A's special case now: construct deletions that change only non-policy rows. The old optimal policy should certify using only changed rows. Then construct a changed-policy-row chain and measure how many gaps require examination. Compare signed influence, distance envelopes, the current global residual test, ordinary prioritized sweeping, and warm-started policy iteration with **total** wall time.
2. Build an independent LSPI reference on fixed random features and a frozen small neural encoder. Verify candidate B's normal-matrix identity after deletion **and after a policy switch** to machine precision. The policy-switch case is the essential test.
3. Before naming either as the main theorem, do a dedicated citation search on local MDP sensitivity and low-rank LSTD/LSPI control updates. If a near-identical result exists, shift the contribution toward sharp deletion-difficulty characterization and controlled deep experiments.
4. A submission-worthy claim would read: *under explicit margin and graph/feature conditions, the method returns the exact policy of a specified retained-data learner with per-request cost governed by the deletion influence region or LSPI switch rank; outside those conditions it detects failure and fully recomputes.* It must demonstrate regimes where this beats warm-started retraining after including indexing, verification, and fallback.
