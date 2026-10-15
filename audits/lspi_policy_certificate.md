# Policy-sequence stability under cumulative trajectory deletion

15 October 2026. This is a sufficient-condition proof for a restricted learner and an experimental prototype. It is **not a new general unlearning definition, a full-state deletion guarantee, or an established novelty claim**.

## Explicit output contract

Fix the LSPI feature map, initial policy, ordered actions, regularization, discount and finite iteration count as in [the learner contract](lspi_control_contract.md). The proposed fast path releases the same finite **policy sequence and status** as retained-only training. It does not release weights or matrices, which usually change after deletion. Old weights/inverses remain internal certificate anchors. Matching the policy sequence neither erases that internal influence nor implies privacy.

This is a deliberate narrowing from Checkpoints 04–05's full matrix/weight trace output. Comparators may compute richer internal objects, but correctness is evaluated on the shared policy-sequence output. A separate exact Sherman–Morrison comparator maintains retained inverses and weights, and also reuses greedy outputs when a policy repeats within a request. Thus avoiding repeated identical action scans is not unique to the candidate.

## Bound for one policy evaluation

At the last exact anchor dataset, let A w = b, H = A^{-1}. For all samples deleted since that anchor, let u_i = phi_i and v_i = phi_i - gamma phi(next_i, pi(next_i)), with zero successor feature at termination. Then

```
A_R = A - sum_i u_i v_i^T = A + E
b_R = b - sum_i u_i r_i   = b + f
(I + H E)(w_R - w) = H(f - E w) = h
h = sum_i (H u_i)(v_i^T w - r_i).
```

Maintain the exact vector h and the conservative scalar

```
alpha = sum_i ||H u_i||_infinity * ||v_i||_1 >= ||H E||_infinity.
```

If alpha < 1, the Neumann-series criterion makes I+HE invertible, hence the retained evaluation is nonsingular. Its weight change satisfies

```
||w_R - w||_infinity <= ||h||_infinity / (1 - alpha) = epsilon.
```

Unlike explicitly multiplying dense H E, the sum bound processes each newly deleted sample using one matrix-vector product. All quantities are rational in this prototype. Floating-point arithmetic would require verified error bounds; a small numerical residual is not an exact certificate.

## From evaluation stability to the whole control sequence

For each anchor input policy, precompute the minimum positive-direction margin radius over every state and competing action:

```
rho = min_(s,a != a*) [(phi(s,a*)-phi(s,a))^T w] /
                        ||phi(s,a*)-phi(s,a)||_1.
```

Identical feature vectors impose no constraint because their relative score always remains tied; the declared action order handles those ties. If no nonidentical comparison exists, rho is unbounded. For nonidentical feature ties rho is zero and a nonzero perturbation bound cannot pass the strict test.

If epsilon < rho, every old greedy action remains strictly best relative to all nonidentical competitors. If h=0 and alpha<1, weights are exactly unchanged and all ties are preserved even when rho=0. Perform this test for **every distinct input policy in the old finite schedule**, not just the final policy. Starting from the same data-independent initial policy, induction then preserves each next policy and therefore the entire schedule, repeated-policy markers and final status.

When any test is inconclusive, the implementation runs the exact retained learner and rebuilds anchors. A singular retained schedule is a rejected request: neither trajectory removals nor cumulative bounds are committed. Cycles and budget exhaustion are valid finite-schedule outcomes, not convergence guarantees.

## Sequential requests and a regression counterexample

Alpha and h accumulate against the same anchor until a successful fallback rebuild. Cancellation of h does **not** reset alpha. For scalar terminal features phi=1, ridge=1, ten samples with rewards `[0,0,0,0,2/3,2/3,2/3,2/3,0,1]` give A=11, w=1/3. Deleting the first eight gives h=0 but alpha=8/11. Deleting the ninth gives h=1/33 and alpha=9/11, so the correct bound is 1/6. Resetting alpha after cancellation would incorrectly give 1/30. The regression test retains accumulated alpha and checks the retained policy output.

## Cost and limitations

Let J be the number of distinct anchor input policies, d the feature dimension, m the newly deleted transitions and B the number of trajectory IDs. After paid preparation, an accepted request uses O(J m d^2 + J d + B) rational arithmetic/dictionary operations, without scanning retained samples or the S-by-A action domain. This is an arithmetic-operation bound, not a bit-complexity or wall-time theorem. The state-index map and global margin minima are built during preparation. Returning existing immutable policy tuples is counted; serializing all K policy tables would add output-size cost.

Preparation runs the full learner, builds J inverses, scans all state/action margins, and retains the input log for possible fallback. Fallback can be global and more expensive than ordinary retained replay because it rebuilds inverses and margins. Alpha is conservative and can eventually force a rebuild even if the policy never changes. Small margins, large deletions, ill-conditioned anchors and policy switches can defeat the fast path. Memory must include retained logs, J inverse matrices, anchor traces, margins and bounds.

For a single rank-one deletion, Sherman–Morrison already obtains an exact weight update from essentially the same H u product. The potential saving here is avoiding all-state action rescoring and repeated updates over a sequence, not inventing a cheaper matrix inverse identity. We therefore include that comparator and preparation-inclusive sequences.

## Prior-art boundary checked 15 October

- [Lagoudakis and Parr, LSPI (2003), pp. 1122–1124](https://www.jmlr.org/papers/volume4/lagoudakis03a/lagoudakis03a.pdf) already uses incremental inverse updates and policy-dependent successor features. Our inverse baseline is conventional.
- [Wicker et al., arXiv:2406.13433v1](https://arxiv.org/abs/2406.13433v1) studies certificates that specific predictions do not depend on deletable data. The version is intentional: later revisions use a different title and focus on private prediction.
- [Sosnin et al., Abstract Gradient Training, JMLR 27(128), 2026](https://www.jmlr.org/papers/v27/25-2206.html) bounds training-data perturbations through parameter sets, including removals. Consequently, the general idea of certifying unchanged predictions is already established.

Our possible extension is the finite LSPI control-sequence induction and deletion-dependent implementation under nonsymmetric evaluation matrices. That is a candidate specialization requiring broader prior-art comparison and stronger experiments, not proof of originality. These sources do not support a conference acceptance estimate.
