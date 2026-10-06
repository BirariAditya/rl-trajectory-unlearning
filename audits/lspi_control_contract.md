# Fixed-feature LSPI replay validation contract

Implemented 5 October and integrated/validated 7 October 2026. This is a correctness prototype and a prospective Track B
checkpoint, not a novelty, optimal-control, or privacy-unlearning result.

## Declared learner and output

`LSPIProblem` fixes an ordered finite state/action domain, a data-independent
rational feature map, a data-independent initial policy, discount, nonnegative
regularization, and a positive integer iteration budget. Regularization is added
to unnormalized sample sums. Floats are interpreted through their decimal strings;
use `Fraction` for an exact intended rational number. A terminal `next_state=None`
has zero successor feature. Input trajectories are named immutable sample groups;
the module validates sample domains, not environment feasibility or timeout meaning.

Each scheduled evaluation solves
`A_R(pi) w = b_R`, where
`A_R(pi) = lambda I + sum_R phi_i (phi_i - gamma phi(next_i, pi(next_i)))^T`
and `b_R = sum_R phi_i r_i`, then takes exact greedy actions with ties resolved by
the declared action order. The full budget runs even after a fixed point or a
policy cycle. Thus the output is that finite learner's result, not an assertion
that LSPI converges or finds the empirical/environment-optimal policy.

The returned trace includes each input policy, exact matrix, right-hand side,
weights, and improved policy. Policy tuples follow the declared state order.
Final weights evaluate the last **input** policy; the returned final policy is
greedy with respect to those weights. They need not be a converged policy/critic
pair when the budget ends. The first repeated policy is reported; a period of one
means a fixed point, a longer period a cycle, and no repeat means budget exhaustion.

Exact Gaussian elimination uses the first nonzero pivot and rejects singular
matrices, even if a nonunique solution exists. Positive ridge does not ensure that
an off-policy nonsymmetric LSTD matrix is invertible. A singular result records
the failing matrix/iteration and has no final weights. No pseudoinverse or larger
ridge silently changes the learner. An unsuccessful deletion is atomic: retained
statistics/data stay unchanged. A reported cycle or exhausted budget is a valid
finite-schedule result and can commit a deletion.

## Two independent assembly paths

`scratch_lspi` rebuilds every matrix and RHS directly from retained transitions at
every step. `LSPIReplayState` stores the initial-policy matrix, RHS, and retained
successor aggregates `G_R(z)=sum_{i: next_i=z} phi_i`. Deletion subtracts a sample's
matrix contribution **at the fixed initial policy**, its reward contribution, and
its successor aggregate contribution. Every request then restarts that same
initial policy and replays the same finite schedule. A target change contributes
`-gamma G_R(z) [phi(z,pi_new(z))-phi(z,pi_old(z))]^T`.

Exact equality at each iteration follows by matrix-identity substitution and
induction using the common deterministic solve/tie rule. The two paths share the
schedule and solver but not matrix assembly. Tests compare every matrix, RHS,
weight, and policy, including singular and cyclic outcomes. The hand-calculated
two-state fixture additionally proves that one-shot data subtraction can return
the wrong final control action when retained policy targets change.

## Work and claim boundary

The prototype uses dense matrix copies and a fresh rational factorization each
step. It scans successor groups to detect changed targets and scores every
declared state/action during greedy improvement. Preparation stores/reads all
logs. Selected work counters expose sample reads, successor checks, target
rank-one additions, solves, and action scores; they are not a complete accounting
of rational bit complexity, memory allocation, dictionary copying, or latency.
No speedup theorem follows from a low-rank matrix difference alone.

`LSPIWork` differs between scratch and replay and is not part of the equality
claim. `samples_read` counts assembly/deletion visits, not input validation or
all preparation reads. `target_rank_one_updates` counts dense outer-product
additions (each quadratic in feature dimension), not constant-time operations.
The successor dictionary is copied in full on deletion and checked each policy
step, including entries whose aggregate has become zero. These costs remain even
when no retained transition is scanned by replay. The prototype makes no local
or sublinear overall-work claim.

Original [LSPI (2003), pp. 1122–1124](https://www.jmlr.org/papers/volume4/lagoudakis03a/lagoudakis03a.pdf)
already gives sample-incremental LSTDQ and Sherman–Morrison updates, and explicitly
explains how policies alter successor features. The identity and exact replay are
validation baselines. A publishable contribution still needs a nontrivial
deletion-dependent work result versus strong incremental LSPI, including index,
policy comparison, feature evaluation, factorization, and preparation costs.
Features/normalizers learned from deletable data are outside this contract.

Validation command:

`python -m unittest discover -s tests -p test_lspi_control.py -v`
