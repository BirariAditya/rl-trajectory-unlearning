# Why deleting samples is not enough for LSPI control

Validated 7 October 2026. This is a small correctness example, not a new theoretical result or a speed benchmark. Reproduce it with `python scripts/run_lspi_demo.py`; see [the contract](../audits/lspi_control_contract.md).

## Setup

Two states x and z each have actions a and b, with a first in the tie order. Use four one-hot features in the order `(x,a), (x,b), (z,a), (z,b)`. Discount and regularization are both 1/2. Regularization is added to sample sums. Every run begins at policy `(a,a)` and performs three policy-evaluation/improvement iterations.

The retained log contains:

- A trajectory `(x,a,0,z)` followed by terminal `(z,a,0)`.
- A terminal transition `(x,b,1/16)`.
- A terminal transition `(z,b,1/4)`.

The forgotten trajectory is terminal `(z,a,1)`. These are complete finite episodes. The initial policy, feature map, discount, action order and iteration count are fixed independently of the data.

## Exact result

| Calculation | Weights in the four feature coordinates | Greedy policy `(x,z)` |
| --- | --- | --- |
| Full-data evaluation at `(a,a)` | `(2/15, 1/24, 2/5, 1/6)` | `(a,a)` |
| Retained evaluation at the old policy (one-shot downdate) | `(0, 1/24, 0, 1/6)` | `(b,b)` |
| Retained evaluation after updating successor targets | `(1/18, 1/24, 0, 1/6)` | `(a,b)` |
| Third retained iteration | Same as the preceding row | `(a,b)` |

The removed sample is at z, but the correct final action at x depends on the retained transition from x to z. After the first improvement, z selects b. Evaluating that new policy changes the target for the **retained** `(x,a)` sample. Its value becomes 1/18, which exceeds the competing 1/24, so x switches back to a. Stopping after the one-shot downdate leaves x wrong relative to the declared three-iteration learner.

## What the implementation checks

The scratch path rebuilds each matrix directly from retained transitions. The replay path maintains the matrix at the fixed initial policy and retained successor aggregates `G(z)`. When a successor policy changes, it applies:

```text
Delta A = -gamma * G(z) * [phi(z,new_action) - phi(z,old_action)]^T.
```

Here `G(z)=(1,0,0,0)`, so one nonzero outer-product correction moves the first row's `-1/2` target entry from the `(z,a)` column to `(z,b)`. The replay reads one deleted sample and then applies this correction; all three matrices, right-hand sides, weight vectors, and greedy policies match independently assembled scratch training exactly.

Work counters are not runtime evidence. Preparation still reads the full log, requests copy statistics, policy improvement scans the declared state/action domain, and every iteration performs a fresh dense rational solve. Original LSPI already discusses policy-dependent successor features and incremental matrix updates. This example establishes a necessary control-replay behavior for our implementation; it does not establish novelty or a complexity advantage.

## Tests beyond the example

The suite also covers shared (non-one-hot) features, a true policy cycle, exact ties, finite-budget behavior, terminal masking, all-data deletion, sequential versus union deletion, and singular matrices. Random shared-feature tests compare every iterate on 25 datasets and three sequential requests per dataset. Singularity means a failed request whose data subtraction is not committed; a finite policy cycle is recorded and the declared iteration schedule still finishes. Equality concerns the learner's trace/policy/weights, not cost metadata, privacy, or an environment-optimal policy.
