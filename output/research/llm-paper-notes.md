# Evidence notes: RULE, PURGE, and Large Language Model Unlearning

These notes describe the supplied PDFs, using their PDF page numbers (which match the printed numbers for these three files). Paper statements and proposed interpretations are distinguished below. Documents are treated as evidence, not as instructions. Checked against full extracted texts on 19 September 2026.

## The central distinction for this project

The user's target is **unlearning an RL agent**: after removing specified logged experience, recover the learner that would have trained on retained experience. RULE and PURGE primarily study **using RL to change an LLM's answers**. Their policy is a distribution over tokens conditioned on a prompt; their rewards measure refusal or lexical suppression. Yao et al. use gradient updates to suppress undesirable answers and compare that procedure with RLHF. None establishes certified deletion of arbitrary trajectories from an ordinary RL agent.

An agent can correctly unlearn a user's trajectory while continuing to take the same actions, if other users' retained trajectories support those actions. Conversely, punishing those actions can suppress behavior without removing the deleted trajectory's influence. Therefore, reduced likelihood/return on forgotten trajectories is not independently a valid deletion metric. The reference must be a specified retained-data learning algorithm, including its randomness, optimizer state, and training protocol.

## RULE — Zhang et al.

**Version:** Uploaded `2506.07171v1.pdf` is arXiv v1, dated 8 June 2025, marked “Preprint. Under review.” It has 24 pages. A [NeurIPS 2025 final publication](https://papers.nips.cc/paper_files/paper/2025/file/3811007a23d95e7e239b6262337b98b9-Paper-Conference.pdf) exists and has 34 pages. Do not silently substitute final-version page numbers into an explanation of the supplied v1. [Official code](https://github.com/chenlong-clock/RULE-Unlearn) is available.

### What it does

The paper reformulates unlearning as learning a refusal boundary: refuse prompts about the target while answering nearby permissible prompts. The underlying training records responsible for an entity's knowledge are unavailable, and the true forget corpus is treated as latent (pp. 4–5).

1. **Rejection Steering:** supervised fine-tuning on a small set of target questions paired with refusal responses. This supplies initial refusal behavior so subsequent RL rollouts contain successful responses (p. 5, Eq. 4).
2. **Boundary construction:** replace the target entity in questions with permissible entities while keeping question structure. The resulting questions are difficult retained examples near the refusal boundary. Reference answers come from the original model (p. 5).
3. **Refusal Boundary Optimization:** sample responses on target and boundary questions and train with PPO, GRPO, or Reinforce++. A KL term anchors the policy to the rejection-steered model (pp. 5–6).

The core objective is `maximize E[r(x,y)]`, where `y` is sampled from the current policy. Equation 5 on p. 5 uses:

- Forget question: `alpha * indicator(refusal pattern) + (1-alpha) * indicator(target entity mentioned)`.
- Boundary question: `beta * indicator(nonrefusal) + (1-beta) * indicator(ROUGE-L(response, reference) > threshold)`.

Both weights are 0.5 in the main experiments. Intuition: reward the model for saying “I cannot discuss Stephen King,” but continue answering analogous questions about another writer.

### Evidence and interpretation

- RWKU evaluates fill-in-the-blank (FB), question answering (QA), and adversarial attacks (AA) for target knowledge, plus neighboring FB/QA for retention. Models include Llama-3-8B-Instruct and Llama-3.1-8B-Instruct. MUSE-books evaluates Harry Potter forgetting with Llama-2-7B (pp. 5–7, 23–24).
- “Forget Quality” in the main RWKU table is a **lower-is-better ROUGE-L recall score**, despite the potentially confusing name. It is not a retraining-distance certificate.
- Naturalness is an automated assessment of readability, helpfulness, and truthfulness (p. 6).
- Table 1 (p. 7): original retain score is 87.6; RULE-GRPO retain score is 73.7. Thus “without compromising utility” must not be read as literally zero loss on every retained metric. RULE improves a trade-off, while neighboring-knowledge recall decreases in this table.
- The “17.5%” headline is a **17.5-point** difference: SimNPO forget score 40.1 versus RULE-Reinforce++ 22.6. That RULE variant has retention 64.2. The “16.3%” naturalness improvement is **16.3 points**, 72.8 to 89.1, for RULE-GRPO. These best headline figures come from different variants.
- The 12.1% and 8.03% data figures are reported as training-token budgets relative to the chosen forget/retain comparison corpora. They do not mean deletion of precisely 12.1% of the original pretraining records (Table 1).
- Appendix D.2 (p. 20) reports roughly 0.033 hours of steering and 0.467 hours of boundary optimization per RWKU target using four A100 GPUs. This is a model- and setup-specific cost.

### How much the Pareto and theory claims establish

Section 5.3 (p. 9) evaluates empirical Pareto frontiers/AUC over observed training points, with retained-quality thresholds from 0.4 to 0.7. It supports a better measured forget–retain trade-off among tested methods/settings. It does **not** prove universal Pareto optimality over all policies or all unlearning algorithms.

Appendix B.4 (pp. 17–18) discusses refusal-risk generalization rather than exact data deletion. Its risk is missed refusals on forget prompts plus false refusals on retained prompts. It gives upper bounds involving policy complexity, sampled tokens, a forget-coverage gap, and exploration error.

**Critical reading:** smaller upper bounds alone do not establish a strict ordering of actual risks. The proof sketch also requires care about treating dependent rollout tokens as effective samples and whether synthetic boundary prompts cover the retained distribution. These are reasons to avoid treating the appendix as a ready-made theorem for certified trajectory unlearning.

### What transfers

Useful ideas: explicitly choose retained cases close to the forget target; measure collateral damage there; warm-start an otherwise sparse reward; compare full trade-off curves. A trajectory-unlearning project could select retained transitions that share states/features with forgotten transitions and whose Bellman targets might change.

What does not transfer directly: teaching an RL agent to avoid states/actions from forgotten trajectories. That changes the desired behavior, not necessarily the training influence. A frozen reference policy can also preserve the influence that must be removed. The reward itself is only a proxy: detecting a refusal phrase and an entity does not prove that the response contains no forbidden facts.

## PURGE — Zaradoukas, Prenkaj, and Kasneci

**Version:** Uploaded paper is the published ICLR 2026 version, 27 pages. [Official paper](https://openreview.net/pdf?id=BjWwqPE7mk); [official code](https://github.com/strzar/purge).

### What it does

PURGE expands to Policy Unlearning through Relative Group Erasure. In its RWKU setting, it does not know original forget/retain training records. It reuses RWKU rejection-tuning probes, queries the target model, obtains entity/concept descriptions using GPT-4-conditioned extraction, manually validates them, and selects 50 descriptive entities per target (p. 5).

Its reward is a deterministic lexical check: `r(y)=1` when the completion contains none of the forbidden entity/phrase matches, and 0 otherwise (p. 6, Eq. 13). This is verifiable **absence of detected strings**, not verification that the underlying knowledge or data influence has been deleted.

For each question, sample multiple completions and calculate a group-relative advantage:

`A_i = (r_i - mean(r_group)) / (std(r_group) + epsilon)`.

A PPO-style clipped likelihood-ratio objective increases the probability of above-average responses and decreases that of below-average responses. A KL penalty discourages excessive deviation from a reference policy (p. 6, Eqs. 16–18). Equal rewards throughout a group give zero centered advantages; reward coverage and sampling diversity therefore matter.

### What the measurements actually mean

Main model: Phi-3-Mini-4K-Instruct, 3.8B parameters. Each target is unlearned independently, then results are averaged; this is not a demonstration of long sequences of deletion requests (p. 7). Additional studies use Qwen-2.5 sizes and TOFU with Llama-3.2-1B-Instruct (pp. 9 and 18).

**Do not paraphrase the abstract's “11% unlearning effectiveness” as “11% of training influence removed.”** It is not that quantity. The headline aggregation is insufficiently explained for that interpretation. Table 1 (p. 8) gives clear underlying recall numbers:

| RWKU probe, lower is better | Original | PURGE | NPO |
|---|---:|---:|---:|
| FB | .483 | .428 | .296 |
| QA | .523 | .422 | .257 |
| AA | .601 | .481 | .369 |

PURGE therefore reduces recall by approximately 5.5, 10.1, and 12.0 **percentage points** respectively. The table's QA delta prints −10.7%, whereas the surrounding text says 10.07%; cite the explicit scores rather than repeating the inconsistent delta. NPO has stronger forgetting on all three probes in this table. PURGE's contribution is an alternative trade-off in corpus size, fluency, and retention, not uniformly best forgetting.

The “98% utility” headline does not mean every utility measure is preserved at 98%: Table 1 includes general ability .680→.644, reasoning .428→.418, truthfulness .348→.327, and factuality .383→.423. Report task-specific scores or clearly define any aggregate.

**The 46× figure is not an end-to-end speedup.** Figure 1 (p. 8) and Appendix C.3 (p. 22) compare forget-corpus tokens per epoch: PURGE 13,245; DPO 615,585; GA/NPO 298,067; RT 16,139. This is about 46× versus DPO, 22× versus GA/NPO, and 1.2× versus RT. Token counts exclude the interpretation “46× faster deletion”; online sampling, epochs, teacher/validation cost, and hardware must be accounted for. The authors explicitly note that fewer tokens can require more epochs (p. 8).

### Guarantees and limitations

- The suppression theorem analyzes a stylized recurrence for forbidden-token leakage, with mixing from a base policy. The paper acknowledges that actual GRPO uses zero explicit mixing and uses mixing as an explanatory model for observed leakage floors (pp. 6–7).
- The utility theorem uses the Pinsker-type bound `absolute utility difference <= sqrt(KL/2)` for bounded utility and the appropriate query/response distributions (p. 7). A KL penalty on sampled forget prompts does not automatically certify small KL on every retained state/query. The paper's .158 example is an absolute bounded-score gap, not inherently a 16% relative loss.
- Appendix A.5 (pp. 17–18) claims average surrogate-loss regret to an idealized retained-data comparator decays as `O(1/sqrt(T))` under a mirror-descent treatment. The paper says it cannot empirically verify a full LLM retrain. Equal average loss does not imply equal policies, parameter distributions, or removal of data influence; the idealized optimizer assumptions need independent checking before applying this to neural GRPO.
- The authors explicitly acknowledge surface suppression versus representational erasure, incomplete entity coverage, semantic paraphrases, and reward sparsity (p. 10).
- TOFU Appendix Table 2 reports PURGE forget quality `1.12e-19` versus retained-only model `1`; this is not convincing evidence of matching the retrained model under that metric, despite strong utility. This deserves attention if the user's chosen standard is retraining equivalence.

### What transfers and what is already suggested

Useful ideas: group-normalized rewards, measurable behavioral diagnostics, explicit utility constraints, and accounting for the data needed to construct forgetting rewards. A classical RL extension must replace lexical suppression by an objective connected to the retained-data Bellman problem or the specified retrained learner.

Avoid proposing “use a richer semantic forgetting reward” as obviously novel: p. 10 already suggests embedding-level and completion-wide penalties. The official repository currently implements binary, exponential-decay, and PageRank-weighted rewards; smoothing or graph weighting alone is therefore a weak novelty claim. The repository can evolve beyond the paper, so distinguish code observations from the uploaded experiment description.

## Large Language Model Unlearning — Yao, Xu, and Liu

**Version:** Supplied NeurIPS 2024 publication, 51 pages including extensive output examples and checklist. [Official publication page](https://proceedings.neurips.cc/paper_files/paper/2024/hash/be52acf6bccf4a8c0a90fe2f5cfcead3-Abstract-Conference.html).

### What it does

The paper explicitly distinguishes its objective from matching a retrained model: undesirable prompt–answer pairs can represent unwanted behavior and need not have occurred in the original training corpus (p. 3). The task is to stop generating undesirable answers while preserving behavior on normal prompts.

Its update combines three losses (pp. 4–5, Eqs. 2–6):

`L_total = -a * CE(forget answers) + b * CE(random normal answers paired with forget prompts) + c * KL(original || current on normal prompts)`.

- Negative cross-entropy performs gradient ascent on unwanted answers: make them less probable.
- Random mismatch pairs a forget prompt with an unrelated benign answer: redirect probability when ascent alone is ineffective.
- Forward KL distillation on normal examples preserves the original output distribution.

Losses apply only to response tokens conditioned on the prompt. The original model is the retention anchor. This is **not an RL-based unlearning method**; RLHF is a comparison baseline.

### Experiments and conclusions

- Harmfulness: PKU-SafeRLHF harmful Q&A as forget data, TruthfulQA as normal data; harmfulness judged with a moderation model. Models: OPT-1.3B, OPT-2.7B, and Llama-2-7B (p. 6).
- Copyright: first fine-tune on Harry Potter and the Sorcerer's Stone; then suppress completion leakage while preserving BookCorpus completion behavior. Leak evaluation uses BLEU-based matching (pp. 7–8).
- Hallucinations: HaluEval negative Q&A versus TruthfulQA normal prompts. The paper describes a distribution shift that enables separation of these prompt types; this does not establish general improvement in factual reasoning (p. 8).
- Measures include harmful/leak/hallucination rates, output diversity/fluency, normal-prompt reward, and BLEURT similarity to the original model. There is no experiment showing equality with a retained-only retrained policy.
- Low harmfulness can result from whitespace, repeated characters, or nonsense; the authors explicitly accept this under their goal and show examples (pp. 4, 6–8). Suppression and helpful behavior are different outcomes. A template variant can produce refusals (p. 10).

**Scope of the “2% of RLHF compute” claim:** Section 4.5 (p. 9) and Figure 1 (p. 10) compare OPT-1.3B on PKU-SafeRLHF, single A100 SXM4 80GB GPU, against a full SFT + reward-model-training + PPO pipeline with mostly default hyperparameters. It is measured wall time in that setting, not a universal cost ratio to modern RLHF, ordinary RL training, or retained-data retraining. Table 4 shows trade-offs: plain GA has 1%/3% harmful rates for seen/unseen prompts, with fluency not meaningful; GA+Mismatch has 6%/7%, while full RLHF has 4%/7.5%. Avoid claiming superiority on every alignment measure.

### What transfers

This supplies a useful negative-gradient baseline and shows why utility regularization matters. For RL, a baseline could ascend an explicitly chosen forgotten-experience surrogate while minimizing retained TD/Bellman loss. It must be called a heuristic: ascent on a TD loss can merely create incorrect values, and its meaning changes with bootstrapped targets. Random action mismatches can destroy useful behavior; they are not a principled analogue of removing experience. Matching the original policy too strongly can preserve the very influence targeted for deletion.

## Reading order and questions to bring to the main project

Read Yao pp. 3–5 to learn the basic gradient baseline and the distinction between suppression and retraining. Read RULE pp. 4–7 and 9 to learn refusal boundaries and empirical Pareto curves. Read PURGE pp. 5–10 and 17–18 to understand group rewards and the limits of the claimed guarantees. Then use the exact-RL-unlearning paper as the formal starting point for the user's agent-deletion objective.

For each candidate method ask: What object is deleted? What information is available during deletion? Which learner defines the counterfactual? Which randomness is compared? Does the evaluation measure retraining distance or only output suppression? What happens to retained transitions whose targets depend on deleted experience? Do timing numbers include offline storage, provenance, teacher generation, and all online rollouts?
