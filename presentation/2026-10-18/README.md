# Comprehensive research presentation: 18 October 2026

Updated 15 October. This folder contains a **33-slide Markdown outline with speaker notes**, not a rendered PPTX. The scope covers the full work so far, with no assumed ten-minute limit. Experimental evidence is frozen through Checkpoint 06. New experiments must receive their own dated report before changing presentation claims.

## Read and rehearse in this order

1. [Learning guide](../../docs/LEARNING_GUIDE.md): problem definition, paper distinctions, equations, tabular repair, LSPI and worked counterexample. Assumes basic RL and explains the math.
2. [Talk track](TALK_TRACK.md): all 30 slide sections, speaker notes, claim–evidence table, backup material and professor questions.
3. [Professor update](../../PROFESSOR_UPDATE.md): concise progress language; the top 14 October update supersedes historical material below it.
4. Checkpoints [01](../../checkpoints/01_exact_local_repair.md), [02](../../checkpoints/02_scc_controls.md), [03](../../checkpoints/03_global_controls_and_lspi.md), [04](../../checkpoints/04_lspi_cost.md), and [05](../../checkpoints/05_lspi_matched_reuse.md): how each stronger comparator changed the interpretation.
   Then [Checkpoint 06](../../checkpoints/06_policy_sequence_certificate.md): how a narrower policy-sequence guarantee enables a new fast path, why it sometimes loses, and which prior art limits novelty.
5. [Implementation plan](../../implementation_plan.md) and [progress log](../../PROGRESS.md): what is complete, what is still a hypothesis, and which gate comes next.

## What to demonstrate

From `C:\Users\adity\Research`:

```powershell
python scripts/run_tabular_demo.py
python scripts/run_lspi_demo.py --output results/presentation_lspi_demo.json
python -m unittest discover -s tests -v
```

Explain the counterfactual target before showing results: the same declared learner trained on the retained trajectories alone. In the LSPI example, show full-data `(a,a)`, incorrect one-shot deletion `(b,b)`, and correct retained-only `(a,b)`. Explain the retained transition whose target changes. The four-iteration cost experiment is separate from this three-step teaching fixture.

Do not run full timing matrices during the talk. Use the archived [tabular global table](../../output/benchmarks/global_checkpoint_summary.md) and [LSPI table](../../output/benchmarks/lspi_checkpoint_summary.md), keeping preparation-inclusive and request-only costs distinct. The archives contain the data and provenance needed to reproduce them later.

## Preparation phases

| Date | Work | Completion condition |
| --- | --- | --- |
| 14 October | Freeze and cross-check evidence; prepare study material | Checkpoint 04, archive, guide, outline and current progress agree |
| 14–15 October | Study the learner, proofs and counterexample | Explain every symbol and the difference between policy fidelity, privacy and retained return without reading notes |
| 15–16 October | Bounded continuation experiments | Checkpoints 05–06 complete: equal reuse, inverse control, policy certificate and allocation passes; study the narrower guarantee and fallback cases |
| 16–17 October | Rehearse the comprehensive presentation | Explain every checkpoint, its strongest control, limitations, and resulting decision; verify numeric slides against sources |
| 18 October | Present research progress and discuss the next claim | Obtain feedback on the deletion model, strongest comparator, and proof needed before neural scale-up |

Evidence preparation and the matched-reuse continuation are complete. Study, rehearsal and further experiments are planned activities, not automatic scheduled jobs.

## Research position to communicate

We have a tested research system and increasingly strong comparisons. Exactness is relative to a specified rational tabular learner or finite fixed-feature LSPI schedule. The latest certificate narrows the output to policy sequence/status and retains old internal anchors; it has conditional lifecycle gains but is not full-state erasure. None of the current mechanisms has established publication-level novelty. No neural/GPU, real-environment return, or privacy result has been produced. Present the proof, gains, negative cases and next controls together.


16 October evidence addendum: [Checkpoint 07](../../checkpoints/07_selective_decision_repair.md) is complete. Slides 30-32 and learning-guide Section 11 explain selective/shared-anchor repair, changed policies, unique-feature controls and memory costs. Freeze these measured claims for rehearsal; numerical and novelty gates remain open.
