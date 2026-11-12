# Phase 4 — Improvement and ablations checklist

Reference: [project plan](fastdepth_edge_ai_plan.md). Improvement A is required by the plan; B is optional if time permits.

## Improvement A — Nano-specific pruning

**Read immediately before pruning:** [FastDepth](../papers/fastdepth.pdf), PDF pages 3 and 6, section III-B and the pruning results in Table VII; then [NetAdapt](../papers/netadapt.pdf), PDF pages 4–6, sections 3.1–3.3 and Algorithm 1. NetAdapt generates candidate reductions, briefly fine-tunes and evaluates them, then retains a candidate under a resource constraint. For the lookup table, read NetAdapt PDF pages 7–8, section 3.4 and Figures 3–4: layer timings are estimates of total latency, so validate shortlisted candidates as complete TensorRT engines. Inspect actual skip connections in code before changing channels.

- [ ] Before tuning, write the target: improve TensorRT FP16 Nano latency over the paper's pruned model, with a maximum allowed RMSE increase (for example, 0.02 m).
- [ ] Establish the reference model's latency, RMSE, δ1, model size, and energy using the recorded benchmark protocol.
- [ ] Map prunable channels and dependencies, including matching channels across additive skip connections.
- [ ] Measure relevant layer or subnetwork configurations on the Nano and build a latency lookup table; record shapes, precision, runtime, and measurement method.
- [ ] Use the lookup table to generate NetAdapt-style channel-pruning proposals for a chosen latency reduction target.
- [ ] Check each proposal for valid channel dimensions and a working forward pass; document how coupled layers are pruned.
- [ ] Fine-tune each viable candidate briefly on the training subset, using a fixed validation protocol to select candidates without training on validation data.
- [ ] Select the most accurate candidate meeting the iteration target; repeat within the available compute budget and record every proposal and decision.
- [ ] Export the final candidate, build its Nano TensorRT FP16 engine, and measure actual end-to-end inference latency under the reference protocol.
- [ ] Evaluate RMSE and δ1 on the same validation set and measure model size and energy per frame.
- [ ] Compare the result with the predeclared target and the paper's pruned model; report the result even if the target is missed.

**Done when:** a comparison table covers latency, RMSE, δ1, size, and energy, and states whether the preset latency and accuracy targets were met.

## Improvement B — Distillation (optional)

**Read only if doing this option:** [Depth Anything V2](../papers/depth_anything_v2.pdf), PDF page 6 (teacher and pseudo-label approach) and page 9, section 7.3 (metric-depth fine-tuning). The paper distinguishes relative and metric models; check the chosen checkpoint and repository for its output representation, input preprocessing, and licence. The proposed per-image alignment and student loss are this project's experiment, not a reproduction of the paper's complete training method.

**Alignment reminder:** for valid training pixels, minimize `Σ(s × teacher + t − target)²`; solve the two-variable normal equations for `s` and `t`. Handle a near-constant teacher output, and verify whether transformation of inverse-depth output is needed before fitting.

- [ ] Decide whether time and compute permit this work; document the decision.
- [ ] Verify the teacher checkpoint's licence, output representation (depth or inverse depth), input requirements, and preprocessing in the paper and code.
- [ ] Generate teacher outputs for training images only and store them with stable image identifiers.
- [ ] Implement and check per-image scale-and-shift alignment over valid training pixels, accounting for the teacher's output representation and degenerate cases.
- [ ] Choose and record α, training schedule, student initialization, and the ground-truth/teacher loss definitions.
- [ ] Train the student without using validation images for teacher alignment or training.
- [ ] Evaluate against the original checkpoint on the held-out validation set and measure latency under the same conditions.

**Done when:** the comparison shows whether the distilled student improves RMSE while keeping the architecture and measured latency comparable.

## Ablations

**Read while designing comparisons:** [NetAdapt](../papers/netadapt.pdf), PDF pages 5–7, for proposal selection and short-term fine-tuning; [Depth Anything V2](../papers/depth_anything_v2.pdf), PDF page 6, for its teacher/student context if using option B. Keep the comparison controlled so the effect of each project choice is measurable.

- [ ] Define controlled comparisons for each method step, keeping data, evaluation, and measurement settings fixed.
- [ ] For pruning, measure at least a candidate without fine-tuning and the corresponding fine-tuned candidate.
- [ ] If distillation is used, include an α = 1 ground-truth-only training baseline and any other comparison needed to isolate the teacher's effect.
- [ ] Save configurations, checkpoints, metrics, and results in an ablation table; explain missing runs and limitations.
