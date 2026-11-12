# FastDepth on Jetson Nano and Arduino UNO Q: Learning and Build Plan

PS13 project, 2401CS62. Written Tue 3 Nov 2026. Reviews: 10 Nov, 17 Nov.

**The story you are building:** a depth-estimation network that misses a real-time budget on the Jetson Nano, an optimization ladder that brings it inside the budget, and one improvement of your own that the original paper did not do.

**Project approval:** Professor approval to move from chess to FastDepth has been obtained.

---

## 1. Mathematics you need

Use this section as a reference when a checklist task needs the topic; you do not need to complete all of the reading or derivations before implementation. Each topic includes a derivation and a number to check your understanding.

### 1.1 Dense regression: losses and depth metrics

- Pixel-wise regression: the model outputs one depth value per pixel; the loss is averaged over valid pixels only (NYU depth maps contain invalid zeros).
- Metrics: RMSE, MAE, REL = mean(|d − d̂| / d), and δ1 = share of valid pixels where max(d/d̂, d̂/d) < 1.25. Also R² over all valid pixels, for PS13.
- **Derive:** why δ1 is scale-relative while RMSE is in metres. Why is a 0.2 m error worse at 1 m than at 8 m under REL but equal under RMSE?
- **Check:** if d = 2.0 m and d̂ = 2.4 m, the δ1 ratio is 1.2, which counts as correct; d̂ = 2.6 m gives 1.3, which does not.

### 1.2 Cost of convolutions

- Standard convolution multiply-adds: H_out · W_out · C_in · C_out · k².
- Depthwise separable convolution: depthwise H·W·C_in·k², plus pointwise H·W·C_in·C_out.
- **Derive:** the ratio separable / standard = 1/C_out + 1/k².
- **Check:** k = 3, C_out = 256 gives about 0.115, so roughly 8.7 times fewer multiply-adds.
- Parameters: count them for both forms, including bias.

### 1.3 Encoder–decoder design

- What the encoder (MobileNet) does to resolution, and why the decoder must upsample back to 224 × 224.
- Nearest-neighbour upsampling followed by convolution (the paper's "NNConv5") versus transposed convolution: cost and checkerboard artefacts.
- Skip connections: additive versus concatenation. **Derive:** how each changes the decoder's channel count and multiply-adds.

### 1.4 Roofline and why fewer operations is not always faster

- Attainable throughput = min(peak compute, arithmetic intensity × memory bandwidth); intensity = operations / bytes moved.
- **Derive:** the arithmetic intensity of a 3 × 3 depthwise convolution against a 1 × 1 pointwise convolution on the same feature map. Which one is memory-bound?
- Use this to explain why FastDepth measured latency instead of trusting multiply-add counts.
- **Check:** the Nano's memory bandwidth is 25.6 GB/s. Moving one FP32 feature map of 112 × 112 × 64 once takes how long at that rate? (About 0.13 ms.)

### 1.5 Structured pruning and NetAdapt

- Removing output channel j of layer l also removes input channel j of layer l + 1.
- With additive skip connections, the two tensors being added must keep identical channels, so those layers are coupled. Work out which layers in FastDepth are coupled.
- NetAdapt's loop: per iteration, generate one proposal per layer that meets a latency reduction target, fine-tune each briefly, keep the most accurate. Know what a latency lookup table is and why it replaces on-device measurement inside the loop.
- **Derive:** if a 3 × 3 convolution goes from 256 to 192 output channels, by what fraction do its own multiply-adds and the next layer's fall?

### 1.6 Floating-point precision

- FP32 versus FP16 layout (sign, exponent, mantissa bits); FP16 range (max 65,504) and spacing near 1.0 (about 0.001).
- Why depth in metres (0–10) is safe in FP16, but some intermediate activations may not be.
- Why the Nano gains nothing from INT8: its GPU has compute capability 5.3, and TensorRT only supports FP32 and FP16 there.

### 1.7 Distillation for dense regression (needed only for Improvement B)

- Loss L = α · L(student, ground truth) + (1 − α) · L(student, teacher).
- Depth Anything V2 predicts relative depth, so it must be aligned before use as a metric teacher. Fit a scale s and shift t per image by least squares: minimise Σ (s · d_T + t − d_target)² over valid pixels.
- **Derive:** the closed-form solution for s and t (a 2 × 2 system of normal equations).
- First check in the Depth Anything V2 paper or code whether the output is depth or inverse depth, because the alignment differs.

### 1.8 Measurement statistics

- Medians and p95 or p99 latency, with confidence intervals across repeated runs.
- Energy per frame = average power × latency, total and idle-subtracted.
- Why a single run's number is not a result.

---

## 2. Paper reference list (read sections as needed)

1. **FastDepth.** Wofk, Ma, Yang, Karaman, Sze, ICRA 2019 · [arXiv 1903.03273](https://arxiv.org/abs/1903.03273) · [code and trained models](https://github.com/dwofk/fast-depth)
   - Read the task-specific sections linked from the phase checklists, focusing on decoder design, pruning, runtime and accuracy when needed.
   - Write down: what their starting baseline is, how much each step saved, and which steps were measured on the CPU versus the GPU.

2. **NetAdapt.** Yang et al., ECCV 2018 · [arXiv 1804.03230](https://arxiv.org/abs/1804.03230)
   - Read the relevant parts of section 3 when implementing pruning and lookup tables.
   - Write down: the constrained objective, the per-iteration loop and how proposals are generated.

3. **Depth Anything V2.** Yang et al., 2024 · [arXiv 2406.09414](https://arxiv.org/abs/2406.09414) · [code](https://github.com/DepthAnything/Depth-Anything-V2). Only needed for Improvement B.
   - If pursuing Improvement B, read the task-specific references in its checklist.
   - Write down: the output type (relative or metric, depth or disparity), the input size and the Small model's licence (Apache-2.0).

---

## 3. Steps

Each step lists the goal, a **done when** test you can verify, and what you must work out yourself.

### Phase 0: Setup (Tue 3 – Wed 4 Nov)

**Step 0.1 — Get compute.**
- Goal: a Kaggle account with GPU access, and Colab as a backup.
- Done when: a Kaggle notebook shows a T4 or P100 GPU in `nvidia-smi`, and your weekly GPU quota is visible in account settings.
- Figure out: what verification Kaggle requires before enabling GPUs, and how to save outputs so a 12-hour session cut-off doesn't lose work.

**Step 0.2 — Get the data without downloading 32 GB to your laptop.**
- Goal: NYU Depth v2 validation data, plus a training subset, inside the notebook.
- Done when: you can display one RGB image next to its depth map, the depth values sit roughly between 0 and 10 m, and you know how many validation images you have.
- Figure out: whether the Hugging Face copy (`sayakpaul/nyu_depth_v2`, HDF5 shards) matches what the FastDepth dataloader expects (keys, shapes, units), and whether its validation split is the standard 654 images.

### Phase 1: Reproduce the paper (Wed 4 – Thu 5 Nov)

**Step 1.1 — Load all four trained models.**
- Goal: run a forward pass with each of the four checkpoints.
- Done when: each returns a depth map of the expected size for a 224 × 224 input.
- Figure out: the code was written for PyTorch 0.4.1. Work out how its checkpoints were saved (whole model or state dict) and what you need from `models.py` to load them in your PyTorch version. **Do this first; if it fails, everything else waits.**

**Step 1.2 — Write your own metrics.**
- Goal: RMSE, MAE, REL, δ1 and R² over valid pixels, written by you, not copied.
- Done when: on the pruned model, your RMSE and δ1 land close to the README's 0.604 m and 0.771.
- Figure out: the exact preprocessing their evaluation uses (resize, crop, normalisation). If your numbers are off, preprocessing is the first suspect.

**Step 1.3 — Count the cost.**
- Goal: parameters and multiply-adds for each of the four models.
- Done when: your count for the pruned model is close to the README's 0.37 billion multiply-adds, and you can explain any gap.
- Figure out: do it once by hand for one encoder block and one decoder block (section 1.2), then with a counting tool for the whole network.

### Phase 2: Prepare for the boards (Thu 5 – Fri 6 Nov)

**Step 2.1 — Export to ONNX.**
- Goal: one ONNX file per model, at a fixed 1 × 3 × 224 × 224 input.
- Done when: ONNX Runtime and PyTorch outputs agree to within about 1e-4 on the same input.
- Figure out: which ONNX opset TensorRT 8.2 on the Nano (JetPack 4.6) accepts, and whether any layer exports to an operation it does not support.

**Step 2.2 — Choose the model that struggles, and fix the budget.**
- Goal: a baseline that a reasonable engineer might deploy first, and a written budget.
- Done when: you have written "real-time = at most 33 ms per frame at batch 1 (30 fps)" in your plan *before* measuring, and named the baseline (for example, the unoptimized MobileNet-NNConv5 in plain PyTorch FP32).
- Figure out: whether to add Depth Anything V2 Small as a heavy reference point, and whether it even converts for TensorRT 8.2. Decide with a quick test, not a guess.

**Step 2.3 — Build the benchmark harness.**
- Goal: one script that loads a model, warms up, times many runs and writes the results to a file.
- Done when: it records median, p95 and p99 latency, the device, runtime, version, precision and model name, and works on your laptop.
- Figure out: how many warm-up runs and timed runs give stable medians, and how to time GPU work correctly (synchronisation).

### Phase 3: Board weekend 1 (Sat 7 – Sun 8 Nov)

**Step 3.1 — Measure the Nano.**
- Goal: the ladder on the Nano: four models × {PyTorch FP32, TensorRT FP32, TensorRT FP16}.
- Done when: every cell has latency, power and memory, and you have an accuracy check from outputs produced on the board.
- Figure out:
  - how to build TensorRT engines on the Nano itself;
  - how to put the Nano in its maximum power mode;
  - how to read power from the onboard INA3221 sensor during a long run;
  - how to compare FP16 outputs against FP32.

**Step 3.2 — Measure the UNO Q.**
- Goal: the same models on the UNO Q's A53 cores.
- Done when: you have latency and power for FP32 on at least one runtime, and a yes/no answer on whether INT8 helps there.
- Figure out: which runtime installs on the UNO Q's Debian, and how to measure power (it has no onboard power sensor I could find).

**Step 3.3 — Draw the ladder.**
- Goal: a plot of latency against RMSE, one point per model and precision, with the 33 ms line.
- Done when: anyone can read off which configurations pass the budget and what accuracy each costs.

### Phase 4: Your improvement (Mon 9 – Fri 13 Nov)

Do A. Add B only if A is finished by Wednesday.

**Improvement A — Nano-specific pruning beyond the paper.**
- Goal: a model faster than the paper's pruned model on the Nano in TensorRT FP16, losing no more than a stated amount of accuracy (choose the limit in advance, for example 0.02 m RMSE).
- Done when: your pruned model meets that target, and a table compares it with the paper's model on latency, RMSE, δ1, size and energy.
- Figure out:
  - build a latency lookup table from your weekend-1 Nano measurements;
  - generate NetAdapt-style proposals from it;
  - keep coupled layers (section 1.5) consistent;
  - fine-tune briefly after each step on your training subset;
  - decide how many iterations your GPU quota allows.

**Improvement B — Distillation from Depth Anything V2 (optional).**
- Goal: the same FastDepth architecture, trained with a teacher, more accurate at identical latency.
- Done when: the distilled student beats the original checkpoint's RMSE on the validation set, and its latency is unchanged.
- Figure out:
  - precompute and store teacher outputs once;
  - align them per image (section 1.7);
  - choose α;
  - make sure no validation image is used for alignment or training.

**Ablations (both improvements).**
- Done when: for each step of your method, you can show results with and without it.
- Examples: pruning without fine-tuning; distillation with α = 1, which means ground truth only.

### Phase 5: Validation and report (Sat 14 – Tue 17 Nov)

**Step 5.1 — Final board runs (Sat 14 – Sun 15 Nov).**
- Done when: the final models are run 5 times each on both boards, with latency, energy per frame and accuracy from board outputs.

**Step 5.2 — Demo (optional).**
- Done when: a camera feed shows live depth on the Nano, with the frame rate displayed. Record a backup video.

**Step 5.3 — Report and slides.**
- Done when: every PS13 metric appears, and the three acts (struggle, optimize, your improvement) are each backed by a table or plot.
- Structure it around the grading sheet: introduction and contribution, literature review, dataset, implementation, methodology and ablations, results.

---

## 4. What can go wrong

| Risk | Early warning | What to do |
| --- | --- | --- |
| Old checkpoints will not load | Step 1.1 fails | Ask on the repo's issues page, or rebuild from `models.py` and the state dict |
| Your accuracy does not match the paper | Step 1.2 is off by more than a little | Preprocessing mismatch; compare against their dataloader line by line |
| TensorRT rejects an operation | Step 2.1 or 3.1 | Re-export with a different opset, or replace the layer with an equivalent |
| GPU quota runs out | Kaggle settings | Use Colab for small jobs; cut the training subset |
| The baseline already meets 33 ms | Step 3.1 | Report it honestly; tighten the budget only with a stated reason, or use a heavier baseline |
| No approval to switch from chess | Phase 0 | Stay with chess and use the heavier CNN evaluator instead |
