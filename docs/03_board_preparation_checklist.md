# Phase 2 — Prepare for the boards checklist

Reference: [project plan](fastdepth_edge_ai_plan.md). Keep the model, data preprocessing, and input shape consistent across runtimes.

## Step 2.1 — Export ONNX

**Read only as needed:** [FastDepth](../papers/fastdepth.pdf), PDF page 3, Figure 2, to recognize the model's upsampling, convolutions, and additions in the exported graph. ONNX and TensorRT support depends on the installed versions, so check their documentation and the actual export; the paper does not establish Nano compatibility.

- [ ] Inspect the Nano's installed JetPack, TensorRT, CUDA, and supported ONNX/operator versions rather than assuming a particular installation.
- [ ] Choose a compatible opset and export each of the four evaluation-mode models with a fixed `1 × 3 × 224 × 224` input.
- [ ] Validate each ONNX graph and inspect export warnings and unsupported operations.
- [ ] Run the same input through PyTorch and ONNX Runtime; compare output shape, finite values, and numerical differences.
- [ ] Investigate differences greater than approximately `1e-4` and record the comparison method and tolerance.
- [ ] Save export commands, environment versions, ONNX files, and comparison results.

**Done when:** all four exports run and agree with PyTorch within the stated tolerance, or any justified exception is documented.

## Step 2.2 — Set baseline and budget before measurement

**Read when choosing the comparison:** [FastDepth](../papers/fastdepth.pdf), PDF pages 4–6, Figure 3 and Tables I–VII. Note which numbers are for the TX2 CPU or GPU and which optimization produced them; they are reference points, not Nano results. If testing Depth Anything V2, its [paper](../papers/depth_anything_v2.pdf), PDF page 1, gives model-scale context; use the repository to verify actual deployment requirements.

- [ ] Write down the baseline model, runtime, precision, batch size, and input shape; a candidate is unoptimized MobileNet-NNConv5 in PyTorch FP32.
- [ ] Record the primary target: **at most 33 ms per frame at batch size 1 (approximately 30 fps)**, and define precisely which processing stages are timed.
- [ ] Record the target before viewing Nano benchmark results and preserve the decision in experiment notes.
- [ ] Decide whether Depth Anything V2 Small is useful as a heavy reference; if so, quickly test export and Nano runtime compatibility, and record the outcome.

**Done when:** the baseline, timing scope, and budget are written down before the Nano measurements.

## Step 2.3 — Build the benchmark harness

**Read when defining the measurement:** [FastDepth](../papers/fastdepth.pdf), PDF page 4, section IV-A and Table II, for the original platform and inference conditions. The key distinction is measured latency versus theoretical MACs: lower MACs do not guarantee lower latency. Use runtime documentation for GPU synchronization and measurement APIs.

- [ ] Implement configurable model loading, input preparation, warm-up, repeated inference, and machine-readable result output.
- [ ] Synchronize GPU work around timed intervals and distinguish inference latency from preprocessing, transfer, and postprocessing if measured separately.
- [ ] Record model, input shape, batch size, device, runtime and versions, precision, warm-up and timed-run counts, and timing scope.
- [ ] Report median, p95, and p99 latency; retain individual samples so repeated-run confidence intervals can be calculated.
- [ ] Experiment with warm-up and sample counts until medians are reasonably stable; record the selected settings.
- [ ] Run the harness on the laptop and check output, timing behavior, and reproducibility.

**Done when:** the harness produces the specified statistics and metadata on the laptop and is ready to adapt to both boards.
