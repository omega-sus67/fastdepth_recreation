# Phase 1 — Reproduce FastDepth checklist

Reference: [project plan](fastdepth_edge_ai_plan.md). Complete checkpoint loading first; later results depend on it.

## Step 1.1 — Load four trained models

**Read alongside loading:** [FastDepth](../papers/fastdepth.pdf), PDF page 3, Figure 2 and section III-A. The encoder extracts features; the decoder upsamples them into a dense depth map, with additive skip connections. Use the repository code and checkpoint definitions as the authority for exact tensor shapes and loading.

- [ ] Clone or obtain the FastDepth code and all four pretrained checkpoints; record source, revision, and checkpoint names.
- [ ] Inspect how each checkpoint was saved and which model classes and dependencies it needs.
- [ ] Set up a reproducible environment; record Python, PyTorch, CUDA, and package versions.
- [ ] Resolve old PyTorch compatibility issues, preserving the model architecture and weights; document changes.
- [ ] Load each model in evaluation mode and run a forward pass on a `1 × 3 × 224 × 224` input.
- [ ] Record each model's output shape and verify that its depth map has the expected dimensions and finite values.
- [ ] Save a minimal repeatable loading and inference command for each checkpoint.

**Done when:** all four checkpoints load and return depth maps of the expected size.

## Step 1.2 — Implement and check metrics

**Read before evaluating:** [FastDepth](../papers/fastdepth.pdf), PDF page 4, section IV-A and Table I, for the authors' dataset and metrics. Apply the repository's actual crop and normalization for reproduction. For each valid pixel, REL is `abs(prediction − depth) / depth`; δ1 counts `max(depth / prediction, prediction / depth) < 1.25`. RMSE is in depth units; δ1 is relative. R² is an additional project requirement.

- [ ] Inspect the original evaluation loader and document resizing, cropping, normalization, depth units, valid-pixel mask, and evaluation region.
- [ ] Implement RMSE, MAE, REL, δ1, and R² over valid pixels; define how values are aggregated across images and how invalid or zero predictions are handled.
- [ ] Check metric calculations on small hand-computable examples, including invalid pixels and the plan's δ1 examples.
- [ ] Evaluate the pruned model on the compatible validation split with the original preprocessing.
- [ ] Compare RMSE and δ1 with the README references of approximately 0.604 m and 0.771; investigate discrepancies starting with preprocessing and split definitions.
- [ ] Save per-model metrics, evaluation configuration, sample counts, and predictions needed to reproduce the result.

**Done when:** the metrics work on known examples and the pruned model's results are close to the reference, or any remaining discrepancy is explained.

## Step 1.3 — Count model cost

**Read while examining layers:** [FastDepth](../papers/fastdepth.pdf), PDF pages 3–5, section III-A and Tables III–IV. Standard convolution MACs are `H × W × Cin × Cout × k²`; depthwise-separable MACs are `H × W × Cin × (k² + Cout)` for matching spatial sizes. Count parameters separately and include biases when present. Compare actual layer shapes before applying these formulas.

- [ ] Calculate parameter and multiply-add counts by hand for one encoder block and one decoder block, including bias parameters where present.
- [ ] Select a counting tool and document its conventions, input shape, and treatment of unsupported operations.
- [ ] Measure parameters and multiply-adds for all four models at `1 × 3 × 224 × 224`.
- [ ] Compare the pruned model with the README's approximately 0.37 billion multiply-adds; explain counting or architecture differences.
- [ ] Save the results in a table with model names and counting assumptions.

**Done when:** all four models have reproducible counts and the pruned model's difference from the reference is understood.
