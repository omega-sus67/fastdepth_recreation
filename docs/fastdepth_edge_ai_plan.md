# FastDepth on Jetson Orin Nano: Complete Implementation, Experiments, Report, and Poster Plan

Updated: 14 November 2026 (Asia/Kolkata). Poster and final report production are deferred by user request. Project: PS13, 2401CS62.

This is the single authoritative detailed plan. [immediate_priority.md](immediate_priority.md) contains only the current next actions and must be rewritten as those actions finish or dependencies change. [agents.md](../agents.md) defines the working rules. The older phase-checklist paths are navigation pages into this document, not separate backlogs.

## 1. Source requirements and interpretation

The following supplied files were inspected, including both guideline pages and the rendered poster slide:

- [Machine Learning Project Report Guidelines](Machine_Learning_Project_Report_Guidelines_Edited.pdf), pages 1–2: required report structure, contributions, numbered material, single-PDF delivery, and the 12–15 page limit including all sections.
- [Updated poster sample](updated%20poster%20sample.pptx), slide 1: landscape, three-column organization; title/identity banner; abstract, introduction, problem statement, key contributions, methodology/architecture figure, numbered algorithm, results table, graphs, conclusion, and references.

The PDF defines mandatory report requirements. The PPTX is the supplied visual/content reference for the poster. Its photonic-sensor application, regression models, numerical improvements, data split, and subject-specific metrics are example content, not FastDepth requirements. Use the structure and replace the example material with our own measured depth-estimation work. In particular, do not copy its practice of selecting a model using test-set scores.

The guidelines do not specify a report font, margin, citation style, poster print size, poster deadline, or separate slide-deck requirements. Use a consistent readable academic format and numbered references; preserve the supplied poster's 16:9 landscape aspect ratio provisionally. Confirm additional institutional formatting only if it is supplied later. These are project formatting decisions, not requirements inferred from the sample.

The report must contain the prescribed literature-review chapter, but implementation does not wait for a course of reading or derivations. Collect citations and evidence alongside implementation. This document is a work plan, not the final report and not a claim that planned experiments have run.

## 2. Objective, scope, and verified starting point

### 2.1 Problem and success criteria

Build a reproducible monocular metric-depth estimation application from RGB images, based on FastDepth, and deploy it on **Jetson Orin Nano**. Establish baseline accuracy, compare architecture/runtime/precision optimizations, implement a structured-pruning improvement, and document its measured accuracy, resource, and latency trade-offs.

The deployment scope is Orin Nano only. The RTX 3060 Laptop GPU with 6 GB VRAM is the development/training/test machine while the board is unavailable. Original Jetson Nano deployment assumptions and other-board comparisons are outside scope.

### PS13 source and agreed scope — 14 November

The user-supplied [problem statement](../papers/Project%20based%20learning%28PBL%29-%20Machine%20Learning.pdf), page 13, explicitly names **NVIDIA Jetson Nano and Arduino UNO Q** for experimental evaluation. The user reports that the professor has approved **Jetson Orin Nano only**. The Orin Nano is a different Jetson product from the original Jetson Nano; this project-specific agreement is user-reported, not written into the source PDF. There is no UNO Q deliverable in the agreed project scope. Do not claim that the original two-board requirement was experimentally completed.

| PS13 requirement | Current evidence and gap |
| --- | --- |
| Edge AI principles and optimization on a public benchmark | FastDepth, NYU-derived data, structured pruning and detailed notes exist; complete training provenance and scene separation still need work. |
| Enhanced, hardware-aware framework; novel mathematical or algorithmic modifications | Ablation-guided structural pruning and recovery are implemented as exploratory methods, but there is no established novelty over prior work and no Orin-guided selection yet. Experimental implementations are one stated support option; a theorem is not necessarily mandatory. |
| Comparative effectiveness and minimized accuracy degradation | Original and pruning candidates have preliminary matched tuning results, but no candidate has shown an improved accuracy–speed trade-off; stronger conventional controls and broader data are needed. |
| Jetson Nano and UNO Q hardware experiments | The user reports professor approval for Orin Nano only; no Orin experiment has been performed. No desktop timing substitutes for a board result. |
| MAE, MSE, RMSE, R²; compression, memory, speed, computational efficiency, power, energy, robustness and generalization | Some accuracy metrics, parameter counts and desktop timing are recorded. MSE can be calculated consistently from pooled squared error but is not currently a separately reported metric; comparable board memory, speed, power/energy and broader robustness/generalization remain pending. |

The scope approval above is based on the user's report of their conversation, not a direct written statement from the professor. Board access and feasible hardware metrics remain to be arranged.

The working title is **FastDepth Reproduction and Hardware-Aware Optimization for Real-Time Monocular Depth Estimation on Jetson Orin Nano**. It may be shortened for the poster. Do not describe hardware-aware optimization as completed until actual Orin measurements support it.

### 2.2 Predeclared experiment targets

- Input contract: batch 1, three RGB channels, model input `1 × 3 × 224 × 224`; record the actual output shape per variant.
- Baseline: the original unpruned MobileNet–NNConv5 in PyTorch FP32, if its trusted weights can be obtained. If unavailable, record the supported substitute and why before comparisons.
- Primary real-time target: **at most 33 ms per frame at batch 1**. The project decision is to assess p95 pipeline latency from a decoded RGB frame in host memory through preprocessing, input transfer, inference, output transfer, and numerical depth postprocessing. Exclude file decoding, camera acquisition, display, and file writing from this target; measure a complete live-demo frame rate separately. Report model-only median/p95/p99 as additional measurements.
- Required runtime comparison: PyTorch FP32, TensorRT FP32, TensorRT FP16, for supported original variants. INT8 is optional after these work and requires representative calibration/quantization and accuracy checks.
- Proposed pruning acceptance limit: final evaluation RMSE increase no greater than **0.02 m** relative to the paper's pruned checkpoint evaluated through our same pipeline, while reducing measured Orin FP16 latency. Freeze this limit before candidate search; it is our project target, not a quoted guideline or paper threshold. Use tuning accuracy for candidate selection and final evaluation for the final acceptance check.
- If the baseline already meets the latency target, keep the target and report success honestly. Quantify further improvements without manufacturing a failing baseline.

### 2.3 What is actually available

- Professor approval to switch to FastDepth has been obtained.
- Upstream source is vendored at recorded revision `e68492011609c9bfb7de6d402da5d1d201d95bd9`, with its license.
- The recommended independent `.venv-desktop` now uses Python 3.12.3, PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128, ONNX Runtime 1.31.1, ONNX Script 0.7.2, and desktop TensorRT 10.3.0. All 70 packages are recorded in `requirements-desktop-lock.txt`; use `scripts/python.sh` to isolate inherited ROS/user imports. The previous shared environment is preserved.
- `scripts/setup_desktop.sh --locked --with-tensorrt` passed in a clean source snapshot with a newly created independent environment. CPU/CUDA, ONNX Runtime, modern ONNX export, and TensorRT engine build/execution passed, including after cache cleanup. See [Phase A results](phase_a_results.md) and `results/environment/fresh-install.json`. `requirements-desktop-tested.txt` remains historical.
- There are two validation archives and 654 extracted HDF5 files. An RGB/depth preview exists at `results/validation_sample.png`. Count alone does not verify split identity, units, or reproduction preprocessing.
- All four original-author checkpoint endpoints timed out over HTTP and HTTPS on 12 November 2026. One community-mirrored pruned checkpoint has been preserved, converted to explicit config/weights, and forward/reload checked; original-author authentication and full-split accuracy are unverified. The three unpruned pretrained variants remain unavailable. Source/provenance and exact failure records are in `results/checkpoints/`.
- A scene-grouped dataset, preliminary evaluation, image/video smoke paths, ONNX export, desktop runtime checks and multiple pruning/training comparisons now exist. Twenty-four tests pass. A bounded NYU-derived sample has 1,857 training and 445 tuning images across 124 named scenes, with conservative suffix grouping and overlap screening. An official-index hypothesis is supported by 31/32 sampled exact RGB matches and predicts no evaluation/train/tune location-group overlap, but most evaluation identities and original pretraining remain unverified. No board engine or Orin result exists. See [progress history](progress_report.md) and [immediate priorities](immediate_priority.md).
- The user requests three to five simple changes they can explain themselves. The [four-experiment walkthrough](learning/four_experiments.md) separates width, selection rule, recovery training and BN mode as four questions; they are not four independently novel methods. A matched 192-channel weight-versus-ablation comparison and BN-mode follow-up are now recorded as exploratory results.
- The [scene-aware pruning proposal](learning/scene_aware_pruning.md) was tested against random, weight and average-ablation selection. A deterministic-mode three-seed repeat produced near-tied tuning RMSE and no reliable superiority; no novelty or generalization claim is established. Desktop TensorRT accuracy was close in aggregate but showed large individual output discrepancies, and ordered timing replays were unstable. Resolve or bound precision/stream behavior before target-device claims.
- Orin Nano access is unavailable. Exact module memory, carrier, JetPack, CUDA, TensorRT, camera, and supported power modes remain to be confirmed.

## 3. Requirements-to-evidence coverage

Every row below needs an artifact or a justified limitation. A script merely existing is not proof that an experiment passed. Section numbers refer to this plan; report chapter numbers refer to the final five-chapter report.

| Supplied requirement | Planned content and work | Required evidence / final location |
| --- | --- | --- |
| Title Page | Project title, author, roll number, department, institution, contact, submission date | Report page 1; verify identity fields, no guessed contact details |
| Abstract | Problem, method, dataset, target device, main measured findings, limitation | Report page 2; write numerical claims after final experiments |
| Table of Contents | Generated chapter/subsection entries and actual pages | Report page 3 |
| List of Figures and Tables | Generated numbered captions and actual pages | Report page 3 alongside contents |
| Introduction: Background and Motivation | Monocular depth and embedded deployment needs | Chapter 1; citations and application context |
| Introduction: Problem Statement | Accuracy/latency/resource trade-off for RGB-to-depth on Orin | Chapter 1; input/output contract and timing boundary |
| Introduction: Objectives | Reproduction, runnable application, runtime comparison, pruning, evidence | Chapter 1; targets in section 2.2 |
| Introduction: Scope of the Work | Orin only, indoor NYU data, desktop development, optional work bounded | Chapter 1; limitations reflected in Chapter 5 |
| Your contribution (minimum 3–5) | Target five distinct evidence-backed contributions | C1–C5 in section 4; at least three completed defensible claims |
| Literature Review: Existing Approaches | FastDepth, NetAdapt, relevant efficient-depth context; teacher only if used | Chapter 2; comparison table and numbered citations |
| Literature Review: Research Gaps | Hardware/runtime transfer, current-framework reproducibility, measured channel choices | Chapter 2; each gap connected to a tested project choice |
| Methodology: Dataset Description | Source, identities, splits, counts, schema, units, access/license notes | Chapter 3; dataset record and hashed manifests |
| Methodology: Data Preprocessing | Exact RGB/depth transformations and valid-pixel policy | Chapter 3; transform configuration and regression tests |
| Methodology: Exploratory Data Analysis (EDA) | Valid-depth coverage, depth distributions, shapes, duplicates, representative scenes | Chapter 3; section 7 outputs and EDA figure |
| Methodology: Proposed Model Architecture | Encoder, decoder, skip connections, pruned dimensions, shape flow | Chapter 3; original architecture figure and variant table |
| Methodology: Algorithm | Reproduction/export workflow and actual pruning/search procedure | Chapter 3; numbered pseudocode with inputs, outputs, stopping rule |
| Methodology: Experimental Setup | Hardware, software, seeds, splits, batch, timing, precision, power settings | Chapter 3; environment records and fixed protocol |
| Results: Performance Evaluation Metrics | RMSE, MAE, REL, delta1, R²; latency, size, memory, power, energy | Chapter 4; equations, units, masks, aggregation and timing scope |
| Results: Experimental Results | Baselines, exports, runtimes, final model, repeated board runs | Chapter 4; raw runs, summary tables, model identifiers |
| Results: Comparison with Existing Models | Four original variants where available; proposed model under same protocol | Chapter 4; measured rows separated from paper-reference rows |
| Results: Graphical Analysis | EDA, accuracy/latency frontier, error/prediction plots, qualitative depth | Chapters 3–4; reproducible plot data and generation commands |
| Results: Discussion of Findings | Effects, trade-offs, errors, unexpected outcomes, variability, budget outcome | Chapter 4; evidence-based interpretation, no unsupported generalization |
| Conclusion: Key Findings | Answer objectives and contribution claims using final results | Chapter 5 |
| Conclusion: Limitations | Dataset/domain, training subset, missing weights/hardware cells, numerical/runtime limits | Chapter 5; all material omissions disclosed |
| Conclusion: Future Research Directions | Justified next experiments such as quantization, distillation, outdoor robustness | Chapter 5; distinguish future work from implemented work |
| References | Papers, code, data, runtime/hardware documentation actually used | Numbered bibliography; all citations resolve |
| Appendix (if required) | Compact reproducibility information only if space remains | Optional page 15, inside total page limit |
| Separate numbered main chapters | Chapters 1–5 each start on a new page | Final PDF visual/page-break inspection |
| Number figures, tables, equations, algorithms and reference them | Stable labels, captions, in-text mentions, generated lists | Report source and final PDF checks |
| Submit one PDF, 12–15 pages including all sections | Target a 14-page self-contained report; 15 only with needed appendix | `deliverables/report/fastdepth_report.pdf`; PDF page-count check |
| Clearly describe methodology, results, conclusions | Trace each claim to configuration, run, and figure/table | Final editorial audit and artifact index |
| Poster follows supplied sample organization | Header, three columns, contributions, architecture, algorithm, results, conclusion, references | Section 17; editable PPTX plus exported PDF, visually checked |

## 4. Planned contributions and claim discipline

The guideline phrase is “Your contribution (minimum 3–5).” Target five distinct contributions, with at least three genuinely completed and supported. Do not split one implementation into artificial novelty claims. Engineering/reproduction contributions are valid project work but must be identified as such rather than presented as new FastDepth architecture research.

| ID | Planned contribution | Work needed | Evidence that permits the final claim |
| --- | --- | --- | --- |
| C1 | A reproducible current-framework FastDepth inference/evaluation pipeline | Environment, legacy checkpoint conversion, verified transforms, model registry and metrics | Fresh-install instructions, conversion checks, baseline accuracy and tested inference commands |
| C2 | A controlled architecture/runtime/precision comparison for this target | Four original variants where accessible, ONNX parity, desktop checks and Orin runs | Same-split accuracy tables and supported board runtime matrix with raw samples |
| C3 | An Orin-guided structured-pruning experiment beyond the reference checkpoint | Dependencies, measured cost estimates, candidates, fine-tuning and final complete-engine validation | Selected model/configuration, candidate log, measured accuracy/latency/energy and target outcome |
| C4 | An integrated deployable image/video application with reproducible measurement | Shared preprocessing, runtime adapters, ONNX/build package, image/video/webcam commands | Board inference and sustained demo; separately defined inference/pipeline timing and resource logs |
| C5 | An ablation and error-analysis study of the accuracy–efficiency trade-off | Before/after fine-tuning, controlled candidate comparisons, depth-bin errors and repeated-run analysis | Ablation table, error plots, qualitative successes/failures, and evidence-backed findings |

Until a row's evidence exists, label it proposed or in progress in drafts. If pruning fails to improve latency, report the implemented method and negative result, not a successful speedup. If a contribution cannot be completed, revise claims to match actual work, maintain at least three defensible distinct claims where possible, and make any remaining evaluator gap explicit.

## 5. Software design, interfaces, and evidence organization

### 5.1 Required components

Implement a small reusable project package under `src/`, with adapters around the preserved upstream source where practical:

- Data: HDF5 reading, manifests, unit validation, paired RGB/depth transforms, train/tuning/evaluation loaders, EDA summaries.
- Models: four-variant registry, explicit architecture/channel configuration, trusted checkpoint import/conversion, model hash and provenance.
- Metrics: tested depth metrics, image-level and pooled aggregation, valid-pixel handling and invalid-prediction diagnostics.
- Runtimes: PyTorch and ONNX Runtime first; TensorRT adapter behind an optional dependency so desktop CPU tools can still run without TensorRT.
- Inference: shared image/directory/video/webcam application, numerical outputs, consistent visualization and error handling.
- Training: losses, seeds, augmentation, optimizer/scheduler, checkpointing/resume, tuning-based selection and controlled fine-tuning.
- Optimization: structural dependencies, candidate generation, measured cost table interface, search logs and stopping rules.
- Benchmarking: synchronization, raw timings, environment metadata, timing boundaries and board telemetry adapter.
- Results: summary/plot generation from saved records, contribution evidence index, report/poster assets.

Keep the desktop software install separate from the board software install. Match export and TensorRT APIs to an explicitly recorded provisional board stack; verify the actual installed stack later. Do not reinstall the existing shared GPU stack as a side effect of documentation or data work.

### 5.2 Planned artifact paths

These paths are specifications; their appearance below does not mean they exist yet.

| Path | Planned contents |
| --- | --- |
| `configs/` | Model, data, training, pruning, export and benchmark configuration |
| `data/manifests/` | Stable training/tuning/evaluation identifiers, sources, hashes and overlap report |
| `checkpoints/` | Original trusted checkpoints, converted weights, fine-tuned candidates and provenance |
| `exports/` | Validated ONNX; board engines with device/runtime/build metadata |
| `results/environment/` | Desktop/board environment and fresh-install records |
| `results/data/` | Dataset audit, preprocessing checks, EDA statistics and figures |
| `results/evaluation/` | Per-image and aggregate accuracy by model/runtime/split |
| `results/training/` | Training curves, saved configurations, seeds and resume checks |
| `results/pruning/` | Candidate structures, cost estimates, fine-tuning and selection history |
| `results/benchmarks/` | Raw timing, telemetry and repeated-run summaries labeled by device |
| `results/figures/` | Regenerable report/poster figures and underlying plot tables |
| `results/evidence_index.csv` | Claim/contribution → artifact → run/configuration → final figure/table |
| `deliverables/report/` | Editable report source, bibliography, final single PDF |
| `deliverables/poster/` | Editable poster source and final PDF |
| `deliverables/demo/` | Demo instructions and backup video if a demo is produced |

Version documents, papers, datasets, code, models, results and final deliverables as requested. Use Git LFS for files too large for normal hosting limits; keep only environments, caches, installed dependencies, build artifacts and local editor/OS files ignored. Keep original guideline/template files intact.

### 5.3 Command contracts to implement

Provide documented commands for `check-environment`, `audit-data`, `infer`, `evaluate`, `model-cost`, `export-onnx`, `compare-runtimes`, `benchmark`, `train`, `prune`, `build-engine` and `make-results`. The exact CLI/module names are implementation choices; do not publish fictitious runnable commands before they exist. Common options should include model/checkpoint, data/manifest, runtime/device, configuration, seed where applicable, and output directory.

A failed or unsupported operation must exit clearly and record the reason. A random-weight structural smoke test must never be presented as pretrained inference, a baseline accuracy result, or a deployment completion.

## 6. Phase A — Reproducible desktop setup and checkpoint acquisition

### A1. Verify and stabilize installation

1. Record current Python, OS, driver, GPU, PyTorch, torchvision, CUDA build, NumPy, ONNX, ONNX Runtime and optional TensorRT versions.
2. Preserve the shared environment. Verify an independent installation in a temporary environment or clean checkout using a tested Python/framework pair and explicit package sources.
3. Make setup idempotent with vendored upstream source; do not require a nested `.git` directory or clone over an occupied source directory.
4. Record enough versions and installation commands to reproduce the tested environment; distinguish a tested dependency record from the historical requirements snapshot.
5. Run package imports, a CUDA operation and a small CPU path. Record failures rather than assuming packages listed in requirements are operational.
6. Document LFS retrieval, project-relative artifact paths and disk/memory expectations. Avoid a second unnecessary full dataset copy during checks.

**Gate A1:** a fresh independent environment passes documented checks, while the current working setup remains usable. **Completed 12 November 2026:** a newly created independent environment installed the pinned packages from a clean source snapshot and passed all five CPU/GPU/export/runtime checks; the legacy environment snapshot and CUDA check remain unchanged. Evidence: [Phase A results](phase_a_results.md), `results/environment/fresh-install.json`, `results/environment/desktop-environment.json`, and `results/environment/legacy-environment-check.json`.

### A2. Obtain and convert original model checkpoints

1. Retry official model links with bounded timeouts; search trustworthy author/upstream-linked mirrors if needed. Save download source, retrieval date, original filename, variant, license information where available and SHA-256.
2. Acquire standard NNConv5, depthwise NNConv5, depthwise+skip-add and pruned depthwise+skip-add weights. Record partial availability honestly.
3. Inspect trusted serialization: whole module versus dictionary/state dict, needed class names, wrapper prefixes and actual pruned topology. Use trusted legacy-loading compatibility only for verified files.
4. Adapt current PyTorch compatibility without changing architecture or weights. Preserve original checkpoints and log changes to loading or compatibility code.
5. Save converted weights plus explicit architecture/channel configuration and a conversion record.
6. Compare legacy and converted outputs where an executable reference is available; otherwise document the narrower evidence available. Verify eval mode, finite outputs and expected shapes for every supported variant.

**Gate A2:** all four trusted pretrained variants have repeatable loading and forward commands, or a precise supported-subset record plus a pending acquisition task. **Partial 12 November 2026:** one community pruned candidate loads/converts with exact legacy/reload parity; actual channels, provenance limits and commands are recorded in [checkpoint documentation](../checkpoints/README.md). Three unpruned architectures pass random structural checks. All four original-author endpoints remain unavailable; original-author authentication and all-original reproduction are pending. Continue independent data/metrics work; any training-from-scratch fallback remains separately labeled.

## 7. Phase B — Dataset verification, preprocessing, and EDA

### B1. Audit source and split identity

1. Inventory validation archives and extracted files; verify counts, readable HDF5 records, expected keys, shape/dtype consistency, and archive/extracted agreement.
2. Trace sample identities to source metadata and the standard FastDepth/NYU evaluation split. Record exact matches or unresolved differences; do not infer correctness solely from 654 files.
3. Confirm depth units and invalid-depth representation using source metadata and observed values. Save conversion rules rather than guessing from plausible ranges.
4. Acquire a manageable documented training subset with persistent storage. Log the source and any preprocessing already applied by the distributor.
5. Reserve a tuning split from training data. Prefer scene/group-separated splitting where scene identifiers are available to reduce near-duplicate leakage.
6. Save sorted manifests, hashes, seed, selection logic and overlap/duplicate checks. Final evaluation is excluded from training, teacher alignment, calibration and candidate selection.

### B2. Reproduce the image/depth transform

1. Compare against upstream loader operations: resize by the original scale, center crop, final `224 × 224` resize, RGB range conversion and depth transformation. Record actual interpolation conventions, axis orders and dtype changes.
2. Replace removed APIs such as `np.asfarray` with equivalent current behavior and test the resulting values. Do not silently substitute ImageNet normalization or a different crop.
3. Verify RGB/depth alignment with saved examples and shape/value checks. Avoid smoothing invalid-depth masks into valid data unnoticed; document how resizing and masking interact for reproduction versus our additional analysis.
4. Keep one shared preprocessing configuration across evaluation, inference, export parity, training validation, desktop benchmark and board benchmark.
5. Separate deterministic evaluation transforms from training augmentation. Record any scale/depth adjustment used during augmentation.

### B3. Produce required EDA evidence

Generate a machine-readable dataset audit and a compact report-ready EDA panel covering:

- Sample counts by split and, if available, scene; file/schema/dimension summaries.
- Valid-pixel fraction per image and total invalid-depth proportion.
- Valid-depth min/max/quantiles and a depth histogram in metres, with counts and axes clearly labeled.
- RGB/depth representative examples spanning near/far scenes and sparse/dense valid coverage, selected by recorded identifiers.
- Corrupt files, missing keys, duplicate or near-duplicate concerns, inconsistent units/shapes and the action taken.
- Training/tuning/evaluation distribution differences and practical limitations of the selected subset.

Use training/tuning EDA for design decisions. A final-evaluation descriptive summary may document the dataset, but do not adapt the method after examining its model errors. If metadata is absent, say that scene-level disjointness cannot be established rather than inventing scene labels.

**Gate B:** loaders produce correctly aligned tensors; schema, units, valid masks and split compatibility are documented; training/tuning/evaluation manifests have explicit non-overlap checks; EDA statistics and at least one reproducible figure are saved.

## 8. Phase C — Metrics, baseline reproduction, cost and usable inference

### C1. Implement and test metrics

For target depth `d`, prediction `p`, and valid ground-truth pixel set `V`, define and number the following in the final report:

- RMSE: `sqrt(sum((p-d)^2) / |V|)`, in metres.
- MAE: `sum(abs(p-d)) / |V|`, in metres.
- REL: `sum(abs(p-d)/d) / |V|`, dimensionless.
- delta1: fraction with `max(p/d, d/p) < 1.25`, dimensionless.
- R²: `1 - sum((p-d)^2) / sum((d-mean(d))^2)`, with explicitly documented behavior for constant targets.

Use positive finite ground-truth depths and any preregistered dataset validity bounds/region. Do not remove pixels merely because predictions are bad. Flag nonfinite predictions as an invalid run; count nonpositive predictions and define a documented positive-epsilon policy for ratio calculations rather than silently discarding them. Save the count affected and test that the policy does not conceal errors.

Implement tests with hand-computable errors, zero/invalid ground truth, empty masks, constant targets, nonpositive/nonfinite predictions and the delta1 threshold boundary. Check metres versus millimetres explicitly. Review upstream metric masking and units before trying to reproduce its number.

Save both pooled-pixel statistics and an image-averaged summary where needed to match the original evaluation. Label aggregation in every table: mean image RMSE is different from pooled-pixel RMSE. R² and uncertainty must use a clearly specified aggregation, not an undocumented average of incompatible quantities.

### C2. Evaluate original variants

1. Run the same verified final evaluation manifest and deterministic transforms for every available original model.
2. Save per-image metrics, aggregate metrics, counts, preprocessing/configuration hash, checkpoint hash and numerical predictions or enough cached outputs to reproduce checks.
3. Compare the pruned model with the upstream reference RMSE approximately 0.604 m and delta1 approximately 0.771. These are cited reference results, not our measured results or a guarantee.
4. Investigate discrepancies in split identity, units, interpolation, crop, masking, aggregation, serialization and model topology; record residual differences.
5. Count parameters, file size, and MACs at the fixed input shape. Define one MAC convention and treatment of resize/add/non-convolution operations. Compare the pruned reference's approximately 0.37 G MACs under that convention.
6. Prepare a model table with architecture, channels, parameters, MACs, checkpoint size and baseline accuracy.

### C3. Implement the application

Support single-image, directory, recorded-video and webcam inference using the same preprocessing. Provide CPU/CUDA selection and runtime/model configuration. Save raw numerical depth in a documented format and separate colorized previews with a consistent scale for qualitative comparisons. Label absolute depth units; per-image display normalization is a visualization choice, not a metric improvement.

Exercise good inputs, missing checkpoints, unreadable files, unsupported runtime selection and failed video sources. Report complete demo FPS separately from model latency. Add a short recorded-video path even without a webcam so the full application can be tested now.

**Gate C:** metrics tests pass, trusted pretrained image/video inference works, baseline metrics for the supported variants are recorded, and reproduction deviations are explained. A screenshot alone does not establish correctness.

## 9. Phase D — Exports, runtime parity, benchmark harness and desktop evidence

### D1. Export portable models

1. Choose a provisional Orin JetPack/TensorRT target from official documentation and record why; actual versions remain pending board access. Do not carry over original-Nano JetPack 4.6/TensorRT 8.2 assumptions. [NVIDIA JetPack documentation](https://developer.nvidia.com/embedded/jetpack-sdk-62) is the source for a chosen stack, not proof of what the future board runs.
2. Export each supported model in eval mode as fixed-shape ONNX with documented opset, names, dtype and model hash. Make dynamic shapes an optional extension only if justified.
3. Run ONNX graph validation; inspect unsupported operators and exporter warnings. Resolve them with equivalent behavior and regression checks.
4. Compare PyTorch and ONNX Runtime on a recorded synthetic input plus representative real samples. Save maximum/mean absolute and relative differences; start with a stated FP32 tolerance around `1e-4`, adjusting only with numerical evidence.
5. Evaluate ONNX accuracy on the same manifest, not just one output tensor.
6. In an isolated compatible desktop environment, exercise TensorRT parsing, engine build and FP32/FP16 inference. Save desktop-specific engine metadata and accuracy checks. Quantify actual depth-metric changes instead of applying FP32 tolerances blindly to FP16.
7. Export ONNX/configuration/sample inputs/expected outputs as the board handoff. Build final engines on the Orin; [NVIDIA engine compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html) prevents treating the desktop engine as a portable board result.

### D2. Implement one benchmark protocol

Use configurable warm-up and measured iterations. Start development with 50 warm-up and 500 measured iterations per run; verify stability and freeze the chosen counts before comparative final experiments. These are initial project choices, not externally mandated counts. All primary latency measurements use batch 1.

Record model/checkpoint/engine hashes, input identifiers and shapes, precision, runtime/version, device, thread settings, seeds, timing scope, power/clock settings where applicable, warm-up/sample counts, and UTC timestamps with readable local dates where useful. GPU synchronization or correctly synchronized CUDA events must match the timing scope; do not measure asynchronous enqueue as completed inference.

Measure separately:

- Model-only execution on prepared tensors.
- The defined primary pipeline boundary from decoded host RGB to numerical depth output.
- Complete sustained video/demo throughput, including acquisition/rendering when enabled.

Save individual latencies and summary median/p95/p99, throughput calculation, warm-up policy and failures. Build repeated-run aggregation that treats repeated runs as independent units and does not invent independence between adjacent video frames. Use fixed input sequences and consistent caching boundaries for comparisons.

Record model file size, parameter count and MACs separately from execution memory. Measure desktop GPU peak memory where possible; board unified/system-memory accounting needs its own documented method. Prepare telemetry interfaces and offline log parsing without claiming board sensors have been validated.

**Gate D:** validated ONNX exports, supported desktop runtime checks, actual parity/accuracy records and a harness that generates raw measurements plus a traceable summary. Desktop results are useful preparation, not estimates promoted to Orin facts.

## 10. Phase E — Training, structured pruning and controlled ablations

This phase's tooling and initial experiments can proceed before board access. The final hardware-aware search needs the measurements in Phase G.

### E1. Build resumable training and fine-tuning

1. Use the verified training and tuning manifests. Start with a pretrained checkpoint when available; record any independently trained fallback as a separate experimental lineage.
2. Choose and record a valid-depth loss, optimizer, scheduler, batch size, seed, augmentations, number of steps/epochs and checkpoint-selection rule. Start with a masked regression loss and justify any additions.
3. Fit memory use to the RTX 3060's 6 GB through measured batch-size choice, optional mixed-precision training and gradient accumulation as needed. Do not imply a larger batch or full-dataset training actually occurred.
4. Verify a short training run, finite loss/gradients and a saved checkpoint. Resume with model, optimizer, scheduler, scaler if used, step/epoch and RNG state; confirm progress continues under the saved configuration.
5. Save train/tuning learning curves and loss/metric trajectories, throughput, elapsed time and resource usage. Use the same tuning criterion and comparable compute budget across controlled experiments.

### E2. Map structure and generate valid candidates

Inventory convolution input/output channels, depthwise groups, pointwise connections, normalization channels, decoder output dimensions and additive-skip dependencies. Removing a channel must update downstream inputs and coupled skip paths consistently. Store architecture configuration for every candidate rather than only weight tensors.

Start with a small preregistered candidate set and validate shape, finite outputs, actual parameter/MAC reduction, checkpoint save/load and ONNX export before running expensive fine-tuning. Weight masking alone does not create a structurally smaller dense model or establish faster inference.

On the desktop, use parameters/MACs and desktop latency only as preliminary cost indicators. Save candidates and accuracy curves now. Do not label their ranking Orin-optimal.

### E3. Complete Orin-guided selection when hardware is available

1. Measure relevant layer/subnetwork shapes on the board in the intended runtime/precision and construct a cost lookup table with measurement conditions and uncertainty.
2. Use a stated latency-reduction target to generate NetAdapt-style proposals while preserving structural dependencies.
3. Briefly fine-tune valid proposals on training data and compare tuning accuracy. Record every proposal, rejection, measured/estimated cost and selection.
4. Validate shortlisted candidates as complete TensorRT engines: fused runtime behavior means a sum of layer timings is only an estimate.
5. Repeat within a predeclared compute/iteration budget and stop on target attainment, plateau, accuracy constraint or resource limit. Save unsuccessful decisions as well as the winner.
6. Finish fine-tuning, freeze the final model using tuning data, and evaluate once under the final evaluation protocol. Compare with the paper's pruned checkpoint using the same runtime, split, preprocessing and timing scope.
7. Report whether the proposed 0.02 m accuracy allowance and lower measured Orin latency were achieved; an unsuccessful target remains a result.

### E4. Required ablations

At minimum compare the reference checkpoint, a structurally pruned candidate before fine-tuning, and the same candidate after fine-tuning. Include a matched fine-tuning-only control if attributing improvement specifically to pruning/search rather than extra training. Compare additional selection choices only when they are part of a claimed contribution; do not inflate the matrix beyond available compute.

Keep manifests, seeds where feasible, loss, training budget, runtime and evaluation settings controlled. Save the exact changed factor and limitations when perfect matching is impossible. Report architecture, accuracy, size and actual target latency for final ablations; show missing hardware cells explicitly.

### E5. Optional extensions, after required work

- Distillation: verify teacher license/output representation, generate teacher predictions only for training samples, implement scale/shift alignment with degenerate-case handling, and compare with a matched ground-truth-only baseline. Use tuning data for settings and final evaluation only after selection. A teacher used for training does not need to run on the edge device.
- INT8: Orin Nano supports INT8 acceleration, but actual model accuracy and speed must be measured. Prepare representative training-only calibration or explicit-quantization inputs; record calibration configuration and re-evaluate all depth metrics. Do not assume quantization improves depthwise-network latency.
- Additional heavy baseline: include only if it helps the research question and can be measured with compatible input/evaluation semantics. It is not needed merely because the poster example lists classical regression models.

**Gate E:** resumable training works, candidate generation preserves topology, controlled desktop ablations exist, and final selected-model claims await or include actual Orin evidence as appropriate.

## 11. Phase F — Board-ready packaging and desktop completion milestone

Create a handoff directory or documented artifact manifest containing ONNX models, hashes, preprocessing configuration, fixed input examples, expected outputs/tolerances, model registry/configuration, runtime commands, engine-build commands, benchmark configuration and result schema. Include a minimal error-diagnostic procedure.

Prepare a board setup guide for the chosen provisional JetPack stack, with actual install commands verified only when board access is possible. Use board-compatible PyTorch packages rather than desktop wheels. Keep TensorRT optional for unrelated tools; detect missing capabilities clearly.

Provide a run sequence: environment inventory → dependency/smoke check → engine build → output parity → image inference → small evaluation → benchmark → telemetry → full evaluation/video demo. Prepare scripts and parsers now; record which commands have only been syntax-checked or desktop-tested.

The maximum desktop-complete milestone is achieved when setup, dataset/EDA, trusted-weight inference, metrics, baseline accuracy, exports, desktop runtime checks, training/resume, preliminary candidates, ablation infrastructure, benchmark tools, report/poster source skeletons and deployment packaging work. At that point board deployment, board results and Orin-aware selection are still explicitly pending.

**Gate F:** the board session can begin from a documented manifest and known expected outputs rather than unfinished desktop application code.

## 12. Phase G — Orin Nano bring-up and measurement

### G1. Inventory and runtime verification

Record actual module and memory variant, carrier, OS, JetPack, CUDA, TensorRT, PyTorch, driver/runtime details, storage, camera if used, cooling, power supply, supported power mode and clocks. Choose a repeatable supported mode; do not copy original-Nano or TX2 sensor paths/commands blindly.

Install compatible dependencies, run the smoke inputs, build engines on the Orin and compare outputs with exported references. Record engine build configuration, workspace limits, precision, versions, warnings and elapsed build time. Build success alone does not establish correct accuracy.

### G2. Execute the required comparison matrix

For each supported original variant, collect PyTorch FP32, TensorRT FP32 and TensorRT FP16 measurements. Add the selected improved model using the same protocol. Evaluate accuracy from outputs actually produced by the board; either evaluate there or transfer raw predictions to the desktop using the identical evaluator.

Each supported cell needs model identity, accuracy, raw and summary timing, memory, power/energy, environment and any failure notes. If a cell cannot run, retain an explicit failed/unsupported row with the observed cause. Optional INT8 rows are labeled as additional experiments.

### G3. Power, memory and thermal methodology

- Discover and document available board input-power sensors and rails; save units and sample cadence. If sensors are unavailable, use a documented external input meter or mark power/energy unmeasured with the reason.
- Collect idle power under the same mode and background conditions, then sustained inference power and temperature over a long enough run for meaningful samples.
- Compute total energy/frame as integrated input power over the measured interval divided by completed frames. Compute idle-subtracted energy using the aligned idle estimate and the same interval. Use average power × time/frame only when those quantities cover the same sustained workload and boundary.
- State whether measurements cover the module, carrier or entire device input. Model-only energy and whole-device input energy must not share an unlabeled column.
- Record memory using a documented board method that respects unified memory; do not add CPU/GPU counters that double-count shared pages.
- Log thermal conditions, throttling indicators and clock changes. Report sustained performance rather than only a cold short burst.

**Gate G:** real board outputs agree within documented tolerances and the supported runtime matrix has traceable timing, accuracy and resource results. These measurements unlock final Orin-guided candidate selection in E3.

## 13. Phase H — Final evaluation, analysis and demonstration

Freeze the final baseline, reference and improved checkpoints/configurations based on tuning evidence. Rerun applicable final configurations **five times** on the Orin under the same protocol; preserve raw timings and telemetry. Record model hashes so posterior file edits cannot silently change a reported result.

Summarize median and tail latency with between-run variability or a documented confidence interval, accuracy from board outputs, model size, memory, input power, total/idle-subtracted energy and sustained conditions. Keep inference/pipeline/video timing scopes separate. Report whether the 33 ms p95 pipeline target and improvement constraint were met.

Evaluate at image/scene level where possible for uncertainty and error analysis. Dense pixels within one image are correlated; treating millions of pixels as independent experimental runs produces misleading confidence. Stratify errors by preregistered depth bins and inspect representative successes/failures with recorded IDs. Do not optimize the model based on final-set failure examples after freezing it.

For a demonstration, exercise recorded video first, then an available camera. Show model/runtime identity, a clearly defined live FPS counter and a consistent depth visualization. Record a backup video and a short reproduction guide. Do not label displayed relative colors as calibrated metric depth without numerical evidence. The live demo is optional in the supplied project plan; a working numerical image/video pipeline remains required for our runnable-project goal.

**Gate H:** final claims, plots, report and poster all refer to a frozen, traceable set of runs; failed targets and incomplete measurements are disclosed.

## 14. Tables, figures, algorithms and result records to produce

### 14.1 Compact evidence set

| Item | Content | Source evidence | Intended use |
| --- | --- | --- | --- |
| Dataset/split table | Source, counts, groups if known, units, validity, overlap policy | Audit and manifests | Methodology |
| Architecture/cost table | Four variants and proposed model, channels, parameters, MACs, size | Model configs and counting records | Methodology and comparison |
| Environment/protocol table | Desktop/board versions, batch, precision, repeats, timing, power mode | Environment and run metadata | Methodology |
| Main results table | RMSE, MAE, REL, delta1, R², p95 pipeline latency, model-only latency, size, memory and energy as space permits | Matching evaluation and benchmark runs | Results; shorter subset on poster |
| Ablation table | Reference, pruned before fine-tuning, pruned after fine-tuning, matched control if required | Controlled training/pruning runs | Results and contribution C5 |
| EDA panel | Depth distribution, validity coverage and examples | Dataset audit | Required EDA |
| Architecture/workflow figure | RGB → paired transforms → MobileNet encoder → decoder/skips → depth → runtime evaluation/pruning loop | Actual implementation | Methodology; poster Figure 1 |
| Accuracy–latency plot | Each model/runtime/precision, RMSE versus measured pipeline latency, 33 ms line | Joined results by model/config/hash | Main graphical analysis |
| Prediction/error panel | Fixed RGB/ground truth/reference/improved/error examples, shared units/scale | Frozen prediction outputs and masks | Qualitative findings |
| Error/fit panel | Depth-bin error and sampled predicted-versus-target scatter with identity line | Held-out outputs, documented sampling | Regression/graphical analysis |
| Training/ablation curves | Training/tuning loss or RMSE and structural/accuracy changes | Actual training logs | Training explanation; space permitting |
| Algorithm 1 | End-to-end model conversion, export, parity and evaluation workflow | Implemented pipeline | Report reproducibility |
| Algorithm 2 | Orin-guided proposal, short fine-tuning, tuning selection, complete-engine check, stopping | Actual pruning search | Report and compact poster algorithm |

Number and caption all included objects. A compact multi-panel figure is preferable to many tiny plots, but every panel needs readable labels, units and a description. Generate plots with standard plotting tools and export standalone PNG/PDF/SVG assets suitable for the final report; do not draw result curves manually.

### 14.2 Result schema and traceability

At minimum record: run ID, source commit, model/weight/config hashes, manifest hash, sample count, device/runtime/precision/version, input/output contract, aggregation/mask policy, metric values and units, warm-up/measured/repeat counts, timing boundary, raw-sample paths, power/memory method and failure status.

Use one stable summary-data source to generate report and poster tables. Each contribution sentence and claimed improvement must link through `results/evidence_index.csv` to its originating runs. Define speedup as baseline latency divided by compared latency under the same timing scope; define percentage error/energy change with the denominator stated. Do not compare a paper's TX2 timing directly with our Orin timing as if only software changed.

## 15. Final report: mandatory structure and page plan

### 15.1 Submission format and page budget

The supplied PDF requires **one PDF of 12–15 total pages, including all sections**. Target 14 pages. The detailed implementation plan can be much longer; it is not subject to the final report's page limit.

| Physical PDF pages | Required content | Planned length |
| --- | --- | --- |
| 1 | Title Page | 1 page |
| 2 | Abstract | 1 page |
| 3 | Table of Contents plus List of Figures and Tables | 1 page |
| 4 | Chapter 1: Introduction | 1 page |
| 5 | Chapter 2: Literature Review | 1 page |
| 6–9 | Chapter 3: Proposed Methodology | 4 pages |
| 10–12 | Chapter 4: Results and Discussion | 3 pages |
| 13 | Chapter 5: Conclusion and Future Scope | 1 page |
| 14 | References | 1 page |
| 15, only if needed | Appendix or reference overflow | At most 1 additional page |

Each main chapter begins on a new page and is numbered 1–5. Sharing page 3 between contents and lists is our space-saving layout choice; preserve both headings. A longer bibliography or required appendix must fit within page 15, not silently exceed the limit. Keep core methodology/results in the report; an external repository link is supporting material, not a replacement for required content.

### 15.2 Preliminary pages

Title page: concise title; verified author name, roll number, department, institution and date; advisor/course details only if known. Do not reuse placeholder identity from the sample poster.

Abstract: draft after experiments. Cover the application problem, FastDepth-based approach, dataset, Orin deployment/runtime comparison, actual measured outcomes and a material limitation. A working target of roughly 180–250 words is editorial guidance, not a word limit stated by the evaluator. Avoid claims based only on planned results.

Contents and lists: generate from actual numbered headings/captions. Ensure page labels match the final exported PDF. No dangling placeholders or unlisted major result figures.

### 15.3 Chapter 1 — Introduction

Include separately identifiable subsections for **Background and Motivation**, **Problem Statement**, **Objectives**, **Scope of the Work**, and **Your Contributions**. State RGB-to-metric-depth, indoor dataset scope, actual target hardware, why deployment cost matters and what is being measured.

State the 33 ms budget, runtime/precision comparison, pruning objective and the distinction between desktop preparation and final hardware evidence. Present C1–C5 only at their actual completion level, using at least three distinct evidence-backed contributions where achievable. Describe reproduction/engineering contributions honestly.

### 15.4 Chapter 2 — Literature Review

Include **Existing Approaches** and **Research Gaps**. Prepare a compact comparison covering FastDepth architecture/decoder/runtime, NetAdapt's measured-resource adaptation and any relevant depth-estimation reference actually used. Include Depth Anything V2 only if used as teacher/reference or clearly discussed as optional future context.

Cite the original FastDepth work for TX2 results and the upstream checkpoint metrics; do not imply it evaluated our Orin. Explain project-specific gaps: current-framework reproducibility, modern runtime portability and measured Orin channel/runtime trade-offs. Link each gap to an implementation or experiment instead of claiming that no prior efficient-depth research exists.

The literature table may use “not evaluated here” for externally reported systems. Do not invent missing benchmark values or turn unavailable code into a measured baseline.

### 15.5 Chapter 3 — Proposed Methodology

Include all six required subsections: **Dataset Description**, **Data Preprocessing**, **Exploratory Data Analysis (EDA)**, **Proposed Model Architecture**, **Algorithm**, and **Experimental Setup**.

Dataset/preprocessing/EDA: source and exact splits; training subset limits; units/masks; duplication checks; paired transforms and augmentation; required EDA panel and observations affecting interpretation.

Architecture: present our original diagram and the four variants plus the actual proposed pruned structure. Label model input/output shapes, MobileNet encoder, depthwise/pointwise decoder, nearest-neighbor upsampling and additive skips. State inherited FastDepth design versus channels/procedure changed by our project. Include cost conventions and structural coupling.

Algorithm: provide numbered pseudocode for the implemented export/evaluation workflow and pruning/search. Include inputs, outputs, tuning criterion, latency/accuracy constraint, structural checks and stopping condition. If using a simplified search rather than full NetAdapt reproduction, say so precisely.

Experimental setup: reproducible versions and hardware, seeds and manifests, training/fine-tuning settings, exporter/opset, runtimes/precision, benchmark warm-up/repeats/synchronization, timing scope, power/thermal conditions and uncertainty method. Record actual settings, not only intended defaults.

### 15.6 Chapter 4 — Results and Discussion

Include **Performance Evaluation Metrics**, **Experimental Results**, **Comparison with Existing Models**, **Graphical Analysis**, and **Discussion of Findings**.

Define numbered metric equations and units/aggregation. Show baseline accuracy/reproduction deviations, export parity, architecture/runtime/precision matrix, selected improvement, final repeated Orin results and controlled ablations. At minimum discuss RMSE, MAE, REL, delta1 and R²; use latency and resource metrics to substantiate the edge deployment objective.

Use the main comparison table, accuracy–latency plot, qualitative depth/error panel and a compact ablation/error panel. Choose a readable subset for the 3-page results budget; retain complete raw tables in the repository. Separate measured rows from cited external references, identify unsupported cells and give reasons.

Interpret which configurations meet the fixed budget, how much accuracy changes, whether size/MAC reduction became actual latency savings, how fine-tuning changes errors, whether FP16/optional INT8 affects accuracy, and where sustained power/thermal behavior limits conclusions. Discuss negative/unexpected outcomes and uncertainty, not only the best run.

### 15.7 Chapter 5 — Conclusion and Future Scope

Include **Key Findings**, **Limitations**, and **Future Research Directions**. Answer the objectives using final evidence. State achieved versus missed targets without changing their definitions.

Limitations may include checkpoint availability, reproduction mismatch, training-subset size, indoor-only domain, scene metadata limits, unavailable telemetry, optional runtimes and uncompleted hardware validation. Future directions should follow from those limitations: broader-domain evaluation, stronger calibration/quantization, distillation, longer controlled search or deployment robustness. Do not advertise future steps as delivered features.

### 15.8 References, appendix and formatting

Use consistent numbered citations and a bibliography that includes papers, upstream code/revision, dataset source and runtime/hardware documentation actually relied on. Verify author/title/year and URLs from primary sources when writing citations. Avoid unrelated references inherited from the poster sample.

Number figures, tables, equations and algorithms, for example Figure 3.1, Table 4.1, Equation (4.1) and Algorithm 3.1. Refer to each included object in the text and give units/caption context. Keep labels readable at final page size, avoid clipped formulas and ensure fonts render/are embedded in the exported PDF.

Appendix is optional and inside the total page limit. If used, include concise execution/configuration information or an artifact index; keep essential methodology and conclusions in the main chapters. Export one submission PDF and inspect `pdfinfo` page count, page breaks, contents/lists, references, figures and text extraction.

**Gate Report:** one readable 12–15 page PDF with all required sections, five separate numbered main chapters, 3–5 defensible contribution statements, required numbered material, correct citations and no invented results.

## 16. Report production workflow and evidence updates

Maintain [progress_report.md](progress_report.md) throughout implementation as the dated record of wins, failures, negative results, fixes and limitations. Link findings through `results/evidence_index.csv` to stable, fingerprinted evidence before latest-result files are overwritten. Preserve failed attempts and distinguish reconstructed session history from saved raw logs. Use this record to draft the methodology, discussion and limitations; keep the immediate-priority file as the current queue rather than a second history.

Create the report source and bibliography skeleton early with the required headings and explicit “pending experiment” markers. Fill dataset/EDA/environment/methodology subsections as their evidence is verified. Reserve result fields until runs exist, then generate tables/plots from the shared summary data.

After final model freeze, write Chapter 4 discussion, Chapter 5 conclusions and the abstract. Check each contribution against its evidence index. Compress the report to the page plan by removing duplicated exposition and choosing compact figures; do not remove mandatory EDA, comparisons, limitations or reproducibility essentials.

Run a final consistency audit: sample counts, split names, units, checkpoint/model names, timing boundaries, precision, power mode and numerical improvements agree across source, figures, report, README and poster. Keep a single final submission PDF and clearly named editable source; do not leave reviewers guessing which draft is final.

## 17. Poster plan based on the supplied sample

### 17.1 Layout to preserve and adapt

The inspected sample is one 16:9 landscape slide, approximately 13.33 × 7.5 inches in slide coordinates. This is its slide aspect/size, not a supplied print-paper requirement. Its layout uses a full-width top banner, three body columns, and a full-width references footer. The middle methodology column is wider than the side columns; the sample uses light blue, light neutral and light green body backgrounds with a warm header/footer and dark borders.

Use that structure with balanced spacing. The sample's dense text and occasional clipping are not requirements to reproduce. Keep content readable, avoid overflow, and check both the editable PPTX and the exported PDF. Preserve institutional logos only as supplied and appropriate to the user's actual institution; verify title/identity rather than carrying over the sample author placeholders.

| Region | Our planned content | Evidence/source |
| --- | --- | --- |
| Full-width header | Short project title, author, roll number, contact, department/institute, appropriate logos | Verified identity and actual title |
| Left column: Abstract | Brief problem, method, dataset, device and strongest actual result | Final report abstract condensed |
| Left column: Introduction | Embedded depth motivation and why accuracy/latency/resources matter | Chapter 1 and numbered citations |
| Left column: Problem Statement | RGB-to-depth on Orin under the declared latency and accuracy constraints | Frozen objective/protocol |
| Middle column: Key Contributions | 3–5 concise completed contributions, aligned with C1–C5 | Evidence index; no pending claims presented as achieved |
| Middle column: Methodology | Large Figure 1 showing real RGB/preprocessing/model/pruning/export/Orin flow | Actual architecture and pipeline |
| Middle column: Algorithm | Short numbered pruning/selection pseudocode with inputs, constraint and stopping rule | Actual implemented algorithm, not photonic optimization text |
| Right column: Result and Discussion | Table 1 comparing baseline/reference/proposed and a small number of readable plots | Frozen summary data with same units/protocol as report |
| Right column: Conclusion | Whether targets were met, measured trade-off and one material limitation | Final board evidence and Chapter 5 |
| Full-width footer: References | A few relevant numbered references cited in poster text | FastDepth, NetAdapt and other actually used sources |

### 17.2 Poster result choices

Use a compact comparison table with model/runtime/precision, RMSE, delta1, p95 pipeline latency and energy/frame if measured. Include MAE/R² in the table only if still readable; they remain required project/report metrics and can appear in a compact secondary panel. Do not copy FOM, adjusted R², photonic input variables or model families from the example.

Prefer an accuracy–latency plot with the 33 ms line plus a qualitative RGB/ground-truth/prediction/error strip. Add a small ablation panel only if it improves clarity. Clearly label device, batch, timing boundary and units in compact captions. Replace text-heavy explanations with the actual architecture/algorithm visual where practical.

Write the poster after key results are frozen, using the same generated numbers as the report. If board access is still missing at a review, label the poster a desktop-progress version and mark Orin deployment/results pending. Do not visually imply a board experiment occurred.

### 17.3 Poster deliverables and acceptance

Produce `deliverables/poster/fastdepth_poster.pptx` and a PDF export, with standalone figure assets retained. Render the exported poster and inspect normal viewing size: title, labels, table, equations/algorithm, references and all three columns must be legible with no clipping, stretched graphics, overlapping text, placeholders or borrowed example numbers.

The report remains its own single 12–15 page PDF; the poster is a separate artifact. A backup demo clip and optional presentation slides are supporting project material, not additions to the report page count.

**Gate Poster:** the sample's content organization is recognizable, all factual claims match final evidence, and editable/PDF versions are visually sound.

## 18. Dependency order, blockers and fallback decisions

### 18.1 Execution order while hardware is unavailable

1. Stabilize setup and start trusted checkpoint acquisition (A).
2. Independently audit data, implement paired preprocessing and create EDA (B).
3. Implement/test metrics, load available models and complete baseline image/video inference and evaluation (C).
4. Export/validate models and build the desktop benchmark/result pipeline (D).
5. Prepare training data, resumable fine-tuning, valid candidate generation and controlled preliminary ablations (E desktop portion).
6. Package the board handoff and complete report/poster methodology skeletons/assets (F, sections 15–17).
7. On board access, run bring-up/baselines (G), use those measurements for final selection (E3), then freeze and repeat final experiments (H).
8. Generate final report, poster and any demo from frozen evidence, then run compliance checks.

The immediate-priority file holds at most five current tasks from this order. It is not an instruction to execute every phase during a documentation-only task.

### 18.2 Risk handling

| Risk / observed blocker | Next action | Claim or scope consequence |
| --- | --- | --- |
| Original checkpoint server times out | Bounded retry, trustworthy mirror search, provenance record; continue independent work | Original accuracy reproduction remains pending; training-from-scratch is separately labeled |
| Legacy checkpoint cannot load | Inspect trusted serialization, wrapper names and actual topology; convert with logged compatibility changes | Do not silently replace weights/architecture or mark the variant supported |
| 654 files differ from standard split | Compare sample identities/metadata and document difference; seek matching manifest | Reference-accuracy comparison is qualified until compatibility is established |
| Training data or GPU memory is insufficient | Document subset, reduce batch, use accumulation, measure resource needs before optional cloud fallback | Report actual training budget and subset; no full-data claim |
| Model accuracy differs from the paper | Audit units, transforms, masks, aggregation and weights | Explain discrepancy rather than tuning final-evaluation preprocessing to chase a number |
| TensorRT rejects export | Match documented stack/opset, inspect operators, patch equivalently and rerun parity | Unsupported cells stay visible; desktop build is not board validation |
| Pruning reduces MACs but not Orin latency | Validate complete engines and revise candidates using actual board measurements | Report negative result and limitations; no speed claim from MACs alone |
| Board remains unavailable | Complete desktop milestone and evidence package; maintain pending board gates | Report/poster identify missing deployment evidence; full hardware objective is not marked achieved |
| Power rail or sensor access is unavailable | Verify available telemetry or use documented external meter | Energy remains unmeasured if no defensible input-power measurement exists |
| Baseline already meets 33 ms | Retain target and quantify measured efficiency changes | No invented struggle narrative or post-hoc target change |
| Report exceeds 15 pages | Apply page budget and compact readable panels, keep appendix within limit | Required sections remain; external files do not substitute for mandatory content |
| Fewer than three defensible contributions exist | Reassess completed engineering/experimental work and finish missing evidence | Do not claim rubric compliance until the contribution requirement is supported |

## 19. Final completion checklist

This checklist tracks the consolidated plan. Update statuses with artifact paths/run evidence as work is completed. Do not infer completion from old dates or from a task appearing in the plan.

- [x] Deployment scope confirmed as Jetson Orin Nano; desktop-first work while hardware is unavailable.
- [x] Supplied evaluator PDF and poster sample inspected; required report content and sample organization mapped above.
- [x] Independent desktop installation, fresh-source setup, all five runtime checks, dependency lock, legacy-environment preservation and cache cleanup verified; see [Phase A results](phase_a_results.md).
- [x] Validation files, audit, byte comparison to the pinned mirror archives and a limited source-level preprocessing parity check exist. Independent MIT archive identity, historical binary parity, mirrored unit-conversion chain and evaluation scene identities remain unresolved.
- [x] Gate A1: reproducible independent setup verified, including fresh-source installation and post-cache-cleanup runtime checks.
- [ ] Gate A2 fully complete: all trusted original checkpoints available/authenticated. Partial evidence exists for one labeled community pruned candidate and conversion tooling; original endpoint access and the remaining pretrained variants are pending.
- [ ] Gate B: dataset/manifests/preprocessing/EDA verified.
- [ ] Gate C: tested metrics, baseline accuracy, model cost and pretrained image/video inference work.
- [ ] Gate D: ONNX/parity, desktop runtime checks and raw-sample benchmark harness work.
- [ ] Gate E desktop portion: training/resume, valid structural candidates and preliminary ablations work.
- [ ] Gate F: board handoff and expected-output package ready.
- [ ] Gate G: Orin software/runtime/output verification and initial measurements complete.
- [ ] Gate E final portion: Orin-guided candidate selection and target outcome evaluated.
- [ ] Gate H: frozen configurations, five repeated final board runs and error/uncertainty analysis complete.
- [ ] At least three distinct completed contributions, targeting C1–C5, linked to evidence.
- [ ] Report contains all preliminary pages, required five chapters/subsections, references and optional appendix within 12–15 pages.
- [ ] All report figures, tables, equations and algorithms are numbered, captioned and referenced; citations resolve.
- [ ] Poster follows the supplied organization, with original project content, readable architecture/algorithm/results and verified identity.
- [ ] Report/poster numbers, scope, units, model identities, claimed improvements and limitations agree with saved runs.
- [ ] Final PDFs visually inspected, report page count checked, editable sources and figure-generation commands preserved.
- [ ] Runnable commands, artifacts, model/data provenance, licenses and Git LFS tracking documented; priority queue updated to the actual remaining work.
