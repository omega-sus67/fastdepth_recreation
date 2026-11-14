# Project progress report and experiment history

Last maintained: 14 November 2026 (Asia/Kolkata). Latest runtime/model evidence: 14 November 2026. Project: FastDepth on Jetson Orin Nano.

This is the persistent record of what worked, what failed, what changed, and what remains unsupported. Use it to write the methodology, results, discussion and limitations of the final report. The [complete plan](fastdepth_edge_ai_plan.md) defines intended work; [immediate priorities](immediate_priority.md) define the next actions. This report records actual history rather than duplicating that queue.

## How to read and maintain this record

- Give significant events stable IDs (`PR-001`, `PR-002`, …), an event date, status, objective, observation, response, validation, evidence and report relevance. Keep failed attempts when a later attempt succeeds; update their resolution and link follow-up evidence.
- Distinguish engineering/tooling failures, experimental negative results, inspection-found risks and external blockers. A fixed script error is not a model ablation or scientific finding.
- Preserve numerical results with units, scope, model/runtime/device, sample count or input contract, and aggregation/tolerance. Link raw evidence instead of reconstructing numbers from memory. Update the [evidence index](../results/evidence_index.csv) when claims or supporting files change.
- Separate **measured**, **historical metadata**, **reconstructed session history**, **planned**, and **unverified**. If a log is missing, say so. Record a failed hypothesis or unsupported configuration with its reason instead of removing it from comparisons.
- Dates below describe the events, not the date of this document's creation. Do not relabel a 12 November check as a new 13 November experiment. Later contradictory evidence should supersede the earlier conclusion visibly, not erase it.
- Report contribution claims only at their demonstrated level. Preserve the distinction between synthetic runtime checks, random-weight model checks, pretrained inference, dataset accuracy, desktop timing and Orin measurements.

The initial entries were compiled on 13 November from saved artifacts and the 12 November work session. Existing evidence was copied into a [preserved Phase A snapshot](../results/history/2026-11-12-phase-a/snapshot_manifest.json), with SHA-256 fingerprints; this did not rerun any experiments. Measurement links below use that snapshot so future latest-result files can change without erasing the baseline. Entries explicitly labeled **reconstructed** retain the observed failure/response, but their original error output was not saved as a standalone project log. The linked implementation and final passing artifacts establish the resulting state, not a preserved transcript of the failed run.

## Current outcome summary

| Area | Latest demonstrated state | What this permits us to say | Outstanding limitation |
| --- | --- | --- | --- |
| Scope | Orin Nano deployment; RTX 3060 Laptop GPU desktop development | The implementation targets Orin Nano and supports desktop preparation | No board runtime, latency, power or thermal results |
| Phase A1 | Complete; independent Python 3.12.3 environment, 70 pinned packages | Desktop setup was reproduced in a clean source copy and new environment | Used a warm download cache; this was not an independent-user or offline reproduction |
| Runtime readiness | FastDepth ONNX and desktop FP32/FP16 TensorRT paths tested | One-sample export parity, desktop timing and FP16 validation metrics recorded | No Orin measurements or repeated-run uncertainty |
| Phase A2 | Partial; one community pruned checkpoint recovered and converted | A concrete pruned model loads and its conversion preserves tested outputs | Original-author authentication and the other three pretrained variants are pending |
| Model validation | Twenty-four tests pass; provisional 654-file evaluation and multi-seed 445-image tuning comparison recorded | Model shape, conversion, metrics and tested resize cases checked | Mirror conversion chain, complete split identity, original checkpoint pretraining and historical binary-environment parity remain unverified |
| Data | 654 evaluation files and a new 1,857-train/445-tune sample across 124 named scenes | Source manifests, conservative grouping, exact checks and a coarse visual-similarity screen recorded | Evaluation scene identities remain unknown; bounded-window sample and perceptual hashes cannot establish complete independence |
| Training and improvement | Scene-aware, average, weight and random pruning compared after three-seed deterministic recovery | Paired tuning results and negative/near-tie findings are recorded | Original pretraining may overlap tuning; no reliable superiority or Orin result; FP16 runtime has individual output outliers |
| Report/poster | Evaluator requirements mapped into the consolidated plan | Submission content and evidence requirements are explicit | Final submissions and main experimental results have not been produced |

All “latest” statements refer to the dated evidence below. The official server was not rechecked merely to create this report.

## Verified wins and measurements

| Result | Recorded value | Scope and interpretation | Evidence |
| --- | --- | --- | --- |
| Independent installation | 70 pinned packages; no external site-package paths detected | Newly created `.venv-desktop` in a clean source snapshot; warm download cache | [Fresh installation](../results/history/2026-11-12-phase-a/results/environment/fresh-install.json) |
| PyTorch CPU/CUDA | Passed | Synthetic CPU convolution and GPU matrix calculation on RTX 3060; not FastDepth timing | [Runtime checks](../results/history/2026-11-12-phase-a/results/environment/desktop-environment.json) |
| ONNX Runtime CPU | Maximum absolute error `0.0` | Synthetic addition graph against its expected output | [Runtime checks](../results/history/2026-11-12-phase-a/results/environment/desktop-environment.json) |
| Modern ONNX export | Maximum absolute error `1.1920928955078125e-07` | Synthetic convolution exported with the dynamo exporter, opset 18, compared against PyTorch | [Runtime checks](../results/history/2026-11-12-phase-a/results/environment/desktop-environment.json) |
| TensorRT FP32 | Maximum absolute error `0.0` | A synthetic engine was built and executed on the desktop GPU | [Runtime checks](../results/history/2026-11-12-phase-a/results/environment/desktop-environment.json) |
| Preserved legacy environment | Package snapshot unchanged; CUDA matrix value `16.0` as expected | Earlier local/shared environment remains usable; no claim of a byte-by-byte audit of all shared files | [Preservation check](../results/history/2026-11-12-phase-a/results/environment/legacy-environment-check.json) |
| Converted pruned topology | `1,360,786` parameters | Actual recovered model, with per-layer channel configuration preserved | [Conversion record](../results/history/2026-11-12-phase-a/results/checkpoints/community-pruned-conversion.json) |
| Converted versus legacy output | Maximum absolute error `0.0`; converted reload bit-exact | Recorded synthetic batch-1 input and the same runtime/device | [Conversion record](../results/history/2026-11-12-phase-a/results/checkpoints/community-pruned-conversion.json) |
| Upstream example comparison | Mean absolute difference `0.00033575919223949313`; maximum `0.0036644935607910156` in the example depth scale | One supplied RGB/TVM prediction example; passes `rtol=1e-3, atol=1e-3`; not a held-out dataset metric or identity authentication | [Model checks](../results/history/2026-11-12-phase-a/results/checkpoints/model-checks.json) |
| Model/loader tests | `8 passed` | Architecture outputs, topology round trip, hash/global/type guards and explicit pruned-topology requirement | [Test output](../results/history/2026-11-12-phase-a/results/environment/model-tests.txt) |
| Cache cleanup | `16,086,982,656` observed additional free bytes, approximately `14.98 GiB` | Filesystem free-space change after pip/uv/source-cache cleanup; not an inference memory reduction | [Cleanup record](../results/history/2026-11-12-phase-a/results/environment/cache-cleanup.json) |

The initial cleanup reading was approximately 14.96 GiB; removing the remaining disposable pip cache directory tree updated the final saved value to approximately 14.98 GiB. Use the final saved measurement consistently. Cache directory totals and reclaimed filesystem space differ because of hardlinked installed package files.

The three unpruned **random-weight** architecture checks record parameter counts of `20,668,194` (NNConv5), `3,960,930` (depthwise NNConv5), and `3,960,930` (depthwise skip-add). These describe the tested architectures, not pretrained accuracy or measured speedups. See [model checks](../results/history/2026-11-12-phase-a/results/checkpoints/model-checks.json).

## Dated progress, failures and responses

### PR-001 — Deployment scope and evaluation requirements

- **Event date:** 12 November 2026. **Status:** decision implemented; experimental deliverables pending.
- **Objective:** align implementation with the actual target hardware and supplied evaluator documents.
- **Observation:** the earlier plan assumed the original Nano and multiple boards; the user specified Orin Nano only. The evaluator requires a single 12–15 page report, five numbered chapters, explicit EDA, comparisons, graphical analysis and 3–5 contribution statements. The poster sample provides a landscape three-column organization.
- **Response:** consolidated the implementation, experiments, report and poster work into one plan; retained the existing phase paths as navigation pages; separated board-only evidence from desktop development.
- **Evidence:** [plan and requirements mapping](fastdepth_edge_ai_plan.md), [report guidelines](Machine_Learning_Project_Report_Guidelines_Edited.pdf), [poster reference](updated%20poster%20sample.pptx). Documentation baseline commit: `075481c`.
- **Report use:** Chapter 1 scope/objectives; Chapter 3 experimental boundaries. This is planning alignment, not a completed scientific contribution.

### PR-002 — Working shared setup was insufficient for independent reproduction

- **Event date:** 12 November 2026. **Status:** resolved for the tested desktop path.
- **Objective:** make the project installable without depending on another project's Python packages.
- **Initial limitation:** the working `.venv` loaded PyTorch/torchvision from `/home/omega_sus/droidrun/venv`, and the historical requirements snapshot did not describe that active installation.
- **Inspection finding:** setup used the presence of a nested upstream `.git` directory to decide whether to clone, even after source vendoring removed that directory. This was a discovered setup risk, not a saved demonstration of a clone failure.
- **Response:** use an independent `.venv-desktop`, pin the full tested dependency set, recognize vendored `models.py`, and leave the legacy/shared environment intact.
- **Validation:** clean-source/new-environment installation of 70 packages; all runtime checks passed. The prior package snapshot remained unchanged and its CUDA check passed.
- **Evidence:** [fresh installation](../results/history/2026-11-12-phase-a/results/environment/fresh-install.json), [dependency lock](../requirements-desktop-lock.txt), [setup implementation](../scripts/setup_desktop.sh), [legacy check](../results/history/2026-11-12-phase-a/results/environment/legacy-environment-check.json).
- **Report use:** Chapter 3 reproducibility/setup and C1 supporting work. Installation used a warm download cache; no fresh external machine was tested.

### PR-003 — Official checkpoints inaccessible; alternatives require provenance labels

- **Event date:** 12 November 2026. **Status:** original download blocker open; partial recovery achieved.
- **Objective:** obtain the four original pretrained models for controlled reproduction.
- **Failure observed:** all four exact original checkpoint URLs timed out over both HTTP and HTTPS in the saved attempt record.
- **Investigation:** inspected upstream issue comments, the listed forks and larger fork file trees, Hugging Face cards, a converted PINTO archive and a Wayback CDX query. Some alternatives were separately trained reimplementations or converted deployment assets, so they were not accepted as interchangeable original checkpoints.
- **Win:** recovered one pruned checkpoint from an upstream fork at immutable commit `bca2e09002f4e214f76b024881988967b8561865`; recorded SHA-256 and preserved the original file.
- **Unresolved:** no original-author checksum or checkpoint-specific license statement was established. The recovered model remains `community_mirror_unverified`; the three unpruned pretrained checkpoints remain missing.
- **Evidence:** [official request failures](../results/history/2026-11-12-phase-a/results/checkpoints/original-source-attempts.json), [source investigation](../results/history/2026-11-12-phase-a/results/checkpoints/source-search.json), [acquisition record](../results/history/2026-11-12-phase-a/results/checkpoints/community-acquisition.json), [source manifest](../configs/checkpoints.json).
- **Report use:** Chapter 3 checkpoint sourcing and Chapter 5 reproduction limitations. Do not write that all original pretrained variants were reproduced.

### PR-004 — Legacy PyTorch checkpoint could not use the modern default loader

- **Event date:** 12 November 2026. **Status:** engineering failure resolved for the recovered candidate. **History basis:** reconstructed; initial raw error log was not preserved.
- **Observed failure:** modern `weights_only=True` loading rejected the legacy persistent module format with `Only persistent_load of storage is allowed, but got module`.
- **Cause established for this file:** PyTorch 0.4-era serialized model/module IDs and obsolete backend/metadata objects need compatibility handling; the model's pruned channel dimensions also cannot be replaced by unpruned defaults.
- **Response:** checksum enforcement and a restricted class/function allowlist for legacy conversion, explicit per-layer topology reconstruction, and a modern config/state-dictionary artifact loaded with `weights_only=True`. Unknown globals and unsupported types are rejected.
- **Win:** recovered topology contains 1,360,786 parameters; converted/legacy outputs match exactly on the recorded input, converted reload is bit-exact, and valid depth-map shapes are produced.
- **Evidence for resolution:** [adapter](../src/fastdepth_project/models.py), [converter](../scripts/convert_checkpoint.py), [conversion results](../results/history/2026-11-12-phase-a/results/checkpoints/community-pruned-conversion.json), [guard/round-trip tests](../tests/test_models.py).
- **Report use:** Chapter 3 implementation/compatibility and C1 supporting evidence. Exact output parity does not authenticate checkpoint origin or establish dataset accuracy.

### PR-005 — Inherited ROS Python paths interfered with project tests

- **Event date:** 12 November 2026. **Status:** engineering failure resolved for the wrapper/setup paths. **History basis:** reconstructed; initial raw error log was not preserved.
- **Observed failure:** pytest automatically discovered a ROS `launch_testing` plugin from `/opt/ros/humble/lib/python3.10/site-packages` and failed on a missing `lark` dependency, despite the project using Python 3.12.
- **Diagnosis:** an activated virtual environment alone did not remove inherited `PYTHONPATH`; this was environment contamination, not a missing FastDepth requirement.
- **Response:** added a project Python runner that sets only the project's `src` import path and disables user site packages; setup clears the inherited Python path. Unrelated ROS dependencies were not added to the project.
- **Validation:** final and fresh-source reports show no external site-package paths, and all eight project tests pass.
- **Evidence for resolution:** [runner](../scripts/python.sh), [setup](../scripts/setup_desktop.sh), [runtime isolation](../results/history/2026-11-12-phase-a/results/environment/desktop-environment.json), [tests](../results/history/2026-11-12-phase-a/results/environment/model-tests.txt).
- **Report use:** Chapter 3 environment reproducibility; include briefly if space permits rather than treating it as a model result.

### PR-006 — TensorRT installation and execution required several corrections

- **Event date:** 12 November 2026. **Status:** engineering failures resolved for the synthetic desktop FP32 check. **History basis:** reconstructed for intermediate errors; final execution is persisted.
- **Failed attempts:** installing bindings/libraries alone left `import tensorrt` unavailable because the facade package was missing. The older facade's build hook launched a nested pip installation, causing duplicate large runtime downloads. The first locked-source path also needed the explicit NVIDIA wheel index. Finally, the checker attempted `len()` on TensorRT `IHostMemory`, which this API does not support.
- **Responses:** install facade/bindings/libraries explicitly; set `NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP=1`; include the NVIDIA index in locked installation; obtain serialized size through the buffer's bytes representation.
- **Win:** clean-source installation succeeded; TensorRT 10.3.0 built and executed the synthetic FP32 engine on the RTX 3060, with zero maximum absolute error for that check.
- **Evidence for resolution:** [runtime requirements](../requirements-tensorrt.txt), [setup](../scripts/setup_desktop.sh), [checker](../scripts/check_environment.py), [fresh installation](../results/history/2026-11-12-phase-a/results/environment/fresh-install.json), [final runtime results](../results/history/2026-11-12-phase-a/results/environment/desktop-environment.json).
- **Report use:** Chapter 3 setup/implementation. The success is not a FastDepth engine export, FP16 validation, performance benchmark or Orin deployment.

### PR-007 — RGB layout error in the upstream-example comparison

- **Event date:** 12 November 2026. **Status:** engineering failure resolved for the supplied example. **History basis:** reconstructed for the failed call.
- **Observed failure:** the vendored RGB example is HWC (`224 × 224 × 3`); adding a batch dimension without transposing produced a channel mismatch in a convolution expecting three input channels.
- **Response:** explicitly transpose HWC to CHW, create a contiguous float tensor and then add the batch dimension.
- **Win:** the recovered model's output agrees with the bundled TVM prediction under the saved tolerance; exact differences are in the verified-measurement table above.
- **Evidence for resolution:** [comparison implementation](../scripts/check_models.py), [saved example comparison](../results/history/2026-11-12-phase-a/results/checkpoints/model-checks.json).
- **Report use:** Chapter 3 input contract and a compatibility check. It reinforces the need for verified shared preprocessing; it does not complete validation-set preprocessing or depth-unit verification.

### PR-008 — Storage recovery without removing project artifacts or installed runtimes

- **Event date:** 12 November 2026. **Status:** engineering maintenance complete for the recorded cleanup.
- **Objective:** recover disposable cache space after installing/testing the independent GPU stack.
- **Observation:** pip/uv cache sizes grew during installation; cache totals were not equal to reclaimable disk space because of hardlinks to installed files.
- **Response:** cleared pip/uv and source/test caches, removed the disposable fresh-install environment after saving its result, and preserved datasets/checkpoints/papers/Git/LFS data and both working environments.
- **Validation:** the saved final cache-cleanup reading shows approximately 14.98 GiB observed additional free space; all runtime checks and eight tests passed afterward.
- **Evidence:** [cleanup record](../results/history/2026-11-12-phase-a/results/environment/cache-cleanup.json), [post-cleanup runtime result](../results/history/2026-11-12-phase-a/results/environment/desktop-environment.json), [tests](../results/history/2026-11-12-phase-a/results/environment/model-tests.txt).
- **Report use:** reproducibility/storage notes or appendix if useful. This is host disk maintenance, not an optimized model-memory or energy result.

### PR-009 — Progress history established for report writing

- **Event date:** 13 November 2026. **Status:** documentation maintenance implemented.
- **Objective:** preserve wins, failed approaches, fixes and limitations so the final report can be evidence-based.
- **Work:** created this history, linked report-ready claims to artifacts, distinguished reconstructed failures from saved results, added an evidence index and required future maintenance in the agent rules.
- **Evidence:** this document, [evidence index](../results/evidence_index.csv), and [working rules](../agents.md).
- **Report use:** an internal writing source. This documentation work is not a new model experiment or independent scientific contribution.

### PR-010 — Validation audit and provisional baseline

- **Event date:** 13 November 2026. **Status:** implementation and full local evaluation completed; reproduction identity and units remain unverified.
- **Objective:** audit local HDF5 records, implement paired validation preprocessing and calculate independent metrics for the community model.
- **Data:** 654/654 files read; one schema; no exact file-hash duplicates or read errors. All raw depth values were positive/finite; range 0.713300–9.986646 native units. File hashes do not prove scene independence.
- **Implementation:** nearest-neighbor Pillow adaptation of the upstream resize/crop/resize sequence; pooled valid-target metrics and mean per-image RMSE/delta1. Thirteen tests passed after adding metric and preprocessing checks. Historical SciPy parity is still unverified.
- **Result:** community checkpoint `8d7f7bf2359f33dac8e1046d21e4fdcfafabb8099addd4272a7642d63dd6c1ca`, RTX 3060 CUDA, 654 samples: pooled RMSE 0.721159, MAE 0.428963, REL 0.165163, delta1 0.771065, R² 0.719858; mean image RMSE 0.603737 and mean image delta1 0.771065. There were 32,815,104 valid transformed targets and 8,884 nonpositive predictions. Values use native, unverified depth units.
- **Interpretation:** mean per-image RMSE and delta1 are close to historical checkpoint metadata; pooled RMSE differs because the aggregation differs. This does not authenticate the checkpoint or establish standard split identity.
- **Evidence:** [evaluation summary](../results/evaluation/2026-11-13-community-baseline/summary.json), [per-image data](../results/evaluation/2026-11-13-community-baseline/per_image.csv), [audit summary](../results/data/2026-11-13-validation-audit/summary.json), [histogram](../results/data/2026-11-13-validation-audit/depth_distribution.png), [metrics explanation](learning/metrics.md).

### PR-011 — Community model ONNX export

- **Event date:** 13 November 2026. **Status:** one-sample desktop CPU parity passed.
- **Method:** export recovered model to ONNX opset 18 and compare PyTorch CPU with ONNX Runtime CPU on local validation sample `00001.h5`.
- **Result:** maximum absolute difference 3.5762786865234375e-06, below 1e-4; output shape 1 × 1 × 224 × 224.
- **Evidence:** [parity result](../exports/2026-11-13-community-pruned/parity.json), [ONNX model](../exports/2026-11-13-community-pruned/model.onnx).
- **Limitations:** one input only; full validation parity, TensorRT accuracy and Orin execution remain pending.

### PR-012 — Desktop PyTorch model-only baseline

- **Event date:** 13 November 2026. **Status:** exploratory desktop benchmark completed.
- **Method:** RTX 3060 Laptop GPU, FP32, batch 1, preallocated zero input, 50 warm-ups and 500 synchronized calls.
- **Result:** median 5.404786 ms, p95 5.663535 ms, p99 5.929362 ms; raw individual measurements retained.
- **Evidence:** [summary](../results/benchmarks/2026-11-13-desktop-pytorch/summary.json), [raw samples](../results/benchmarks/2026-11-13-desktop-pytorch/latencies_ms.csv).
- **Limitations:** a single desktop run with a zero tensor; excludes preprocessing and transfers. It is neither an Orin measurement nor the end-to-end 33 ms target.

### PR-013 — Desktop TensorRT FP32/FP16 comparison

- **Event date:** 13 November 2026. **Status:** exploratory desktop comparison completed; board evaluation pending.
- **Method:** both engines built on RTX 3060 from the same ONNX model, batch 1, zero tensor, 50 warm-ups, 500 synchronized model-only calls. FP16 flag allows mixed precision. The FP32 engine's default TF32 setting was not recorded.
- **Result:** TensorRT FP32 median 0.520262 ms, p95 0.549775 ms; FP16-enabled median 0.391760 ms, p95 0.409146 ms. The runtime printed a default-stream synchronization warning; that warning is reconstructed from console output, not a saved standalone log.
- **Accuracy:** the FP16 engine used the same 654-file manifest as PR-010. Mean per-image RMSE 0.603775 versus PyTorch FP32 0.603737; delta1 0.771055 versus 0.771065. Pooled RMSE 0.721116 versus 0.721159. Depth units and split identity remain unverified.
- **Evidence:** [FP32 summary](../results/benchmarks/2026-11-13-desktop-tensorrt-fp32/summary.json), [FP16 summary](../results/benchmarks/2026-11-13-desktop-tensorrt-fp16/summary.json), [FP16 evaluation](../results/evaluation/2026-11-13-community-tensorrt-fp16/summary.json).
- **Limits:** one run each, zero tensor, default stream; no repeated-run uncertainty, Orin latency/power, end-to-end target or guaranteed all-FP16 execution.

### PR-014 — Image and recorded-video inference

- **Event date:** 13 November 2026. **Status:** desktop image inference and three-frame generated-video smoke test passed; webcam remains pending.
- **Method:** same community checkpoint and preprocessing; image output includes numerical depth and normalized preview. Video resizes each input frame to 640 × 480, applies per-frame preview normalization, writes MP4 and omits audio.
- **Evidence:** [image metadata](../results/inference/2026-11-13-sample/metadata.json), [video metadata](../results/inference/2026-11-13-video-smoke/metadata.json), [image script](../scripts/infer_image.py), [video script](../scripts/infer_video.py).
- **Limits:** aspect ratio can change for video; preview color is relative per frame; the three-frame throughput is a smoke test, not a stable performance measurement.

### PR-015 — Bounded training-data acquisition timed out

- **Event date:** 13 November 2026. **Status:** partial download, unresolved.
- **Method:** stream the first Hugging Face NYU training shard with a 768 MiB read budget and collect 16 examples from each of two distinct scenes.
- **Result:** network read timed out after 49,070,080 bytes; 16 files from `basement_0001a` were saved, no second scene. The saved manifest contains the timeout message; original stack trace was not captured.
- **Evidence:** [subset manifest](../data/nyu_depth_v2/preliminary_2026-11-13/manifest.json), [download implementation](../scripts/download_training_subset.py).
- **Limit:** one scene cannot supply scene-disjoint training and tuning; evaluation scene overlap remains unknown.

### PR-016 — Correlated training/resume smoke test

- **Event date:** 13 November 2026. **Status:** smoke test and resume passed; not a final training result.
- **Method:** 12 training and four tuning examples from the same preliminary scene; Adam, learning rate 1e-5, seed 42, masked MAE; exact file hashes checked against local validation files, scene overlap unverified.
- **Result:** first epoch training/tuning MAE 1.01010/0.65698; after resuming for epoch 2, 0.95651/0.91595 in native unverified units. The tuning loss increased. No claim about pruning improvement or generalization is supported.
- **Evidence:** [first epoch](../results/training/2026-11-13-correlated-smoke/summary.json), [resume](../results/training/2026-11-13-correlated-resume/summary.json), [training code](../scripts/train_smoke.py).
- **Limit:** tiny same-scene split, unknown near-duplicate/scene overlap with evaluation, and resume equivalence versus an uninterrupted run has not been tested.

### PR-017 — Reported Orin software and decision explanations

- **Event date:** 13 November 2026. **Status:** screenshot values transcribed; direct board verification pending.
- **Input:** two user-supplied screenshots show a chat summary reporting Orin Nano 8 GB, L4T 36.4.7, JetPack package 6.2.1, CUDA 12.6, TensorRT/bindings 10.3.0, Python 3.10.12 and installed `trtexec`. PyTorch version and usable-memory figure are not established by the visible text.
- **Finding:** [NVIDIA's JetPack archive](https://developer.nvidia.com/embedded/jetpack-archive) associates the standard JetPack 6.2.1 release with L4T 36.4.4. The reported package/BSP pairing is a discrepancy for direct investigation; its cause is unknown and it does not establish an incompatibility.
- **Response:** prepared [version notes and read-only verification commands](learning/jetson_software.md), added an [oral-defense guide](learning/oral_defense.md), and required explanations of mechanisms, alternatives, decisions, code, evidence, failures and likely questions for future work.
- **Evidence:** [preserved screenshot manifest](../results/history/2026-11-13-jetson-screenshots/manifest.json), [first screenshot](../results/history/2026-11-13-jetson-screenshots/WhatsApp%20Image%202026-11-13%20at%2012.13.16.jpeg), [second screenshot](../results/history/2026-11-13-jetson-screenshots/WhatsApp%20Image%202026-11-13%20at%2012.13.17.jpeg).
- **Limits:** these are photographs of a chat summary rather than raw system output; no board execution, installation, performance or compatibility validation occurred.

### PR-018 — Source identity and preprocessing parity

- **Event date:** 13 November 2026. **Status:** mirror and tested preprocessing verified; original archive/unit chain not independently proven.
- **Evidence:** the two local archives match the pinned Hugging Face revision's LFS SHA-256 values, and all 654 extracted HDF5 members match local archive bytes. The dataset card attributes its preprocessed files to FastDepth. The original NYU documentation specifies metres for its labeled depth arrays, but the entire mirrored HDF5 conversion chain is not independently established.
- **Comparison:** SciPy 1.1.0 source functions, adapted for current NumPy/Pillow, were compared with the project's Pillow preprocessing on five real samples and a synthetic pattern. Depth and RGB tensors matched exactly after the upstream float32 cast. This is not a test in historical SciPy/Pillow binaries.
- **Failure:** the initial reference comparison failed because NumPy removed `ndarray.tostring`; the error was observed in the session, not saved as standalone raw output. An AST compatibility substitution to `tobytes` allowed the comparison.
- **Artifacts:** [archive comparison](../results/data/2026-11-13-source-verification/archive-parity.json), [dataset provenance](../results/data/2026-11-13-source-verification/dataset-provenance.json), [preprocessing comparison](../results/data/2026-11-13-source-verification/preprocessing-parity-v2.json), [explanation](learning/dataset_preprocessing.md).

### PR-019 — Scene-grouped training and tuning subset

- **Event date:** 13 November 2026. **Status:** small exploratory subset acquired; evaluation scene overlap remains unresolved.
- **Method:** two pinned Hugging Face training shard prefixes, 16 samples from `basement_0001a` for training and 16 from `bedroom_0017` for tuning. Exact decoded RGB comparisons found no overlap between groups or with 654 evaluation samples. Near-duplicate and evaluation-scene overlap were not ruled out.
- **Failure and response:** the first new acquisition downloaded data but failed while converting a relative output path to an absolute-root-relative path; [failure record](../data/nyu_depth_v2/scene_split_2026-11-13/failure.json) and its partial artifact were preserved. The path handling was fixed and a distinct retry directory succeeded.
- **Artifacts:** [subset manifest](../data/nyu_depth_v2/scene_split_2026-11-13-retry/manifest.json), [overlap results](../results/data/2026-11-13-source-verification/scene-overlap.json), [acquisition script](../scripts/acquire_scene_subset.py).
- **Limit:** one scene per group and 16 samples each are inadequate for broad generalization; the training group includes filename `00001.h5`, excluded by upstream's internal training/holdout convention.

### PR-020 — First structural pruning candidate and desktop timing

- **Event date:** 13 November 2026. **Status:** structural candidate implemented; no reliable speed or accuracy improvement demonstrated.
- **Method:** predetermined magnitude rule retained 12 of 16 channels after the final skip connection, with coupled pointwise convolution and batch-normalization parameters sliced consistently. Starting parameters 1,360,786; candidate 1,360,550, a reduction of 236 (about 0.0173%). Keep-all parity, shape, gradient and bad-argument tests passed.
- **Tuning result:** original unmodified mean-image RMSE 0.116838; candidate before fine-tuning 0.295998; candidate after one matched epoch 0.271104, in unverified native units, on just 16 tuning images from one scene. Delta1 decreased from 0.758793 to 0.747539 after fine-tuning. None is a held-out final result.
- **Timing:** three alternating RTX 3060 PyTorch FP32 model-only runs gave median-of-run-medians 6.179140 ms (control) and 6.148260 ms (candidate), with individual pairs moving in both directions. Uncontrolled power/thermal conditions and three non-randomized pairs do not establish a speedup; no Orin latency was measured.
- **Artifacts:** [candidate specification](../checkpoints/candidates/2026-11-13-lastdecoder12/pruning.json), [predeclared experiment](../configs/experiment_2026-11-13_lastdecoder12.json), [candidate tuning](../results/training/2026-11-13-scene-pruned/summary.json), [repeated timing](../results/benchmarks/2026-11-13-pruning-timing-comparison.json), [pruning explanation](learning/pruning.md).

### PR-021 — Matched fine-tuning control and resume parity

- **Event date:** 13 November 2026. **Status:** exploratory control and one resume comparison completed; negative tuning outcome retained.
- **Method:** same 16 training and 16 tuning images, seed 42, Adam 1e-5, batch 1, masked MAE, one epoch for both unpruned control and pruned candidate. No final validation samples were used to select the candidate.
- **Result:** control tuning mean-image RMSE rose from 0.116838 to 0.400989; the candidate improved from 0.295998 to 0.271104 but remained worse than the original control before training. With an uninterrupted two-epoch control versus one epoch plus resume, model state dictionaries matched exactly in this one run. PR-023 subsequently compared BN training and evaluation modes; the modes affected results, but that comparison did not isolate a single mechanism.
- **Artifacts:** [control](../results/training/2026-11-13-scene-control/summary.json), [candidate](../results/training/2026-11-13-scene-pruned/summary.json), [resume comparison](../results/training/2026-11-13-control-resume-parity.json), [training explanation](learning/training.md).
- **Limits:** only two scenes, 32 images total; evaluation scene leakage unverified; depth units not fully traced; not a generalization or successful optimization claim.

### PR-022 — Prepare a Jetson engine-build command

- **Event date:** 13 November 2026. **Status:** prepared and desktop guard tested; board execution pending.
- **Work:** added a Jetson-only `trtexec` build script for the reported TensorRT 10.x path; it records inputs, exact command, build log and hashes and refuses to overwrite an output. It correctly rejected the x86 desktop and passed shell syntax checks. The guard output was observed in the session and not saved as a standalone raw log.
- **Artifacts:** [script](../scripts/build_jetson_engine.sh), [deployment explanation](learning/deployment.md).
- **Limits:** no board build, parser acceptance, accuracy or performance result exists. The reported BSP/JetPack discrepancy remains unverified.

### PR-023 — Matched batch-normalization mode ablation

- **Event date:** 13 November 2026. **Status:** exploratory follow-up; final model improvement not demonstrated.
- **Question:** does BN mode affect the degradation seen after a one-epoch control fine-tune? Compare the same initial checkpoint, 16 training and 16 tuning samples, seed 42, order, Adam 1e-5, batch 1 and epoch budget. The BN-eval run uses stored running statistics for forward/backward and prevents buffer updates while keeping affine parameters trainable; therefore this changes more than buffer updates alone.
- **Result:** control mean-image tuning RMSE was 0.116838 before training, 0.400989 after normal BN training, and 0.122571 after BN-eval training. All 114 BN running-buffer tensors changed in the normal run; zero changed in BN-eval mode, while 76 BN affine tensors changed. The candidate under BN-eval mode scored 0.294076 versus 0.271104 in normal BN mode and 0.295998 before training; both remain worse than the original unmodified control on this tiny tuning scene. Native depth units remain unverified.
- **Interpretation:** BN mode materially changes this specific exploratory outcome, but it changes forward statistics as well as updates; do not claim the experiment isolates which mechanism caused degradation. This ablation was motivated by previously observed tuning results; repeated tuning-set inspection and one-scene sampling limit interpretation.
- **Evidence:** [BN-mode comparison](../results/training/2026-11-13-bn-mode-comparison.json), [BN-eval control](../results/training/2026-11-13-control-frozenbn/summary.json), [BN-eval candidate](../results/training/2026-11-13-pruned-frozenbn/summary.json), [training notes](learning/training.md). The pre-run control configuration was clarified after execution to spell out the forward-pass semantics; the executed flag/hyperparameters were unchanged.

## Open gaps and negative-result register

| Gap | Latest evidence / date | Consequence for report claims | What would resolve it |
| --- | --- | --- | --- |
| Original checkpoint access | Failed saved URLs, 12 November | Cannot claim four-original-model reproduction | Acquire files with credible provenance and validate each |
| Recovered-model identity | Community provenance label, 12 November | Cannot claim authenticated original weights | Original-author checksum or equivalent provenance evidence |
| Dataset accuracy | Provisional 654-file evaluation recorded in PR-010 | Do not claim exact paper reproduction or verified metre units | Verify split identities, units and historical preprocessing parity |
| Dataset identity/units/transforms | Acquisition/count only | 654 files alone do not establish standard split compatibility | Audited identities, units, paired transforms and manifests |
| Training/tuning split and training | Two different named scenes and exact RGB overlap checks in PR-019/021 | Exploratory comparison only; scene overlap with evaluation and near-duplicates remain unverified | More diverse training/tuning data and provenance checks |
| Model-specific ONNX/TensorRT path | One-sample ONNX parity and FP16 654-file desktop metrics in PR-011/013 | No Orin inference or repeated-run uncertainty claim | Board validation and repeated runs |
| Orin hardware access | Reported version screenshots in PR-017; direct access unavailable | No verified runtime, real-time, power, memory, energy or thermal claim | Raw board version checks, execution and repeated measurements |
| Hardware-aware pruning improvement | One fixed 12-channel candidate and matched comparison, PR-020/021 | No claim that our method improves the reference; the tiny tuning result was worse than original | More robust candidates/data and Orin measurements |

The preliminary tuning loss increase in PR-016 is recorded but is too limited and correlated to support a general conclusion. The first structural candidate was worse than the original unmodified control on the tiny tuning scene, and desktop timing did not establish a reliable speedup; neither observation establishes a universal result. The failures above are acquisition/tooling/compatibility issues. Add unsuccessful model experiments when they actually occur; do not turn these setup issues into synthetic ablations.

## Material to reuse in the final report

### Evidence-backed wording available now

- **Methodology/setup:** “An independent Python 3.12 environment with 70 pinned packages was installed from a clean source snapshot and verified using CPU, CUDA, ONNX export, ONNX Runtime and TensorRT execution checks on an RTX 3060 Laptop GPU. The installation used a warm download cache.”
- **Checkpoint compatibility:** “A community-mirrored pruned FastDepth checkpoint was converted to an explicit architecture/state-dictionary format while preserving its channel configuration. Converted and legacy predictions were identical for the recorded synthetic input, and the converted model reloaded bit-exactly.”
- **Limitations:** “The original-author checkpoint endpoints were inaccessible during acquisition. One community checkpoint was recovered, but original-author identity and held-out accuracy were not established by the compatibility checks.”

These sentences describe the current evidence, not the future abstract/conclusions. Revisit them after actual dataset and board experiments; avoid copying full installation debugging history into the 12–15 page report.

### Contribution and chapter coverage

| Planned contribution / report area | Current support | Required before stronger claim |
| --- | --- | --- |
| C1: modern inference/evaluation pipeline | Independent environment, legacy conversion, structural/loader tests | Verified transforms, metrics, baseline evaluation and usable inference commands |
| C2: architecture/runtime/precision comparison | Runtime tools execute synthetic checks; random architectures construct | Comparable trained weights, model exports, same-split accuracy and actual Orin matrix |
| C3: Orin-guided pruning | No experiment evidence yet | Valid candidates, tuning logs, board-guided selection and final outcome |
| C4: deployable image/video application and measurement | Desktop image and video smoke tests exist | Board handoff, sustained demo and defined end-to-end measurement |
| C5: ablation/error analysis | No model ablation or dataset error analysis yet | Controlled training/pruning comparisons and saved error plots/results |
| Chapter 3: dataset/preprocessing/EDA | File acquisition and initial preview only | Audited dataset, manifests, EDA and transforms |
| Chapter 4: model accuracy/efficiency findings | Provisional 654-file desktop metrics and exploratory runtimes | Split/units verification, repeated runs and final board measurements |
| Chapter 5: limitations | Checkpoint, data and hardware evidence gaps recorded | Reassess which gaps remain at submission |

The five planned contribution labels are not five completed contributions. No final report claim should imply the 33 ms budget or proposed pruning accuracy target has been met.

### PR-024 — First-decoder structural pruning and matched tuning run

- **Date / phase:** 13 November 2026; exploratory model optimization, implementation evidence and negative-result analysis.
- **Objective:** test whether pruning an internal decoder width yields a larger physical parameter reduction while keeping the skip addition valid. We chose 160 of 200 channels by downstream pointwise input-weight L1 magnitude; this is a fixed heuristic, not a target-device search.
- **Implementation:** `prune_first_decoder` slices first-stage pointwise outputs and BN, next-stage depthwise filters and BN, and next pointwise inputs. The depthwise group count changes with the width; the next pointwise output remains 256 channels. Original: 1,360,786 parameters; candidate: 1,328,906 (31,880 fewer, about 2.34%). Keep-all output matched exactly; shapes, depthwise groups and gradients were tested.
- **Failure and response:** the first generated candidate was structurally valid but its checkpoint variant name incorrectly said `lastdecoder` because of a hard-coded suffix. The original artifact is preserved at `checkpoints/candidates/2026-11-13-firstdecoder160/`. We corrected the metadata and generated a separate candidate at `checkpoints/candidates/2026-11-13-firstdecoder160-corrected/`, which was used for the experiment.
- **Configuration:** `configs/experiment_2026-11-13_firstdecoder160.json` was written before the run; same manifest, seed 42, Adam 1e-5, one epoch, batch size 1 and frozen-BN mode as the matched control. The hypothesis and BN choice followed previous tuning inspection. See the candidate pruning JSON for checkpoint SHA-256 and kept indices; the training summary records source and resulting hashes, device and all metrics.
- **Result:** on 16 tuning images, mean per-image RMSE in native units was 0.146354 before and 0.170767 after training. The original matched control scored 0.116838 before and 0.122571 after training. The candidate therefore did not beat the unmodified original on this tuning sample. No candidate latency or board measurement was made; fewer parameters alone do not establish a speedup.
- **Limits / next:** only one training scene and one tuning scene; repeated tuning inspection; unknown final-evaluation scene overlap; no generalization or Orin claim. Acquire diverse scenes and predeclare follow-up comparisons. See EV-036–EV-038 and the pruning/training topic notes.

### PR-025 — Expand scene-grouped NYU samples

- **Date / status:** 14 November 2026; acquired and audited. **Question:** why only 16 training images? The 654 local evaluation files are not a training set; the full training mirror has twelve large shards and had not been downloaded. Earlier runs deliberately used bounded prefixes.
- **Command / source:** `bash scripts/python.sh scripts/acquire_multiscene_subset.py --per-scene 32 --prefix-mib 48 --output data/nyu_depth_v2/multiscene_2026-11-14-expanded` retrieved prefixes of six shards from the pinned revision `50579a0b591445f91bf7269b28611ad1cc6a05d2`. Four scene prefixes yielded 128 training files; two distinct scene prefixes yielded 64 tuning files. A prior eight-per-scene acquisition yielded the 32-image score subset and is retained separately.
- **Validation:** `scripts/verify_scene_split.py` checked scene labels against source member names, sample hashes and decoded RGB overlap. No exact RGB overlap was found between train/tune or with the 654 evaluation files. See EV-039 and manifests for identities and hashes.
- **Limitations:** sample hashes are locally computed and do not independently authenticate the mirror; adjacent frames can be highly correlated and no near-duplicate check was made. Evaluation scene names are unknown. Bedroom_0017 was used for tuning in the earlier exploratory run but is now part of training; comparisons across those experiments are not independent. Four training/two tuning scenes remain a narrow convenience sample.

### PR-026 — Single-channel ablation and a gradual-width recovery sweep

- **Date / objective:** 14 November 2026; test whether training-loss sensitivity can guide smaller physically pruned decoder widths better than the prior weight-only trial. The goal is a useful accuracy–latency trade-off, not a presumption that pruning improves accuracy.
- **Configuration:** initial protocol `configs/experiment_2026-11-14_ablation.json`; amendment `configs/experiment_2026-11-14_expanded_recovery.json` recorded after the ablation scan and before new tuning runs. The scan used 32 images, eight from each of four training scenes. The amendment uses 128 train/64 tune images in the same six scenes for every control and candidate. The previous 32 scoring images are included in expanded training.
- **Method:** `scripts/score_ablation.py` zeros each of 200 channels individually after the second-stage depthwise block, measuring the change in mean per-image masked MAE. This placement preserves correspondence with physical removal despite batch normalization. The training baseline was 0.196314 native units. All 200 scores, source checkpoint/manifest hashes, device and metric definition are in EV-040. We removed the lowest-scoring 8, 16 or 24 channels from separate copies of the original model. Individual effects do not capture interactions; this is a fixed ranking and width sweep, not iterative rescoring.
- **Structure and checks:** 192/184/176 channels gave 1,354,410/1,348,034/1,341,658 parameters versus 1,360,786 original. A test matched an individual mask against physical channel removal within numerical tolerance; full suite 18 passed (EV-042). This does not guarantee group effects or latency improvement.
- **Recovery:** one epoch, Adam 1e-5, seed 42, batch 1, BN evaluation mode, identical data and ordering; RTX 3060 CUDA. Mean-image RMSE on 64 tuning images before→after: original 0.428998→0.454392; 192 0.573695→0.475736; 184 0.711172→0.528955; 176 0.909395→0.587736. Native units remain unverified. Each pruned candidate improved relative to its own starting point, yet all remained worse than the original unmodified baseline; the trained control also worsened. Source and output checkpoint hashes and other metrics are in EV-041 and the four per-run summaries.
- **Limits / relevance:** one seed and epoch, correlated frames, a repeatedly examined tuning group, uncertain evaluation scene provenance, no Orin execution. These are useful methodology and negative/partial recovery findings, not proof of generalization or hardware speedup. A distillation recovery comparison is proposed only after a suitable compact candidate and matching baseline are established.

### PR-027 — Desktop latency check of all ablation widths

- **Date / objective:** 14 November 2026; explore whether the three fixed-width physical candidates yield a desktop model-only speed benefit.
- **Protocol:** `configs/experiment_2026-11-14_ablation_timing.json`, recorded after tuning inspection. Original and all three untrained candidates timed on RTX 3060 in PyTorch FP32, batch one, zero input; 50 warmups and 300 synchronized calls per run. Forward and reverse ordering yielded two runs per width. `scripts/benchmark_pytorch.py` preserves full latency CSVs and checkpoint hashes.
- **Results:** medians (ms), first/second pass: control 5.351/5.444; 192 channels 5.361/5.535; 184 channels 5.459/5.536; 176 channels 5.442/5.453. EV-043 links the summary and run hashes. No candidate establishes a speedup in these measurements; two runs with uncontrolled thermal/power conditions are insufficient for small-difference inference.
- **Limitation / response:** desktop PyTorch is not the target Orin TensorRT path and timing excludes data preparation, copies and visualization. We cannot claim these results predict Orin ranking. No distillation trial was run; it is a proposed recovery method once a promising accuracy–latency candidate exists, though an Orin signal is currently unavailable.

### PR-028 — Reconcile supplied PS13 with user-selected hardware scope

- **Date / evidence:** 14 November 2026. Read page 13 of the user-supplied problem-statement PDF (EV-044); this is a requirements review, not a model experiment.
- **Finding:** PS13 explicitly calls for experimental evaluation on NVIDIA Jetson Nano and Arduino UNO Q, and asks for an enhanced hardware-aware framework and comparisons across accuracy, compression, memory, speed, computation, power, energy, robustness and generalization. The user previously selected Jetson Orin Nano only and excluded UNO Q. Orin Nano is a different product from the original Jetson Nano. The source text itself does not approve these scope changes.
- **Assessment:** FastDepth/NYU, optimization, mathematical explanations and comparisons are directionally aligned. No target-board evaluation, Orin-aware candidate selection, demonstrated overall improvement, or novelty claim has yet been established. Some requested metrics are absent or desktop-only, and dataset coverage remains limited.
- **Response:** add a source-to-evidence mapping to the consolidated plan and record professor confirmation of Orin-only scope as an open dependency. Continue the user's Orin-only work; do not add UNO Q deliverables without user instruction. Ask the professor whether the substitution, omitted second board and available hardware/testing conditions are acceptable.
- **Later update (14 November):** the user reports they discussed the hardware scope with the professor and Orin Nano only was approved. This resolves the scope question for this project based on the user's report; it is not a direct written approval or a modification of the supplied PDF. Board access remains unresolved.

### PR-029 — Simple matched scoring and BN recovery controls

- **Date / objective:** 14 November 2026. After the user requested 3–5 readily explainable experiments, compare two simple channel-selection methods at equal width and two BN modes for the same candidate. The user's report that the professor approved Orin-only scope is recorded in PR-028.
- **Prior knowledge / protocol:** `configs/experiment_2026-11-14_simple_controls.json` was created before these new runs but after inspecting the earlier ablation sweep and tuning set; this is exploratory follow-up, not blind confirmation. All comparisons use the expanded manifest (128 train/64 tuning images; four/two scenes), original community checkpoint lineage, Adam 1e-5, batch one, seed 42, one epoch and RTX 3060 CUDA. Mean per-image RMSE uses native units. Data/source/output hashes are in the individual summaries and EV-045.
- **Scoring experiment:** `scripts/prune_candidate.py --stage first --keep 192 --output checkpoints/candidates/2026-11-14-weight-192` generated a physically pruned 192-channel weight-score candidate. Both it and the earlier ablation-score candidate have 1,354,410 parameters. After training with `scripts/train_scene.py ... --freeze-bn`, weight-score tuning RMSE was 0.440975→0.448607, versus ablation-score 0.573695→0.475736. The original was 0.428998 before and 0.454392 after matched training. Weight scoring was better on this tuning set; the trained weight model beat the trained control but not the original unmodified model. No Orin or generalization claim follows.
- **BN experiment:** reran the same ablation checkpoint with `scripts/train_scene.py` and no `--freeze-bn`. The identical starting tuning RMSE of 0.573695 became 3.526827, versus 0.475736 with BN eval mode. Training-mode changes both normalization statistics used during forward/backward and running-buffer updates; this run does not isolate either mechanism. Affine parameters train in both modes.
- **Interpretation / follow-up:** both new results use one seed and repeatedly inspected tuning images from only two scenes; control confounders and near-duplicate leakage remain. The small weight-model-versus-trained-control difference is not robustly established. Record experiments as understandable engineering and scientific tests, not novel inventions or proven Orin optimizations. A detailed teaching walkthrough is `docs/learning/four_experiments.md`.

### PR-030 — Acquire diverse scene-grouped NYU-derived data

- **Date / objective:** 14 November 2026; obtain a broader indoor RGB–depth training/tuning sample while retaining space, reproducibility and the 654-image evaluation collection. This is additional data from the same pinned NYU mirror, not a different independently captured dataset.
- **Pilot:** `scripts/acquire_diverse_nyu.py --windows-per-shard 1 --window-mib 8 --per-window 4 --output data/nyu_depth_v2/diverse_2026-11-14-pilot` obtained 48 pairs from 12 named scenes; no failed requests. This confirmed range sampling was feasible; the pilot is preserved separately.
- **Full acquisition:** `configs/data_2026-11-14_diverse_nyu.json` fixed the window and split parameters before the run. `scripts/acquire_diverse_nyu.py --windows-per-shard 8 --window-mib 32 --per-window 24 --output data/nyu_depth_v2/diverse_2026-11-14` used 96 bounded range requests across 12 tar shards at pinned revision `50579a0b591445f91bf7269b28611ad1cc6a05d2`. It returned 2,304 RGB–depth files across 125 named scenes, with no failed ranges. The [manifest](../data/nyu_depth_v2/diverse_2026-11-14/manifest.json) records byte ranges, offsets, hashes and selected source members. Window sampling is not uniform scene/frame sampling, and downloaded ranges do not independently authenticate full shard contents.
- **Discovered split risk:** source names `kitchen_0045a`/`kitchen_0045b` and `furniture_store_0002b`/`furniture_store_0002d` were initially placed in different groups; `basement_0001c` could share a location with previously used `basement_0001a`. These are possible same-location variants inferred from names, not verified physical overlap. The initial split remains preserved but is superseded.
- **Corrective response:** before regrouping, recorded `configs/data_2026-11-14_diverse_regroup.json`; ran `scripts/regroup_nyu_scenes.py --manifest data/nyu_depth_v2/diverse_2026-11-14/manifest.json --exclude-manifest data/nyu_depth_v2/multiscene_2026-11-14-expanded/manifest.json --output data/nyu_depth_v2/diverse_2026-11-14-regrouped`. The new manifest has 1,857 train pairs from 98 named scenes/94 inferred location groups, and 445 tune pairs from 26 named scenes/24 inferred location groups. It excludes `basement_0001c` and groups suffix variants; source and output file hashes were checked. EV-046 and EV-047 record the new manifest and group audit.
- **Validation:** exact decoded RGB overlap checks found none between train/tune and the local 654 evaluation files (EV-048). A 64-bit difference-hash screen found no pairs within Hamming distance 3 across 826,365 train–tune, 1,214,478 train–evaluation and 291,030 tune–evaluation pairs (EV-049). This screening can miss near duplicates and cannot identify shared real-world locations. A four-sample loader/preprocessing/model-forward smoke test produced finite outputs on desktop CUDA (EV-050); it did not train the model or measure accuracy. Full test suite: 21 passed (EV-051).
- **Limitations / follow-up:** only downloaded archive windows, possible adjacent-frame correlation, no authenticated upstream per-member hashes, unknown evaluation-scene identities and no proof that other differently named scenes are physically distinct. Prior model results use the old six-scene sample and must not be represented as new-dataset results. Fix the next experiment protocol and use the regrouped manifest before comparing pruning methods.

### PR-031 — Compare existing models on the larger tuning split

- **Date / objective:** 14 November 2026. Compare the current baseline and previously created pruning/recovery checkpoints on the 445-image tuning subset, without training on the new 1,857-image training group or using the 654-file evaluation set for candidate selection.
- **Configuration / method:** fixed `configs/experiment_2026-11-14_new_tuning_comparison.json` before the run; `bash scripts/python.sh scripts/compare_tuning_models.py --config configs/experiment_2026-11-14_new_tuning_comparison.json --output results/evaluation/2026-11-14-new-tuning-comparison`. The script checks the manifest and overlap-report hashes, validates sample hashes, loads/preprocesses each image once, evaluates each of ten checkpoints in PyTorch FP32 on RTX 3060, and saves pooled, per-image and per-scene metrics. The summary records the checkpoint SHA-256, parameter count, device and runtime. No training occurred.
- **Result:** mean per-image tuning RMSE in native units: original 0.300091; original trained on prior small subset 0.327769; weight-192 before/after prior recovery 0.306396/0.328357; ablation-192 before/after prior frozen-BN recovery 0.367881/0.330332; recovered ablation-184 0.364513; recovered ablation-176 0.366903; normal-BN ablation-192 1.105387; recovered weight-160 0.332615. The 160-channel recovery used a different earlier split and is not a matched training-budget comparison. Pooled RMSE and δ1, all per-scene metrics, checkpoint hashes and distinctions between prior training conditions are in EV-052; 4,450 per-image rows are retained.
- **Analysis / validation:** original had the best mean-image score; weight-192 without recovery was the closest. Recovery improved ablation-192 relative to its own starting point, not to the unmodified model. No candidate establishes an accuracy improvement over the original. An independent calculation reconstructed image, scene and pooled RMSE from saved per-image records to maximum absolute difference 2.78e-16 (EV-053). Twenty-one tests passed (EV-055).
- **Desktop speed:** `configs/experiment_2026-11-14_new_model_timing.json` fixed two forward/reverse passes, 50 warm-ups and 200 synchronized calls per run. Representative original/recovered model medians ranged around 5.3–5.6 ms in desktop PyTorch FP32, with overlapping results (EV-054 and raw per-call CSVs); no reliable speedup is established. Temperature/power were uncontrolled, order was not randomized, and a CPU-only test suite ran during early timing. These are not Orin, TensorRT or end-to-end figures.
- **Limitations and report use:** the 26 named tuning scenes contain correlated images from bounded training-shard windows. The original community checkpoint's pretraining data is unverified and may contain these NYU images or scenes; the 654 evaluation scene identities remain unknown. The tuning set is now inspected and cannot serve as an untouched test. Results support an exploratory comparison and failure analysis, not generalization, causal BN mechanism, or target-hardware improvement.

### PR-032 — Investigate official NYU evaluation identities

- **Date / objective:** 14 November 2026; investigate the local 654-image evaluation set without treating it as training or selection data. [Official NYU documentation](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) states the original labeled MATLAB file contains RGB, metric depth and scene metadata; the [FastDepth repository](https://github.com/dwofk/fast-depth) identifies its preprocessed NYU data source. The recovered community checkpoint's precise pretraining image list and original-author checksum remain unknown.
- **Availability:** bounded HEAD requests found the official labeled MATLAB file reachable at 2,972,037,809 bytes; both checked original FastDepth data endpoints timed out. The [HEAD record](../results/data/2026-11-14-provenance-head-check.json) did not download or authenticate a complete file. An initial `fsspec` HTTP attempt failed because the environment lacked its `requests`/`aiohttp` HTTP dependencies; instead a custom validated HTTP range reader was implemented. That failed command's raw log was not preserved; this is reconstructed session history.
- **Exact checks:** `scripts/audit_official_nyu_identity.py` read a bounded part of the original file. Among its first 16 labeled indices, six occur locally, and all six RGB arrays matched the one-based filename/index convention exactly (EV-056). A second evenly spaced sample of 32 local evaluation filenames yielded 31 exact RGB matches to the hypothesized official image index (EV-057). The non-exact case, `00567.h5` versus official index 566, has 99.57% equal RGB byte values, mean absolute byte difference 0.0614 and maximum 89; the reason is unknown (EV-058).
- **Inferred scene mapping:** `scripts/audit_nyu_scene_mapping.py` applied the supported but incompletely verified filename/index hypothesis to all 654 files and retrieved original scene metadata. The result contains 215 distinct inferred evaluation scene names and no overlap with new training/tuning suffix-normalized location groups (EV-059). This is useful evidence, but the other files are inferred rather than exact-matched, naming-based grouping is heuristic, and different names may still represent the same real location. Original checkpoint pretraining overlap is still unverified.

### PR-033 — Scene-aware channel selection and three-seed recovery

- **Date / objective:** 14 November 2026; implement the user's requested simple potential course-project contribution and compare it fairly with weight, average-ablation and random pruning. This is an exploratory extension to known ideas, not a verified claim of research novelty or Orin-aware selection. Related group-sensitive compression and random pruning work are linked in [the teaching note](learning/scene_aware_pruning.md).
- **Predeclared protocol:** `configs/experiment_2026-11-14_scene_aware.json` fixed 192 retained channels, a training-only score sample of 32 selected named scenes × 4 frames, coefficient 0.5, one epoch, batch 4, Adam 1e-5, BN evaluation mode and seeds 42/43/44. Post-run name inspection showed two calibration scene names have the same inferred location key, so the sample comprises 31 inferred groups, not 32 necessarily independent places. Original, weight, average-ablation and scene-aware candidates share the same data/order for each seed; random masks vary with seed and therefore mask and order effects are not separated. The tuning manifest had been inspected previously; the 654-file evaluation was not used for model selection.
- **Implementation:** for channel j and training scene s, measure the per-image masked MAE change after masking the channel, then select by mean scene loss change plus 0.5 times the largest scene loss change. `scripts/score_scene_aware.py` records all per-scene losses and selected-image hashes (EV-060). A two-scene/three-channel smoke test passed. `scripts/build_scene_candidates.py` physically created equal-width weight, average, scene-aware and three random candidates with exact reload checks (EV-061); the mean and scene-aware rules removed three different channels from their groups of eight.
- **Recovery / reproducibility:** `scripts/train_scene_batch.py` trained 15 runs using the 1,857-image training set and saved model/optimizer/RNG state; it does not implement resume. The run audit verified 15 input/output hashes, data/settings and same shuffled order within each seed (EV-062). During the sweep the unchanged intended masked-loss arithmetic was moved into a function for testing; the old complete source and full-run bit-parity were not saved. The [reconstructed refactor note](../results/history/2026-11-14-scene-aware/trainer-refactor-note.txt) describes which processes used each version. Post-hoc replays found nonidentical model states and a tuning RMSE difference larger than the small observed method gap; see PR-035.
- **PyTorch tuning results:** 22 checkpoints were evaluated on 445 images from 26 named scenes / 24 inferred location groups. Before additional recovery, original mean-image RMSE was 0.300091 native units, average-ablation 0.290574 and scene-aware 0.284237. After recovery, three-seed mean-image RMSE averages were original 0.283717, weight 0.284839, average-ablation 0.283696, scene-aware 0.282248 and random 0.285755 (EV-063, EV-064). Scene-aware beat average ablation in 15, 19 and 10 of 24 inferred location groups by seed; its worst-group RMSE was not uniformly better. Scene-aware had 6,376 fewer parameters than the starting model. All outcomes and per-scene data are retained; the 0.001448 difference in seed-mean RMSE against average ablation is small and not a statistical/generalization or hardware claim.
- **Validation and limitations:** 9,790 per-image rows independently reconstruct pooled and scene/image means to maximum absolute difference 2.22e-16 (EV-065); full suite 24 passed (EV-066). The test set scene identity mapping is supported but incomplete (PR-032), pretraining overlap unknown, calibration set is small, tuning scenes have unequal sample counts including one image in one location group, tuning has been inspected, only one width/lambda/epoch was tested and no Orin result exists. `docs/learning/scene_aware_pruning.md` gives equations, commands, alternatives and questions. Desktop TensorRT checks are separately reported when complete.
- **Follow-up:** the deterministic repeat in PR-036 narrows the scene-aware versus average-ablation three-seed mean RMSE gap to 0.000141 and leaves the trained original 0.000131 lower than scene-aware. Use the repeat for current discussion; neither set establishes reliable superiority.

### PR-034 — Desktop TensorRT accuracy, latency and outlier diagnosis

- **Date / objective:** 14 November 2026; compare seed-42 original and all four pruning candidates on desktop TensorRT FP16-enabled engines, using the same 445 tuning images as PyTorch. The five models and method were fixed in `configs/experiment_2026-11-14_scene_tensorrt.json` before engine builds and this tuning comparison. This is not a final held-out, end-to-end or Orin experiment.
- **Export and correction:** CPU PyTorch–ONNX Runtime parity passed on one training image for all five models, maximum difference 2.38e-6–3.81e-6 (EV-068). Three initial reports mislabeled the supplied training image as validation because `scripts/export_onnx.py` had hard-coded its scope. The initial outputs remain; the script was corrected and the three exports regenerated under new paths, retaining logs. The other two outputs recorded the correct scope.
- **Accuracy:** each engine was built locally on the RTX 3060 using TensorRT 10.3, FP16 tactics enabled (other operations may remain FP32). Mean-image tuning RMSE TensorRT/PyTorch: original 0.282107/0.281928; weight 0.285215/0.284867; average-ablation 0.282520/0.282455; scene-aware 0.281560/0.281534; random 0.284596/0.284614 (EV-069). `scripts/compare_tensorrt_tuning.py` v2 records per-image runtime-output differences; the first comparison is preserved and v2 adds diagnostics. EV-070 verifies aggregation from 4,450 saved image/runtime rows to 2.22e-16.
- **Output outliers:** despite close mean RMSE, maximum absolute pixel difference versus PyTorch reached 1.396948 native units for original, 0.850651 weight, 0.415020 average, 0.062409 scene-aware and 0.741938 random. At an exploratory 0.1-unit threshold, original had at least one flagged pixel in 21 images (1,100 of 22,328,320 output pixels); scene-aware had none in this one run (EV-071). This threshold is not a predefined deployment acceptance criterion and these are output differences, not errors relative to ground truth. A post-hoc FP16-disabled desktop TensorRT build on the worst original image had max difference 0.143082 (EV-072), but default TF32 and engine tactics were not isolated; the cause remains unknown.
- **Latency:** saved engines were replayed with 50 warm-ups and 300 synchronized calls, in forward/reverse order, under `configs/experiment_2026-11-14_scene_tensorrt_replay.json`. Original medians were 0.388433/0.324522 ms; scene-aware 0.324578/0.316602 ms (EV-073). Runtime varied notably, model order was not randomized, thermal/power state was uncontrolled, there was one build per model, and TensorRT warned default-stream execution may synchronize. No reliable speedup or Orin inference follows.
- **Report use / next:** explain precision-versus-accuracy trade-offs, outlier analysis and measured latency limitations, while locating sensitive operations and verifying target-device output before claims. Full details are in [scene-aware learning note](learning/scene_aware_pruning.md).

### PR-035 — Same-seed variability and deterministic replay

- **Date / question:** 14 November 2026; determine whether a small scene-aware tuning advantage is distinguishable from repeat-run variation, prompted by the mid-sweep loss-function refactor documented in PR-033.
- **Post-hoc replay:** repeated the original model with the same seed 43 and recorded sample order using the refactored script. The model states differed in 114 of 228 tensors, maximum absolute difference 0.000608 (EV-074). Tuning mean-image RMSE was 0.284011 versus 0.286014, a 0.002003 difference (EV-075), larger than the 0.001448 three-seed mean difference between scene-aware and average-ablation. This does not identify a cause.
- **Same-code replay:** a second independent run of the same refactored code and seed also differed from the first in 114 tensors, maximum 0.000816, despite identical order (EV-076). Thus the refactor alone cannot explain all variability. Environment/runtime effects and GPU nondeterministic operations remain hypotheses; no root cause was proven.
- **Deterministic diagnostic:** `configs/experiment_2026-11-14_deterministic_replay.json` fixed one original checkpoint/seed before two full replays. Launched each Python process with `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `--deterministic`, `torch.use_deterministic_algorithms(True)`, cuDNN deterministic true and benchmark false. A smoke run passed; both full checkpoints had identical SHA-256 and all 228 model-state tensors matched exactly in this environment (EV-077). This is one model and one seed, not a cross-platform guarantee or a retroactive fix for PR-033.
- **Response:** `configs/experiment_2026-11-14_deterministic_sweep.json` records a follow-up repeat of the same 15 model/seed combinations, reusing one verified deterministic original seed-43 replay. The prior tuning results were already inspected. The observed non-deterministic 0.001448 method gap must not be reported as a reliable improvement on its own.

### PR-036 — Deterministic repeat of scene-aware comparison

- **Date / objective:** 14 November 2026; repeat the predefined 192-channel comparison with deterministic algorithms after finding same-seed variation larger than the initial method gap. `configs/experiment_2026-11-14_deterministic_sweep.json` was recorded after earlier tuning results were inspected and before this repeat; this is not a blinded or independent experiment.
- **Protocol:** same original checkpoint, four pruning rules and masks, 1,857 training images, 445 tuning images, one epoch, batch four, Adam 1e-5 and frozen BN as PR-033; deterministic PyTorch/cuDNN settings and `CUBLAS_WORKSPACE_CONFIG=:4096:8` were requested in each process. Seeds 42/43/44; reuse the earlier byte-identical deterministic original seed-43 run. The saved run audit checks all 15 expected runs, settings, source/output hashes and matching per-seed data orders (EV-078). This does not guarantee every method or other device is bit-reproducible.
- **Mean-image tuning RMSE, native units:** original seeds 0.282433/0.284597/0.282133, mean 0.283054; weight 0.284435/0.286175/0.283858, mean 0.284823; average-ablation 0.280680/0.284865/0.284434, mean 0.283326; scene-aware 0.283273/0.284344/0.281939, mean 0.283186; random 0.284623/0.285725/0.284725, mean 0.285024 (EV-079, EV-080). Original mean is 0.000131 lower than scene-aware; scene-aware mean is 0.000141 lower than average-ablation. These tiny differences do not establish superiority or robustness.
- **Location and verification:** scene-aware beat average ablation in 13, 15 and 15 of 24 inferred location groups for seeds 42, 43 and 44; its worst-group RMSE was lower only for seed 44 (EV-080). Group sizes are unequal and include one image. Independent reconstruction of 9,790 per-image records matched aggregate RMSE to 2.22e-16 (EV-081). The 654-file final evaluation set was not used for selection.
- **Limits / next:** original checkpoint pretraining may overlap NYU tuning; tuning previously inspected; one width/lambda/epoch; random mask and order vary together; only three seeds; scene names are imperfect identity proxies. This repeat does not establish novelty, generalization or Orin improvement. Desktop TensorRT checks of deterministic seed-42 original and scene-aware checkpoints were fixed before these tuning results; report their results separately once complete.

### PR-037 — Deterministic-checkpoint TensorRT check and comparison plot

- **Date / objective:** 14 November 2026; test the deterministic seed-42 original and scene-aware checkpoints on desktop TensorRT. `configs/experiment_2026-11-14_deterministic_tensorrt.json` fixed both models before deterministic tuning results were inspected. This supplements, rather than replaces, PR-034's non-deterministic-checkpoint results.
- **Export:** CPU PyTorch–ONNX Runtime checks on a training input passed for both deterministic checkpoints: max absolute errors 5.245e-6 and 2.146e-6 (EV-082). The test covers one input and is not Orin parity.
- **Accuracy and outliers:** over all 445 tuning images, mean-image RMSE was TensorRT/PyTorch 0.282669/0.282433 for original and 0.283390/0.283273 for scene-aware (EV-083). Maximum absolute output differences were 2.110585 and 0.166729 native units respectively, both on `bathroom_0050/00076.h5`; at an arbitrary 0.1-unit diagnostic cutoff, 27 original images and one scene-aware image contained at least one flagged pixel (EV-084). These are runtime-output differences, not ground-truth prediction errors or a predefined deployment tolerance. The output differences, preprocessing and scene identities need further investigation before deployment conclusions.
- **Timing:** two ordered 300-call replay passes after 50 warmups yielded original medians 0.384894/0.326886 ms and scene-aware 0.326095/0.330172 ms (EV-085). The large drift between original passes, default-stream synchronization warning, uncontrolled power/thermal state and single engine build per model preclude a reliable speedup claim. Timing is desktop-only, model-only, FP16 tactics enabled and not Orin.
- **Plot and checks:** the [deterministic comparison plot](../results/figures/2026-11-14-deterministic-scene-comparison-v3.png) shows all seed values with a labeled zoomed vertical scale (EV-086). The first version had footer overlap, the second fixed it, and the third explicitly labeled the zoom; earlier versions are retained. EV-087 records 24 tests passed after implementation. None of this establishes generalization or verified novelty.

### PR-038 — Document and trace the recovered FastDepth architecture

- **Date / objective:** 14 November 2026; provide a detailed, teachable architecture explanation for the user's professor discussion. This is documentation and structural inspection, not a new model optimization or accuracy experiment.
- **Method:** loaded `checkpoints/converted/community-skipadd-pruned.pth` on CPU in evaluation mode, used one seed-42 synthetic `[1,3,224,224]` input and traced the 14 encoder blocks, five decoder stages, three additive skip connections and final head. A manually staged forward pass matched the model's own output exactly on that input. The one-off inline inspection command was not saved as a standalone script; the input seed, checkpoint SHA-256, runtime, all dimensions and results are preserved in EV-088.
- **Observed:** encoder has 1,136,648 trainable parameters and decoder including the head has 224,138, totaling 1,360,786. The final 16→1 pointwise head also contains batch normalization and ReLU. The trace used synthetic data and does not establish prediction accuracy, original-author checkpoint authenticity or Jetson performance.
- **Response:** expanded [model architecture](learning/model_architecture.md) with the complete actual layer dimensions, 5×5 depthwise/1×1 pointwise and skip-add operations, head behavior, worked operation counts, input transforms, training/inference distinction and oral-defense questions. No model or prior experiment was changed.

## Template for the next substantial event

Copy this structure, assign the next unused ID, and retain the resolved failure history:

```text
### PR-039 — Short factual event title
- Event date and date recorded (if different):
- Phase / report section / contribution supported:
- Status and evidence type:
- Objective or hypothesis:
- Model/checkpoint hash, data manifest, device/runtime/precision, configuration/seed:
- Command or run ID:
- What happened, including failed or unsupported attempts:
- Measured result, units, sample/repeat count, aggregation/tolerance:
- Diagnosis: confirmed cause versus hypothesis:
- Change or response and why:
- Follow-up validation and evidence paths:
- What the evidence permits us to claim:
- Remaining limitation and next dependency:
```

For model experiments, preserve the original configuration/raw samples and comparison baseline even when the result is poor. Capture future failing command output when practical; do not replace it with a polished success narrative. Read-only inspections that reveal no new finding need not create an event.
