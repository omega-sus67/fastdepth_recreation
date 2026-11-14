# Explaining and defending the project

This is a living question-and-answer guide. Each answer distinguishes the reason for a choice, alternatives, implementation, evidence and limitations. Update it with every substantial code or methodological decision; do not memorize a performance claim that has not been measured. For deeper detail see the separate [topic notes](README.md) and [progress history](../progress_report.md).

## What are we building?

We use FastDepth to predict a dense depth map from one RGB image. The output estimates a depth value per pixel; it is learned inference, not a physical distance measurement. The target is deployment on Jetson Orin Nano, with a focus on accuracy versus latency and resource cost. We inherited the FastDepth architecture and a community checkpoint; our project work includes adapting, validating and comparing the pipeline. We must credit inherited work and describe our actual additions accurately.

## Why choose RMSE?

RMSE is the square root of the mean squared difference between prediction and target over valid pixels. Squaring gives large errors more influence, which helps reveal severe depth mistakes; taking the square root returns the result to the depth array's units. FastDepth also reports RMSE, making comparison possible if the data, processing and aggregation match. Its weakness is sensitivity to outliers, so we also report MAE, relative error, delta1 and R². Depth units and exact evaluation equivalence still need verification.

## Why is pooled RMSE different from mean per-image RMSE?

For each image, calculate the mean squared error then its square root. Averaging these image RMSE values yields **mean per-image RMSE**. Pooling every valid pixel first and then taking one square root yields **pooled RMSE**. The square root is nonlinear, so the answers generally differ even when all images have equal numbers of valid pixels. Our provisional run reports 0.603737 mean per-image and 0.721159 pooled, in unverified native units. Comparing the latter directly with a published mean-image RMSE would be misleading. See [metrics](metrics.md) and the saved run for exact definitions.

## What is delta1 and why use it?

For positive target depth `d` and prediction `p`, calculate `max(p/d, d/p)`. A pixel passes if that ratio is **strictly less than 1.25**. Delta1 is the fraction of valid pixels that pass; higher is better. If target is 2 and prediction is 2.4, the ratio is 1.2, so it passes. At 2.5, the ratio equals 1.25 and fails because the inequality is strict. Unlike RMSE, delta1 is relative to depth and has no depth unit. It is useful alongside RMSE because FastDepth reports it and because it communicates the share of predictions within a relative tolerance. It does not describe the size of errors beyond the threshold.

## Why also use MAE, REL and R²?

MAE averages absolute error and is less dominated by large deviations than RMSE. REL averages `abs(p-d)/d`, so the same absolute error matters more for nearby targets; very small valid depths can produce large contributions. R² compares squared error with a constant mean-target baseline; it can be negative and is undefined if all valid target values are identical. These measures answer different questions and should not be collapsed into one score. All must use a declared validity mask, units and aggregation.

## What happens to zero or invalid pixels?

The current evaluator considers a target valid if it is finite and positive. It excludes invalid targets, rejects nonfinite predictions at valid targets and counts nonpositive predictions rather than removing them. For delta1 ratios only, a nonpositive prediction is clamped to a small epsilon to avoid division by zero. This policy prevents discarding difficult predictions; it can differ from the legacy code, so comparisons must state both policies. The current local audit found all raw depth pixels positive and finite, but that observation is specific to these 654 files.

## What is TensorRT and why use it?

TensorRT is NVIDIA's inference optimizer/runtime. It parses a supported network, selects implementation tactics for the target hardware and builds an engine. It can fuse operations and use reduced-precision tactics where configured. The motivation is to measure whether the same network executes more efficiently; no universal speedup or accuracy preservation is guaranteed. ONNX is the graph/weight interchange format; TensorRT is one way to compile and run that graph. We must compare the same model, inputs, precision, timing boundaries and accuracy.

## How was our pipeline set up?

On the desktop, we created a separate Python 3.12 environment with pinned PyTorch/CUDA, ONNX, ONNX Runtime and TensorRT dependencies. A wrapper prevents unrelated ROS paths from contaminating imports. A clean source-snapshot installation passed CPU/GPU and synthetic runtime checks. We recovered a community checkpoint, recorded its checksum, converted its legacy model to explicit architecture plus weights, and checked legacy/converted outputs. We then loaded HDF5 RGB/depth pairs, applied a documented resize/crop/resize sequence, evaluated 654 samples, exported ONNX and compared one CPU output. Finally we measured exploratory desktop model-only timings and evaluated an FP16-enabled desktop engine. Exact commands and evidence are in [setup](setup.md), [dataset](dataset_preprocessing.md), [deployment](deployment.md) and [progress history](../progress_report.md).

## Did we improve the model on Orin Nano?

No Orin improvement has been demonstrated yet. On the RTX 3060 desktop, one synchronized model-only run had median 0.520 ms for TensorRT FP32 and 0.392 ms with FP16 tactics enabled. Both used 50 warm-ups and 500 calls on a preallocated zero tensor; FP16 can still use mixed tactics, and the FP32 TF32 setting was not recorded. These are single desktop runs, exclude preprocessing/transfers, and cannot establish the 33 ms end-to-end Orin outcome. The 654-file FP16-enabled evaluation has similar provisional accuracy to desktop PyTorch, but split identity, units and exact historical transform parity are unresolved. Full explanations are in [quantization](quantization.md).

## Did we implement pruning ourselves?

The available checkpoint is **already pruned**; that pruning is inherited, not our novel contribution. We reconstructed its architecture and preserved channel configuration during conversion. We implemented one new candidate that physically removes four channels after the final skip connection, and ran a tiny controlled tuning comparison. This is a first exploratory proposal, not a demonstrated improvement: it removes only 236 parameters and remains worse than the original unmodified model on the small tuning scene. Actual target-device measurements and broader data are still required. See [pruning](pruning.md).

## Why not trust the training smoke test as an improvement?

The download stopped after one scene. The smoke test used 12 training and four tuning images from that same scene, so images can be correlated and scene overlap with final evaluation remains unknown. It only demonstrates that training and checkpoint resume execute. Tuning loss increased on the resumed epoch. A defensible experiment needs disjoint source groups, an appropriate control, reproducible settings and evaluation of frozen choices on the held-out set.

## What software does the target board have?

The supplied screenshots report Orin Nano 8 GB, L4T 36.4.7, JetPack package 6.2.1, CUDA 12.6, TensorRT and Python bindings 10.3.0, Python 3.10.12 and installed `trtexec`. They are a chat summary rather than raw board output. NVIDIA's archive associates standard JetPack 6.2.1 with L4T 36.4.4; an updated BSP/package combination may explain the difference, but the reason is unknown. Verify both independently on the board. See [reported Jetson software](jetson_software.md).

## Template for every new decision or code change

1. **Question:** What problem or constraint prompted the choice?
2. **Mechanism:** Explain the concept and relevant tensor shapes, equations, data flow or runtime behavior in plain language.
3. **Alternatives:** What plausible alternatives exist, and what trade-off made this choice appropriate here?
4. **Implementation:** Which files/functions/commands and configurations implement it? Include a small concrete example where useful.
5. **Evidence:** What was tested, on which data/device, with what result? Link the raw output and state units, scope and aggregation.
6. **Limitations:** What has not been tested, what can go wrong, and what would change the conclusion?
7. **Defense:** Add likely follow-up questions and concise answers to the relevant topic note or this guide.

## Why did we remove only four channels, and is that an improvement?

The last decoder's 16 channels are downstream of the skip additions, so this is a conservative place to implement and verify channel surgery without altering paired skip branches. We kept 12 channels based on the magnitude of their final 1×1 output weights and sliced the coupled convolution/batch-normalization tensors. This removes 236 of 1,360,786 parameters, around 0.0173%. The channel count and rule were fixed before the tuning experiment. The candidate's tuning RMSE was worse than the original unmodified model and no Orin latency was measured; therefore this is an implemented pruning experiment, not a proven improvement. A deeper search needs device-specific measurements and careful coupling of depthwise and skip channels.

## How do we know preprocessing matches the original?

We compared against the actual SciPy 1.1.0 `imresize` source, with small compatibility changes for current NumPy/Pillow. For five real samples and one synthetic pattern, the final RGB and depth tensors matched exactly after the original pipeline's float32 conversion. This supports the current transform on tested inputs; it is not a full historical binary-environment reproduction, nor proof that the local 654 files exactly equal the original MIT archive. The two local archives do match hashes recorded for the pinned Hugging Face revision, and its dataset card attributes the files to FastDepth. The original NYU documentation describes labeled depth arrays in metres, but the exact conversion chain of this particular mirror has not been independently verified.

## Did the pruning candidate make inference faster?

We do not have a defensible speedup claim. In three alternating desktop model-only comparisons, the median of run medians was 6.179 ms for the starting model and 6.148 ms for the candidate, and individual pairs moved in both directions. Only 236 parameters were removed; the measurements were not randomized or controlled for power/thermals. Also, the candidate's tuning accuracy was worse than the original unmodified model on our small tuning scene. These are useful negative/uncertain results: reduced parameter count alone is not proof of real deployment improvement.

## Did we use the held-out evaluation set to choose pruning?

No. The one candidate and training schedule were fixed in `configs/experiment_2026-11-13_lastdecoder12.json` before its training comparison. We used one named training scene and another named tuning scene. Exact decoded RGB matches with the 654 local evaluation files were checked and none found, but evaluation scene names are unknown and near-duplicate leakage is not ruled out. We have not evaluated this candidate on the final 654 files or claimed broad generalization. The tiny sample size and uncertain scene relationship are important limitations.

## Why did fine-tuning worsen the original model, and what did we test?

The exact cause is not proven. One plausible factor was batch normalization with batch size one and a single training scene. We compared normal BN training mode with BN evaluation mode under otherwise matched settings. Mean-image RMSE on the small tuning scene was 0.400989 after normal mode and 0.122571 after BN evaluation mode, versus 0.116838 before training. BN evaluation mode both uses stored statistics in the forward pass and prevents their updates; it does not freeze the trainable scale and offset. Our checkpoint comparison confirmed zero changed BN running buffers in that mode and changed affine parameters. This is evidence that the modes behaved differently in one small test, not proof of which specific mechanism caused the degradation or a general result. The pruned model still scored worse than the original unmodified model.

## How did you prune through a depthwise convolution?

We selected 160 of the first decoder's 200 internal channels using the L1 magnitude of their weights in the next pointwise convolution. We removed corresponding output filters and batch-normalization entries, then removed the same depthwise filters and set their group count to 160. We also removed corresponding input channels in the next pointwise convolution. Its 256 output channels remain unchanged, keeping the skip addition compatible. This removed 31,880 parameters, but the candidate's 16-image tuning RMSE worsened relative to the unmodified model; no speedup is established. The score does not account for activations or hardware, and the repeatedly inspected tuning sample is too small for generalization.

## Why ablation rather than weight magnitude?

We can temporarily zero one channel at the input of the next pointwise convolution and measure the change in masked depth MAE on training images. This directly probes the impact on the task loss, while a weight-only score ignores actual activations and normalization. We rank channels by individual loss increase, then physically prune several low-scoring channels and measure the group effect on different tuning scenes. It is not guaranteed optimal: channels interact and our convenience sample is small. The mask is placed after the depthwise normalization because a normalization offset could make an earlier zero nonzero. A unit test checks one-channel mask versus structural removal.

## When would knowledge distillation help?

Once a compact candidate has a promising measured latency–accuracy trade-off, compare ordinary recovery training against a matched student trained with both ground-truth depth loss and a fixed original teacher's prediction loss. The teacher is not updated; tune the relative loss weight using training/tuning data. Distillation may restore accuracy but can also inherit teacher errors; a desktop latency signal is provisional until Orin measurement. This is a proposed next experiment, not an implemented or proven improvement.

## Did task-loss ablation beat simple weight scoring?

Not in our matched exploratory test. At 192 channels, both had the same parameter count and recovery settings; weight-based scoring gave 0.448607 mean-image tuning RMSE after recovery versus 0.475736 for ablation. Both were worse than the original unmodified model's 0.428998. These values come from 64 images in two scenes already inspected during development. The result does not prove weight scoring is universally better; individual ablations can miss channel interactions and the scoring training subset may not represent the tuning scenes. The [four-experiment walkthrough](four_experiments.md) shows the complete comparisons and explanations.

## Did the earlier comparison of existing checkpoints show an improved model?

On the larger 445-image tuning set, the original model had mean per-image RMSE 0.300091; the closest existing candidate, weight-selected 192 channels before recovery, scored 0.306396. The recovered ablation-selected 192 model scored 0.330332, an improvement over its own pre-recovery 0.367881 but not over the original. Two desktop PyTorch timing passes did not establish a reliable speedup. These checkpoints were trained on earlier small subsets, not the newly acquired training split. The tuning frames are from NYU mirror training shards and could overlap the unknown original checkpoint pretraining data, so this is not a provably independent generalization test or Orin evaluation. See the [walkthrough](four_experiments.md) for all ten models and metric definitions.

## What happened in the scene-aware experiment?

We ranked channels using their mean loss change across 32 training scenes plus half their largest scene-specific loss change, then compared the resulting 192-channel model with weight, average-ablation and random selections. After matched one-epoch recovery, mean-image tuning RMSE averaged over three seeds was 0.282248 for scene-aware versus 0.283696 for average ablation and 0.283717 for the trained original. A post-hoc same-seed replay of the original differed by 0.002003 RMSE, larger than that 0.001448 method gap, for a reason not yet isolated. The result is a small exploratory difference on an already-inspected NYU-derived tuning set; original pretraining may overlap it. It is not proof of reliable improvement, novelty, statistical significance or Orin improvement. Worst-location performance was not consistently better across the three seeds.

We then requested deterministic algorithms and repeated the comparison: the three-seed mean tuning RMSE was 0.283186 for scene-aware, 0.283326 for average ablation and 0.283054 for the trained original. The gaps are tiny, and the original was slightly better in this repeat. One same-model, same-seed deterministic replay produced byte-identical checkpoints on this PC, but that is not a guarantee across devices or all models. The comparison was run after earlier tuning inspection and does not establish generalization.

## Did TensorRT preserve every depth prediction?

No. Aggregate tuning RMSE stayed close to PyTorch for all five seed-42 engines, but the original engine had one output pixel differing by as much as 1.396948 native units. A diagnostic threshold of 0.1 flagged 21 original-engine images. This compares runtime outputs, not prediction error against ground truth. The default CUDA stream also produced a TensorRT synchronization warning, and repeated desktop timing varied, so we need precision-outlier investigation and board measurement before concluding the conversion is suitable or faster. See [details](scene_aware_pruning.md#desktop-tensorrt-accuracy-and-timing--14-november).

For the deterministic seed-42 checkpoints, a separate TensorRT build produced maximum output differences of 2.110585 for the original and 0.166729 for scene-aware, again on the tuning images and in native units. This single-seed comparison is not evidence that one method will always be numerically safer. A diagnostic on a previously observed outlier reduced the maximum when FP16 tactics were disabled, but different tactics and default TF32 were not isolated.
