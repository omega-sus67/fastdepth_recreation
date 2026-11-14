# Our FastDepth project story

This is a learning-oriented account of the work recorded through 14 November 2026. It follows the decisions and results in order, including attempts that did not work. The [progress report](progress_report.md) is the authoritative detailed history with run identifiers and evidence; the [current priorities](immediate_priority.md) describe what remains. Some early failures were reconstructed from session history because their original error logs were not saved.

## 1. We defined the problem

We want a model that takes one RGB image and predicts a depth value for each pixel, running efficiently on a Jetson Orin Nano. Depth estimation can help a device understand the approximate geometry of a scene without relying on a depth sensor at inference time. We chose FastDepth as the starting point because it was designed for efficient monocular depth estimation.

The supplied PS13 originally mentions two boards. The user subsequently reported that the professor approved an Orin-only scope. We have not yet accessed that board, so all measured runtime results so far are from the desktop, not the Jetson.

We reviewed the supplied evaluation guidelines and created a [single detailed plan](fastdepth_edge_ai_plan.md), a changing immediate work queue, and separate teaching notes. The poster and final report are deferred. We also established a progress history so unsuccessful experiments remain available for an honest discussion.

## 2. We made the project reproducible

The desktop environment needed to be independent of unrelated installed software. We created a project virtual environment, setup scripts and a Python wrapper. An inherited ROS path had caused pytest to load an unrelated plugin and fail; isolating project commands addressed that interference. We also adjusted setup to recognize the vendored FastDepth source without requiring its own nested Git repository.

**What we learned:** a working program on one machine is not automatically reproducible. Python versions, package paths, CUDA and framework versions all affect whether another machine can execute it. The desktop environment is not a Jetson installation recipe: the board uses a different CPU architecture and its actual software versions must be checked.

## 3. We found and recovered a pretrained model

The original checkpoint sources were inaccessible during our attempts. We located a community-mirrored checkpoint, recorded its provenance and converted it for the current project. The legacy checkpoint initially failed with the modern default PyTorch loader; a controlled conversion produced a usable checkpoint. Its identity has **not** been authenticated against the original author's file, and it is already a pruned model. Thus, when we say “baseline” below, we mean this recovered community model before our additional experiments—not a verified original unpruned FastDepth checkpoint.

The recovered model has 1,360,786 parameters and accepts a batch of RGB images shaped `[1, 3, 224, 224]`; it returns `[1, 1, 224, 224]` depth predictions. The detailed component-by-component explanation is in [model architecture](learning/model_architecture.md).

**What we learned:** model weights, architecture and preprocessing are all part of a reproducible model. A community copy can be useful, but its origin and limitations must be stated.

## 4. We built a baseline and checked the data pipeline

We implemented the image/depth preprocessing and evaluation metrics, then checked the adapted preprocessing against upstream source on selected cases. One early inference attempt failed because an image was arranged as height × width × channels while the convolution expected channels first. Correcting the layout allowed the supplied example to run.

On 654 local evaluation images, the baseline obtained pooled RMSE **0.721159** and δ1 **0.771065**. Mean per-image RMSE was **0.603737**. RMSE measures the size of depth errors, giving larger errors greater influence; δ1 is the fraction of valid predictions within a multiplicative accuracy threshold. Pooled and mean-image RMSE aggregate errors differently and should not be compared as if they were the same number. Depth units remain unverified. See [progress PR-010](progress_report.md#pr-010--validation-audit-and-provisional-baseline) for the complete metrics and valid-pixel details.

We also investigated the evaluation images' identity. Thirty-one of 32 sampled RGB images exactly matched hypothesized official NYU images; one was near-identical. This supports the investigation but does not verify every image or prove all scene locations are independent.

## 5. We made the model usable and exported it

We added image and recorded-video inference, exported the model to ONNX, and tested desktop runtime output. ONNX represents the model in a form TensorRT can attempt to import; it does not itself guarantee that a particular board or runtime will accept or reproduce the model. We added desktop benchmarking and a guarded script to build a TensorRT engine on the Jetson. That board script has not yet been tested on the board.

The desktop FP16-enabled TensorRT baseline had aggregate accuracy close to PyTorch on the 654 images. That was encouraging, but later tests exposed individual prediction outliers, so aggregate accuracy alone is not sufficient to certify runtime parity.

## 6. We tried pruning and recovery training

Structured pruning physically removes channels and associated connections, making a smaller network rather than merely setting individual weights to zero. We first tried a candidate on a very small tuning set of 16 images from one scene. Fine-tuning improved that candidate relative to its own pruned starting point, but it remained worse than the unmodified baseline. A matched unpruned training control also worsened in that early run.

We examined batch normalization (BN), which maintains statistics used to normalize activations. With a tiny training setup, the BN mode strongly affected outcomes. For one later matched candidate, mean-image RMSE after training was **0.475736** with BN in evaluation mode and **3.526827** with normal BN training mode. This comparison does not isolate whether the difference came from the statistics used in training, updates to stored statistics, or both.

We gradually tested first-decoder widths of 192, 184 and 176 channels, selected through individual-channel ablation. After matched recovery, their mean-image tuning RMSE values were **0.475736, 0.528955 and 0.587736** respectively, against **0.428998** for the unmodified model on that experiment's 64-image tuning set. Removing more channels hurt accuracy in this small experiment. Desktop timing did not convincingly show a speedup.

We also compared selection by weight magnitude against ablation scoring. On that same small setup, the 192-channel weight-scored candidate was better than the ablation-selected candidate after matched training, but neither beat the unmodified original. Full experimental context is in [the four experiments](learning/four_experiments.md).

**What we learned:** fewer parameters do not guarantee lower latency, and fine-tuning does not automatically restore accuracy. A matched unpruned control is necessary to distinguish the effect of pruning from the effect of training.

## 7. We broadened the dataset

Sixteen images from one scene could not represent the range of indoor environments. We acquired a larger NYU-derived subset using bounded downloads, then grouped scenes conservatively to reduce overlap between splits. An initial acquisition encountered a path-handling error after downloading data; we preserved the failure and fixed the retry.

The resulting split contains **1,857 training pairs from 98 named scenes** and **445 tuning pairs from 26 named scenes**. We used the tuning set to compare candidates and make decisions; it is therefore not an untouched final test set. Exact-image and perceptual checks found no flagged cross-split matches under the recorded checks, but these do not prove the absence of similar images or overlapping physical locations. The community checkpoint's pretraining data is unknown and could overlap this NYU-derived data. We did not download the entire NYU dataset.

Comparing ten existing checkpoints on the 445 tuning images gave mean-image RMSE **0.300091** for the unmodified baseline and **0.306396** for the closest candidate, weight-selected 192 before recovery. That comparison did not establish an accuracy improvement.

## 8. We explored a scene-aware idea

To investigate a simple potential novel contribution, we tried scoring channel importance with attention to different scenes, alongside average ablation, weight-based and random selection. The idea is to avoid choosing channels solely because they perform well on a dominant scene. We used matched training settings and multiple seeds to compare the methods.

An initial result looked promising, but a same-seed replay varied by more than the small apparent advantage. We therefore repeated the comparison with deterministic settings. Three-seed mean tuning RMSE was **0.283054** for the trained original, **0.283326** for average ablation and **0.283186** for scene-aware selection. The differences are tiny, and the trained original was slightly better than scene-aware on this aggregate. The scene-aware candidate removed 6,376 parameters, approximately 0.47% of the baseline.

**What we learned:** an interesting idea is not automatically a demonstrated improvement or established novelty. We need stronger evidence and a related-work comparison before making either claim. Reproducibility checks can change the interpretation of a result.

## 9. We found a deployment accuracy question

For deterministic seed-42 models on the 445-image tuning set, FP16-enabled TensorRT and PyTorch had close mean-image RMSE: **0.282669 versus 0.282433** for the trained original, and **0.283390 versus 0.283273** for scene-aware. Yet their largest individual output differences were **2.110585** and **0.166729** native units respectively. These are differences between runtime predictions, not ground-truth errors. The cause and an appropriate acceptance tolerance remain unresolved.

Disabling FP16 in one earlier post-hoc desktop test reduced an outlier, but other settings were not isolated, so it did not prove the cause. Desktop latency varied across runs and does not establish Orin latency or an end-to-end speedup.

**What we learned:** averages can hide important individual discrepancies. An engine that builds and runs still needs per-input output checks, accuracy evaluation and measurements on the target device.

## 10. Where we stand

We have a working desktop baseline, data and evaluation tools, several controlled pruning experiments, ONNX exports, and a preliminary board-build script. We have useful negative results and a potential scene-aware idea to discuss, but no reliable demonstrated model improvement, authenticated original checkpoint, or Jetson measurement. The board's reported software versions come from screenshots, not direct board inspection.

Next we need to investigate TensorRT outliers and package a reproducible baseline test. When the Orin becomes available, we must verify its software, build its engine locally, compare predictions, and measure accuracy and both model-only and end-to-end latency under recorded conditions. The [Jetson software notes](learning/jetson_software.md) explain the version checks; the [oral defense guide](learning/oral_defense.md) covers questions about our choices.
