# Script navigation

Run project Python scripts from the repository root with `bash scripts/python.sh scripts/<name>.py ...`; the wrapper selects the independent desktop environment and avoids unrelated Python paths. Most scripts offer `--help`. See the setup notes (local documentation), experiment index (local documentation) and [working rules](../agents.md) for context and reproducibility requirements.

## Setup and models

| Script | Purpose |
| --- | --- |
| [setup_desktop.sh](setup_desktop.sh), [python.sh](python.sh) | Create/check the desktop environment and run commands with the project interpreter. |
| [check_environment.py](check_environment.py) | Check installed packages, runtime availability and environment isolation. |
| [download_checkpoints.py](download_checkpoints.py), [convert_checkpoint.py](convert_checkpoint.py) | Acquire/check the supported checkpoint sources and convert the recovered legacy checkpoint. |
| [check_models.py](check_models.py) | Check model structures and supported checkpoint behavior. |

## Data acquisition and checks

| Script | Purpose |
| --- | --- |
| [download_validation.sh](download_validation.sh), [extract_validation.py](extract_validation.py) | Download and extract the local validation archives. |
| [download_training_subset.py](download_training_subset.py), [acquire_scene_subset.py](acquire_scene_subset.py) | Earlier bounded training-sample acquisition approaches. |
| [acquire_multiscene_subset.py](acquire_multiscene_subset.py) | Acquire small scene-grouped samples from pinned training-shard prefixes. |
| [acquire_diverse_nyu.py](acquire_diverse_nyu.py), [regroup_nyu_scenes.py](regroup_nyu_scenes.py) | Sample bounded windows across pinned shards and conservatively regroup possible same-location scene variants. |
| [inspect_validation.py](inspect_validation.py), [audit_validation.py](audit_validation.py) | Inspect sample files and audit the local evaluation collection. |
| [verify_validation_archive.py](verify_validation_archive.py), [verify_preprocessing.py](verify_preprocessing.py) | Check downloaded archive members and compare tested preprocessing against adapted SciPy source. |
| [verify_scene_split.py](verify_scene_split.py) | Verify sample hashes, scene separation and exact RGB overlap with the local evaluation images. |
| [audit_near_duplicates.py](audit_near_duplicates.py) | Flag visually similar train/tuning/evaluation image pairs using a coarse perceptual hash; flags require review. |
| [audit_official_nyu_identity.py](audit_official_nyu_identity.py), [audit_nyu_scene_mapping.py](audit_nyu_scene_mapping.py) | Match bounded samples against the official NYU labeled file and investigate an inferred evaluation-scene mapping using remote byte ranges. |

## Training and pruning

| Script | Purpose |
| --- | --- |
| [train_smoke.py](train_smoke.py) | Earlier small training/resume smoke test on correlated samples. |
| [train_scene.py](train_scene.py) | Run a controlled scene-grouped training and tuning experiment. |
| [score_ablation.py](score_ablation.py) | Score decoder channels by individual training-loss ablations. |
| [score_scene_aware.py](score_scene_aware.py), [build_scene_candidates.py](build_scene_candidates.py) | Score individual channels across training scenes and create equal-width comparison candidates. |
| [audit_scene_runs.py](audit_scene_runs.py), [summarize_scene_experiment.py](summarize_scene_experiment.py) | Check matched recovery run identities and summarize paired results by inferred location group. |
| [prune_candidate.py](prune_candidate.py) | Create a physically smaller candidate using supported channel-selection rules. |
| [train_scene_batch.py](train_scene_batch.py) | Run the fixed one-epoch batched recovery protocol; saves optimizer/RNG state but does not implement resume. |

## Inference, evaluation and deployment

| Script | Purpose |
| --- | --- |
| [infer_image.py](infer_image.py), [infer_video.py](infer_video.py) | Run desktop image or video depth inference. |
| [evaluate_validation.py](evaluate_validation.py), [evaluate_tensorrt.py](evaluate_tensorrt.py) | Evaluate the PyTorch model or a supported TensorRT engine. |
| [compare_tuning_models.py](compare_tuning_models.py) | Evaluate existing checkpoints on a scene-grouped tuning manifest and save per-image, per-scene and pooled metrics. |
| [compare_tensorrt_tuning.py](compare_tensorrt_tuning.py) | Compare desktop TensorRT engines with PyTorch checkpoints on the tuning manifest. |
| [export_onnx.py](export_onnx.py) | Export a supported model for runtime validation and deployment. |
| [benchmark_pytorch.py](benchmark_pytorch.py), [benchmark_tensorrt.py](benchmark_tensorrt.py) | Save model-only desktop timing samples and summaries. |
| [replay_tensorrt_timing.py](replay_tensorrt_timing.py) | Repeat timing on already-built desktop engines in a recorded order. |
| [build_jetson_engine.sh](build_jetson_engine.sh) | Prepare a guarded TensorRT engine build on a Jetson; it has not yet been tested on the target board. |
| [run_tensorrt_smoke.py](run_tensorrt_smoke.py) | Run a fixed synthetic input through a local engine and compare numerical output; tested on desktop, not Jetson. |

Use a new output path for each experiment. The script's help and saved configuration define exact arguments; examples in training (local documentation) and the experiment index (local documentation) show existing workflows. Desktop setup and engine results do not establish Jetson compatibility or performance.
