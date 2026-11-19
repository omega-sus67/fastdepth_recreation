# Script guide

Run desktop Python commands from the repository root with `bash scripts/python.sh scripts/<script>.py`. Most scripts accept `--help`. Use a new output directory for each run.

## Core workflow

| Purpose | Scripts |
| --- | --- |
| Set up and check the desktop environment | `setup_desktop.sh`, `python.sh`, `check_environment.py` |
| Acquire and verify the baseline | `download_checkpoints.py`, `convert_checkpoint.py`, `check_models.py` |
| Prepare and check the dataset | `download_validation.sh`, `extract_validation.py`, `acquire_diverse_nyu.py`, `regroup_nyu_scenes.py`, `verify_scene_split.py` |
| Create and recover a student | `prune_candidate.py`, `train_distill.py` |
| Export and calibrate models | `export_onnx.py`, `prepare_int8_calibration.py`, `quantize_int8.py` |
| Prepare Jetson inputs and references | `prepare_jetson_tuning.py`, `prepare_jetson_evaluation.py`, `prepare_jetson_references.py` |
| Package and run the sweep | `package_latency_sweep.sh`, `build_jetson_engine.sh`, `build_jetson_qdq.sh`, `run_latency_sweep_jetson.sh` |
| Evaluate and select on the board | `evaluate_jetson_bundle.py`, `select_latency_candidate.py`, `run_orin_followup.sh` |
| Run image and video demos | `infer_image.py`, `demo_simple.py`, `infer_video.py`, `compare_video.py` |
| Make visual comparisons | `compare_visual_photos.py`, `make_poster_depth_figure.py` |

The baseline and selected width176 student are the defaults in the image and video demos. `prepare_jetson_references.py` defaults to the same pair; supply repeated `--model NAME=CHECKPOINT` arguments when preparing references for all four sweep widths.

The 12-configuration sweep was run on the Jetson Orin Nano. See [results](../results/README.md) for its saved measurements. The prepared `jetson_handoff/` folder contains its own guide and data; it is kept locally rather than pushed to GitHub. Video inference is a desktop-tested PyTorch application, not a TensorRT or Jetson video benchmark.
