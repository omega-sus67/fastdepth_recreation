# Deployment and benchmarking

## Desktop TensorRT check of the scene-aware experiment

Five fixed seed-42 models were exported and built as TensorRT FP16-enabled engines on the RTX 3060. All 445 tuning images were compared against the matching PyTorch models, and saved engines were timed twice in ordered forward/reverse passes. The original engine's median changed from 0.388433 to 0.324522 ms across passes, and TensorRT warned that the default CUDA stream can add synchronization; these measurements do not establish a reliable speedup. Differences of up to 1.396948 native depth units occurred at individual output pixels even though aggregate RMSE was close. See the [scene-aware note](scene_aware_pruning.md#desktop-tensorrt-accuracy-and-timing--14-november) for the full evidence and explanation. These are desktop model-only tests; the board engine and end-to-end timing remain pending.

**Status: desktop model export and exploratory TensorRT checks completed; Orin measurements pending.** ONNX stores a computation graph and weights. An exporter traces/captures a model and translates its operations; a validation step checks the graph and compares numerical output with the PyTorch model on identical inputs. An ONNX file is the portable handoff, while TensorRT builds an engine optimized for the target runtime and hardware. Build the final engine on the Orin and record JetPack, CUDA, TensorRT, power mode and build settings. See [reported target-board versions](jetson_software.md) and their verification limits before preparing board-specific commands.

For fair timing, use batch size 1 and the same input sequence. Warm up the runtime, synchronize GPU work at the timing boundary, retain individual samples and report median, p95 and p99. Model-only latency excludes image preparation and transfers; the project's 33 ms target is p95 from decoded RGB in host memory through preprocessing, transfers, inference and numerical depth output. Camera capture and display are separately measured in complete-demo FPS. Never compare timings with different boundaries as if they represented the same work.

Board power and energy require a documented sensor or external meter, measurement boundary, idle baseline, sample duration and number of completed frames. Total energy per frame is integrated power divided by completed frames; idle-subtracted energy uses a matching idle baseline. Cooling and clock/power modes affect sustained performance. A desktop TensorRT engine or desktop latency does not prove Orin portability, speed, power or accuracy.

## Desktop evidence — 13 November

The community pruned checkpoint was exported to ONNX opset 18. On one local validation image, PyTorch CPU and ONNX Runtime CPU outputs differed by at most 3.5763e-6, below the chosen 1e-4 absolute tolerance. This one-sample comparison does not establish full-set parity. `exports/2026-11-13-community-pruned/parity.json` records the exact model hashes and settings.

A PyTorch FP32 model-only desktop test with 50 warm-ups and 500 synchronized calls on an RTX 3060 Laptop GPU recorded median 5.405 ms and p95 5.664 ms. This uses a preallocated all-zero tensor and excludes preprocessing and transfers, so it cannot be compared directly to the project's 33 ms end-to-end target or to Orin results. Raw samples and metadata are in `results/benchmarks/2026-11-13-desktop-pytorch/`.

The image CLI accepts 480×640 RGB inputs and saves raw NumPy depth plus a separately normalized color preview. The video CLI resizes frames to 640×480, which can distort their aspect ratio, and writes a per-frame-normalized preview with no audio. A three-frame generated video smoke test ran; its short-run FPS is not a benchmark. Run `bash scripts/python.sh scripts/infer_image.py <image> --output <new-directory>` or the analogous `infer_video.py` command. Webcam handling is still pending.

## Why compare independent runs?

A single sequence of 500 calls describes latency within one run, but the device can change clock, temperature and background load between runs. We therefore alternated the unmodified and structurally pruned models three times on the desktop; raw samples and per-run percentiles are saved separately. These were not randomized and power/thermal conditions were not controlled, so the small differences are descriptive rather than evidence of a statistically reliable improvement. See [pruning](pruning.md) for the actual values. Orin trials must use their own board, supported power mode and full-pipeline measurement.

## Preparing a board-built engine

`scripts/build_jetson_engine.sh` accepts an ONNX file, a new output directory and `fp32` or `fp16`. It checks that it is running on an AArch64 system with `/etc/nv_tegra_release`, locates `trtexec`, saves the ONNX hash and environment, then invokes the TensorRT 10.x `--onnx`, `--saveEngine`, `--skipInference` flags; the FP16 option adds `--fp16`. It saves the exact command, full build log and engine hash. NVIDIA's [10.x migration reference](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/api/tensorrt-8x-to-10x-trtexec.html) documents the relevant build-only flag; [NVIDIA's TensorRT reference](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-trtexec.html) shows the 10.x precision flag.

Example **for the actual board, once its versions are confirmed**:

```bash
bash scripts/build_jetson_engine.sh exports/2026-11-13-community-pruned/model.onnx results/jetson/build-fp16 fp16
```

The script was syntax-checked and correctly rejected execution on the x86 desktop. It has not been run on the Orin; this is a preparation artifact, not evidence that the board parser accepts the graph or that a board engine works. After building, compare board outputs and validation metrics against the same reference inputs before benchmarking. The default `fp32` configuration does not explicitly disable TensorRT TF32; inspect and record the actual board configuration before claiming strict FP32 arithmetic.
