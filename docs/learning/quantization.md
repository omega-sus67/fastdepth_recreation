# Precision and quantization

## Desktop FP16-enabled comparison on 445 tuning images

Five seed-42 models were exported and run through TensorRT 10.3 on the desktop; FP16 tactics were enabled, without assuming that every layer used FP16. Their aggregate mean-image RMSE remained close to PyTorch FP32, but some individual pixel predictions differed substantially: up to 1.396948 native depth units for the original model. This is a runtime-output difference, not a ground-truth error or an Orin result. The [scene-aware experiment note](scene_aware_pruning.md#desktop-tensorrt-accuracy-and-timing--14-november) gives all model pairs, outlier counts, the arbitrary diagnostic threshold, raw records and limitations. This was not INT8 quantization and does not establish precision suitability on Orin.

**Status: one desktop FP32 versus FP16-enabled TensorRT comparison and full local accuracy evaluation completed; Orin tests remain pending.** FP32 and FP16 are floating-point formats with different precision and range. TensorRT may execute suitable operations in FP16 when configured to do so. FP16 can change intermediate activations and therefore the final predicted depth. A correct comparison needs the same input, model weights and preprocessing, plus output differences and full-set accuracy; successful engine construction alone is insufficient.

INT8 uses integer representations and scale information to approximate real values. Post-training quantization commonly determines activation scales from representative calibration examples. Calibration data must come from training data, not the held-out final evaluation set. Quantization-aware training is another approach that models quantization during training. Either path requires checking that the chosen JetPack/TensorRT version supports the graph and precision.

Orin Nano supports INT8 acceleration, but this does not guarantee a speedup for every model or preserve depth accuracy. Compare FP32, FP16 and any INT8 candidate with matched timing boundaries, accuracy metrics, calibration settings and full engine logs. Synthetic desktop TensorRT FP32 execution has passed; FastDepth FP16/INT8 and any Orin result remain unmeasured.

## Initial desktop experiment — 13 November

Both TensorRT engines were built from the same ONNX file on the RTX 3060, with batch size 1, 50 warm-ups and 500 synchronized model-only samples using a preallocated zero tensor. FP32 median latency was 0.520262 ms (p95 0.549775 ms); FP16-enabled median was 0.391760 ms (p95 0.409146 ms). This is a single desktop run for each configuration. TensorRT warned that using the default CUDA stream may add synchronization overhead; the warning was observed in the console and not captured in a separate raw log. The FP32 engine's TF32 setting was not recorded. These constraints limit interpretation; neither timing is the end-to-end target or an Orin result.

On the identical 654-file manifest, FP16-enabled mean per-image RMSE was 0.603775 and delta1 0.771055, versus PyTorch FP32 0.603737 and 0.771065, respectively, in native unverified depth units. Pooled RMSE was 0.721116 versus 0.721159. FP16 flags permit a mixture of FP16/FP32 tactics; the engine is not necessarily entirely FP16. No INT8 model has been built. Source summaries and per-image results are in `results/benchmarks/` and `results/evaluation/`.
