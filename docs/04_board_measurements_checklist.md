# Phase 3 — Board measurements checklist

Reference: [project plan](fastdepth_edge_ai_plan.md). Preserve raw measurements and document any unavailable configuration or deviation.

## Step 3.1 — Measure the Jetson Nano

**Read before interpreting results:** [FastDepth](../papers/fastdepth.pdf), PDF page 4, section IV-A and Table II, for how the paper reports TX2 inference and power. Record the Nano's own conditions; these are different hardware. Use installed NVIDIA documentation for engine building, power modes, and sensor paths.

- [ ] Record board model, JetPack, CUDA, TensorRT and PyTorch versions, power mode, cooling, clocks, and ambient/test conditions where available.
- [ ] Configure the Nano's maximum power mode and document the commands and settings actually used.
- [ ] Transfer models and test inputs; build each TensorRT engine on the Nano, recording build settings and errors.
- [ ] Benchmark all four models in PyTorch FP32, TensorRT FP32, and TensorRT FP16 using the same input and timing scope.
- [ ] Record median, p95, and p99 latency and raw samples for every supported cell; repeat runs and record variability.
- [ ] Measure memory and power for each cell; document the INA3221 sensor rail, sampling method, idle measurement, and measurement duration.
- [ ] Compute total and idle-subtracted energy per frame using a documented power and timing method.
- [ ] Save board-produced outputs and evaluate accuracy with the same validation data and metrics as the reference run.
- [ ] Compare TensorRT FP16 and FP32 outputs and accuracy, and investigate material differences.
- [ ] Record failed or unsupported cells and the reason rather than silently omitting them.

**Done when:** each supported model/runtime/precision cell has latency, power, memory, and an accuracy check from board outputs, with gaps explained.

## Step 3.2 — Measure the Arduino UNO Q

**Context:** [FastDepth](../papers/fastdepth.pdf), PDF pages 4–6, reports CPU and GPU measurements on a TX2, not an UNO Q. Consult the actual board and runtime documentation for supported execution and power measurement.

- [ ] Record the board's OS, hardware, available memory, runtime versions, and test conditions.
- [ ] Identify and install a compatible inference runtime on its A53 cores; document model conversion and execution commands.
- [ ] Benchmark FP32 models where supported using the same timing scope and input shape; save raw samples and latency statistics.
- [ ] Test a suitable INT8 path with representative calibration data if supported; compare accuracy and latency with FP32 and record a yes/no conclusion with evidence.
- [ ] Choose and document a power measurement method, including any external meter, measurement boundary, idle draw, and limitations.
- [ ] Record latency and power for at least one FP32 runtime; document unsupported models or unavailable measurements.

**Done when:** FP32 latency and power are recorded for at least one runtime, and the INT8 result or a concrete feasibility limitation is documented.

## Step 3.3 — Draw the optimization ladder

**Read while plotting:** [FastDepth](../papers/fastdepth.pdf), PDF pages 4–6, Figure 3 and Tables I, V–VII, for examples of showing the effects of successive design choices. Plot your own Nano measurements and label the timing scope.

- [ ] Join each measured configuration's latency and RMSE using the matching model, precision, and evaluation setup.
- [ ] Plot latency against RMSE with clear model/runtime/precision labels and the 33 ms target line.
- [ ] State which configurations meet the budget and quantify their accuracy trade-offs.
- [ ] Save the plot, underlying data, and generation command.

**Done when:** the plot makes the budget outcome and accuracy cost readable for each measured configuration.
