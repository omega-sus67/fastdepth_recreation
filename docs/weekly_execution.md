# Execution schedule — 13–15 November

Estimates are active working time, not guarantees; data downloads, training, debugging and hardware access can change them. The deadline is Sunday 15 November. Poster and final report production are out of scope at the user's request. Maintain `progress_report.md` and topic documents as work is verified.

1. **Friday:** audit validation data, implement preprocessing and depth metrics, run baseline evaluation, and document dataset, architecture and metrics (estimated 8–12 hours).
2. **Saturday:** image/video inference, ONNX export and parity, desktop benchmarks, training-data acquisition and resumable fine-tuning (estimated 8–12 hours).
3. **Sunday:** implement and test practical pruning/precision candidates, produce controlled comparisons, package Orin setup and document deployment and limitations (estimated 8–12 hours).

Hardware-dependent work is pending until an Orin Nano is available; desktop measurements cannot establish board improvements. The original three unpruned checkpoints and the recovered checkpoint's original-author provenance remain unresolved. If the training data or original checkpoints remain unavailable, prioritize a validated application, reproducible benchmark and clearly labeled limitations over unsupported improvement claims.

Write separate detailed files under `docs/learning/` for setup, dataset and preprocessing, model architecture, metrics, training, pruning, quantization, deployment and experiments as the corresponding work proceeds. Each must distinguish implemented behavior, planned work, assumptions and measured results.
