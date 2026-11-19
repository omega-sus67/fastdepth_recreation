# Jetson Orin sweep, 16 November 2026

- [Comparison table](comparison.csv): 12 tuning rows and six evaluation rows.
- [Tuning results](results/orin-sweep-visit/tuning/): all four model widths in FP32, FP16 and INT8.
- [Evaluation results](results/orin-sweep-visit/evaluation/): baseline and selected width176 in the three modes.
- [Selection](results/orin-sweep-visit/selection.json): why width176 was selected from the tuning data.
- [Status](results/orin-sweep-visit/status.csv), [evaluation status](results/orin-sweep-visit/evaluation-status.csv) and [environment](results/orin-sweep-visit/environment.log).
- [Builds](results/orin-sweep-visit/build/): build metadata and logs; board-specific engines are retained locally and ignored by Git.

These are actual device results. FP32 is configured without explicitly disabling TF32; FP16 permits mixed precision and explicit INT8 does not imply every operation runs in INT8. Latency is model-only. The evaluation collection was inspected previously and depth units are unverified.
