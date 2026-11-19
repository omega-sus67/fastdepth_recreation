# Jetson results

These measurements were collected on the Jetson Orin Nano on 16 November 2026.

| File or folder | What it contains |
| --- | --- |
| [comparison.csv](comparison.csv) | The main table: 12 tuning rows and six evaluation rows, including accuracy and latency. |
| [sweep/tuning](sweep/tuning/) | Results for baseline, width192, width176 and width160 in FP32, FP16 and INT8. Each folder contains a summary and timing samples. |
| [sweep/evaluation](sweep/evaluation/) | Results on the 654-image collection for baseline and the selected width176 student in the three modes. |
| [sweep/selection.json](sweep/selection.json) | The tuning-based selection decision and accuracy gates. |
| [sweep/build](sweep/build/) | Build logs and metadata. The 12 device-specific TensorRT engines are retained locally and excluded from Git. |
| [sweep/logs](sweep/logs/) | Console logs grouped into `build`, `tuning` and `evaluation`, with one file per configuration. |
| [sweep/status.csv](sweep/status.csv) and [evaluation-status.csv](sweep/evaluation-status.csv) | Which builds and evaluations completed. |
| [sweep/environment.log](sweep/environment.log) and [tegrastats.log](sweep/tegrastats.log) | Software, power-mode and sampled device telemetry. |
| [follow_up/profiling](follow_up/profiling/) | Repeat timings and layer profiles for baseline and width176. |
| [follow_up/image_comparisons](follow_up/image_comparisons/) | Saved paired PyTorch depth predictions for the report images. |

Only baseline and width176 were evaluated on the 654-image collection; all four widths were compared on the tuning set. The evaluation images were inspected previously and their depth units are unverified. FP32 was configured without explicitly disabling TF32; FP16 and INT8 can use mixed execution. Latency excludes preprocessing and transfers. INT8 lost accuracy, and the measurements do not establish a reliable pruning speedup.

The saved raw logs and summaries retain the paths recorded when the experiments ran; those old paths describe the original run and have not been rewritten. Current configuration files now use the renamed paths, so their file hashes differ from those recorded during the original run. The original files and paths can be inspected in Git commit `e3df169`.
