# Results for the 12 Jetson configurations

Start with [comparison.csv](jetson_orin_2026-11-16/comparison.csv). It contains the actual results for the four model widths and three precision modes: **12 tuning rows**, followed by **six evaluation rows** for the baseline and the selected width176 student. Width192 and width160 were not evaluated on the 654-image collection in that run.

| Location | Contents |
| --- | --- |
| [Jetson sweep](jetson_orin_2026-11-16/results/orin-sweep-visit/) | The original build, tuning and evaluation logs, 12 tuning results, six evaluation results, selection record and telemetry. The build subfolders also contain the original board-specific engines locally. |
| [Follow-up](jetson_followup_2026-11-16/) | Baseline and width176 timing repeats, profiles and saved paired predictions. |
| [Training](training/) | The three trained student checkpoints and their training summaries. The baseline is in `../checkpoints/converted/`. |
| [Figures](figures/) | The poster comparison and provenance of the six report comparisons. |
| [Inference](inference/) | One paired video smoke test and its synthetic input. |
| [Data checks](data/) and [checkpoint checks](checkpoints/) | Evidence about the data split, preprocessing and baseline provenance. |
| [Evidence index](evidence_index.csv) | Fingerprints of the retained evidence. |

Each tuning or evaluation subfolder contains its own `summary.json` and timing samples. The selection record chooses width176 using the FP16 tuning measurements and the project's accuracy and timing-variability rules. The 654-image collection had been inspected previously; it is not an untouched test. INT8 lost accuracy and the measurements did not establish a reliable pruning speedup.

Earlier unrelated results were removed from the working directory. Tracked historical files can be inspected at Git commit `6c947dd`; ignored or previously untracked files are not guaranteed recoverable. The local progress report records the project's development and the cleanup.
