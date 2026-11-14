# Results and evidence

Start with the [experiment index](../docs/experiment_index.md) for a question-by-question overview, the relevant settings, checkpoints, outcomes and links to raw measurements. For the history of decisions, failures and limitations, read the [progress report](../docs/progress_report.md). The [evidence index](evidence_index.csv) maps stable evidence IDs to artifact paths and SHA-256 fingerprints.

| Folder | What it contains |
| --- | --- |
| [evaluation/](evaluation/) | Desktop model accuracy summaries, evaluation manifests and per-image results. |
| [training/](training/) | Fine-tuning and pruning comparisons, trained checkpoints and tuning metrics. |
| [benchmarks/](benchmarks/) | Desktop model-only latency summaries and raw timing samples; some folders contain desktop TensorRT engines. |
| [data/](data/) | Dataset audits, archive checks, preprocessing verification and split-overlap reports. |
| [checkpoints/](checkpoints/) | Checkpoint acquisition, conversion, provenance and structural checks. |
| [inference/](inference/) | Image and video inference examples. |
| [environment/](environment/) | Desktop installation and runtime checks. |
| [history/](history/) | Preserved snapshots, earlier checks and test outputs. |

Results are organized by type and then run date/name. Similar-looking folders can contain different configurations or controls: read the run's `summary.json` and its configuration before comparing values. `latencies_ms.csv` contains individual timing samples, while a benchmark summary specifies the runtime, scope, warm-up and device. Desktop results do not establish Jetson Orin Nano performance.

Keep earlier attempts, including negative findings and failed runs. Do not overwrite a run directory or edit an indexed evidence artifact to make an updated result: create a new run, document its relationship to earlier work, and add its evidence fingerprint. The [project rules](../agents.md) define the full procedure.
