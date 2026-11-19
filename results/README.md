# Results and evidence

Start with the experiment index (local documentation) for a question-by-question overview, the relevant settings, checkpoints, outcomes and links to raw measurements. For the history of decisions, failures and limitations, read the progress report (local documentation). The [evidence index](evidence_index.csv) maps stable evidence IDs to artifact paths and SHA-256 fingerprints.

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

The current sweep checkpoints and distillation controls are retained. Superseded checkpoint binaries, some old desktop engines and intermediate poster layouts were retired on 19 November 2026; their summaries and measurement records remain. [Cleanup manifest](history/2026-11-19-cleanup/manifest.json) records the paths and hashes. Historical configurations referencing retired binaries need those files restored or regenerated before they can run again. A Git-tracked file may be recoverable from a commit that contains it; an untracked file is not necessarily recoverable.

Keep earlier attempts, including negative findings and failed runs. Do not overwrite a run directory or edit an indexed evidence artifact to make an updated result: create a new run, document its relationship to earlier work, and add its evidence fingerprint. The [project rules](../agents.md) define the full procedure.
