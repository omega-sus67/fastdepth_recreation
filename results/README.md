# Results: start here

This folder contains the project’s saved results, organized by purpose:

| Folder | What you will find |
| --- | --- |
| [jetson_results](jetson_results/) | The actual Orin measurements: comparison table, all 12 tuning runs, six evaluation runs, selection, logs and follow-up. |
| [training](training/) | The three trained student checkpoints: `width_192`, `width_176` and `width_160`. |
| [data](data/) | Dataset, split, overlap and preprocessing checks. |
| [checkpoints](checkpoints/) | Baseline checkpoint provenance and verification. |
| [figures](figures/) | The poster figure and provenance for the report comparison images. |
| [inference](inference/) | An example paired video and its sample input. |
| [evidence_index.csv](evidence_index.csv) | Paths and SHA-256 fingerprints of retained evidence. |

The baseline is in `../checkpoints/converted/`. The 176-channel student was selected on the tuning set. The 654-image evaluation collection was previously inspected; it is not an untouched test. See the [Jetson results guide](jetson_results/README.md) for what the numbers mean.
