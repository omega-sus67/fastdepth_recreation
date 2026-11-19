# Dataset checks

- `training_tuning_overlap.json`: the split and exact-overlap checks used by the sweep configuration.
- `near_duplicates.json` and `scene_groups.json`: additional dataset checks.
- `evaluation_dataset_audit/`: audit of the 654 evaluation files, including sample inventory and depth distribution.
- `source_and_preprocessing/`: source and preprocessing verification records.

The actual datasets are kept locally in `../../data/nyu_depth_v2/` and are not pushed to GitHub. Scene independence and the depth-unit conversion are not fully verified.
