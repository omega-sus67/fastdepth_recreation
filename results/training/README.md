# Trained students

- `width_192/`: 192-channel student.
- `width_176/`: 176-channel student selected on the Jetson tuning results.
- `width_160/`: 160-channel student.

Each folder has a PyTorch `checkpoint.pth` and its original training `summary.json`. The baseline checkpoint is `../../checkpoints/converted/community-skipadd-pruned.pth`. The precision modes are made during export and TensorRT engine building; these are four sets of model weights, not 12 separately trained models.
