# FastDepth edge AI project

Start with the [task checklists](docs/README.md) and [paper index](papers/README.md). The local setup targets the desktop's NVIDIA GPU; it does not install Jetson packages.

## Start here

```bash
source .venv/bin/activate
python scripts/check_environment.py
python scripts/inspect_validation.py  # inspect the validation data
```

The project's `.venv` reuses PyTorch 2.9.1, torchvision 0.24.1, and CUDA 13 libraries from `/home/omega_sus/droidrun/venv`. Its `edgeai_shared_env.pth` file adds that environment's Python 3.12 packages to the import path; FastDepth's extra packages are installed locally. Activate this project's environment as usual. The shared environment was not modified and must remain at that location. Changes to its packages can also affect this project. The environment check prints where PyTorch and torchvision were loaded from.

Running `scripts/setup_desktop.sh` on this machine preserves the configured shared environment and checks it without downloading another GPU stack. On a new machine without that configuration, the script creates an independent environment using Python 3 and CUDA 12.8 PyTorch wheels; a compatible NVIDIA driver and [uv](https://docs.astral.sh/uv/getting-started/installation/) are required. `requirements-desktop-tested.txt` records the original independent installation before sharing was enabled; it is not the current environment's lock file. Desktop wheels do not work on Jetson Nano: install a PyTorch build and TensorRT compatible with the board's actual JetPack separately.

## Layout

- `docs/` — plan and task-by-task checklists.
- `papers/` — the three reference papers.
- `third_party/fast-depth/` — original upstream source, pinned by the setup script to commit `e68492011609c9bfb7de6d402da5d1d201d95bd9`; includes an example RGB/depth pair under `deploy/data/`.
- `data/nyu_depth_v2/raw/` — downloaded validation archives; `scripts/download_validation.sh` resumes their download. The dataset card describes 654 validation examples sourced from FastDepth, but split identity and preprocessing still need verification against the original evaluation.
- `checkpoints/` — pretrained models; see its README for the download blocker.
- `notebooks/`, `src/`, `scripts/`, `tests/` — project work.
- `exports/`, `results/` — generated models and measurements. The data inspection preview is `results/validation_sample.png`.

The 654 validation files have also been extracted under `data/nyu_depth_v2/extracted/val`; recreate them with `python scripts/extract_validation.py`. The upstream evaluation script assumes a different relative data path, so it will need an adapter or path adjustment before use.

The full training data is not downloaded. Large datasets, generated outputs, the virtual environment, checkpoints, and the upstream checkout are ignored by Git. The original FastDepth code targets PyTorch 0.4.1; loading its checkpoints and running its evaluation code on current PyTorch and NumPy remain project tasks (for example, NumPy 2 removed `np.asfarray`, which its loader uses), not verified capabilities of this setup.
