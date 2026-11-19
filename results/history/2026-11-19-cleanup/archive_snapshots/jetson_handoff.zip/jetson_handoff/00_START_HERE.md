# Start here: FastDepth on the Jetson Orin Nano

This folder is the complete **prepared experiment handoff**. Copy the whole `jetson_handoff` folder to the Jetson Orin Nano. It contains the model files, 445 tuning RGB/depth pairs, 654 evaluation pairs, reference predictions, code and this guide. You do **not** need to download the dataset or train models on the board. You may need internet to install software that is missing on the Jetson.

The board experiment has not yet been run. The baseline is a community checkpoint that was already pruned before we obtained it. The 192, 176 and 160 variants prune it further and were trained on the PC. INT8 accuracy was substantially worse in our desktop checks, so treat those runs as experiments, not confirmed improvements. We do not yet know the Jetson latency or energy use.

## Which images are included and how are they tested?

The images needed for the accuracy comparison **are included**. They have already been converted from their original files into arrays so the Jetson can load them without downloading the dataset:

- `data/nyu_depth_v2/jetson-tuning-445/inputs.npy` contains 445 prepared RGB images; `targets.npy` contains their corresponding ground-truth depth maps. These are used to compare and select candidate models.
- `data/nyu_depth_v2/jetson-evaluation-654/inputs.npy` contains 654 prepared RGB images; `targets.npy` contains their corresponding ground-truth depth maps. These are used to evaluate the baseline and selected student after selection.

For every pair, the test feeds the RGB image to the model, compares its predicted depth map with the included ground-truth map, and calculates RMSE, δ1 and the other metrics. The images are stored together in `.npy` arrays, not as individual JPEG or PNG files. The full original training dataset is not included because the models have already been trained. An additional ordinary photograph is only needed for the optional visual demo in section 7; it is not needed for the accuracy test.

## Before you go

1. Copy the entire `jetson_handoff` folder to a USB drive or other storage. Keep the folders and filenames unchanged. The prepared tuning and evaluation images are included; do not upload this folder to GitHub. A USB drive with at least 8 GB free is a practical starting point; the board also needs free space for the folder and the engines it builds. Actual engine sizes and build time are unknown.
2. Bring the Jetson power supply, cooling, display and keyboard, your USB drive and an internet connection if possible. Plan for a long uninterrupted session: there are 12 engine builds plus evaluation. Keep power mode and cooling consistent throughout.
3. Eight 640×480 evaluation photographs are already included for the visual comparison in section 7. You can optionally bring your own image for the separate one-image demo.

## 1. Copy the folder onto the Jetson

Use the file manager to copy `jetson_handoff` from the USB drive to the Jetson user's home folder. Open a terminal and run:

```bash
cd ~/jetson_handoff
pwd
sha256sum --quiet -c SHA256SUMS
```

The checksum check may take a few minutes. No output and exit status `0` means all listed files match. Check the status immediately:

```bash
echo $?
```

If a file fails or is missing, recopy it before continuing. Every later command assumes the terminal is in `~/jetson_handoff`. If you copied it elsewhere, replace the `cd` path accordingly. You do not need to extract an archive.

## 2. Record what is installed

```bash
mkdir -p results
{ date -Is; uname -a; cat /etc/nv_tegra_release; cat /proc/device-tree/model; printf '\n'; python3 --version; dpkg-query -W nvidia-l4t-core nvidia-jetpack 'libnvinfer*' 'python3-libnvinfer*'; free -h; df -h .; } 2>&1 | tee results/board-info.txt
command -v trtexec || ls -l /usr/src/tensorrt/bin/trtexec
python3 -c "import numpy, tensorrt, torch; print('NumPy:',numpy.__version__,'TensorRT:',tensorrt.__version__,'PyTorch:',torch.__version__,'CUDA:',torch.version.cuda,'GPU available:',torch.cuda.is_available()); assert torch.cuda.is_available(); print((torch.ones(2, device='cuda') + 1).cpu())"
```

The last command must print `GPU available: True` and a tensor. The build tool must either appear on PATH or exist at `/usr/src/tensorrt/bin/trtexec`. The runner uses `python3`; if you activate a virtual environment, verify that its `python3` passes this exact command. The information photographed earlier is not a substitute for these checks: use the versions this board reports.

## 3. Install missing software only if the checks fail

First inspect the board's JetPack repositories and simulate installing common tools:

```bash
apt-cache policy libnvinfer-bin python3-libnvinfer python3-numpy
sudo apt-get update
sudo apt-get -s install libnvinfer-bin python3-libnvinfer python3-numpy
```

Read the simulation. If the packages are compatible with the installed JetPack and the proposed changes are appropriate for the board, install:

```bash
sudo apt-get install libnvinfer-bin python3-libnvinfer python3-numpy
```

If CUDA-enabled PyTorch is missing, use NVIDIA's [Jetson PyTorch installation guide](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) and choose a wheel for the **actual JetPack, ARM64, Python and CUDA versions** recorded above. A desktop PyTorch wheel or an arbitrary `pip install torch` may not support the Jetson GPU. Some wheels have additional prerequisites; check their release notes. The following creates a virtual environment that can see the board's system TensorRT Python bindings:

```bash
sudo apt-get install python3-venv python3-pip libopenblas-dev
python3 -m venv --system-site-packages ~/fastdepth-jetson-venv
source ~/fastdepth-jetson-venv/bin/activate
python -m pip install --no-cache-dir 'numpy<2'
```

Once you have positively identified the compatible NVIDIA wheel, set `JETSON_TORCH_WHEEL` to its URL or local wheel path and install it:

```bash
python -m pip install --no-cache-dir "$JETSON_TORCH_WHEEL"
python3 -c "import numpy, tensorrt, torch; print(numpy.__version__, tensorrt.__version__, torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available(); print((torch.ones(2,device='cuda')+1).cpu())"
```

Keep that virtual environment active for the following commands. If you cannot identify a compatible wheel, preserve the board inventory and installation errors; do not guess. `docs/jetson_first_test.md` has more background in its dependency section, although its older experiment commands are superseded by this guide.

## 4. Run the whole comparison

```bash
cd ~/jetson_handoff
bash scripts/run_latency_sweep_jetson.sh results/orin-sweep-visit
```

Use this command once the checks pass. The output directory must not exist beforehand. On a retry, change `orin-sweep-visit` to a **new** name and preserve the previous directory. The script:

1. Records the board environment and, when available, `tegrastats` output.
2. Builds baseline, width192, width176 and width160 in FP32-configured, FP16-enabled and explicit INT8 modes on the Jetson.
3. Evaluates each successful engine on the same 445 tuning samples and records accuracy, runtime differences and three timing rounds. Each round uses 50 warmups and 300 timed inferences.
4. Selects an eligible student using the recorded FP16 tuning accuracy and target timings; if some builds fail, selection only covers successful students.
5. Evaluates the baseline and selected student with available precision engines on the separate 654-file collection.

The script continues after individual failures and records them. If it exits with a failure code, do not delete the output: partial results and logs are useful. Keep the computer powered and cooled, and avoid other GPU-heavy tasks.

## 5. Check and save the results

```bash
cat results/orin-sweep-visit/status.csv
cat results/orin-sweep-visit/selection.json
cat results/orin-sweep-visit/evaluation-status.csv
ls results/orin-sweep-visit
```

`status.csv` lists each model/precision build and tuning outcome. `selection.json` identifies the chosen width, gates and missing candidates. `evaluation-status.csv` reports the baseline and selected student's evaluation outcomes. If selection failed, inspect `selection.log`. If a build or evaluation failed, read the corresponding `*_build.log`, `*_tuning.log` or `*_evaluation.log` and preserve it.

Inside `tuning/` and `evaluation/`, each successful model directory contains `summary.json` (pooled RMSE, MAE, REL, δ1, R² and latency summaries), `per_image.csv` (image-level measurements) and `latency_ms.csv` (individual timed runs). The `build/` directories hold board-built engines and build logs. `environment.log` records versions and power mode; `tegrastats.log` is present if the tool was available.

Copy the **entire** `results/orin-sweep-visit` folder and `results/board-info.txt` to your USB drive and bring them back to the PC, including failed logs. If you used another output name, copy that folder instead.

## 6. What the numbers do and do not show

- Lower RMSE is better; higher δ1 is better. Compare the same metric, precision and sample collection. Depth units of the source mirror are not independently verified.
- Model-only timing is synchronized inference with preallocated GPU tensors. It excludes decoding, resizing, host-device copies and display. Prepared-input-to-CPU-output timing includes transfers but still excludes full image processing.
- FP16-enabled permits mixed precision; explicit INT8 Q/DQ does not prove every operation executes in INT8. The INT8 exports lost substantial accuracy on desktop. Compare the Jetson results rather than assuming improvement.
- Both collections have been inspected before; the 654-file set is not a pristine test. Baseline pretraining overlap and complete scene independence are unknown.
- The timing order is fixed and can be affected by temperature and clocks. `tegrastats` alone does not establish energy per inference or peak model memory; those need a defined separate measurement method.

## 7. Create eight visual comparisons for the report

The folder `visual_photos/` contains eight real 640×480 RGB photographs from the same NYU-derived 654-file evaluation collection. Their source identities, hashes and selection method are recorded in `visual_photos/manifest.json`. These are previously inspected evaluation images; scene identities and the original checkpoint's training overlap are unknown. They were selected at evenly spaced manifest positions, without inspecting the model outputs, and are not independent new evidence.

After section 4 has selected a student, create all eight comparison figures using **PyTorch** on the Jetson. These are not TensorRT output images or a latency measurement. This visual step additionally needs a torchvision build compatible with the Jetson PyTorch, Pillow, h5py and Matplotlib. Check first:

```bash
python3 -c "import torch, torchvision, PIL, h5py, matplotlib; print('Image-demo imports OK')"
```

If system packages are missing, inspect and simulate the available packages before installation:

```bash
sudo apt-get -s install python3-pil python3-h5py python3-matplotlib
sudo apt-get install python3-pil python3-h5py python3-matplotlib
```

For torchvision, select a version built for the actual NVIDIA Jetson PyTorch installation. Generic desktop builds may not work. Re-run the import check. Then run:

```bash
python3 scripts/compare_visual_photos.py --selection results/orin-sweep-visit/selection.json --output results/visual-comparison --device cuda
```

If you gave the sweep output a different name, change the `--selection` path. `results/visual-comparison` must be a new directory. The script reads the selected width from `selection.json`; if no width was selected, inspect the sweep logs first. It writes eight `*_comparison.png` figures, the underlying baseline and student depth arrays, and a manifest with model hashes and the selection record. Copy the entire `results/visual-comparison` folder back along with the sweep results.

Each figure shows original RGB, baseline depth and selected-student depth. Within **each figure**, both predictions use the same color scale, so matching colors mean matching predicted depth values for that photograph. Scales can differ between photographs; read their color bars. The original RGB panel shows the full photo, but the network uses a center crop and resize, so the depth image covers only the cropped region. Depth units are unverified. A visually pleasing map alone does not prove accuracy; use the ground-truth metrics from the full evaluation too. Prior desktop FP16 TensorRT showed output outliers, so do not imply that these PyTorch maps are identical to board TensorRT outputs.

To test your own 640×480 image instead, the older `scripts/demo_simple.py` interface remains available; supply `--checkpoint` for the chosen student. Its individual previews use automatic per-image scaling, so they are less suitable for direct visual comparison than the paired figures above.

## What is in this folder?

| Path | Purpose |
| --- | --- |
| `00_START_HERE.md` | This guide. |
| `SHA256SUMS` | Checksums for the transferred files. |
| `scripts/run_latency_sweep_jetson.sh` | Main 12-configuration board runner. |
| `scripts/build_jetson_engine.sh`, `scripts/build_jetson_qdq.sh` | Build floating-point or explicit INT8 TensorRT engines on the board. |
| `scripts/evaluate_jetson_bundle.py` | Measure accuracy, PyTorch-reference differences and timed inference for one engine. |
| `scripts/select_latency_candidate.py` | Select an eligible student from FP16 tuning accuracy and repeated board latency. |
| `scripts/compare_visual_photos.py` | Creates matched-scale baseline/student figures for all eight included photos using the selected student checkpoint. |
| `scripts/demo_simple.py`, `scripts/infer_image.py` | Optional PyTorch single-image demonstration. |
| `visual_photos/` | Eight original RGB photographs and their provenance manifest; they come from the included evaluation collection. |
| `exports/` | Baseline and student ONNX graphs; four normal graphs and four explicitly quantized graphs. FP32 and FP16 engines are built from the same normal graph. |
| `data/nyu_depth_v2/jetson-tuning-445/` | Prepared tuning images, depth targets and manifest. |
| `data/nyu_depth_v2/jetson-evaluation-654/` | Prepared evaluation images, targets and manifest. |
| `data/nyu_depth_v2/latency-*-references/` | Desktop PyTorch predictions used to inspect runtime differences. |
| `data/nyu_depth_v2/int8-calibration-2026-11-16/manifest.json` | Records the training-only calibration sample selection; calibration has already been done. |
| `checkpoints/` and `results/training/` | PyTorch baseline and student checkpoint files for the optional demo. |
| `configs/` | Experiment settings and declared selection gates. |
| `src/`, `third_party/fast-depth/` | Project code and licensed upstream FastDepth implementation. |
| `docs/` | Additional notes; their historical commands may refer to older workflows. Use this guide for the current visit. |

Only the prepared handoff is included, not the full training dataset or desktop Python environment. A working JetPack/TensorRT/CUDA and compatible Python dependencies must already be installed or be installed on the board as described above.
