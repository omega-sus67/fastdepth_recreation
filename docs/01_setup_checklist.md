# Phase 0 — Setup checklist

Reference: [project plan](fastdepth_edge_ai_plan.md), sections 1–3. Record links, versions, paths, and decisions alongside the relevant task or in experiment notes.

## Project approval and reading approach

- [x] Obtain professor approval to switch from chess to FastDepth.
- [ ] Start setup immediately. Read the short task-specific references in later checklists only when you reach those tasks; the plan's mathematics is a reference, not a prerequisite to starting.

## Step 0.1 — Obtain compute

- [ ] Create or sign in to Kaggle and enable GPU access, completing any required verification.
- [ ] Start a GPU notebook and run `nvidia-smi`; record GPU type, software versions, and available quota.
- [ ] Set up Colab as a fallback and confirm GPU access if available.
- [ ] Decide where code, checkpoints, datasets, and results will be stored persistently; verify that outputs survive notebook termination.
- [ ] Record session limits and a restart procedure for long-running work.

**Done when:** a Kaggle notebook reports a T4 or P100 GPU, quota is visible, and outputs can be saved persistently.

## Step 0.2 — Obtain and inspect NYU Depth v2

**Read when inspecting the split:** [FastDepth](../papers/fastdepth.pdf), PDF page 4, section IV-A (experimental setup). The paper describes its NYU Depth v2 evaluation; compare it with the actual repository loader before assuming another dataset copy uses the same split and preprocessing.

- [ ] Locate the dataset in notebook-accessible storage without downloading the full dataset to the laptop.
- [ ] Inspect the Hugging Face copy and the FastDepth data loader: compare split definitions, HDF5 keys, image/depth shapes, data types, units, and invalid-depth conventions.
- [ ] Identify the standard evaluation split and verify whether the chosen validation split contains the standard 654 images; record any mismatch.
- [ ] Select a training subset without overlap with validation and record its identifiers and selection method.
- [ ] Write or adapt a loader as needed; display one RGB image beside its depth map and inspect depth values and invalid pixels.
- [ ] Record dataset source, split sizes, preprocessing assumptions, and access instructions.

**Done when:** the image and depth map display correctly, depth units are understood, and validation count and split compatibility are known.
