# Phase 5 — Final validation and report checklist

Reference: [project plan](fastdepth_edge_ai_plan.md). Use the predeclared budget and improvement target when interpreting results.

## Step 5.1 — Final board runs

**Read when preparing tables:** [FastDepth](../papers/fastdepth.pdf), PDF pages 4–6, especially Tables I–II and VII, for examples of reporting accuracy, latency, and power. Your repeated-run variability and energy calculations are additional project measurements.

- [ ] Select the final baseline, optimized reference, and improved models; record exact checkpoint and engine identifiers.
- [ ] Run each applicable final configuration five times on each board using the same documented protocol.
- [ ] Save raw latency and power samples, environment settings, model outputs, and run identifiers.
- [ ] Calculate median and tail latency, variability or confidence intervals across repeated runs, total and idle-subtracted energy per frame, and accuracy from board outputs.
- [ ] Check for thermal throttling, unsupported configurations, missing data, and differences between boards; document limitations.
- [ ] Produce final comparison tables with units, timing scope, and sample counts.

**Done when:** five runs per applicable final configuration support the reported latency, energy, and accuracy results, with any gaps explained.

## Step 5.2 — Demo (optional)

- [ ] Connect and test a camera on the Nano; match its preprocessing to model input requirements.
- [ ] Display live depth with an accurately defined frame-rate counter and identify the model and runtime used.
- [ ] Check sustained operation and record a backup video.

**Done when:** the live demo and backup video are usable, if this optional step is pursued.

## Step 5.3 — Report and slides

**Read for related work only when writing:** [FastDepth](../papers/fastdepth.pdf), PDF pages 1 and 3–6; [NetAdapt](../papers/netadapt.pdf), PDF pages 4–8; and, only if used, [Depth Anything V2](../papers/depth_anything_v2.pdf), PDF pages 1, 6 and 9. Cite the papers for their own claims and distinguish them from measurements made on your boards.

- [ ] Check the grading sheet and include every required PS13 metric, including R².
- [ ] Describe the problem, contribution, related work, dataset and splits, implementation, measurement methodology, and ablations.
- [ ] Present the baseline, optimization ladder, and project improvement with supporting tables or plots.
- [ ] Report the 33 ms budget, the predeclared accuracy-loss limit, actual results, uncertainty, and any failed or unsupported configurations.
- [ ] Explain preprocessing, valid-pixel treatment, device/runtime versions, power measurement boundaries, and reproducibility instructions.
- [ ] Cite papers, code, datasets, and model licences as appropriate.
- [ ] Prepare slides and check that every displayed number matches the saved experimental results.

**Done when:** the report and slides cover the grading criteria and all claims are traceable to saved evidence.
