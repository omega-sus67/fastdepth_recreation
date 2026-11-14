# Project working rules

## Scope and priorities

- The deployment target is **Jetson Orin Nano**. Do not introduce original Jetson Nano or other-board deliverables without a user request.
- The current development machine has an RTX 3060 Laptop GPU with 6 GB VRAM. The Orin Nano is unavailable; finish independent desktop implementation and experiments while hardware tasks remain pending.
- Read `docs/immediate_priority.md` before starting implementation and follow the first actionable task. Use `docs/fastdepth_edge_ai_plan.md` as the single authoritative implementation and submission plan.
- Prioritize runnable software and reproducible experiments. Build detailed teaching explanations alongside implementation; the user does not need to finish reading before work can proceed.

## Explain every implementation and decision

- For every substantive code change, experiment, configuration choice or methodological decision, update the relevant separate `docs/learning/` topic file in the same work session. Explain the purpose, underlying concept, inputs/outputs and tensor shapes or equations where relevant, exact implementation and commands, alternatives considered, reasons and trade-offs, tests/results, failure cases and remaining unknowns.
- Write for oral examination as well as reproduction: add likely professor questions and defensible answers to the relevant topic note or `docs/learning/oral_defense.md`. Explain not just what was done but why, what an alternative would change, and what the evidence can and cannot establish. Break concepts down carefully enough that the user can explain them independently.
- Keep examples, data units, masks, aggregation, numerical results, hardware and version claims consistent with primary evidence and the progress report. Clearly mark proposed approaches and inferred explanations; do not pretend every reasonable choice was experimentally proven optimal.
- Use `docs/learning/jetson_software.md` and the user-provided photographs under `papers/` as reported target-board context. They are screenshots of a summary, not direct board telemetry; keep version discrepancies and unseen components explicitly unverified until raw board checks are available. Do not install desktop wheels on Jetson or claim an engine/benchmark works on the board based on desktop evidence.
- Preserve the user's choice to defer poster and final report production; detailed learning documents and progress/evidence records should continue to grow while implementation proceeds.

## Maintaining immediate_priority.md

- The only active priority file is `docs/immediate_priority.md`; do not recreate a root-level copy. Maintain links relative to its `docs/` location.
- Treat this file as the current work queue, not an append-only journal or a duplicate of the full plan.
- Rewrite it whenever a listed step is completed, a dependency or blocker changes, new evidence changes the order, or the user changes scope. Update it before handing back a turn that made relevant progress.
- Keep a last-updated date, a concise verified state, an ordered list of at most five immediate tasks, concrete acceptance criteria, current blockers with the next action, and the hardware-dependent work being deferred.
- Remove completed work from the active queue, record its evidence in the appropriate plan status/result/document, and promote the next useful task. Retain a short verified-state summary rather than a growing completion log.
- Recheck claims against actual files and results. Distinguish a planned command from a command that ran successfully. A random-weight forward pass does not establish pretrained inference or accuracy.
- If a task is blocked, state the specific dependency and continue independent authorized work; do not let unavailable weights or hardware stop unrelated implementation.
- Synchronize affected README text, plan status and assumptions when scope or verified capabilities change. Do not mark pending fresh-install or board checks complete from desktop checks.

## Maintaining the progress report and evidence

- Maintain `docs/progress_report.md` as the persistent history of wins, failed attempts, negative results, fixes, decisions and unresolved limitations. It supports final report writing; it is separate from the rewritten immediate-priority queue and the detailed plan.
- Read its current outcome summary before substantive experiments. Before handing back work that changes capabilities, experimental evidence, blockers or interpretation, append/update a dated entry with a stable progress ID and synchronize the current summary. Preserve the original failed attempt when recording a later resolution.
- Record the objective/hypothesis, actual command or run ID, model/checkpoint/configuration/manifest identifiers, device/runtime/precision, seed where applicable, observations, measured values with units/sample counts, diagnosis, response, validation, evidence paths, report relevance and remaining limits. Separate a confirmed cause from a hypothesis.
- Record event date separately from date recorded when compiling history later. Capture future failures' stdout/stderr and exit status when practical; label reconstructed session history and missing logs explicitly. Never fabricate evidence or claim a new run solely because documentation was updated.
- Distinguish tooling errors from scientific negative results, random/synthetic checks from pretrained accuracy, inherited checkpoint metrics from remeasured results, and desktop results from Orin measurements. State what each result permits us to claim and which report/contribution sections still need evidence.
- Preserve important evidence in stable run-specific paths before a checker overwrites a latest-result file. Do not overwrite `results/history/2026-11-12-phase-a/` or other historical baselines with new runs; add a new dated/run-specific snapshot and mark superseding findings in the history.
- Maintain `results/evidence_index.csv` with progress IDs, claim scope, artifact paths, SHA-256 fingerprints, report placement and limitations. Keep fingerprints consistent with the referenced files; the progress report is a narrative source, not a substitute for raw measurements.
- Routine read-only work without a new finding does not need a history entry. Documentation-only progress must not be counted as a model experiment or scientific contribution.

## Evaluator requirements and the consolidated plan

- Maintain `docs/fastdepth_edge_ai_plan.md` as the single detailed implementation, experiment, report and poster plan; do not create competing phase backlogs.
- Treat `docs/Machine_Learning_Project_Report_Guidelines_Edited.pdf` as the supplied report requirements and `docs/updated poster sample.pptx` as the poster layout/content reference. Preserve both source files.
- Keep a requirements-to-evidence mapping in the plan. Cover all prescribed preliminary pages, five numbered chapters and subsections, references, optional appendix, and properly numbered/referenced figures, tables, equations and algorithms.
- The final report is one PDF of 12–15 total pages including all sections. The detailed plan has no such page limit. Target 3–5 distinct evidence-backed project contributions without mislabeling reproduction or engineering work as research novelty.
- Produce dataset/preprocessing/EDA evidence, existing-model comparisons, graphical analysis, findings and limitations as actual work. A requirement listed in the plan is not a completed requirement.
- Use the poster's landscape three-column organization, title/identity, contributions, architecture, algorithm, results, conclusion and references. Replace its photonic examples with this project's own measured results, and keep report/poster values consistent through shared evidence.

## Implementation and environment

- Use the independent `.venv-desktop` through `bash scripts/python.sh ...` for project commands; the wrapper excludes inherited ROS/user Python package paths. Reproduce the verified package set with `bash scripts/setup_desktop.sh --locked --with-tensorrt` and `requirements-desktop-lock.txt`.
- Preserve the existing working environment and the shared packages in `/home/omega_sus/droidrun/venv`. Provide and verify an independent installation path without changing that shared environment as a side effect.
- Use project-relative configurable paths; record actual Python, framework, CUDA, runtime, and GPU versions. Do not call the historical requirements snapshot a lock file for the active shared environment.
- Treat `third_party/fast-depth/` as vendored upstream source with recorded revision and license. Prefer project adapters in `src/`; document any necessary compatibility patch. Do not require a nested `.git` directory for setup to recognize the source.
- Preserve the architecture and channel configuration when converting original checkpoints. Record source, license where available, checksum, variant, and conversion command. Do not substitute random or independently trained weights for original pretrained results.
- Keep inference, evaluation, export, and training preprocessing consistent. Validate meaningful boundaries such as metrics, tensor shapes, checkpoint conversion, training resume, and cross-runtime output parity.
- Run checks appropriate to each change. Use short training smoke runs before long experiments; preserve seeds, configuration, manifests, checkpoints, and resume state.

## Experiments and board validation

- Keep training, tuning/candidate-selection, and final evaluation samples separate. Record manifests and overlap checks; do not train or repeatedly select pruning/quantization settings using the final evaluation split.
- Record depth units, valid-pixel rules, aggregation, and treatment of invalid predictions for RMSE, MAE, REL, delta1, and R². Investigate upstream preprocessing and metric quirks instead of silently copying them.
- Declare the batch-1 latency budget (at most 33 ms), timing scope, and improvement accuracy limit before target-device measurement. Report honestly if the baseline already meets the budget.
- Label desktop, reference-paper, and Orin measurements by their actual source. Preserve raw timing samples, warm-up settings, synchronization, versions, device conditions, and accuracy results.
- Select the Orin software target from official documentation and confirm installed versions on the actual board. Do not assume original-Nano JetPack 4.6/TensorRT 8.2 constraints or reject INT8 categorically.
- Package ONNX and reproducible build commands; build final TensorRT engines on the Orin Nano. Desktop engine execution does not establish Jetson portability.
- Build pruning and fine-tuning tools on the desktop, but require actual Orin measurements for final hardware-aware candidate selection, latency, power, memory, thermal, and energy claims.

## Repository contents

- Keep planning documents, papers, source, checkpoints, exports, and experiment results in scope for version control. The user requested that datasets remain local and not be pushed to GitHub.
- Use `.gitignore` for local datasets, environments, installed dependency directories, caches, build products, and local editor/OS files. Preserve papers, checkpoints, results, and vendored source.
- Use Git LFS for files exceeding the host's ordinary Git size limits and preserve existing LFS tracking. Never delete project artifacts merely to reduce repository size.
- Preserve user changes and upstream licenses. Make completion claims only for work actually performed and verified.
