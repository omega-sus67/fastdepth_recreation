# FastDepth project documentation

Start with [the project story](project_story.md) for a chronological explanation of our decisions and what we learned, or [immediate priorities](immediate_priority.md) for the current next actions. Follow [agents.md](../agents.md) when working on the project and updating that queue.

The [experiment index](experiment_index.md) links each main question to its settings, model, measurements and limitations. Use the [dataset guide](../data/README.md), [script guide](../scripts/README.md) and [results guide](../results/README.md) to navigate the implementation and evidence.

The [progress report](progress_report.md) preserves wins, failures, fixes, decisions and remaining limitations for final report writing. Its [evidence index](../results/evidence_index.csv) links claims to fingerprinted artifacts, including a preserved Phase A baseline. Update the history as meaningful work progresses; it is distinct from the short next-step queue.

The [complete project plan](fastdepth_edge_ai_plan.md) is the **single authoritative detailed plan** for implementation, experiments, report and poster. It maps every supplied evaluator requirement to work and evidence, separates desktop work from Orin-dependent validation, and includes the final completion checklist. The deployment target is Jetson Orin Nano; development currently uses the RTX 3060 Laptop GPU.

[Phase A results](phase_a_results.md) record the verified independent installation, runtime checks, checkpoint recovery/conversion, cache cleanup and remaining original-weight limitations. This is an evidence record, not a second implementation plan.

## Supplied evaluation references

- [Report guidelines](Machine_Learning_Project_Report_Guidelines_Edited.pdf): prescribed sections, contributions, numbering, and one 12–15 page PDF including all sections.
- [Poster sample](updated%20poster%20sample.pptx): landscape three-column layout and expected content organization. Its photonic-sensor results and model choices are example content, not this project's results.

## Navigate the consolidated plan

- [Requirements-to-evidence coverage](fastdepth_edge_ai_plan.md#3-requirements-to-evidence-coverage)
- [Contributions and evidence](fastdepth_edge_ai_plan.md#4-planned-contributions-and-claim-discipline)
- [Setup and checkpoints](fastdepth_edge_ai_plan.md#6-phase-a--reproducible-desktop-setup-and-checkpoint-acquisition)
- [Dataset, preprocessing and EDA](fastdepth_edge_ai_plan.md#7-phase-b--dataset-verification-preprocessing-and-eda)
- [Metrics, baseline and inference](fastdepth_edge_ai_plan.md#8-phase-c--metrics-baseline-reproduction-cost-and-usable-inference)
- [Exports and benchmarking](fastdepth_edge_ai_plan.md#9-phase-d--exports-runtime-parity-benchmark-harness-and-desktop-evidence)
- [Training, pruning and ablations](fastdepth_edge_ai_plan.md#10-phase-e--training-structured-pruning-and-controlled-ablations)
- [Board handoff and desktop milestone](fastdepth_edge_ai_plan.md#11-phase-f--board-ready-packaging-and-desktop-completion-milestone)
- [Orin bring-up and measurements](fastdepth_edge_ai_plan.md#12-phase-g--orin-nano-bring-up-and-measurement)
- [Final evaluation and demonstration](fastdepth_edge_ai_plan.md#13-phase-h--final-evaluation-analysis-and-demonstration)
- [Report structure and page budget](fastdepth_edge_ai_plan.md#15-final-report-mandatory-structure-and-page-plan)
- [Poster plan](fastdepth_edge_ai_plan.md#17-poster-plan-based-on-the-supplied-sample)
- [Final completion checklist](fastdepth_edge_ai_plan.md#19-final-completion-checklist)

The consolidated plan contains the former phase navigation and completion checklist; maintain task details and status there rather than creating duplicate backlogs. Papers are indexed in [papers](../papers/README.md); reading is reference work, not a prerequisite in the immediate implementation queue.

[Weekly execution schedule](weekly_execution.md) and [separate learning notes](learning/README.md) track the current deadline-focused work. Poster and final report production are deferred at the user’s request.
