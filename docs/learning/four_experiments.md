# Four understandable FastDepth experiments

These are **four experimental questions**, not four claimed inventions or four proven improvements. They reuse the same FastDepth checkpoint and a small number of controlled changes, so you can explain both the mechanism and the evidence. Experiments 1 and 2 use the same underlying physical channel surgery; experiments 3 and 4 change recovery training. The recovered starting checkpoint is already a community-mirrored pruned model, not an authenticated original unpruned model. See [pruning](pruning.md), [training](training.md), [metrics](metrics.md), and [oral defense](oral_defense.md) for deeper explanations.

## Common setup and how to read the numbers

- **Task:** input an RGB image; output an estimated depth for every pixel. Our experiments use 224×224 input and produce a 224×224 depth map.
- **Starting network:** 1,360,786 parameters; first decoder's internal width is 200 channels. A channel is one feature map, a 2D array learned to represent some aspect of an image.
- **Data:** 128 training images from four scenes, 64 tuning images from two different scenes. Nearby frames may be similar; evaluation scene identities and near-duplicates remain uncertain. The 654-image evaluation set was not used to score or compare these candidates.
- **Training:** a single epoch, batch size 1, Adam, learning rate 0.00001, seed 42 and mean absolute error on finite positive targets. The fixed BN condition uses stored running statistics and still updates BN scale and bias. The normal BN condition uses the current batch's statistics and updates the running statistics.
- **Metric in the tables:** mean of per-image RMSE on the 64 tuning images, native depth units; lower is better. For image i, `RMSE_i = sqrt(mean_valid_pixels((prediction - target)^2))`; mean-image RMSE is the arithmetic mean of those 64 image-level values. This differs from pooled RMSE, which takes one square root over all valid pixels from all images. Neither should be compared directly to an RMSE from a different dataset or aggregation.
- **Caveat:** choices follow inspection of earlier tuning results, and the same small tuning set has now been examined several times. One seed and epoch cannot establish generalization. These are exploratory measurements, not a final test or Orin performance.

## Experiment 1: How many channels can we remove?

**Question:** Does a narrower decoder give a useful size–accuracy trade-off? We physically remove 8, 16, or 24 of the 200 internal channels, leaving widths 192, 184, and 176. Removing one feature map requires slicing its producer's pointwise filters and BN values, then the following depthwise filters and BN values, and finally the next pointwise input weights. Its depthwise group count changes to match the new width. We keep the following stage's 256 outputs, preserving the skip-add shape.

For this sweep the same ablation ranking was used for each width and each candidate was made independently from the starting model. This is a gradual **width sweep**, not iterative rescoring after each removal. All three used the same frozen-BN training settings:

| Model | Parameters | Before recovery | After recovery |
| --- | ---: | ---: | ---: |
| Starting model | 1,360,786 | 0.428998 | 0.454392 |
| 192 channels | 1,354,410 | 0.573695 | 0.475736 |
| 184 channels | 1,348,034 | 0.711172 | 0.528955 |
| 176 channels | 1,341,658 | 0.909395 | 0.587736 |

Each pruned candidate recovers some accuracy, but every result is worse than the unmodified model on this tuning set. Removing more channels also worsened these measured tuning results. That is not a universal law; it is what happened here. Two desktop PyTorch FP32 timing runs did not show a convincing speedup; no Orin timing exists. See `results/training/2026-11-14-ablation-comparison.json` and `results/benchmarks/2026-11-14-ablation-comparison.json`.

**Why prune at all?** A smaller model can require fewer computations or less memory, but hardware and tensor dimensions determine actual latency. **Why change the depthwise group count?** A depthwise convolution has one independent filter group per channel; after removing channels its group count must match the remaining channels.

## Experiment 2: Which channels should we remove?

**Question:** Given the same width, is it better to remove channels based on weight magnitude or on their effect on depth loss? At width 192 we tried two methods:

1. **Weight magnitude:** for each input channel of the following pointwise convolution, sum the absolute values of its weights. Remove eight channels with the smallest sum. It is fast but ignores channel activations and task loss.
2. **Loss ablation:** for each channel, temporarily suppress its contribution and measure the change in masked training MAE on 32 images from four training scenes. Remove the eight channels whose individual removal increased loss least. This uses data but costs 200 repeated scans and ignores interactions between multiple removed channels.

Both candidates have 1,354,410 parameters and use the same source checkpoint, data, order, optimizer and frozen-BN recovery settings. The weight method removed channels `[2, 4, 28, 34, 125, 127, 181, 195]`; ablation removed `[7, 22, 39, 80, 96, 152, 188, 197]`.

| Selection | Tuning RMSE before recovery | After recovery |
| --- | ---: | ---: |
| Starting model | 0.428998 | 0.454392 |
| Weight magnitude | 0.440975 | 0.448607 |
| Loss ablation | 0.573695 | 0.475736 |

On this small, repeatedly inspected tuning set, the simple weight rule scored better than the ablation rule before and after recovery. The weight candidate after training scored better than the **trained control** (0.448607 versus 0.454392), but worse than the **original model before any extra training** (0.428998). The small difference from the trained control is not an established general improvement. Neither rule is proven best on other scenes, and neither has a measured Orin latency. The follow-up protocol was recorded in `configs/experiment_2026-11-14_simple_controls.json` after previous tuning results were known and before the two new runs; results are in `results/training/2026-11-14-simple-controls.json`.

**Why might the ablation rule do worse?** It measured each channel alone on a small training subset; removing eight simultaneously can create interactions and the training scenes differ from the tuning scenes. These are possible explanations, not confirmed diagnoses. **Why compare at equal width?** It fixes parameter count and the structural change so the principal intended difference is channel selection; the experiment does not remove every possible source of variation.

## Experiment 3: Does recovery fine-tuning help?

**Question:** Can the remaining weights adapt after pruning? For the 192-channel ablation model, we train for one epoch on labeled training images. For every pixel with positive finite target depth, compute absolute error `|prediction - target|`; average it, backpropagate and let Adam update the surviving weights. Using the same input and target pairs matters because the objective is to recover depth prediction, not just imitate an arbitrary output.

Its mean-image tuning RMSE improved from **0.573695 to 0.475736**. However, the unmodified original was **0.428998** before training, and the original model's matched training control worsened to **0.454392**. Thus recovery helped this damaged model relative to itself; it did not restore or improve the original score. The 192-channel weight candidate instead changed from **0.440975 to 0.448607**, so training was not uniformly beneficial. More scenes, seeds and schedules are needed to understand when it helps.

**Why include the trained control?** To avoid attributing the effect of extra training to pruning. **Why is MAE used for training but RMSE reported?** MAE is a simple objective with a constant-magnitude gradient away from zero; RMSE places more emphasis on large errors and is a complementary evaluation measure. Neither choice is automatically optimal.

## Experiment 4: Which batch-normalization mode during recovery?

**Question:** How does training with a batch of one behave when BN uses current-batch statistics versus stored statistics? In normal training mode BN uses the batch statistics in the forward pass and updates running statistics; in evaluation mode it uses the previously stored statistics and does not update those buffers. In both cases the BN scale and offset remain trainable. This is a change to the recovery procedure and resulting weights, not a new pruning architecture.

For the same 192-channel ablation checkpoint, data, optimizer, one-epoch budget and seed, the initial tuning RMSE was **0.573695** in both runs. It ended at **0.475736** with BN in evaluation mode versus **3.526827** in normal BN training mode. The latter is a large degradation on these tuning images. One hypothesis is that batch-one and a small, scene-skewed training distribution made statistics unstable; our experiment changes both the statistics used during training and buffer updates simultaneously, so it does not establish which mechanism caused the difference. Both are one-run observations on a repeatedly inspected tuning set.

**Does frozen BN mean the whole layer is frozen?** No. Here only the use and update of batch statistics change; its trainable scale and offset still receive gradient updates. **Does this prove frozen BN is always better?** No. Different data, batch sizes, schedules or models can change the outcome.

## Separate deployment experiment: FP32 and FP16

FP16 is a numerical precision setting, not a different channel-pruning architecture. A desktop TensorRT engine with FP16 tactics enabled produced pooled RMSE 0.721116 on 654 local evaluation images, compared with 0.721159 for desktop PyTorch FP32; other TensorRT operations may remain FP32. The runtimes differ, so this is a useful cross-runtime check rather than a clean isolation of precision. We also measured one desktop FP16-enabled TensorRT model-only run, median 0.392 ms. We cannot infer the precision effect on Orin speed from this. See [quantization](quantization.md) and [deployment](deployment.md).

## How to present this honestly

Explain the baseline first, then each question, its one intended change, the comparison, the result and its limitation. These are controlled experiments and engineering changes; channel pruning, magnitude ranking, ablation and BN modes are established concepts, so do not claim their invention. A finding can be valuable even if the candidate fails: our tests show recovery, a weight-versus-ablation contrast, and sensitivity to BN mode on this small dataset. A claim of a better edge model still requires broader held-out evaluation and a measured Orin accuracy–latency comparison.

## Comparison on the newer 445-image tuning set — 14 November

The tables above describe the **earlier** 64-image tuning set. We subsequently evaluated ten existing checkpoints on the new, separately grouped tuning subset: 445 images from 26 named scenes (24 inferred location groups). The protocol was fixed in `configs/experiment_2026-11-14_new_tuning_comparison.json` before this run. No model was trained on the new 1,857-image training split in this comparison. Each image was loaded once, preprocessed once, and fed to every checkpoint in evaluation mode on the desktop GPU. Results and per-image records are in `results/evaluation/2026-11-14-new-tuning-comparison/`.

| Existing model | Prior recovery training | Parameters | Mean image RMSE | Pooled δ1 |
| --- | --- | ---: | ---: | ---: |
| Original community model | None | 1,360,786 | 0.300091 | 0.960962 |
| Original control | Prior six-scene split; frozen BN, one epoch | 1,360,786 | 0.327769 | 0.952772 |
| Weight-selected 192 | None | 1,354,410 | 0.306396 | 0.960224 |
| Weight-selected 192 | Prior six-scene split; frozen BN, one epoch | 1,354,410 | 0.328357 | 0.952164 |
| Ablation-selected 192 | None | 1,354,410 | 0.367881 | 0.949689 |
| Ablation-selected 192 | Prior six-scene split; frozen BN, one epoch | 1,354,410 | 0.330332 | 0.952660 |
| Ablation-selected 184 | Prior six-scene split; frozen BN, one epoch | 1,348,034 | 0.364513 | 0.944021 |
| Ablation-selected 176 | Prior six-scene split; frozen BN, one epoch | 1,341,658 | 0.366903 | 0.944029 |
| Ablation-selected 192 | Prior six-scene split; **normal** BN, one epoch | 1,354,410 | 1.105387 | 0.766680 |
| Weight-selected 160 | Earlier two-scene split; frozen BN, one epoch | 1,328,906 | 0.332615 | 0.946916 |

**Interpretation.** The original model had the lowest mean per-image RMSE. Weight-selected 192 before recovery was closest, 0.006306 native depth units higher; after recovery on the old data it was 0.328357. The frozen-BN ablation-selected 192 model improved relative to its own untrained 0.367881, but was worse than the original. The normal-BN checkpoint degraded markedly in this comparison. The 160-channel candidate had a different prior training set and therefore is not a matched recovery comparison with 192/184/176.

**Why do pooled and mean-image RMSE differ?** Each image's RMSE is the square root of its mean squared valid-pixel error. Mean-image RMSE averages those square roots equally; pooled RMSE averages all squared valid-pixel errors first and takes one square root. The original model's pooled RMSE is 0.363704, not 0.300091. The report also averages scene mean-image RMSE equally across named scenes; it is not the same as image-weighting when scene sample counts differ. `results/evaluation/2026-11-14-new-tuning-aggregation-check.json` reconstructs the image, pooled and scene results from the 4,450 per-image records with maximum difference below 3e-16.

**Limits.** Frames within a scene are correlated and the split was chosen from bounded archive windows. Scene identities of the 654 final evaluation files are unknown. Moreover, this tuning set comes from the mirror's training shards, and the community model's original training data is not authenticated: the model may already have seen some of these images or scenes during pretraining. Excluding the small sets used in our subsequent experiments does not establish independence from that original training. This tuning set has now been inspected; selecting candidates based on it makes it unsuitable as an untouched final test. No measured result here establishes Orin accuracy or latency.

The comparison command is `bash scripts/python.sh scripts/compare_tuning_models.py --config configs/experiment_2026-11-14_new_tuning_comparison.json --output <new-directory>`. It verifies the overlap report and manifest hashes, loads each saved model without training, computes per-image depth metrics and records per-scene and pooled summaries. The per-image CSV retains all 445 × 10 records; the matched configuration and checkpoint hashes identify what was compared. It uses the same metric mask and preprocessing as the earlier pipeline.

Two additional desktop timing passes compared the original and recovered variants using PyTorch FP32, one zero input, 50 warm-ups and 200 synchronized calls per run. Medians in milliseconds, first/second pass: original 5.352/5.521; weight-192 5.578/5.347; ablation-192 5.352/5.498; ablation-184 5.500/5.431; ablation-176 5.325/5.580; weight-160 5.567/5.551. Run order was forward and reverse, power/thermal conditions were uncontrolled, and a CPU-only test suite ran during the first part of timing. Differences change direction and cannot establish reliable speedup. This is not TensorRT, Orin or end-to-end timing. See `configs/experiment_2026-11-14_new_model_timing.json` and `results/benchmarks/2026-11-14-new-model-comparison.json`.
