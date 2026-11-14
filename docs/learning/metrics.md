# Depth metrics and evaluation

## Valid pixels and aggregation

Ground-truth pixels are valid when the target is finite and greater than zero. Invalid targets are excluded. At valid targets, a nonfinite prediction causes an error; a nonpositive prediction is counted, remains in error metrics, and is clamped only for the ratio calculation using epsilon `10^-6`. This avoids silently improving a score by dropping bad predictions. Results currently aggregate all valid pixels across all images; this differs from averaging a metric separately for each image.

## Definitions

For prediction `p`, ground truth `d`, and N valid pixels: RMSE is `sqrt(sum((p-d)^2)/N)` and MAE is `sum(abs(p-d))/N`. Both have the same units as the depth arrays. RMSE penalizes larger errors more heavily. REL is `sum(abs(p-d)/d)/N`, a dimensionless error relative to target depth. Delta1 is the fraction for which `max(p/d,d/p) < 1.25`; larger is better. R² is `1 - sum((p-d)^2)/sum((d-mean(d))^2)`: 1 means exact predictions, 0 matches the constant target mean under this pooled definition, and negative values are possible. If all targets are constant, the denominator is zero and R² is reported as null.

The implementation is `src/fastdepth_project/metrics.py`; tests cover hand-calculated values, invalid targets/predictions, empty masks, constant targets and mismatched shapes. The evaluation script saves both pooled metrics and the mean of image-level RMSE/delta1; these are different quantities and must be labeled. The authors' saved checkpoint metrics and their TX2 results are historical references, not measurements made by this project. Depth units and preprocessing parity still require verification before claiming exact reproduction.

## First desktop baseline — 13 November

The community checkpoint was evaluated over all 654 locally extracted files with the implemented Pillow preprocessing. There were 32,815,104 valid transformed pixels and 8,884 nonpositive predictions. Pooled RMSE was 0.721159, MAE 0.428963, REL 0.165163, delta1 0.771065 and R² 0.719858. Mean per-image RMSE was 0.603737, demonstrating that averaging each image's RMSE is not the same operation as taking the square root after pooling squared errors. The numerical closeness of mean image RMSE and delta1 to the checkpoint's historical saved metrics is encouraging, but split identity, source authenticity, exact resize parity and units remain unverified. See `results/evaluation/2026-11-13-community-baseline/` for manifest, per-image rows and configuration.
