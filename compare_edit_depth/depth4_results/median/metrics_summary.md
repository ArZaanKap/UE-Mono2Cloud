# Depth Consistency Metrics: Original vs Edited Image

## Methodology

### Scaling Method: MEDIAN
All model predictions are aligned to ground truth using **median scaling** (scale only, no shift),
computed from the **original image only**. The same scale factor is then applied to both original
and edited predictions. This ensures fair comparison across models with different output scales.

### Unchanged Regions
A pixel is considered **unchanged** if the mean absolute RGB difference between the original
and edited image is **<= 10** (on a 0-255 scale). This threshold identifies regions where
the edit did not visually modify the image.

### What We're Measuring
For a perfectly consistent depth model, unchanged regions should produce **identical depth
predictions** regardless of edits elsewhere in the image. Lower values = better consistency.

## Metrics Comparison

| Metric | depth_anything | depth_pro | metric3d |
|--------|--------|--------|--------|
| Scale factor | 0.5451 | 0.8047 | 0.2878 |
| Orig vs GT (m) | 0.0826 | 0.0499 | 0.1281 |
| Edit vs GT (m) | 0.0974 | 0.0959 | 0.2627 |
| Edit vs GT RMSE (m) | 0.1528 | 0.1132 | - |
| MAE (m) | 0.0188 | 0.0832 | 0.1769 |
| RMSE (m) | 0.0356 | 0.0974 | 0.2035 |
| Max diff (m) | 1.5835 | 2.3885 | 1.6631 |
| Std dev (m) | 0.0330 | 0.0521 | 0.1013 |
| % diff > 0.1m | 1.03 | 33.45 | 76.13 |
| % diff > 0.5m | 0.06 | 0.14 | 0.21 |

## Interpretation

- **Scale factor**: Least-squares scale applied to align model output to GT (closer to 1.0 = better native metric accuracy).
- **Orig vs GT (m)**: MAE between scaled original prediction and GT in unchanged regions (baseline accuracy).
- **Edit vs GT (m)**: MAE between scaled edited prediction and GT in unchanged regions. If higher than Orig vs GT, the edit caused accuracy drift.
- **MAE (m)**: Mean Absolute Error between original and edited predictions in unchanged regions. Lower = more consistent.
- **RMSE (m)**: Root Mean Square Error. More sensitive to large errors.
- **Max diff (m)**: Worst-case depth difference in unchanged regions.
- **Std dev (m)**: Standard deviation of depth differences.
- **% diff > 0.1m / 0.5m**: Percentage of unchanged pixels with depth differences exceeding threshold.

---
*Last updated: depth_pro*
