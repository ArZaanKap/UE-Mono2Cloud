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

| Metric | depth_anything | depth_pro | zoedepth | metric3d |
|--------|--------|--------|--------|--------|
| Scale factor | 0.0248 | 0.9031 | 0.7749 | 0.2914 |
| Orig vs GT (m) | 0.5630 | 0.0736 | 0.1536 | 0.0952 |
| Edit vs GT (m) | 0.5798 | 0.0470 | 0.2109 | 0.2114 |
| MAE (m) | 0.0483 | 0.0572 | 0.1636 | 0.1495 |
| RMSE (m) | 0.1037 | 0.1084 | 0.1821 | 0.1748 |
| Max diff (m) | 3.3045 | 1.9116 | 1.3544 | 1.4724 |
| Std dev (m) | 0.0975 | 0.0941 | 0.0822 | 0.0926 |
| % diff > 0.1m | 12.14 | 19.10 | 73.58 | 63.65 |
| % diff > 0.5m | 0.31 | 0.20 | 0.43 | 0.22 |

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
*Last updated: metric3d*
