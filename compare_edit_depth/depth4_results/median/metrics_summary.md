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
| Scale factor | 0.0223 | 0.8048 | 0.2878 |
| Orig vs GT (m) | 0.3643 | 0.0497 | 0.1281 |
| Edit vs GT (m) | 0.3312 | 0.0976 | 0.2627 |
| MAE (m) | 0.0804 | 0.0851 | 0.1769 |
| RMSE (m) | 0.1304 | 0.1143 | 0.2035 |
| Max diff (m) | 2.5082 | 2.0659 | 1.6631 |
| Std dev (m) | 0.1041 | 0.0773 | 0.1013 |
| % diff > 0.1m | 32.96 | 33.62 | 76.13 |
| % diff > 0.5m | 0.25 | 0.26 | 0.21 |

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
