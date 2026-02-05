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
| Scale factor | 0.0223 | 0.8048 | 0.6451 | 0.2878 |
| Orig vs GT (m) | 0.3799 | 0.0493 | 0.1460 | 0.1377 |
| Edit vs GT (m) | 0.3424 | 0.0921 | 0.1320 | 0.2724 |
| MAE (m) | 0.0779 | 0.0844 | 0.1141 | 0.1739 |
| RMSE (m) | 0.1195 | 0.1046 | 0.1433 | 0.2028 |
| Max diff (m) | 2.8001 | 2.3918 | 1.1252 | 1.7501 |
| Std dev (m) | 0.0922 | 0.0626 | 0.0969 | 0.1051 |
| % diff > 0.1m | 30.62 | 31.26 | 46.92 | 73.72 |
| % diff > 0.5m | 0.19 | 0.14 | 0.14 | 0.35 |

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
