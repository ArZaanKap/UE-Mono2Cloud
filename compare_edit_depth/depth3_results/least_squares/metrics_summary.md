# Depth Consistency Metrics: Original vs Edited Image

## Methodology

### Scaling Method: LEAST_SQUARES
All model predictions are aligned to ground truth using **least-squares scaling** (scale + shift),
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
| Scale factor | 0.5630 | 0.8221 | 0.3187 |
| Orig vs GT (m) | 0.0622 | 0.0513 | 0.0677 |
| Edit vs GT (m) | 0.0618 | 0.0702 | 0.1467 |
| Edit vs GT RMSE (m) | 0.0884 | 0.1050 | - |
| MAE (m) | 0.0318 | 0.0421 | 0.1426 |
| RMSE (m) | 0.0596 | 0.0901 | 0.1716 |
| Max diff (m) | 2.2350 | 1.8621 | 1.5474 |
| Std dev (m) | 0.0568 | 0.0821 | 0.0973 |
| % diff > 0.1m | 1.70 | 12.89 | 59.57 |
| % diff > 0.5m | 0.17 | 0.24 | 0.30 |

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
