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
| Scale factor | 0.0143 | 0.8223 | 0.3187 |
| Orig vs GT (m) | 0.0620 | 0.0513 | 0.0677 |
| Edit vs GT (m) | 0.0601 | 0.0704 | 0.1467 |
| MAE (m) | 0.0277 | 0.0418 | 0.1426 |
| RMSE (m) | 0.0643 | 0.0936 | 0.1716 |
| Max diff (m) | 1.9410 | 1.7650 | 1.5474 |
| Std dev (m) | 0.0614 | 0.0856 | 0.0973 |
| % diff > 0.1m | 3.02 | 12.52 | 59.57 |
| % diff > 0.5m | 0.27 | 0.29 | 0.30 |

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
