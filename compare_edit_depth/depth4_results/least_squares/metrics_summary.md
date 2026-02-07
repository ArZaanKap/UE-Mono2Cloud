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
| Scale factor | 0.6001 | 0.7747 | 0.3509 |
| Orig vs GT (m) | 0.0552 | 0.0434 | 0.0711 |
| Edit vs GT (m) | 0.0567 | 0.0866 | 0.2155 |
| Edit vs GT RMSE (m) | 0.0940 | 0.1069 | - |
| MAE (m) | 0.0207 | 0.0801 | 0.2157 |
| RMSE (m) | 0.0392 | 0.0938 | 0.2481 |
| Max diff (m) | 1.7433 | 2.2994 | 2.0277 |
| Std dev (m) | 0.0363 | 0.0502 | 0.1235 |
| % diff > 0.1m | 1.16 | 30.74 | 85.30 |
| % diff > 0.5m | 0.07 | 0.13 | 0.27 |

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
