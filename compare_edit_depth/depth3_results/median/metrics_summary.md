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
| Scale factor | 0.0248 | 0.9031 | 0.2914 |
| Orig vs GT (m) | 0.5027 | 0.0671 | 0.0825 |
| Edit vs GT (m) | 0.5123 | 0.0458 | 0.1805 |
| MAE (m) | 0.0482 | 0.0459 | 0.1304 |
| RMSE (m) | 0.1118 | 0.1028 | 0.1569 |
| Max diff (m) | 3.3756 | 1.9385 | 1.4149 |
| Std dev (m) | 0.1067 | 0.0940 | 0.0890 |
| % diff > 0.1m | 10.79 | 13.83 | 52.03 |
| % diff > 0.5m | 0.46 | 0.30 | 0.23 |

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
