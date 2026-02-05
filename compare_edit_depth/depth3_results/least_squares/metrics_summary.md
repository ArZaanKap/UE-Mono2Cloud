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

| Metric | depth_anything | depth_pro | zoedepth | metric3d |
|--------|--------|--------|--------|--------|
| Scale factor | 0.0143 | 0.8223 | 0.8650 | 0.3187 |
| Orig vs GT (m) | 0.0706 | 0.0532 | 0.1265 | 0.0722 |
| Edit vs GT (m) | 0.0698 | 0.0828 | 0.2092 | 0.1630 |
| MAE (m) | 0.0278 | 0.0521 | 0.1826 | 0.1635 |
| RMSE (m) | 0.0596 | 0.0987 | 0.2033 | 0.1912 |
| Max diff (m) | 1.9001 | 1.7405 | 1.5119 | 1.6104 |
| Std dev (m) | 0.0561 | 0.0856 | 0.0918 | 0.1012 |
| % diff > 0.1m | 3.62 | 17.01 | 78.20 | 70.50 |
| % diff > 0.5m | 0.19 | 0.19 | 1.06 | 0.28 |

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
*Last updated: depth_anything*
