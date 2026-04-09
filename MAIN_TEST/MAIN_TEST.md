# Notebook Guide: Image to Point Cloud

The two main notebooks take a UE render + edited image and produce a coloured `.las` point cloud.

| Notebook | Depth model | Output folder |
|---|---|---|
| `MAIN_TEST/img_to_pointcloud_depth_pro.ipynb` | Depth Pro | `MAIN_TEST/pointclouds_depth_pro/` |
| `MAIN_TEST/img_to_pointcloud_depth_pro_legacy.ipynb` | Depth Pro (legacy variant) | `MAIN_TEST/pointclouds_depth_pro_legacy/` |
| `MAIN_TEST/img_to_pointcloud_da3.ipynb` | Depth Anything 3 (DA3 Giant / DA3 Nested) | `MAIN_TEST/pointclouds_da3/` |
| `MAIN_TEST/img_to_pointcloud_marigold.py` | Marigold-DC | `MAIN_TEST/pointclouds_marigold/` |

Both follow the same pipeline. Depth Pro currently gives better results (lower RMSE). DA3 is the newer alternative.

---

## Configuration (top of each notebook)

```python
DATASET          = "test2"       # folder name inside data/
CAMERA_FOV       = 90            # degrees — matches UE scene camera
GT_TO_CENTIMETERS = 10000.0      # for SceneDepth.exr files (do not change)
```

For `MAIN_TEST/img_to_pointcloud_da3.ipynb` also set:
```python
MODEL_VARIANT = "da3_giant"      # or "da3_nested"
```

---

## Pipeline Steps

```
1. Load data          — original EXR, edited PNG, GT SceneDepth EXR
2. Sky detection      — find sky threshold from GT depth histogram
3. Change detection   — GeSCF mask: which pixels changed vs original
4. Depth prediction   — run model on edited image
5. Calibration / completion
   Depth Pro / DA3: least-squares fit on unchanged pixels → scale + shift
   Marigold-DC: sparse-guided depth completion with UE GT on unchanged pixels
6. Point cloud        — back-project, sky filter, colour, export .las
```

---

## Sky Detection — Density Knee Method

UE's sky dome renders at varying depths (no single far-clip value), producing a flat sparse tail in the depth histogram. A simple gap-based threshold fails.

**Method:** Histogram GT depth in 0.25m bins → smooth → find the peak of the dense scene cluster → walk right until bin count drops below 5% of peak. That "knee" is the sky threshold.

Example (test2): peak at 1.4m, knee at 4.88m → scene = 97.3%, sky tail = 2.7%.

---

## Change Detection — GeSCF

1. Hook SAM ViT-B block 8 Q/K/V attention features for original and edited images
2. Cosine distance → distance map → Gaussian smooth → normalise
3. Adaptive threshold: `mean + k·std` where `k = clip(skewness, 1, 3)`
4. SAM segment refinement: include any segment with >30% overlap with initial mask

---

## Calibration

Depth Pro / DA3 output relative depth. Align to GT using unchanged pixels:

```python
depth_calibrated = pred * scale + shift   # least-squares fit
```

Fit is on unchanged, non-sky pixels where GT is reliable. A p98 cap on predictions excludes model sky overestimates from the fit.

---

## Sky Filtering — Changed vs Unchanged

The GT sky mask reflects the **original** scene. Edited content (e.g. a ladder placed in front of sky) would be wrongly removed if the GT mask is applied everywhere.

```
Unchanged pixels → trust GT sky mask
Changed pixels   → use predicted depth: if depth > depth_cap → sky, else keep
```

`depth_cap` = max GT depth in non-sky region (e.g. 4.87m for test2).
After scaling: `depth_cap_scaled = depth_cap * scale + shift`.

```python
sky_unchanged = sky_full & ~changed_full
sky_changed   = changed_full & (depth_full > depth_cap_scaled)
sky_combined  = sky_unchanged | sky_changed
```

---

## Point Cloud Projection

Pinhole camera model:

```python
focal = width / (2 * tan(FOV/2))
x = (px_x - cx) * z / focal
y = (px_y - cy) * z / focal
```

LAS axis convention: X = depth (forward), Y = −x (right), Z = −y (up).
