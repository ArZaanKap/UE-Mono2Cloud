# Change Detection Results

This folder contains the script for comparing change detection methods, and the pre-computed mask outputs for each dataset.

---

## What this does

Given an original UE render and a user-edited version, we need to know **which pixels changed** so we can calibrate depth only on the parts of the scene that stayed the same.

Four methods are compared side-by-side:

| Method | How it works |
|---|---|
| **RGB threshold** | Pixel-level colour difference > threshold |
| **DINOv2** | Feature distance on 37×37 patch grid (518×518 input, 4 layers concat) |
| **GeSCF** | SAM ViT-B block 8 attention features, adaptive threshold (mean + k·std where k = clip(skewness, 1, 3)) |
| **Cross-attention** | From vendored `Robust-Scene-Change-Detection` repo |

**GeSCF performs best** — it handles lighting changes and subtle edits better than RGB thresholding, and doesn't require the heavy DINOv2 feature pipeline.

---

## How to run

```bash
# Run all 4 methods on a dataset and save results
python change_detection_results/test_change_detection.py --dataset depth4
```

Outputs go to `change_detection_results/{dataset}/`.

---

## Outputs

```
change_detection_results/
  depth4/
    rgb_depth4_mask.npy          # Binary change mask (True = changed)
    dinov2_depth4_mask.npy
    gescf_depth4_mask.npy
    rgb_depth4.png               # Visualisation of the mask
    dinov2_depth4.png
    gescf_depth4.png
    summary_depth4.png           # Side-by-side comparison of all methods
```

The `.npy` masks are loaded by the depth evaluation scripts and the point cloud notebooks.

---

## Notes

- Masks are pre-computed once and reused across depth experiments — you don't need to re-run this unless you change the dataset or method.
- The cross-attention baseline requires the vendored `Robust-Scene-Change-Detection` repo; the others have no extra dependencies.
