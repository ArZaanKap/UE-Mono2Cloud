# Change Detection Results

This folder contains the script for comparing change detection methods, and the pre-computed mask outputs for each dataset.

---

## What this does

Given an original UE render and a user-edited version, we need to know **which pixels changed** so we can calibrate depth only on the parts of the scene that stayed the same.

Seven methods are compared side-by-side:

| Method | How it works |
|---|---|
| **RGB threshold** | Pixel-level colour difference > threshold |
| **DINOv2** | Feature distance on 37×37 patch grid (518×518 input, 4 layers concat) |
| **DINOv3** | Updated DINOv3 feature baseline |
| **GeSCF** | SAM ViT-B block 8 attention features, adaptive threshold (mean + k·std where k = clip(skewness, 1, 3)) |
| **Official GeSCF** | Official pretrained GeSCF weights from `mask_models/gescf-official/` |
| **ViewDelta** | From vendored `mask_models/viewdelta-scd/` |
| **Cross-attention** | From vendored `mask_models/Robust-Scene-Change-Detection/` |

**GeSCF performs best** — handles lighting changes and subtle edits better than RGB thresholding, and doesn't require the heavy DINOv2 feature pipeline.

For pair datasets (`new*`), a **GT mask** derived from `|depth_gt_edit − depth_gt_orig| > threshold` provides a ground-truth reference for evaluating change detection quality — see `gt_mask_comparison.ipynb`.

---

## How to run

```bash
# Run all methods on a dataset and save results
python change_detection_results/test_change_detection.py --dataset new0

# Skip expensive baselines
python change_detection_results/test_change_detection.py --dataset new4 --skip-dino --skip-crossattn
```

Outputs go to `change_detection_results/{dataset}/`.

Default method parameters are in `params.py` — edit there to keep the sweep notebooks in sync.

---

## Datasets processed

| Dataset | GT mask available? | Notes |
|---|---|---|
| `new0` | yes (via depth diff) | |
| `new1` | yes | |
| `new2` | yes | |
| `new2_2` | yes | Variant of new2 |
| `new3` | yes | |
| `new4` | yes | |
| `depth4` | no | AI edit |
| `depth3` | no | AI edit |
| `concrete1` | no | AI edit |
| `test2` | no | AI edit |

---

## Outputs

```
change_detection_results/
  new0/
    rgb_new0.png               # Visualisation of each mask
    dinov2_new0.png
    gescf_new0.png
    gt_mask_new0.png           # GT mask from depth diff (pair datasets only)
    summary_new0.png           # Side-by-side comparison of all methods
    detection_scores.json      # IoU / F1 scores vs GT mask (pair datasets only)
```

The `.npy` masks are loaded by v1 depth evaluation scripts and point cloud notebooks.
For v2 (`compare_edit_depth2.py`), the change mask is derived from GT depth diff directly — `.npy` files are not needed.

---

## Notes

- Masks are pre-computed once and reused across depth experiments — you don't need to re-run unless you change the dataset or method.
- The cross-attention baseline requires the vendored `Robust-Scene-Change-Detection` repo; the others have no extra dependencies beyond the main `requirements.txt`.
- `gt_mask_comparison.ipynb` — interactive notebook comparing predicted masks against the GT depth-diff mask across all pair datasets.
