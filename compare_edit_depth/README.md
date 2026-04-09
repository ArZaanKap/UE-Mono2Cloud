# Compare Edit Depth

Scripts for evaluating how well the depth calibration works — i.e. how close the scaled monocular prediction is to UE ground truth on unchanged regions.

---

## Two scripts, two strategies

### `compare_edit_depth2.py` — **recommended (v2)**
Runs the depth model on the edited image only. Fits scale+shift using GT on unchanged pixels.
This matches the real production scenario: you only have the edited image, not the original.

### `compare_edit_depth.py` — v1
Runs the model on both original and edited images. Learns scale from the original vs GT, then applies it to the edited prediction.
More expensive (two model runs), and less realistic.

| | v1 | v2 |
|---|---|---|
| Scale learned from | original prediction vs GT (all pixels) | edited prediction vs GT (unchanged pixels only) |
| Model runs | 2 | 1 |
| Output folder | `{dataset}_results/` | `{dataset}_results2/` |
| Comparable metric | "Edit vs GT MAE" | "Edit vs GT MAE" |

---

## How to run

```bash
# v2 (recommended)
python compare_edit_depth/compare_edit_depth2.py --model dpro --dataset depth4 --mask-model gescf

# v1
python compare_edit_depth/compare_edit_depth.py --model depth_pro --dataset depth4 --mask-model gescf
```

**`--model` options:** `dpro`, `da2`, `da3_giant`, `da3_nested`, `marigold_dc`
**`--mask-model` options:** `gescf`, `dinov2`, `rgb`

## Tested depth models

This repo has tested the following depth models across the depth-comparison scripts and saved results:

| Model | Script support | Notes |
|---|---|---|
| Depth Pro | v1 and v2 | Main baseline and current best performer |
| Depth Anything V2 Metric | v1 and v2 | Referred to as `depth_anything` in v1 and `da2` in v2 |
| Depth Anything 3 Giant 1.1 | v2 only | Current DA3 option |
| Depth Anything 3 Nested Giant 1.1 | v2 only | Current DA3 option |
| Marigold-DC | v2 only | Uses sparse UE GT depth on unchanged pixels as guidance; no least-squares fit |
| Metric3D v2 | v1 only | Legacy experiment retained in `compare_edit_depth.py` and `compare_edit_depth/v1/` results folders |

If you want the current, production-like evaluation path, use `compare_edit_depth2.py`. Metric3D v2 appears only in the older v1 script and its saved `v1/depth3_results/` and `v1/depth4_results/` outputs.

---

## Outputs

```
compare_edit_depth/
  v1/
    depth4_results/
      least_squares/   # Results from compare_edit_depth.py
      median/
  v2/
    depth4_results2/
      least_squares/   # Results from compare_edit_depth2.py
      median/
      guided_completion/ # Results for Marigold-DC using sparse GT guidance
```

Each subfolder contains:
- A visualisation image (2×3 or 3×3 grid of depth maps and error maps)
- A JSON/markdown file with the numeric metrics

### Metrics (on unchanged, non-sky pixels)

| Metric | Meaning |
|---|---|
| MAE | Mean absolute error in metres |
| RMSE | Root mean square error in metres |
| % > 0.1m | Fraction of pixels with error above 10 cm |
| % > 0.5m | Fraction of pixels with error above 50 cm |

## Marigold-DC note

`marigold_dc` is not just another monocular model. It consumes:

- the edited RGB image
- a sparse metric depth map built from UE GT on unchanged pixels, with `0` in changed / unknown regions

That means the change mask is still used, but indirectly: it defines which GT pixels become sparse guidance. Because the model already uses known depth during inference, `compare_edit_depth2.py` skips the usual median / least-squares calibration step for this model.

On this machine, getting Marigold-DC to run cleanly also required a few wrapper details:

- convert edited images to plain RGB before inference because the official CLI rejects 4-channel PNG input
- resize the edited RGB and sparse depth guide together before inference so their dimensions match exactly
- cap the Marigold input to a 768-pixel long edge to fit the RTX 3070 Ti 8 GB GPU without CUDA OOM

---

## Best result so far

`depth4`, Depth Pro, GeSCF mask, least-squares: **MAE = 4.2 cm, RMSE = 6.5 cm**
