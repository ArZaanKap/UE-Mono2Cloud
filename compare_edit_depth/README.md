# Compare Edit Depth

Scripts for evaluating depth calibration accuracy — how close a scaled monocular prediction
is to Unreal Engine ground truth, on both unchanged regions and the newly added objects.

---

## Two scripts, two strategies

### `compare_edit_depth2.py` — **recommended (v2)**

Runs the depth model on the edited image only. Derives the change mask directly from
the GT depth difference `|depth_gt_edit − depth_gt_orig| > threshold` (no `.npy` masks needed).
Fits scale+shift on unchanged pixels, then evaluates on both unchanged and changed pixels.

This matches the real production scenario: one edited image, one model run, calibrate on
what you know is still correct, evaluate on the new objects.

**Requires two `SceneDepth` EXRs in the dataset folder** (original + edited). Raises an error
if only one is found — those datasets do not have GT for the edited scene.

### `compare_edit_depth.py` — v1 (legacy)

Runs the model on both original and edited images. Learns scale from the original vs GT
(all pixels), then applies it to the edited prediction. More expensive (two model runs)
and less realistic — in deployment you only ever have the edited image.

| | v1 | v2 |
|---|---|---|
| Scale learned from | original prediction vs GT (all pixels) | edited prediction vs GT (unchanged pixels only) |
| Change mask source | `.npy` file from `change_detection_results/` | GT depth diff (`\|edit − orig\| > threshold`) |
| Model runs | 2 | 1 |
| Output folder | `v1/{dataset}_results/` | `v2/{dataset}_results2/` |
| Datasets supported | `depth4`, `concrete1`, `test2` | `new0`, `new1` (and any with two GT depth EXRs) |

---

## How to run

```bash
# v2 (recommended) — requires two SceneDepth EXRs in data/{dataset}/
python compare_edit_depth/compare_edit_depth2.py --model dpro --dataset new0
python compare_edit_depth/compare_edit_depth2.py --model da3_giant --dataset new1 --scaling ls

# v1 (legacy, AI-edited datasets only)
python compare_edit_depth/compare_edit_depth.py --model dpro --dataset depth4 --mask-model gescf
```

**`--model` options:** `dpro`, `da3_giant`, `da3_nested`, `marigold_dc`

**`--scaling` options:** `ls` (least-squares, default), `median`

**`--change-threshold`:** depth difference in metres to mark a pixel as changed (default `0.05`)

---

## Tested depth models

| Model | Script support | Notes |
|---|---|---|
| Depth Pro | v1 and v2 | Main baseline, current best performer on unchanged regions |
| Depth Anything 3 Giant 1.1 | v2 only | Current DA3 option |
| Depth Anything 3 Nested Giant 1.1 | v2 only | Current DA3 option |
| Marigold-DC | v2 only | Uses sparse UE GT depth on unchanged pixels as guidance; no scale fit |
| Depth Anything V2 Metric | v1 only | Legacy (`da2` in v1); not in v2 |
| Metric3D v2 | v1 only | Legacy; appears only in `v1/` saved results |

---

## Datasets

| Dataset | GT edit depth? | Scripts | Notes |
|---|---|---|---|
| `new0` | yes | v2 | New UE-rendered edits — full evaluation |
| `new1` | yes | v2 | New UE-rendered edits — full evaluation |
| `depth4` | no | v1 | AI-edited image, original GT only |
| `concrete1` | no | v1 | AI-edited image, original GT only |
| `test2` | no | v1 | AI-edited image, original GT only |

---

## Metrics

### v2 (GT mode)

Evaluated against `depth_gt_edit` separately on unchanged and changed pixels.

| Metric | Region | Meaning |
|---|---|---|
| MAE | unchanged | Mean absolute error in metres — calibration quality |
| RMSE | unchanged | Root mean square error in metres |
| δ1 | unchanged | % pixels where `max(pred/GT, GT/pred) < 1.25` |
| MAE | **changed** | Error on newly added objects — the core new capability |
| RMSE | **changed** | Root mean square error on new objects |
| δ1 / δ2 / δ3 | **changed** | Standard monocular benchmark thresholds (1.25 / 1.25² / 1.25³) |

δ1/δ2/δ3 are standard monocular depth benchmark metrics, enabling comparison with published results.

### v1 (AI-edited datasets)

Evaluated against `depth_gt_orig` on unchanged pixels only (GT for the edit does not exist).

| Metric | Meaning |
|---|---|
| MAE | Mean absolute error in metres |
| RMSE | Root mean square error |
| % > 0.1 m | Fraction of pixels with error above 10 cm |
| % > 0.5 m | Fraction of pixels with error above 50 cm |

---

## v1 vs v2 comparison (clean with GT datasets)

With two GT depth maps available, the comparison is unambiguous — both strategies are
evaluated against the same target (`depth_gt_edit`) on the same changed pixels:

- **v1**: fit scale from original prediction vs `depth_gt_orig` (all pixels); apply to edited
  prediction; evaluate on changed pixels vs `depth_gt_edit`
- **v2**: fit scale from edited prediction vs `depth_gt_edit` (unchanged pixels only);
  evaluate on changed pixels vs `depth_gt_edit`

Same evaluation target, same pixels → directly comparable MAE/δ1 on new objects.

---

## Outputs

```
compare_edit_depth/
  v1/
    depth4_results/
      least_squares/   # results from compare_edit_depth.py
      median/
  v2/
    new0_results2/
      least_squares/
        dpro_visualization.png     # 2×4 grid: originals, GT depth, GT mask, error maps
        dpro_metrics.json          # MAE, RMSE, δ1/δ2/δ3 on unchanged + changed pixels
        gt_mask_new0.png           # visual check of the GT change mask
      median/
      guided_completion/           # Marigold-DC only
```

### Visualization layout (v2, 2×4 grid)

| | Col 0 | Col 1 | Col 2 | Col 3 |
|---|---|---|---|---|
| **Row 0** | Original image | Edited image | GT depth (edit) | GT change mask |
| **Row 1** | Scaled prediction | Full-image error | Error (unchanged) | Error (changed) |

---

## Marigold-DC note

`marigold_dc` is not just another monocular model. It consumes:

- the edited RGB image
- a sparse metric depth map built from UE GT on unchanged pixels (`0` elsewhere)

Because the model already uses known depth during inference, the usual scale fit is skipped.
Results go to `guided_completion/` instead of `least_squares/` or `median/`.

On this machine, Marigold-DC required:
- convert edited images to plain RGB (CLI rejects 4-channel PNG)
- resize RGB and sparse depth guide together (dimensions must match exactly)
- cap to 768-pixel long edge to fit RTX 3070 Ti 8 GB without CUDA OOM

---

## Best results so far

| Dataset | Model | Region | MAE | RMSE | δ1 |
|---|---|---|---|---|---|
| `depth4` (v1, unchanged only) | Depth Pro + GeSCF | unchanged | 4.2 cm | 6.5 cm | — |
| `new0` / `new1` (v2) | — | changed | TBD | TBD | TBD |
