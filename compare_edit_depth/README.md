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

**Requires two `SceneDepth` EXRs in the dataset folder** (original + edited). All `new*`
datasets have this — use them for evaluation. Raises an error if only one is found.

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
| Datasets | `depth4`, `concrete1`, `test2` | all `new*` datasets |

---

## How to run

```bash
# v2 (recommended) — requires two SceneDepth EXRs in data/{dataset}/
python compare_edit_depth/compare_edit_depth2.py --model dpro --dataset new0
python compare_edit_depth/compare_edit_depth2.py --model da3_giant --dataset new1 --scaling ls
python compare_edit_depth/compare_edit_depth2.py --model depthlab --dataset new4

# Run all models on a dataset at once
python compare_edit_depth/compare_edit_depth2.py --all-models --dataset new2

# v1 (legacy, AI-edited datasets only)
python compare_edit_depth/compare_edit_depth.py --model dpro --dataset depth4 --mask-model gescf
```

**`--model` options:** `dpro`, `da3_giant`, `da3_nested`, `marigold_dc`, `depthlab`

**`--scaling` options:** `ls` (least-squares, default), `median` — ignored for `marigold_dc` / `depthlab`

**`--change-threshold`:** depth difference in metres to mark a pixel as changed (default `0.05`)

**`--all-models`:** runs every model sequentially on the given dataset

---

## Tested depth models

| Model | Script support | Notes |
|---|---|---|
| Depth Pro | v1 and v2 | Best on changed regions across most datasets |
| Depth Anything 3 Giant 1.1 | v2 only | Current DA3 option; best on unchanged (low MAE) |
| Depth Anything 3 Nested Giant 1.1 | v2 only | Current DA3 option |
| Marigold-DC | v2 only | Diffusion depth completion; sparse GT as guidance; results in `guided_completion/` |
| DepthLab | v2 only | Dual-branch diffusion; excellent on unchanged but fails on changed pixels |
| Depth Anything V2 Metric | v1 only | Legacy (`da2` in v1); not in v2 |
| Metric3D v2 | v1 only | Legacy; appears only in `v1/` saved results |

---

## Datasets

| Dataset | GT edit depth? | Scripts | Notes |
|---|---|---|---|
| `new0` | yes | v2 | ~5.6% changed |
| `new1` | yes | v2 | ~8.9% changed |
| `new2` | yes | v2 | ~2.8% changed; GBuffer passes |
| `new3` | yes | v2 | ~11.6% changed; GBuffer passes |
| `new4` | yes | v2 | ~3.4% changed; GBuffer passes |
| `depth4` | no | v1 | AI-edited image, original GT only |
| `concrete1` | no | v1 | AI-edited image, original GT only |
| `test2` | no | v1 | AI-edited image, original GT only |

`new2_2` has change detection results but is not included in `compare_edit_depth2.py`.

---

## Metrics

### v2 (GT mode)

Evaluated against `depth_gt_edit` separately on unchanged and changed pixels.

| Metric | Region | Meaning |
|---|---|---|
| MAE | unchanged | Mean absolute error in metres — calibration quality |
| RMSE | unchanged | Root mean square error in metres |
| δ1 | unchanged | % pixels where `max(pred/GT, GT/pred) < 1.25` |
| MAE | **changed** | Error on newly added objects — the core capability |
| RMSE | **changed** | Root mean square error on new objects |
| δ1 / δ2 / δ3 | **changed** | Standard monocular benchmark thresholds (1.25 / 1.25² / 1.25³) |
| sna_mean / sna_median | both | Surface Normal Alignment — mean/median angular error in degrees (lower = better) |
| pct_11 / pct_22 / pct_30 | both | % pixels with SNA < 11.25° / 22.5° / 30° (higher = better) |

δ1/δ2/δ3 are standard monocular depth benchmark metrics, enabling comparison with published results.
SNA metrics are computed on later datasets (`new2`–`new4`) where `_WorldNormal.exr` is present.

### v1 (AI-edited datasets)

Evaluated against `depth_gt_orig` on unchanged pixels only (GT for the edit does not exist).

| Metric | Meaning |
|---|---|
| MAE | Mean absolute error in metres |
| RMSE | Root mean square error |
| % > 0.1 m | Fraction of pixels with error above 10 cm |
| % > 0.5 m | Fraction of pixels with error above 50 cm |

---

## Results on pair datasets (v2, changed region, least-squares)

MAE in metres on the changed pixels (new objects). Best per dataset in **bold**.

| Dataset | Depth Pro | DA3 Giant | DA3 Nested | DepthLab |
|---|---|---|---|---|
| new0 | 0.103 | **0.075** | **0.071** | — |
| new1 | **0.069** | 0.163 | 0.171 | — |
| new2 | **0.122** | 0.212 | 0.274 | 1.276 |
| new3 | **0.250** | 0.288 | 0.306 | — |
| new4 | **0.167** | 0.225 | 0.262 | 0.748 |

DepthLab on **unchanged** pixels: MAE ~1.4–2.6 cm (best of all models), d1 ≈ 100%.
DepthLab on **changed** pixels: MAE ~75–128 cm — the model propagates wrong depth to pixels
without guidance rather than generating new geometry.

Depth Pro is consistently best on changed regions. DA3 Giant / Nested tend to be better on
unchanged regions (lower MAE) but worse on new objects.

---

## Marigold-DC note

`marigold_dc` consumes the edited RGB image plus a sparse metric depth map built from UE GT
on unchanged pixels (`0` elsewhere). Results go to `guided_completion/` instead of
`least_squares/` or `median/`. No scale fit — guidance is used directly.

On this machine (RTX 3070 Ti 8 GB), Marigold-DC required:
- convert edited images to plain RGB (CLI rejects 4-channel PNG)
- resize RGB and sparse depth guide together (dimensions must match exactly)
- cap to 768-pixel long edge to avoid CUDA OOM

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
        metrics_data.json          # MAE, RMSE, δ1/δ2/δ3, SNA on unchanged + changed pixels
        gt_mask_new0.png           # visual check of the GT change mask
      median/
      guided_completion/           # Marigold-DC only
      depthlab/                    # DepthLab only
```

### Visualization layout (v2, 2×4 grid)

| | Col 0 | Col 1 | Col 2 | Col 3 |
|---|---|---|---|---|
| **Row 0** | Original image | Edited image | GT depth (edit) | GT change mask |
| **Row 1** | Scaled prediction | Full-image error | Error (unchanged) | Error (changed) |
