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

**`--model` options:** `dpro`, `da3_giant`, `da3_nested`, `da3_large`, `marigold_dc`, `depthlab`,
`unidepth_vitl`, `promptda_vitl`, `promptda_vitl_guided`, `moge2`, `unik3d`, `hyden`

**`--scaling` options:** `ls` (least-squares, default), `median` — ignored for `marigold_dc`,
`depthlab`, and `promptda_vitl_guided`

**`--change-threshold`:** depth difference in metres to mark a pixel as changed (default from `params.json`)

**`--all-models`:** runs every model sequentially on the given dataset

---

## Tested depth models

| Model | Key | Script support | Notes |
|---|---|---|---|
| Depth Pro | `dpro` | v1 and v2 | Most consistent on changed regions across datasets |
| Depth Anything 3 Giant 1.1 | `da3_giant` | v2 only | Native res (~1526px); best on new0/new3/new4 |
| Depth Anything 3 Nested Giant 1.1 | `da3_nested` | v2 only | Default `process_res=1024` |
| Depth Anything 3 Large 1.1 | `da3_large` | v2 only | Native res; inconsistent on small changed regions |
| Marigold-DC | `marigold_dc` | v2 only | Diffusion depth completion; sparse GT as guidance |
| DepthLab | `depthlab` | v2 only | Dual-branch diffusion; perfect on unchanged, fails on changed |
| Lotus-2 | tested only | removed | Briefly tested in v2; not kept because it was slow and not competitive enough |
| UniDepth-V2 ViT-L | `unidepth_vitl` | v1 and v2 | Metric model (no LS needed); raw output ~1.3–1.6× scale on UE synthetic; after LS d1=1.0 |
| PromptDA ViT-L (monocular) | `promptda_vitl` | v2 only | Relative output like DA3; scale 15–40× on UE; not intended use case |
| PromptDA ViT-L (guided) | `promptda_vitl_guided` | v2 only | Sparse GT guidance at unchanged pixels; fill strategy is the bottleneck — see below |
| MoGe-2 ViT-L | `moge2` | v1 and v2 | Relative depth model; can optionally use known camera FoV |
| UniK3D ViT-L | `unik3d` | v1 and v2 | Metric 3D model; used here via its depth output |
| HyDen DA2-Large | `hyden` | v1 and v2 | Relative depth model from MetaDepth; uses internal 518px ViT resize while preserving aspect ratio |
| DepthFM | tested only | — | Clearly weaker for this use case; removed |
| Depth Anything V2 Large | tested only | — | Not competitive; removed |
| Depth Anything V2 Metric | `da2` | v1 only | Legacy |
| Metric3D v2 | — | v1 only | Legacy |

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

DA3 Giant runs at native (~1526px); DA3 Nested at `process_res=1024`; DA3 Large at native.
Previous DA3 Giant results at `process_res=1024` shown in parentheses for reference.

| Dataset | Depth Pro | DA3 Giant native | DA3 Nested 1024px | DA3 Large native | DepthLab |
|---|---|---|---|---|---|
| new0 | 0.103 | **0.047** (was 0.063) | 0.057 | 0.059 | — |
| new1 | **0.069** | 0.134 (was 0.188) | 0.184 | 0.206 | — |
| new2 | **0.122** | 0.243 (was 0.234) | 0.218 | 0.391 | 1.276 |
| new3 | 0.250 | 0.171 (was 0.245) | 0.289 | **0.073** | — |
| new4 | 0.167 | **0.161** (was 0.168) | 0.202 | 0.382 | 0.748 |

DepthLab on **unchanged** pixels: MAE ~1.4–2.6 cm (best of all models), d1 ≈ 100%.
DepthLab on **changed** pixels: MAE ~75–128 cm — the model propagates wrong depth to pixels
without guidance rather than generating new geometry.

Depth Pro is most consistent on changed regions. DA3 Giant at native resolution improves
substantially over 1024px (new0: 0.063→0.047, new1: 0.188→0.134, new3: 0.245→0.171) and now
beats Depth Pro on new0 and new4. DA3 Large at native excels on new3 (large changed region,
11.6%) but fails on small changed regions (new2: 2.8%, new4: 3.4%).
DA3 Nested was tested at native but showed inconsistent gains — kept at 1024px.

---

## PromptDA guided — fill strategy experiments

PromptDA guided (`promptda_vitl_guided`) takes a sparse depth prompt at unchanged pixels
and infers depth for the changed region. The model architecture is DA3 backbone + prompt
fusion in the decoder neck. It outputs relative depth, normalised by the prompt's own
min/max and denormalised back; the HF processor expects prompt depths in **millimetres**
(it divides internally by 1000 to convert to metres).

### The fill problem

The sparse prompt has GT values at unchanged pixels and zeros at changed (unknown) pixels.
Passing zeros directly to PromptDA corrupts the normalisation (`depth_min = 0` collapses
the prompt range), producing near-zero output everywhere. The unknown pixels must be
**filled** before passing to the model.

All experiments below use `experiments_promptda_fill.py` and `experiments_promptda_ensemble.py`.

### Oracle experiment

Providing perfect GT depth at changed pixels (i.e., a fully dense GT prompt) gives
MAE 0.05–0.08m on changed pixels across all datasets. The model is not the bottleneck —
the fill quality is.

### Fill strategies tested

| Strategy | new0 | new1 | new2 | new3 | new4 | Notes |
|---|---|---|---|---|---|---|
| NN fill (baseline) | 0.177 | 0.324 | 0.443 | 0.778 | 0.261 | Nearest valid GT pixel; current default |
| DA3 Giant fill | **0.055** | 0.138 | 0.288 | **0.145** | **0.148** | DA3 LS-calibrated on unchanged; 1.5–5.4× improvement over NN |
| DepthPro fill | 0.131 | **0.100** | **0.144** | 0.231 | 0.195 | Best on new1/new2 where DA3 underperforms |
| Self-refinement (2-pass) | 0.176 | 0.316 | 0.463 | 0.633 | 0.268 | Pass-1 (NN) output fed back as fill for pass-2; negligible improvement |
| Blended (NN + DA3, σ=30px) | 0.144 | 0.154 | 0.517 | 0.201 | 0.279 | Distance-weighted blend; inconsistent |

**Key finding:** fill quality at changed pixels is the dominant factor. A calibrated monocular
prediction (DA3 or DepthPro) as fill vastly outperforms NN propagation, especially on large
changed regions (new3: 0.778→0.145). Self-refinement does not work — when the first pass has
large errors, feeding that output back as guidance cannot self-correct.

### Ensemble strategies

Both DA3 and DepthPro win on different datasets — DA3 on new0/new3/new4, DepthPro on new1/new2.
Ensemble strategies attempt to combine them automatically (`experiments_promptda_ensemble.py`).

| Strategy | new0 | new1 | new2 | new3 | new4 | Description |
|---|---|---|---|---|---|---|
| DA3 alone | **0.055** | 0.138 | 0.288 | **0.145** | 0.148 | Ref |
| DPro alone | 0.131 | 0.100 | **0.144** | 0.231 | 0.195 | Ref |
| **A: Simple average** | 0.083 | **0.085** | 0.175 | 0.187 | **0.141** | (DA3_cal + DPro_cal) / 2 |
| B: RMSE-weighted | 0.082 | **0.084** | 0.190 | 0.186 | **0.136** | Weight ∝ 1/RMSE on unchanged pixels |
| C: Local confidence | 0.070 | 0.096 | 0.188 | 0.202 | 0.139 | Gaussian-spread residuals (σ=60px) |
| D: Hard selection | **0.055** | 0.100 | 0.288 | **0.145** | 0.148 | Pick model with lower unchanged RMSE |

**Hard selection (D)** is correct on 4/5 datasets but fails on new2: DA3 fits unchanged pixels
better (RMSE 0.033 vs 0.043) yet is worse on changed pixels (0.288 vs 0.144). The unchanged
RMSE does not reliably predict changed-pixel performance when the changed region has
structurally different content from the rest of the scene.

**Simple average (A)** is the most robust choice. It beats both individual models on new1
(0.085 < 0.100 and 0.138) and new4 (0.141 < 0.148 and 0.195) via ensemble averaging, and
gracefully lands between them on the new2 failure case (0.175) rather than picking wrong.

### Summary: PromptDA guided MAE on changed pixels

| Strategy | new0 | new1 | new2 | new3 | new4 | Avg |
|---|---|---|---|---|---|---|
| NN fill (baseline) | 0.177 | 0.324 | 0.443 | 0.778 | 0.261 | 0.397 |
| DA3 fill | 0.055 | 0.138 | 0.288 | 0.145 | 0.148 | 0.155 |
| DepthPro fill | 0.131 | 0.100 | 0.144 | 0.231 | 0.195 | 0.160 |
| **Average ensemble (A)** | **0.083** | **0.085** | **0.175** | **0.187** | **0.141** | **0.134** |
| Oracle (GT fill) | ~0.07 | ~0.06 | ~0.05 | ~0.08 | ~0.06 | ~0.064 |

**Recommended fill strategy: simple average of DA3-calibrated and DepthPro-calibrated depth.**
3× better than NN fill, within 0.07m of oracle, never catastrophically wrong.

### Density invariance

As few as 0.5% of unchanged GT pixels as guidance gives the same result as 100%. The model's
performance is determined by the spatial NN structure of the fill, not the number of guidance
points.

---

## UniDepthV2 note

UniDepthV2 (`unidepth_vitl`) outputs absolute metric depth — no calibration is needed in
principle. However on UE synthetic scenes it systematically underestimates depth by 1.3–1.6×.
After LS calibration on unchanged pixels, δ1 = 1.0 (effectively perfect relative accuracy).

v2 calibration (fit on edited-image unchanged pixels) is much more reliable than v1 (fit on
original image). UniDepthV2 shifts its global scale estimate when scene content changes, so a
scale factor derived from the original image does not transfer cleanly to the edited image —
v1 unchanged MAE was 1.28–1.48m. v2 avoids this by fitting scale on the unchanged pixels of
the edited image directly.

Brief tuning note: the current default in `compare_edit_depth2.py` is UniDepth with
UE-derived pinhole intrinsics passed in, plus a 95th-percentile trimmed least-squares refit
on unchanged pixels. This was the best single universal setting across `new0`–`new4`
overall: average changed-region MAE `0.1095m`, beating the other monocular baselines on
average. The following UniDepth-specific variants were tested and rejected as defaults:
forcing near-full-size input (slower, usually worse), `resolution_level=0` (faster but less
accurate), and plain RGB-only inference without camera intrinsics (slightly worse average).

Raw (unscaled) MAE per dataset is stored in `metrics_data.json` under `raw_edit_unchanged` and
`raw_edit_changed` for this model only.

---

## Depth Pro note

Depth Pro has a fixed 1536×1536 square input. For 16:9 images (1526×858) it squashes the
image — height is stretched ~1.79×. Despite this distortion the model handles it well in
practice; it was trained on varied aspect ratios at this resolution.

**Letterbox padding was tested and rejected.** Padding to a square with black bars before
inference consistently hurt results (new1: 0.069→0.134m, new3: 0.250→0.261m). The model's
FOV estimation head is confused by large black regions, corrupting the depth structure in
the padded areas. Passing explicit `f_px` from `params.json` would not help because the
LS calibration on unchanged pixels already absorbs any metric scale error.

The original squash behaviour is kept.

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

## Lotus-2 note

Lotus-2 was tested briefly in the v2 benchmark and is mentioned here for record-keeping only.
It was removed from the active script because it was much slower on this GPU and did not earn
its complexity with better enough results than the stronger/faster baselines.

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
        dpro_visualization.png     # 3×4 grid: originals, GT depth, GT mask, error maps, normals
        metrics_data.json          # MAE, RMSE, δ1/δ2/δ3, SNA on unchanged + changed pixels
        gt_mask_new0.png           # visual check of the GT change mask
      median/
      guided_completion/           # Marigold-DC only
      depthlab/                    # DepthLab only
      promptda_guided/             # promptda_vitl_guided (NN fill baseline) only
```

### Visualization layout (v2, 3×4 grid)

| | Col 0 | Col 1 | Col 2 | Col 3 |
|---|---|---|---|---|
| **Row 0** | Original image | Edited image | GT depth (edit) | GT change mask |
| **Row 1** | Scaled prediction | Full-image error | Error (unchanged) | Error (changed) |
| **Row 2** | GT surface normals | Pred surface normals | Normal error (full) | Normal error (changed) |

Row 2 is populated only when `_WorldNormal.exr` and camera rotation params are present in
`params.json` (new2–new4). Otherwise row 2 is blank with an explanatory note.
