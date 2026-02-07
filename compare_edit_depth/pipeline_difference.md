# Pipeline Difference: v1 vs v2

Both are test scripts to evaluate calibration strategies for the main pipeline (img -> depth -> point cloud).
Both use pre-computed change masks from `change_detection_results/` (toggle with `--mask-model dinov2|gescf`).

**v1 (`compare_edit_depth.py`)** — Calibrate on original image.
Fits original prediction to GT (all pixels) to get scale factor, applies same factor to edited prediction.

**v2 (`compare_edit_depth2.py`)** — Calibrate on edited image (unchanged regions only).
Fits edited prediction to GT using unchanged pixels only, applies that factor to both predictions.

## Output comparison

**v1** (3x3 grid) — runs model on both images:

| Row | Left | Centre | Right |
|-----|------|--------|-------|
| 1 | Original image | Edited image | Change mask |
| 2 | GT depth | Pred original (scaled) | Pred edited (scaled) |
| 3 | Depth diff (unchanged) | Error: original - GT | Error: edited - GT |

**v2** (2x3 grid) — runs model on edited image only:

| Row | Left | Centre | Right |
|-----|------|--------|-------|
| 1 | Original image | Edited image | Change mask |
| 2 | GT depth | Pred edited (scaled) | Error: edited - GT |

**Comparable metric**: "Edit vs GT MAE" (unchanged regions) — same measurement, different scaling.

| | v1 | v2 |
|---|---|---|
| Scale source | original pred vs GT (all pixels) | edited pred vs GT (unchanged only) |
| Model runs | 2 (original + edited) | 1 (edited only) |
| Output folder | `{dataset}_results/` | `{dataset}_results2/` |
