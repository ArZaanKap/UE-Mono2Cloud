# v1 vs v2

## v1 — `compare_edit_depth.py`
Runs the model on the **original** image. Fits scale to original prediction vs GT across all pixels.
Applies the same scale to the edited prediction. Evaluates on unchanged pixels only.

**Problem:** unrealistic — in production you only have the edited image. Also, GT for
the edited scene is required to evaluate changed pixels, which this script cannot provide.

Results live under `compare_edit_depth/v1/`.

## v2 — `compare_edit_depth2.py` (use this)
Runs the model on the **edited** image only. Derives the GT change mask directly from
`|depth_gt_edit − depth_gt_orig| > 0.05 m` — no pre-computed `.npy` masks needed.
Fits scale to the edited prediction vs GT on unchanged pixels. Evaluates on both
unchanged and changed pixels, reporting MAE, RMSE, and δ1/δ2/δ3.

This is the realistic scenario: one image, one model run, calibrate on what you know
is still correct, measure error on the new objects.

**Requires two SceneDepth EXRs** — only datasets rendered entirely in UE (e.g. `new0`, `new1`).

Results live under `compare_edit_depth/v2/`.

---

The "Edit vs GT MAE (unchanged)" metric means the same thing in both scripts — calibration
accuracy on pixels that did not change. v2 also adds evaluation on changed pixels, which v1
cannot do because it does not have GT depth for the edited scene.
