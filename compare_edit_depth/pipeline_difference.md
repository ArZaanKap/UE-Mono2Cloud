# v1 vs v2

## v1 — `compare_edit_depth.py`
Runs the model on the **original** image. Fits scale to original prediction vs GT across all pixels. Applies the same scale to the edited prediction.

**Problem:** unrealistic — in production you only have the edited image.

## v2 — `compare_edit_depth2.py` (use this)
Runs the model on the **edited** image only. Fits scale to the edited prediction vs GT, but **only on unchanged pixels** (from the pre-computed `.npy` mask).

This is the realistic scenario: one image, one model run, calibrate on what you know is still correct.

---

The "Edit vs GT MAE" metric means the same thing in both — error on unchanged pixels after scaling. v2 just learns the scale from the right source.
