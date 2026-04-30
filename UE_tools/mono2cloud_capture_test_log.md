# Mono2Cloud Capture Test Log

Updated: 2026-04-29

## Goal

Make Mono2Cloud's Unreal capture path produce beauty images that are as deterministic and as close as possible to a manual High Resolution Screenshot capture, so downstream DINOv2 masks stay stable.

## Current Capture Config

- Script: `UE_tools/mono2cloud_editor_mvp.py`
- Capture size: `1526 x 858`
- Warmup ticks: `24`
- Pipeline folder tags:
  - `depth_pro -> dpro`
  - `da3_giant -> da3g`

## What We Tested

### 1. Folder naming cleanup

- Kept the outer session folder long:
  - `<Saved>/Mono2Cloud/<timestamp>_<camera>/`
- Shortened inner pipeline folders and result names:
  - `dpro/` or `da3g/`
  - `result.las`
  - `result_summary.json`
  - `debug/`

### 2. Depth visualization range

- Changed depth preview/debug visualization upper percentile from `99` to `99.9`.

### 3. Capture warmup

- Added viewport warmup before `HighResShot`.
- Increased default warmup from `8` to `24` ticks on 2026-04-29.
- Reason:
  - likely gives TAA / TSR / shadows / lighting history time to settle after piloting the camera.

### 3b. Deterministic capture overrides

Added fixed capture-time overrides in `UE_tools/mono2cloud_editor_mvp.py`:

- `capture_screen_percentage = None`
- `capture_aa_quality = None`
- `force_realtime_viewport = True`

Behavior:

- Script can apply these before warmup/capture when set.
- Script restores previous `r.ScreenPercentage` and `sg.AntiAliasingQuality` afterward.
- Script also writes these values into the session `params.json`.

Reason:

- reduce differences caused by inheriting editor viewport settings
- make scripted captures more reproducible across runs

Observed on 2026-04-29:

- forcing `r.ScreenPercentage = 100` and `sg.AntiAliasingQuality = 4` made the image look worse / less pretty than the manual capture
- current recommendation is:
  - keep `capture_warmup_ticks = 24`
  - keep `force_realtime_viewport = True`
  - do not force screen percentage or AA quality for now

### 4. UE-script original vs manual original comparison

Compared:

- UE-script original:
  - `D:\Documents\Unreal Projects\MyProjectDepth2\Saved\Mono2Cloud\20260429_121436_CameraActor\original.png`
- Manual original:
  - `data/new2/HighresScreenshot00000.png`

Comparison bundle:

- `temp/ue_vs_new2_original_compare/`

Result after the manual image was re-captured to match dimensions:

- Both images: `1526 x 858`
- Mean absolute RGB difference: about `4.22`
- RMSE: about `7.10`

Interpretation:

- The images are closer now that framing matches.
- They are still not identical.
- Remaining difference is likely due to viewport/render-state differences rather than filename or aspect mismatch.

### 5. DINOv2 mask using UE-script original

Ran `change_detection_results` DINOv2 manually using:

- original:
  - latest UE-script capture `original.png`
- edited:
  - `data/new2/HighresScreenshot00001.png`

Saved outputs:

- `data/new2/ue_script_original_latest.png`
- `data/new2/dinov2_ueorig_mask.png`
- `data/new2/dinov2_ueorig_overlay.png`
- `data/new2/dinov2_ueorig_diff.png`
- `data/new2/dinov2_ueorig_meta.json`

Observed:

- Changed fraction: about `12.24%`

Interpretation:

- This differs noticeably from the existing `change_detection_results` `new2` DINOv2 output.
- So the original beauty image alone is enough to move the mask significantly.

## Current Working Hypothesis

The Mono2Cloud scripted capture still differs from the manual High Resolution Screenshot mostly because it inherits editor viewport state.

Most likely factors:

- temporal AA / TSR history settling
- editor viewport screen percentage
- editor scalability / AA quality
- exposure / post-process state
- realtime viewport state during warmup

Less likely primary cause:

- buffer visualization dumping itself
- result-folder naming

## Next Tests

1. Force deterministic capture settings in the script before `HighResShot`:
   - fixed screen percentage
   - fixed AA scalability
   - ensure realtime viewport during warmup
2. Re-run:
   - manual high-res screenshot
   - Mono2Cloud scripted capture
   - image diff bundle
   - DINOv2 mask comparison
3. If needed, tune:
   - `capture_screen_percentage`
   - `capture_aa_quality`
   - `capture_warmup_ticks`

## Notes

- If we want the most reproducible beauty capture long-term, we may need a more controlled render path than the live editor viewport, such as Movie Render Queue or a dedicated capture workflow.
