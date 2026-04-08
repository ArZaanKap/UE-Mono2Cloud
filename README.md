# UE Mono2Cloud

Turn AI-edited room images into 3D point clouds using monocular depth estimation, calibrated against Unreal Engine ground truth.

## Workflow

```
UE scene -> export RGB + GT depth (EXR)
         -> edit the RGB image
         -> detect changed regions
         -> run a monocular depth model on the edited image
         -> fit prediction to GT on unchanged regions
         -> back-project to a coloured point cloud (.las)
```

## Repo Guide

| Path | What it does |
|------|--------------|
| `img_to_pointcloud.ipynb` | Main notebook pipeline using Depth Pro for edited-image depth, GT-based calibration, and LAS export |
| `img_to_pointcloud2.ipynb` | Newer notebook variant using Depth Anything 3 (`da3_giant` / `da3_nested`) instead of Depth Pro |
| `mask_tests.ipynb` | Experimental notebook for tuning GeSCF-style change masks and thin-structure recovery |
| `compare_edit_depth/compare_edit_depth.py` | Evaluate depth consistency by scaling from the original-image prediction, then comparing original vs edited depth on unchanged regions |
| `compare_edit_depth/compare_edit_depth2.py` | Evaluate edited-image depth only, scaling from unchanged edited pixels vs GT |
| `compare_edit_depth/pipeline_difference.md` | Short note describing the difference between the two comparison scripts |
| `change_detection_results/test_change_detection.py` | Compare change-detection methods: RGB threshold, DINOv2 features, GeSCF-style SAM features, and pretrained DINOv2 + cross-attention |
| `analyze_depth.py` | Verify EXR depth channels and confirm the `GT_TO_CENTIMETERS` conversion for UE `SceneDepth` files |
| `data_analysis_report.txt` | Earlier analysis notes on Unreal depth formats and model alignment |
| `Depth-Anything-3/` | Vendored third-party repo used by the DA3 notebook and `compare_edit_depth2.py` |
| `Robust-Scene-Change-Detection/` | Vendored third-party repo used by the optional cross-attention change-detection baseline |

## Current State

- The end-to-end point-cloud workflow lives in notebooks, not in a packaged Python module.
- The Depth Pro path is in `img_to_pointcloud.ipynb`.
- The newer Depth Anything 3 path is in `img_to_pointcloud2.ipynb`.
- The scripted evaluation flow is centered on the `depth3` and `depth4` datasets.
- The notebooks currently use `test*` datasets, while the comparison scripts use `depth*` datasets.

## Depth Calibration

Monocular depth predictions are aligned to UE ground truth with a linear fit on unchanged pixels:

```python
depth_calibrated = prediction * scale + shift
```

The unchanged region mask comes from precomputed change-detection results, usually `gescf_{dataset}_mask.npy` or `dinov2_{dataset}_mask.npy` in `change_detection_results/{dataset}/`.

GT depth conversion for Unreal `SceneDepth.exr` files is:

```python
depth_meters = raw_value * 10000 / 100
```

## Metrics

The comparison scripts report metrics in meters on unchanged regions.

| Metric | Description |
|--------|-------------|
| `Orig vs GT MAE` | Baseline error for the original-image prediction after scaling |
| `Edit vs GT MAE` | Error for the edited-image prediction after scaling |
| `MAE` | Mean absolute difference between original and edited predictions on unchanged pixels |
| `RMSE` | Root mean square error |
| `% > 0.1m / 0.5m` | Fraction of unchanged pixels above the given error threshold |

`compare_edit_depth` writes JSON and markdown summaries per model. The notebooks print summary metrics inline and export LAS files.

## Data Layout

```
data/
  depth3/
  depth4/
  test1/
  test2/
  ...
    HighresScreenshot00000.exr
    HighresScreenshot00000_SceneDepth.exr
    *edit*.png

change_detection_results/
  depth3/
  depth4/
    gescf_{dataset}_mask.npy
    dinov2_{dataset}_mask.npy
    summary_{dataset}.png

weights/
  sam_vit_b_01ec64.pth

pointclouds/
pointclouds2/
```

## Quick Start

```bash
# Run change detection for a dataset
python change_detection_results/test_change_detection.py --dataset depth4

# Compare original vs edited depth (scale learned from original prediction)
python compare_edit_depth/compare_edit_depth.py --model depth_pro --dataset depth4 --mask-model gescf

# Compare edited-image depth only (scale learned from unchanged edited pixels)
python compare_edit_depth/compare_edit_depth2.py --model dpro --dataset depth4 --mask-model gescf
```

Notebook config lives at the top of each notebook. The main settings are `DATASET`, `CAMERA_FOV`, `GT_TO_CENTIMETERS`, and the chosen depth model variant.

## Dependencies

There is no pinned environment file in this repo yet. Based on current imports, the project expects:

- Python 3.10+
- PyTorch
- `OpenEXR`, `Imath`, `Pillow`, `numpy`, `scipy`, `opencv-python`, `matplotlib`, `laspy`
- `depth_pro`
- `segment-anything`
- `transformers`
- The vendored `Depth-Anything-3/` repo for DA3 experiments
- The vendored `Robust-Scene-Change-Detection/` repo for the optional cross-attention baseline
