# UE Mono2Cloud

Turn AI-edited room images into 3D point clouds using monocular depth estimation, calibrated against Unreal Engine ground truth.

## Workflow

```
UE Scene → Export RGB + GT Depth (EXR)
                    ↓
         AI edits the RGB image
                    ↓
      Detect changed regions (GeSCF / DINOv2)
                    ↓
      Run depth model on edited image
                    ↓
      Calibrate prediction → GT via unchanged regions
                    ↓
      Generate colored point cloud (.las)
```

## Project Structure

| File | What it does |
|------|--------------|
| `img_to_pointcloud.ipynb` | End-to-end pipeline: change detection, depth estimation (Depth Pro), least-squares calibration, LAS export |
| `compare_edit_depth/compare_edit_depth.py` | Evaluate depth consistency — runs model on original + edited image, scales from **original**, reports metrics on unchanged regions |
| `compare_edit_depth/compare_edit_depth2.py` | Same evaluation but scales from **edited image** unchanged regions instead |
| `compare_all_models.py` | Benchmark depth models (Depth Pro, Depth Anything V2, Metric3D v2) against GT |
| `change_detection_results/test_change_detection.py` | Compare change detection methods: RGB threshold, DINOv2, GeSCF (SAM Q/K/V), DINOv2+CrossAttn |
| `analyze_depth.py` | Inspect EXR depth files — verify channels, units, and `GT_TO_CENTIMETERS` |

## Depth Calibration

Monocular depth models output relative or coarsely-metric depth. We align predictions to UE ground truth using least-squares fitting on unchanged regions:

```
depth_calibrated = prediction * scale + shift
```

The change detection mask (GeSCF by default) identifies which pixels were not modified by the AI edit, so only those are used for fitting.

## Metrics

All evaluation scripts report metrics **in meters on unchanged regions**:

| Metric | Description |
|--------|-------------|
| MAE | Mean absolute error between prediction and GT |
| RMSE | Root mean square error (penalises large errors) |
| Edit vs GT MAE | How well the edited prediction matches GT after scaling |
| Orig vs GT MAE | Baseline — how well the original prediction matches GT |
| % > 0.1m / 0.5m | Fraction of pixels with error above threshold |

`compare_edit_depth` saves results to JSON + markdown tables per model. `img_to_pointcloud` prints MAE/RMSE to console only.

## Data Layout

```
data/
  depth3/                         # Test scene 1
  depth4/                         # Test scene 2
    HighresScreenshot00000.exr        # Original RGB (EXR, linear)
    HighresScreenshot00000_SceneDepth.exr  # GT depth
    *edit*.png                        # AI-edited image
change_detection_results/
  {dataset}/
    gescf_{dataset}_mask.npy          # Pre-computed change masks
weights/
  sam_vit_b_01ec64.pth              # SAM weights (auto-downloaded)
```

GT depth conversion: `depth_meters = raw_value * 10000 / 100`

## Quick Start

```python
# img_to_pointcloud.ipynb
DATASET = "depth4"
CAMERA_FOV = 90.0
GT_TO_CENTIMETERS = 10000.0
```

```bash
# Compare depth consistency across models
python compare_edit_depth/compare_edit_depth.py --model depth_pro --dataset depth4

# Run change detection
python change_detection_results/test_change_detection.py --dataset depth4

# Benchmark models against GT
python compare_all_models.py
```

## Requirements

- Python 3.10+, PyTorch + CUDA
- `depth_pro`, `segment-anything`, `transformers`
- `OpenEXR`, `laspy`, `scipy`, `opencv-python`
