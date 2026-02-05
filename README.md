# UE Mono2Cloud

**Turn AI-generated room concepts into 3D point clouds for Unreal Engine.**

## What is this?

DepthDreamer lets designers quickly visualize AI-generated interior concepts in 3D. Start with a half-finished room in Unreal Engine, use AI to "decorate" it, then bring that vision back into 3D space as a point cloud.

## Workflow

```
UE Scene (incomplete) → Export RGB + Depth
                              ↓
                     AI decorates the RGB
                              ↓
              Depth model estimates depth on both images
                              ↓
               Use GT depth to calibrate scale factor
                              ↓
              Apply scale to decorated depth output
                              ↓
                   Generate colored point cloud
                              ↓
                     Import back into UE
```

## Why?

- **Fast iteration**: See AI concepts in 3D without manual modeling
- **Spatial understanding**: Evaluate if AI suggestions work in the actual space
- **Design exploration**: Quickly test multiple decoration styles in context

## Key Components

| File | Purpose |
|------|---------|
| `img_to_pointcloud.ipynb` | Main pipeline: depth estimation → point cloud generation |
| `compare_all_models.py` | Benchmark depth models (Depth Pro, ZoeDepth, Metric3D, Depth Anything) |
| `analyze_depth.py` | Analyze UE depth exports and verify calibration |

## Depth Models Compared

| Model | RMSE | Best For |
|-------|------|----------|
| Depth Pro | 5.5cm | Production (best accuracy) |
| ZoeDepth | 6.1cm | Good alternative |
| Metric3D v2 | 6.8cm | Lightweight option |
| Depth Anything V2 | 8.0cm | Fast inference |

## Quick Start

```python
# In img_to_pointcloud.ipynb:
DATASET = "depth4"
MODEL_TYPE = "depth_pro"
SCALING_METHOD = "least_squares"
```

## Requirements

- Python 3.10+
- PyTorch with CUDA
- Depth Pro, ZoeDepth (separate venv), or other depth models
- Open3D, laspy for point cloud I/O

## File Formats

- **RGB**: `HighresScreenshot00000.exr` (UE high-res screenshot)
- **Depth GT**: `HighresScreenshot00000_SceneDepth.exr` (Z-buffer, use `GT_TO_CENTIMETERS=10000`)
- **Output**: `.las` point cloud (importable to UE via plugins)
