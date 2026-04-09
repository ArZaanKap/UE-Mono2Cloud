# UE Mono2Cloud

**Goal:** Take a room scene rendered in Unreal Engine, let a user edit the image (e.g. add furniture or objects), then reconstruct a metric 3D point cloud of the edited scene — without ever re-rendering in UE.

The core challenge: monocular depth models give relative depth, not metric. We solve this by calibrating against UE's ground truth depth on the parts of the image that *weren't* changed.

---

## Pipeline

```
UE render ──► RGB image (.exr)  +  Ground truth depth (.exr)
                │
                ▼
         User edits the RGB image  (adds objects, changes scene)
                │
                ▼
         Change detection  →  mask of what changed vs what stayed the same
                │
                ▼
         Monocular depth model  →  predicted depth for the edited image
                │
                ▼
         Calibration  →  fit prediction to GT on unchanged pixels (least-squares)
                │
                ▼
         Back-project pixels to 3D  →  export coloured point cloud (.las)
```

---

## Where to Start

| If you want to... | Go to |
|---|---|
| Generate a point cloud from an edited image | `MAIN_TEST/img_to_pointcloud_depth_pro.ipynb` (Depth Pro) or `MAIN_TEST/img_to_pointcloud_da3.ipynb` (DA3) |
| Generate a point cloud with sparse UE depth guidance | `MAIN_TEST/img_to_pointcloud_marigold.ipynb` |
| Understand change detection and compare methods | `change_detection_results/` |
| Evaluate how well the depth calibration works | `compare_edit_depth/` |
| Understand the dataset format | `data/README.md` |
| Verify the GT depth conversion formula | `analyze_depth.py` |

---

## Repo Map

```
UE_depth/
│
├── MAIN_TEST/
│   ├── img_to_pointcloud_depth_pro.ipynb          # Main Depth Pro notebook
│   ├── img_to_pointcloud_depth_pro_legacy.ipynb   # Older Depth Pro variant
│   ├── img_to_pointcloud_da3.ipynb                # DA3 notebook
│   ├── img_to_pointcloud_marigold.ipynb           # Marigold-DC pipeline
│   ├── pointclouds_depth_pro/                     # Depth Pro outputs
│   ├── pointclouds_depth_pro_legacy/              # Legacy Depth Pro outputs
│   ├── pointclouds_da3/                           # DA3 outputs
│   └── pointclouds_marigold/                      # Marigold-DC outputs
├── analyze_depth.py              # One-off: verify GT depth unit conversion
│
├── change_detection_results/     # Scripts + outputs for change detection (masking) experiments
├── compare_edit_depth/           # Scripts + outputs for depth calibration evaluation
│
├── data/                         # Input datasets (UE renders + edited images)
│
├── Depth-Anything-3/             # Vendored: DA3 model repo (third-party)
├── Robust-Scene-Change-Detection/ # Vendored: cross-attention change detection (third-party)
│
├── weights/                      # SAM model weights (gitignored)
└── checkpoints/                  # Depth Pro weights (gitignored)
```

---

## Key Concepts

### GT Depth Conversion
UE exports `SceneDepth` as a Z-buffer (perpendicular distance, not Euclidean). Convert to metres:
```python
depth_metres = raw_value * 10000 / 100
```
Verified with a flat wall at 90cm (`data/depth_gt2`).

### Calibration
Monocular models give relative depth. We fit a linear scale+shift on unchanged pixels:
```python
depth_calibrated = prediction * scale + shift   # least-squares fit to GT
```
Typical scale factor: 0.54–0.80 (models tend to overestimate depth).

### Sky Handling
Sky pixels are detected via a "density knee" on the GT depth histogram — the point where the dense scene cluster transitions to a sparse sky tail. Importantly, the GT sky mask only applies to *unchanged* pixels; changed regions (e.g. a new object placed in front of sky) use the predicted depth to decide what to keep.

### Depth Models
| Model | Notebook | Script flag |
|---|---|---|
| Depth Pro | `MAIN_TEST/img_to_pointcloud_depth_pro.ipynb` | `--model dpro` |
| Depth Anything 3 Giant | `MAIN_TEST/img_to_pointcloud_da3.ipynb` | `--model da3_giant` |
| Depth Anything 3 Nested | `MAIN_TEST/img_to_pointcloud_da3.ipynb` | `--model da3_nested` |

### Tested Depth Models
These are the monocular depth models that have been tested in this repo so far.

| Model | Status | Where it appears |
|---|---|---|
| Depth Pro | Current | `MAIN_TEST/img_to_pointcloud_depth_pro.ipynb`, `compare_edit_depth.py`, `compare_edit_depth2.py` |
| Depth Anything V2 Metric | Tested in evaluation | Early analysis notes, `compare_edit_depth.py`, `compare_edit_depth2.py` |
| Depth Anything 3 Giant 1.1 | Current | `MAIN_TEST/img_to_pointcloud_da3.ipynb`, `compare_edit_depth2.py` |
| Depth Anything 3 Nested Giant 1.1 | Current | `MAIN_TEST/img_to_pointcloud_da3.ipynb`, `compare_edit_depth2.py` |
| Marigold-DC | Evaluation baseline | `compare_edit_depth2.py`; uses sparse UE GT depth on unchanged pixels |
| Metric3D v2 | Legacy evaluation only | `compare_edit_depth.py` and saved `compare_edit_depth/v1/*_results/` metrics |

Metric3D v2 was tested in the older v1 comparison pipeline, but it is not part of the current recommended v2 workflow or the point-cloud notebooks.

### Change Detection Methods
| Method | Description |
|---|---|
| RGB threshold | Simple pixel difference |
| DINOv2 | Feature distance on 37×37 patch grid |
| GeSCF | SAM ViT-B attention features, adaptive threshold — **best performer** |
| Cross-attention | From vendored Robust-Scene-Change-Detection repo |

---

## Quick Start

```bash
# 1. Pre-compute change masks for a dataset
python change_detection_results/test_change_detection.py --dataset depth4

# 2. Evaluate depth calibration (v2 — recommended)
python compare_edit_depth/compare_edit_depth2.py --model dpro --dataset depth4 --mask-model gescf

# Optional: evaluate sparse-guided depth completion
python compare_edit_depth/compare_edit_depth2.py --model marigold_dc --dataset depth4 --mask-model gescf

# 3. Generate a point cloud (open notebook, set DATASET at top, run all cells)
#    MAIN_TEST/img_to_pointcloud_depth_pro.ipynb
#    MAIN_TEST/img_to_pointcloud_da3.ipynb

# 4. Generate a point cloud with Marigold-DC (open notebook, run all cells)
#    MAIN_TEST/img_to_pointcloud_marigold.ipynb
```

Notebook settings to configure at the top of each notebook:
- `DATASET` — which folder in `data/` to use
- `CAMERA_FOV` — field of view in degrees (90° for our UE scenes)
- `GT_TO_CENTIMETERS` — depth unit conversion (10000.0)

---

## Results Summary

Best configuration: **Depth Pro + GeSCF mask + least-squares calibration**

| Dataset | MAE (unchanged regions) | RMSE | % pixels > 10cm error |
|---|---|---|---|
| depth4 | 4.2 cm | 6.5 cm | 5.4% |

---

## Dependencies

GPU is required for practical use of this repo. The models may fall back to CPU in some places, but the full pipeline is too slow to be usable without a GPU.

Install the public Python packages with:

```bash
pip install -r requirements.txt
```

Setup steps that are not covered by `requirements.txt` are documented in `setup.md`.
