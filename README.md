# UE Mono2Cloud

**Goal:** Take a room scene rendered in Unreal Engine, let a user edit the image (e.g. add furniture or objects), then reconstruct a metric 3D point cloud of the edited scene — without ever re-rendering in UE.

The core challenge: monocular depth models give relative depth, not metric. We solve this by calibrating against UE's ground truth depth on the parts of the image that *weren't* changed.

---

## Pipeline

```
UE render ──► RGB image (.png/.exr)  +  Ground truth depth (.exr)
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
                         or: diffusion depth completion guided by GT on unchanged pixels
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
| Explore surface normals / SNA metrics | `UE_understanding/` |

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
│
├── analyze_depth.py              # One-off: verify GT depth unit conversion
│
├── change_detection_results/     # Scripts + outputs for change detection experiments
├── compare_edit_depth/           # Scripts + outputs for depth calibration evaluation
│
├── data/                         # Input datasets (UE renders + edited images)
│
├── UE_understanding/             # Experiments with world normals and SNA metrics
│
├── depth_models/
│   ├── Depth-Anything-3/         # Vendored: DA3 model repo
│   ├── DepthLab/                 # Vendored: DepthLab diffusion model
│   └── Marigold-DC/              # Vendored: Marigold-DC diffusion model
│
├── mask_models/
│   ├── gescf-official/           # Vendored: official GeSCF weights
│   ├── Robust-Scene-Change-Detection/  # Vendored: cross-attention change detection
│   └── viewdelta-scd/            # Vendored: ViewDelta change detection
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
Typical scale factor: 0.54–5.0 (varies by model and scene depth range).

### Sky Handling
Sky pixels are detected via a "density knee" on the GT depth histogram — the point where the dense scene cluster transitions to a sparse sky tail. Importantly, the GT sky mask only applies to *unchanged* pixels; changed regions (e.g. a new object placed in front of sky) use the predicted depth to decide what to keep.

### Depth Models

| Model | Notebook | Script flag |
|---|---|---|
| Depth Pro | `MAIN_TEST/img_to_pointcloud_depth_pro.ipynb` | `--model dpro` |
| Depth Anything 3 Giant | `MAIN_TEST/img_to_pointcloud_da3.ipynb` | `--model da3_giant` |
| Depth Anything 3 Nested | `MAIN_TEST/img_to_pointcloud_da3.ipynb` | `--model da3_nested` |

### Tested Depth Models

| Model | Status | Where it appears |
|---|---|---|
| Depth Pro | Current | `MAIN_TEST/img_to_pointcloud_depth_pro.ipynb`, `compare_edit_depth.py`, `compare_edit_depth2.py` |
| Depth Anything 3 Giant 1.1 | Current | `MAIN_TEST/img_to_pointcloud_da3.ipynb`, `compare_edit_depth2.py` |
| Depth Anything 3 Nested Giant 1.1 | Current | `MAIN_TEST/img_to_pointcloud_da3.ipynb`, `compare_edit_depth2.py` |
| Marigold-DC | Evaluated | `compare_edit_depth2.py`; sparse UE GT on unchanged pixels as guidance |
| DepthLab | Evaluated | `compare_edit_depth2.py`; excellent on unchanged, fails on changed pixels |
| Depth Anything V2 Metric | Legacy | Early analysis notes, `compare_edit_depth.py` |
| Metric3D v2 | Legacy | `compare_edit_depth.py` and saved `compare_edit_depth/v1/*_results/` metrics |

### Change Detection Methods

| Method | Description |
|---|---|
| RGB threshold | Simple pixel difference |
| DINOv2 / DINOv3 | Feature distance on 37×37 patch grid |
| GeSCF | SAM ViT-B attention features, adaptive threshold — **best performer** |
| Official GeSCF | Official pretrained GeSCF weights |
| ViewDelta | Vendored from `mask_models/viewdelta-scd/` |
| Cross-attention | From vendored `Robust-Scene-Change-Detection` repo |

---

## Quick Start

```bash
# 1. Pre-compute change masks for a dataset
python change_detection_results/test_change_detection.py --dataset new0

# 2. Evaluate depth calibration on all models (v2 — recommended)
python compare_edit_depth/compare_edit_depth2.py --all-models --dataset new0

# 3. Generate a point cloud (open notebook, set DATASET at top, run all cells)
#    MAIN_TEST/img_to_pointcloud_depth_pro.ipynb
#    MAIN_TEST/img_to_pointcloud_da3.ipynb
#    MAIN_TEST/img_to_pointcloud_marigold.ipynb
```

Notebook settings to configure at the top of each notebook:
- `DATASET` — which folder in `data/` to use
- `CAMERA_FOV` — field of view in degrees (90° for our UE scenes)
- `GT_TO_CENTIMETERS` — depth unit conversion (10000.0)

---

## Results Summary

Evaluated on pair datasets (`new*`) — both original and edited renders from UE, so GT depth is available for the new objects. Metric: MAE on changed pixels (lower is better).

| Dataset | Depth Pro | DA3 Giant | DA3 Nested | Changed % |
|---|---|---|---|---|
| new0 | 10.3 cm | 7.5 cm | **7.1 cm** | 5.6% |
| new1 | **6.9 cm** | 16.3 cm | 17.1 cm | 8.9% |
| new2 | **12.2 cm** | 21.2 cm | 27.4 cm | 2.8% |
| new3 | **25.0 cm** | 28.8 cm | 30.6 cm | 11.6% |
| new4 | **16.7 cm** | 22.5 cm | 26.2 cm | 3.4% |

Depth Pro is the most consistent winner on changed regions. DA3 Giant/Nested tend to achieve lower unchanged-region error but fall behind on new objects.

DepthLab (tested on new2, new4): MAE ~1–3 cm on unchanged (best of all models), but ~75–128 cm on changed — it propagates incorrect depth to pixels without guidance rather than generating new geometry.

Previous best on legacy dataset: **Depth Pro + GeSCF mask + least-squares → MAE 4.2 cm, RMSE 6.5 cm** on `depth4` (unchanged pixels only; no GT for changed objects).

---

## Dependencies

GPU is required for practical use of this repo. The models may fall back to CPU in some places, but the full pipeline is too slow to be usable without a GPU.

Install the public Python packages with:

```bash
pip install -r requirements.txt
```

Setup steps that are not covered by `requirements.txt` are documented in `setup.md`.
