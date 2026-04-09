# Setup

This file covers the setup steps that are not fully handled by [`requirements.txt`](./requirements.txt).

The goal is that a new person can start from a fresh clone of this repo and follow the steps below to get the whole working environment ready.

GPU is required for practical use of this repo. Some code paths may technically run on CPU, but the full pipeline is too slow to be considered a usable setup without a GPU.

## 1. Create and activate a virtual environment

From the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 2. Install PyTorch

If you are using a GPU, install the correct CUDA build of PyTorch first from the official PyTorch install page, then come back here.

If you are using CPU only, this is usually enough:

```powershell
pip install torch torchvision
```

## 3. Install the main repo requirements

```powershell
pip install -r requirements.txt
```

This installs the public pip packages used directly by the tracked scripts and notebooks.

## 4. Download the manual model weights

### 4a. Depth Pro checkpoint

The `depth-pro` package expects this file to exist at:

```text
checkpoints/depth_pro.pt
```

Create the folder if needed:

```powershell
New-Item -ItemType Directory -Force checkpoints
```

Then download the official Depth Pro checkpoint from the Apple Depth Pro project and save it as:

```text
checkpoints/depth_pro.pt
```

Project page:

```text
https://github.com/apple/ml-depth-pro
```

### 4b. SAM ViT-B checkpoint

This repo uses the SAM ViT-B checkpoint at:

```text
weights/sam_vit_b_01ec64.pth
```

You can download it manually with PowerShell:

```powershell
New-Item -ItemType Directory -Force weights
Invoke-WebRequest `
  -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" `
  -OutFile "weights/sam_vit_b_01ec64.pth"
```

Note: `change_detection_results/test_change_detection.py` will also auto-download this file if it is missing.

## 5. Install the local DA3 repo

This is required for:

- `MAIN_TEST/img_to_pointcloud_da3.ipynb`
- `compare_edit_depth/compare_edit_depth2.py` when using `--model da3`, `--model da3_giant`, or `--model da3_nested`

Clone the repo into the project root:

```powershell
git clone https://github.com/ByteDance-Seed/Depth-Anything-3 .\Depth-Anything-3
```

Then install it:

```powershell
pip install -e .\Depth-Anything-3
```

Why this step matters:

- the notebook/script imports `depth_anything_3`
- DA3 has extra dependencies not covered by this repo's `requirements.txt`
- DA3 model weights are downloaded automatically from Hugging Face on first run

## 6. Install the optional CrossAttention change-detection repo

This is only needed if you want to run the optional CrossAttention baseline in:

- `change_detection_results/test_change_detection.py` without `--skip-crossattn`

Clone it with submodules:

```powershell
git clone --recursive https://github.com/ChadLin9596/Robust-Scene-Change-Detection .\Robust-Scene-Change-Detection
```

Then install its local packages:

```powershell
pip install -e .\Robust-Scene-Change-Detection\thirdparties\py_utils
pip install --no-deps -e .\Robust-Scene-Change-Detection
```

The pretrained CrossAttention checkpoints are handled by that repo's own loader.

If you do not need this baseline, you can skip this whole section and use:

```powershell
python .\change_detection_results\test_change_detection.py --dataset depth4 --skip-crossattn
```

## 7. Install the optional Marigold-DC repo

This is needed only if you want to run the new sparse-guided depth-completion baseline in:

- `compare_edit_depth/compare_edit_depth2.py` with `--model marigold_dc`
- `MAIN_TEST/img_to_pointcloud_marigold.py`

Clone the repo into the project root:

```powershell
git clone https://github.com/prs-eth/Marigold-DC .\Marigold-DC
```

Then install its requirements into the same environment:

```powershell
pip install -r .\Marigold-DC\requirements.txt
```

Why this step matters:

- `marigold_dc` is run via the official `python -m marigold_dc` entrypoint
- it needs the `diffusers`-based dependencies from the Marigold-DC repo
- it uses a sparse metric depth map built from UE GT on unchanged pixels

Practical note from bringing this up in this repo:

- the Marigold install pulled `numpy>=2`; after testing, the environment was restored to `numpy<2` so it still matches this repo's pinned requirements
- the wrapper here converts `edit.png` to RGB before inference because the official CLI rejects 4-channel PNG input
- the wrapper resizes the RGB image and sparse depth guide together before inference so their dimensions match exactly
- the wrapper caps Marigold to a 768-pixel long edge on this RTX 3070 Ti 8 GB system to avoid CUDA OOM
- the sparse guide uses UE GT on unchanged pixels and `0` elsewhere; `--processing_resolution 0` is passed so the CLI does not silently resize again after that

## 8. Verify the environment

Run these imports from the repo root:

```powershell
@'
import depth_pro
import OpenEXR
import laspy
import cv2
import transformers
from segment_anything import sam_model_registry
print("core imports ok")
try:
    from depth_anything_3.api import DepthAnything3
    print("da3 import ok")
except Exception as e:
    print("da3 import missing:", e)
try:
    from robust_scene_change_detect.models import get_model_from_pretrained
    print("crossattn import ok")
except Exception as e:
    print("crossattn import missing:", e)
try:
    import diffusers
    print("marigold deps ok")
except Exception as e:
    print("marigold deps missing:", e)
'@ | python -
```

## 9. Quick smoke test

Minimal working path for the core repo:

```powershell
python .\change_detection_results\test_change_detection.py --dataset depth4 --skip-crossattn
python .\compare_edit_depth\compare_edit_depth2.py --model dpro --dataset depth4 --mask-model gescf --no-show
```

If both commands run, the main non-DA3 pipeline is set up correctly.

If Marigold-DC is installed, you can also test:

```powershell
python .\compare_edit_depth\compare_edit_depth2.py --model marigold_dc --dataset depth4 --mask-model gescf --no-show
python .\MAIN_TEST\img_to_pointcloud_marigold.py --dataset depth4 --no-show
```

## What is optional vs required

Required for the main recommended pipeline:

- virtual environment
- PyTorch
- `pip install -r requirements.txt`
- `checkpoints/depth_pro.pt`
- `weights/sam_vit_b_01ec64.pth`

Required only for DA3 workflows:

- `Depth-Anything-3/`
- `pip install -e .\Depth-Anything-3`

Required only for Marigold-DC evaluation:

- `Marigold-DC/`
- `pip install -r .\Marigold-DC\requirements.txt`

Required only for the optional CrossAttention baseline:

- `Robust-Scene-Change-Detection/`
- `pip install -e .\Robust-Scene-Change-Detection\thirdparties\py_utils`
- `pip install --no-deps -e .\Robust-Scene-Change-Detection`
