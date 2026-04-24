"""
Change Detection Model Comparison
==================================
Runs change detection models on a dataset image pair and writes binary masks
(.npy) and visualisations (.png) to change_detection_results/output/{dataset}/

Default parameters are in params.py — edit there to keep sweep notebooks in sync.
SAM checkpoint paths are also configured in params.py (SAM2_BASELINE / SAM3_BASELINE).

Usage
-----
  # Run all default models on a dataset
  python change_detection_results/test_change_detection.py --dataset new3

  # Run every model at once
  python change_detection_results/test_change_detection.py --dataset new3 \
      --models rgb dinov2 dinov3 gescf ogescf viewdelta crossattn sam2 sam3

  # Tune per-model parameters
  python change_detection_results/test_change_detection.py --dataset new3 \
      --models dinov2 dinov3 gescf ogescf crossattn viewdelta \
      --rgb-threshold 30 \
      --dino-threshold 0.3 --dino-sigma 2 --dino-min-area 500 --dino-dilate 3 \
      --dinov3-threshold 0.3 --dinov3-sigma 2 --dinov3-min-area 500 --dinov3-dilate 3 \
      --gescf-threshold 0.5 \
      --ogescf-points 32 --ogescf-iou 0.88 --ogescf-stability 0.95 \
      --crossattn-threshold 0.4 --crossattn-model dino_2Cross_PSCD \
      --viewdelta-prompt "object changes" --viewdelta-threshold 0.2

  # SAM 3.1 — auto VLM prompt, or override model / prompt manually
  python change_detection_results/test_change_detection.py --dataset new3 \
      --models sam2 sam3 \
      --sam3-vlm-model google/gemma-4-E2B-it \
      --sam3-text-prompt "construction equipment,scaffolding"

  # Score against ground-truth depth EXRs
  python change_detection_results/test_change_detection.py --dataset new3 --change-threshold 0.02 --masks-only


  # Skip inference — only re-visualise masks already saved in the dataset folder
  python change_detection_results/test_change_detection.py --dataset new3 --masks-only

  # Suppress the summary figure (faster batch runs)
  python change_detection_results/test_change_detection.py --dataset new3 --no-show

Available models: rgb  dinov2  dinov3  gescf  ogescf  viewdelta  crossattn  sam2  sam3  dinov3_sam2  dinov2_sam2
"""

import os, sys, glob, argparse, json, warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", message=".*use_fast.*")
warnings.filterwarnings("ignore", message=".*grid_sample.*")
warnings.filterwarnings("ignore", message=".*Xet Storage.*")
warnings.filterwarnings("ignore", message=".*timm.models.layers.*")
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy import ndimage

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_ROOT  = os.path.join(SCRIPT_DIR, "output")
sys.path.insert(0, SCRIPT_DIR)

from params import (DINO_BASELINE, DINOV3_BASELINE, GESCF_BASELINE, RGB_BASELINE,
                    CROSSATTN_BASELINE, VIEWDELTA_BASELINE, OFFICIAL_GESCF_BASELINE,
                    SAM2_BASELINE, SAM3_BASELINE, DINOV3_SAM2_BASELINE, DINOV2_SAM2_BASELINE,
                    DATASET_CHANGE_THRESHOLDS)

DEFAULT_DATASET          = "new3"
DEFAULT_CHANGE_THRESHOLD = 0.0   # metres — for GT mask derivation from depth diff

AVAILABLE_DATASETS = ['depth4', 'concrete1', 'test2', 'new0', 'new1', 'new2', 'new2_2', 'new3', 'new4']


# ---------------------------------------------------------------------------
# Image / depth loading
# ---------------------------------------------------------------------------

def load_exr_depth(exr_path, gt_to_cm=10000.0):
    """Load depth from EXR file, convert to metres."""
    import OpenEXR, Imath
    exr_file = OpenEXR.InputFile(exr_path)
    header   = exr_file.header()
    dw       = header['dataWindow']
    width    = dw.max.x - dw.min.x + 1
    height   = dw.max.y - dw.min.y + 1
    FLOAT    = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = list(header['channels'].keys())
    for chan_name in ['R', 'SceneDepth', 'Z']:
        if chan_name in channels:
            channel_str = exr_file.channel(chan_name, FLOAT)
            depth = np.frombuffer(channel_str, dtype=np.float32).reshape(height, width).copy()
            break
    else:
        raise ValueError(f"No depth channel found in {exr_path}")
    return (depth * gt_to_cm) / 100.0


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_exr_rgb(exr_path):
    import OpenEXR, Imath
    exr = OpenEXR.InputFile(exr_path)
    dw  = exr.header()['dataWindow']
    w   = dw.max.x - dw.min.x + 1
    h   = dw.max.y - dw.min.y + 1
    FLT = Imath.PixelType(Imath.PixelType.FLOAT)
    rgb = np.stack(
        [np.frombuffer(exr.channel(c, FLT), dtype=np.float32).reshape(h, w) for c in 'RGB'],
        axis=-1,
    )
    return Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8))


def load_image(path):
    if path.lower().endswith('.exr'):
        return load_exr_rgb(path)
    return Image.open(path).convert('RGB')


def find_image_pair(folder):
    """Return (original_path, edited_path).

    Handles two naming conventions:
    - UE 00000/00001: two SceneDepth EXRs present → sorted PNGs (first=orig, second=edit)
    - Legacy: EXR original + PNG with 'edit' in filename
    """
    files = sorted(os.listdir(folder))

    scene_depth_exrs = sorted([
        f for f in files
        if 'SceneDepth' in f and 'WorldUnits' not in f and f.lower().endswith('.exr')
    ])
    rgb_pngs = sorted([
        f for f in files
        if f.lower().endswith('.png') and 'depth' not in f.lower()
    ])

    # UE GT mode: two SceneDepth EXRs → sorted PNGs are original + edited
    if len(scene_depth_exrs) >= 2 and len(rgb_pngs) >= 2:
        return (
            os.path.join(folder, rgb_pngs[0]),
            os.path.join(folder, rgb_pngs[1]),
        )

    original = next(
        (os.path.join(folder, f) for f in files
         if f.lower().endswith('.exr')
         and 'depth' not in f.lower()
         and 'scenedepth' not in f.lower()
         and 'normal' not in f.lower()),
        None,
    )
    edited = next(
        (os.path.join(folder, f) for f in files
         if 'edit' in f.lower() and f.lower().endswith(('.png', '.jpg', '.exr'))),
        None,
    )
    return original, edited


def find_extra_masks(folder):
    """Return list of (name, path) for any PNG files with 'mask' in the filename."""
    files = sorted(os.listdir(folder))
    return [
        (os.path.splitext(f)[0], os.path.join(folder, f))
        for f in files
        if 'mask' in f.lower() and f.lower().endswith('.png')
    ]


def find_depth_pair(folder):
    """Return (depth_gt_orig, depth_gt_edit) — edit depth may be None."""
    files = sorted(os.listdir(folder))
    scene_depth_exrs = sorted([
        f for f in files
        if 'SceneDepth' in f and 'WorldUnits' not in f and f.lower().endswith('.exr')
    ])
    if len(scene_depth_exrs) >= 2:
        return (
            os.path.join(folder, scene_depth_exrs[0]),
            os.path.join(folder, scene_depth_exrs[1]),
        )
    if len(scene_depth_exrs) == 1:
        return os.path.join(folder, scene_depth_exrs[0]), None
    return None, None


# ---------------------------------------------------------------------------
# Change detection methods
# ---------------------------------------------------------------------------

def rgb_threshold_mask(img1, img2,
                        threshold=RGB_BASELINE['threshold'],
                        min_area=RGB_BASELINE['min_area'],
                        dilate_iter=RGB_BASELINE['dilate_iter']):
    arr1 = np.array(img1).astype(float)
    arr2 = np.array(img2).astype(float)
    diff = np.abs(arr1 - arr2).mean(axis=2)
    mask = diff > threshold
    mask = _refine(mask, min_area=min_area, dilate_iter=dilate_iter)
    return mask, diff


def dinov2_feature_mask(img1, img2,
                         threshold=DINO_BASELINE['threshold'],
                         sigma=DINO_BASELINE['sigma'],
                         min_area=DINO_BASELINE['min_area'],
                         dilate_iter=DINO_BASELINE['dilate_iter'],
                         model_name="facebook/dinov2-with-registers-base"):
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    family = "DINOv3" if "dinov3" in model_name.lower() else "DINOv2"
    print(f"  Loading {family} ({model_name}) on {device} ...")
    try:
        processor = AutoImageProcessor.from_pretrained(
            model_name,
            local_files_only=True,
            size={"height": 518, "width": 518},
            crop_size={"height": 518, "width": 518},
        )
    except Exception:
        try:
            processor = AutoImageProcessor.from_pretrained(
                model_name,
                size={"height": 518, "width": 518},
                crop_size={"height": 518, "width": 518},
            )
        except Exception as first_error:
            # Older DINO checkpoints can share the DINOv2 preprocessing.
            print("  (no registered processor — falling back to dinov2-base preprocessor)")
            try:
                processor = AutoImageProcessor.from_pretrained(
                    "facebook/dinov2-base",
                    local_files_only=True,
                    size={"height": 518, "width": 518},
                    crop_size={"height": 518, "width": 518},
                )
            except Exception as fallback_error:
                try:
                    processor = AutoImageProcessor.from_pretrained(
                        "facebook/dinov2-base",
                        size={"height": 518, "width": 518},
                        crop_size={"height": 518, "width": 518},
                    )
                except Exception:
                    raise first_error from fallback_error

    try:
        model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device).eval()
    except Exception:
        model = AutoModel.from_pretrained(model_name).to(device).eval()
    num_register = int(getattr(model.config, "num_register_tokens", 0) or 0)

    with torch.no_grad():
        inp1 = processor(images=img1, return_tensors="pt").to(device)
        inp2 = processor(images=img2, return_tensors="pt").to(device)
        out1 = model(**inp1, output_hidden_states=True)
        out2 = model(**inp2, output_hidden_states=True)
        skip = 1 + num_register
        f1 = torch.cat([out1.hidden_states[i][:, skip:, :] for i in [3, 6, 9, 12]], dim=-1)
        f2 = torch.cat([out2.hidden_states[i][:, skip:, :] for i in [3, 6, 9, 12]], dim=-1)
        distance = (1 - (F.normalize(f1, dim=-1) * F.normalize(f2, dim=-1)).sum(dim=-1)).squeeze()

    h, w = np.array(img1).shape[:2]
    grid = int(np.sqrt(distance.shape[0]))
    dist_map = F.interpolate(
        distance.reshape(1, 1, grid, grid), size=(h, w), mode='bilinear', align_corners=False,
    ).squeeze().cpu().numpy()
    dist_map = ndimage.gaussian_filter(dist_map, sigma=sigma)

    mask = _refine(dist_map > threshold, min_area=min_area, dilate_iter=dilate_iter)
    return mask, dist_map


def gescf_feature_mask(img1, img2,
                        threshold=None,
                        points_per_side=GESCF_BASELINE['points_per_side'],
                        pred_iou_thresh=GESCF_BASELINE['pred_iou_thresh'],
                        stability_score_thresh=GESCF_BASELINE['stability_score_thresh'],
                        min_mask_region_area=GESCF_BASELINE['min_mask_region_area'],
                        overlap_frac=GESCF_BASELINE['overlap_frac']):
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        raise ImportError("segment-anything not installed. Run: pip install segment-anything")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    weights_dir  = os.path.join(PROJECT_ROOT, "mask_models", "weights")
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, "sam_vit_b_01ec64.pth")
    if not os.path.exists(weights_path):
        print("  Downloading SAM ViT-B weights (~375 MB) ...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            weights_path,
        )

    print(f"  Loading SAM ViT-B on {device} ...")
    sam = sam_model_registry["vit_b"](checkpoint=weights_path).to(device).eval()

    h, w = np.array(img1).shape[:2]

    captured = {}
    hook = sam.image_encoder.blocks[8].attn.qkv.register_forward_hook(
        lambda m, i, o: captured.__setitem__("qkv", o.detach())
    )

    from segment_anything.utils.transforms import ResizeLongestSide
    img_size  = sam.image_encoder.img_size
    transform = ResizeLongestSide(img_size)
    scale     = img_size / max(h, w)
    feat_h    = int(h * scale + 0.5) // 16
    feat_w    = int(w * scale + 0.5) // 16

    def _prep(pil_img):
        t = transform.apply_image(np.array(pil_img))
        return sam.preprocess(torch.as_tensor(t, device=device).permute(2, 0, 1).unsqueeze(0).float())

    with torch.no_grad():
        sam.image_encoder(_prep(img1)); qkv1 = captured["qkv"]
        sam.image_encoder(_prep(img2)); qkv2 = captured["qkv"]
    hook.remove()

    with torch.no_grad():
        f1, f2 = qkv1.squeeze(0), qkv2.squeeze(0)
        dist_map = (1 - (F.normalize(f1, dim=-1) * F.normalize(f2, dim=-1)).sum(dim=-1)).cpu().numpy()

    dist_map = dist_map[:feat_h, :feat_w]
    dist_map = F.interpolate(
        torch.tensor(dist_map).float().unsqueeze(0).unsqueeze(0),
        size=(h, w), mode='bilinear', align_corners=False,
    ).squeeze().numpy()
    dist_map = ndimage.gaussian_filter(dist_map, sigma=4)
    d_min, d_max = dist_map.min(), dist_map.max()
    if d_max - d_min > 1e-8:
        dist_map = (dist_map - d_min) / (d_max - d_min)

    if threshold is None:
        from scipy.stats import skew
        sk        = skew(dist_map.ravel())
        k         = float(np.clip(sk, 1.0, 3.0))
        threshold = float(dist_map.mean() + k * dist_map.std())
        print(f"  Adaptive threshold: {threshold:.4f}  (skew={sk:.2f}, k={k:.2f})")
    else:
        print(f"  Fixed threshold: {threshold:.4f}")

    initial_mask = dist_map > threshold

    print(f"  Running SAM (points_per_side={points_per_side}) ...")
    mask_gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )
    sam_masks = mask_gen.generate(np.array(img2))
    print(f"  SAM generated {len(sam_masks)} segments")

    refined_mask = np.zeros((h, w), dtype=bool)
    for seg in sam_masks:
        m = seg["segmentation"]
        if m.sum() > 0 and np.logical_and(m, initial_mask).sum() / m.sum() > overlap_frac:
            refined_mask |= m

    if refined_mask.sum() == 0:
        print("  SAM refinement empty — falling back to initial threshold mask")
        refined_mask = initial_mask

    return refined_mask, dist_map


def dino_crossattn_mask(img1, img2, threshold=0.5, pretrained="dino_2Cross_PSCD"):
    """DINOv2 + CrossAttention pretrained scene change detection (ICRA 2025)."""
    try:
        from robust_scene_change_detect.models import get_model_from_pretrained
    except ImportError as e:
        raise ImportError(
            "robust-scene-change-detection not installed.\n"
            "  cd {root}\n"
            "  git clone https://github.com/ChadLin9596/Robust-Scene-Change-Detection\n"
            "  cd Robust-Scene-Change-Detection\n"
            "  git submodule update --init --recursive\n"
            "  pip install -e thirdparties/py_utils && pip install --no-deps -e ."
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading DINOv2+CrossAttn ({pretrained}) on {device} ...")
    model = get_model_from_pretrained(pretrained)
    if hasattr(model, 'module'):
        model = model.module
    model = model.to(device).eval()

    h, w = np.array(img1).shape[:2]

    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize((504, 504)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    t0 = preprocess(img1).unsqueeze(0).to(device)
    t1 = preprocess(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(t0, t1)

    if pred.dim() == 4 and pred.shape[-1] == 2:
        prob_map = torch.softmax(pred, dim=-1)[0, :, :, 1].cpu().numpy()
    elif pred.dim() == 4 and pred.shape[1] == 2:
        prob_map = torch.softmax(pred, dim=1)[0, 1].cpu().numpy()
    else:
        prob_map = torch.sigmoid(pred).squeeze().cpu().numpy()

    print(f"  Threshold: {threshold}  (prob range: {prob_map.min():.3f}–{prob_map.max():.3f})")

    pred_mask = Image.fromarray((prob_map > threshold).astype(np.uint8) * 255)
    mask_full = np.array(pred_mask.resize((w, h), Image.NEAREST)) > 127
    prob_full = np.array(
        Image.fromarray((prob_map * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    ).astype(np.float32) / 255.0

    return mask_full, prob_full


def official_gescf_mask(img_t0_path, img_t1_path,
                        output_size=OFFICIAL_GESCF_BASELINE['output_size'],
                        feature_facet=OFFICIAL_GESCF_BASELINE['feature_facet'],
                        feature_layer=OFFICIAL_GESCF_BASELINE['feature_layer'],
                        embedding_layer=OFFICIAL_GESCF_BASELINE['embedding_layer'],
                        points_per_side=OFFICIAL_GESCF_BASELINE['points_per_side'],
                        pred_iou_thresh=OFFICIAL_GESCF_BASELINE['pred_iou_thresh'],
                        stability_score_thresh=OFFICIAL_GESCF_BASELINE['stability_score_thresh']):
    """Official GeSCF (CVPR 2025) — SAM ViT-H + SuperPoint coarse alignment + geometric-semantic mask matching."""
    import argparse

    gescf_src = os.path.join(PROJECT_ROOT, "mask_models", "gescf-official", "src")
    if gescf_src not in sys.path:
        sys.path.insert(0, gescf_src)

    try:
        from framework import GeSCF
    except ImportError as e:
        raise ImportError(
            f"Official GeSCF not found. Expected at {gescf_src}\n"
            "  git clone https://github.com/1124jaewookim/towards-generalizable-scene-change-detection.git gescf-official"
        ) from e

    weights_dir = os.path.join(gescf_src, "pretrained_weight")
    os.makedirs(weights_dir, exist_ok=True)

    sam_path = os.path.join(weights_dir, "sam_vit_h_4b8939.pth")
    if not os.path.exists(sam_path):
        print("  Downloading SAM ViT-H (~2.5 GB) ...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", sam_path
        )

    sp_path = os.path.join(weights_dir, "superpoint_v1.pth")
    if not os.path.exists(sp_path):
        print("  Downloading SuperPoint weights (~5 MB) ...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth",
            sp_path,
        )

    args = argparse.Namespace(
        test_dataset='Random',
        output_size=output_size,
        img_t0_path=img_t0_path,
        img_t1_path=img_t1_path,
        gt_path=None,
        feature_facet=feature_facet,
        feature_layer=feature_layer,
        embedding_layer=embedding_layer,
        sam_backbone='vit_h',
        pseudo_backbone='vit_h',
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
    )

    # framework.py uses CWD-relative paths — chdir to src/ for the duration of inference
    old_cwd = os.getcwd()
    os.chdir(gescf_src)
    try:
        model = GeSCF(args)
        mask_out = model(img_t0_path, img_t1_path)   # (output_size, output_size) uint8 0/1
    finally:
        os.chdir(old_cwd)

    # Resize to original image resolution
    orig = np.array(Image.open(img_t0_path))
    h, w = orig.shape[:2]
    mask_full = np.array(
        Image.fromarray(mask_out.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
    ) > 127

    # GeSCF has no probability map — use the binary mask as float diff_map
    diff_map = mask_full.astype(np.float32)
    return mask_full, diff_map


def sam2_mask(img1, img2, checkpoint, model_cfg='configs/sam2.1/sam2.1_hiera_l.yaml',
              diff_thresh=15, dilate=8, min_area_frac=0.001):
    """SAM 2 change mask: pixel diff → bounding box of largest changed region → SAM 2 box prompt.
    Needs only the two RGB images — no text, no GT depth."""
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        raise ImportError(
            "SAM 2 not installed. Run:\n"
            "  pip install git+https://github.com/facebookresearch/sam2.git\n"
            "Checkpoints: https://github.com/facebookresearch/sam2#model-description"
        )

    arr1 = np.array(img1.convert('RGB'), dtype=np.float32)
    arr2 = np.array(img2.convert('RGB'), dtype=np.float32)
    diff = np.abs(arr1 - arr2).max(axis=-1)          # max over RGB channels
    raw  = diff > diff_thresh
    if dilate > 0:
        raw = ndimage.binary_dilation(raw, iterations=dilate)

    H, W = arr1.shape[:2]
    labeled, n = ndimage.label(raw)
    if n == 0 or ndimage.sum(raw, labeled, range(1, n + 1)).max() < H * W * min_area_frac:
        print("  [SAM2] No significant diff region — using full image as box.")
        box = np.array([0, 0, W, H], dtype=np.float32)
    else:
        sizes = ndimage.sum(raw, labeled, range(1, n + 1))
        best  = int(np.argmax(sizes)) + 1
        comp  = labeled == best
        rows  = np.where(np.any(comp, axis=1))[0]
        cols  = np.where(np.any(comp, axis=0))[0]
        pad   = 12
        box   = np.array([max(0, cols[0] - pad), max(0, rows[0] - pad),
                           min(W, cols[-1] + pad), min(H, rows[-1] + pad)], dtype=np.float32)
    print(f"  [SAM2] Diff bbox: {box.astype(int).tolist()}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  [SAM2] Loading model on {device} …")
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))
    with torch.inference_mode():
        predictor.set_image(np.array(img2.convert('RGB')))
        masks, scores, _ = predictor.predict(
            point_coords=None, point_labels=None,
            box=box, multimask_output=True,
        )
    mask = masks[int(np.argmax(scores))].astype(bool)
    return mask, diff


def dinov3_sam2_mask(img1, img2,
                     checkpoint,
                     model_cfg='configs/sam2.1/sam2.1_hiera_l.yaml',
                     dino_threshold=DINOV3_SAM2_BASELINE['dino_threshold'],
                     sigma=DINOV3_SAM2_BASELINE['sigma'],
                     dilate=DINOV3_SAM2_BASELINE['dilate'],
                     model_name=DINOV3_SAM2_BASELINE['model_name'],
                     min_area_frac=0.001):
    """DINO feature distance map → one box per significant changed region → SAM2.

    Finds all changed regions above min_area_frac, runs SAM2 once per box,
    and unions the resulting masks so multiple changed objects are captured.
    """
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        raise ImportError(
            "SAM 2 not installed. Run:\n"
            "  pip install git+https://github.com/facebookresearch/sam2.git"
        )

    # Step 1 — DINO feature distance map
    tag = 'DINOv3→SAM2' if 'dinov3' in model_name.lower() else 'DINOv2→SAM2'
    _, dist_map = dinov2_feature_mask(
        img1, img2,
        threshold=dino_threshold,
        sigma=sigma,
        min_area=0,
        dilate_iter=0,
        model_name=model_name,
    )

    # Step 2 — find ALL significant changed regions
    H, W = dist_map.shape
    raw = dist_map > dino_threshold
    if dilate > 0:
        raw = ndimage.binary_dilation(raw, iterations=dilate)

    labeled, n = ndimage.label(raw)
    min_pixels = H * W * min_area_frac
    boxes = []
    if n > 0:
        sizes = ndimage.sum(raw, labeled, range(1, n + 1))
        for idx, size in enumerate(sizes, 1):
            if size < min_pixels:
                continue
            comp = labeled == idx
            rows = np.where(np.any(comp, axis=1))[0]
            cols = np.where(np.any(comp, axis=0))[0]
            pad  = 12
            boxes.append(np.array([
                max(0, cols[0] - pad), max(0, rows[0] - pad),
                min(W, cols[-1] + pad), min(H, rows[-1] + pad),
            ], dtype=np.float32))

    if not boxes:
        print(f"  [{tag}] No significant feature diff — using full image as box.")
        boxes = [np.array([0, 0, W, H], dtype=np.float32)]

    print(f"  [{tag}] {len(boxes)} region(s) found")
    for b in boxes:
        print(f"  [{tag}]   bbox: {b.astype(int).tolist()}")

    # Step 3 — SAM2: one predict per region, union all masks
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  [{tag}] Loading SAM2 on {device} …")
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))
    combined = np.zeros((H, W), dtype=bool)
    with torch.inference_mode():
        predictor.set_image(np.array(img2.convert('RGB')))
        for box in boxes:
            masks, scores, _ = predictor.predict(
                point_coords=None, point_labels=None,
                box=box, multimask_output=True,
            )
            combined |= masks[int(np.argmax(scores))].astype(bool)
    return combined, dist_map


def sam3_mask(img1, img2, checkpoint,
              text_prompt=SAM3_BASELINE['text_prompt'],
              vlm_model=SAM3_BASELINE['vlm_model']):
    """
    SAM 3.1 change mask: VLM generates a precise text description of what
    changed, which is passed as a text prompt to SAM 3.1.

    If text_prompt is provided it is used directly (VLM step is skipped).
    text_prompt may be a string, comma-separated string, or list of prompts.
    Otherwise vlm_model is loaded to auto-generate the description.

    Requires:
      pip install git+https://github.com/facebookresearch/sam3.git
    Weights (gated): https://huggingface.co/facebook/sam3.1
    """
    # Import shared helpers from sam_mask_infer (same directory)
    from sam_mask_infer import generate_vlm_description, run_sam3_text

    prompt = text_prompt
    if prompt is None:
        print(f"  [SAM3] No text_prompt — running VLM ({vlm_model}) to generate one ...")
        prompt = generate_vlm_description(
            img1,
            img2,
            vlm_model,
        )
        print(f"  [SAM3] VLM raw output: '{prompt}'")
        print(f"  [SAM3] Prompt fed to SAM3: '{prompt}'")
    else:
        print(f"  [SAM3] Using text prompt: '{prompt}'")

    mask = run_sam3_text(img2, prompt, checkpoint)
    # Return a dummy diff map (binary mask as float) since SAM3 has no scalar diff map
    return mask, mask.astype(np.float32)


def viewdelta_mask(img1, img2,
                   text_prompt=VIEWDELTA_BASELINE['text_prompt'],
                   threshold=VIEWDELTA_BASELINE['threshold']):
    """ViewDelta text-conditioned scene change detection (ICCV 2025)."""
    viewdelta_dir = os.path.join(PROJECT_ROOT, "mask_models", "viewdelta-scd")
    if viewdelta_dir not in sys.path:
        sys.path.insert(0, viewdelta_dir)
    try:
        from ViewDelta.model.transformer_args import TransformerModelArgs
        from ViewDelta.embedders import get_embedders, get_model_features_from_image
        from ViewDelta.model.model_feature_segmentor import TextConditionedDecoder
    except ImportError as e:
        raise ImportError(
            "ViewDelta not found. Clone the repo:\n"
            f"  git clone https://github.com/drags99/viewdelta-scd.git {viewdelta_dir}\n"
            "  pip install einops kornia lightning timm"
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"

    weights_dir      = os.path.join(PROJECT_ROOT, "mask_models", "weights")
    checkpoint_path  = os.path.join(weights_dir, "viewdelta_checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        print("  Downloading ViewDelta checkpoint (~201 MB) ...")
        import urllib.request
        os.makedirs(weights_dir, exist_ok=True)
        urllib.request.urlretrieve(
            "https://huggingface.co/hoskerelab/ViewDelta/resolve/main/viewdelta_checkpoint.pth",
            checkpoint_path,
        )

    model_args = TransformerModelArgs(
        text_embeddings="siglip",
        image_embeddings="dinov2",
        use_multiscale=False,
        use_separation_tokens=False,
        depth=12, dim=768, mlp_dim=3072, heads=12,
        checkpoint_attn=False, checkpoint_ff=False,
    )
    model_args.text_tokens          = 64
    model_args.text_embedding_dim   = 1024
    model_args.img_tokens           = 257
    model_args.image_embedding_dim  = 1024

    print(f"  Loading ViewDelta checkpoint ...")
    model = TextConditionedDecoder(model_args).to(device).eval()
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    print(f"  Loading embedders (DINOv2-large + SigLIP) ...")
    emb = get_embedders(model_args)

    h, w    = np.array(img1).shape[:2]
    img1_s  = img1.resize((256, 256))
    img2_s  = img2.resize((256, 256))

    with torch.no_grad():
        f1 = get_model_features_from_image(
            img1_s, emb["image_model"], emb["image_processor"], model_args
        ).to(device)
        f2 = get_model_features_from_image(
            img2_s, emb["image_model"], emb["image_processor"], model_args
        ).to(device)

        text_tokens = emb["text_processor"](
            text=text_prompt, padding="max_length", return_tensors="pt"
        )
        # text_model stays on CPU — run there and move output to device afterwards
        text_feats = emb["text_model"](**text_tokens)["last_hidden_state"].detach().to(device)

        output      = model(f1, f2, text_feats)
        prob_256    = torch.softmax(output, dim=1)[0, 1].cpu().numpy()
        print(f"  prob_map range: {prob_256.min():.4f}–{prob_256.max():.4f}  "
              f"mean={prob_256.mean():.4f}")
        seg_256     = prob_256 > threshold

    prob_full = np.array(
        Image.fromarray((prob_256 * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    ).astype(np.float32) / 255.0
    mask_full = np.array(
        Image.fromarray(seg_256.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
    ) > 127

    return mask_full, prob_full


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _score_mask(pred_mask, gt_changed):
    """Compute precision, recall, F1, IoU of pred_mask vs ground-truth changed mask."""
    tp = int(np.logical_and(pred_mask,  gt_changed).sum())
    fp = int(np.logical_and(pred_mask,  ~gt_changed).sum())
    fn = int(np.logical_and(~pred_mask, gt_changed).sum())
    tn = int(np.logical_and(~pred_mask, ~gt_changed).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return dict(precision=precision, recall=recall, f1=f1, iou=iou,
                tp=tp, fp=fp, fn=fn, tn=tn)


def _refine(mask, min_area=500, dilate_iter=2):
    mask = mask.copy()
    labeled, n = ndimage.label(mask)
    for i in range(1, n + 1):
        if np.sum(labeled == i) < min_area:
            mask[labeled == i] = False
    if dilate_iter > 0:
        mask = ndimage.binary_dilation(mask, iterations=dilate_iter)
    return ndimage.binary_fill_holes(mask)


def _save_method_png(original_img, edited_img, diff_map, mask, label, out_dir, dataset, vmax=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(original_img); axes[0, 0].set_title("Original"); axes[0, 0].axis("off")
    axes[0, 1].imshow(edited_img);   axes[0, 1].set_title("Edited");   axes[0, 1].axis("off")

    im = axes[1, 0].imshow(diff_map, cmap="hot", vmin=0, vmax=vmax)
    axes[1, 0].set_title(f"{label} difference map"); axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    overlay = np.zeros((*mask.shape, 4))
    overlay[mask] = [1, 0, 0, 0.45]
    axes[1, 1].imshow(edited_img); axes[1, 1].imshow(overlay)
    axes[1, 1].set_title(f"{label} mask ({mask.mean()*100:.2f}% changed)"); axes[1, 1].axis("off")

    plt.suptitle(f"{label} — {dataset}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fname = f"{label.lower().replace(' ', '_').replace('+', '_')}_{dataset}.png"
    path  = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_summary_png(original_img, edited_img, results, out_dir, dataset, gt_changed=None):
    n     = len(results)
    ncols = max(3, n)
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 8))
    plt.subplots_adjust(hspace=0.15, wspace=0.05)

    # Row 0: Original, Edited, GT change mask
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title("Original"); axes[0, 0].axis("off")

    axes[0, 1].imshow(edited_img)
    axes[0, 1].set_title("Edited"); axes[0, 1].axis("off")

    if gt_changed is not None:
        overlay_gt = np.zeros((*gt_changed.shape, 4), dtype=np.float32)
        overlay_gt[gt_changed]  = [1, 0, 0, 0.6]
        overlay_gt[~gt_changed] = [0, 1, 0, 0.15]
        axes[0, 2].imshow(edited_img)
        axes[0, 2].imshow(overlay_gt)
        axes[0, 2].set_title(f"GT change mask\n({gt_changed.mean()*100:.2f}% changed)")
    else:
        axes[0, 2].text(0.5, 0.5, "GT mask\nnot available",
                        ha='center', va='center', transform=axes[0, 2].transAxes, fontsize=11)
    axes[0, 2].axis("off")

    for j in range(3, ncols):
        axes[0, j].axis("off")

    # Row 1: method masks (F1/IoU in title when GT scores are available)
    for i, (name, data) in enumerate(results.items()):
        mask = data["mask"]
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask] = [1, 0, 0, 0.5]
        axes[1, i].imshow(edited_img)
        axes[1, i].imshow(overlay)
        title = f"{name}\n({mask.mean()*100:.2f}%)"
        if gt_changed is not None and 'scores' in data:
            s = data['scores']
            title += f"\nF1={s['f1']:.2f}  IoU={s['iou']:.2f}"
        axes[1, i].set_title(title); axes[1, i].axis("off")

    for i in range(n, ncols):
        axes[1, i].axis("off")

    plt.suptitle(f"Summary — {dataset}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"summary_{dataset}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _clean_old_outputs(out_dir, dataset):
    # Only remove the summary PNG — individual model PNGs are overwritten in-place.
    summary = os.path.join(out_dir, f"summary_{dataset}.png")
    if os.path.exists(summary):
        os.remove(summary)
        print(f"Cleaned: summary_{dataset}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='Run change detection methods on a dataset pair.')
    p.add_argument('--dataset',             default=DEFAULT_DATASET,
                   choices=AVAILABLE_DATASETS)
    p.add_argument('--rgb-threshold',       type=int,   default=RGB_BASELINE['threshold'])
    p.add_argument('--dino-threshold',      type=float, default=DINO_BASELINE['threshold'])
    p.add_argument('--dino-sigma',          type=int,   default=DINO_BASELINE['sigma'])
    p.add_argument('--dino-min-area',       type=int,   default=DINO_BASELINE['min_area'])
    p.add_argument('--dino-dilate',         type=int,   default=DINO_BASELINE['dilate_iter'])
    p.add_argument('--gescf-threshold',     type=float, default=None,
                   help='GeSCF threshold (default: adaptive/skewness-based)')
    p.add_argument('--crossattn-threshold', type=float, default=CROSSATTN_BASELINE['threshold'])
    p.add_argument('--crossattn-model',     default=CROSSATTN_BASELINE['model'],
                   choices=['dino_2Cross_CMU', 'dino_2Cross_PSCD', 'dino_2Cross_DiffCMU'])
    p.add_argument('--dinov3-threshold', type=float, default=DINOV3_BASELINE['threshold'])
    p.add_argument('--dinov3-sigma',     type=int,   default=DINOV3_BASELINE['sigma'])
    p.add_argument('--dinov3-min-area',  type=int,   default=DINOV3_BASELINE['min_area'])
    p.add_argument('--dinov3-dilate',    type=int,   default=DINOV3_BASELINE['dilate_iter'])
    p.add_argument('--ogescf-points',     type=int,   default=OFFICIAL_GESCF_BASELINE['points_per_side'])
    p.add_argument('--ogescf-iou',        type=float, default=OFFICIAL_GESCF_BASELINE['pred_iou_thresh'])
    p.add_argument('--ogescf-stability',  type=float, default=OFFICIAL_GESCF_BASELINE['stability_score_thresh'])
    p.add_argument('--viewdelta-prompt',    default=VIEWDELTA_BASELINE['text_prompt'],
                   help='Text prompt for ViewDelta (e.g. "all changes", "object changes")')
    p.add_argument('--viewdelta-threshold', type=float, default=VIEWDELTA_BASELINE['threshold'],
                   help='Probability threshold for ViewDelta mask (default 0.15)')
    p.add_argument('--sam3-text-prompt', default=SAM3_BASELINE['text_prompt'],
                   help='Manual SAM 3 text prompt or comma-separated prompt list. '
                        'If unset, the configured VLM generates one automatically.')
    p.add_argument('--sam3-vlm-model', default=SAM3_BASELINE['vlm_model'],
                   help='SAM 3 Gemma 4 VLM for auto prompt generation '
                        '(e.g. google/gemma-4-E2B-it)')
    p.add_argument('--models', nargs='+',
                   choices=['rgb', 'dinov2', 'dinov3', 'gescf', 'ogescf',
                            'viewdelta', 'crossattn', 'sam2', 'sam3', 'dinov3_sam2', 'dinov2_sam2'],
                   default=['rgb', 'dinov2', 'dinov3', 'gescf', 'ogescf', 'viewdelta', 'crossattn'],
                   metavar='MODEL',
                   help='Models to run. Choices: rgb dinov2 dinov3 gescf ogescf viewdelta crossattn sam2 sam3 dinov3_sam2 dinov2_sam2. '
                        'SAM checkpoint paths are configured in params.py.')
    p.add_argument('--no-show',        action='store_true')
    p.add_argument('--masks-only',     action='store_true',
                   help='Skip all model inference — only visualise extra masks found in the dataset folder')
    # GT depth scoring (optional — enables per-method scoring vs ground truth)
    p.add_argument('--gt-depth-orig',    default=None,
                   help='Path to original GT depth EXR (for GT mask derivation)')
    p.add_argument('--gt-depth-edit',    default=None,
                   help='Path to edited GT depth EXR (for GT mask derivation)')
    p.add_argument('--change-threshold', type=float, default=None,
                   help='Depth diff threshold (m) to derive GT change mask '
                        '(default: per-dataset value from DATASET_CHANGE_THRESHOLDS in params.py)')
    args = p.parse_args()
    if args.change_threshold is None:
        args.change_threshold = DATASET_CHANGE_THRESHOLDS.get(args.dataset, DEFAULT_CHANGE_THRESHOLD)

    data_dir = os.path.join(PROJECT_ROOT, "data", args.dataset)
    out_dir  = os.path.join(OUTPUT_ROOT, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("CHANGE DETECTION MODEL COMPARISON")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")

    original_path, edited_path = find_image_pair(data_dir)
    if not original_path or not edited_path:
        raise FileNotFoundError(f"Could not find image pair in {data_dir}")
    print(f"Original: {os.path.basename(original_path)}")
    print(f"Edited:   {os.path.basename(edited_path)}")

    # Auto-detect GT depths if not provided, or fall back to auto-discovery
    gt_depth_orig_path = args.gt_depth_orig
    gt_depth_edit_path = args.gt_depth_edit
    if gt_depth_orig_path is None and gt_depth_edit_path is None:
        auto_orig, auto_edit = find_depth_pair(data_dir)
        if auto_orig and auto_edit:
            gt_depth_orig_path = auto_orig
            gt_depth_edit_path = auto_edit
            print(f"Auto-detected GT depths:")
            print(f"  orig: {os.path.basename(gt_depth_orig_path)}")
            print(f"  edit: {os.path.basename(gt_depth_edit_path)}")

    gt_scoring = (gt_depth_orig_path is not None and gt_depth_edit_path is not None)
    if gt_scoring:
        print(f"GT scoring enabled (threshold={args.change_threshold} m)")
    else:
        print("GT scoring disabled (no GT depth pair found)")

    extra_masks = find_extra_masks(data_dir)
    if extra_masks:
        print(f"Extra masks found: {', '.join(n for n, _ in extra_masks)}")

    original_img = load_image(original_path)
    edited_img   = load_image(edited_path)
    if edited_img.size != original_img.size:
        edited_img = edited_img.resize(original_img.size, Image.BILINEAR)
    print(f"Image size: {original_img.size}")

    # ── Load GT depths and derive GT change mask ─────────────────────────────
    gt_changed = None
    if gt_scoring:
        depth_gt_orig = load_exr_depth(gt_depth_orig_path)
        depth_gt_edit = load_exr_depth(gt_depth_edit_path)
        h, w = np.array(original_img).shape[:2]
        # Resize depth maps to image resolution if needed
        def _resize_depth(d, h, w):
            if d.shape == (h, w):
                return d
            from PIL import Image as _PIL
            return np.array(
                _PIL.fromarray(d.astype(np.float32)).resize((w, h), _PIL.BILINEAR)
            )
        depth_gt_orig = _resize_depth(depth_gt_orig, h, w)
        depth_gt_edit = _resize_depth(depth_gt_edit, h, w)
        gt_changed = np.abs(depth_gt_edit - depth_gt_orig) > args.change_threshold
        print(f"GT change mask: {gt_changed.sum():,} changed px "
              f"({gt_changed.mean()*100:.2f}%) at threshold {args.change_threshold} m")

        # Save GT mask visualisation
        depth_diff = np.abs(depth_gt_edit - depth_gt_orig)
        _save_method_png(original_img, edited_img, depth_diff, gt_changed,
                         "GT mask", out_dir, args.dataset,
                         vmax=np.percentile(depth_diff, 99))

    _clean_old_outputs(out_dir, args.dataset)
    results = {}
    run = set() if args.masks_only else set(args.models)
    print(f"Models: {', '.join(sorted(run)) or '(none — masks-only)'}")

    # ── RGB ──────────────────────────────────────────────────────────────────
    if 'rgb' in run:
        print("\n--- RGB threshold ---")
        rgb_mask, rgb_diff = rgb_threshold_mask(
            original_img, edited_img,
            threshold=args.rgb_threshold,
            min_area=RGB_BASELINE['min_area'],
            dilate_iter=RGB_BASELINE['dilate_iter'],
        )
        print(f"  threshold={args.rgb_threshold}  changed={rgb_mask.mean()*100:.2f}%")
        results['RGB'] = {'mask': rgb_mask, 'diff_map': rgb_diff}
        _save_method_png(original_img, edited_img, rgb_diff, rgb_mask, "RGB", out_dir, args.dataset, vmax=80)

    # ── DINOv2 ───────────────────────────────────────────────────────────────
    if 'dinov2' in run:
        print("\n--- DINOv2 ---")
        try:
            dino_mask, dino_diff = dinov2_feature_mask(
                original_img, edited_img,
                threshold=args.dino_threshold,
                sigma=args.dino_sigma,
                min_area=args.dino_min_area,
                dilate_iter=args.dino_dilate,
            )
            print(f"  threshold={args.dino_threshold}  sigma={args.dino_sigma}  changed={dino_mask.mean()*100:.2f}%")
            results['DINOv2'] = {'mask': dino_mask, 'diff_map': dino_diff}
            _save_method_png(original_img, edited_img, dino_diff, dino_mask,
                             "DINOv2", out_dir, args.dataset, vmax=0.5)
        except Exception as e:
            print(f"  ERROR: {e}")

    # ── DINOv3 ───────────────────────────────────────────────────────────────
    if 'dinov3' in run:
        print("\n--- DINOv3 ---")
        try:
            dv3_mask, dv3_diff = dinov2_feature_mask(
                original_img, edited_img,
                threshold=args.dinov3_threshold,
                sigma=args.dinov3_sigma,
                min_area=args.dinov3_min_area,
                dilate_iter=args.dinov3_dilate,
                model_name=DINOV3_BASELINE['model_name'],
            )
            print(f"  threshold={args.dinov3_threshold}  sigma={args.dinov3_sigma}  changed={dv3_mask.mean()*100:.2f}%")
            results['DINOv3'] = {'mask': dv3_mask, 'diff_map': dv3_diff}
            _save_method_png(original_img, edited_img, dv3_diff, dv3_mask,
                             "DINOv3", out_dir, args.dataset, vmax=0.5)
        except Exception as e:
            print(f"  ERROR: {e}")

    # ── GeSCF ────────────────────────────────────────────────────────────────
    if 'gescf' in run:
        print("\n--- GeSCF ---")
        try:
            gescf_mask, gescf_diff = gescf_feature_mask(
                original_img, edited_img,
                threshold=args.gescf_threshold,
                **GESCF_BASELINE,
            )
            print(f"  changed={gescf_mask.mean()*100:.2f}%")
            results['GeSCF'] = {'mask': gescf_mask, 'diff_map': gescf_diff}
            _save_method_png(original_img, edited_img, gescf_diff, gescf_mask,
                             "GeSCF", out_dir, args.dataset, vmax=1.0)
        except Exception as e:
            print(f"  SKIPPED: {e}")

    # ── Official GeSCF ───────────────────────────────────────────────────────
    if 'ogescf' in run:
        print("\n--- Official GeSCF (ViT-H, layer 17/32) ---")
        try:
            og_mask, og_diff = official_gescf_mask(
                original_path, edited_path,
                points_per_side=args.ogescf_points,
                pred_iou_thresh=args.ogescf_iou,
                stability_score_thresh=args.ogescf_stability,
            )
            print(f"  changed={og_mask.mean()*100:.2f}%")
            results['OfficialGeSCF'] = {'mask': og_mask, 'diff_map': og_diff}
            _save_method_png(original_img, edited_img, og_diff, og_mask,
                             "Official GeSCF", out_dir, args.dataset, vmax=1.0)
        except Exception as e:
            print(f"  SKIPPED: {e}")

    # ── ViewDelta ─────────────────────────────────────────────────────────────
    if 'viewdelta' in run:
        print(f"\n--- ViewDelta (prompt: '{args.viewdelta_prompt}') ---")
        try:
            vd_mask, vd_prob = viewdelta_mask(
                original_img, edited_img,
                text_prompt=args.viewdelta_prompt,
                threshold=args.viewdelta_threshold,
            )
            print(f"  changed={vd_mask.mean()*100:.2f}%")
            results['ViewDelta'] = {'mask': vd_mask, 'diff_map': vd_prob}
            _save_method_png(original_img, edited_img, vd_prob, vd_mask,
                             "ViewDelta", out_dir, args.dataset, vmax=1.0)
        except Exception as e:
            print(f"  SKIPPED: {e}")

    # ── CrossAttn ────────────────────────────────────────────────────────────
    if 'crossattn' in run:
        print("\n--- CrossAttn ---")
        try:
            ca_mask, ca_prob = dino_crossattn_mask(
                original_img, edited_img,
                threshold=args.crossattn_threshold,
                pretrained=args.crossattn_model,
            )
            print(f"  changed={ca_mask.mean()*100:.2f}%")
            results['CrossAttn'] = {'mask': ca_mask, 'diff_map': ca_prob}
            _save_method_png(original_img, edited_img, ca_prob, ca_mask,
                             "CrossAttn", out_dir, args.dataset, vmax=1.0)
        except Exception as e:
            print(f"  SKIPPED: {e}")

    # ── SAM 2 ────────────────────────────────────────────────────────────────────
    if 'sam2' in run:
        ckpt2 = os.path.join(PROJECT_ROOT, SAM2_BASELINE['checkpoint'])
        if not os.path.isfile(ckpt2):
            print(f"\n--- SAM 2 skipped (checkpoint not found: {SAM2_BASELINE['checkpoint']}) ---")
        else:
            print("\n--- SAM 2 (pixel diff → box prompt) ---")
            try:
                s2_mask, s2_diff = sam2_mask(
                    original_img, edited_img,
                    checkpoint=ckpt2,
                    model_cfg=SAM2_BASELINE['model_cfg'],
                    diff_thresh=SAM2_BASELINE['diff_thresh'],
                    dilate=SAM2_BASELINE['dilate'],
                )
                print(f"  changed={s2_mask.mean()*100:.2f}%")
                results['SAM2'] = {'mask': s2_mask, 'diff_map': s2_diff}
                _save_method_png(original_img, edited_img, s2_diff, s2_mask,
                                 "SAM2", out_dir, args.dataset, vmax=80)
            except Exception as e:
                print(f"  SKIPPED: {e}")

    # ── DINOv3 → SAM2 ────────────────────────────────────────────────────────────
    if 'dinov3_sam2' in run:
        ckpt_d3s2 = os.path.join(PROJECT_ROOT, DINOV3_SAM2_BASELINE['checkpoint'])
        if not os.path.isfile(ckpt_d3s2):
            print(f"\n--- DINOv3→SAM2 skipped (checkpoint not found: {DINOV3_SAM2_BASELINE['checkpoint']}) ---")
        else:
            print("\n--- DINOv3→SAM2 (feature diff → per-region box prompts) ---")
            try:
                d3s2_mask, d3s2_diff = dinov3_sam2_mask(
                    original_img, edited_img,
                    checkpoint=ckpt_d3s2,
                    model_cfg=DINOV3_SAM2_BASELINE['model_cfg'],
                    dino_threshold=DINOV3_SAM2_BASELINE['dino_threshold'],
                    sigma=DINOV3_SAM2_BASELINE['sigma'],
                    dilate=DINOV3_SAM2_BASELINE['dilate'],
                    model_name=DINOV3_SAM2_BASELINE['model_name'],
                )
                print(f"  changed={d3s2_mask.mean()*100:.2f}%")
                results['DINOv3→SAM2'] = {'mask': d3s2_mask, 'diff_map': d3s2_diff}
                _save_method_png(original_img, edited_img, d3s2_diff, d3s2_mask,
                                 "DINOv3_SAM2", out_dir, args.dataset, vmax=0.5)
            except Exception as e:
                print(f"  SKIPPED: {e}")

    # ── DINOv2 → SAM2 ────────────────────────────────────────────────────────────
    if 'dinov2_sam2' in run:
        ckpt_d2s2 = os.path.join(PROJECT_ROOT, DINOV2_SAM2_BASELINE['checkpoint'])
        if not os.path.isfile(ckpt_d2s2):
            print(f"\n--- DINOv2→SAM2 skipped (checkpoint not found: {DINOV2_SAM2_BASELINE['checkpoint']}) ---")
        else:
            print("\n--- DINOv2→SAM2 (feature diff → per-region box prompts) ---")
            try:
                d2s2_mask, d2s2_diff = dinov3_sam2_mask(
                    original_img, edited_img,
                    checkpoint=ckpt_d2s2,
                    model_cfg=DINOV2_SAM2_BASELINE['model_cfg'],
                    dino_threshold=DINOV2_SAM2_BASELINE['dino_threshold'],
                    sigma=DINOV2_SAM2_BASELINE['sigma'],
                    dilate=DINOV2_SAM2_BASELINE['dilate'],
                    model_name=DINOV2_SAM2_BASELINE['model_name'],
                )
                print(f"  changed={d2s2_mask.mean()*100:.2f}%")
                results['DINOv2→SAM2'] = {'mask': d2s2_mask, 'diff_map': d2s2_diff}
                _save_method_png(original_img, edited_img, d2s2_diff, d2s2_mask,
                                 "DINOv2_SAM2", out_dir, args.dataset, vmax=0.5)
            except Exception as e:
                print(f"  SKIPPED: {e}")

    # ── SAM 3.1 (VLM → text prompt → segmentation) ───────────────────────────────
    if 'sam3' in run:
        ckpt3 = os.path.join(PROJECT_ROOT, SAM3_BASELINE['checkpoint'])
        if not os.path.exists(ckpt3):
            print(f"\n--- SAM 3.1 skipped (weights not found: {SAM3_BASELINE['checkpoint']}) ---")
        else:
            print("\n--- SAM 3.1 (VLM → text prompt) ---")
            try:
                s3_mask, s3_diff = sam3_mask(
                    original_img, edited_img,
                    checkpoint=ckpt3,
                    text_prompt=args.sam3_text_prompt,
                    vlm_model=args.sam3_vlm_model,
                )
                print(f"  changed={s3_mask.mean()*100:.2f}%")
                results['SAM3'] = {'mask': s3_mask, 'diff_map': s3_diff}
                _save_method_png(original_img, edited_img, s3_diff, s3_mask,
                                 "SAM3", out_dir, args.dataset, vmax=1.0)
            except Exception as e:
                print(f"  SKIPPED: {e}")

    # ── GT mask scoring (before summary so scores appear in image titles) ────────
    detection_scores = None
    if gt_scoring and gt_changed is not None and results:
        detection_scores = {
            'change_threshold_m': args.change_threshold,
            'gt_changed_pixels':  int(gt_changed.sum()),
            'gt_changed_frac':    float(gt_changed.mean()),
            'methods': {},
        }
        for name, data in results.items():
            mask = data['mask']
            if mask.shape != gt_changed.shape:
                mask = np.array(
                    Image.fromarray(mask.astype(np.uint8) * 255).resize(
                        (gt_changed.shape[1], gt_changed.shape[0]), Image.NEAREST
                    )
                ) > 127
            scores = _score_mask(mask, gt_changed)
            detection_scores['methods'][name] = scores
            results[name]['scores'] = scores  # stored so summary can show F1/IoU in titles

    # ── Extra mask visualisation + scoring ───────────────────────────────────
    if extra_masks:
        h_img, w_img = np.array(original_img).shape[:2]
        for mask_name, mask_path in extra_masks:
            extra_arr = np.array(
                Image.open(mask_path).convert('L').resize((w_img, h_img), Image.NEAREST)
            ) > 127
            _save_method_png(original_img, edited_img, extra_arr.astype(np.float32),
                             extra_arr, mask_name, out_dir, args.dataset, vmax=1.0)
            print(f"Saved: {mask_name}_{args.dataset}.png  ({extra_arr.mean()*100:.2f}% changed)")

            if results:
                if detection_scores is None:
                    detection_scores = {}
                detection_scores.setdefault('extra_masks', {})
                print(f"\n{'='*65}")
                print(f"SCORING vs EXTRA MASK: {mask_name}")
                print(f"{'='*65}")
                print(f"{'Method':<14} {'Precision':>10} {'Recall':>10} {'F1':>10} {'IoU':>10}")
                print("-" * 65)
                mask_method_scores = {}
                for name, data in results.items():
                    pred = data['mask']
                    if pred.shape != extra_arr.shape:
                        pred = np.array(
                            Image.fromarray(pred.astype(np.uint8) * 255).resize(
                                (extra_arr.shape[1], extra_arr.shape[0]), Image.NEAREST
                            )
                        ) > 127
                    s = _score_mask(pred, extra_arr)
                    mask_method_scores[name] = s
                    print(f"{name:<14} {s['precision']:>10.3f} {s['recall']:>10.3f} "
                          f"{s['f1']:>10.3f} {s['iou']:>10.3f}")
                print("=" * 65)
                detection_scores['extra_masks'][mask_name] = {
                    'changed_pixels': int(extra_arr.sum()),
                    'changed_frac':   float(extra_arr.mean()),
                    'methods':        mask_method_scores,
                }

    if results:
        _save_summary_png(original_img, edited_img, results, out_dir, args.dataset,
                          gt_changed=gt_changed)

    # ── Print scores + save JSON ──────────────────────────────────────────────
    if detection_scores is not None:
        if 'methods' in detection_scores and detection_scores['methods']:
            print("\n" + "=" * 65)
            print("SCORING vs GT CHANGE MASK")
            print("=" * 65)
            print(f"{'Method':<14} {'Precision':>10} {'Recall':>10} {'F1':>10} {'IoU':>10}")
            print("-" * 65)
            for name in results:
                s = detection_scores['methods'][name]
                print(f"{name:<14} {s['precision']:>10.3f} {s['recall']:>10.3f} "
                      f"{s['f1']:>10.3f} {s['iou']:>10.3f}")
            print("=" * 65)

        scores_path = os.path.join(out_dir, "detection_scores.json")
        with open(scores_path, 'w') as f:
            json.dump(detection_scores, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
