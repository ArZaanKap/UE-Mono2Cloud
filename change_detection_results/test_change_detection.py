"""
Change Detection Model Comparison
=================================
Tests different approaches for detecting changes between original and edited images:
1. RGB Threshold - Simple pixel difference baseline
2. DINOv2 - Semantic feature-based difference (518px, multi-layer, registers)
3. GeSCF-style - SAM Q/K/V attention features + adaptive thresholding (CVPR 2025-inspired)
4. DINOv2+CrossAttn - Pretrained scene change detection (ICRA 2025)

Each method produces its own {method}_{dataset}.png with 2x2 layout, plus a
summary_{dataset}.png comparing all masks side by side.
"""

import os
import sys
import glob
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy import ndimage

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Parent directory (UE_depth)
DEFAULT_DATASET = "depth4"


def load_exr_rgb(exr_path):
    """Load RGB channels from EXR file."""
    import OpenEXR
    import Imath
    exr_file = OpenEXR.InputFile(exr_path)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    rgb = []
    for c in ['R', 'G', 'B']:
        channel_str = exr_file.channel(c, FLOAT)
        channel = np.frombuffer(channel_str, dtype=np.float32).reshape(height, width)
        rgb.append(channel)

    img = np.stack(rgb, axis=-1)
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


def load_image(path):
    """Load image from EXR or standard format."""
    if path.lower().endswith('.exr'):
        return load_exr_rgb(path)
    else:
        return Image.open(path).convert('RGB')


def find_image_pair(folder):
    """Find original and edited images in folder."""
    files = os.listdir(folder)

    # Find original RGB: EXR without 'depth' or 'scenedepth' in name
    original = None
    for f in files:
        if f.endswith('.exr') and 'depth' not in f.lower() and 'scenedepth' not in f.lower():
            original = os.path.join(folder, f)
            break

    # Find edited image: look for 'edit' in name
    edited = None
    for f in files:
        if 'edit' in f.lower() and (f.endswith('.png') or f.endswith('.jpg') or f.endswith('.exr')):
            edited = os.path.join(folder, f)
            break

    return original, edited


# ---------------------------------------------------------------------------
# Change detection methods
# ---------------------------------------------------------------------------

def rgb_threshold_mask(img1, img2, threshold=10):
    """
    Simple RGB threshold-based change detection.
    Returns mask where True = changed pixel.
    """
    arr1 = np.array(img1).astype(float)
    arr2 = np.array(img2).astype(float)

    # Mean absolute difference across RGB channels
    diff = np.abs(arr1 - arr2).mean(axis=2)
    changed_mask = diff > threshold

    return changed_mask, diff


def dinov2_feature_mask(img1, img2, threshold=0.3, model_name="facebook/dinov2-with-registers-base"):
    """
    DINOv2 feature-based change detection.
    Uses 518x518 input (37x37 patches), multi-layer feature concatenation
    from layers {3,6,9,12}, and Gaussian smoothing for clean spatial maps.
    Returns mask where True = changed region.
    """
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading DINOv2 model ({model_name})...")

    # 518x518 gives 37x37 patches (5x more spatial detail than 224x224)
    processor = AutoImageProcessor.from_pretrained(model_name, size={"height": 518, "width": 518}, crop_size={"height": 518, "width": 518})
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Number of register tokens to skip (reg4 models have 4)
    num_register = 4 if "reg" in model_name else 0

    # Process images
    with torch.no_grad():
        inputs1 = processor(images=img1, return_tensors="pt").to(device)
        inputs2 = processor(images=img2, return_tensors="pt").to(device)

        # Get multi-layer features (layers 3, 6, 9, 12 for ViT-B)
        outputs1 = model(**inputs1, output_hidden_states=True)
        outputs2 = model(**inputs2, output_hidden_states=True)

        # Skip CLS token (idx 0) and register tokens (idx 1..num_register)
        skip = 1 + num_register
        selected1 = [outputs1.hidden_states[i][:, skip:, :] for i in [3, 6, 9, 12]]
        selected2 = [outputs2.hidden_states[i][:, skip:, :] for i in [3, 6, 9, 12]]
        features1 = torch.cat(selected1, dim=-1)  # [1, num_patches, 4*hidden_dim]
        features2 = torch.cat(selected2, dim=-1)

        # Compute cosine similarity per patch
        features1_norm = features1 / features1.norm(dim=-1, keepdim=True)
        features2_norm = features2 / features2.norm(dim=-1, keepdim=True)
        similarity = (features1_norm * features2_norm).sum(dim=-1)  # [1, num_patches]

        # Convert to distance (1 - similarity)
        distance = 1 - similarity.squeeze()  # [num_patches] (keep as tensor)

    # Reshape to 2D patch grid (37x37 for 518x518 input)
    num_patches = distance.shape[0]
    grid_size = int(np.sqrt(num_patches))
    distance_map = distance.reshape(1, 1, grid_size, grid_size)

    # Upsample to original image size using F.interpolate
    h, w = np.array(img1).shape[:2]
    distance_map_full = F.interpolate(distance_map, size=(h, w), mode='bilinear', align_corners=False)
    distance_map_full = distance_map_full.squeeze().cpu().numpy()

    # Gaussian smoothing to remove blocky patch artifacts (sigma=4, per AnomalyDINO)
    distance_map_full = ndimage.gaussian_filter(distance_map_full, sigma=4)

    # Threshold to get binary mask
    changed_mask = distance_map_full > threshold

    return changed_mask, distance_map_full


def gescf_feature_mask(img1, img2, threshold=None):
    """
    GeSCF-style change detection (CVPR 2025-inspired).

    Uses SAM ViT encoder's internal Q/K/V attention features (not just the
    final embedding) to compute cross-image cosine similarity, then applies
    adaptive skewness-based thresholding from the GeSCF paper.

    Requires: pip install segment-anything
    Weights: sam_vit_b_01ec64.pth (auto-downloaded if missing)

    Key differences from naive SAM feature comparison:
      - Uses Q/K/V from intermediate attention layers (richer than final output)
      - Adaptive threshold based on distance-map skewness (not a fixed value)
      - SAM automatic mask generator filters the result into clean segments
    """
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        raise ImportError(
            "segment-anything not installed. Run: pip install segment-anything"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- locate / download weights ---
    weights_dir = os.path.join(PROJECT_ROOT, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, "sam_vit_b_01ec64.pth")

    if not os.path.exists(weights_path):
        print("  Downloading SAM ViT-B weights (~375 MB) ...")
        import urllib.request
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        urllib.request.urlretrieve(url, weights_path)
        print("  Download complete.")

    print(f"  Loading SAM ViT-B (segment-anything) on {device} ...")
    sam = sam_model_registry["vit_b"](checkpoint=weights_path)
    sam.to(device)
    sam.eval()

    h, w = np.array(img1).shape[:2]

    # --- hook to capture Q/K/V from an intermediate encoder block ---
    # SAM ViT-B uses global attention at indices [2, 5, 8, 11]; other blocks
    # use windowed attention (which partitions tokens and breaks our reshape).
    # We use block 8: a global-attention block near the end of the encoder.
    captured = {}

    def _hook_qkv(module, input, output):
        """Capture the linear projection output before reshape into Q/K/V."""
        captured["qkv"] = output.detach()

    target_block = sam.image_encoder.blocks[8]
    hook_handle = target_block.attn.qkv.register_forward_hook(_hook_qkv)

    # --- preprocess both images ---
    from segment_anything.utils.transforms import ResizeLongestSide
    img_size = sam.image_encoder.img_size
    transform = ResizeLongestSide(img_size)
    patch_size = 16  # SAM ViT patch size

    def _prepare(pil_img):
        img_np = np.array(pil_img)
        img_t = transform.apply_image(img_np)
        img_t = torch.as_tensor(img_t, device=device).permute(2, 0, 1).unsqueeze(0).float()
        img_t = sam.preprocess(img_t)
        return img_t

    # Compute the resized (pre-padding) dimensions so we can crop the feature map
    scale = img_size / max(h, w)
    resized_h = int(h * scale + 0.5)
    resized_w = int(w * scale + 0.5)
    feat_h = resized_h // patch_size  # rows of real content in 64x64 feature map
    feat_w = resized_w // patch_size
    print(f"  Image {w}x{h} -> SAM {resized_w}x{resized_h} -> feat {feat_w}x{feat_h} (of {img_size // patch_size}x{img_size // patch_size})")

    img1_t = _prepare(img1)
    img2_t = _prepare(img2)

    # --- forward pass to capture Q/K/V features ---
    with torch.no_grad():
        sam.image_encoder(img1_t)
        qkv1 = captured["qkv"]  # [B, H_feat, W_feat, D] (spatial dims kept)

        sam.image_encoder(img2_t)
        qkv2 = captured["qkv"]

    hook_handle.remove()

    # --- compute per-token cosine distance ---
    with torch.no_grad():
        f1 = qkv1.squeeze(0)  # [H_feat_full, W_feat_full, D]
        f2 = qkv2.squeeze(0)

        f1_norm = f1 / (f1.norm(dim=-1, keepdim=True) + 1e-8)
        f2_norm = f2 / (f2.norm(dim=-1, keepdim=True) + 1e-8)
        cosine_sim = (f1_norm * f2_norm).sum(dim=-1)  # [H_feat_full, W_feat_full]
        dist_map_full_feat = (1 - cosine_sim).cpu().numpy()

    # Crop to actual content region (remove zero-padding area)
    dist_map = dist_map_full_feat[:feat_h, :feat_w]

    # Upsample cropped feature map to original image size
    dist_map_tensor = torch.tensor(dist_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    dist_map_full = F.interpolate(dist_map_tensor, size=(h, w), mode='bilinear', align_corners=False)
    dist_map_full = dist_map_full.squeeze().numpy()

    # Gaussian smoothing
    dist_map_full = ndimage.gaussian_filter(dist_map_full, sigma=4)

    # Normalise to [0, 1]
    dmin, dmax = dist_map_full.min(), dist_map_full.max()
    if dmax - dmin > 1e-8:
        dist_map_full = (dist_map_full - dmin) / (dmax - dmin)

    # --- adaptive threshold (skewness-based, from GeSCF) ---
    if threshold is None:
        from scipy.stats import skew
        sk = skew(dist_map_full.ravel())
        # GeSCF: if distribution is right-skewed (most pixels similar, few changed),
        # use mean + k*std where k scales with skewness
        mu = dist_map_full.mean()
        sigma = dist_map_full.std()
        k = min(max(sk, 1.0), 3.0)  # clamp skewness factor
        threshold = mu + k * sigma
        print(f"  Adaptive threshold: {threshold:.4f}  (skew={sk:.2f}, k={k:.2f})")
    else:
        print(f"  Fixed threshold: {threshold:.4f}")

    initial_mask = dist_map_full > threshold

    # --- SAM automatic mask generator for segment-level refinement ---
    print("  Running SAM automatic mask generator for refinement ...")
    mask_gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,      # coarser grid for speed on CPU
        pred_iou_thresh=0.80,
        stability_score_thresh=0.85,
        min_mask_region_area=200,
    )
    img2_np = np.array(img2)
    sam_masks = mask_gen.generate(img2_np)
    print(f"  SAM generated {len(sam_masks)} candidate masks")

    # Keep SAM segments that overlap significantly with the initial change mask
    refined_mask = np.zeros((h, w), dtype=bool)
    for seg in sam_masks:
        seg_mask = seg["segmentation"]
        overlap = np.logical_and(seg_mask, initial_mask).sum()
        seg_area = seg_mask.sum()
        if seg_area > 0 and overlap / seg_area > 0.3:
            refined_mask |= seg_mask

    # Fall back to initial mask if SAM refinement removed everything
    if refined_mask.sum() == 0:
        print("  SAM refinement produced empty mask, using initial threshold mask")
        refined_mask = initial_mask

    return refined_mask, dist_map_full


def dino_crossattn_mask(img1, img2, threshold=0.3, pretrained="dino_2Cross_CMU"):
    """
    DINOv2 + CrossAttention pretrained scene change detection (ICRA 2025).

    Uses the Robust-Scene-Change-Detection repo which provides a DINOv2-small
    backbone with cross-attention heads, pretrained for binary scene change
    detection.

    Available pretrained models:
      - dino_2Cross_CMU    (outdoor, CMU Seasons)
      - dino_2Cross_PSCD   (indoor/outdoor, PSCD dataset)
      - dino_2Cross_DiffCMU

    Args:
        threshold: Probability threshold for "changed" class (default 0.3).
                   Lower = more sensitive. The model's default argmax uses 0.5.

    Returns binary mask (True = changed) and a soft probability map.
    """
    try:
        from robust_scene_change_detect.models import get_model_from_pretrained
    except ImportError as e:
        raise ImportError(
            "robust-scene-change-detection not installed.\n"
            "Install it:\n"
            f"  cd {PROJECT_ROOT}\n"
            "  git clone https://github.com/ChadLin9596/Robust-Scene-Change-Detection\n"
            "  cd Robust-Scene-Change-Detection\n"
            "  git submodule update --init --recursive\n"
            "  pip install -e thirdparties/py_utils && pip install --no-deps -e ."
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading DINOv2+CrossAttn ({pretrained}) on {device} ...")
    model = get_model_from_pretrained(pretrained)
    # Unwrap DataParallel for CPU compatibility
    if hasattr(model, 'module'):
        model = model.module
    model = model.to(device)
    model.eval()

    h, w = np.array(img1).shape[:2]

    # Preprocess: resize to 504x504 (must be 14*n for ViT patch size)
    target_size = 504
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    t0 = preprocess(img1).unsqueeze(0).to(device)  # [1, 3, 504, 504]
    t1 = preprocess(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(t0, t1)  # logits [1, H', W', 2]

    # The model outputs per-pixel class logits [B, H', W', 2]
    if pred.dim() == 4 and pred.shape[-1] == 2:
        prob_map = torch.softmax(pred, dim=-1)[0, :, :, 1].cpu().numpy()
    elif pred.dim() == 4 and pred.shape[1] == 2:
        prob_map = torch.softmax(pred, dim=1)[0, 1].cpu().numpy()
    else:
        prob_map = torch.sigmoid(pred).squeeze().cpu().numpy()

    # Apply threshold on probability (not argmax which forces 0.5)
    pred_mask = prob_map > threshold
    print(f"  Threshold: {threshold}  (prob range: {prob_map.min():.3f} - {prob_map.max():.3f})")

    # Resize mask and probability map back to original image dimensions
    pred_mask_pil = Image.fromarray(pred_mask.astype(np.uint8) * 255)
    pred_mask_full = np.array(pred_mask_pil.resize((w, h), Image.NEAREST)) > 127

    prob_pil = Image.fromarray((prob_map * 255).astype(np.uint8))
    prob_map_full = np.array(prob_pil.resize((w, h), Image.BILINEAR)).astype(np.float32) / 255.0

    return pred_mask_full, prob_map_full


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def refine_mask(mask, min_area=500, dilate_iter=2):
    """
    Refine binary mask by removing small regions and smoothing.
    """
    mask = mask.copy()
    # Remove small connected components
    labeled, num_features = ndimage.label(mask)
    for i in range(1, num_features + 1):
        if np.sum(labeled == i) < min_area:
            mask[labeled == i] = False

    # Dilate to fill gaps
    mask = ndimage.binary_dilation(mask, iterations=dilate_iter)

    # Fill holes
    mask = ndimage.binary_fill_holes(mask)

    return mask


def compute_mask_metrics(mask1, mask2):
    """Compute IoU and other metrics between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union if union > 0 else 0

    # Dice coefficient
    dice = 2 * intersection / (mask1.sum() + mask2.sum()) if (mask1.sum() + mask2.sum()) > 0 else 0

    return {'iou': iou, 'dice': dice, 'pixels_mask1': mask1.sum(), 'pixels_mask2': mask2.sum()}


def clean_old_outputs(output_folder, dataset):
    """Remove old per-method and summary PNGs from previous runs."""
    patterns = [
        os.path.join(output_folder, f"*_{dataset}.png"),
        os.path.join(output_folder, f"change_detection_{dataset}.png"),
        os.path.join(output_folder, f"sweep_{dataset}.png"),
    ]
    removed = []
    for pat in patterns:
        for f in glob.glob(pat):
            os.remove(f)
            removed.append(os.path.basename(f))
    if removed:
        print(f"Cleaned old outputs: {', '.join(removed)}")


def save_method_png(original_img, edited_img, diff_map, mask, method_label,
                    output_folder, dataset, vmax=None):
    """
    Save a 2x2 per-method visualization PNG.
      Top-left:     Original image
      Top-right:    Edited image
      Bottom-left:  Difference / distance heatmap
      Bottom-right: Binary mask overlaid on edited image
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(edited_img)
    axes[0, 1].set_title("Edited")
    axes[0, 1].axis("off")

    im = axes[1, 0].imshow(diff_map, cmap="hot", vmin=0, vmax=vmax)
    axes[1, 0].set_title(f"{method_label} Difference Map")
    axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Overlay mask on edited image
    overlay = np.array(edited_img).copy()
    overlay_rgba = np.zeros((*mask.shape, 4))
    overlay_rgba[mask] = [1, 0, 0, 0.45]
    axes[1, 1].imshow(edited_img)
    axes[1, 1].imshow(overlay_rgba)
    pct = mask.mean() * 100
    axes[1, 1].set_title(f"{method_label} Mask ({pct:.1f}% changed)")
    axes[1, 1].axis("off")

    plt.suptitle(f"{method_label} — {dataset}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    fname = f"{method_label.lower().replace(' ', '_').replace('+', '_')}_{dataset}.png"
    path = os.path.join(output_folder, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def save_summary_png(original_img, edited_img, results, output_folder, dataset):
    """
    Save a summary comparison PNG with all masks shown side by side in one row.
    """
    method_names = list(results.keys())
    n = len(method_names)
    # Columns: original, edited, then one column per method mask
    ncols = 2 + n
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

    axes[0].imshow(original_img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(edited_img)
    axes[1].set_title("Edited")
    axes[1].axis("off")

    for i, name in enumerate(method_names):
        ax = axes[2 + i]
        data = results[name]
        mask = data["mask"]
        overlay_rgba = np.zeros((*mask.shape, 4))
        overlay_rgba[mask] = [1, 0, 0, 0.5]
        ax.imshow(edited_img)
        ax.imshow(overlay_rgba)
        pct = mask.mean() * 100
        ax.set_title(f"{name}\n({pct:.1f}%)")
        ax.axis("off")

    plt.suptitle(f"Summary — {dataset}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_folder, f"summary_{dataset}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Sweep (kept for backwards compatibility)
# ---------------------------------------------------------------------------

def sweep_thresholds(original_img, edited_img, output_folder, dataset, no_show=False):
    """
    Grid search over RGB and DINOv2 thresholds to find best mask parameters.
    Computes feature maps once, then varies thresholds cheaply.
    Reports best single method and best combo by IoU when reference mask exists.
    """
    from transformers import AutoImageProcessor, AutoModel

    h, w = np.array(original_img).shape[:2]

    # --- Compute all diff maps once ---

    # RGB
    arr1 = np.array(original_img).astype(float)
    arr2 = np.array(edited_img).astype(float)
    rgb_diff = np.abs(arr1 - arr2).mean(axis=2)
    print("Computed RGB diff map")

    # DINOv2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino_model_name = "facebook/dinov2-with-registers-base"
    print(f"Loading DINOv2 model ({dino_model_name})...")
    processor = AutoImageProcessor.from_pretrained(dino_model_name, size={"height": 518, "width": 518}, crop_size={"height": 518, "width": 518})
    model = AutoModel.from_pretrained(dino_model_name).to(device)
    model.eval()
    num_register = 4 if "reg" in dino_model_name else 0

    with torch.no_grad():
        inputs1 = processor(images=original_img, return_tensors="pt").to(device)
        inputs2 = processor(images=edited_img, return_tensors="pt").to(device)
        outputs1 = model(**inputs1, output_hidden_states=True)
        outputs2 = model(**inputs2, output_hidden_states=True)
        skip = 1 + num_register
        selected1 = [outputs1.hidden_states[i][:, skip:, :] for i in [3, 6, 9, 12]]
        selected2 = [outputs2.hidden_states[i][:, skip:, :] for i in [3, 6, 9, 12]]
        features1 = torch.cat(selected1, dim=-1)
        features2 = torch.cat(selected2, dim=-1)
        f1_norm = features1 / features1.norm(dim=-1, keepdim=True)
        f2_norm = features2 / features2.norm(dim=-1, keepdim=True)
        distance = 1 - (f1_norm * f2_norm).sum(dim=-1).squeeze()

    num_patches = distance.shape[0]
    grid_size = int(np.sqrt(num_patches))
    distance_map = distance.reshape(1, 1, grid_size, grid_size)
    dino_diff = F.interpolate(distance_map, size=(h, w), mode='bilinear', align_corners=False)
    dino_diff = dino_diff.squeeze().cpu().numpy()
    dino_diff = ndimage.gaussian_filter(dino_diff, sigma=4)
    del model, processor
    torch.cuda.empty_cache()
    print("Computed DINOv2 diff map")

    # --- Threshold ranges per method ---
    methods = {
        'RGB':   {'diff': rgb_diff,   'thresholds': [10, 15, 20, 25, 30, 40], 'min_area': 100},
        'DINOv2':{'diff': dino_diff,  'thresholds': [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5], 'min_area': 500},
    }

    # Load reference mask if available
    # --- Single-method sweep ---
    print("\n" + "=" * 60)
    print("SINGLE METHOD SWEEP")
    print("=" * 60)

    best_single = []
    for name, cfg in methods.items():
        print(f"\n--- {name} ---")
        print(f"{'Thresh':>8} {'Pct%':>6}")
        for t in cfg['thresholds']:
            mask = refine_mask((cfg['diff'] > t).copy(), min_area=cfg['min_area'])
            pct = mask.mean() * 100
            print(f"{t:>8} {pct:>6.1f}")
            best_single.append({'method': name, 'thresh': t, 'pct': pct, 'mask': mask})

    # --- Combo sweep (pairwise intersections) ---
    print("\n" + "=" * 60)
    print("COMBO SWEEP (pairwise intersections)")
    print("=" * 60)

    method_names = list(methods.keys())
    best_combo = []
    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            n1, n2 = method_names[i], method_names[j]
            c1, c2 = methods[n1], methods[n2]
            combo_name = f"{n1} & {n2}"
            print(f"\n--- {combo_name} ---")
            print(f"{n1:>8} {n2:>8} {'Pct%':>6}")
            for t1 in c1['thresholds']:
                m1 = refine_mask((c1['diff'] > t1).copy(), min_area=c1['min_area'])
                for t2 in c2['thresholds']:
                    m2 = refine_mask((c2['diff'] > t2).copy(), min_area=c2['min_area'])
                    inter_mask = m1 & m2
                    pct = inter_mask.mean() * 100
                    print(f"{t1:>8} {t2:>8} {pct:>6.1f}")
                    best_combo.append({'combo': combo_name, 't1': t1, 't2': t2,
                                       'pct': pct, 'mask': inter_mask})

    # --- Visualization: RGB x DINOv2 grid ---
    rgb_thresholds = methods['RGB']['thresholds']
    dino_thresholds = methods['DINOv2']['thresholds']
    n_rows = len(rgb_thresholds)
    n_cols = len(dino_thresholds)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))

    for i, rt in enumerate(rgb_thresholds):
        rgb_m = refine_mask((rgb_diff > rt).copy(), min_area=100)
        for j, dt in enumerate(dino_thresholds):
            dino_m = refine_mask((dino_diff > dt).copy(), min_area=500)
            inter_mask = rgb_m & dino_m
            inter_pct = inter_mask.mean() * 100

            ax = axes[i, j]
            ax.imshow(edited_img)
            overlay = np.zeros((*inter_mask.shape, 4))
            overlay[inter_mask] = [1, 0, 0, 0.5]
            ax.imshow(overlay)
            ax.axis('off')

            label = f"{inter_pct:.1f}%"
            ax.set_title(label, fontsize=8)

            if i == 0:
                ax.set_title(f"D={dt:.2f}\n{label}", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"RGB={rt}", fontsize=9)

    plt.suptitle(f'Mask Sweep: RGB threshold (rows) x DINOv2 threshold (cols)\n{dataset}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    sweep_path = os.path.join(output_folder, f"sweep_{dataset}.png")
    plt.savefig(sweep_path, dpi=150, bbox_inches='tight')
    print(f"\nSweep visualization saved: {sweep_path}")

    if not no_show:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Test change detection models')
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET,
                        help='Dataset folder name (e.g., depth4)')
    parser.add_argument('--rgb-threshold', type=int, default=25,
                        help='RGB difference threshold (0-255)')
    parser.add_argument('--dino-threshold', type=float, default=0.35,
                        help='DINOv2 feature distance threshold (0-1)')
    parser.add_argument('--gescf-threshold', type=float, default=None,
                        help='GeSCF threshold (default: adaptive/skewness-based)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display figures')
    parser.add_argument('--skip-dino', action='store_true',
                        help='Skip DINOv2 method')
    parser.add_argument('--skip-gescf', action='store_true',
                        help='Skip GeSCF-style SAM method')
    parser.add_argument('--skip-crossattn', action='store_true',
                        help='Skip DINOv2+CrossAttn method')
    parser.add_argument('--crossattn-threshold', type=float, default=0.5,
                        help='CrossAttn probability threshold (default 0.5, lower=more sensitive)')
    parser.add_argument('--crossattn-model', type=str, default='dino_2Cross_PSCD',
                        choices=['dino_2Cross_CMU', 'dino_2Cross_PSCD', 'dino_2Cross_DiffCMU'],
                        help='CrossAttn pretrained model variant')
    parser.add_argument('--sweep', action='store_true',
                        help='Run threshold sweep over RGB + DINOv2')
    args = parser.parse_args()

    # Setup paths
    data_folder = os.path.join(PROJECT_ROOT, "data", args.dataset)
    output_folder = os.path.join(SCRIPT_DIR, args.dataset)
    os.makedirs(output_folder, exist_ok=True)

    print("=" * 60)
    print("CHANGE DETECTION MODEL COMPARISON")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")

    # Find images
    original_path, edited_path = find_image_pair(data_folder)
    if not original_path or not edited_path:
        raise FileNotFoundError(f"Could not find image pair in {data_folder}")

    print(f"Original: {os.path.basename(original_path)}")
    print(f"Edited:   {os.path.basename(edited_path)}")

    # Load images
    original_img = load_image(original_path)
    edited_img = load_image(edited_path)

    # Resize edited to match original if needed
    if edited_img.size != original_img.size:
        edited_img = edited_img.resize(original_img.size, Image.BILINEAR)

    print(f"Image size: {original_img.size}")

    if args.sweep:
        sweep_thresholds(original_img, edited_img, output_folder, args.dataset, args.no_show)
        print("\nDone!")
        return

    # Clean old outputs
    clean_old_outputs(output_folder, args.dataset)

    results = {}

    # -----------------------------------------------------------------------
    # Method 1: RGB Threshold (baseline)
    # -----------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Method 1: RGB Threshold Baseline")
    print("-" * 40)
    rgb_mask, rgb_diff = rgb_threshold_mask(original_img, edited_img, threshold=args.rgb_threshold)
    rgb_mask_refined = refine_mask(rgb_mask.copy(), min_area=100)
    print(f"  Threshold: {args.rgb_threshold}")
    print(f"  Changed pixels: {rgb_mask.sum():,} ({rgb_mask.mean()*100:.1f}%)")
    print(f"  After refinement: {rgb_mask_refined.sum():,} ({rgb_mask_refined.mean()*100:.1f}%)")
    results['RGB'] = {'mask': rgb_mask_refined, 'diff_map': rgb_diff}
    np.save(os.path.join(output_folder, f"rgb_{args.dataset}_mask.npy"), rgb_mask_refined)
    save_method_png(original_img, edited_img, rgb_diff, rgb_mask_refined,
                    "RGB", output_folder, args.dataset, vmax=80)

    # -----------------------------------------------------------------------
    # Method 2: DINOv2 Features
    # -----------------------------------------------------------------------
    if not args.skip_dino:
        print("\n" + "-" * 40)
        print("Method 2: DINOv2 Semantic Features")
        print("-" * 40)
        try:
            dino_mask, dino_diff = dinov2_feature_mask(
                original_img, edited_img,
                threshold=args.dino_threshold,
                model_name="facebook/dinov2-with-registers-base"
            )
            dino_mask_refined = refine_mask(dino_mask.copy(), min_area=500)
            print(f"  Threshold: {args.dino_threshold}")
            print(f"  Changed pixels: {dino_mask.sum():,} ({dino_mask.mean()*100:.1f}%)")
            print(f"  After refinement: {dino_mask_refined.sum():,} ({dino_mask_refined.mean()*100:.1f}%)")
            results['DINOv2'] = {'mask': dino_mask_refined, 'diff_map': dino_diff}
            np.save(os.path.join(output_folder, f"dinov2_{args.dataset}_mask.npy"), dino_mask_refined)
            save_method_png(original_img, edited_img, dino_diff, dino_mask_refined,
                            "DINOv2", output_folder, args.dataset, vmax=0.5)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # -----------------------------------------------------------------------
    # Method 3: GeSCF-style (SAM Q/K/V attention features)
    # -----------------------------------------------------------------------
    if not args.skip_gescf:
        print("\n" + "-" * 40)
        print("Method 3: GeSCF-style (SAM Q/K/V features)")
        print("-" * 40)
        try:
            gescf_mask, gescf_diff = gescf_feature_mask(
                original_img, edited_img,
                threshold=args.gescf_threshold,
            )
            # No extra refine_mask — gescf already does SAM-based refinement
            print(f"  Changed pixels: {gescf_mask.sum():,} ({gescf_mask.mean()*100:.1f}%)")
            results['GeSCF'] = {'mask': gescf_mask, 'diff_map': gescf_diff}
            np.save(os.path.join(output_folder, f"gescf_{args.dataset}_mask.npy"), gescf_mask)
            save_method_png(original_img, edited_img, gescf_diff, gescf_mask,
                            "GeSCF", output_folder, args.dataset, vmax=1.0)
        except Exception as e:
            print(f"  SKIPPED: {e}")
            import traceback
            traceback.print_exc()

    # -----------------------------------------------------------------------
    # Method 4: DINOv2+CrossAttention (pretrained scene change detection)
    # -----------------------------------------------------------------------
    if not args.skip_crossattn:
        print("\n" + "-" * 40)
        print("Method 4: DINOv2+CrossAttn (ICRA 2025)")
        print("-" * 40)
        try:
            crossattn_mask, crossattn_prob = dino_crossattn_mask(
                original_img, edited_img,
                threshold=args.crossattn_threshold,
                pretrained=args.crossattn_model,
            )
            print(f"  Changed pixels: {crossattn_mask.sum():,} ({crossattn_mask.mean()*100:.1f}%)")
            results['CrossAttn'] = {'mask': crossattn_mask, 'diff_map': crossattn_prob}
            np.save(os.path.join(output_folder, f"crossattn_{args.dataset}_mask.npy"), crossattn_mask)
            save_method_png(original_img, edited_img, crossattn_prob, crossattn_mask,
                            "CrossAttn", output_folder, args.dataset, vmax=1.0)
        except Exception as e:
            print(f"  SKIPPED: {e}")
            import traceback
            traceback.print_exc()

    # -----------------------------------------------------------------------
    # Comparison & Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPARISON (pairwise IoU)")
    print("=" * 60)

    method_keys = list(results.keys())
    for i in range(len(method_keys)):
        for j in range(i + 1, len(method_keys)):
            name_a, name_b = method_keys[i], method_keys[j]
            metrics = compute_mask_metrics(results[name_a]['mask'], results[name_b]['mask'])
            print(f"  {name_a:>10} vs {name_b:<10} IoU: {metrics['iou']:.3f}")

    # Save summary PNG
    if results:
        save_summary_png(original_img, edited_img, results, output_folder,
                         args.dataset)

    print("\nDone!")


if __name__ == "__main__":
    main()
