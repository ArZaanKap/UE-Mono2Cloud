"""
Change Detection Model Comparison
==================================
Runs RGB / DINOv2 / GeSCF / CrossAttn change detection on a dataset image pair
and writes binary masks (.npy) and visualisations (.png) to
    change_detection_results/{dataset}/

Default parameters are read from params.py — edit that file to keep the sweep
notebooks and this script in sync.

Usage:
    python change_detection_results/test_change_detection.py --dataset depth4
    python change_detection_results/test_change_detection.py --dataset depth4 --skip-dino --skip-crossattn
"""

import os, sys, glob, argparse, json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy import ndimage

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from params import DINO_BASELINE, GESCF_BASELINE, RGB_BASELINE, CROSSATTN_BASELINE

DEFAULT_DATASET          = "new0"
DEFAULT_CHANGE_THRESHOLD = 0.05   # metres — for GT mask derivation from depth diff

AVAILABLE_DATASETS = ['depth4', 'concrete1', 'test2', 'new0', 'new1']


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
    print(f"  Loading DINOv2 ({model_name}) on {device} ...")
    processor = AutoImageProcessor.from_pretrained(
        model_name,
        size={"height": 518, "width": 518},
        crop_size={"height": 518, "width": 518},
    )
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    num_register = 4 if "reg" in model_name else 0

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

    weights_dir  = os.path.join(PROJECT_ROOT, "weights")
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
    axes[1, 1].set_title(f"{label} mask ({mask.mean()*100:.1f}% changed)"); axes[1, 1].axis("off")

    plt.suptitle(f"{label} — {dataset}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fname = f"{label.lower().replace(' ', '_').replace('+', '_')}_{dataset}.png"
    path  = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
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
        axes[0, 2].set_title(f"GT change mask\n({gt_changed.mean()*100:.1f}% changed)")
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
        title = f"{name}\n({mask.mean()*100:.1f}%)"
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
    print(f"  Saved: {path}")


def _clean_old_outputs(out_dir, dataset):
    removed = []
    for pat in [f"*_{dataset}.png", f"change_detection_{dataset}.png"]:
        for f in glob.glob(os.path.join(out_dir, pat)):
            os.remove(f); removed.append(os.path.basename(f))
    if removed:
        print(f"Cleaned: {', '.join(removed)}")


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
    p.add_argument('--skip-dino',      action='store_true')
    p.add_argument('--skip-gescf',     action='store_true')
    p.add_argument('--skip-crossattn', action='store_true')
    p.add_argument('--no-show',        action='store_true')
    # GT depth scoring (optional — enables per-method scoring vs ground truth)
    p.add_argument('--gt-depth-orig',    default=None,
                   help='Path to original GT depth EXR (for GT mask derivation)')
    p.add_argument('--gt-depth-edit',    default=None,
                   help='Path to edited GT depth EXR (for GT mask derivation)')
    p.add_argument('--change-threshold', type=float, default=DEFAULT_CHANGE_THRESHOLD,
                   help='Depth diff threshold (m) to derive GT change mask')
    args = p.parse_args()

    data_dir = os.path.join(PROJECT_ROOT, "data", args.dataset)
    out_dir  = os.path.join(SCRIPT_DIR, args.dataset)
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
              f"({gt_changed.mean()*100:.1f}%) at threshold {args.change_threshold} m")

    _clean_old_outputs(out_dir, args.dataset)
    results = {}

    # ── RGB ──────────────────────────────────────────────────────────────────
    print("\n--- RGB threshold ---")
    rgb_mask, rgb_diff = rgb_threshold_mask(
        original_img, edited_img,
        threshold=args.rgb_threshold,
        min_area=RGB_BASELINE['min_area'],
        dilate_iter=RGB_BASELINE['dilate_iter'],
    )
    print(f"  threshold={args.rgb_threshold}  changed={rgb_mask.mean()*100:.1f}%")
    results['RGB'] = {'mask': rgb_mask, 'diff_map': rgb_diff}
    np.save(os.path.join(out_dir, f"rgb_{args.dataset}_mask.npy"), rgb_mask)
    _save_method_png(original_img, edited_img, rgb_diff, rgb_mask, "RGB", out_dir, args.dataset, vmax=80)

    # ── DINOv2 ───────────────────────────────────────────────────────────────
    if not args.skip_dino:
        print("\n--- DINOv2 ---")
        try:
            dino_mask, dino_diff = dinov2_feature_mask(
                original_img, edited_img,
                threshold=args.dino_threshold,
                sigma=args.dino_sigma,
                min_area=args.dino_min_area,
                dilate_iter=args.dino_dilate,
            )
            print(f"  threshold={args.dino_threshold}  sigma={args.dino_sigma}  changed={dino_mask.mean()*100:.1f}%")
            results['DINOv2'] = {'mask': dino_mask, 'diff_map': dino_diff}
            np.save(os.path.join(out_dir, f"dinov2_{args.dataset}_mask.npy"), dino_mask)
            _save_method_png(original_img, edited_img, dino_diff, dino_mask,
                             "DINOv2", out_dir, args.dataset, vmax=0.5)
        except Exception as e:
            print(f"  ERROR: {e}")

    # ── GeSCF ────────────────────────────────────────────────────────────────
    if not args.skip_gescf:
        print("\n--- GeSCF ---")
        try:
            gescf_mask, gescf_diff = gescf_feature_mask(
                original_img, edited_img,
                threshold=args.gescf_threshold,
                **GESCF_BASELINE,
            )
            print(f"  changed={gescf_mask.mean()*100:.1f}%")
            results['GeSCF'] = {'mask': gescf_mask, 'diff_map': gescf_diff}
            np.save(os.path.join(out_dir, f"gescf_{args.dataset}_mask.npy"), gescf_mask)
            _save_method_png(original_img, edited_img, gescf_diff, gescf_mask,
                             "GeSCF", out_dir, args.dataset, vmax=1.0)
        except Exception as e:
            print(f"  SKIPPED: {e}")

    # ── CrossAttn ────────────────────────────────────────────────────────────
    if not args.skip_crossattn:
        print("\n--- CrossAttn ---")
        try:
            ca_mask, ca_prob = dino_crossattn_mask(
                original_img, edited_img,
                threshold=args.crossattn_threshold,
                pretrained=args.crossattn_model,
            )
            print(f"  changed={ca_mask.mean()*100:.1f}%")
            results['CrossAttn'] = {'mask': ca_mask, 'diff_map': ca_prob}
            np.save(os.path.join(out_dir, f"crossattn_{args.dataset}_mask.npy"), ca_mask)
            _save_method_png(original_img, edited_img, ca_prob, ca_mask,
                             "CrossAttn", out_dir, args.dataset, vmax=1.0)
        except Exception as e:
            print(f"  SKIPPED: {e}")

    # ── Pairwise IoU summary ──────────────────────────────────────────────────
    if len(results) > 1:
        print("\n--- Pairwise IoU ---")
        keys = list(results.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = results[keys[i]]['mask'], results[keys[j]]['mask']
                inter = np.logical_and(a, b).sum()
                union = np.logical_or(a, b).sum()
                iou   = inter / union if union > 0 else 0.0
                print(f"  {keys[i]:>10} vs {keys[j]:<10}  IoU={iou:.3f}")

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

    if results:
        _save_summary_png(original_img, edited_img, results, out_dir, args.dataset,
                          gt_changed=gt_changed)

    # ── Print scores + save JSON ──────────────────────────────────────────────
    if detection_scores is not None:
        print("\n" + "=" * 65)
        print("SCORING vs GT CHANGE MASK")
        print("=" * 65)
        print(f"{'Method':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'IoU':>10}")
        print("-" * 65)
        for name in results:
            s = detection_scores['methods'][name]
            print(f"{name:<12} {s['precision']:>10.3f} {s['recall']:>10.3f} "
                  f"{s['f1']:>10.3f} {s['iou']:>10.3f}")
        print("=" * 65)

        scores_path = os.path.join(out_dir, "detection_scores.json")
        with open(scores_path, 'w') as f:
            json.dump(detection_scores, f, indent=2)
        print(f"\nScores saved: {scores_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
