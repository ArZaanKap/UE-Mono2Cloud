"""
Test whether applying sRGB gamma correction to the EXR before change detection
has any measurable effect on the GeSCF mask and downstream depth metrics.

Two loading strategies compared:
  A (current)  - linear:  clip(exr, 0, 1) * 255  →  uint8
  B (proposed) - gamma:   apply sRGB curve first  →  uint8

Runs GeSCF once per strategy, then compares masks.

If the dataset folder contains both original and edited SceneDepth EXRs, the
script also derives a GT change mask directly from depth diff. Otherwise GT
comparison is skipped.
"""

import os
import sys
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "change_detection_results"))

import OpenEXR, Imath
FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

DATASET       = "depth4"
DATASET_DIR   = os.path.join(PROJECT_ROOT, "data", DATASET)
EXR_PATH      = os.path.join(DATASET_DIR, "HighresScreenshot00000.exr")
EDIT_PATH     = os.path.join(DATASET_DIR, "edit.png")
GT_CHANGE_THRESHOLD = 0.0


# ── Two EXR loading strategies ──────────────────────────────────────────────

def load_exr_raw_float(path):
    """Return float32 HxWx3 array clipped to [0, 1]."""
    exr = OpenEXR.InputFile(path)
    dw = exr.header()['dataWindow']
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    rgb = np.stack(
        [np.frombuffer(exr.channel(c, FLOAT), np.float32).reshape(h, w) for c in 'RGB'],
        axis=-1,
    )
    return np.clip(rgb, 0.0, 1.0)


def load_scene_depth(path, gt_to_cm=10000.0):
    """Load a single-channel SceneDepth EXR and convert to metres."""
    exr = OpenEXR.InputFile(path)
    dw = exr.header()['dataWindow']
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    depth = None
    for chan_name in ['R', 'SceneDepth', 'Z']:
        try:
            depth = np.frombuffer(exr.channel(chan_name, FLOAT), np.float32).reshape(h, w)
            break
        except Exception:
            pass
    if depth is None:
        raise RuntimeError(f"Could not read depth channel from {path}")
    return depth / gt_to_cm


def try_build_gt_mask(dataset_dir, threshold=0.0):
    """
    Derive GT mask from the two SceneDepth EXRs in a pair dataset.

    Returns (mask, depth_diff) or (None, None) if the dataset does not contain
    both original and edited GT depth files.
    """
    depth_files = sorted(
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if 'SceneDepth' in f and 'WorldUnits' not in f and f.lower().endswith('.exr')
    )
    if len(depth_files) < 2:
        return None, None

    depth_gt_orig = load_scene_depth(depth_files[0])
    depth_gt_edit = load_scene_depth(depth_files[1])
    depth_diff = depth_gt_edit - depth_gt_orig
    gt_changed = np.abs(depth_diff) > threshold
    return gt_changed, depth_diff


def linear_to_srgb(x):
    """IEC 61966-2-1 sRGB forward curve (vectorised)."""
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1.0 / 2.4) - 0.055)


def load_A_linear(path):
    """Current pipeline: no gamma."""
    rgb = load_exr_raw_float(path)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def load_B_gamma(path):
    """Proposed: apply sRGB gamma before quantising."""
    rgb = load_exr_raw_float(path)
    rgb_srgb = linear_to_srgb(rgb)
    return Image.fromarray((np.clip(rgb_srgb, 0.0, 1.0) * 255).astype(np.uint8))


# ── Helpers ──────────────────────────────────────────────────────────────────

def mask_stats(mask, label):
    pct = mask.mean() * 100
    print(f"  {label:25s}  changed={pct:.2f}%")
    return pct


def compare_masks(a, b, label_a, label_b):
    agree     = (a == b).mean() * 100
    only_a    = (a & ~b).mean() * 100
    only_b    = (~a & b).mean() * 100
    union     = (a | b)
    iou       = (a & b).sum() / union.sum() if union.any() else 1.0
    print(f"\n  Agreement ({label_a} vs {label_b}):")
    print(f"    Pixel agreement : {agree:.2f}%")
    print(f"    IoU             : {iou:.4f}")
    print(f"    Only in {label_a:6s}  : {only_a:.2f}%")
    print(f"    Only in {label_b:6s}  : {only_b:.2f}%")
    return agree, iou


def image_stats(img_a, img_b, label_a, label_b):
    arr_a = np.array(img_a).astype(np.int16)
    arr_b = np.array(img_b).astype(np.int16)
    diff  = np.abs(arr_a - arr_b)
    print(f"\n  Pixel diff ({label_a} original vs {label_b} original):")
    print(f"    MAE             : {diff.mean():.3f} / 255")
    print(f"    Max abs         : {diff.max()}")
    print(f"    > 5 levels      : {(diff > 5).mean()*100:.2f}% of pixels")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    from test_change_detection import gescf_feature_mask

    print("=" * 60)
    print("GAMMA CORRECTION EFFECT TEST")
    print("=" * 60)

    import gc
    import torch

    # 1. Load images
    print("\n[1] Loading images...")
    orig_A = load_A_linear(EXR_PATH)
    orig_B = load_B_gamma(EXR_PATH)
    edit   = Image.open(EDIT_PATH).convert("RGB")

    # Resize edit to EXR resolution — the notebook always does this before GeSCF
    # (running the mask generator on a 2752×1536 image causes OOM on 8 GB VRAM)
    edit_at_exr_res = edit.resize(orig_A.size, Image.LANCZOS)

    print(f"  EXR path  : {EXR_PATH}")
    print(f"  Edit path : {EDIT_PATH}")
    print(f"  EXR size  : {orig_A.size}  |  edit original: {edit.size}  |  edit resized: {edit_at_exr_res.size}")
    image_stats(orig_A, orig_B, "A-linear", "B-gamma")

    # Sanity: how similar are original and edit at EXR resolution?
    arr_orig = np.array(orig_A).astype(np.int16)
    arr_edit = np.array(edit_at_exr_res).astype(np.int16)
    diff_ae  = np.abs(arr_orig - arr_edit)
    print(f"\n  EXR(linear) vs edit (resized) MAE : {diff_ae.mean():.3f}")

    # 2. Run GeSCF - method A (linear)
    print("\n[2] Running GeSCF — A (linear, no gamma)...")
    mask_A, _ = gescf_feature_mask(orig_A, edit_at_exr_res)

    # Free GPU memory before second run
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"  GPU after A: {torch.cuda.memory_allocated()/1e6:.0f} MB allocated")

    # 3. Run GeSCF - method B (gamma)
    print("\n[3] Running GeSCF — B (sRGB gamma)...")
    mask_B, _ = gescf_feature_mask(orig_B, edit_at_exr_res)

    # 4. Derive GT mask if the dataset has both original and edited depth GT
    print("\n[4] Deriving GT mask from SceneDepth EXRs (if available)...")
    mask_baseline, depth_diff = try_build_gt_mask(DATASET_DIR, GT_CHANGE_THRESHOLD)
    if mask_baseline is not None:
        print(f"  Derived GT mask from pair depth files in: {DATASET_DIR}")
        print(f"  GT changed pixels: {mask_baseline.sum():,} ({mask_baseline.mean()*100:.2f}%)")
    else:
        print("  Pair SceneDepth EXRs not found — skipping GT comparison.")

    # 5. Report
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\nMask sizes (% changed pixels):")
    mask_stats(mask_A,        "A: linear (current)")
    mask_stats(mask_B,        "B: gamma  (proposed)")
    if mask_baseline is not None:
        mask_stats(mask_baseline, "GT mask (ground truth)")

    compare_masks(mask_A, mask_B, "A", "B")
    if mask_baseline is not None:
        compare_masks(mask_A, mask_baseline, "A", "baseline")
        compare_masks(mask_B, mask_baseline, "B", "baseline")

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    agree_AB, iou_AB = compare_masks(mask_A, mask_B, "A", "B")
    if iou_AB >= 0.98:
        print("\n  Gamma correction has NO meaningful effect on GeSCF (IoU >= 0.98).")
        print("  The current linear loading is fine.")
    elif iou_AB >= 0.90:
        print("\n  Gamma correction has a SMALL effect (IoU 0.90–0.98).")
        print("  Likely edge-region differences only — probably not worth changing.")
    else:
        print(f"\n  Gamma correction has a NOTABLE effect (IoU {iou_AB:.3f}).")
        print("  Consider switching to gamma-corrected loading.")


if __name__ == "__main__":
    main()
