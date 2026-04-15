"""
Compare Depth Predictions: Original vs Edited Image  (v1)
=========================================================
v1 strategy: calibrate scale on the ORIGINAL image (all pixels vs depth_gt_orig),
apply the same scale to the edited prediction, evaluate on unchanged and changed
pixels vs depth_gt_edit.

Requires both original and edited SceneDepth EXRs (GT mode only — no .npy masks).
GT change mask is derived from |depth_gt_edit − depth_gt_orig| > threshold and
saved as a PNG so you can visually verify it is correct.

Args:
    --model             dpro | da3_giant | da3_nested          (default: dpro)
    --dataset           new0 | new1 | depth4 | concrete1 | test2  (default: new0)
    --scaling           ls | median                              (default: ls)
    --change-threshold  float in metres                          (default: 0.03)
    --no-show           suppress interactive plot window

Usage examples:
    # defaults: dpro, new0, least-squares
    python compare_edit_depth/compare_edit_depth.py

    python compare_edit_depth/compare_edit_depth.py --model da3_giant --dataset new1
    python compare_edit_depth/compare_edit_depth.py --model dpro --dataset new0 --scaling median
    python compare_edit_depth/compare_edit_depth.py --model dpro --dataset new0 --change-threshold 0.05 --no-show
"""

import os
import sys
import subprocess
import tempfile
import argparse
import json
import numpy as np
import matplotlib
from PIL import Image
import OpenEXR
import Imath

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT   = os.path.dirname(SCRIPT_DIR)
OUTPUT_FOLDER  = os.path.join(SCRIPT_DIR, "v1")
GT_TO_CENTIMETERS        = 10000.0
DEFAULT_CHANGE_THRESHOLD = 0.05   # metres

AVAILABLE_DATASETS = ['new0', 'new1']
DEFAULT_DATASET          = 'new0'

AVAILABLE_MODELS = {
    'da3_giant':  'DA3 Giant 1.1',
    'da3_nested': 'DA3 Nested Giant 1.1',
    'dpro':       'Depth Pro',
}

DA3_HF_MODELS = {
    'da3_giant':  'depth-anything/DA3-GIANT-1.1',
    'da3_nested': 'depth-anything/DA3NESTED-GIANT-LARGE-1.1',
}

DA3_MODEL_NAMES = {
    'da3_giant':  'da3-giant',
    'da3_nested': 'da3nested-giant-large',
}


# ---------------------------------------------------------------------------
# Image / depth loading
# ---------------------------------------------------------------------------

def load_exr_rgb(exr_path):
    exr_file = OpenEXR.InputFile(exr_path)
    header   = exr_file.header()
    dw       = header['dataWindow']
    width    = dw.max.x - dw.min.x + 1
    height   = dw.max.y - dw.min.y + 1
    FLOAT    = Imath.PixelType(Imath.PixelType.FLOAT)
    rgb = []
    for c in ['R', 'G', 'B']:
        channel_str = exr_file.channel(c, FLOAT)
        channel     = np.frombuffer(channel_str, dtype=np.float32).reshape(height, width)
        rgb.append(channel)
    img = np.stack(rgb, axis=-1)
    img = np.clip(img, 0, 1)
    return Image.fromarray((img * 255).astype(np.uint8))


def load_image(path):
    if path.lower().endswith('.exr'):
        return load_exr_rgb(path)
    return Image.open(path).convert('RGB')


def load_exr_depth(exr_path, gt_to_cm=GT_TO_CENTIMETERS):
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
# File discovery
# ---------------------------------------------------------------------------

def find_files(folder):
    """Return (original_rgb, edited_rgb, depth_gt_orig, depth_gt_edit).

    Requires two SceneDepth EXRs (GT mode). Raises FileNotFoundError otherwise.
    Files are sorted; index-0 = original, index-1 = edited.
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

    if len(scene_depth_exrs) < 2:
        raise FileNotFoundError(
            f"GT mode requires two SceneDepth EXRs in {folder}, "
            f"found {len(scene_depth_exrs)}. "
            "This dataset does not have an edited GT depth."
        )
    if len(rgb_pngs) < 2:
        raise FileNotFoundError(
            f"GT mode requires two PNG images in {folder}, found {len(rgb_pngs)}."
        )

    return (
        os.path.join(folder, rgb_pngs[0]),
        os.path.join(folder, rgb_pngs[1]),
        os.path.join(folder, scene_depth_exrs[0]),
        os.path.join(folder, scene_depth_exrs[1]),
    )


# ---------------------------------------------------------------------------
# Model inference (subprocess)
# ---------------------------------------------------------------------------

def run_model_subprocess(model_name, rgb_path, output_path):
    rgb_path_safe    = rgb_path.replace('\\', '/')
    output_path_safe = output_path.replace('\\', '/')

    exr_loader = '''
def load_exr_rgb(path):
    import OpenEXR, Imath
    exr_file = OpenEXR.InputFile(path)
    header   = exr_file.header()
    dw       = header["dataWindow"]
    width    = dw.max.x - dw.min.x + 1
    height   = dw.max.y - dw.min.y + 1
    FLOAT    = Imath.PixelType(Imath.PixelType.FLOAT)
    rgb = []
    for c in ["R", "G", "B"]:
        channel_str = exr_file.channel(c, FLOAT)
        channel     = np.frombuffer(channel_str, dtype=np.float32).reshape(height, width)
        rgb.append(channel)
    img = np.stack(rgb, axis=-1)
    img = np.clip(img, 0, 1)
    return Image.fromarray((img * 255).astype(np.uint8))

def load_image(path):
    if path.lower().endswith(".exr"):
        return load_exr_rgb(path)
    return Image.open(path).convert("RGB")
'''

    if model_name in DA3_HF_MODELS:
        hf_model_id    = DA3_HF_MODELS[model_name]
        da3_model_name = DA3_MODEL_NAMES[model_name]
        script = f'''
import gc, sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors import safe_open
torch.set_grad_enabled(False)
{exr_loader}
da3_src = Path(r"{PROJECT_ROOT}") / "Depth-Anything-3" / "src"
if da3_src.exists() and str(da3_src) not in sys.path:
    sys.path.insert(0, str(da3_src))
from depth_anything_3.api import DepthAnything3
device = "cuda" if torch.cuda.is_available() else "cpu"
with torch.device(device):
    da3_model = DepthAnything3(model_name="{da3_model_name}")
da3_model.eval()
model_file = hf_hub_download(repo_id="{hf_model_id}", filename="model.safetensors")
state = da3_model.state_dict(keep_vars=True)
with safe_open(model_file, framework="pt", device="cpu") as h:
    for name in h.keys():
        if name in state:
            src = h.get_tensor(name)
            with torch.no_grad():
                state[name].copy_(src.to(device))
            del src
if torch.cuda.is_available():
    torch.cuda.empty_cache()
img = load_image("{rgb_path_safe}")
with torch.no_grad():
    prediction = da3_model.inference([img])
depth = prediction.depth[0]
np.save("{output_path_safe}", depth)
print(f"OK: shape={{depth.shape}}, range={{depth.min():.2f}}-{{depth.max():.2f}}")
'''

    elif model_name == 'dpro':
        depth_pro_checkpoint = os.path.join(PROJECT_ROOT, "checkpoints", "depth_pro.pt").replace('\\', '/')
        script = f'''
import torch
import numpy as np
from dataclasses import replace
from PIL import Image
torch.set_grad_enabled(False)
{exr_loader}
import depth_pro
from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
device = "cuda" if torch.cuda.is_available() else "cpu"
precision = torch.half if device == "cuda" else torch.float32
config = replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri="{depth_pro_checkpoint}")
model, transform = depth_pro.create_model_and_transforms(config=config, device=device, precision=precision)
model.eval()
img = load_image("{rgb_path_safe}")
image_tensor = transform(img).to(device)
prediction = model.infer(image_tensor, f_px=None)
depth = prediction["depth"].cpu().numpy().squeeze()
np.save("{output_path_safe}", depth)
print(f"OK: shape={{depth.shape}}, range={{depth.min():.2f}}-{{depth.max():.2f}}")
'''
    else:
        raise ValueError(f"Unknown model: {model_name}")

    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True, text=True, timeout=1800,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Subprocess failed:\n{result.stderr}")
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_depth_metrics(pred, gt, mask):
    """MAE, RMSE, δ1/δ2/δ3 on pixels where mask is True and values are valid."""
    valid = mask & (gt > 0.01) & np.isfinite(pred) & np.isfinite(gt) & (pred > 0.01)
    if valid.sum() == 0:
        return dict(n=0, mae=float('nan'), rmse=float('nan'),
                    d1=float('nan'), d2=float('nan'), d3=float('nan'))
    p = pred[valid]
    g = gt[valid]
    err   = p - g
    ratio = np.maximum(p / g, g / p)
    return dict(
        n    = int(valid.sum()),
        mae  = float(np.mean(np.abs(err))),
        rmse = float(np.sqrt(np.mean(err ** 2))),
        d1   = float(np.mean(ratio < 1.25)),
        d2   = float(np.mean(ratio < 1.25 ** 2)),
        d3   = float(np.mean(ratio < 1.25 ** 3)),
    )


def save_gt_mask_png(original_img, edited_img, depth_gt_orig, depth_gt_edit,
                     gt_changed, depth_diff, threshold, out_path):
    """Save a 1×4 visual check of the GT change mask."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    plt.subplots_adjust(wspace=0.06)

    axes[0].imshow(original_img)
    axes[0].set_title('Original', fontsize=11); axes[0].axis('off')

    axes[1].imshow(edited_img)
    axes[1].set_title('Edited', fontsize=11); axes[1].axis('off')

    im = axes[2].imshow(depth_diff, cmap='RdBu_r',
                        vmin=-max(0.5, float(np.abs(depth_diff).max())),
                        vmax= max(0.5, float(np.abs(depth_diff).max())))
    axes[2].set_title(f'GT depth diff (edit − orig)', fontsize=11); axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label='m')

    overlay = np.zeros((*gt_changed.shape, 4), dtype=np.float32)
    overlay[gt_changed]  = [1, 0, 0, 0.6]   # red = changed
    overlay[~gt_changed] = [0, 1, 0, 0.15]  # faint green = unchanged
    axes[3].imshow(edited_img)
    axes[3].imshow(overlay)
    axes[3].set_title(
        f'GT change mask (threshold={threshold} m)\n'
        f'{gt_changed.sum():,} changed  ({gt_changed.mean()*100:.1f}%)',
        fontsize=11)
    axes[3].axis('off')

    plt.suptitle('GT Change Mask — visual check', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  GT mask saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='v1: Compare depth — calibrate on original, evaluate on changed pixels'
    )
    parser.add_argument('--model', default='dpro', choices=list(AVAILABLE_MODELS.keys()))
    parser.add_argument('--scaling', default='ls', choices=['median', 'ls'])
    parser.add_argument('--dataset', default=DEFAULT_DATASET, choices=AVAILABLE_DATASETS)
    parser.add_argument('--change-threshold', type=float, default=DEFAULT_CHANGE_THRESHOLD,
                        help='Depth difference (m) to mark a pixel as changed')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    model_key      = args.model
    model_name     = AVAILABLE_MODELS[model_key]
    scaling_method = args.scaling
    scaling_folder = 'median' if scaling_method == 'median' else 'least_squares'
    scaling_prefix = 'med'   if scaling_method == 'median' else 'ls'
    dataset        = args.dataset
    input_folder   = os.path.join(PROJECT_ROOT, "data", dataset)

    output_subfolder = os.path.join(OUTPUT_FOLDER, f"{dataset}_results", scaling_folder)
    os.makedirs(output_subfolder, exist_ok=True)

    output_filename = f"{scaling_prefix}_cmp_{model_key}.png"
    output_path     = os.path.join(output_subfolder, output_filename)

    print("=" * 70)
    print("DEPTH COMPARISON v1 — calibrate on original, evaluate on changed pixels")
    print("=" * 70)
    print(f"Model:   {model_name}")
    print(f"Scaling: {scaling_folder}")
    print(f"Dataset: {dataset}")
    print(f"Change threshold: {args.change_threshold} m")

    # ── Discover files ──────────────────────────────────────────────────────
    original_path, edited_path, gt_orig_path, gt_edit_path = find_files(input_folder)
    print(f"\nOriginal RGB:  {os.path.basename(original_path)}")
    print(f"Edited   RGB:  {os.path.basename(edited_path)}")
    print(f"GT depth orig: {os.path.basename(gt_orig_path)}")
    print(f"GT depth edit: {os.path.basename(gt_edit_path)}")

    # ── Load images & GT depths ─────────────────────────────────────────────
    original_img  = load_image(original_path)
    edited_img    = load_image(edited_path)
    depth_gt_orig = load_exr_depth(gt_orig_path)
    depth_gt_edit = load_exr_depth(gt_edit_path)

    target_shape = depth_gt_orig.shape
    print(f"\nGT depth orig: {depth_gt_orig.shape}  {depth_gt_orig.min():.2f}–{depth_gt_orig.max():.2f} m")
    print(f"GT depth edit: {depth_gt_edit.shape}  {depth_gt_edit.min():.2f}–{depth_gt_edit.max():.2f} m")

    # ── GT change mask ───────────────────────────────────────────────────────
    depth_diff   = depth_gt_edit - depth_gt_orig
    gt_changed   = np.abs(depth_diff) > args.change_threshold
    gt_unchanged = ~gt_changed
    print(f"\nGT mask: changed={gt_changed.sum():,} ({gt_changed.mean()*100:.1f}%)  "
          f"unchanged={gt_unchanged.sum():,}")

    # Save GT mask visual check
    mask_png_path = os.path.join(output_subfolder, f"gt_mask_{dataset}.png")
    save_gt_mask_png(original_img, edited_img, depth_gt_orig, depth_gt_edit,
                     gt_changed, depth_diff, args.change_threshold, mask_png_path)

    # ── Run model on both images ────────────────────────────────────────────
    def _infer(img_path, label):
        print(f"\nRunning {model_name} on {label} ...")
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            tmp = f.name
        try:
            out = run_model_subprocess(model_key, img_path, tmp)
            print(f"  {out}")
            return np.load(tmp)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    depth_original = _infer(original_path, "ORIGINAL")
    depth_edited   = _infer(edited_path,   "EDITED")

    # ── Resize predictions to GT shape ──────────────────────────────────────
    def _resize(arr, shape):
        if arr.shape == shape:
            return arr
        return np.array(Image.fromarray(arr.astype(np.float32)).resize(
            (shape[1], shape[0]), Image.BILINEAR))

    depth_original = _resize(depth_original, target_shape)
    depth_edited   = _resize(depth_edited,   target_shape)
    if edited_img.size != original_img.size:
        edited_img = edited_img.resize(original_img.size, Image.BILINEAR)

    # ── Scale fit on original image (all valid pixels vs depth_gt_orig) ─────
    valid_for_fit = (
        (depth_gt_orig > 0.1) & (depth_gt_orig < 100)
        & np.isfinite(depth_original) & np.isfinite(depth_gt_orig)
    )
    p_fit = depth_original[valid_for_fit].flatten()
    g_fit = depth_gt_orig[valid_for_fit].flatten()

    if scaling_method == 'median':
        scale = float(np.median(g_fit) / np.median(p_fit))
        shift = 0.0
    else:
        A = np.vstack([p_fit, np.ones_like(p_fit)]).T
        scale, shift = [float(x) for x in np.linalg.lstsq(A, g_fit, rcond=None)[0]]

    depth_original_scaled = depth_original * scale + shift
    depth_edited_scaled   = depth_edited   * scale + shift
    print(f"\nScale fit on original (all pixels): scale={scale:.4f}, shift={shift:.4f} m")

    # ── Metrics ──────────────────────────────────────────────────────────────
    orig_unch  = compute_depth_metrics(depth_original_scaled, depth_gt_orig,  gt_unchanged)
    unch_m     = compute_depth_metrics(depth_edited_scaled,   depth_gt_edit,  gt_unchanged)
    ch_m       = compute_depth_metrics(depth_edited_scaled,   depth_gt_edit,  gt_changed)

    print("\n" + "=" * 65)
    print("METRICS")
    print("=" * 65)
    print(f"{'Region':<22} {'n':>8} {'MAE (m)':>10} {'RMSE (m)':>10} {'δ1':>8} {'δ2':>8} {'δ3':>8}")
    print("-" * 65)
    for label, m in [
        ("orig vs GT_orig (unch)", orig_unch),
        ("edit vs GT_edit (unch)", unch_m),
        ("edit vs GT_edit (chng)", ch_m),
    ]:
        print(f"{label:<22} {m['n']:>8,} {m['mae']:>10.4f} {m['rmse']:>10.4f} "
              f"{m['d1']:>8.3f} {m['d2']:>8.3f} {m['d3']:>8.3f}")
    print("=" * 65)

    # ── Save JSON metrics ────────────────────────────────────────────────────
    metrics_json_path = os.path.join(output_subfolder, "metrics_data.json")
    if os.path.exists(metrics_json_path):
        with open(metrics_json_path) as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    all_metrics[model_key] = {
        'scale': scale, 'shift': shift,
        'change_threshold_m': args.change_threshold,
        'gt_changed_frac': float(gt_changed.mean()),
        'orig_unchanged': orig_unch,
        'edit_unchanged': unch_m,
        'edit_changed':   ch_m,
    }

    with open(metrics_json_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved: {metrics_json_path}")

    # ── Main visualization (2×4) ─────────────────────────────────────────────
    from scipy import ndimage

    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    plt.subplots_adjust(hspace=0.18, wspace=0.06)

    vmin_d = min(depth_gt_orig.min(), depth_gt_edit.min())
    vmax_d = max(depth_gt_orig.max(), depth_gt_edit.max())

    # Row 0: Original, Edited, orig pred scaled, edit pred scaled
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image', fontsize=11); axes[0, 0].axis('off')

    axes[0, 1].imshow(edited_img)
    axes[0, 1].set_title('Edited Image', fontsize=11); axes[0, 1].axis('off')

    im = axes[0, 2].imshow(depth_original_scaled, cmap='turbo', vmin=vmin_d, vmax=vmax_d)
    axes[0, 2].set_title(f'{model_name}\n(Original, scaled)', fontsize=11); axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04, label='m')

    im = axes[0, 3].imshow(depth_edited_scaled, cmap='turbo', vmin=vmin_d, vmax=vmax_d)
    axes[0, 3].set_title(f'{model_name}\n(Edited, scaled)', fontsize=11); axes[0, 3].axis('off')
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04, label='m')

    # Row 1: GT depth (edit), GT change mask, error on unchanged, error on changed
    im = axes[1, 0].imshow(depth_gt_edit, cmap='turbo', vmin=vmin_d, vmax=vmax_d)
    axes[1, 0].set_title('GT Depth (Edit)', fontsize=11); axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04, label='m')

    axes[1, 1].imshow(gt_changed.astype(np.uint8), cmap='RdYlGn_r')
    axes[1, 1].set_title(
        f'GT Change Mask\n{gt_changed.mean()*100:.1f}% changed  (thr={args.change_threshold}m)',
        fontsize=11)
    axes[1, 1].axis('off')

    err_unch = np.where(
        ndimage.binary_erosion(gt_unchanged, iterations=2),
        depth_edited_scaled - depth_gt_edit, np.nan,
    )
    im = axes[1, 2].imshow(err_unch, cmap='PuOr', vmin=-0.5, vmax=0.5)
    axes[1, 2].set_title(
        f'Error (unchanged)\nMAE={unch_m["mae"]:.3f}m  RMSE={unch_m["rmse"]:.3f}m', fontsize=11)
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04, label='m')

    err_ch = np.where(
        ndimage.binary_erosion(gt_changed, iterations=1),
        depth_edited_scaled - depth_gt_edit, np.nan,
    )
    im = axes[1, 3].imshow(err_ch, cmap='PuOr', vmin=-0.5, vmax=0.5)
    axes[1, 3].set_title(
        f'Error (changed)\nMAE={ch_m["mae"]:.3f}m  δ1={ch_m["d1"]:.3f}', fontsize=11)
    axes[1, 3].axis('off')
    plt.colorbar(im, ax=axes[1, 3], fraction=0.046, pad=0.04, label='m')

    plt.suptitle(
        f'v1 Depth Comparison: {model_name} ({scaling_folder}) — GT mask\n'
        f'dataset={dataset}  threshold={args.change_threshold}m',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {output_path}")

    if not args.no_show:
        plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
