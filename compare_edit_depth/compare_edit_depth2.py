"""
Scaled Depth Map: Calibrate on Edited Image  (v2)
=================================================
v2 strategy: run model on the EDITED image only, calibrate by fitting
unchanged pixels of the prediction to depth_gt_edit, then evaluate on
both unchanged and changed pixels.

Requires both original and edited SceneDepth EXRs (GT mode only — no .npy masks).
GT change mask is derived from |depth_gt_edit − depth_gt_orig| > threshold and
saved as a PNG so you can visually verify it is correct.

Key difference from v1 (compare_edit_depth.py):
  v1: scale factor from original prediction vs GT (all pixels)
  v2: scale factor from edited prediction vs GT (unchanged pixels only)

Args:
    --model             dpro | da3_giant | da3_nested | marigold_dc  (default: dpro)
    --dataset           new0 | new1 | depth4 | concrete1 | test2     (default: new0)
    --scaling           ls | median   (ignored for marigold_dc)       (default: ls)
    --change-threshold  float in metres                               (default: 0.05)
    --no-show           suppress interactive plot window

Usage examples:
    # defaults: dpro, new0, least-squares
    python compare_edit_depth/compare_edit_depth2.py

    python compare_edit_depth/compare_edit_depth2.py --model da3_giant --dataset new1
    python compare_edit_depth/compare_edit_depth2.py --model dpro --dataset new0 --scaling median
    python compare_edit_depth/compare_edit_depth2.py --model marigold_dc --dataset new0
    python compare_edit_depth/compare_edit_depth2.py --model dpro --dataset new0 --change-threshold 0.02 --no-show
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
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT    = os.path.dirname(SCRIPT_DIR)
OUTPUT_FOLDER   = os.path.join(SCRIPT_DIR, "v2")
GT_TO_CENTIMETERS        = 10000.0
DEFAULT_CHANGE_THRESHOLD = 0.05   # metres
MARIGOLD_MAX_RESOLUTION  = 768

AVAILABLE_DATASETS = ['new0', 'new1']
DEFAULT_DATASET          = 'new0'

AVAILABLE_MODELS = {
    'da3_giant':   'DA3 Giant 1.1',
    'da3_nested':  'DA3 Nested Giant 1.1',
    'dpro':        'Depth Pro',
    'marigold_dc': 'Marigold-DC',
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

def run_model_subprocess(model_name, rgb_path, output_path, sparse_depth_path=None):
    rgb_path_safe    = rgb_path.replace('\\', '/')
    output_path_safe = output_path.replace('\\', '/')
    python_exe       = sys.executable

    if model_name == 'marigold_dc':
        marigold_repo = os.path.join(PROJECT_ROOT, "Marigold-DC")
        if not os.path.isdir(marigold_repo):
            raise FileNotFoundError(f"Marigold-DC repo not found at: {marigold_repo}")
        if sparse_depth_path is None:
            raise ValueError("Marigold-DC requires a sparse_depth_path")
        result = subprocess.run(
            [python_exe, '-m', 'marigold_dc',
             '--in-image', rgb_path,
             '--in-depth', sparse_depth_path,
             '--out-depth', output_path,
             '--processing_resolution', '0'],
            capture_output=True, text=True, timeout=3600, cwd=marigold_repo,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Subprocess failed (exit {result.returncode}):\n"
                f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )
        return result.stdout.strip()

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
        [python_exe, '-c', script],
        capture_output=True, text=True, timeout=1800,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess failed (exit {result.returncode}):\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
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

    diff_abs_max = max(0.5, float(np.abs(depth_diff).max()))
    im = axes[2].imshow(depth_diff, cmap='RdBu_r', vmin=-diff_abs_max, vmax=diff_abs_max)
    axes[2].set_title('GT depth diff (edit − orig)', fontsize=11); axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label='m')

    overlay = np.zeros((*gt_changed.shape, 4), dtype=np.float32)
    overlay[gt_changed]  = [1, 0, 0, 0.6]
    overlay[~gt_changed] = [0, 1, 0, 0.15]
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


def build_sparse_guidance(gt_depth, unchanged_mask):
    """Build sparse metric depth for Marigold-DC from unchanged GT pixels."""
    valid = unchanged_mask & (gt_depth > 0.1) & (gt_depth < 100) & np.isfinite(gt_depth)
    return np.where(valid, gt_depth, 0.0).astype(np.float32), valid


def get_processing_shape(height, width, max_res):
    longest = max(height, width)
    if longest <= max_res:
        return height, width
    scale = max_res / longest
    h = max(8, int(round((height * scale) / 8.0) * 8))
    w = max(8, int(round((width  * scale) / 8.0) * 8))
    return h, w


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='v2: Calibrate depth on edited image (unchanged pixels), evaluate on changed pixels'
    )
    parser.add_argument('--model', default='dpro', choices=list(AVAILABLE_MODELS.keys()))
    parser.add_argument('--scaling', default='ls', choices=['median', 'ls'],
                        help='Scaling method; ignored for marigold_dc')
    parser.add_argument('--dataset', default=DEFAULT_DATASET, choices=AVAILABLE_DATASETS)
    parser.add_argument('--change-threshold', type=float, default=DEFAULT_CHANGE_THRESHOLD,
                        help='Depth difference (m) to mark a pixel as changed')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    model_key      = args.model
    model_name     = AVAILABLE_MODELS[model_key]
    scaling_method = args.scaling
    scaling_folder = ('guided_completion' if model_key == 'marigold_dc'
                      else ('median' if scaling_method == 'median' else 'least_squares'))
    dataset        = args.dataset
    input_folder   = os.path.join(PROJECT_ROOT, "data", dataset)

    output_subfolder = os.path.join(OUTPUT_FOLDER, f"{dataset}_results2", scaling_folder)
    os.makedirs(output_subfolder, exist_ok=True)

    output_filename = f"{model_key}_visualization.png"
    output_path     = os.path.join(output_subfolder, output_filename)

    print("=" * 70)
    print("DEPTH COMPARISON v2 — calibrate on edited image (unchanged pixels)")
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
    print(f"\nGT depth edit: {depth_gt_edit.shape}  {depth_gt_edit.min():.2f}–{depth_gt_edit.max():.2f} m")

    # ── GT change mask ───────────────────────────────────────────────────────
    depth_diff   = depth_gt_edit - depth_gt_orig
    gt_changed   = np.abs(depth_diff) > args.change_threshold
    gt_unchanged = ~gt_changed
    print(f"GT mask: changed={gt_changed.sum():,} ({gt_changed.mean()*100:.1f}%)  "
          f"unchanged={gt_unchanged.sum():,}")

    # Save GT mask visual check
    mask_png_path = os.path.join(output_subfolder, f"gt_mask_{dataset}.png")
    save_gt_mask_png(original_img, edited_img, depth_gt_orig, depth_gt_edit,
                     gt_changed, depth_diff, args.change_threshold, mask_png_path)

    # ── Build Marigold-DC sparse guidance from unchanged GT pixels ───────────
    sparse_guidance, valid_guidance_mask = build_sparse_guidance(depth_gt_edit, gt_unchanged)
    if model_key == 'marigold_dc' and valid_guidance_mask.sum() == 0:
        raise ValueError("No valid unchanged GT pixels for Marigold-DC guidance")

    # ── Run model on EDITED image only ──────────────────────────────────────
    print(f"\nRunning {model_name} on EDITED image...")

    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        output_edited = f.name

    sparse_guidance_path = None
    marigold_image_path  = None
    try:
        if model_key == 'marigold_dc':
            mar_h, mar_w = get_processing_shape(target_shape[0], target_shape[1], MARIGOLD_MAX_RESOLUTION)
            mar_guidance = sparse_guidance
            if (mar_h, mar_w) != target_shape:
                mar_guidance = np.array(
                    Image.fromarray(sparse_guidance).resize((mar_w, mar_h), Image.NEAREST),
                    dtype=np.float32,
                )
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
                sparse_guidance_path = f.name
            np.save(sparse_guidance_path, mar_guidance)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                marigold_image_path = f.name
            edited_img.resize((mar_w, mar_h), Image.BILINEAR).save(marigold_image_path)
            print(f"  Marigold input: {mar_w}×{mar_h}  guidance pixels: {np.count_nonzero(mar_guidance):,}")
            out = run_model_subprocess(
                model_key, marigold_image_path, output_edited,
                sparse_depth_path=sparse_guidance_path,
            )
        else:
            out = run_model_subprocess(model_key, edited_path, output_edited)
        if out:
            print(f"  {out}")
        depth_edited = np.load(output_edited)
    finally:
        for p in [output_edited, sparse_guidance_path, marigold_image_path]:
            if p and os.path.exists(p):
                os.remove(p)

    # ── Resize prediction ────────────────────────────────────────────────────
    if depth_edited.shape != target_shape:
        depth_edited = np.array(
            Image.fromarray(depth_edited.astype(np.float32)).resize(
                (target_shape[1], target_shape[0]), Image.BILINEAR
            )
        )
    if edited_img.size != original_img.size:
        edited_img = edited_img.resize(original_img.size, Image.BILINEAR)

    # ── Scale fit on unchanged pixels vs depth_gt_edit ───────────────────────
    if model_key == 'marigold_dc':
        scale, shift = 1.0, 0.0
        depth_scaled = depth_edited.copy()
        print("(Marigold-DC: no scale fit — guided completion)")
    else:
        valid_fit = (
            gt_unchanged
            & (depth_gt_edit > 0.1) & (depth_gt_edit < 100)
            & np.isfinite(depth_edited) & np.isfinite(depth_gt_edit)
        )
        p_fit = depth_edited[valid_fit].flatten()
        g_fit = depth_gt_edit[valid_fit].flatten()

        if scaling_method == 'median':
            scale = float(np.median(g_fit) / np.median(p_fit))
            shift = 0.0
            depth_scaled = depth_edited * scale
        else:
            A = np.vstack([p_fit, np.ones_like(p_fit)]).T
            scale, shift = [float(x) for x in np.linalg.lstsq(A, g_fit, rcond=None)[0]]
            depth_scaled = depth_edited * scale + shift
        print(f"\nScale fit on unchanged pixels: scale={scale:.4f}, shift={shift:.4f} m")

    # ── Metrics ──────────────────────────────────────────────────────────────
    unch_m = compute_depth_metrics(depth_scaled, depth_gt_edit, gt_unchanged)
    ch_m   = compute_depth_metrics(depth_scaled, depth_gt_edit, gt_changed)

    print("\n" + "=" * 65)
    print("METRICS")
    print("=" * 65)
    print(f"{'Region':<22} {'n':>8} {'MAE (m)':>10} {'RMSE (m)':>10} {'δ1':>8} {'δ2':>8} {'δ3':>8}")
    print("-" * 65)
    for label, m in [("edit vs GT_edit (unch)", unch_m), ("edit vs GT_edit (chng)", ch_m)]:
        print(f"{label:<22} {m['n']:>8,} {m['mae']:>10.4f} {m['rmse']:>10.4f} "
              f"{m['d1']:>8.3f} {m['d2']:>8.3f} {m['d3']:>8.3f}")
    print("=" * 65)

    # ── Save JSON metrics ────────────────────────────────────────────────────
    metrics_entry = {
        'scale': scale, 'shift': shift,
        'scaling': scaling_folder,
        'change_threshold_m': args.change_threshold,
        'gt_changed_frac': float(gt_changed.mean()),
        'edit_unchanged': unch_m,
        'edit_changed':   ch_m,
    }
    if model_key == 'marigold_dc':
        metrics_entry['guidance_pixels'] = int(valid_guidance_mask.sum())

    metrics_path = os.path.join(output_subfolder, "metrics_data.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    all_metrics[model_key] = metrics_entry

    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")

    # ── Main visualization (2×4) ─────────────────────────────────────────────
    from scipy import ndimage

    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    plt.subplots_adjust(hspace=0.18, wspace=0.06)

    vmin_d = depth_gt_edit.min()
    vmax_d = depth_gt_edit.max()

    # Row 0: Original, Edited, GT depth (edit), GT change mask
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image', fontsize=11); axes[0, 0].axis('off')

    axes[0, 1].imshow(edited_img)
    axes[0, 1].set_title('Edited Image', fontsize=11); axes[0, 1].axis('off')

    im = axes[0, 2].imshow(depth_gt_edit, cmap='turbo', vmin=vmin_d, vmax=vmax_d)
    axes[0, 2].set_title('GT Depth (Edit)', fontsize=11); axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04, label='m')

    axes[0, 3].imshow(gt_changed.astype(np.uint8), cmap='RdYlGn_r')
    axes[0, 3].set_title(
        f'GT Change Mask\n{gt_changed.mean()*100:.1f}% changed  (thr={args.change_threshold}m)',
        fontsize=11)
    axes[0, 3].axis('off')

    # Row 1: Pred scaled, full error, error unchanged, error changed
    suffix = 'Dense (Marigold)' if model_key == 'marigold_dc' else 'Scaled (unch fit)'
    im = axes[1, 0].imshow(depth_scaled, cmap='turbo', vmin=vmin_d, vmax=vmax_d)
    axes[1, 0].set_title(f'{model_name}\n({suffix})', fontsize=11); axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04, label='m')

    full_err = depth_scaled - depth_gt_edit
    im = axes[1, 1].imshow(full_err, cmap='PuOr', vmin=-0.5, vmax=0.5)
    axes[1, 1].set_title('Error vs GT (full image)', fontsize=11); axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04, label='m')

    err_unch = np.where(
        ndimage.binary_erosion(gt_unchanged, iterations=2),
        full_err, np.nan,
    )
    im = axes[1, 2].imshow(err_unch, cmap='PuOr', vmin=-0.5, vmax=0.5)
    axes[1, 2].set_title(
        f'Error (unchanged)\nMAE={unch_m["mae"]:.3f}m  RMSE={unch_m["rmse"]:.3f}m', fontsize=11)
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04, label='m')

    err_ch = np.where(
        ndimage.binary_erosion(gt_changed, iterations=1),
        full_err, np.nan,
    )
    im = axes[1, 3].imshow(err_ch, cmap='PuOr', vmin=-0.5, vmax=0.5)
    axes[1, 3].set_title(
        f'Error (changed)\nMAE={ch_m["mae"]:.3f}m  δ1={ch_m["d1"]:.3f}', fontsize=11)
    axes[1, 3].axis('off')
    plt.colorbar(im, ax=axes[1, 3], fraction=0.046, pad=0.04, label='m')

    plt.suptitle(
        f'v2 Depth (edit calibration): {model_name} ({scaling_folder}) — GT mask\n'
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
