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
    --model             dpro | da3_giant | da3_nested | marigold_dc | depthlab  (default: dpro)
    --dataset           new0 | new1 | depth4 | concrete1 | test2     (default: new0)
    --scaling           ls | median   (ignored for marigold_dc / depthlab)     (default: ls)
    --change-threshold  float in metres                               (default: 0.05)
    --no-show           suppress interactive plot window
    --all-models        run all available models sequentially on the given dataset

Usage examples:
    # defaults: dpro, new0, least-squares
    python compare_edit_depth/compare_edit_depth2.py

    python compare_edit_depth/compare_edit_depth2.py --model da3_giant --dataset new1
    python compare_edit_depth/compare_edit_depth2.py --model dpro --dataset new0 --scaling median
    python compare_edit_depth/compare_edit_depth2.py --model marigold_dc --dataset new0
    python compare_edit_depth/compare_edit_depth2.py --model depthlab --dataset new0
    python compare_edit_depth/compare_edit_depth2.py --model dpro --dataset new0 --change-threshold 0.02 --no-show
    python compare_edit_depth/compare_edit_depth2.py --all-models --dataset new1
"""

import os
import sys
import subprocess
import tempfile
import argparse
import json
import warnings
import numpy as np
import matplotlib
from PIL import Image
import OpenEXR
import Imath

matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore', 'FigureCanvasAgg is non-interactive')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT    = os.path.dirname(SCRIPT_DIR)
OUTPUT_FOLDER   = os.path.join(SCRIPT_DIR, "v2")
GT_TO_CENTIMETERS        = 10000.0
DEFAULT_CHANGE_THRESHOLD = 0.0   # metres
MARIGOLD_MAX_RESOLUTION  = 768
DEPTHLAB_MAX_RESOLUTION  = 768

AVAILABLE_DATASETS = ['new0', 'new1','new2','new3','new4']
DEFAULT_DATASET          = 'new0'

AVAILABLE_MODELS = {
    'da3_giant':   'DA3 Giant 1.1',
    'da3_nested':  'DA3 Nested Giant 1.1',
    'dpro':        'Depth Pro',
    'marigold_dc': 'Marigold-DC',
    'depthlab':    'DepthLab',
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

def run_model_subprocess(model_name, rgb_path, output_path, sparse_depth_path=None, mask_path=None):
    rgb_path_safe    = rgb_path.replace('\\', '/')
    output_path_safe = output_path.replace('\\', '/')
    python_exe       = sys.executable

    if model_name == 'marigold_dc':
        marigold_repo = os.path.join(PROJECT_ROOT, "depth_models", "Marigold-DC")
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

    if model_name == 'depthlab':
        depthlab_repo = os.path.join(PROJECT_ROOT, "depth_models", "DepthLab")
        if not os.path.isdir(depthlab_repo):
            raise FileNotFoundError(f"DepthLab repo not found at: {depthlab_repo}")
        if sparse_depth_path is None or mask_path is None:
            raise ValueError("DepthLab requires sparse_depth_path and mask_path")
        infer_script = os.path.join(SCRIPT_DIR, "depthlab_infer.py")
        result = subprocess.run(
            [python_exe, infer_script,
             '--in-image',       rgb_path,
             '--in-depth',       sparse_depth_path,
             '--in-mask',        mask_path,
             '--out-depth',      output_path,
             '--depthlab-dir',   depthlab_repo,
             '--denoise-steps',  '20',
             '--processing-res', '0'],
            capture_output=True, text=True, timeout=3600,
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
da3_src = Path(r"{PROJECT_ROOT}") / "depth_models" / "Depth-Anything-3" / "src"
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
        depth_pro_checkpoint = os.path.join(PROJECT_ROOT, "depth_models", "checkpoints", "depth_pro.pt").replace('\\', '/')
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


def _load_world_normal(path):
    """Load WorldNormal EXR -> unit normals in UE world space (X=fwd, Y=right, Z=up)."""
    exr_file = OpenEXR.InputFile(path)
    header   = exr_file.header()
    dw       = header['dataWindow']
    width    = dw.max.x - dw.min.x + 1
    height   = dw.max.y - dw.min.y + 1
    FLOAT    = Imath.PixelType(Imath.PixelType.FLOAT)
    ch = []
    for c in ['R', 'G', 'B']:
        buf = exr_file.channel(c, FLOAT)
        ch.append(np.frombuffer(buf, dtype=np.float32).reshape(height, width).copy())
    raw = np.stack(ch, axis=-1)
    N   = raw - 1.0   # UE stores world normals as N+1 in [0,2]; decode: raw-1
    return N / (np.linalg.norm(N, axis=-1, keepdims=True) + 1e-6)


def _normals_from_depth(depth_m, fx, fy, cx, cy):
    H, W = depth_m.shape
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    pts = np.stack([(uu - cx) * depth_m / fx,
                    (vv - cy) * depth_m / fy,
                    depth_m.copy()], axis=-1)
    # np.gradient handles image boundaries correctly (no wrap-around artefacts)
    dy, dx = np.gradient(pts, axis=(0, 1))
    n = np.cross(dx, dy)
    return -n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6)


def _disc_mask(depth_m, dilate=3):
    """Pixels at depth discontinuities where finite-diff normals are unreliable."""
    dx = np.abs(np.roll(depth_m, -1, axis=1) - np.roll(depth_m, 1, axis=1))
    dy = np.abs(np.roll(depth_m, -1, axis=0) - np.roll(depth_m, 1, axis=0))
    from scipy.ndimage import binary_dilation
    disc = (dx > 0.3) | (dy > 0.3)
    return binary_dilation(disc, iterations=dilate) if dilate > 0 else disc


def _ue_cam_to_world(pitch_deg, yaw_deg, roll_deg):
    """
    Camera-to-world rotation for a UE camera.
    UE world: X=forward, Y=right, Z=up. Screen space: X=right, Y=down, Z=depth.
    Rotation order: Rz(yaw) @ Ry(pitch) @ Rx(roll)  (extrinsic Z->Y->X).
    Fill in pitch/yaw/roll from UE Details panel > Transform > Rotation.

    Sign convention: UE positive-pitch = nose-up, but standard Ry(+θ) = nose-down,
    so pitch is negated. Same logic applies to roll.
    """
    p = -np.radians(pitch_deg)   # negate: UE +pitch = nose-up = standard Ry(-p)
    y =  np.radians(yaw_deg)
    r = -np.radians(roll_deg)    # negate: UE +roll = CW-fwd = standard Rx(-r)
    Rz = np.array([[ np.cos(y), -np.sin(y), 0],
                   [ np.sin(y),  np.cos(y), 0],
                   [0, 0, 1]], dtype=float)
    Ry = np.array([[ np.cos(p), 0, np.sin(p)],
                   [0, 1, 0],
                   [-np.sin(p), 0, np.cos(p)]], dtype=float)
    Rx = np.array([[1, 0, 0],
                   [0,  np.cos(r), -np.sin(r)],
                   [0,  np.sin(r),  np.cos(r)]], dtype=float)
    # Base: UE default camera (P=Y=R=0) looks along world +X
    # screen-right -> world +Y,  screen-down -> world -Z,  depth -> world +X
    R_base = np.array([[0, 0, 1],
                       [1, 0, 0],
                       [0, -1, 0]], dtype=float)
    return Rz @ Ry @ Rx @ R_base


_SNA_NAN = dict(sna_mean=float('nan'), sna_median=float('nan'),
                pct_11=float('nan'), pct_22=float('nan'), pct_30=float('nan'))


def compute_sna(depth_pred, gt_normals_world, mask, fx, fy, cx, cy, R_cam_to_world):
    """Surface Normal Alignment vs UE WorldNormal EXR (degrees + threshold %).
    Depth discontinuities are excluded from the mask — finite-diff normals at
    edges are unreliable and would inflate the reported SNA values.
    """
    disc  = _disc_mask(depth_pred)
    valid = mask & ~disc & np.isfinite(depth_pred) & (depth_pred > 0.1)
    if valid.sum() == 0:
        return _SNA_NAN.copy()
    n_cam   = _normals_from_depth(depth_pred, fx, fy, cx, cy)
    n_world = (R_cam_to_world @ n_cam.reshape(-1, 3).T).T.reshape(n_cam.shape)
    n_world = n_world / (np.linalg.norm(n_world, axis=-1, keepdims=True) + 1e-6)
    dot = np.clip((n_world * gt_normals_world).sum(axis=-1), -1.0, 1.0)
    ang = np.degrees(np.arccos(dot))
    a   = ang[valid]
    return dict(
        sna_mean   = float(a.mean()),
        sna_median = float(np.median(a)),
        pct_11     = float((a < 11.25).mean() * 100),
        pct_22     = float((a < 22.5).mean()  * 100),
        pct_30     = float((a < 30.0).mean()  * 100),
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
    parser.add_argument('--all-models', action='store_true',
                        help='Run all available models sequentially on the given dataset')
    args = parser.parse_args()

    if args.all_models:
        print("STARTING")
        for mk in AVAILABLE_MODELS:
            print(f"\n--- {AVAILABLE_MODELS[mk]} ---")
            cmd = [sys.executable, __file__,
                   '--model', mk,
                   '--dataset', args.dataset,
                   '--scaling', args.scaling,
                   '--change-threshold', str(args.change_threshold)]
            if args.no_show:
                cmd.append('--no-show')
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"\n[WARN] {AVAILABLE_MODELS[mk]} failed (exit {result.returncode}), skipping.\n")
        print("\nFINISHED")
        return

    model_key      = args.model
    model_name     = AVAILABLE_MODELS[model_key]
    scaling_method = args.scaling
    scaling_folder = ('guided_completion' if model_key == 'marigold_dc'
                      else 'depthlab'       if model_key == 'depthlab'
                      else ('median' if scaling_method == 'median' else 'least_squares'))
    dataset        = args.dataset
    input_folder   = os.path.join(PROJECT_ROOT, "data", dataset)

    output_subfolder = os.path.join(OUTPUT_FOLDER, f"{dataset}_results2", scaling_folder)
    os.makedirs(output_subfolder, exist_ok=True)

    output_filename = f"{model_key}_visualization.png"
    output_path     = os.path.join(output_subfolder, output_filename)

    # ── Discover files ──────────────────────────────────────────────────────
    original_path, edited_path, gt_orig_path, gt_edit_path = find_files(input_folder)

    # ── Load images & GT depths ─────────────────────────────────────────────
    original_img  = load_image(original_path)
    edited_img    = load_image(edited_path)
    depth_gt_orig = load_exr_depth(gt_orig_path)
    depth_gt_edit = load_exr_depth(gt_edit_path)

    target_shape = depth_gt_orig.shape
    H_gt, W_gt   = target_shape

    # ── Load camera params ───────────────────────────────────────────────────
    cam_params_path = os.path.join(input_folder, "camera_params.json")
    cp = {}
    if os.path.exists(cam_params_path):
        with open(cam_params_path) as f:
            cp = json.load(f)
    _fov = cp.get('fov_deg') or 90.0
    if cp.get('fov_deg') is None:
        print(f"  [warn] fov_deg not set in camera_params.json — assuming {_fov}°")
    _fx = (W_gt / 2.0) / np.tan(np.radians(_fov) / 2.0)
    _fy, _cx, _cy = _fx, W_gt / 2.0, H_gt / 2.0

    # ── GT change mask ───────────────────────────────────────────────────────
    depth_diff   = depth_gt_edit - depth_gt_orig
    gt_changed   = np.abs(depth_diff) > args.change_threshold
    gt_unchanged = ~gt_changed

    mask_png_path = os.path.join(output_subfolder, f"gt_mask_{dataset}.png")
    save_gt_mask_png(original_img, edited_img, depth_gt_orig, depth_gt_edit,
                     gt_changed, depth_diff, args.change_threshold, mask_png_path)

    # ── Load WorldNormal for SNA ─────────────────────────────────────────────
    sna_ready = False
    R_cam_to_world = gt_normals_edit = None
    if None not in (cp.get('pitch_deg'), cp.get('yaw_deg'), cp.get('roll_deg')):
        R_cam_to_world = _ue_cam_to_world(cp['pitch_deg'], cp['yaw_deg'], cp['roll_deg'])
        wn_edit = gt_edit_path.replace('_SceneDepth.exr', '_WorldNormal.exr')
        if os.path.exists(wn_edit):
            gt_normals_edit = _load_world_normal(wn_edit)
            sna_ready = True

    # ── Build sparse guidance from unchanged GT pixels (Marigold-DC + DepthLab) ─
    sparse_guidance, valid_guidance_mask = build_sparse_guidance(depth_gt_edit, gt_unchanged)
    guided_models = {'marigold_dc', 'depthlab'}
    if model_key in guided_models and valid_guidance_mask.sum() == 0:
        raise ValueError(f"No valid unchanged GT pixels for {model_name} guidance")

    # ── Run model on EDITED image only ──────────────────────────────────────

    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        output_edited = f.name

    sparse_guidance_path = None
    marigold_image_path  = None
    depthlab_mask_path   = None
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
            out = run_model_subprocess(
                model_key, marigold_image_path, output_edited,
                sparse_depth_path=sparse_guidance_path,
            )

        elif model_key == 'depthlab':
            dl_h, dl_w = get_processing_shape(target_shape[0], target_shape[1], DEPTHLAB_MAX_RESOLUTION)
            dl_guidance = sparse_guidance
            dl_mask     = gt_changed.astype(np.float32)   # 1=predict, 0=known
            if (dl_h, dl_w) != target_shape:
                dl_guidance = np.array(
                    Image.fromarray(sparse_guidance).resize((dl_w, dl_h), Image.NEAREST),
                    dtype=np.float32,
                )
                dl_mask = np.array(
                    Image.fromarray(dl_mask).resize((dl_w, dl_h), Image.NEAREST),
                    dtype=np.float32,
                )
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
                sparse_guidance_path = f.name
            np.save(sparse_guidance_path, dl_guidance)
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
                depthlab_mask_path = f.name
            np.save(depthlab_mask_path, dl_mask)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                marigold_image_path = f.name
            edited_img.resize((dl_w, dl_h), Image.BILINEAR).save(marigold_image_path)
            run_model_subprocess(
                model_key, marigold_image_path, output_edited,
                sparse_depth_path=sparse_guidance_path,
                mask_path=depthlab_mask_path,
            )

        else:
            run_model_subprocess(model_key, edited_path, output_edited)
        depth_edited = np.load(output_edited)
    finally:
        for p in [output_edited, sparse_guidance_path, marigold_image_path, depthlab_mask_path]:
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
    if model_key in {'marigold_dc', 'depthlab'}:
        scale, shift = 1.0, 0.0
        depth_scaled = depth_edited.copy()
        print(f"({model_name}: no scale fit — guided completion)")
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

    sna_unch = _SNA_NAN.copy()
    sna_ch   = _SNA_NAN.copy()
    if sna_ready:
        sna_unch = compute_sna(depth_scaled, gt_normals_edit, gt_unchanged, _fx, _fy, _cx, _cy, R_cam_to_world)
        sna_ch   = compute_sna(depth_scaled, gt_normals_edit, gt_changed,   _fx, _fy, _cx, _cy, R_cam_to_world)

    def _sf(s): return f"{s['sna_mean']:8.2f}" if not np.isnan(s['sna_mean']) else "     ---"

    print("\n" + "=" * 75)
    print("METRICS")
    print("=" * 75)
    print(f"{'Region':<22} {'n':>8} {'MAE (m)':>10} {'RMSE (m)':>10} {'d1':>8} {'d2':>8} {'d3':>8} {'SNA(deg)':>8}")
    print("-" * 75)
    for label, m, sna in [("edit vs GT_edit (unch)", unch_m, sna_unch), ("edit vs GT_edit (chng)", ch_m, sna_ch)]:
        print(f"{label:<22} {m['n']:>8,} {m['mae']:>10.4f} {m['rmse']:>10.4f} "
              f"{m['d1']:>8.3f} {m['d2']:>8.3f} {m['d3']:>8.3f} {_sf(sna)}")
    print("=" * 75)

    # ── Save JSON metrics ────────────────────────────────────────────────────
    metrics_entry = {
        'scale': scale, 'shift': shift,
        'scaling': scaling_folder,
        'change_threshold_m': args.change_threshold,
        'gt_changed_frac': float(gt_changed.mean()),
        'edit_unchanged': {**unch_m, **sna_unch},
        'edit_changed':   {**ch_m,   **sna_ch},
    }
    if model_key in {'marigold_dc', 'depthlab'}:
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

    # ── Main visualization (3×4) ─────────────────────────────────────────────
    from scipy import ndimage

    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    plt.subplots_adjust(hspace=0.22, wspace=0.06)

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
    suffix = ('Dense (Marigold)'  if model_key == 'marigold_dc'
              else 'Dense (DepthLab)' if model_key == 'depthlab'
              else 'Scaled (unch fit)')
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

    # Row 2: Surface normal alignment
    if sna_ready:
        # GT world normals -> RGB (UE stores as N+1, already decoded in _load_world_normal)
        gt_normal_rgb = np.clip((gt_normals_edit + 1) / 2, 0, 1)
        axes[2, 0].imshow(gt_normal_rgb)
        axes[2, 0].set_title('GT Surface Normals', fontsize=11)
        axes[2, 0].axis('off')

        # Pred world normals from depth_scaled -> RGB
        n_cam_pred   = _normals_from_depth(depth_scaled, _fx, _fy, _cx, _cy)
        n_world_pred = (R_cam_to_world @ n_cam_pred.reshape(-1, 3).T).T.reshape(n_cam_pred.shape)
        n_world_pred = n_world_pred / (np.linalg.norm(n_world_pred, axis=-1, keepdims=True) + 1e-6)
        pred_normal_rgb = np.clip((n_world_pred + 1) / 2, 0, 1)
        axes[2, 1].imshow(pred_normal_rgb)
        axes[2, 1].set_title(f'{model_name}\nPred Surface Normals', fontsize=11)
        axes[2, 1].axis('off')

        # Angular error (full image)
        dot     = np.clip((n_world_pred * gt_normals_edit).sum(axis=-1), -1.0, 1.0)
        ang_err = np.degrees(np.arccos(dot))
        im = axes[2, 2].imshow(ang_err, cmap='hot_r', vmin=0, vmax=45)
        axes[2, 2].set_title(
            f'Normal Angular Error (full)\nSNA mean={sna_unch["sna_mean"]:.1f}° (unch)', fontsize=11)
        axes[2, 2].axis('off')
        plt.colorbar(im, ax=axes[2, 2], fraction=0.046, pad=0.04, label='°')

        # Angular error for changed pixels only
        disc       = _disc_mask(depth_scaled)
        ang_err_ch = np.where(
            ndimage.binary_erosion(gt_changed, iterations=1) & ~disc,
            ang_err, np.nan,
        )
        im = axes[2, 3].imshow(ang_err_ch, cmap='hot_r', vmin=0, vmax=45)
        axes[2, 3].set_title(
            f'Normal Error (changed)\nSNA={sna_ch["sna_mean"]:.1f}°  <11.25°={sna_ch["pct_11"]:.1f}%',
            fontsize=11)
        axes[2, 3].axis('off')
        plt.colorbar(im, ax=axes[2, 3], fraction=0.046, pad=0.04, label='°')
    else:
        for col in range(4):
            axes[2, col].axis('off')
        axes[2, 0].text(
            0.5, 0.5,
            'Surface normals not available\n(missing camera params or WorldNormal EXR)',
            ha='center', va='center', transform=axes[2, 0].transAxes, fontsize=11,
        )

    plt.suptitle(
        f'v2 Depth (edit calibration): {model_name} ({scaling_folder}) — GT mask\n'
        f'dataset={dataset}  threshold={args.change_threshold}m',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')


if __name__ == "__main__":
    main()
