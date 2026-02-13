"""
Scaled Depth Map: Calibrate on Edited Image (Unchanged Regions)
================================================================
Runs a depth model on the edited image only, then calibrates by fitting
the prediction (unchanged regions) to GT depth.

Key difference from v1 (compare_edit_depth.py):
  v1: scale factor from original prediction vs GT (all pixels)
  v2: scale factor from edited prediction vs GT (unchanged pixels only)

The comparable metric is "Edit vs GT MAE" — same measurement, different scaling.
"""

import os
import sys
import subprocess
import tempfile
import argparse
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import OpenEXR
import Imath

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Parent directory (UE_depth)
OUTPUT_FOLDER = SCRIPT_DIR  # Output to same folder as script
GT_TO_CENTIMETERS = 10000.0

AVAILABLE_DATASETS = ['depth3', 'depth4']

AVAILABLE_MODELS = {
    'da2': 'DA V2 Metric',
    'da3': 'DA3 Metric Large',
    'da3_giant': 'DA3 Giant 1.1',
    'da3_nested': 'DA3 Nested Giant 1.1',
    'dpro': 'Depth Pro',
}

DA3_HF_MODELS = {
    'da3': 'depth-anything/DA3METRIC-LARGE',
    'da3_giant': 'depth-anything/DA3-GIANT-1.1',
    'da3_nested': 'depth-anything/DA3NESTED-GIANT-LARGE-1.1',
}


def load_exr_rgb(exr_path):
    """Load RGB channels from EXR file."""
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


def load_exr_depth(exr_path, gt_to_cm=GT_TO_CENTIMETERS):
    """Load depth from EXR file, convert to meters."""
    exr_file = OpenEXR.InputFile(exr_path)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = list(header['channels'].keys())

    for chan_name in ['R', 'SceneDepth', 'Z']:
        if chan_name in channels:
            channel_str = exr_file.channel(chan_name, FLOAT)
            depth = np.frombuffer(channel_str, dtype=np.float32).reshape(height, width).copy()
            break
    else:
        raise ValueError(f"No depth channel found in {exr_path}")

    depth_m = (depth * gt_to_cm) / 100.0
    return depth_m


def find_files(folder):
    """Find original RGB, edited RGB, and depth GT files in folder."""
    files = os.listdir(folder)

    # Find original RGB: EXR without 'depth' or 'scenedepth' in name
    original_rgb = None
    for f in files:
        if f.endswith('.exr') and 'depth' not in f.lower() and 'scenedepth' not in f.lower():
            original_rgb = os.path.join(folder, f)
            break

    # Find edited image: look for 'edit' in name
    edited_rgb = None
    for f in files:
        if 'edit' in f.lower() and (f.endswith('.png') or f.endswith('.jpg') or f.endswith('.exr')):
            edited_rgb = os.path.join(folder, f)
            break

    # Find depth GT: SceneDepth EXR (not WorldUnits)
    depth_gt = None
    for f in files:
        if 'SceneDepth' in f and 'WorldUnits' not in f and f.endswith('.exr'):
            depth_gt = os.path.join(folder, f)
            break

    return original_rgb, edited_rgb, depth_gt


def run_model_subprocess(model_name, rgb_path, output_path):
    """Run a depth model in a separate subprocess."""
    rgb_path_safe = rgb_path.replace('\\', '/')
    output_path_safe = output_path.replace('\\', '/')

    exr_loader = '''
def load_exr_rgb(path):
    import OpenEXR
    import Imath
    exr_file = OpenEXR.InputFile(path)
    header = exr_file.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    rgb = []
    for c in ["R", "G", "B"]:
        channel_str = exr_file.channel(c, FLOAT)
        channel = np.frombuffer(channel_str, dtype=np.float32).reshape(height, width)
        rgb.append(channel)
    img = np.stack(rgb, axis=-1)
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

def load_image(path):
    if path.lower().endswith(".exr"):
        return load_exr_rgb(path)
    else:
        return Image.open(path).convert("RGB")
'''

    if model_name == 'da2':
        script = f'''
import torch
import numpy as np
from PIL import Image
torch.set_grad_enabled(False)
{exr_loader}
from transformers import pipeline
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf", device=0)
img = load_image("{rgb_path_safe}")
result = pipe(img)
depth = result["predicted_depth"].numpy()
np.save("{output_path_safe}", depth)
print(f"OK: shape={{depth.shape}}, range={{depth.min():.2f}}-{{depth.max():.2f}}")
'''

    elif model_name in DA3_HF_MODELS:
        hf_model_id = DA3_HF_MODELS[model_name]
        script = f'''
import torch
import numpy as np
from PIL import Image
torch.set_grad_enabled(False)
{exr_loader}
from depth_anything_3.api import DepthAnything3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepthAnything3.from_pretrained("{hf_model_id}")
model = model.to(device=device)
img = load_image("{rgb_path_safe}")
prediction = model.inference([img])
depth = prediction.depth[0]
np.save("{output_path_safe}", depth)
print(f"OK: shape={{depth.shape}}, range={{depth.min():.2f}}-{{depth.max():.2f}}")
'''

    elif model_name == 'dpro':
        script = f'''
import torch
import numpy as np
from PIL import Image
torch.set_grad_enabled(False)
{exr_loader}
import depth_pro
device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = depth_pro.create_model_and_transforms(device=device)
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

    python_exe = sys.executable

    result = subprocess.run(
        [python_exe, '-c', script],
        capture_output=True,
        text=True,
        timeout=1800
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess failed (exit code {result.returncode}):\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}"
        )

    return result.stdout.strip()


def load_change_mask(mask_model, dataset):
    """Load a pre-computed change mask from change_detection_results/."""
    mask_dir = os.path.join(PROJECT_ROOT, "change_detection_results", dataset)
    mask_path = os.path.join(mask_dir, f"{mask_model}_{dataset}_mask.npy")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(
            f"Mask not found: {mask_path}\n"
            f"Run test_change_detection.py --dataset {dataset} first."
        )
    mask = np.load(mask_path)
    print(f"  Loaded mask: {mask_path}")
    print(f"  Changed pixels: {mask.sum():,} ({mask.mean()*100:.1f}%)")
    return mask.astype(bool)


def main():
    parser = argparse.ArgumentParser(description='Generate scaled depth map for edited image using GT-calibrated scale factor')
    parser.add_argument('--model', type=str, default='dpro',
                        choices=list(AVAILABLE_MODELS.keys()),
                        help=f'Model to use: {list(AVAILABLE_MODELS.keys())}')
    parser.add_argument('--scaling', type=str, default='ls',
                        choices=['median', 'ls'],
                        help='Scaling method: median or ls (least_squares)')
    parser.add_argument('--dataset', type=str, default='depth4',
                        choices=AVAILABLE_DATASETS,
                        help=f'Dataset to use: {AVAILABLE_DATASETS}')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display the figure (useful for headless)')
    parser.add_argument('--mask-model', type=str, default='gescf',
                        choices=['dinov2', 'gescf'],
                        help='Change detection mask to use (from test_change_detection.py output)')
    args = parser.parse_args()

    model_key = args.model
    model_name = AVAILABLE_MODELS[model_key]
    scaling_method = args.scaling
    scaling_folder = 'median' if scaling_method == 'median' else 'least_squares'
    mask_model = args.mask_model
    dataset = args.dataset
    input_folder = os.path.join(PROJECT_ROOT, "data", dataset)

    # Create output subfolder: compare_edit_depth/{dataset}_results2/{scaling}/
    output_subfolder = os.path.join(OUTPUT_FOLDER, f"{dataset}_results2", scaling_folder)
    os.makedirs(output_subfolder, exist_ok=True)

    # Auto-generate output filename
    output_filename = f"{model_key}_visualization.png"
    output_path = os.path.join(output_subfolder, output_filename)

    print("=" * 70)
    print("SCALED DEPTH MAP GENERATOR")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Scaling: {scaling_folder}")
    print(f"Mask model: {mask_model}")
    print(f"Dataset: {dataset}")
    print(f"Output: {dataset}_results2/{scaling_folder}/{output_filename}")

    # Find files
    original_path, edited_path, gt_depth_path = find_files(input_folder)

    if not original_path:
        raise FileNotFoundError("Could not find original RGB image")
    if not edited_path:
        raise FileNotFoundError("Could not find edited image (looking for 'edit' in filename)")
    if not gt_depth_path:
        raise FileNotFoundError("Could not find ground truth depth")

    original_path = os.path.abspath(original_path)
    edited_path = os.path.abspath(edited_path)
    gt_depth_path = os.path.abspath(gt_depth_path)

    print(f"\nOriginal: {os.path.basename(original_path)}")
    print(f"Edited:   {os.path.basename(edited_path)}")
    print(f"GT Depth: {os.path.basename(gt_depth_path)}")

    # Load images
    original_img = load_image(original_path)
    edited_img = load_image(edited_path)
    gt_depth = load_exr_depth(gt_depth_path)

    print(f"\nOriginal size: {original_img.size}")
    print(f"Edited size:   {edited_img.size}")
    print(f"GT depth shape: {gt_depth.shape}")
    print(f"GT depth range: {gt_depth.min():.2f}m - {gt_depth.max():.2f}m")

    # Run model on EDITED image only
    print("\n" + "-" * 50)
    print(f"Running {model_name} on EDITED image...")

    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        output_edited = f.name

    try:
        output = run_model_subprocess(model_key, edited_path, output_edited)
        print(f"  {output}")
        depth_edited = np.load(output_edited)
    finally:
        if os.path.exists(output_edited):
            os.remove(output_edited)

    # Resize prediction to match GT if needed
    target_shape = gt_depth.shape
    if depth_edited.shape != target_shape:
        depth_edited = np.array(Image.fromarray(depth_edited.astype(np.float32)).resize(
            (target_shape[1], target_shape[0]), Image.BILINEAR))

    # Resize edited image for display if needed
    if edited_img.size != original_img.size:
        edited_img = edited_img.resize(original_img.size, Image.BILINEAR)

    # Load pre-computed change mask from change_detection_results/
    print(f"\nLoading change mask (model: {mask_model})...")
    changed_mask = load_change_mask(mask_model, dataset)

    # Resize mask to match GT if needed
    if changed_mask.shape != target_shape:
        changed_mask = np.array(Image.fromarray(changed_mask.astype(np.uint8) * 255).resize(
            (target_shape[1], target_shape[0]), Image.NEAREST)) > 127

    unchanged_mask = ~changed_mask

    # Scale edited prediction using UNCHANGED REGIONS vs GT
    # (This is the key difference from v1, which scales using the original prediction)
    print("\n" + "-" * 50)
    print("Computing scale factor from EDITED prediction (unchanged regions)...")

    valid_mask = unchanged_mask & (gt_depth > 0.1) & (gt_depth < 100) & np.isfinite(depth_edited) & np.isfinite(gt_depth)
    pred_valid = depth_edited[valid_mask].flatten()
    gt_valid = gt_depth[valid_mask].flatten()

    if scaling_method == 'median':
        scale = np.median(gt_valid) / np.median(pred_valid)
        shift = 0.0
        depth_edited_scaled = depth_edited * scale
        print(f"Scaling (median from edited unchanged): scale={scale:.4f}")
    else:
        A = np.vstack([pred_valid, np.ones_like(pred_valid)]).T
        result = np.linalg.lstsq(A, gt_valid, rcond=None)
        scale, shift = result[0]
        depth_edited_scaled = depth_edited * scale + shift
        print(f"Scaling (least_squares from edited unchanged): scale={scale:.4f}, shift={shift:.4f}")

    # Metrics: how well does scaled edited prediction match GT in unchanged regions?
    edit_unchanged = depth_edited_scaled[unchanged_mask]
    gt_unchanged = gt_depth[unchanged_mask]
    mae_edit_vs_gt = np.mean(np.abs(edit_unchanged - gt_unchanged))
    rmse_edit_vs_gt = np.sqrt(np.mean((edit_unchanged - gt_unchanged) ** 2))
    max_diff = np.max(np.abs(edit_unchanged - gt_unchanged))
    std_diff = np.std(edit_unchanged - gt_unchanged)
    pct_above_01m = np.mean(np.abs(edit_unchanged - gt_unchanged) > 0.1) * 100
    pct_above_05m = np.mean(np.abs(edit_unchanged - gt_unchanged) > 0.5) * 100

    # Statistics
    print("\n" + "=" * 70)
    print(f"DEPTH STATISTICS (after {scaling_folder} scaling)")
    print("=" * 70)
    print(f"\nGT Depth:       min={gt_depth.min():.2f}m, max={gt_depth.max():.2f}m, median={np.median(gt_depth):.2f}m")
    print(f"Pred Edited:    min={depth_edited_scaled.min():.2f}m, max={depth_edited_scaled.max():.2f}m, median={np.median(depth_edited_scaled):.2f}m")

    print(f"\nEdited vs GT (unchanged regions, {unchanged_mask.sum()} pixels, {unchanged_mask.mean()*100:.1f}%):")
    print(f"  MAE:              {mae_edit_vs_gt:.4f}m")
    print(f"  RMSE:             {rmse_edit_vs_gt:.4f}m")
    print(f"  Max diff:         {max_diff:.4f}m")
    print(f"  Std dev:          {std_diff:.4f}m")
    print(f"  % diff > 0.1m:    {pct_above_01m:.2f}%")
    print(f"  % diff > 0.5m:    {pct_above_05m:.2f}%")

    # Save metrics JSON
    metrics_data = {
        'Scale factor': float(scale),
        'Edit vs GT (m)': float(mae_edit_vs_gt),
        'RMSE (m)': float(rmse_edit_vs_gt),
        'Max diff (m)': float(max_diff),
        'Std dev (m)': float(std_diff),
        '% diff > 0.1m': float(pct_above_01m),
        '% diff > 0.5m': float(pct_above_05m),
    }

    metrics_path = os.path.join(output_subfolder, f"{model_key}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")

    # Create visualization (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    plt.subplots_adjust(hspace=0.15, wspace=0.05)

    # Row 1: Images
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(edited_img)
    axes[0, 1].set_title('Edited Image', fontsize=12)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(changed_mask, cmap='gray')
    axes[0, 2].set_title(f'Changed Regions ({mask_model})', fontsize=12)
    axes[0, 2].axis('off')

    # Row 2: Depth maps + error
    vmin_depth = gt_depth.min()
    vmax_depth = gt_depth.max()

    im0 = axes[1, 0].imshow(gt_depth, cmap='turbo', vmin=vmin_depth, vmax=vmax_depth)
    axes[1, 0].set_title('Ground Truth Depth', fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.04, label='meters')

    im1 = axes[1, 1].imshow(depth_edited_scaled, cmap='turbo', vmin=vmin_depth, vmax=vmax_depth)
    axes[1, 1].set_title(f'{model_name}\n(Edited, Scaled)', fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04, label='meters')

    from scipy import ndimage
    unchanged_mask_clean = ndimage.binary_erosion(unchanged_mask, iterations=2)
    error_edit_signed = depth_edited_scaled - gt_depth
    error_edit_masked = np.where(unchanged_mask_clean, error_edit_signed, np.nan)
    error_range = 0.5
    im2 = axes[1, 2].imshow(error_edit_masked, cmap='PuOr', vmin=-error_range, vmax=error_range)
    axes[1, 2].set_title(f'Error: Edited - GT\n(Unchanged, MAE={mae_edit_vs_gt:.3f}m)', fontsize=12)
    axes[1, 2].axis('off')
    plt.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04, label='meters')

    plt.suptitle(f'Scaled Depth: {model_name} ({scaling_folder}, mask: {mask_model})', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")

    if not args.no_show:
        plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
