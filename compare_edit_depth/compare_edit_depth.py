"""
Compare Depth Predictions: Original vs Edited Image
====================================================
Compares depth model predictions between the original GT image and an edited
version to visualize how edits affect depth estimation.

Output visualization shows:
1. Ground Truth depth
2. Depth prediction on original image
3. Depth prediction on edited image
4. Difference between predictions (2) and (3)
"""

import os
import sys
import subprocess
import tempfile
import argparse
import numpy as np
import pandas as pd
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
    'depth_anything': 'Depth Anything V2 Metric',
    'depth_pro': 'Depth Pro',
    'metric3d': 'Metric3D v2',
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

    if model_name == 'depth_anything':
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

    elif model_name == 'depth_pro':
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

    elif model_name == 'metric3d':
        script = f'''
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
torch.set_grad_enabled(False)
{exr_loader}
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load("yvanyin/metric3d", "metric3d_vit_small", pretrain=True)
model = model.to(device).eval()

img = load_image("{rgb_path_safe}")
max_size = 800
w, h = img.size
if max(w, h) > max_size:
    scale = max_size / max(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img_tensor = transform(img).unsqueeze(0).to(device)
pred_depth, _, _ = model.inference({{"input": img_tensor}})
depth = pred_depth.squeeze().cpu().numpy()
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
        timeout=300
    )

    if result.returncode != 0:
        raise RuntimeError(f"Subprocess failed: {result.stderr}")

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
    parser = argparse.ArgumentParser(description='Compare depth predictions: original vs edited image')
    parser.add_argument('--model', type=str, default='depth_pro',
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
    scaling_prefix = 'med' if scaling_method == 'median' else 'ls'
    dataset = args.dataset
    input_folder = os.path.join(PROJECT_ROOT, "data", dataset)

    # Create output subfolder: compare_edit_depth/{dataset}_results/{scaling}/
    output_subfolder = os.path.join(OUTPUT_FOLDER, f"{dataset}_results", scaling_folder)
    os.makedirs(output_subfolder, exist_ok=True)

    # Auto-generate output filename: {med/ls}_cmp_{model_key}.png
    output_filename = f"{scaling_prefix}_cmp_{model_key}.png"
    output_path = os.path.join(output_subfolder, output_filename)

    print("=" * 70)
    print("DEPTH COMPARISON: ORIGINAL vs EDITED IMAGE")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Scaling: {scaling_folder}")
    print(f"Dataset: {dataset}")
    print(f"Mask model: {args.mask_model}")
    print(f"Output: {dataset}_results/{scaling_folder}/{output_filename}")

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

    # Run model on both images
    print("\n" + "-" * 50)
    print(f"Running {model_name} on ORIGINAL image...")

    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        output_original = f.name

    try:
        output = run_model_subprocess(model_key, original_path, output_original)
        print(f"  {output}")
        depth_original = np.load(output_original)
    finally:
        if os.path.exists(output_original):
            os.remove(output_original)

    print(f"\nRunning {model_name} on EDITED image...")

    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        output_edited = f.name

    try:
        output = run_model_subprocess(model_key, edited_path, output_edited)
        print(f"  {output}")
        depth_edited = np.load(output_edited)
    finally:
        if os.path.exists(output_edited):
            os.remove(output_edited)

    # Resize predictions to match GT if needed
    target_shape = gt_depth.shape
    if depth_original.shape != target_shape:
        depth_original = np.array(Image.fromarray(depth_original.astype(np.float32)).resize(
            (target_shape[1], target_shape[0]), Image.BILINEAR))

    if depth_edited.shape != target_shape:
        depth_edited = np.array(Image.fromarray(depth_edited.astype(np.float32)).resize(
            (target_shape[1], target_shape[0]), Image.BILINEAR))

    # Resize edited image for display if needed
    if edited_img.size != original_img.size:
        edited_img = edited_img.resize(original_img.size, Image.BILINEAR)

    # Scale predictions to GT (computed from ORIGINAL only)
    # This ensures fair comparison: same scale factor applied to both predictions
    mask = (gt_depth > 0.1) & (gt_depth < 100) & np.isfinite(depth_original) & np.isfinite(gt_depth)
    pred_valid = depth_original[mask].flatten()
    gt_valid = gt_depth[mask].flatten()

    if scaling_method == 'median':
        # Median scaling: scale only, no shift
        scale = np.median(gt_valid) / np.median(pred_valid)
        shift = 0.0
        depth_original_scaled = depth_original * scale
        depth_edited_scaled = depth_edited * scale
        print(f"\nScaling (median from original): scale={scale:.4f}")
    else:
        # Least squares scaling: scale + shift
        A = np.vstack([pred_valid, np.ones_like(pred_valid)]).T
        result = np.linalg.lstsq(A, gt_valid, rcond=None)
        scale, shift = result[0]
        depth_original_scaled = depth_original * scale + shift
        depth_edited_scaled = depth_edited * scale + shift
        print(f"\nScaling (least_squares from original): scale={scale:.4f}, shift={shift:.4f}")

    # Compute difference on scaled predictions
    depth_diff = depth_edited_scaled - depth_original_scaled

    # Load pre-computed change mask from change_detection_results/
    mask_model = args.mask_model
    print(f"\nLoading change mask (model: {mask_model})...")
    changed_mask = load_change_mask(mask_model, dataset)

    # Resize mask to match GT if needed
    if changed_mask.shape != gt_depth.shape:
        changed_mask = np.array(Image.fromarray(changed_mask.astype(np.uint8) * 255).resize(
            (gt_depth.shape[1], gt_depth.shape[0]), Image.NEAREST)) > 127

    unchanged_mask = ~changed_mask

    # Metrics for unchanged regions only
    depth_diff_unchanged = depth_diff[unchanged_mask]
    abs_diff_unchanged = np.abs(depth_diff_unchanged)

    mae_unchanged = np.mean(abs_diff_unchanged)
    rmse_unchanged = np.sqrt(np.mean(depth_diff_unchanged ** 2))
    max_diff_unchanged = np.max(abs_diff_unchanged)
    std_unchanged = np.std(depth_diff_unchanged)
    pct_above_01m = np.mean(abs_diff_unchanged > 0.1) * 100
    pct_above_05m = np.mean(abs_diff_unchanged > 0.5) * 100

    # Alignment metrics: how well do predictions match GT in unchanged regions?
    gt_unchanged = gt_depth[unchanged_mask]
    orig_unchanged = depth_original_scaled[unchanged_mask]
    edit_unchanged = depth_edited_scaled[unchanged_mask]

    mae_orig_vs_gt = np.mean(np.abs(orig_unchanged - gt_unchanged))
    mae_edit_vs_gt = np.mean(np.abs(edit_unchanged - gt_unchanged))
    rmse_edit_vs_gt = np.sqrt(np.mean((edit_unchanged - gt_unchanged) ** 2))

    # Statistics
    print("\n" + "=" * 70)
    print(f"DEPTH STATISTICS (after {scaling_folder} scaling)")
    print("=" * 70)
    print(f"\nGT Depth:       min={gt_depth.min():.2f}m, max={gt_depth.max():.2f}m, median={np.median(gt_depth):.2f}m")
    print(f"Pred Original:  min={depth_original_scaled.min():.2f}m, max={depth_original_scaled.max():.2f}m, median={np.median(depth_original_scaled):.2f}m")
    print(f"Pred Edited:    min={depth_edited_scaled.min():.2f}m, max={depth_edited_scaled.max():.2f}m, median={np.median(depth_edited_scaled):.2f}m")

    print(f"\nUnchanged regions ({unchanged_mask.sum()} pixels, {unchanged_mask.mean()*100:.1f}% of image):")
    print(f"  MAE:              {mae_unchanged:.4f}m")
    print(f"  RMSE:             {rmse_unchanged:.4f}m")
    print(f"  Max diff:         {max_diff_unchanged:.4f}m")
    print(f"  Std dev:          {std_unchanged:.4f}m")
    print(f"  % diff > 0.1m:    {pct_above_01m:.2f}%")
    print(f"  % diff > 0.5m:    {pct_above_05m:.2f}%")

    print(f"\nAlignment to GT (unchanged regions):")
    print(f"  Original vs GT MAE:  {mae_orig_vs_gt:.4f}m")
    print(f"  Edited vs GT MAE:    {mae_edit_vs_gt:.4f}m")
    print(f"  Edited vs GT RMSE:   {rmse_edit_vs_gt:.4f}m")
    print(f"  Drift from GT:       {mae_edit_vs_gt - mae_orig_vs_gt:+.4f}m")

    print(f"\nOverall Difference (Edited - Original):")
    print(f"  Min:    {depth_diff.min():.4f}m")
    print(f"  Max:    {depth_diff.max():.4f}m")
    print(f"  Mean:   {np.mean(depth_diff):.4f}m")
    print(f"  Std:    {np.std(depth_diff):.4f}m")
    print(f"  MAE:    {np.mean(np.abs(depth_diff)):.4f}m")

    # Update metrics summary markdown file (in scaling subfolder)
    metrics_md_path = os.path.join(output_subfolder, "metrics_summary.md")
    metrics_json_path = os.path.join(output_subfolder, "metrics_data.json")

    metrics_data = {
        'Scale factor': float(scale),
        'Orig vs GT (m)': float(mae_orig_vs_gt),
        'Edit vs GT (m)': float(mae_edit_vs_gt),
        'Edit vs GT RMSE (m)': float(rmse_edit_vs_gt),
        'MAE (m)': float(mae_unchanged),
        'RMSE (m)': float(rmse_unchanged),
        'Max diff (m)': float(max_diff_unchanged),
        'Std dev (m)': float(std_unchanged),
        '% diff > 0.1m': float(pct_above_01m),
        '% diff > 0.5m': float(pct_above_05m),
    }

    # Load existing data or create new
    import json
    if os.path.exists(metrics_json_path):
        with open(metrics_json_path, 'r') as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    # Update data for this model
    all_metrics[model_key] = metrics_data

    # Save JSON (for data persistence)
    with open(metrics_json_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # Generate markdown file
    models_order = [m for m in AVAILABLE_MODELS.keys() if m in all_metrics]
    metric_names = list(metrics_data.keys())

    if scaling_method == 'median':
        scaling_desc = "**median scaling** (scale only, no shift)"
    else:
        scaling_desc = "**least-squares scaling** (scale + shift)"

    md_content = f"""# Depth Consistency Metrics: Original vs Edited Image

## Methodology

### Scaling Method: {scaling_folder.upper()}
All model predictions are aligned to ground truth using {scaling_desc},
computed from the **original image only**. The same scale factor is then applied to both original
and edited predictions. This ensures fair comparison across models with different output scales.

### Unchanged Regions
A pixel is considered **unchanged** if the mean absolute RGB difference between the original
and edited image is **<= 10** (on a 0-255 scale). This threshold identifies regions where
the edit did not visually modify the image.

### What We're Measuring
For a perfectly consistent depth model, unchanged regions should produce **identical depth
predictions** regardless of edits elsewhere in the image. Lower values = better consistency.

## Metrics Comparison

"""

    # Build table header
    header = "| Metric |"
    separator = "|--------|"
    for model in models_order:
        header += f" {model} |"
        separator += "--------|"

    md_content += header + "\n" + separator + "\n"

    # Build table rows
    for metric in metric_names:
        row = f"| {metric} |"
        for model in models_order:
            value = all_metrics[model].get(metric, None)
            if value is None:
                row += " - |"
                continue
            if '%' in metric:
                row += f" {value:.2f} |"
            elif 'Scale' in metric:
                row += f" {value:.4f} |"
            else:
                row += f" {value:.4f} |"
        md_content += row + "\n"

    md_content += f"""
## Interpretation

- **Scale factor**: Least-squares scale applied to align model output to GT (closer to 1.0 = better native metric accuracy).
- **Orig vs GT (m)**: MAE between scaled original prediction and GT in unchanged regions (baseline accuracy).
- **Edit vs GT (m)**: MAE between scaled edited prediction and GT in unchanged regions. If higher than Orig vs GT, the edit caused accuracy drift.
- **MAE (m)**: Mean Absolute Error between original and edited predictions in unchanged regions. Lower = more consistent.
- **RMSE (m)**: Root Mean Square Error. More sensitive to large errors.
- **Max diff (m)**: Worst-case depth difference in unchanged regions.
- **Std dev (m)**: Standard deviation of depth differences.
- **% diff > 0.1m / 0.5m**: Percentage of unchanged pixels with depth differences exceeding threshold.

---
*Last updated: {model_key}*
"""

    with open(metrics_md_path, 'w') as f:
        f.write(md_content)

    print(f"\nMetrics summary updated: {metrics_md_path}")

    # Create visualization (3x3 layout) with tight spacing
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
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

    # Row 2: Depth maps
    # Use FIXED colormap range based on GT for consistent visualization across all models
    vmin_depth = gt_depth.min()
    vmax_depth = gt_depth.max()

    im0 = axes[1, 0].imshow(gt_depth, cmap='turbo', vmin=vmin_depth, vmax=vmax_depth)
    axes[1, 0].set_title('Ground Truth Depth', fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.04, label='meters')

    im1 = axes[1, 1].imshow(depth_original_scaled, cmap='turbo', vmin=vmin_depth, vmax=vmax_depth)
    axes[1, 1].set_title(f'{model_name}\n(Original Image)', fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04, label='meters')

    im2 = axes[1, 2].imshow(depth_edited_scaled, cmap='turbo', vmin=vmin_depth, vmax=vmax_depth)
    axes[1, 2].set_title(f'{model_name}\n(Edited Image)', fontsize=12)
    axes[1, 2].axis('off')
    plt.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04, label='meters')

    # Row 3: Analysis maps
    # Depth difference masked to unchanged regions only (erode mask to remove edge artifacts)
    from scipy import ndimage
    # Erode the unchanged mask slightly to remove edge artifacts from resizing/compression
    unchanged_mask_clean = ndimage.binary_erosion(unchanged_mask, iterations=2)
    depth_diff_masked = np.where(unchanged_mask_clean, depth_diff, np.nan)
    diff_range = 0.5  # Fixed +/- 0.5m range for difference visualization
    im3 = axes[2, 0].imshow(depth_diff_masked, cmap='RdBu_r', vmin=-diff_range, vmax=diff_range)
    axes[2, 0].set_title('Depth Diff (Unchanged Only)\n(Edited - Original)', fontsize=12)
    axes[2, 0].axis('off')
    plt.colorbar(im3, ax=axes[2, 0], fraction=0.046, pad=0.04, label='meters')

    # Signed Error: Original vs GT (masked to unchanged regions only)
    error_orig_signed = depth_original_scaled - gt_depth
    error_orig_masked = np.where(unchanged_mask_clean, error_orig_signed, np.nan)
    error_range = 0.5  # Fixed +/- range for error visualization
    im4 = axes[2, 1].imshow(error_orig_masked, cmap='PuOr', vmin=-error_range, vmax=error_range)
    axes[2, 1].set_title(f'Error: Original - GT\n(Unchanged, MAE={mae_orig_vs_gt:.3f}m)', fontsize=12)
    axes[2, 1].axis('off')
    plt.colorbar(im4, ax=axes[2, 1], fraction=0.046, pad=0.04, label='meters')

    # Signed Error: Edited vs GT (masked to unchanged regions only)
    error_edit_signed = depth_edited_scaled - gt_depth
    error_edit_masked = np.where(unchanged_mask_clean, error_edit_signed, np.nan)
    im5 = axes[2, 2].imshow(error_edit_masked, cmap='PuOr', vmin=-error_range, vmax=error_range)
    axes[2, 2].set_title(f'Error: Edited - GT\n(Unchanged, MAE={mae_edit_vs_gt:.3f}m)', fontsize=12)
    axes[2, 2].axis('off')
    plt.colorbar(im5, ax=axes[2, 2], fraction=0.046, pad=0.04, label='meters')

    plt.suptitle(f'Depth Comparison: Original vs Edited ({model_name}, mask: {mask_model})', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Always save to output folder
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")

    if not args.no_show:
        plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
