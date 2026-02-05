"""
Composite Depth Map Generator
=============================
Creates a composite depth map where:
- Unchanged regions: Use GT depth directly
- Changed regions: Use scaled model prediction (scale computed from unchanged regions)

This differs from compare_edit_depth.py which measures model consistency.
This script generates an aligned depth output suitable for downstream tasks.
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
import torch

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
depth = np.array(result["depth"])
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


def refine_mask(mask, min_area=500, dilate_iter=2):
    """
    Refine binary mask by removing small regions and smoothing.
    """
    from scipy import ndimage
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


def dinov2_feature_mask(img1, img2, threshold=0.35, model_name="facebook/dinov2-base"):
    """
    DINOv2 feature-based change detection.
    Extracts patch features and computes cosine distance.
    Returns mask where True = changed region.
    """
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading DINOv2 for semantic change detection...")

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Process images
    with torch.no_grad():
        inputs1 = processor(images=img1, return_tensors="pt").to(device)
        inputs2 = processor(images=img2, return_tensors="pt").to(device)

        # Get patch features (excluding CLS token)
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

        # last_hidden_state shape: [batch, num_patches+1, hidden_dim]
        features1 = outputs1.last_hidden_state[:, 1:, :]
        features2 = outputs2.last_hidden_state[:, 1:, :]

        # Compute cosine similarity per patch
        features1_norm = features1 / features1.norm(dim=-1, keepdim=True)
        features2_norm = features2 / features2.norm(dim=-1, keepdim=True)
        similarity = (features1_norm * features2_norm).sum(dim=-1)

        # Convert to distance (1 - similarity)
        distance = 1 - similarity.squeeze().cpu().numpy()

    # Reshape to 2D patch grid
    num_patches = distance.shape[0]
    grid_size = int(np.sqrt(num_patches))
    distance_map = distance.reshape(grid_size, grid_size)

    # Upsample to original image size
    h, w = np.array(img1).shape[:2]
    distance_map_full = np.array(Image.fromarray(distance_map.astype(np.float32)).resize(
        (w, h), Image.BILINEAR))

    # Threshold to get binary mask
    changed_mask = distance_map_full > threshold

    return changed_mask


def create_composite_depth(gt_depth, pred_scaled, unchanged_mask):
    """
    Create composite depth map:
    - Unchanged regions: Use GT depth directly
    - Changed regions: Use scaled prediction

    Args:
        gt_depth: Ground truth depth (H, W)
        pred_scaled: Scaled model prediction (H, W)
        unchanged_mask: Boolean mask where True = unchanged (H, W)

    Returns:
        composite_depth: Blended depth map (H, W)
    """
    composite_depth = np.where(unchanged_mask, gt_depth, pred_scaled)
    return composite_depth


def main():
    parser = argparse.ArgumentParser(description='Generate composite depth map from GT and model prediction')
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
    parser.add_argument('--rgb-only', action='store_true',
                        help='Use RGB threshold only (skip DINOv2 for faster processing)')
    args = parser.parse_args()

    model_key = args.model
    model_name = AVAILABLE_MODELS[model_key]
    scaling_method = args.scaling
    scaling_folder = 'median' if scaling_method == 'median' else 'least_squares'
    dataset = args.dataset
    input_folder = os.path.join(PROJECT_ROOT, "data", dataset)

    # Create output subfolder: compare_edit_depth/{dataset}_results2/{scaling}/
    output_subfolder = os.path.join(OUTPUT_FOLDER, f"{dataset}_results2", scaling_folder)
    os.makedirs(output_subfolder, exist_ok=True)

    print("=" * 70)
    print("COMPOSITE DEPTH MAP GENERATOR")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Scaling: {scaling_folder}")
    print(f"Dataset: {dataset}")
    print(f"Output folder: {dataset}_results2/{scaling_folder}/")

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

    # Run model on edited image ONLY
    print("\n" + "-" * 50)
    print(f"Running {model_name} on EDITED image...")

    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        output_edited = f.name

    try:
        output = run_model_subprocess(model_key, edited_path, output_edited)
        print(f"  {output}")
        depth_pred = np.load(output_edited)
    finally:
        if os.path.exists(output_edited):
            os.remove(output_edited)

    # Resize prediction to match GT if needed
    target_shape = gt_depth.shape
    if depth_pred.shape != target_shape:
        depth_pred = np.array(Image.fromarray(depth_pred.astype(np.float32)).resize(
            (target_shape[1], target_shape[0]), Image.BILINEAR))

    # Resize edited image for display if needed
    if edited_img.size != original_img.size:
        edited_img = edited_img.resize(original_img.size, Image.BILINEAR)

    # Compute change mask using intersection of RGB and DINOv2
    orig_arr = np.array(original_img).astype(float)
    edit_arr = np.array(edited_img).astype(float)
    img_diff = np.abs(orig_arr - edit_arr).mean(axis=2)

    # Method 1: RGB threshold
    rgb_threshold = 25  # pixel difference threshold
    rgb_changed_mask = img_diff > rgb_threshold
    rgb_changed_mask = refine_mask(rgb_changed_mask.copy(), min_area=500, dilate_iter=2)

    if args.rgb_only:
        # RGB-only mode (faster)
        changed_mask = rgb_changed_mask
        print(f"\nUsing RGB-only mask: {changed_mask.mean()*100:.1f}% changed")
    else:
        # Method 2: DINOv2 semantic features
        print("\nComputing semantic change mask...")
        dino_changed_mask = dinov2_feature_mask(original_img, edited_img, threshold=0.35)
        dino_changed_mask = refine_mask(dino_changed_mask.copy(), min_area=500, dilate_iter=2)

        # Intersection: high-confidence changes (both methods agree)
        changed_mask = rgb_changed_mask & dino_changed_mask

        print(f"  RGB changed: {rgb_changed_mask.mean()*100:.1f}%")
        print(f"  DINOv2 changed: {dino_changed_mask.mean()*100:.1f}%")
        print(f"  Intersection: {changed_mask.mean()*100:.1f}% (high-confidence)")

    unchanged_mask = ~changed_mask

    # Compute scale factor from unchanged regions
    print("\n" + "-" * 50)
    print("Computing scale factor from unchanged regions...")

    # Valid mask: unchanged regions with valid depth values
    valid_mask = unchanged_mask & (gt_depth > 0.1) & (gt_depth < 100) & np.isfinite(depth_pred) & np.isfinite(gt_depth)
    pred_valid = depth_pred[valid_mask].flatten()
    gt_valid = gt_depth[valid_mask].flatten()

    if scaling_method == 'median':
        # Median scaling: scale only, no shift
        scale = np.median(gt_valid) / np.median(pred_valid)
        shift = 0.0
        pred_scaled = depth_pred * scale
        print(f"Scaling (median): scale={scale:.4f}")
    else:
        # Least squares scaling: scale + shift
        A = np.vstack([pred_valid, np.ones_like(pred_valid)]).T
        result = np.linalg.lstsq(A, gt_valid, rcond=None)
        scale, shift = result[0]
        pred_scaled = depth_pred * scale + shift
        print(f"Scaling (least_squares): scale={scale:.4f}, shift={shift:.4f}")

    # Create composite depth map
    print("\nCreating composite depth map...")
    composite_depth = create_composite_depth(gt_depth, pred_scaled, unchanged_mask)

    # Compute statistics
    n_unchanged = unchanged_mask.sum()
    n_changed = changed_mask.sum()
    n_total = unchanged_mask.size
    pct_unchanged = n_unchanged / n_total * 100
    pct_changed = n_changed / n_total * 100

    # Verify composite construction
    gt_in_unchanged = composite_depth[unchanged_mask]
    gt_ref_unchanged = gt_depth[unchanged_mask]
    unchanged_match = np.allclose(gt_in_unchanged, gt_ref_unchanged)

    pred_in_changed = composite_depth[changed_mask]
    pred_ref_changed = pred_scaled[changed_mask]
    changed_match = np.allclose(pred_in_changed, pred_ref_changed)

    print(f"\nComposite depth statistics:")
    print(f"  Shape: {composite_depth.shape}")
    print(f"  Range: {composite_depth.min():.2f}m - {composite_depth.max():.2f}m")
    print(f"  Unchanged pixels: {n_unchanged} ({pct_unchanged:.1f}%)")
    print(f"  Changed pixels: {n_changed} ({pct_changed:.1f}%)")
    print(f"  Unchanged == GT: {unchanged_match}")
    print(f"  Changed == Scaled Pred: {changed_match}")

    # Save outputs
    print("\n" + "-" * 50)
    print("Saving outputs...")

    # Save composite depth as .npy
    depth_output_path = os.path.join(output_subfolder, f"{model_key}_composite_depth.npy")
    np.save(depth_output_path, composite_depth)
    print(f"  Composite depth: {depth_output_path}")

    # Save metrics JSON
    metrics_data = {
        'model': model_key,
        'scaling_method': scaling_method,
        'scale': float(scale),
        'shift': float(shift),
        'n_unchanged': int(n_unchanged),
        'n_changed': int(n_changed),
        'pct_unchanged': float(pct_unchanged),
        'pct_changed': float(pct_changed),
        'depth_min': float(composite_depth.min()),
        'depth_max': float(composite_depth.max()),
        'unchanged_match_gt': bool(unchanged_match),
        'changed_match_pred': bool(changed_match),
    }

    metrics_path = os.path.join(output_subfolder, f"{model_key}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"  Metrics: {metrics_path}")

    # Create visualization (2x2 grid)
    print("\nCreating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    plt.subplots_adjust(hspace=0.2, wspace=0.1)

    # Use consistent colormap range based on GT
    vmin_depth = gt_depth.min()
    vmax_depth = gt_depth.max()

    # Top-left: GT Depth
    im0 = axes[0, 0].imshow(gt_depth, cmap='turbo', vmin=vmin_depth, vmax=vmax_depth)
    axes[0, 0].set_title('Ground Truth Depth', fontsize=12)
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04, label='meters')

    # Top-right: Model Prediction (scaled)
    im1 = axes[0, 1].imshow(pred_scaled, cmap='turbo', vmin=vmin_depth, vmax=vmax_depth)
    axes[0, 1].set_title(f'{model_name} Prediction (Scaled)', fontsize=12)
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04, label='meters')

    # Bottom-left: Composite Output
    im2 = axes[1, 0].imshow(composite_depth, cmap='turbo', vmin=vmin_depth, vmax=vmax_depth)
    axes[1, 0].set_title('Composite Depth Output', fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04, label='meters')

    # Bottom-right: Change Mask Overlay on edited image
    axes[1, 1].imshow(edited_img)
    # Create overlay: red for changed regions
    overlay = np.zeros((*changed_mask.shape, 4))
    overlay[changed_mask] = [1, 0, 0, 0.4]  # Red with alpha
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title(f'Change Mask Overlay\n({pct_changed:.1f}% changed)', fontsize=12)
    axes[1, 1].axis('off')

    plt.suptitle(f'Composite Depth: {model_name} ({scaling_folder})', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save visualization
    viz_output_path = os.path.join(output_subfolder, f"{model_key}_visualization.png")
    plt.savefig(viz_output_path, dpi=150, bbox_inches='tight')
    print(f"  Visualization: {viz_output_path}")

    if not args.no_show:
        plt.show()

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_subfolder}/")
    print(f"  - {model_key}_composite_depth.npy")
    print(f"  - {model_key}_metrics.json")
    print(f"  - {model_key}_visualization.png")


if __name__ == "__main__":
    main()
