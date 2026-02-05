"""
Comprehensive Depth Model Comparison Script
============================================
Compares Depth Anything V2 Metric, Depth Pro, and Metric3D v2
on the same test dataset with ground truth evaluation.

Each model runs in a separate subprocess to prevent crashes from
memory corruption when switching between incompatible model libraries.
"""

import os
import sys
import subprocess
import tempfile
import numpy as np
from PIL import Image
import OpenEXR
import Imath

# Configuration
DATASET = "depth4"
INPUT_FOLDER = f"./data/{DATASET}"
GT_TO_CENTIMETERS = 10000.0  # For SceneDepth files
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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
    """Find RGB and depth files in folder."""
    files = os.listdir(folder)

    # Find RGB: prefer base EXR (no 'depth' or 'scenedepth' in name)
    rgb_file = None
    for f in files:
        if f.endswith('.exr') and 'depth' not in f.lower() and 'scenedepth' not in f.lower():
            rgb_file = os.path.join(folder, f)
            break

    # Fallback to PNG if no suitable EXR
    if not rgb_file:
        for f in files:
            if f.endswith('.png') and 'depth' not in f.lower() and 'edit' not in f.lower():
                rgb_file = os.path.join(folder, f)
                break

    # Find depth GT: prefer SceneDepth EXR (not WorldUnits)
    depth_file = None
    for f in files:
        if 'SceneDepth' in f and 'WorldUnits' not in f and f.endswith('.exr'):
            depth_file = os.path.join(folder, f)
            break

    return rgb_file, depth_file


def compute_metrics(pred, gt, mask=None):
    """Compute depth estimation metrics."""
    if mask is None:
        mask = (gt > 0.1) & (gt < 100) & np.isfinite(pred) & np.isfinite(gt)

    pred_valid = pred[mask]
    gt_valid = gt[mask]

    if len(pred_valid) == 0:
        return {}

    rmse = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))
    mae = np.mean(np.abs(pred_valid - gt_valid))
    rel_err = np.mean(np.abs(pred_valid - gt_valid) / gt_valid) * 100
    ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
    delta1 = np.mean(ratio < 1.25) * 100

    return {'rmse': rmse, 'mae': mae, 'rel_err': rel_err, 'delta1': delta1}


def scale_prediction(pred, gt, method='least_squares'):
    """Scale prediction to match GT."""
    mask = (gt > 0.1) & (gt < 100) & np.isfinite(pred) & np.isfinite(gt)
    pred_valid = pred[mask].flatten()
    gt_valid = gt[mask].flatten()

    if method == 'none':
        return pred, 1.0, 0.0
    elif method == 'median':
        scale = np.median(gt_valid) / np.median(pred_valid)
        return pred * scale, scale, 0.0
    elif method == 'least_squares':
        A = np.vstack([pred_valid, np.ones_like(pred_valid)]).T
        result = np.linalg.lstsq(A, gt_valid, rcond=None)
        scale, shift = result[0]
        return pred * scale + shift, scale, shift


def run_model_subprocess(model_name, rgb_path, output_path):
    """Run a depth model in a separate subprocess to avoid memory conflicts."""
    # Convert Windows paths to forward slashes for Python string safety
    rgb_path_safe = rgb_path.replace('\\', '/')
    output_path_safe = output_path.replace('\\', '/')

    # Common EXR loading code
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


def main():
    print("=" * 70)
    print("COMPREHENSIVE DEPTH MODEL COMPARISON")
    print("=" * 70)
    print(f"Dataset: {DATASET}")
    print("(Each model runs in isolated subprocess to prevent crashes)")

    # Find files
    rgb_path, depth_path = find_files(INPUT_FOLDER)
    rgb_path = os.path.abspath(rgb_path)
    depth_path = os.path.abspath(depth_path)
    print(f"\nRGB: {os.path.basename(rgb_path)}")
    print(f"GT:  {os.path.basename(depth_path)}")

    # Load RGB and GT
    rgb_image = load_image(rgb_path)
    gt_depth = load_exr_depth(depth_path)
    print(f"\nImage size: {rgb_image.size}")
    print(f"GT depth shape: {gt_depth.shape}")
    print(f"GT depth range: {gt_depth.min():.2f}m - {gt_depth.max():.2f}m, median={np.median(gt_depth):.2f}m")

    results = {}

    models = [
        ('depth_anything', 'Depth Anything V2'),
        ('depth_pro', 'Depth Pro'),
        ('metric3d', 'Metric3D v2'),
    ]

    for model_key, model_name in models:
        print("\n" + "-" * 50)
        print(f"Running {model_name} (subprocess)...")

        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            output_path = f.name

        try:
            # Run model in subprocess
            output = run_model_subprocess(model_key, rgb_path, output_path)
            print(f"  {output}")

            # Load result
            depth = np.load(output_path)

            # Resize to match GT
            depth_resized = np.array(Image.fromarray(depth.astype(np.float32)).resize(
                (gt_depth.shape[1], gt_depth.shape[0]), Image.BILINEAR))

            print(f"  Resized: {depth_resized.shape}, median={np.median(depth_resized):.2f}m")
            results[model_name] = depth_resized

        except Exception as e:
            print(f"  Error: {e}")

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    # Compare all models
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(f"\nGround Truth median: {np.median(gt_depth):.2f}m")

    scaling_methods = ['none', 'median', 'least_squares']

    all_results = []

    for model_name, pred_depth in results.items():
        print(f"\n{model_name}:")
        print(f"  Raw output median: {np.median(pred_depth):.2f}m")

        for method in scaling_methods:
            scaled, scale, shift = scale_prediction(pred_depth, gt_depth, method)
            metrics = compute_metrics(scaled, gt_depth)

            if method == 'none':
                print(f"  [{method:12}] RMSE={metrics['rmse']:.4f}m  MAE={metrics['mae']:.4f}m  RelErr={metrics['rel_err']:.1f}%  d1={metrics['delta1']:.1f}%")
            elif method == 'median':
                print(f"  [{method:12}] RMSE={metrics['rmse']:.4f}m  MAE={metrics['mae']:.4f}m  RelErr={metrics['rel_err']:.1f}%  d1={metrics['delta1']:.1f}%  (scale={scale:.4f})")
            else:
                print(f"  [{method:12}] RMSE={metrics['rmse']:.4f}m  MAE={metrics['mae']:.4f}m  RelErr={metrics['rel_err']:.1f}%  d1={metrics['delta1']:.1f}%  (scale={scale:.4f}, shift={shift:.4f})")

            all_results.append({
                'model': model_name,
                'method': method,
                **metrics,
                'scale': scale,
                'shift': shift
            })

    # Find best
    print("\n" + "=" * 70)
    print("WINNER ANALYSIS")
    print("=" * 70)

    if all_results:
        best = min(all_results, key=lambda x: x['rmse'])
        print(f"\nBest overall (by RMSE):")
        print(f"  Model:  {best['model']}")
        print(f"  Method: {best['method']}")
        print(f"  RMSE:   {best['rmse']:.4f}m")
        print(f"  MAE:    {best['mae']:.4f}m")
        print(f"  RelErr: {best['rel_err']:.2f}%")
        print(f"  Delta1: {best['delta1']:.1f}%")

    # Final comparison table
    print("\n" + "=" * 70)
    print("FINAL COMPARISON TABLE (least_squares scaling)")
    print("=" * 70)
    print(f"\n{'Model':<20} {'RMSE':>8} {'MAE':>8} {'RelErr':>8} {'Delta1':>8} {'Scale':>8}")
    print("-" * 70)

    for r in sorted(all_results, key=lambda x: x['rmse']):
        if r['method'] == 'least_squares':
            print(f"{r['model']:<20} {r['rmse']:>7.4f}m {r['mae']:>7.4f}m {r['rel_err']:>7.1f}% {r['delta1']:>7.1f}% {r['scale']:>8.4f}")


if __name__ == "__main__":
    main()
