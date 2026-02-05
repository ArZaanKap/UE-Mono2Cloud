"""
Change Detection Model Comparison
=================================
Tests different approaches for detecting changes between original and edited images:
1. RGB Threshold - Simple pixel difference baseline
2. DINOv2 - Semantic feature-based difference (HuggingFace)
3. SAM-based - Uses SAM encoder features + mask refinement (inspired by GeSCF CVPR 2025)

This generates masks identifying changed regions (e.g., added objects).
"""

import os
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


def dinov2_feature_mask(img1, img2, threshold=0.3, model_name="facebook/dinov2-base"):
    """
    DINOv2 feature-based change detection.
    Extracts patch features and computes cosine distance.
    Returns mask where True = changed region.
    """
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading DINOv2 model ({model_name})...")

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
        # Remove CLS token (first token)
        features1 = outputs1.last_hidden_state[:, 1:, :]  # [1, num_patches, hidden_dim]
        features2 = outputs2.last_hidden_state[:, 1:, :]

        # Compute cosine similarity per patch
        features1_norm = features1 / features1.norm(dim=-1, keepdim=True)
        features2_norm = features2 / features2.norm(dim=-1, keepdim=True)
        similarity = (features1_norm * features2_norm).sum(dim=-1)  # [1, num_patches]

        # Convert to distance (1 - similarity)
        distance = 1 - similarity.squeeze().cpu().numpy()  # [num_patches]

    # Reshape to 2D patch grid
    # DINOv2 uses 14x14 patches for 224x224 input by default
    num_patches = distance.shape[0]
    grid_size = int(np.sqrt(num_patches))
    distance_map = distance.reshape(grid_size, grid_size)

    # Upsample to original image size
    h, w = np.array(img1).shape[:2]
    distance_map_full = np.array(Image.fromarray(distance_map.astype(np.float32)).resize(
        (w, h), Image.BILINEAR))

    # Threshold to get binary mask
    changed_mask = distance_map_full > threshold

    return changed_mask, distance_map_full


def sam_feature_mask(img1, img2, threshold=0.3, model_name="facebook/sam-vit-base"):
    """
    SAM-based change detection inspired by GeSCF (CVPR 2025).
    Uses SAM's image encoder to extract features and compute differences.
    Then uses SAM's mask decoder with point prompts for refined masks.
    """
    from transformers import SamModel, SamProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading SAM model ({model_name})...")

    processor = SamProcessor.from_pretrained(model_name)
    model = SamModel.from_pretrained(model_name).to(device)
    model.eval()

    h, w = np.array(img1).shape[:2]

    with torch.no_grad():
        # Process both images
        inputs1 = processor(img1, return_tensors="pt").to(device)
        inputs2 = processor(img2, return_tensors="pt").to(device)

        # Get image embeddings from SAM encoder
        # Shape: [batch, 256, 64, 64] for SAM-base
        embeddings1 = model.get_image_embeddings(inputs1["pixel_values"])
        embeddings2 = model.get_image_embeddings(inputs2["pixel_values"])

        # Compute feature difference (L2 distance per spatial location)
        diff = (embeddings1 - embeddings2).pow(2).sum(dim=1).sqrt()  # [1, 64, 64]
        diff = diff.squeeze().cpu().numpy()

        # Normalize to 0-1
        diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

    # Upsample difference map to original size
    diff_map_full = np.array(Image.fromarray(diff.astype(np.float32)).resize(
        (w, h), Image.BILINEAR))

    # Initial mask from threshold
    initial_mask = diff_map_full > threshold

    # Find centroid of changed region for SAM point prompt
    if initial_mask.sum() > 0:
        # Get coordinates of changed pixels
        y_coords, x_coords = np.where(initial_mask)

        # Find the point with maximum difference (most confident change)
        diff_values = diff_map_full[initial_mask]
        max_idx = np.argmax(diff_values)
        point_x, point_y = x_coords[max_idx], y_coords[max_idx]

        # Also get center of mass as backup
        center_y, center_x = int(y_coords.mean()), int(x_coords.mean())

        print(f"  Change detected at ({point_x}, {point_y}), using SAM for refinement...")

        # Use SAM to generate refined mask with point prompt
        with torch.no_grad():
            # Use the edited image for mask generation
            inputs = processor(
                img2,
                input_points=[[[point_x, point_y]]],  # Single point
                return_tensors="pt"
            ).to(device)

            outputs = model(**inputs)

            # Get the best mask (highest IoU prediction)
            masks = processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )

            # Take the mask with highest predicted IoU
            iou_scores = outputs.iou_scores.squeeze().cpu().numpy()
            best_mask_idx = np.argmax(iou_scores)
            refined_mask = masks[0][0, best_mask_idx].numpy()

            print(f"  SAM mask IoU scores: {iou_scores}, selected idx {best_mask_idx}")
    else:
        print("  No changes detected above threshold")
        refined_mask = initial_mask

    return refined_mask, diff_map_full


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


def main():
    parser = argparse.ArgumentParser(description='Test change detection models')
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET,
                        help='Dataset folder name (e.g., depth4)')
    parser.add_argument('--rgb-threshold', type=int, default=25,
                        help='RGB difference threshold (0-255)')
    parser.add_argument('--dino-threshold', type=float, default=0.35,
                        help='DINOv2 feature distance threshold (0-1)')
    parser.add_argument('--sam-threshold', type=float, default=0.3,
                        help='SAM feature distance threshold (0-1)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display figure')
    parser.add_argument('--skip-dino', action='store_true',
                        help='Skip DINOv2 method')
    parser.add_argument('--skip-sam', action='store_true',
                        help='Skip SAM method')
    args = parser.parse_args()

    # Setup paths
    data_folder = os.path.join(PROJECT_ROOT, "data", args.dataset)
    output_folder = SCRIPT_DIR  # Output to same folder as script
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

    results = {}

    # Method 1: RGB Threshold (baseline)
    print("\n" + "-" * 40)
    print("Method 1: RGB Threshold Baseline")
    print("-" * 40)
    rgb_mask, rgb_diff = rgb_threshold_mask(original_img, edited_img, threshold=args.rgb_threshold)
    rgb_mask_refined = refine_mask(rgb_mask.copy(), min_area=100)
    print(f"  Threshold: {args.rgb_threshold}")
    print(f"  Changed pixels: {rgb_mask.sum():,} ({rgb_mask.mean()*100:.1f}%)")
    print(f"  After refinement: {rgb_mask_refined.sum():,} ({rgb_mask_refined.mean()*100:.1f}%)")
    results['rgb'] = {'mask': rgb_mask_refined, 'diff_map': rgb_diff}

    # Method 2: DINOv2 Features
    if not args.skip_dino:
        print("\n" + "-" * 40)
        print("Method 2: DINOv2 Semantic Features")
        print("-" * 40)
        try:
            dino_mask, dino_diff = dinov2_feature_mask(
                original_img, edited_img,
                threshold=args.dino_threshold,
                model_name="facebook/dinov2-base"
            )
            dino_mask_refined = refine_mask(dino_mask.copy(), min_area=500)
            print(f"  Threshold: {args.dino_threshold}")
            print(f"  Changed pixels: {dino_mask.sum():,} ({dino_mask.mean()*100:.1f}%)")
            print(f"  After refinement: {dino_mask_refined.sum():,} ({dino_mask_refined.mean()*100:.1f}%)")
            results['dinov2'] = {'mask': dino_mask_refined, 'diff_map': dino_diff}
        except Exception as e:
            print(f"  Error: {e}")
            results['dinov2'] = None
    else:
        results['dinov2'] = None

    # Method 3: SAM-based (GeSCF-inspired)
    if not args.skip_sam:
        print("\n" + "-" * 40)
        print("Method 3: SAM-based (GeSCF-inspired)")
        print("-" * 40)
        try:
            sam_mask, sam_diff = sam_feature_mask(
                original_img, edited_img,
                threshold=args.sam_threshold,
                model_name="facebook/sam-vit-base"
            )
            sam_mask_refined = refine_mask(sam_mask.copy(), min_area=200, dilate_iter=1)
            print(f"  Threshold: {args.sam_threshold}")
            print(f"  Changed pixels: {sam_mask.sum():,} ({sam_mask.mean()*100:.1f}%)")
            print(f"  After refinement: {sam_mask_refined.sum():,} ({sam_mask_refined.mean()*100:.1f}%)")
            results['sam'] = {'mask': sam_mask_refined, 'diff_map': sam_diff}
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results['sam'] = None
    else:
        results['sam'] = None

    # Compute intersection mask (high-confidence changes)
    if results.get('dinov2'):
        intersection_mask = results['rgb']['mask'] & results['dinov2']['mask']
        results['intersection'] = {'mask': intersection_mask}
        print("\n" + "-" * 40)
        print("Method 4: Intersection (RGB AND DINOv2)")
        print("-" * 40)
        print(f"  Changed pixels: {intersection_mask.sum():,} ({intersection_mask.mean()*100:.1f}%)")
        print("  (High-confidence: both methods agree)")

    # Compare masks
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    if results.get('dinov2'):
        metrics = compute_mask_metrics(results['rgb']['mask'], results['dinov2']['mask'])
        print(f"RGB vs DINOv2 IoU: {metrics['iou']:.3f}")

    if results.get('sam'):
        metrics = compute_mask_metrics(results['rgb']['mask'], results['sam']['mask'])
        print(f"RGB vs SAM IoU: {metrics['iou']:.3f}")

    if results.get('dinov2') and results.get('sam'):
        metrics = compute_mask_metrics(results['dinov2']['mask'], results['sam']['mask'])
        print(f"DINOv2 vs SAM IoU: {metrics['iou']:.3f}")

    # Visualization - 3 rows now
    num_methods = 1 + (1 if results.get('dinov2') else 0) + (1 if results.get('sam') else 0)
    fig, axes = plt.subplots(num_methods, 4, figsize=(16, 4 * num_methods))
    if num_methods == 1:
        axes = axes.reshape(1, -1)
    plt.subplots_adjust(hspace=0.25, wspace=0.1)

    row = 0

    # Row 1: RGB method
    axes[row, 0].imshow(original_img)
    axes[row, 0].set_title('Original Image')
    axes[row, 0].axis('off')

    axes[row, 1].imshow(edited_img)
    axes[row, 1].set_title('Edited Image')
    axes[row, 1].axis('off')

    axes[row, 2].imshow(results['rgb']['diff_map'], cmap='hot')
    axes[row, 2].set_title(f'RGB Difference Map\n(threshold={args.rgb_threshold})')
    axes[row, 2].axis('off')

    axes[row, 3].imshow(results['rgb']['mask'], cmap='gray')
    axes[row, 3].set_title(f'RGB Mask\n({results["rgb"]["mask"].mean()*100:.1f}% changed)')
    axes[row, 3].axis('off')
    row += 1

    # Row 2: DINOv2 method
    if results.get('dinov2'):
        axes[row, 0].imshow(results['dinov2']['diff_map'], cmap='hot', vmin=0, vmax=0.5)
        axes[row, 0].set_title(f'DINOv2 Feature Distance')
        axes[row, 0].axis('off')

        axes[row, 1].imshow(results['dinov2']['mask'], cmap='gray')
        axes[row, 1].set_title(f'DINOv2 Mask\n({results["dinov2"]["mask"].mean()*100:.1f}% changed)')
        axes[row, 1].axis('off')

        # Overlay on image
        overlay = np.array(edited_img).copy()
        overlay[results['dinov2']['mask']] = [0, 255, 0]
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title('DINOv2 Overlay')
        axes[row, 2].axis('off')

        # Comparison with RGB
        combined = np.zeros((*results['rgb']['mask'].shape, 3), dtype=np.uint8)
        combined[results['rgb']['mask'] & results['dinov2']['mask']] = [255, 255, 0]
        combined[results['rgb']['mask'] & ~results['dinov2']['mask']] = [255, 0, 0]
        combined[~results['rgb']['mask'] & results['dinov2']['mask']] = [0, 255, 0]
        axes[row, 3].imshow(combined)
        axes[row, 3].set_title('RGB(red) vs DINOv2(green)\nYellow=both')
        axes[row, 3].axis('off')
        row += 1

    # Row 3: SAM method
    if results.get('sam'):
        axes[row, 0].imshow(results['sam']['diff_map'], cmap='hot', vmin=0, vmax=1)
        axes[row, 0].set_title(f'SAM Feature Distance')
        axes[row, 0].axis('off')

        axes[row, 1].imshow(results['sam']['mask'], cmap='gray')
        axes[row, 1].set_title(f'SAM Mask\n({results["sam"]["mask"].mean()*100:.1f}% changed)')
        axes[row, 1].axis('off')

        # Overlay on image
        overlay = np.array(edited_img).copy()
        overlay[results['sam']['mask']] = [0, 128, 255]
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title('SAM Overlay')
        axes[row, 2].axis('off')

        # Comparison with RGB
        combined = np.zeros((*results['rgb']['mask'].shape, 3), dtype=np.uint8)
        combined[results['rgb']['mask'] & results['sam']['mask']] = [255, 255, 0]
        combined[results['rgb']['mask'] & ~results['sam']['mask']] = [255, 0, 0]
        combined[~results['rgb']['mask'] & results['sam']['mask']] = [0, 128, 255]
        axes[row, 3].imshow(combined)
        axes[row, 3].set_title('RGB(red) vs SAM(blue)\nYellow=both')
        axes[row, 3].axis('off')

    plt.suptitle(f'Change Detection Comparison: {args.dataset}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    output_path = os.path.join(output_folder, f"change_detection_{args.dataset}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")

    # Save masks as numpy
    np.save(os.path.join(output_folder, f"mask_rgb_{args.dataset}.npy"), results['rgb']['mask'])
    if results.get('dinov2'):
        np.save(os.path.join(output_folder, f"mask_dinov2_{args.dataset}.npy"), results['dinov2']['mask'])
    if results.get('sam'):
        np.save(os.path.join(output_folder, f"mask_sam_{args.dataset}.npy"), results['sam']['mask'])
    if results.get('intersection'):
        np.save(os.path.join(output_folder, f"mask_intersection_{args.dataset}.npy"), results['intersection']['mask'])
        print(f"Recommended mask (intersection) saved: mask_intersection_{args.dataset}.npy")

    if not args.no_show:
        plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
