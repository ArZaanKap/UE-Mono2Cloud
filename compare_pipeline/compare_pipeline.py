"""
compare_pipeline.py
====================
Single-run end-to-end pipeline evaluation: one mask model + one depth model.

Pipeline:
  1. Mask model predicts changed pixels between original and edited images
  2. Depth model runs on the edited image
  3. Depth calibrated (scale+shift) on pixels predicted as UNCHANGED
  4. Evaluated against GT depth on both GT-changed and GT-unchanged pixels

Results saved to:
  compare_pipeline/outputs/{dataset}/
    ├── metrics.json
    ├── {mask}__{depth}_visualization.png
    └── pointclouds/{mask}__{depth}.las

Usage:
    python compare_pipeline/compare_pipeline.py --dataset new0 --mask gescf --depth unik3d
    python compare_pipeline/compare_pipeline.py --dataset new1 --mask dinov2 --depth dpro
    python compare_pipeline/compare_pipeline.py --dataset new3 --mask ogescf --depth da3_giant
Depth models:  unik3d | unidepth_vitl | moge2 | da3_giant | dpro
Mask models:   gescf | ogescf | dinov2
"""

import os, sys, json, argparse, warnings, time, tempfile, math
import numpy as np
import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
warnings.filterwarnings("ignore")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_ROOT  = os.path.join(SCRIPT_DIR, "outputs")

sys.path.insert(0, os.path.join(PROJECT_ROOT, "compare_edit_depth"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "change_detection_results"))

from compare_edit_depth2 import (
    load_image, load_exr_depth, find_files,
    run_model_subprocess,
    fit_metric_alignment, aligned_prediction_to_metric_depth,
    compute_depth_metrics,
    DA3_DEFAULT_PROCESS_RES,
    UNIDEPTH_TRIM_KEEP_PERCENT,
)
from test_change_detection import dinov2_feature_mask, gescf_feature_mask, official_gescf_mask
from params import DINO_BASELINE, GESCF_BASELINE, OFFICIAL_GESCF_BASELINE

DEPTH_MODELS = ['unik3d', 'unidepth_vitl', 'moge2', 'da3_giant', 'dpro']
MASK_MODELS  = ['gescf', 'ogescf', 'dinov2']
DATASETS     = ['new0', 'new1', 'new2', 'new3', 'new4']

DEPTH_LABELS = {
    'unik3d':        'UniK3D',
    'unidepth_vitl': 'UniDepth-V2',
    'moge2':         'MoGe-2',
    'da3_giant':     'DA3 Giant',
    'dpro':          'Depth Pro',
}
MASK_LABELS = {
    'gescf':  'GeSCF',
    'ogescf': 'Official GeSCF',
    'dinov2': 'DINOv2',
}

DEFAULT_CHANGE_THRESHOLD = 0.0


# ---------------------------------------------------------------------------
# Mask inference
# ---------------------------------------------------------------------------

def run_mask_model(mask_key, img1, img2, original_path, edited_path):
    """Return (mask: bool H×W, diff_map: float H×W)."""
    if mask_key == 'dinov2':
        return dinov2_feature_mask(img1, img2, **DINO_BASELINE)
    if mask_key == 'gescf':
        return gescf_feature_mask(img1, img2, **GESCF_BASELINE)
    if mask_key == 'ogescf':
        return official_gescf_mask(original_path, edited_path, **OFFICIAL_GESCF_BASELINE)
    raise ValueError(f"Unknown mask model: {mask_key}")


# ---------------------------------------------------------------------------
# Depth inference
# ---------------------------------------------------------------------------

def run_depth_inference(model_key, edited_path, output_path, fov_deg, H_gt, W_gt, cp):
    """Run depth model subprocess on edited image, save .npy to output_path."""
    fx = (W_gt / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    unidepth_intrinsics = [
        [float(fx), 0.0, float(W_gt / 2.0)],
        [0.0, float(fx), float(H_gt / 2.0)],
        [0.0, 0.0, 1.0],
    ]
    run_model_subprocess(
        model_key, edited_path, output_path,
        da3_process_res=DA3_DEFAULT_PROCESS_RES.get(model_key, 1024),
        unidepth_use_camera=(model_key == 'unidepth_vitl'),
        unidepth_intrinsics=unidepth_intrinsics,
        moge2_fov_x=(float(fov_deg) if model_key == 'moge2' and cp.get('fov_deg') else None),
    )


# ---------------------------------------------------------------------------
# Calibration + evaluation
# ---------------------------------------------------------------------------

def calibrate_and_eval(depth_raw, depth_gt_edit, calib_mask, gt_changed, gt_unchanged, model_key):
    """
    Fit scale+shift on calib_mask pixels, evaluate on GT changed/unchanged.
    Returns (depth_calibrated, changed_metrics, unchanged_metrics, fit_info).
    On failure returns (None, None, None, error_string).
    """
    alignment_domain = 'disparity' if model_key == 'hyden' else 'depth'
    trim_keep = UNIDEPTH_TRIM_KEEP_PERCENT if model_key == 'unidepth_vitl' else None
    try:
        scale, shift, fit_info = fit_metric_alignment(
            depth_raw, depth_gt_edit, calib_mask,
            scaling_method='ls',
            trim_keep_percent=trim_keep,
            alignment_domain=alignment_domain,
        )
    except ValueError as e:
        return None, None, None, str(e)

    depth_cal = aligned_prediction_to_metric_depth(
        depth_raw, scale, shift, scaling_method='ls', alignment_domain=alignment_domain
    )
    ch_m   = compute_depth_metrics(depth_cal, depth_gt_edit, gt_changed)
    unch_m = compute_depth_metrics(depth_cal, depth_gt_edit, gt_unchanged)
    return depth_cal, ch_m, unch_m, fit_info


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_visualization(original_img, edited_img, depth_gt_edit, gt_changed,
                       pred_mask, depth_cal, ch_m, unch_m,
                       mask_label, depth_label, dataset, out_path):
    """2×4 grid: [original | edited | pred mask | GT mask] / [GT depth | calib depth | unch error | ch error]"""
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    plt.subplots_adjust(hspace=0.28, wspace=0.06)

    valid_gt = depth_gt_edit[depth_gt_edit > 0]
    vmin_d = float(np.percentile(valid_gt, 0.1))
    vmax_d = float(np.percentile(valid_gt, 99.9))

    # Row 0 — Original | Edited | Predicted mask | GT mask
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original', fontsize=11); axes[0, 0].axis('off')

    axes[0, 1].imshow(edited_img)
    axes[0, 1].set_title('Edited', fontsize=11); axes[0, 1].axis('off')

    ov_pred = np.zeros((*pred_mask.shape, 4), dtype=np.float32)
    ov_pred[pred_mask]  = [1.0, 0.3, 0.0, 0.55]
    ov_pred[~pred_mask] = [0.0, 0.8, 0.0, 0.12]
    axes[0, 2].imshow(edited_img); axes[0, 2].imshow(ov_pred)
    axes[0, 2].set_title(f'{mask_label} mask\n({pred_mask.mean()*100:.1f}% changed)', fontsize=11)
    axes[0, 2].axis('off')

    ov_gt = np.zeros((*gt_changed.shape, 4), dtype=np.float32)
    ov_gt[gt_changed]  = [1.0, 0.0, 0.0, 0.55]
    ov_gt[~gt_changed] = [0.0, 1.0, 0.0, 0.12]
    axes[0, 3].imshow(edited_img); axes[0, 3].imshow(ov_gt)
    axes[0, 3].set_title(f'GT change mask\n({gt_changed.mean()*100:.1f}% changed)', fontsize=11)
    axes[0, 3].axis('off')

    # Row 1 — GT depth | Calibrated depth | Error on GT-unchanged | Error on GT-changed
    im = axes[1, 0].imshow(depth_gt_edit, cmap='turbo', vmin=vmin_d, vmax=vmax_d)
    axes[1, 0].set_title('GT Depth (edit)', fontsize=11); axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04, label='m')

    im = axes[1, 1].imshow(depth_cal, cmap='turbo', vmin=vmin_d, vmax=vmax_d)
    axes[1, 1].set_title(f'{depth_label}\ncalib on {mask_label} unchanged', fontsize=11)
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04, label='m')

    full_err = depth_cal - depth_gt_edit

    err_unch = np.where(ndimage.binary_erosion(~gt_changed, iterations=2), full_err, np.nan)
    im = axes[1, 2].imshow(err_unch, cmap='PuOr', vmin=-0.5, vmax=0.5)
    axes[1, 2].set_title(
        f'Error (GT-unchanged)\nMAE={unch_m["mae"]:.3f}m  δ1={unch_m["d1"]:.3f}', fontsize=11)
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04, label='m')

    err_ch = np.where(ndimage.binary_erosion(gt_changed, iterations=1), full_err, np.nan)
    im = axes[1, 3].imshow(err_ch, cmap='PuOr', vmin=-0.5, vmax=0.5)
    axes[1, 3].set_title(
        f'Error (GT-changed)\nMAE={ch_m["mae"]:.3f}m  δ1={ch_m["d1"]:.3f}', fontsize=11)
    axes[1, 3].axis('off')
    plt.colorbar(im, ax=axes[1, 3], fraction=0.046, pad=0.04, label='m')

    plt.suptitle(
        f'{mask_label} mask → {depth_label} — {dataset}\n'
        f'Changed  MAE={ch_m["mae"]:.4f}m  RMSE={ch_m["rmse"]:.4f}m  δ1={ch_m["d1"]:.3f}   |   '
        f'Unchanged  MAE={unch_m["mae"]:.4f}m  δ1={unch_m["d1"]:.3f}',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='End-to-end pipeline: one mask model + one depth model on one dataset'
    )
    parser.add_argument('--dataset', required=True, choices=DATASETS, nargs='+',
                        help='Dataset(s) to evaluate, e.g. --dataset new0 new1 new3')
    parser.add_argument('--mask',    required=True, choices=MASK_MODELS,
                        help='Change detection model: gescf | ogescf | dinov2')
    parser.add_argument('--depth',   required=True, choices=DEPTH_MODELS,
                        help='Depth model: unik3d | unidepth_vitl | moge2 | da3_giant | dpro')
    args = parser.parse_args()

    mask_key  = args.mask
    depth_key = args.depth

    for dataset in args.dataset:
        _run(dataset, mask_key, depth_key)


MIN_DEPTH_M = 0.001


def export_las(edited_img, depth_full, fov_deg, out_path):
    rgb = np.array(edited_img)
    h, w = depth_full.shape
    if rgb.shape[:2] != (h, w):
        rgb = np.array(edited_img.resize((w, h), Image.BILINEAR))

    valid = np.isfinite(depth_full) & (depth_full > MIN_DEPTH_M)
    if not np.any(valid):
        raise ValueError("No valid pixels for point cloud export")

    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    focal  = w / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    z = depth_full[valid]
    x = ((xx[valid] - cx) * z) / focal
    y = ((yy[valid] - cy) * z) / focal
    colors = rgb[valid]

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header=header)
    las.x = z
    las.y = -x
    las.z = -y
    las.red   = colors[:, 0].astype(np.uint16) * 257
    las.green = colors[:, 1].astype(np.uint16) * 257
    las.blue  = colors[:, 2].astype(np.uint16) * 257
    las.write(str(out_path))
    return int(z.size)


def _update_metrics(dataset_dir, combo_key, result):
    """Upsert one combo entry into outputs/{dataset}/metrics.json."""
    metrics_path = os.path.join(dataset_dir, "metrics.json")
    all_metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            all_metrics = json.load(f)
    all_metrics[combo_key] = result
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Updated: {metrics_path}  ({len(all_metrics)} combos)")


def _run(dataset, mask_key, depth_key):
    # ── Paths ─────────────────────────────────────────────────────────────────
    data_dir   = os.path.join(PROJECT_ROOT, "data", dataset)
    dataset_dir = os.path.join(OUTPUT_ROOT, dataset)
    combo_key  = f"{mask_key}__{depth_key}"
    os.makedirs(dataset_dir, exist_ok=True)

    # ── Camera / scene params ──────────────────────────────────────────────────
    cp = {}
    params_path = os.path.join(data_dir, "params.json")
    if os.path.exists(params_path):
        with open(params_path) as f:
            cp = json.load(f)
    fov_deg = cp.get('fov_deg') or 90.0
    change_threshold = cp.get('change_threshold_m', DEFAULT_CHANGE_THRESHOLD)

    # ── Load images + GT depths ────────────────────────────────────────────────
    print(f"\nDataset:     {dataset}")
    print(f"Mask model:  {MASK_LABELS[mask_key]}")
    print(f"Depth model: {DEPTH_LABELS[depth_key]}")
    print(f"Output:      {dataset_dir}\n")

    original_path, edited_path, gt_orig_path, gt_edit_path = find_files(data_dir)

    original_img  = load_image(original_path)
    edited_img    = load_image(edited_path)
    depth_gt_orig = load_exr_depth(gt_orig_path)
    depth_gt_edit = load_exr_depth(gt_edit_path)
    target_shape  = depth_gt_orig.shape
    H_gt, W_gt    = target_shape

    if edited_img.size != original_img.size:
        edited_img = edited_img.resize(original_img.size, Image.BILINEAR)

    gt_changed   = np.abs(depth_gt_edit - depth_gt_orig) > change_threshold
    gt_unchanged = ~gt_changed
    print(f"GT change mask: {gt_changed.mean()*100:.1f}% changed  (threshold={change_threshold}m)")

    # ── Step 1: Run mask model ─────────────────────────────────────────────────
    print(f"\n[1/3] Running mask model: {MASK_LABELS[mask_key]} ...")
    t0 = time.perf_counter()
    pred_mask, _ = run_mask_model(mask_key, original_img, edited_img, original_path, edited_path)
    mask_time = time.perf_counter() - t0

    if pred_mask.shape != target_shape:
        pred_mask = np.array(
            Image.fromarray(pred_mask.astype(np.uint8) * 255).resize(
                (W_gt, H_gt), Image.NEAREST
            )
        ) > 127
    print(f"     {pred_mask.mean()*100:.1f}% predicted changed  ({mask_time:.1f}s)")

    # ── Step 2: Run depth model ────────────────────────────────────────────────
    print(f"\n[2/3] Running depth model: {DEPTH_LABELS[depth_key]} ...")
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        tmp_depth = f.name
    try:
        t0 = time.perf_counter()
        run_depth_inference(depth_key, edited_path, tmp_depth, fov_deg, H_gt, W_gt, cp)
        depth_time = time.perf_counter() - t0
        depth_raw = np.load(tmp_depth)
    finally:
        if os.path.exists(tmp_depth):
            os.remove(tmp_depth)

    if depth_raw.shape != target_shape:
        depth_raw = np.array(
            Image.fromarray(depth_raw.astype(np.float32)).resize(
                (W_gt, H_gt), Image.BILINEAR
            )
        )
    print(f"     shape={depth_raw.shape}  range=[{depth_raw.min():.2f}, {depth_raw.max():.2f}]m  ({depth_time:.1f}s)")

    # ── Step 3: Calibrate + evaluate ──────────────────────────────────────────
    print(f"\n[3/3] Calibrating on predicted-unchanged pixels, evaluating on GT masks ...")
    depth_cal, ch_m, unch_m, fit_info = calibrate_and_eval(
        depth_raw, depth_gt_edit,
        calib_mask=~pred_mask,
        gt_changed=gt_changed,
        gt_unchanged=gt_unchanged,
        model_key=depth_key,
    )

    if depth_cal is None:
        print(f"     FAILED: {fit_info}")
        _update_metrics(dataset_dir, combo_key, {
            'mask_model': mask_key, 'depth_model': depth_key,
            'dataset': dataset, 'error': fit_info,
        })
        return

    print(f"\n{'='*60}")
    print(f"RESULTS  —  {MASK_LABELS[mask_key]} → {DEPTH_LABELS[depth_key]}  ({dataset})")
    print(f"{'='*60}")
    print(f"{'Region':<18} {'n':>8} {'MAE':>9} {'RMSE':>9} {'δ1':>8} {'δ2':>8} {'δ3':>8}")
    print(f"{'-'*60}")
    for label, m in [('GT-changed', ch_m), ('GT-unchanged', unch_m)]:
        print(f"{label:<18} {m['n']:>8,} {m['mae']:>9.4f} {m['rmse']:>9.4f} "
              f"{m['d1']:>8.3f} {m['d2']:>8.3f} {m['d3']:>8.3f}")
    print(f"{'='*60}")

    # ── Save outputs ───────────────────────────────────────────────────────────
    _update_metrics(dataset_dir, combo_key, {
        'mask_model':         mask_key,
        'depth_model':        depth_key,
        'dataset':            dataset,
        'change_threshold_m': change_threshold,
        'changed':            ch_m,
        'unchanged':          unch_m,
        'fit_info':           fit_info,
        'pred_changed_frac':  float(pred_mask.mean()),
        'gt_changed_frac':    float(gt_changed.mean()),
        'mask_wall_s':        round(mask_time, 2),
        'depth_wall_s':       round(depth_time, 2),
    })

    save_visualization(
        original_img, edited_img, depth_gt_edit, gt_changed,
        pred_mask, depth_cal, ch_m, unch_m,
        MASK_LABELS[mask_key], DEPTH_LABELS[depth_key], dataset,
        os.path.join(dataset_dir, f"{combo_key}_visualization.png"),
    )

    pc_dir = os.path.join(dataset_dir, "pointclouds")
    os.makedirs(pc_dir, exist_ok=True)
    las_path = os.path.join(pc_dir, f"{combo_key}.las")
    n_pts = export_las(edited_img, depth_cal, fov_deg, las_path)
    print(f"  Saved: {las_path}  ({n_pts:,} points)")

    print("\nDone.")


if __name__ == "__main__":
    main()
