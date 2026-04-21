"""
experiment_normals.py
=====================
Systematically tests surface normal estimation methods and discontinuity
threshold strategies against GT WorldNormal EXR to find the most accurate
combination for the SNA metric.

Normal methods tested:
  roll2      - current: 2px central diff via np.roll (has edge-wrap bug)
  gradient   - np.gradient: proper boundary handling
  sobel      - 3x3 Sobel weighted gradient
  scharr     - 3x3 Scharr gradient (better isotropy than Sobel)
  plane3     - 3x3 neighbourhood SVD plane fit (most accurate on smooth surfaces)

Discontinuity thresholds tested:
  fixed_30cm  - |delta_d| > 0.30m  (current)
  fixed_10cm  - |delta_d| > 0.10m
  rel_5pct    - |delta_d| / d > 5%
  combined    - fixed_30cm OR rel_5pct

Dilate amounts also swept for the winning combination.

Usage:
    python UE_understanding/experiment_normals.py
    python UE_understanding/experiment_normals.py --datasets new2 new3
    python UE_understanding/experiment_normals.py --no-show
"""

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import OpenEXR
import Imath
from PIL import Image
from scipy.ndimage import binary_dilation, convolve, sobel as scipy_sobel

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "outputs")

GT_TO_METERS    = 10000.0 / 100.0
SKY_THRESHOLD_M = 500.0
DISC_DILATE_PX  = 3        # default dilate for disc mask


# ---------------------------------------------------------------------------
# EXR / camera loaders  (same as validate_sna_gt)
# ---------------------------------------------------------------------------

def load_scene_depth(path):
    exr  = OpenEXR.InputFile(path)
    dw   = exr.header()['dataWindow']
    w    = dw.max.x - dw.min.x + 1
    h    = dw.max.y - dw.min.y + 1
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = list(exr.header()['channels'].keys())
    for chan in ['R', 'SceneDepth', 'Z']:
        if chan in channels:
            buf = exr.channel(chan, FLOAT)
            arr = np.frombuffer(buf, dtype=np.float32).reshape(h, w).copy()
            return arr * GT_TO_METERS
    raise ValueError(f"No depth channel in {path}")


def load_gt_world_normals(path):
    """Load WorldNormal EXR with correct UE decoding: N = raw - 1, then normalise."""
    exr  = OpenEXR.InputFile(path)
    dw   = exr.header()['dataWindow']
    w    = dw.max.x - dw.min.x + 1
    h    = dw.max.y - dw.min.y + 1
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    ch = []
    for c in ['R', 'G', 'B']:
        buf = exr.channel(c, FLOAT)
        ch.append(np.frombuffer(buf, dtype=np.float32).reshape(h, w).copy())
    raw = np.stack(ch, axis=-1)
    N   = raw - 1.0
    return N / (np.linalg.norm(N, axis=-1, keepdims=True) + 1e-6)


def ue_cam_to_world(pitch_deg, yaw_deg, roll_deg):
    """Fixed sign convention: UE +pitch=nose-up=standard Ry(-p), same for roll."""
    p = -np.radians(pitch_deg)
    y =  np.radians(yaw_deg)
    r = -np.radians(roll_deg)
    Rz = np.array([[ np.cos(y), -np.sin(y), 0],
                   [ np.sin(y),  np.cos(y), 0],
                   [0, 0, 1]], dtype=float)
    Ry = np.array([[ np.cos(p), 0, np.sin(p)],
                   [0, 1, 0],
                   [-np.sin(p), 0, np.cos(p)]], dtype=float)
    Rx = np.array([[1, 0, 0],
                   [0,  np.cos(r), -np.sin(r)],
                   [0,  np.sin(r),  np.cos(r)]], dtype=float)
    R_base = np.array([[0, 0, 1],
                       [1, 0, 0],
                       [0, -1, 0]], dtype=float)
    return Rz @ Ry @ Rx @ R_base


# ---------------------------------------------------------------------------
# Normal estimation methods
# ---------------------------------------------------------------------------

def _build_pts(depth, fx, fy, cx, cy):
    H, W = depth.shape
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    return np.stack([(uu - cx) * depth / fx,
                     (vv - cy) * depth / fy,
                     depth.copy()], axis=-1)


def _orient_toward_camera(n):
    """Flip normals pointing away from camera (cam-space Z > 0 = away)."""
    n = n.copy()
    flip = n[..., 2] > 0
    n[flip] *= -1
    return n


def normals_roll2(depth, fx, fy, cx, cy):
    """Current method: 2px central diff via np.roll (wraps at image edges)."""
    pts = _build_pts(depth, fx, fy, cx, cy)
    dx  = np.roll(pts, -1, axis=1) - np.roll(pts, 1, axis=1)
    dy  = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    n   = np.cross(dx, dy)
    return -n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6)


def normals_gradient(depth, fx, fy, cx, cy):
    """np.gradient: handles boundaries without wrapping."""
    pts  = _build_pts(depth, fx, fy, cx, cy)
    # gradient returns [d/drow, d/dcol] for 2D; we want [d/dv, d/du]
    dy_g, dx_g = np.gradient(pts, axis=(0, 1))
    n = np.cross(dx_g, dy_g)
    return -n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6)


def normals_sobel(depth, fx, fy, cx, cy):
    """3x3 Sobel filter applied to each XYZ component of the point cloud."""
    pts = _build_pts(depth, fx, fy, cx, cy)
    dx  = np.stack([scipy_sobel(pts[..., i], axis=1) for i in range(3)], axis=-1)
    dy  = np.stack([scipy_sobel(pts[..., i], axis=0) for i in range(3)], axis=-1)
    n   = np.cross(dx, dy)
    return -n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6)


# Scharr 3x3 kernels (more isotropic than Sobel)
_SCHARR_X = np.array([[-3,  0,  3],
                       [-10, 0, 10],
                       [-3,  0,  3]], dtype=float) / 32.0
_SCHARR_Y = np.array([[-3, -10, -3],
                       [ 0,   0,  0],
                       [ 3,  10,  3]], dtype=float) / 32.0


def normals_scharr(depth, fx, fy, cx, cy):
    """3x3 Scharr filter (better rotational symmetry than Sobel)."""
    pts = _build_pts(depth, fx, fy, cx, cy)
    dx  = np.stack([convolve(pts[..., i], _SCHARR_X) for i in range(3)], axis=-1)
    dy  = np.stack([convolve(pts[..., i], _SCHARR_Y) for i in range(3)], axis=-1)
    n   = np.cross(dx, dy)
    return -n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6)


def normals_plane3(depth, fx, fy, cx, cy):
    """
    3x3 neighbourhood SVD plane fitting.
    For each pixel, fit a plane to its 3x3 window of 3D points and
    return the normal as the smallest eigenvector of the covariance matrix.
    Fully vectorised using numpy broadcasting.
    """
    pts  = _build_pts(depth, fx, fy, cx, cy)   # (H, W, 3)
    H, W = depth.shape

    # Pad so all pixels get a full 3x3 neighbourhood (edge replication)
    padded = np.pad(pts, ((1, 1), (1, 1), (0, 0)), mode='edge')

    # Stack 9 neighbours: (H, W, 9, 3)
    neighbors = []
    for dv in (-1, 0, 1):
        for du in (-1, 0, 1):
            neighbors.append(padded[1+dv : 1+dv+H, 1+du : 1+du+W, :])
    stacked = np.stack(neighbors, axis=2)       # (H, W, 9, 3)

    centroid = stacked.mean(axis=2, keepdims=True)
    centered = stacked - centroid               # (H, W, 9, 3)

    cov = np.einsum('...ki,...kj->...ij', centered, centered)   # (H, W, 3, 3)
    _, vecs = np.linalg.eigh(cov)              # eigenvalues ascending; vecs (H,W,3,3)
    normals  = vecs[..., 0]                    # smallest eigenvector (H, W, 3)
    normals /= (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-6)

    return _orient_toward_camera(normals)


# 5x5 version for comparison
def normals_plane5(depth, fx, fy, cx, cy):
    """5x5 neighbourhood SVD plane fitting (smoother, less edge-aware)."""
    pts  = _build_pts(depth, fx, fy, cx, cy)
    H, W = depth.shape
    padded = np.pad(pts, ((2, 2), (2, 2), (0, 0)), mode='edge')
    neighbors = []
    for dv in range(-2, 3):
        for du in range(-2, 3):
            neighbors.append(padded[2+dv : 2+dv+H, 2+du : 2+du+W, :])
    stacked  = np.stack(neighbors, axis=2)     # (H, W, 25, 3)
    centroid = stacked.mean(axis=2, keepdims=True)
    centered = stacked - centroid
    cov      = np.einsum('...ki,...kj->...ij', centered, centered)
    _, vecs  = np.linalg.eigh(cov)
    normals  = vecs[..., 0]
    normals /= (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-6)
    return _orient_toward_camera(normals)


NORMAL_METHODS = {
    'roll2':    normals_roll2,
    'gradient': normals_gradient,
    'sobel':    normals_sobel,
    'scharr':   normals_scharr,
    'plane3':   normals_plane3,
    'plane5':   normals_plane5,
}


# ---------------------------------------------------------------------------
# Discontinuity masks
# ---------------------------------------------------------------------------

def _grad_mag(depth):
    dx = np.abs(np.roll(depth, -1, axis=1) - np.roll(depth, 1, axis=1))
    dy = np.abs(np.roll(depth, -1, axis=0) - np.roll(depth, 1, axis=0))
    return np.maximum(dx, dy)


def disc_fixed(depth, threshold=0.30, dilate=DISC_DILATE_PX):
    m = _grad_mag(depth) > threshold
    return binary_dilation(m, iterations=dilate) if dilate > 0 else m


def disc_relative(depth, fraction=0.05, dilate=DISC_DILATE_PX):
    rel = _grad_mag(depth) / (depth + 1e-6)
    m   = rel > fraction
    return binary_dilation(m, iterations=dilate) if dilate > 0 else m


def disc_combined(depth, dilate=DISC_DILATE_PX):
    m = (_grad_mag(depth) > 0.30) | (_grad_mag(depth) / (depth + 1e-6) > 0.05)
    return binary_dilation(m, iterations=dilate) if dilate > 0 else m


DISC_METHODS = {
    'fixed_30cm': lambda d: disc_fixed(d, 0.30),
    'fixed_10cm': lambda d: disc_fixed(d, 0.10),
    'rel_5pct':   lambda d: disc_relative(d, 0.05),
    'combined':   disc_combined,
}


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def stats(ang_map, mask):
    a = ang_map[mask & np.isfinite(ang_map)]
    if len(a) == 0:
        return dict(n=0, mean=float('nan'), median=float('nan'),
                    p25=float('nan'), p75=float('nan'),
                    pct11=float('nan'), pct22=float('nan'))
    return dict(
        n      = len(a),
        mean   = float(a.mean()),
        median = float(np.median(a)),
        p25    = float(np.percentile(a, 25)),
        p75    = float(np.percentile(a, 75)),
        pct11  = float((a < 11.25).mean() * 100),
        pct22  = float((a < 22.5).mean()  * 100),
    )


def angular_error(n1, n2):
    dot = np.clip((n1 * n2).sum(axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


# ---------------------------------------------------------------------------
# Run one dataset/image
# ---------------------------------------------------------------------------

def run_dataset(folder, img_idx, cp):
    files      = sorted(os.listdir(folder))
    depth_exrs = sorted([f for f in files if 'SceneDepth' in f and 'WorldUnits' not in f and f.endswith('.exr')])
    normal_exrs= sorted([f for f in files if 'WorldNormal' in f and f.endswith('.exr')])

    depth      = load_scene_depth(os.path.join(folder, depth_exrs[img_idx]))
    gt_n_world = load_gt_world_normals(os.path.join(folder, normal_exrs[img_idx]))

    H, W = depth.shape
    fov  = float(cp['fov_deg'])
    fx   = (W / 2.0) / np.tan(np.radians(fov) / 2.0)
    fy, cx, cy = fx, W / 2.0, H / 2.0
    R_c2w = ue_cam_to_world(cp['pitch_deg'], cp['yaw_deg'], cp['roll_deg'])

    sky_mask = depth > SKY_THRESHOLD_M
    base_valid = ~sky_mask & (depth > 0.1)

    results = {}
    for nm_name, nm_fn in NORMAL_METHODS.items():
        t0 = time.time()
        n_cam   = nm_fn(depth, fx, fy, cx, cy)
        n_world = (R_c2w @ n_cam.reshape(-1, 3).T).T.reshape(H, W, 3)
        n_world /= (np.linalg.norm(n_world, axis=-1, keepdims=True) + 1e-6)
        ang     = angular_error(n_world, gt_n_world)
        elapsed = time.time() - t0

        for disc_name, disc_fn in DISC_METHODS.items():
            disc  = disc_fn(depth)
            valid = base_valid & ~disc
            s = stats(ang, valid)
            results[(nm_name, disc_name)] = {**s, 'secs': elapsed, 'ang': ang,
                                              'valid': valid, 'n_world': n_world}
    return results, depth, gt_n_world, base_valid, H, W


# ---------------------------------------------------------------------------
# Dilate sweep for the winning combination
# ---------------------------------------------------------------------------

def dilate_sweep(depth, ang_map, base_valid, nm_fn, fx, fy, cx, cy, cp):
    """Sweep dilate amounts 0..6 for fixed_30cm and rel_5pct, best normal method."""
    out = {}
    for dilate in (0, 1, 2, 3, 5):
        for disc_type in ('fixed_30cm', 'rel_5pct'):
            if disc_type == 'fixed_30cm':
                disc = disc_fixed(depth, 0.30, dilate)
            else:
                disc = disc_relative(depth, 0.05, dilate)
            valid = base_valid & ~disc
            s = stats(ang_map, valid)
            out[(disc_type, dilate)] = s
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Normal method experiments')
    parser.add_argument('--datasets', nargs='+', default=['new2', 'new3'])
    parser.add_argument('--image', type=int, default=0)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {}   # dataset -> results dict
    all_depth   = {}
    all_gt_wn   = {}
    all_bvalid  = {}

    for dataset in args.datasets:
        folder  = os.path.join(PROJECT_ROOT, "data", dataset)
        cp_path = os.path.join(folder, "camera_params.json")
        if not os.path.exists(cp_path):
            print(f"  Skipping {dataset}: no camera_params.json")
            continue
        with open(cp_path) as f:
            cp = json.load(f)
        if any(cp.get(k) is None for k in ('fov_deg','pitch_deg','yaw_deg','roll_deg')):
            print(f"  Skipping {dataset}: camera_params.json incomplete")
            continue

        print(f"\n=== {dataset} (image {args.image}) ===")
        res, depth, gt_wn, bvalid, H, W = run_dataset(folder, args.image, cp)
        all_results[dataset] = res
        all_depth[dataset]   = depth
        all_gt_wn[dataset]   = gt_wn
        all_bvalid[dataset]  = bvalid

    if not all_results:
        sys.exit("No datasets with camera params found.")

    # ── Print results table ─────────────────────────────────────────────────
    nm_names   = list(NORMAL_METHODS.keys())
    disc_names = list(DISC_METHODS.keys())

    print(f"\n{'='*100}")
    print(f"RESULTS  (mean angular error on clean pixels, lower is better)")
    print(f"{'='*100}")

    for dataset in args.datasets:
        if dataset not in all_results:
            continue
        res = all_results[dataset]
        print(f"\n  {dataset}")
        header = f"  {'Method':<12}" + "".join(f"  {d:<14}" for d in disc_names)
        print(header)
        print("  " + "-"*80)
        for nm in nm_names:
            row = f"  {nm:<12}"
            for disc in disc_names:
                s = res.get((nm, disc), {})
                if s.get('n', 0) == 0:
                    row += f"  {'---':<14}"
                else:
                    row += f"  {s['mean']:5.1f}deg ({s['pct11']:4.1f}%<11)  "
            print(row)

    # Find best (nm, disc) by mean, averaged across datasets
    combo_scores = {}
    for nm in nm_names:
        for disc in disc_names:
            means = []
            for dataset in args.datasets:
                if dataset in all_results:
                    s = all_results[dataset].get((nm, disc), {})
                    if s.get('n', 0) > 0 and not np.isnan(s['mean']):
                        means.append(s['mean'])
            if means:
                combo_scores[(nm, disc)] = np.mean(means)

    best_combo = min(combo_scores, key=combo_scores.get)
    best_nm, best_disc = best_combo
    print(f"\n  Best combination: normal={best_nm}, disc={best_disc} "
          f"(avg mean={combo_scores[best_combo]:.2f} deg)")

    # ── Dilate sweep for best normal method ─────────────────────────────────
    print(f"\n  Dilate sweep for normal={best_nm}:")
    print(f"  {'Type':<14} {'Dilate':>8}  {'Mean':>7}  {'Median':>8}  {'<11%':>7}  {'n':>10}")
    print("  " + "-"*65)

    for dataset in args.datasets:
        if dataset not in all_results:
            continue
        res  = all_results[dataset]
        dep  = all_depth[dataset]
        bv   = all_bvalid[dataset]
        # Get ang map for best nm
        ang_map_key = (best_nm, list(DISC_METHODS.keys())[0])
        ang  = res[ang_map_key]['ang']
        print(f"  [{dataset}]")
        for disc_type in ('fixed_30cm', 'rel_5pct'):
            for dilate in (0, 1, 2, 3, 5):
                if disc_type == 'fixed_30cm':
                    disc = disc_fixed(dep, 0.30, dilate)
                else:
                    disc = disc_relative(dep, 0.05, dilate)
                valid = bv & ~disc
                s = stats(ang, valid)
                print(f"  {disc_type:<14} {dilate:>8}  {s['mean']:7.2f}  "
                      f"{s['median']:8.2f}  {s['pct11']:7.1f}  {s['n']:>10,}")

    # ── Figure ───────────────────────────────────────────────────────────────
    # For each dataset: show bar chart of methods x disc + error heatmaps for
    # best and worst methods side by side.

    n_ds = len([d for d in args.datasets if d in all_results])
    fig, axes = plt.subplots(n_ds + 1, len(disc_names) + 2,
                             figsize=(5*(len(disc_names)+2), 5*(n_ds+1)))
    if n_ds + 1 == 1:
        axes = axes[np.newaxis, :]
    plt.subplots_adjust(hspace=0.35, wspace=0.15)

    # Row per dataset: mean error bar chart for each disc threshold
    for di, dataset in enumerate([d for d in args.datasets if d in all_results]):
        res = all_results[dataset]
        for dj, disc in enumerate(disc_names):
            ax  = axes[di, dj]
            means = [res.get((nm, disc), {}).get('mean', float('nan')) for nm in nm_names]
            pct11 = [res.get((nm, disc), {}).get('pct11', float('nan')) for nm in nm_names]
            colors = ['#d33' if nm == 'roll2' else '#38a' for nm in nm_names]
            if best_disc == disc:
                colors = ['#d33' if nm == 'roll2' else ('#2a2' if nm == best_nm else '#38a')
                          for nm in nm_names]
            bars = ax.bar(nm_names, means, color=colors)
            ax.set_title(f'{dataset} | {disc}\nmean deg (lower=better)', fontsize=8)
            ax.set_ylim(0, 40)
            ax.set_ylabel('Mean error (deg)', fontsize=7)
            for bar, m, p in zip(bars, means, pct11):
                if not np.isnan(m):
                    ax.text(bar.get_x() + bar.get_width()/2, m + 0.5,
                            f'{m:.1f}\n({p:.0f}%)', ha='center', va='bottom',
                            fontsize=6)
            ax.tick_params(axis='x', labelsize=6)

        # Error heatmaps: best vs roll2 for this dataset with best disc
        ang_best = res[(best_nm, best_disc)]['ang']
        ang_roll = res[('roll2', best_disc)]['ang']
        valid_best = res[(best_nm, best_disc)]['valid']

        vmax = float(np.nanpercentile(ang_best[valid_best], 95)) if valid_best.any() else 30

        ax_best = axes[di, -2]
        ax_best.imshow(np.where(valid_best, ang_best, np.nan), cmap='hot', vmin=0, vmax=vmax)
        ax_best.set_title(f'{dataset} | BEST: {best_nm}\nmean={res[(best_nm,best_disc)]["mean"]:.1f} deg',
                          fontsize=8)
        ax_best.axis('off')

        ax_roll = axes[di, -1]
        ax_roll.imshow(np.where(valid_best, ang_roll, np.nan), cmap='hot', vmin=0, vmax=vmax)
        ax_roll.set_title(f'{dataset} | baseline: roll2\nmean={res[("roll2",best_disc)]["mean"]:.1f} deg',
                          fontsize=8)
        ax_roll.axis('off')

    # Last row: combined bar chart across all datasets
    ax_sum = axes[-1, 0]
    nm_avg = {nm: np.nanmean([all_results[d].get((nm, best_disc), {}).get('mean', float('nan'))
                               for d in args.datasets if d in all_results])
              for nm in nm_names}
    colors = ['#d33' if nm == 'roll2' else ('#2a2' if nm == best_nm else '#38a')
              for nm in nm_names]
    bars = ax_sum.bar(nm_names, [nm_avg[nm] for nm in nm_names], color=colors)
    ax_sum.set_title(f'Avg across datasets | disc={best_disc}\n(red=baseline  green=best)',
                     fontsize=9)
    ax_sum.set_ylabel('Mean error (deg)')
    ax_sum.set_ylim(0, 40)
    for bar, nm in zip(bars, nm_names):
        v = nm_avg[nm]
        if not np.isnan(v):
            ax_sum.text(bar.get_x() + bar.get_width()/2, v + 0.3, f'{v:.1f}',
                        ha='center', va='bottom', fontsize=7)

    # Summary text
    ax_txt = axes[-1, 1]
    ax_txt.axis('off')
    lines = ['EXPERIMENT SUMMARY', '=' * 30, '']
    lines += ['All combos (avg mean deg):', '']
    sorted_combos = sorted(combo_scores.items(), key=lambda x: x[1])
    for (nm, disc), score in sorted_combos[:12]:
        marker = ' <-- BEST' if (nm, disc) == best_combo else ''
        lines.append(f'  {nm:<10} {disc:<14} {score:5.2f}{marker}')
    lines += ['', f'Ceiling error (best method):', f'  mean ~ {combo_scores[best_combo]:.1f} deg']
    ax_txt.text(0.02, 0.98, '\n'.join(lines), transform=ax_txt.transAxes,
                fontsize=7, va='top', ha='left', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

    # Hide unused axes in last row
    for col in range(2, len(disc_names) + 2):
        axes[-1, col].axis('off')

    plt.suptitle('Surface Normal Estimation Method Comparison\n'
                 'GT SceneDepth -> normals -> world space  vs  GT WorldNormal EXR',
                 fontsize=12, fontweight='bold')

    out_path = os.path.join(OUTPUT_DIR, 'normal_method_experiment.png')
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"\nSaved: {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
