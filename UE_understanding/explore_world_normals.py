"""
explore_world_normals.py
========================
Validates the Surface Normal Alignment (SNA) metric on GT depth pairs.

Shows SNA heatmaps split by unchanged vs changed pixels so you can see
spatially where the metric fires and how large the errors are in each region.

Usage:
    python ue_understanding/explore_world_normals.py              # new0
    python ue_understanding/explore_world_normals.py --dataset new1
    python ue_understanding/explore_world_normals.py --no-show
"""

import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import OpenEXR
import Imath
from scipy.ndimage import binary_dilation

matplotlib.use("Agg")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "outputs")

GT_TO_METERS             = 10000.0 / 100.0
DEFAULT_FOV_DEG          = 90.0
DISC_THRESHOLD_M         = 0.3    # mask depth-discontinuity pixels (unreliable normals)

DEFAULT_DATASET          = "new2"  # ← edit this


# ---------------------------------------------------------------------------
# EXR loading
# ---------------------------------------------------------------------------

def _exr_open(path):
    exr = OpenEXR.InputFile(path)
    dw  = exr.header()['dataWindow']
    w   = dw.max.x - dw.min.x + 1
    h   = dw.max.y - dw.min.y + 1
    return exr, h, w


def load_scene_depth(path):
    exr, h, w = _exr_open(path)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    header_channels = list(exr.header()['channels'].keys())
    for chan in ['R', 'SceneDepth', 'Z']:
        if chan in header_channels:
            buf = exr.channel(chan, FLOAT)
            arr = np.frombuffer(buf, dtype=np.float32).reshape(h, w).copy()
            return arr * GT_TO_METERS
    raise ValueError(f"No depth channel in {path}")


# ---------------------------------------------------------------------------
# Dataset file discovery
# ---------------------------------------------------------------------------

def find_files(dataset):
    folder = os.path.join(PROJECT_ROOT, "data", dataset)
    files  = sorted(os.listdir(folder))
    depth_exrs = sorted([f for f in files
                         if 'SceneDepth' in f and 'WorldUnits' not in f and f.endswith('.exr')])
    if len(depth_exrs) < 2:
        raise FileNotFoundError(f"Need 2 SceneDepth EXRs in {folder}, found {len(depth_exrs)}")
    return {
        'folder': folder,
        'depth0': os.path.join(folder, depth_exrs[0]),
        'depth1': os.path.join(folder, depth_exrs[1]),
    }


# ---------------------------------------------------------------------------
# Normal utilities
# ---------------------------------------------------------------------------

def angular_error_deg(n1, n2):
    dot = np.clip((n1 * n2).sum(axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def normals_from_depth(depth_m, fx, fy, cx, cy):
    H, W = depth_m.shape
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    X = (uu - cx) * depth_m / fx
    Y = (vv - cy) * depth_m / fy
    Z = depth_m.copy()
    pts = np.stack([X, Y, Z], axis=-1)
    dx = np.roll(pts, -1, axis=1) - np.roll(pts,  1, axis=1)
    dy = np.roll(pts, -1, axis=0) - np.roll(pts,  1, axis=0)
    normals = np.cross(dx, dy)
    mag = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / (mag + 1e-6)


def discontinuity_mask(depth):
    """True at pixels where depth gradient is too large for reliable normals."""
    dx = np.abs(np.roll(depth, -1, axis=1) - np.roll(depth, 1, axis=1))
    dy = np.abs(np.roll(depth, -1, axis=0) - np.roll(depth, 1, axis=0))
    return (dx > DISC_THRESHOLD_M) | (dy > DISC_THRESHOLD_M)


def sna_stats(depth_pred, depth_gt, valid_mask, fx, fy, cx, cy):
    """
    Surface Normal Alignment (SNA) — camera space only.

    Interpretation:
      ~0 deg  : predicted depth reproduces the same surface shape as GT
      high deg: model flattens curved surfaces or smears geometry at edges
    """
    n_pred = normals_from_depth(depth_pred, fx, fy, cx, cy)
    n_gt   = normals_from_depth(depth_gt,   fx, fy, cx, cy)
    ang    = angular_error_deg(n_pred, n_gt)
    v      = valid_mask & np.isfinite(ang)
    if v.sum() == 0:
        return dict(mean=float('nan'), median=float('nan'), p90=float('nan'), n=0, ang_map=ang)
    a = ang[v]
    return dict(
        mean    = float(a.mean()),
        median  = float(np.median(a)),
        p90     = float(np.percentile(a, 90)),
        n       = int(v.sum()),
        ang_map = ang,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='SNA metric validation on GT depth pairs')
    parser.add_argument('--dataset', default=DEFAULT_DATASET)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files  = find_files(args.dataset)
    depth0 = load_scene_depth(files['depth0'])
    depth1 = load_scene_depth(files['depth1'])

    H, W    = depth0.shape
    fov_rad = np.radians(DEFAULT_FOV_DEG)
    fx      = (W / 2.0) / np.tan(fov_rad / 2.0)
    fy, cx, cy = fx, W / 2.0, H / 2.0

    sky_mask      = (depth0 > 500) | (depth1 > 500)
    disc          = discontinuity_mask(depth0) | discontinuity_mask(depth1)
    valid         = (~sky_mask) & (depth0 > 0.1) & (depth1 > 0.1) & (~disc)

    # Exact-equality split: UE GT pixels are bit-identical when unchanged
    depth_changed = (depth0 != depth1) & ~sky_mask
    # Dilate changed mask so boundary pixels (whose normals are polluted by
    # neighbouring changed-depth values via finite differences) are excluded
    changed_dilated = binary_dilation(depth_changed, iterations=3)
    unch_v = valid & ~changed_dilated
    ch_v   = valid &  depth_changed

    sna     = sna_stats(depth0, depth1, valid, fx, fy, cx, cy)
    ang_map = sna['ang_map']

    # Shared colormap scale: 95th percentile of valid angles
    vmax = float(np.nanpercentile(ang_map[valid], 95))

    print(f"\nSNA (GT vs GT)  --  {args.dataset}")
    print(f"  Unchanged : mean={ang_map[unch_v].mean():.2f}°  "
          f"median={np.median(ang_map[unch_v]):.2f}°  "
          f"p90={np.percentile(ang_map[unch_v], 90):.2f}°  n={unch_v.sum():,}")
    print(f"  Changed   : mean={ang_map[ch_v].mean():.2f}°  "
          f"median={np.median(ang_map[ch_v]):.2f}°  "
          f"p90={np.percentile(ang_map[ch_v], 90):.2f}°  n={ch_v.sum():,}")

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    plt.subplots_adjust(wspace=0.12)

    # Panel 1 — GT change mask
    change_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    change_rgb[unch_v]        = [200, 220, 255]   # light blue = unchanged (used)
    change_rgb[depth_changed] = [255,  80,  60]   # red = changed (used)
    # grey = dilated border zone excluded from both
    border = changed_dilated & ~depth_changed & ~sky_mask
    change_rgb[border] = [160, 160, 160]
    axes[0].imshow(change_rgb)
    axes[0].set_title(
        f'GT change mask  (red=changed  blue=unchanged  grey=border excluded)\n'
        f'Changed: {depth_changed.sum():,}   Unchanged: {unch_v.sum():,}   Border: {border.sum():,}',
        fontsize=10)
    axes[0].axis('off')

    # Panel 2 — SNA on unchanged pixels only
    unch_map = np.where(unch_v, ang_map, np.nan)
    im2 = axes[1].imshow(unch_map, cmap='hot', vmin=0, vmax=vmax)
    axes[1].set_title(
        f'SNA — unchanged pixels\n'
        f'mean={ang_map[unch_v].mean():.1f}°  median={np.median(ang_map[unch_v]):.1f}°  '
        f'p90={np.percentile(ang_map[unch_v], 90):.1f}°',
        fontsize=10)
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='degrees')

    # Panel 3 — SNA on changed pixels only
    ch_map = np.where(ch_v, ang_map, np.nan)
    im3 = axes[2].imshow(ch_map, cmap='hot', vmin=0, vmax=vmax)
    axes[2].set_title(
        f'SNA — changed pixels\n'
        f'mean={ang_map[ch_v].mean():.1f}°  median={np.median(ang_map[ch_v]):.1f}°  '
        f'p90={np.percentile(ang_map[ch_v], 90):.1f}°',
        fontsize=10)
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, label='degrees')

    plt.suptitle(
        f'Surface Normal Alignment — {args.dataset}  (vmax={vmax:.1f}° = 95th pct)',
        fontsize=13, fontweight='bold')

    out = os.path.join(OUTPUT_DIR, f'{args.dataset}_surface_normal_alignment.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
