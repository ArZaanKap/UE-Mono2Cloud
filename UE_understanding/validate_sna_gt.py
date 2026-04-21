"""
validate_sna_gt.py
==================
Validates the SNA (Surface Normal Alignment) pipeline ceiling accuracy by:
  1. Loading GT SceneDepth EXR and computing surface normals via finite differences
     (the same pipeline used in compare_edit_depth)
  2. Comparing those normals to the GT WorldNormal EXR

This answers two questions:
  A. Is the WorldNormal EXR decoding correct?  (tests raw*2-1 vs raw/pi-1 vs raw)
  B. What is the ceiling angular error of the finite-difference method?
     (even with perfect depth, edge noise from finite diffs sets a floor)

Only works on datasets with camera_params.json fully filled in.
Filled-in datasets: new2, new3

Usage:
    python UE_understanding/validate_sna_gt.py
    python UE_understanding/validate_sna_gt.py --dataset new3
    python UE_understanding/validate_sna_gt.py --dataset new2 --image 1
    python UE_understanding/validate_sna_gt.py --no-show
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import OpenEXR
import Imath
from PIL import Image
from scipy.ndimage import binary_dilation

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "outputs")

GT_TO_METERS        = 10000.0 / 100.0   # UE SceneDepth raw → metres
DISC_THRESHOLD_M    = 0.3               # depth jump (m) flagged as discontinuity
DISC_DILATE_PX      = 3                 # dilate disc mask to hide edge artefacts
SKY_THRESHOLD_M     = 500.0            # pixels further than this are sky/void


# ---------------------------------------------------------------------------
# EXR loaders
# ---------------------------------------------------------------------------

def _exr_channels(path, channel_names):
    exr  = OpenEXR.InputFile(path)
    dw   = exr.header()['dataWindow']
    w    = dw.max.x - dw.min.x + 1
    h    = dw.max.y - dw.min.y + 1
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    arrays = []
    for c in channel_names:
        buf = exr.channel(c, FLOAT)
        arrays.append(np.frombuffer(buf, dtype=np.float32).reshape(h, w).copy())
    return arrays, h, w


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
    raise ValueError(f"No depth channel found in {path}")


def load_world_normal_raw(path):
    """Return raw float32 R/G/B channels stacked as (H,W,3) — no decoding yet."""
    (R, G, B), h, w = _exr_channels(path, ['R', 'G', 'B'])
    return np.stack([R, G, B], axis=-1)


def decode_world_normal(raw, mode):
    """
    Three candidate decodings of the WorldNormal EXR.
      'std'  : N = raw * 2 - 1       (standard UE remap: stored as (N+1)/2)
      'pi'   : N = raw / pi - 1      (current compare_edit_depth code)
      'direct': N = raw              (raw float; possible if EXR stores signed)
    Returns unit normals (H,W,3).
    """
    if mode == 'std':
        N = raw * 2.0 - 1.0
    elif mode == 'pi':
        N = raw / np.pi - 1.0
    elif mode == 'shift':
        N = raw - 1.0
    elif mode == 'direct':
        N = raw.copy()
    else:
        raise ValueError(f"Unknown decoding mode: {mode}")
    mag = np.linalg.norm(N, axis=-1, keepdims=True)
    return N / (mag + 1e-6)


# ---------------------------------------------------------------------------
# Camera / geometry helpers  (identical to compare_edit_depth)
# ---------------------------------------------------------------------------

def ue_cam_to_world(pitch_deg, yaw_deg, roll_deg):
    """
    Build the camera-to-world rotation matrix for a UE camera.
    UE world: X=forward, Y=right, Z=up.
    Camera screen: X=right, Y=down, Z=depth.

    UE sign conventions differ from standard math:
      Pitch: UE positive = nose UP. Standard Ry(θ) positive = nose DOWN.
             Fix: negate pitch before Ry.
      Yaw:   UE positive = turn right (CW from above). Standard Rz(θ) CW from
             above for positive θ as well. No sign change needed.
      Roll:  UE positive = CW when looking forward. Standard Rx positive = CCW.
             Fix: negate roll before Rx.  (Roll=0 for all current datasets.)
    """
    p = -np.radians(pitch_deg)   # negate: UE positive-pitch = nose-up = standard negative-Ry
    y =  np.radians(yaw_deg)
    r = -np.radians(roll_deg)    # negate: UE positive-roll = CW-fwd = standard negative-Rx
    Rz = np.array([[ np.cos(y), -np.sin(y), 0],
                   [ np.sin(y),  np.cos(y), 0],
                   [0, 0, 1]], dtype=float)
    Ry = np.array([[ np.cos(p), 0, np.sin(p)],
                   [0, 1, 0],
                   [-np.sin(p), 0, np.cos(p)]], dtype=float)
    Rx = np.array([[1, 0, 0],
                   [0,  np.cos(r), -np.sin(r)],
                   [0,  np.sin(r),  np.cos(r)]], dtype=float)
    # Default UE camera (P=Y=R=0): depth→world+X, right→world+Y, down→world-Z
    R_base = np.array([[0, 0, 1],
                       [1, 0, 0],
                       [0, -1, 0]], dtype=float)
    return Rz @ Ry @ Rx @ R_base


def normals_from_depth(depth_m, fx, fy, cx, cy):
    """Camera-space surface normals via central-difference cross product."""
    H, W = depth_m.shape
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    pts = np.stack([(uu - cx) * depth_m / fx,
                    (vv - cy) * depth_m / fy,
                    depth_m.copy()], axis=-1)
    dx = np.roll(pts, -1, axis=1) - np.roll(pts, 1, axis=1)
    dy = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    n  = np.cross(dx, dy)
    return -n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6)


def discontinuity_mask(depth, threshold=DISC_THRESHOLD_M, dilate=DISC_DILATE_PX):
    """Pixels at or near a large depth jump — normals unreliable there."""
    dx = np.abs(np.roll(depth, -1, axis=1) - np.roll(depth, 1, axis=1))
    dy = np.abs(np.roll(depth, -1, axis=0) - np.roll(depth, 1, axis=0))
    disc = (dx > threshold) | (dy > threshold)
    if dilate > 0:
        disc = binary_dilation(disc, iterations=dilate)
    return disc


def angular_error_deg(n1, n2):
    dot = np.clip((n1 * n2).sum(axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def vis_normals(normals_world):
    """Visualize world-space normals as RGB: (N+1)/2, clipped to [0,1]."""
    return np.clip((normals_world + 1.0) / 2.0, 0, 1)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def stats_dict(ang_map, mask):
    a = ang_map[mask & np.isfinite(ang_map)]
    if len(a) == 0:
        return dict(n=0, mean=float('nan'), median=float('nan'),
                    p10=float('nan'), p90=float('nan'),
                    pct11=float('nan'), pct22=float('nan'))
    return dict(
        n      = len(a),
        mean   = float(a.mean()),
        median = float(np.median(a)),
        p10    = float(np.percentile(a, 10)),
        p90    = float(np.percentile(a, 90)),
        pct11  = float((a < 11.25).mean() * 100),
        pct22  = float((a < 22.5).mean()  * 100),
    )


def print_stats(label, s):
    print(f"  {label:<28}  n={s['n']:>7,}  "
          f"mean={s['mean']:5.1f}°  median={s['median']:5.1f}°  "
          f"p10={s['p10']:4.1f}°  p90={s['p90']:5.1f}°  "
          f"<11°={s['pct11']:5.1f}%  <22°={s['pct22']:5.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Validate SNA ceiling accuracy: GT depth → normals vs GT WorldNormal EXR'
    )
    parser.add_argument('--dataset', default='new2',
                        help='Dataset folder under data/ (needs camera_params.json)')
    parser.add_argument('--image', type=int, default=0, choices=[0, 1],
                        help='Which image pair index to use (0=original, 1=edited)')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load camera params ───────────────────────────────────────────────────
    folder  = os.path.join(PROJECT_ROOT, "data", args.dataset)
    cp_path = os.path.join(folder, "camera_params.json")
    if not os.path.exists(cp_path):
        sys.exit(f"No camera_params.json in {folder}")
    with open(cp_path) as f:
        cp = json.load(f)
    for key in ('fov_deg', 'pitch_deg', 'yaw_deg', 'roll_deg'):
        if cp.get(key) is None:
            sys.exit(f"camera_params.json missing '{key}' — "
                     f"fill in from UE Details panel first.\n"
                     f"Datasets with full params: new2, new3")

    fov_deg   = float(cp['fov_deg'])
    R_c2w     = ue_cam_to_world(cp['pitch_deg'], cp['yaw_deg'], cp['roll_deg'])

    # ── Discover files ───────────────────────────────────────────────────────
    all_files    = sorted(os.listdir(folder))
    depth_exrs   = sorted([f for f in all_files
                            if 'SceneDepth' in f and 'WorldUnits' not in f
                            and f.endswith('.exr')])
    normal_exrs  = sorted([f for f in all_files
                            if 'WorldNormal' in f and f.endswith('.exr')])
    rgb_pngs     = sorted([f for f in all_files
                            if f.endswith('.png') and 'depth' not in f.lower()])

    if args.image >= len(depth_exrs):
        sys.exit(f"--image {args.image} out of range ({len(depth_exrs)} depth EXRs found)")
    if args.image >= len(normal_exrs):
        sys.exit(f"--image {args.image} out of range ({len(normal_exrs)} WorldNormal EXRs found)")

    depth_path  = os.path.join(folder, depth_exrs[args.image])
    normal_path = os.path.join(folder, normal_exrs[args.image])
    rgb_path    = os.path.join(folder, rgb_pngs[args.image]) if args.image < len(rgb_pngs) else None

    print(f"\n=== SNA GT Validation: {args.dataset}  image={args.image} ===")
    print(f"  SceneDepth : {depth_exrs[args.image]}")
    print(f"  WorldNormal: {normal_exrs[args.image]}")
    print(f"  FOV={fov_deg}deg  pitch={cp['pitch_deg']}deg  yaw={cp['yaw_deg']}deg  roll={cp['roll_deg']}deg")

    # ── Load data ────────────────────────────────────────────────────────────
    depth   = load_scene_depth(depth_path)
    raw_wn  = load_world_normal_raw(normal_path)
    rgb_img = Image.open(rgb_path).convert('RGB') if rgb_path else None

    H, W = depth.shape
    fx   = (W / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    fy, cx, cy = fx, W / 2.0, H / 2.0

    print(f"  Image size: {W}x{H},  fx={fx:.1f}")
    print(f"\n  WorldNormal EXR raw value range:")
    print(f"    R: [{raw_wn[...,0].min():.4f}, {raw_wn[...,0].max():.4f}]")
    print(f"    G: [{raw_wn[...,1].min():.4f}, {raw_wn[...,1].max():.4f}]")
    print(f"    B: [{raw_wn[...,2].min():.4f}, {raw_wn[...,2].max():.4f}]")

    # ── Compute derived normals ──────────────────────────────────────────────
    n_cam   = normals_from_depth(depth, fx, fy, cx, cy)
    n_world = (R_c2w @ n_cam.reshape(-1, 3).T).T.reshape(H, W, 3)
    n_world = n_world / (np.linalg.norm(n_world, axis=-1, keepdims=True) + 1e-6)

    # ── Masks ────────────────────────────────────────────────────────────────
    sky_mask  = depth > SKY_THRESHOLD_M
    disc_mask = discontinuity_mask(depth)
    valid_all = ~sky_mask & (depth > 0.1)
    valid_clean = valid_all & ~disc_mask  # exclude discontinuities

    # ── Test three decodings ─────────────────────────────────────────────────
    MODES = [
        ('std',    'raw*2-1  (standard UE)'),
        ('pi',     'raw/pi-1 (current code)'),
        ('shift',  'raw-1    (shift by 1)'),
        ('direct', 'raw      (signed float)'),
    ]

    results = {}
    print(f"\n{'-'*85}")
    print(f"  Decoding comparison (all valid pixels, including discontinuities):")
    print(f"{'-'*85}")

    for mode, label in MODES:
        gt_n  = decode_world_normal(raw_wn, mode)
        ang   = angular_error_deg(n_world, gt_n)
        s_all   = stats_dict(ang, valid_all)
        s_clean = stats_dict(ang, valid_clean)
        results[mode] = dict(gt_n=gt_n, ang=ang, s_all=s_all, s_clean=s_clean, label=label)
        print(f"\n  [{label}]")
        print_stats("all valid",             s_all)
        print_stats("clean (no disc/sky)",   s_clean)

    print(f"{'-'*85}")

    # Identify best decoding by mean error on clean pixels
    best_mode = min(results, key=lambda m: results[m]['s_clean']['mean'])
    print(f"\n  Best decoding (lowest mean on clean pixels): [{results[best_mode]['label']}]")

    # ── Camera-space sanity check ────────────────────────────────────────────
    # Compare camera-space derived normals to best decoded GT normals rotated
    # back to camera space — isolates rotation error from decoding error
    R_w2c = R_c2w.T
    gt_n_best = results[best_mode]['gt_n']
    gt_n_cam  = (R_w2c @ gt_n_best.reshape(-1, 3).T).T.reshape(H, W, 3)
    gt_n_cam  = gt_n_cam / (np.linalg.norm(gt_n_cam, axis=-1, keepdims=True) + 1e-6)
    ang_camspace = angular_error_deg(n_cam, gt_n_cam)
    s_cam = stats_dict(ang_camspace, valid_clean)
    print(f"\n  Camera-space check (same decoding, no R_c2w applied):")
    print_stats("  clean",  s_cam)
    print("  (If this is ~same as world-space error, the rotation matrix is fine;")
    print("   if much lower, there's a rotation mismatch.)")

    # ── Figure ───────────────────────────────────────────────────────────────
    # Layout: 3 rows x 4 cols
    #  Row 0: RGB | GT depth | derived normals | GT WN (best decoding)
    #  Row 1: disc mask | err(std) | err(pi) | err(shift)
    #  Row 2: err(direct) | err(best,clean) | histogram | text stats

    best_ang = results[best_mode]['ang']
    vmax_err = float(np.nanpercentile(best_ang[valid_all], 95)) if valid_all.any() else 45.0
    vmax_err = max(vmax_err, 5.0)

    fig, axes = plt.subplots(3, 4, figsize=(22, 15))
    plt.subplots_adjust(hspace=0.22, wspace=0.08)

    # Row 0
    if rgb_img is not None:
        axes[0, 0].imshow(rgb_img)
    else:
        axes[0, 0].text(0.5, 0.5, 'No RGB', ha='center', va='center',
                        transform=axes[0, 0].transAxes)
    axes[0, 0].set_title('RGB Image', fontsize=10); axes[0, 0].axis('off')

    im = axes[0, 1].imshow(depth, cmap='turbo',
                            vmin=np.nanpercentile(depth[valid_all], 1) if valid_all.any() else 0,
                            vmax=np.nanpercentile(depth[valid_all], 99) if valid_all.any() else 100)
    axes[0, 1].set_title('GT SceneDepth (m)', fontsize=10); axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04, label='m')

    axes[0, 2].imshow(vis_normals(n_world))
    axes[0, 2].set_title('Derived normals (world)\n[GT depth + R_cam2world]', fontsize=10)
    axes[0, 2].axis('off')

    axes[0, 3].imshow(np.clip(vis_normals(results[best_mode]['gt_n']), 0, 1))
    axes[0, 3].set_title(f'GT WorldNormal EXR\n[{results[best_mode]["label"]}]', fontsize=10)
    axes[0, 3].axis('off')

    # Row 1 — disc mask + first 3 mode error maps
    disc_vis = np.zeros((H, W, 3), dtype=np.uint8)
    disc_vis[valid_clean]           = [100, 200, 100]
    disc_vis[valid_all & disc_mask] = [255, 100,  50]
    disc_vis[sky_mask]              = [ 50,  50, 150]
    axes[1, 0].imshow(disc_vis)
    n_disc = int((valid_all & disc_mask).sum())
    axes[1, 0].set_title(
        f'Pixel mask\nGreen=clean  Orange=disc({n_disc:,})  Blue=sky', fontsize=10)
    axes[1, 0].axis('off')

    for col, (mode, _) in enumerate(MODES[:3]):
        ang = results[mode]['ang']
        s   = results[mode]['s_clean']
        im  = axes[1, col + 1].imshow(
            np.where(valid_clean, ang, np.nan),
            cmap='hot', vmin=0, vmax=vmax_err)
        axes[1, col + 1].set_title(
            f'Error [{results[mode]["label"]}]\n'
            f'mean={s["mean"]:.1f}  med={s["median"]:.1f}  <11={s["pct11"]:.0f}%',
            fontsize=9)
        axes[1, col + 1].axis('off')
        plt.colorbar(im, ax=axes[1, col + 1], fraction=0.046, pad=0.04, label='deg')

    # Row 2 — 4th mode, best clean, histogram, text
    mode4, _ = MODES[3]
    ang4 = results[mode4]['ang']
    s4   = results[mode4]['s_clean']
    im = axes[2, 0].imshow(
        np.where(valid_clean, ang4, np.nan),
        cmap='hot', vmin=0, vmax=vmax_err)
    axes[2, 0].set_title(
        f'Error [{results[mode4]["label"]}]\n'
        f'mean={s4["mean"]:.1f}  med={s4["median"]:.1f}  <11={s4["pct11"]:.0f}%',
        fontsize=9)
    axes[2, 0].axis('off')
    plt.colorbar(im, ax=axes[2, 0], fraction=0.046, pad=0.04, label='deg')

    s = results[best_mode]['s_clean']
    im = axes[2, 1].imshow(
        np.where(valid_clean, best_ang, np.nan),
        cmap='hot', vmin=0, vmax=vmax_err)
    axes[2, 1].set_title(
        f'BEST [{results[best_mode]["label"]}] clean\n'
        f'mean={s["mean"]:.1f}  med={s["median"]:.1f}  <11={s["pct11"]:.0f}%',
        fontsize=9)
    axes[2, 1].axis('off')
    plt.colorbar(im, ax=axes[2, 1], fraction=0.046, pad=0.04, label='deg')

    ax_hist = axes[2, 2]
    bins = np.linspace(0, 90, 91)
    for mode, label_str in MODES:
        a = results[mode]['ang'][valid_clean]
        a = a[np.isfinite(a)]
        ax_hist.hist(a, bins=bins, alpha=0.55, label=label_str, density=True)
    ax_hist.axvline(11.25, color='k', lw=0.8, linestyle='--', label='11.25 deg')
    ax_hist.axvline(22.5,  color='k', lw=0.8, linestyle=':',  label='22.5 deg')
    ax_hist.set_xlabel('Angular error (deg)')
    ax_hist.set_ylabel('Density')
    ax_hist.set_title('Error histogram (clean pixels)')
    ax_hist.legend(fontsize=7)
    ax_hist.set_xlim(0, 90)

    ax_txt = axes[2, 3]
    ax_txt.axis('off')
    lines = [
        f'Dataset: {args.dataset}  image {args.image}',
        f'FOV:{fov_deg} pitch:{cp["pitch_deg"]} yaw:{cp["yaw_deg"]}',
        f'Size:{W}x{H}  fx={fx:.1f}px',
        f'',
        f'Valid all:    {valid_all.sum():>8,}',
        f'Clean(nodis): {valid_clean.sum():>8,}',
        f'Disc excl:    {(valid_all & disc_mask).sum():>8,}',
        f'',
        f'BEST: {results[best_mode]["label"]}',
        f'',
    ]
    for mode, lbl in MODES:
        s = results[mode]['s_clean']
        lines += [
            f'[{lbl}]',
            f'  mean={s["mean"]:.1f} med={s["median"]:.1f}',
            f'  <11={s["pct11"]:.0f}%  <22={s["pct22"]:.0f}%',
            f'',
        ]
    ax_txt.text(0.02, 0.98, '\n'.join(lines),
                transform=ax_txt.transAxes, fontsize=8,
                va='top', ha='left', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

    plt.suptitle(
        f'SNA GT Validation -- {args.dataset} image {args.image}\n'
        f'GT SceneDepth -> finite-diff normals -> R_cam2world  vs  GT WorldNormal EXR\n'
        f'(ceiling accuracy of depth-derived normals with perfect depth)',
        fontsize=12, fontweight='bold')

    out_path = os.path.join(OUTPUT_DIR, f'{args.dataset}_sna_gt_validation_img{args.image}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
