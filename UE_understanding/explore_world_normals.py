"""
explore_world_normals.py
========================
Explore WorldNormal EXR files from new0 / new1 datasets and evaluate
their utility as an additional metric for the compare_edit_depth pipeline.

Three main questions:
  1. What does the raw WorldNormal data look like and how is it encoded?
  2. Does the normal angular difference between frames make a useful GT
     change signal, complementary to the depth-diff mask?
  3. Normal Consistency Error (NCE): after calibrating depth, back-project
     to 3D, estimate surface normals in camera space, and compare them to
     normals from GT depth.  This captures surface-shape accuracy
     (orientation) independently of depth magnitude.

ENCODING NOTE (UE HighresScreenshot, linear EXR):
  raw in [0, 2*pi]  =>  N = raw / pi - 1  (maps to approx [-1, 1])
  Then re-normalise each pixel vector to unit length.

NCE COORDINATE NOTE:
  GT WorldNormal is in world space; normals estimated from a depth map via
  back-projection are in camera space.  These cannot be compared directly
  without the camera-to-world rotation matrix.
  Instead we define NCE purely in camera space:
      NCE = angular_error( normals_from_depth(depth_pred),
                           normals_from_depth(depth_gt) )
  When depth_pred == depth_gt this is 0; it grows as depth errors cause
  wrong surface curvature / edge orientation.  No rotation matrix needed.

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
from PIL import Image

matplotlib.use("Agg")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "outputs")

GT_TO_METERS = 10000.0 / 100.0   # UE raw SceneDepth R-channel -> metres

# Camera intrinsics: UE HighresScreenshot default horizontal FOV is 90 deg
DEFAULT_FOV_DEG = 90.0

# Change detection thresholds
DEPTH_CHANGE_THRESHOLD_M    = 0.05   # same as compare_edit_depth2 default
NORMAL_CHANGE_THRESHOLD_DEG = 5.0


# ---------------------------------------------------------------------------
# EXR loading helpers
# ---------------------------------------------------------------------------

def _exr_open(path):
    exr = OpenEXR.InputFile(path)
    dw  = exr.header()['dataWindow']
    w   = dw.max.x - dw.min.x + 1
    h   = dw.max.y - dw.min.y + 1
    return exr, h, w


def load_world_normal_raw(path):
    """Load WorldNormal EXR -> raw (H, W, 3) float32, channels R/G/B."""
    exr, h, w = _exr_open(path)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    ch = []
    for c in ['R', 'G', 'B']:
        buf = exr.channel(c, FLOAT)
        ch.append(np.frombuffer(buf, dtype=np.float32).reshape(h, w).copy())
    return np.stack(ch, axis=-1)


def decode_world_normal(raw):
    """
    Decode UE WorldNormal EXR to unit surface normal vectors in world space.

    Encoding: raw = (N + 1) * pi   =>   N = raw / pi - 1
    Max raw = 2*pi when N = 1.  Decoded |N| ~ 1.04, so we re-normalise.
    """
    N   = raw / np.pi - 1.0
    mag = np.linalg.norm(N, axis=-1, keepdims=True)
    return N / (mag + 1e-6)


def load_scene_depth(path):
    """Load SceneDepth EXR -> (H, W) float32 in metres."""
    exr, h, w = _exr_open(path)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    header_channels = list(exr.header()['channels'].keys())
    for chan in ['R', 'SceneDepth', 'Z']:
        if chan in header_channels:
            buf = exr.channel(chan, FLOAT)
            arr = np.frombuffer(buf, dtype=np.float32).reshape(h, w).copy()
            return arr * GT_TO_METERS
    raise ValueError(f"No depth channel in {path}")


def load_rgb(path):
    """Load PNG or EXR -> (H, W, 3) uint8."""
    if path.lower().endswith('.png'):
        return np.array(Image.open(path).convert('RGB'))
    exr, h, w = _exr_open(path)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    ch = []
    for c in ['R', 'G', 'B']:
        buf = exr.channel(c, FLOAT)
        ch.append(np.frombuffer(buf, dtype=np.float32).reshape(h, w).copy())
    img = np.stack(ch, axis=-1)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Dataset file discovery
# ---------------------------------------------------------------------------

def find_files(dataset):
    folder = os.path.join(PROJECT_ROOT, "data", dataset)
    files  = sorted(os.listdir(folder))

    normal_exrs = sorted([f for f in files if 'WorldNormal' in f and f.endswith('.exr')])
    depth_exrs  = sorted([f for f in files
                          if 'SceneDepth' in f and 'WorldUnits' not in f and f.endswith('.exr')])
    rgb_pngs    = sorted([f for f in files if f.endswith('.png')])

    if len(normal_exrs) < 2:
        raise FileNotFoundError(f"Need 2 WorldNormal EXRs in {folder}, found {len(normal_exrs)}")
    if len(depth_exrs) < 2:
        raise FileNotFoundError(f"Need 2 SceneDepth EXRs in {folder}, found {len(depth_exrs)}")

    return {
        'folder':  folder,
        'normal0': os.path.join(folder, normal_exrs[0]),
        'normal1': os.path.join(folder, normal_exrs[1]),
        'depth0':  os.path.join(folder, depth_exrs[0]),
        'depth1':  os.path.join(folder, depth_exrs[1]),
        'rgb0':    os.path.join(folder, rgb_pngs[0]) if len(rgb_pngs) > 0 else None,
        'rgb1':    os.path.join(folder, rgb_pngs[1]) if len(rgb_pngs) > 1 else None,
    }


# ---------------------------------------------------------------------------
# Normal utilities
# ---------------------------------------------------------------------------

def normal_to_color(n):
    """Unit normal (H,W,3) in [-1,1] -> RGB uint8 for display (standard map)."""
    return ((n * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)


def angular_error_deg(n1, n2):
    """Per-pixel angular error (degrees) between two unit normal maps."""
    dot = np.clip((n1 * n2).sum(axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def normals_from_depth(depth_m, fx, fy, cx, cy):
    """
    Estimate per-pixel surface normals from a depth map (camera space).

    Back-projects pixels to 3D, then takes the cross-product of neighbouring
    point vectors.  Result is (H, W, 3) unit normals in camera space where
    Z is the depth direction.

    Note: these are in CAMERA space, not world space.  Use only for NCE
    (comparing two depth maps against each other), not against GT WorldNormal
    directly.
    """
    H, W = depth_m.shape
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    X = (uu - cx) * depth_m / fx
    Y = (vv - cy) * depth_m / fy
    Z = depth_m.copy()
    pts = np.stack([X, Y, Z], axis=-1)          # (H, W, 3) in camera space

    # Central finite differences (roll avoids boundary allocation)
    dx = np.roll(pts, -1, axis=1) - np.roll(pts,  1, axis=1)
    dy = np.roll(pts, -1, axis=0) - np.roll(pts,  1, axis=0)

    normals = np.cross(dx, dy)                  # (H, W, 3)
    mag     = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / (mag + 1e-6)


def nce_stats(depth_pred, depth_gt, valid_mask, fx, fy, cx, cy):
    """
    Normal Consistency Error (NCE) — camera space only.

    Computes normals from both depth maps and returns per-pixel angular
    error statistics over valid_mask pixels.

    Interpretation:
      ~0 deg  : predicted depth reproduces the same surface shape as GT
      high deg: model flattens curved surfaces or smears geometry at edges

    In compare_edit_depth2:
      depth_pred = depth_scaled (model output, calibrated)
      depth_gt   = depth_gt_edit (UE ground truth for edited frame)
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
    parser = argparse.ArgumentParser(description='WorldNormal EXR exploration')
    parser.add_argument('--dataset', default='new0', choices=['new0', 'new1'])
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = find_files(args.dataset)

    print(f"\n{'='*65}")
    print(f"WorldNormal Exploration  --  dataset: {args.dataset}")
    print(f"{'='*65}")
    print(f"  Normal orig : {os.path.basename(files['normal0'])}")
    print(f"  Normal edit : {os.path.basename(files['normal1'])}")
    print(f"  Depth  orig : {os.path.basename(files['depth0'])}")
    print(f"  Depth  edit : {os.path.basename(files['depth1'])}")

    # ── Load everything ────────────────────────────────────────────────────
    raw0   = load_world_normal_raw(files['normal0'])
    raw1   = load_world_normal_raw(files['normal1'])
    norm0  = decode_world_normal(raw0)
    norm1  = decode_world_normal(raw1)
    depth0 = load_scene_depth(files['depth0'])
    depth1 = load_scene_depth(files['depth1'])
    rgb0   = load_rgb(files['rgb0']) if files['rgb0'] else None
    rgb1   = load_rgb(files['rgb1']) if files['rgb1'] else None

    H, W = depth0.shape

    # ── Raw encoding report ────────────────────────────────────────────────
    print(f"\n--- Raw EXR channel stats (before decode) ---")
    for i, c in enumerate('RGB'):
        v = raw0[..., i]
        print(f"  raw {c}: [{v.min():.4f}, {v.max():.4f}]  mean={v.mean():.4f}")
    print(f"  Encoding: N = raw / pi - 1  (max raw ~= 2*pi = 6.28)")
    mag_before_renorm = np.linalg.norm(raw0 / np.pi - 1, axis=-1)
    print(f"  Decoded |N| before re-norm: mean={mag_before_renorm.mean():.4f}  (target 1.0)")

    print(f"\n--- Decoded unit normals (world space) ---")
    for i, c in enumerate('XYZ'):
        v = norm0[..., i]
        print(f"  N.{c}: [{v.min():.3f}, {v.max():.3f}]  mean={v.mean():.3f}")
    print(f"  UE convention: X=right, Y=forward, Z=up")
    print(f"  Z mean={norm0[...,2].mean():.3f} > 0 => most surfaces face upward (floor + ceiling dominate)")

    # ── Change signals ─────────────────────────────────────────────────────
    depth_diff = np.abs(depth1 - depth0)
    ang_diff   = angular_error_deg(norm0, norm1)   # world-space GT normals

    sky_mask       = (depth1 > 500) | (depth0 > 500)
    depth_changed  = (depth_diff > DEPTH_CHANGE_THRESHOLD_M)    & ~sky_mask
    normal_changed = (ang_diff   > NORMAL_CHANGE_THRESHOLD_DEG) & ~sky_mask

    both   = depth_changed & normal_changed
    d_only = depth_changed  & ~normal_changed
    n_only = ~depth_changed & normal_changed

    print(f"\n--- Change signal comparison (frame 0 vs frame 1) ---")
    print(f"  Depth changed  (>{DEPTH_CHANGE_THRESHOLD_M}m)  : {depth_changed.mean()*100:.2f}%")
    print(f"  Normal changed (>{NORMAL_CHANGE_THRESHOLD_DEG} deg): {normal_changed.mean()*100:.2f}%")
    print(f"  Both agree              : {both.mean()*100:.2f}%")
    print(f"  Depth-only change       : {d_only.mean()*100:.2f}%   (depth moves, normal stable)")
    print(f"  Normal-only change      : {n_only.mean()*100:.2f}%   (normal flips, depth barely moves)")
    print(f"  Ang err on changed px   : mean={ang_diff[depth_changed].mean():.1f} deg"
          f"  median={np.median(ang_diff[depth_changed]):.1f} deg")
    print(f"  Ang err on unchanged px : mean={ang_diff[~depth_changed & ~sky_mask].mean():.1f} deg"
          f"  (noise floor)")
    print(f"\n  Key insight: normal-only region ({n_only.mean()*100:.1f}%) = contact/shadow edges")
    print(f"  where an object touches original geometry.  Depth barely changes but")
    print(f"  normal direction flips.  Combined (depth OR normal) mask is more complete.")

    # ── NCE prototype (camera space) ───────────────────────────────────────
    fov_rad = np.radians(DEFAULT_FOV_DEG)
    fx      = (W / 2.0) / np.tan(fov_rad / 2.0)
    fy      = fx
    cx, cy  = W / 2.0, H / 2.0

    valid = (~sky_mask) & (depth1 > 0.1) & (depth0 > 0.1)

    # Validation pass: compare normals-from-GT-depth0 vs normals-from-GT-depth1.
    # On unchanged pixels depth is the same, so NCE ~ 0 deg.
    # On changed pixels depth differs, so NCE is large.
    # This validates the metric works before applying to predicted depth.
    nce_gt = nce_stats(depth0, depth1, valid, fx, fy, cx, cy)

    print(f"\n--- NCE prototype (camera space, GT vs GT) ---")
    print(f"  Interpretation: normals from depth0 vs normals from depth1 (both GT).")
    print(f"  On unchanged pixels NCE should be ~0; on changed pixels it should be large.")
    unch_v = valid & ~depth_changed
    ch_v   = valid & depth_changed
    ang_map = nce_gt['ang_map']
    print(f"  NCE on unchanged pixels : mean={ang_map[unch_v].mean():.2f} deg  "
          f"median={np.median(ang_map[unch_v]):.2f} deg  (should be near 0)")
    print(f"  NCE on changed   pixels : mean={ang_map[ch_v].mean():.2f} deg  "
          f"median={np.median(ang_map[ch_v]):.2f} deg  (should be large)")
    print(f"\n  In compare_edit_depth2, replace depth0 with depth_scaled from the model.")
    print(f"  NCE then measures how well the model preserves surface shape.")

    # ── Figure 1: Overview (2x4) ──────────────────────────────────────────
    fig1, axes1 = plt.subplots(2, 4, figsize=(24, 10))
    plt.subplots_adjust(hspace=0.22, wspace=0.06)

    nc0   = normal_to_color(norm0)
    nc1   = normal_to_color(norm1)
    sky_d = float(depth1[~sky_mask].max()) if (~sky_mask).any() else 20.0

    if rgb0 is not None:
        axes1[0, 0].imshow(rgb0)
    axes1[0, 0].set_title('Original RGB', fontsize=10); axes1[0, 0].axis('off')

    if rgb1 is not None:
        axes1[0, 1].imshow(rgb1)
    axes1[0, 1].set_title('Edited RGB', fontsize=10); axes1[0, 1].axis('off')

    axes1[0, 2].imshow(nc0)
    axes1[0, 2].set_title('WorldNormal -- Original\nX->R  Y->G  Z->B', fontsize=10)
    axes1[0, 2].axis('off')

    axes1[0, 3].imshow(nc1)
    axes1[0, 3].set_title('WorldNormal -- Edited\nX->R  Y->G  Z->B', fontsize=10)
    axes1[0, 3].axis('off')

    im = axes1[1, 0].imshow(depth0, cmap='turbo', vmin=0, vmax=sky_d)
    axes1[1, 0].set_title('GT Depth -- Original (m)', fontsize=10); axes1[1, 0].axis('off')
    plt.colorbar(im, ax=axes1[1, 0], fraction=0.046, pad=0.04)

    im = axes1[1, 1].imshow(depth1, cmap='turbo', vmin=0, vmax=sky_d)
    axes1[1, 1].set_title('GT Depth -- Edited (m)', fontsize=10); axes1[1, 1].axis('off')
    plt.colorbar(im, ax=axes1[1, 1], fraction=0.046, pad=0.04)

    im = axes1[1, 2].imshow(depth_diff, cmap='hot', vmin=0, vmax=1.0)
    axes1[1, 2].set_title('Depth diff |edit - orig| (m)', fontsize=10); axes1[1, 2].axis('off')
    plt.colorbar(im, ax=axes1[1, 2], fraction=0.046, pad=0.04)

    ang_disp = np.where(~sky_mask, ang_diff, np.nan)
    im = axes1[1, 3].imshow(ang_disp, cmap='hot', vmin=0, vmax=30)
    axes1[1, 3].set_title('Normal angular diff (deg)', fontsize=10); axes1[1, 3].axis('off')
    plt.colorbar(im, ax=axes1[1, 3], fraction=0.046, pad=0.04)

    plt.suptitle(f'WorldNormal Exploration -- {args.dataset}', fontsize=13, fontweight='bold')
    out1 = os.path.join(OUTPUT_DIR, f'{args.dataset}_overview.png')
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out1}")

    # ── Figure 2: Change mask agreement ──────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(wspace=0.08)

    axes2[0].imshow(depth_changed.astype(np.uint8) * 200, cmap='Reds', vmin=0, vmax=255)
    axes2[0].set_title(
        f'Depth change mask\n(>{DEPTH_CHANGE_THRESHOLD_M}m)  '
        f'{depth_changed.mean()*100:.1f}% changed', fontsize=11)
    axes2[0].axis('off')

    axes2[1].imshow(normal_changed.astype(np.uint8) * 200, cmap='Blues', vmin=0, vmax=255)
    axes2[1].set_title(
        f'Normal change mask\n(>{NORMAL_CHANGE_THRESHOLD_DEG} deg)  '
        f'{normal_changed.mean()*100:.1f}% changed', fontsize=11)
    axes2[1].axis('off')

    # 3-colour agreement overlay
    agree = np.zeros((*depth_changed.shape, 3), dtype=np.uint8)
    agree[d_only] = [220, 50,  50]    # red: depth only
    agree[n_only] = [50,  80, 220]    # blue: normal only
    agree[both]   = [240, 140,  0]    # orange: both
    axes2[2].imshow(agree)
    axes2[2].set_title(
        'Change signal agreement\nRed=depth only  Blue=normal only  Orange=both',
        fontsize=11)
    axes2[2].axis('off')

    plt.suptitle(
        f'GT Change Masks: Depth-diff vs Normal-angular-diff -- {args.dataset}\n'
        f'Normal-only region ({n_only.mean()*100:.1f}%) = contact/shadow edges '
        f'that depth misses but normals catch',
        fontsize=12, fontweight='bold')
    out2 = os.path.join(OUTPUT_DIR, f'{args.dataset}_change_masks.png')
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"Saved: {out2}")

    # ── Figure 3: Normal channel distributions ─────────────────────────
    fig3, axes3 = plt.subplots(2, 3, figsize=(18, 8))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    for row, (n, label) in enumerate([(norm0, 'Original'), (norm1, 'Edited')]):
        for col, (comp, color, name) in enumerate(
            [(0, 'red', 'X (right)'), (1, 'green', 'Y (forward)'), (2, 'blue', 'Z (up)')]):
            ax = axes3[row, col]
            vals = n[..., comp].ravel()
            ax.hist(vals, bins=100, color=color, alpha=0.7, density=True)
            ax.set_title(f'{label} -- Normal {name}', fontsize=9)
            ax.set_xlim(-1.1, 1.1)
            ax.axvline(0, color='k', lw=0.8, ls='--')
            ax.set_ylabel('Density', fontsize=8)

    plt.suptitle(
        f'WorldNormal channel distributions -- {args.dataset}\n'
        'Z (up) skewed positive = floor + ceiling dominate;  Y (forward) symmetric = balanced walls',
        fontsize=11, fontweight='bold')
    out3 = os.path.join(OUTPUT_DIR, f'{args.dataset}_normal_distributions.png')
    plt.savefig(out3, dpi=150, bbox_inches='tight')
    print(f"Saved: {out3}")

    # ── Figure 4: NCE prototype (camera space) ───────────────────────────
    n_cam0 = normals_from_depth(depth0, fx, fy, cx, cy)
    n_cam1 = normals_from_depth(depth1, fx, fy, cx, cy)
    nce_map = np.where(valid, ang_map, np.nan)

    fig4, axes4 = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(wspace=0.08)

    axes4[0].imshow(normal_to_color(n_cam0))
    axes4[0].set_title('Normals from GT depth (orig)\ncamera space, cross-product method', fontsize=10)
    axes4[0].axis('off')

    axes4[1].imshow(normal_to_color(n_cam1))
    axes4[1].set_title('Normals from GT depth (edit)\ncamera space', fontsize=10)
    axes4[1].axis('off')

    im = axes4[2].imshow(nce_map, cmap='hot', vmin=0, vmax=20)
    axes4[2].set_title(
        f'NCE: angular error orig vs edit normals\n'
        f'Unchanged: mean={ang_map[unch_v].mean():.1f} deg   '
        f'Changed: mean={ang_map[ch_v].mean():.1f} deg\n'
        f'(replaces orig depth with depth_pred in production)',
        fontsize=10)
    axes4[2].axis('off')
    plt.colorbar(im, ax=axes4[2], fraction=0.046, pad=0.04, label='degrees')

    plt.suptitle(
        f'NCE Prototype (camera space) -- {args.dataset}\n'
        'In production: replace depth0 with depth_scaled from the model.\n'
        'NCE ~ 0 on flat surfaces; grows where model flattens curved geometry or smears edges.',
        fontsize=11, fontweight='bold')
    out4 = os.path.join(OUTPUT_DIR, f'{args.dataset}_nce_prototype.png')
    plt.savefig(out4, dpi=150, bbox_inches='tight')
    print(f"Saved: {out4}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("SUMMARY -- what WorldNormal adds to the pipeline")
    print(f"{'='*65}")
    print(f"""
1. RICHER GT CHANGE MASK
   Combining depth diff (>{DEPTH_CHANGE_THRESHOLD_M}m) with normal angular diff (>{NORMAL_CHANGE_THRESHOLD_DEG} deg)
   catches {n_only.mean()*100:.1f}% extra pixels that depth alone misses -- typically
   contact zones where an added object touches the original surface:
   the normal flips (e.g. floor -> object side) but depth barely moves.
   Potential use: --change-mode flag in compare_edit_depth2.

2. NORMAL CONSISTENCY ERROR (NCE) -- a new depth evaluation metric
   Definition (camera space only, no rotation matrix needed):
     n_pred = normals_from_depth(depth_scaled, fx, fy, cx, cy)
     n_gt   = normals_from_depth(depth_gt_edit, fx, fy, cx, cy)
     NCE    = mean angular_error(n_pred, n_gt)  [degrees]

   Why it matters vs existing metrics:
     MAE / RMSE  -- absolute depth accuracy (scale / magnitude)
     delta1/2/3  -- what fraction of predictions are within a scale ratio
     NCE         -- surface geometry accuracy (shape, curvature, edges)

   A model can have low MAE (global scale is right) but high NCE because
   curved surfaces are predicted as flat, or edges are smeared.  NCE
   catches this independently of scale.

   How to add to compare_edit_depth2:
     - load WorldNormal EXR via load_world_normal() from this file
     - after computing depth_scaled, call nce_stats(depth_scaled, depth_gt_edit, ...)
     - report NCE (unchanged), NCE (changed) alongside MAE / RMSE / delta1

   NCE on unchanged pixels validates calibration quality (shape of existing
   scene).  NCE on changed pixels evaluates new-object geometry recovery.
""")

    print(f"Outputs: {OUTPUT_DIR}")
    print("Done.")

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
