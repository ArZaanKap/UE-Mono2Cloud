from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import laspy
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from change_detection_results.dinox_backend import dinox_change_mask
from change_detection_results.params import DINO_BASELINE, DINOX_BASELINE, DINOV3_BASELINE
from change_detection_results.test_change_detection import (
    dinov2_feature_mask,
    find_depth_pair,
    find_image_pair,
)
from compare_edit_depth.compare_edit_depth2 import (
    compute_depth_metrics,
    fit_depth_alignment,
    load_exr_depth,
    load_image,
)
from MAIN.pipeline_io import choose_output_dir, discover_input_files

MODEL_LABEL = "DA3 Giant 1.1"
DA3_HF_MODEL = "depth-anything/DA3-GIANT-1.1"
DA3_MODEL_NAME = "da3-giant"
DA3_PROCESS_RES = 0
DEFAULT_DATASET = "test2"
DEFAULT_FOV_DEG = 90.0
MIN_DEPTH_M = 0.001
MASK_MODEL_CHOICES = ("dinov2", "dinov3", "dinox")


def resize_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = shape_hw
    if mask.shape == (target_h, target_w):
        return mask.astype(bool)
    return cv2.resize(
        mask.astype(np.uint8),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def resize_depth(depth: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = shape_hw
    if depth.shape == (target_h, target_w):
        return depth.astype(np.float32)
    return cv2.resize(depth.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def load_fov(dataset_dir: Path, fallback_fov: float) -> float:
    params_path = dataset_dir / "params.json"
    if not params_path.exists():
        return fallback_fov
    with params_path.open("r", encoding="utf-8") as f:
        params = json.load(f)
    fov = params.get("fov_deg")
    return float(fov) if fov is not None else fallback_fov


def run_change_mask(
    original_img: Image.Image,
    edited_img: Image.Image,
    *,
    mask_model: str,
    dinox_text_prompt: str | None,
    dinox_token_env: str,
) -> tuple[np.ndarray, np.ndarray, str, dict]:
    if mask_model == "dinov2":
        params = dict(DINO_BASELINE)
        params["model_name"] = "facebook/dinov2-with-registers-base"
        changed_mask, dist_map = dinov2_feature_mask(
            original_img,
            edited_img,
            threshold=DINO_BASELINE["threshold"],
            sigma=DINO_BASELINE["sigma"],
            min_area=DINO_BASELINE["min_area"],
            dilate_iter=DINO_BASELINE["dilate_iter"],
            model_name=params["model_name"],
        )
        return changed_mask, dist_map, "DINOv2", params

    if mask_model == "dinov3":
        params = dict(DINOV3_BASELINE)
        changed_mask, dist_map = dinov2_feature_mask(
            original_img,
            edited_img,
            threshold=DINOV3_BASELINE["threshold"],
            sigma=DINOV3_BASELINE["sigma"],
            min_area=DINOV3_BASELINE["min_area"],
            dilate_iter=DINOV3_BASELINE["dilate_iter"],
            model_name=DINOV3_BASELINE["model_name"],
        )
        return changed_mask, dist_map, "DINOv3", params

    if mask_model == "dinox":
        params = dict(DINOX_BASELINE)
        params["token_env"] = dinox_token_env
        params["text_prompt"] = dinox_text_prompt
        changed_mask, dist_map = dinox_change_mask(
            original_img,
            edited_img,
            token_env=dinox_token_env,
            model_name=DINOX_BASELINE["model_name"],
            prompt_text=dinox_text_prompt,
            bbox_threshold=DINOX_BASELINE["bbox_threshold"],
            iou_threshold=DINOX_BASELINE["iou_threshold"],
            match_iou=DINOX_BASELINE["match_iou"],
            min_area=DINOX_BASELINE["min_area"],
            dilate_iter=DINOX_BASELINE["dilate_iter"],
        )
        return changed_mask, dist_map, "DINO-X", params

    raise ValueError(f"Unsupported mask model: {mask_model}")


def build_preview(
    original_img: Image.Image,
    edited_img: Image.Image,
    dist_map: np.ndarray,
    changed_mask: np.ndarray,
    gt_depth: np.ndarray,
    depth_scaled_gt: np.ndarray,
    out_path: Path,
    mask_label: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    axes[0].imshow(original_img)
    axes[0].set_title("Original")
    axes[1].imshow(edited_img)
    axes[1].set_title("Edited")
    im2 = axes[2].imshow(dist_map, cmap="magma")
    axes[2].set_title(f"{mask_label} score")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    overlay = np.zeros((*changed_mask.shape, 4), dtype=np.float32)
    overlay[changed_mask] = [1.0, 0.0, 0.0, 0.55]
    overlay[~changed_mask] = [0.0, 1.0, 0.0, 0.10]
    axes[3].imshow(np.array(edited_img.resize((changed_mask.shape[1], changed_mask.shape[0]))))
    axes[3].imshow(overlay)
    axes[3].set_title("Changed mask")

    vmin = float(np.nanpercentile(gt_depth, 1))
    vmax = float(np.nanpercentile(gt_depth, 99.9))
    im4 = axes[4].imshow(gt_depth, cmap="turbo", vmin=vmin, vmax=vmax)
    axes[4].set_title("GT depth")
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

    im5 = axes[5].imshow(depth_scaled_gt, cmap="turbo", vmin=vmin, vmax=vmax)
    axes[5].set_title("Calibrated depth")
    plt.colorbar(im5, ax=axes[5], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_mask_overlay(
    edited_img: Image.Image,
    changed_mask: np.ndarray,
    out_path: Path,
) -> None:
    overlay = np.array(edited_img.resize((changed_mask.shape[1], changed_mask.shape[0]))).copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    alpha = changed_mask[..., None].astype(np.float32) * 0.45
    overlay = (overlay.astype(np.float32) * (1.0 - alpha) + red.astype(np.float32) * alpha).astype(np.uint8)
    Image.fromarray(overlay).save(out_path)


def cleanup_stale_debug_outputs(output_dir: Path, debug_dir: Path, stem: str) -> None:
    stale_paths = [
        output_dir / f"{stem}_preview.png",
        output_dir / f"{stem}_mask_overlay.png",
        debug_dir / "05_changed_overlay.png",
        debug_dir / "08_gt_sky_mask.png",
        debug_dir / "09_final_sky_mask_fullres.png",
    ]
    for path in stale_paths:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass


def save_debug_artifacts(
    debug_dir: Path,
    original_img: Image.Image,
    edited_img: Image.Image,
    dist_map: np.ndarray,
    changed_mask: np.ndarray,
    gt_depth: np.ndarray,
    depth_scaled_gt: np.ndarray,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)

    original_img.save(debug_dir / "01_original.png")
    edited_img.save(debug_dir / "02_edited.png")

    plt.imsave(debug_dir / "03_dinov2_distance.png", dist_map, cmap="magma")
    plt.imsave(debug_dir / "04_changed_mask.png", changed_mask.astype(np.uint8) * 255, cmap="gray")

    vmin = float(np.nanpercentile(gt_depth, 1))
    vmax = float(np.nanpercentile(gt_depth, 99.9))
    plt.imsave(debug_dir / "06_gt_depth_original.png", gt_depth, cmap="turbo", vmin=vmin, vmax=vmax)
    plt.imsave(debug_dir / "07_calibrated_depth.png", depth_scaled_gt, cmap="turbo", vmin=vmin, vmax=vmax)


def export_las(
    edited_img: Image.Image,
    depth_full: np.ndarray,
    fov_deg: float,
    out_path: Path,
) -> int:
    rgb = np.array(edited_img)
    h, w = depth_full.shape
    if rgb.shape[:2] != (h, w):
        rgb = np.array(edited_img.resize((w, h), Image.BILINEAR))

    valid = (
        np.isfinite(depth_full)
        & (depth_full > MIN_DEPTH_M)
    )
    if not np.any(valid):
        raise ValueError("No valid pixels remain for point cloud export")

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    focal = w / (2.0 * math.tan(math.radians(fov_deg) / 2.0))

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
    las.red = (colors[:, 0].astype(np.uint16) * 257)
    las.green = (colors[:, 1].astype(np.uint16) * 257)
    las.blue = (colors[:, 2].astype(np.uint16) * 257)
    las.write(str(out_path))
    return int(z.size)


def run_da3_giant_subprocess(rgb_path: str, output_path: str, process_res: int) -> None:
    exr_loader = """
def load_exr_rgb(path):
    import OpenEXR, Imath
    from PIL import Image
    import numpy as np
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
    return Image.fromarray((img * 255).astype(np.uint8))

def load_image(path):
    from PIL import Image
    if path.lower().endswith(".exr"):
        return load_exr_rgb(path)
    return Image.open(path).convert("RGB")
"""
    script = f"""
import gc
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

torch.set_grad_enabled(False)

project_root = Path(r"{PROJECT_ROOT}")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

{exr_loader}

da3_src = project_root / "depth_models" / "Depth-Anything-3" / "src"
if str(da3_src) not in sys.path:
    sys.path.insert(0, str(da3_src))

from da3_weight_loader import load_model_streaming
from depth_anything_3.api import DepthAnything3

device = "cuda" if torch.cuda.is_available() else "cpu"
with torch.device(device):
    model = DepthAnything3(model_name="{DA3_MODEL_NAME}")
model = model.eval()

weights_path = hf_hub_download(repo_id="{DA3_HF_MODEL}", filename="model.safetensors")
load_model_streaming(model, weights_path, strict=True, log_every=50)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

img = load_image(r"{rgb_path}")
process_res = max(img.size) if {process_res} <= 0 else {process_res}
with torch.no_grad():
    prediction = model.inference([img], process_res=process_res)
depth = prediction.depth[0].astype(np.float32)
np.save(r"{output_path}", depth)
print(f"OK: shape={{depth.shape}}, range={{depth.min():.3f}}-{{depth.max():.3f}}")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"DA3 subprocess failed (exit {result.returncode}):\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MAIN pipeline: DINOv2 mask + DA3 Giant depth + LAS export"
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Folder under data/ (legacy mode)")
    parser.add_argument("--input-dir", default=None, help="Absolute or relative folder containing original RGB, original SceneDepth EXR, edited RGB, and params.json")
    parser.add_argument("--fov-deg", type=float, default=None, help="Override camera FOV in degrees")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: MAIN/pointclouds_dinov2_da3_giant)")
    parser.add_argument("--mask-model", default="dinov2", choices=MASK_MODEL_CHOICES)
    parser.add_argument("--dinox-text-prompt", default=DINOX_BASELINE["text_prompt"])
    parser.add_argument("--dinox-token-env", default=DINOX_BASELINE["token_env"])
    parser.add_argument(
        "--da3-process-res",
        type=int,
        default=DA3_PROCESS_RES,
        help="Longest-edge process resolution passed to DA3. 0 = native resolution.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else None
    if input_dir is not None:
        original_path, edited_path, gt_depth_path = discover_input_files(input_dir)
        dataset_label = input_dir.name
        params_dir = input_dir
    else:
        dataset_dir = PROJECT_ROOT / "data" / args.dataset
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")
        original_path, edited_path, gt_depth_path = discover_input_files(dataset_dir)
        dataset_label = args.dataset
        params_dir = dataset_dir

    output_dir_name = "da3g" if input_dir is not None else "pointclouds_dinov2_da3_giant"
    output_dir = choose_output_dir(PROJECT_ROOT, args.output_dir, input_dir, output_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_img = load_image(original_path)
    edited_img = load_image(edited_path)
    gt_depth = load_exr_depth(gt_depth_path)

    edited_for_mask = edited_img.resize(original_img.size, Image.BILINEAR) if edited_img.size != original_img.size else edited_img

    valid_depth_mask = np.isfinite(gt_depth) & (gt_depth > MIN_DEPTH_M)

    changed_mask, dist_map, mask_label, mask_params = run_change_mask(
        original_img,
        edited_for_mask,
        mask_model=args.mask_model,
        dinox_text_prompt=args.dinox_text_prompt,
        dinox_token_env=args.dinox_token_env,
    )
    changed_mask_gt = resize_mask(changed_mask, gt_depth.shape)
    unchanged_mask_gt = ~changed_mask_gt

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp_file:
        depth_tmp = Path(tmp_file.name)
    try:
        print(f"Running {MODEL_LABEL} on edited image...")
        print(
            f"{mask_label} params:",
            json.dumps(mask_params),
        )
        print(f"DA3 process_res={args.da3_process_res} (0 means native, matching compare_edit_depth)")
        run_da3_giant_subprocess(edited_path, str(depth_tmp), args.da3_process_res)
        pred_depth_full = np.load(depth_tmp).astype(np.float32)
    finally:
        if depth_tmp.exists():
            depth_tmp.unlink()

    pred_depth_gt = resize_depth(pred_depth_full, gt_depth.shape)
    valid_base = (
        unchanged_mask_gt
        & valid_depth_mask
        & np.isfinite(pred_depth_gt)
        & np.isfinite(gt_depth)
        & (pred_depth_gt > MIN_DEPTH_M)
    )
    if not np.any(valid_base):
        raise ValueError("No valid unchanged pixels available for LS calibration")

    pred_cap = float(np.percentile(pred_depth_gt[valid_base], 98.0))
    fit_mask = valid_base & (pred_depth_gt <= pred_cap)
    if int(fit_mask.sum()) < 16:
        fit_mask = valid_base

    scale, shift, fit_info = fit_depth_alignment(
        pred_depth_gt,
        gt_depth,
        fit_mask,
        scaling_method="ls",
        trim_keep_percent=None,
    )

    depth_scaled_gt = pred_depth_gt * scale + shift
    depth_full = resize_depth(pred_depth_full, edited_img.size[::-1]) * scale + shift

    fov_deg = float(args.fov_deg) if args.fov_deg is not None else load_fov(params_dir, DEFAULT_FOV_DEG)

    stem = f"{dataset_label}_{args.mask_model}_da3_giant"
    las_path = output_dir / "result.las"
    summary_path = output_dir / "result_summary.json"
    debug_dir = output_dir / "debug"
    preview_path = debug_dir / "00_preview.png"
    mask_overlay_path = debug_dir / "05_mask_overlay.png"
    debug_dir.mkdir(parents=True, exist_ok=True)
    cleanup_stale_debug_outputs(output_dir, debug_dir, stem)

    build_preview(
        original_img,
        edited_for_mask,
        dist_map,
        changed_mask_gt,
        gt_depth,
        depth_scaled_gt,
        preview_path,
        mask_label,
    )
    save_mask_overlay(
        edited_for_mask,
        changed_mask_gt,
        mask_overlay_path,
    )
    save_debug_artifacts(
        debug_dir,
        original_img,
        edited_for_mask,
        dist_map,
        changed_mask_gt,
        gt_depth,
        depth_scaled_gt,
    )
    point_count = export_las(edited_img, depth_full, fov_deg, las_path)

    metrics = compute_depth_metrics(depth_scaled_gt, gt_depth, valid_base)
    summary = {
        "dataset": dataset_label,
        "input_dir": str(params_dir),
        "model": MODEL_LABEL,
        "mask_model": mask_label,
        "mask_params": mask_params,
        "da3_hf_model": DA3_HF_MODEL,
        "da3_model_name": DA3_MODEL_NAME,
        "da3_process_res": args.da3_process_res,
        "scale": scale,
        "shift": shift,
        "fit_info": fit_info,
        "pred_cap_p98": pred_cap,
        "fov_deg": fov_deg,
        "changed_pixels": int(changed_mask_gt.sum()),
        "changed_fraction": float(changed_mask_gt.mean()),
        "unchanged_metrics_vs_gt": metrics,
        "point_count": point_count,
        "preview_path": str(preview_path),
        "mask_overlay_path": str(mask_overlay_path),
        "las_path": str(las_path),
        "debug_dir": str(debug_dir),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved preview: {preview_path}")
    print(f"Saved mask overlay: {mask_overlay_path}")
    print(f"Saved debug PNGs: {debug_dir}")
    print(f"Saved point cloud: {las_path}")
    print(f"Saved summary: {summary_path}")
    print(
        f"Changed={changed_mask_gt.mean() * 100:.2f}% | "
        f"scale={scale:.4f}, shift={shift:.4f}m | "
        f"points={point_count:,}"
    )


if __name__ == "__main__":
    main()
