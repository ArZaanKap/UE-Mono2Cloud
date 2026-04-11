"""Compare a UE screenshot EXR against its PNG export."""

import argparse
import os

import Imath
import OpenEXR
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_FOLDER = os.path.join(PROJECT_ROOT, "data", "concrete1")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "comparison_outputs")
FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)


def find_pair(folder):
    """Find the screenshot EXR/PNG pair while ignoring depth EXRs."""
    files = os.listdir(folder)

    exr_candidates = sorted(
        f for f in files
        if f.lower().endswith(".exr") and "scenedepth" not in f.lower()
    )
    png_candidates = sorted(f for f in files if f.lower().endswith(".png"))

    if not exr_candidates:
        raise FileNotFoundError(f"No screenshot EXR found in {folder}")
    if not png_candidates:
        raise FileNotFoundError(f"No PNG found in {folder}")

    exr_name = exr_candidates[0]
    exr_stem = os.path.splitext(exr_name)[0].lower()

    matching_png = next(
        (f for f in png_candidates if os.path.splitext(f)[0].lower() == exr_stem),
        png_candidates[0],
    )

    return os.path.join(folder, exr_name), os.path.join(folder, matching_png)


def load_exr_rgb(path):
    """Load EXR RGB channels into float32 HxWx3 array."""
    exr_file = OpenEXR.InputFile(path)
    header = exr_file.header()
    data_window = header["dataWindow"]
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1
    channels = header["channels"].keys()

    rgb = []
    for channel_name in ("R", "G", "B"):
        if channel_name not in channels:
            raise ValueError(f"Missing {channel_name} channel in {path}")
        channel = np.frombuffer(
            exr_file.channel(channel_name, FLOAT),
            dtype=np.float32,
        ).reshape(height, width)
        rgb.append(channel)

    return np.stack(rgb, axis=-1)


def load_png_rgb(path):
    """Load a standard RGB image as uint8 HxWx3 array."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def compute_metrics(exr_rgb, png_rgb):
    """Compute float-space and 8-bit-space similarity statistics."""
    if exr_rgb.shape[:2] != png_rgb.shape[:2]:
        raise ValueError(
            "Image size mismatch: "
            f"EXR={exr_rgb.shape[1]}x{exr_rgb.shape[0]}, "
            f"PNG={png_rgb.shape[1]}x{png_rgb.shape[0]}"
        )

    png_float = png_rgb.astype(np.float32) / 255.0
    exr_clipped = np.clip(exr_rgb, 0.0, 1.0)
    exr_u8 = np.clip(np.rint(exr_clipped * 255.0), 0, 255).astype(np.uint8)

    diff_float = exr_clipped - png_float
    diff_u8 = exr_u8.astype(np.int16) - png_rgb.astype(np.int16)
    abs_diff_u8 = np.abs(diff_u8)

    float_rmse = float(np.sqrt(np.mean(diff_float ** 2)))
    psnr = float("inf") if float_rmse == 0 else float(20.0 * np.log10(1.0 / float_rmse))

    metrics = {
        "shape": tuple(int(v) for v in exr_rgb.shape),
        "exr_min": float(exr_rgb.min()),
        "exr_max": float(exr_rgb.max()),
        "png_min": int(png_rgb.min()),
        "png_max": int(png_rgb.max()),
        "pixels_clipped_low_pct": float(np.mean(exr_rgb < 0.0) * 100.0),
        "pixels_clipped_high_pct": float(np.mean(exr_rgb > 1.0) * 100.0),
        "float_mae": float(np.mean(np.abs(diff_float))),
        "float_rmse": float_rmse,
        "float_max_abs": float(np.max(np.abs(diff_float))),
        "psnr_db": psnr,
        "correlation": float(np.corrcoef(exr_clipped.reshape(-1), png_float.reshape(-1))[0, 1]),
        "exact_match_pct": float(np.mean(diff_u8 == 0) * 100.0),
        "within_1_level_pct": float(np.mean(abs_diff_u8 <= 1) * 100.0),
        "within_2_levels_pct": float(np.mean(abs_diff_u8 <= 2) * 100.0),
        "within_3_levels_pct": float(np.mean(abs_diff_u8 <= 3) * 100.0),
        "u8_mae": float(np.mean(abs_diff_u8)),
        "u8_max_abs": int(np.max(abs_diff_u8)),
        "per_channel_u8_mae": [float(v) for v in abs_diff_u8.reshape(-1, 3).mean(axis=0)],
        "per_channel_signed_mean": [float(v) for v in diff_u8.reshape(-1, 3).mean(axis=0)],
    }

    return metrics, exr_clipped, exr_u8, abs_diff_u8


def verdict_from_metrics(metrics):
    """Turn raw metrics into a quick human-readable verdict."""
    if metrics["exact_match_pct"] == 100.0:
        return "The EXR and PNG are bit-identical after EXR -> 8-bit conversion."
    if metrics["within_1_level_pct"] >= 99.0 and metrics["u8_max_abs"] <= 3:
        return "They are effectively the same visually, with only tiny 8-bit rounding differences."
    if metrics["within_3_levels_pct"] >= 99.0 and metrics["u8_mae"] <= 1.0:
        return "They are very close overall and should look almost the same, but they are not pixel-identical."
    if metrics["within_1_level_pct"] >= 95.0 and metrics["u8_max_abs"] <= 10:
        return "They are very close overall, but not identical."
    return "There is a noticeable difference between the EXR-derived image and the PNG."


def save_report_figure(exr_u8, png_rgb, abs_diff_u8, output_path, title):
    """Save a compact visual comparison figure."""
    max_diff_map = abs_diff_u8.max(axis=-1)
    hist_max = int(max(abs_diff_u8.max(), 5))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plt.subplots_adjust(hspace=0.2, wspace=0.1)

    axes[0, 0].imshow(exr_u8)
    axes[0, 0].set_title("EXR converted to 8-bit")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(png_rgb)
    axes[0, 1].set_title("PNG")
    axes[0, 1].axis("off")

    heatmap = axes[1, 0].imshow(max_diff_map, cmap="inferno", vmin=0, vmax=max(3, int(max_diff_map.max())))
    axes[1, 0].set_title("Per-pixel max abs diff (0-255)")
    axes[1, 0].axis("off")
    plt.colorbar(heatmap, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].hist(
        abs_diff_u8.reshape(-1),
        bins=np.arange(-0.5, hist_max + 1.5, 1.0),
        color="#3b82f6",
        edgecolor="black",
    )
    axes[1, 1].set_title("Absolute diff histogram")
    axes[1, 1].set_xlabel("Difference in 8-bit levels")
    axes[1, 1].set_ylabel("Count")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare a screenshot EXR against its PNG.")
    parser.add_argument(
        "--folder",
        default=DEFAULT_FOLDER,
        help="Folder containing the EXR/PNG pair.",
    )
    parser.add_argument("--exr", help="Optional explicit EXR path.")
    parser.add_argument("--png", help="Optional explicit PNG path.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the visual report will be saved.",
    )
    args = parser.parse_args()

    folder = os.path.abspath(args.folder)
    if args.exr and args.png:
        exr_path = os.path.abspath(args.exr)
        png_path = os.path.abspath(args.png)
    else:
        exr_path, png_path = find_pair(folder)

    exr_rgb = load_exr_rgb(exr_path)
    png_rgb = load_png_rgb(png_path)

    metrics, exr_clipped, exr_u8, abs_diff_u8 = compute_metrics(exr_rgb, png_rgb)
    verdict = verdict_from_metrics(metrics)

    os.makedirs(args.output_dir, exist_ok=True)
    pair_name = os.path.splitext(os.path.basename(exr_path))[0]
    output_path = os.path.join(args.output_dir, f"{pair_name}_exr_vs_png.png")

    title = f"{os.path.basename(exr_path)} vs {os.path.basename(png_path)}"
    save_report_figure(exr_u8, png_rgb, abs_diff_u8, output_path, title)

    print("=" * 70)
    print("UE SCREENSHOT COMPARISON: EXR vs PNG")
    print("=" * 70)
    print(f"EXR:    {exr_path}")
    print(f"PNG:    {png_path}")
    print(f"Size:   {metrics['shape'][1]} x {metrics['shape'][0]}")
    print("")
    print("EXR range before clipping:")
    print(f"  min={metrics['exr_min']:.6f}, max={metrics['exr_max']:.6f}")
    print(f"  below 0: {metrics['pixels_clipped_low_pct']:.4f}%")
    print(f"  above 1: {metrics['pixels_clipped_high_pct']:.4f}%")
    print("")
    print("Float-space comparison (EXR clipped to [0,1] vs PNG/255):")
    print(f"  MAE:       {metrics['float_mae']:.6f}")
    print(f"  RMSE:      {metrics['float_rmse']:.6f}")
    print(f"  Max abs:   {metrics['float_max_abs']:.6f}")
    print(f"  Corr:      {metrics['correlation']:.6f}")
    print(f"  PSNR:      {metrics['psnr_db']:.2f} dB")
    print("")
    print("8-bit comparison (round(EXR*255) vs PNG):")
    print(f"  Exact match:         {metrics['exact_match_pct']:.2f}%")
    print(f"  Within 1 level:      {metrics['within_1_level_pct']:.2f}%")
    print(f"  Within 2 levels:     {metrics['within_2_levels_pct']:.2f}%")
    print(f"  Within 3 levels:     {metrics['within_3_levels_pct']:.2f}%")
    print(f"  Mean abs diff:       {metrics['u8_mae']:.4f}")
    print(f"  Max abs diff:        {metrics['u8_max_abs']}")
    print(
        "  Per-channel MAE:     "
        f"R={metrics['per_channel_u8_mae'][0]:.4f}, "
        f"G={metrics['per_channel_u8_mae'][1]:.4f}, "
        f"B={metrics['per_channel_u8_mae'][2]:.4f}"
    )
    print("")
    print(f"Verdict: {verdict}")
    print(f"Report:  {output_path}")


if __name__ == "__main__":
    main()
