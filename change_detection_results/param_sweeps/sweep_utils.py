from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display


def resolve_project_root() -> Path:
    """Return the repo root by searching upward from the current working directory."""
    cwd = Path.cwd().resolve()
    for root in (cwd, *cwd.parents):
        if (root / "data").exists() and (root / "change_detection_results").exists():
            change_dir = root / "change_detection_results"
            sweep_dir = change_dir / "param_sweeps"
            for path in (change_dir, sweep_dir):
                path_str = str(path)
                if path_str not in sys.path:
                    sys.path.insert(0, path_str)
            return root
    raise FileNotFoundError(
        f"Could not locate the repo root from cwd={cwd}. "
        "Expected a parent directory containing both 'data' and 'change_detection_results'."
    )


def load_dataset_pair(dataset: str):
    """Load the original and edited image pair for a dataset."""
    project_root = resolve_project_root()
    from test_change_detection import find_image_pair, load_image

    data_dir = project_root / "data" / dataset
    original_path, edited_path = find_image_pair(str(data_dir))
    if not original_path or not edited_path:
        raise FileNotFoundError(f"Could not find an image pair in {data_dir}")
    original_img = load_image(original_path)
    edited_img = load_image(edited_path)
    return project_root, data_dir, original_img, edited_img, Path(original_path), Path(edited_path)


def overlay_mask(image, mask, color=(255, 0, 0), alpha=0.45):
    """Return an RGB array with a semi-transparent mask overlay."""
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    mask = np.asarray(mask).astype(bool)
    out = arr.copy()
    overlay = np.zeros_like(out)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]
    out[mask] = (1 - alpha) * out[mask] + alpha * overlay[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def show_image_pair(original_img, edited_img, title=None):
    """Display the dataset pair once at the top of a notebook."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(original_img)
    axes[0].set_title("Original")
    axes[1].imshow(edited_img)
    axes[1].set_title("Edited")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    display(fig)
    plt.close(fig)


def prompt_label(prompt) -> str:
    """Compact label for SAM3 prompt sweeps."""
    if prompt is None:
        return "auto-vlm"
    if isinstance(prompt, (list, tuple)):
        return "list: " + " | ".join(str(item) for item in prompt)
    return str(prompt)


def show_sweep(results, edited_img, title, cols=4, baseline_key=None):
    """Plot edited-image overlays for a dict of named sweep results."""
    keys = list(results.keys())
    if not keys:
        print("No sweep results to display.")
        return

    rows = math.ceil(len(keys) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 4.8 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, key in zip(axes, keys):
        entry = results[key]
        mask = np.asarray(entry["mask"]).astype(bool)
        ax.imshow(overlay_mask(edited_img, mask))

        title_lines = [str(key)]
        if baseline_key is not None and key == baseline_key:
            title_lines[0] += " [baseline]"
        subtitle = entry.get("subtitle")
        if subtitle:
            title_lines.append(str(subtitle))
        title_lines.append(f"changed={mask.mean() * 100:.2f}%")
        ax.set_title("\n".join(title_lines), fontsize=10)
        ax.axis("off")

    for ax in axes[len(keys):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    display(fig)
    plt.close(fig)
