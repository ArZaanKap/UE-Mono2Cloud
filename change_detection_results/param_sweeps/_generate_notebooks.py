from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n") + "\n",
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n") + "\n",
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_SETUP = """
import gc
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

cwd = Path.cwd().resolve()
PROJECT_ROOT = None
for candidate in (cwd, *cwd.parents):
    if (candidate / 'data').exists() and (candidate / 'change_detection_results').exists():
        PROJECT_ROOT = candidate.resolve()
        break
if PROJECT_ROOT is None:
    raise FileNotFoundError(
        f"Could not locate the repo root from cwd={cwd}. "
        "Expected a parent directory containing both 'data' and 'change_detection_results'."
    )

CHANGE_DIR = PROJECT_ROOT / 'change_detection_results'
SWEEP_DIR = CHANGE_DIR / 'param_sweeps'
for path in (CHANGE_DIR, SWEEP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device.upper()}')
"""


DINO_NOTEBOOK = notebook([
    md("""
    # DINOv3 Parameter Sweep

    Sweeps the same four post-processing knobs as the DINOv2 notebook, but runs them
    against the DINOv3 baseline configured in `params.py`.

    | Param | What it does |
    |---|---|
    | `threshold` | Feature-distance cutoff. Lower = bigger mask. |
    | `sigma` | Gaussian smoothing on the upsampled patch map. Higher = less blocky, but can over-spread changes. |
    | `min_area` | Removes tiny connected components. Lower = keeps finer detail. |
    | `dilate_iter` | Expands / smooths surviving regions. Higher = fills gaps but can overgrow edges. |
    """),
    code(COMMON_SETUP),
    code("""
from params import DINOV3_BASELINE
from sweep_utils import load_dataset_pair, show_image_pair, show_sweep
from test_change_detection import dinov2_feature_mask

DATASET = 'new3'   # <- change dataset here

project_root, data_dir, original_img, edited_img, original_path, edited_path = load_dataset_pair(DATASET)
print(f'Dataset: {DATASET}')
print(f'Using model: {DINOV3_BASELINE["model_name"]}')
print(DINOV3_BASELINE)
"""),
    code("""
show_image_pair(original_img, edited_img, title=f'{DATASET}: original vs edited')
"""),
    md("""
    ## Sweep 1 - `threshold`

    Other params held at the DINOv3 baseline from `params.py`.
    """),
    code("""
thr_values = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
thr_results = {}

for value in thr_values:
    mask, diff_map = dinov2_feature_mask(
        original_img, edited_img,
        threshold=value,
        sigma=DINOV3_BASELINE['sigma'],
        min_area=DINOV3_BASELINE['min_area'],
        dilate_iter=DINOV3_BASELINE['dilate_iter'],
        model_name=DINOV3_BASELINE['model_name'],
    )
    thr_results[f'thr={value:.2f}'] = {
        'mask': mask,
        'diff_map': diff_map,
        'subtitle': f"sigma={DINOV3_BASELINE['sigma']}  area={DINOV3_BASELINE['min_area']}  dilate={DINOV3_BASELINE['dilate_iter']}",
    }

show_sweep(
    thr_results,
    edited_img,
    title=f'DINOv3 threshold sweep - {DATASET}',
    baseline_key=f"thr={DINOV3_BASELINE['threshold']:.2f}",
)
"""),
    md("""
    ## Sweep 2 - `sigma`

    Other params: threshold, min-area and dilation stay at the DINOv3 baseline.
    """),
    code("""
sigma_values = [0, 1, 2, 4, 6, 8]
sigma_results = {}

for value in sigma_values:
    mask, diff_map = dinov2_feature_mask(
        original_img, edited_img,
        threshold=DINOV3_BASELINE['threshold'],
        sigma=value,
        min_area=DINOV3_BASELINE['min_area'],
        dilate_iter=DINOV3_BASELINE['dilate_iter'],
        model_name=DINOV3_BASELINE['model_name'],
    )
    sigma_results[f'sigma={value}'] = {
        'mask': mask,
        'diff_map': diff_map,
        'subtitle': f"thr={DINOV3_BASELINE['threshold']:.2f}  area={DINOV3_BASELINE['min_area']}  dilate={DINOV3_BASELINE['dilate_iter']}",
    }

show_sweep(
    sigma_results,
    edited_img,
    title=f'DINOv3 sigma sweep - {DATASET}',
    baseline_key=f"sigma={DINOV3_BASELINE['sigma']}",
)
"""),
    md("""
    ## Sweep 3 - `min_area`

    Helps trim speckle noise after thresholding and smoothing.
    """),
    code("""
area_values = [50, 100, 200, 500, 1000, 2000]
area_results = {}

for value in area_values:
    mask, diff_map = dinov2_feature_mask(
        original_img, edited_img,
        threshold=DINOV3_BASELINE['threshold'],
        sigma=DINOV3_BASELINE['sigma'],
        min_area=value,
        dilate_iter=DINOV3_BASELINE['dilate_iter'],
        model_name=DINOV3_BASELINE['model_name'],
    )
    area_results[f'area={value}'] = {
        'mask': mask,
        'diff_map': diff_map,
        'subtitle': f"thr={DINOV3_BASELINE['threshold']:.2f}  sigma={DINOV3_BASELINE['sigma']}  dilate={DINOV3_BASELINE['dilate_iter']}",
    }

show_sweep(
    area_results,
    edited_img,
    title=f'DINOv3 min-area sweep - {DATASET}',
    baseline_key=f"area={DINOV3_BASELINE['min_area']}",
)
"""),
    md("""
    ## Sweep 4 - `dilate_iter`

    Useful for filling holes or spreading coverage slightly after connected-component cleanup.
    """),
    code("""
dilate_values = [0, 1, 2, 3, 4, 6]
dilate_results = {}

for value in dilate_values:
    mask, diff_map = dinov2_feature_mask(
        original_img, edited_img,
        threshold=DINOV3_BASELINE['threshold'],
        sigma=DINOV3_BASELINE['sigma'],
        min_area=DINOV3_BASELINE['min_area'],
        dilate_iter=value,
        model_name=DINOV3_BASELINE['model_name'],
    )
    dilate_results[f'dilate={value}'] = {
        'mask': mask,
        'diff_map': diff_map,
        'subtitle': f"thr={DINOV3_BASELINE['threshold']:.2f}  sigma={DINOV3_BASELINE['sigma']}  area={DINOV3_BASELINE['min_area']}",
    }

show_sweep(
    dilate_results,
    edited_img,
    title=f'DINOv3 dilation sweep - {DATASET}',
    baseline_key=f"dilate={DINOV3_BASELINE['dilate_iter']}",
)
"""),
    md("""
    ## Combos

    Edit the non-baseline entries with the settings that looked best above, then re-run this cell.
    """),
    code("""
combo_configs = {
    'baseline': dict(**DINOV3_BASELINE),
    'lower-threshold': dict(**DINOV3_BASELINE, threshold=0.10),
    'less-smoothing': dict(**DINOV3_BASELINE, sigma=2),
    'stronger-dilate': dict(**DINOV3_BASELINE, dilate_iter=4),
}

combo_results = {}
for name, cfg in combo_configs.items():
    mask, diff_map = dinov2_feature_mask(
        original_img, edited_img,
        threshold=cfg['threshold'],
        sigma=cfg['sigma'],
        min_area=cfg['min_area'],
        dilate_iter=cfg['dilate_iter'],
        model_name=cfg['model_name'],
    )
    combo_results[name] = {
        'mask': mask,
        'diff_map': diff_map,
        'subtitle': (
            f"thr={cfg['threshold']:.2f}  sigma={cfg['sigma']}  "
            f"area={cfg['min_area']}  dilate={cfg['dilate_iter']}"
        ),
    }

show_sweep(combo_results, edited_img, title=f'DINOv3 combo comparison - {DATASET}', baseline_key='baseline')
"""),
    code("""
gc.collect()
if device == 'cuda':
    torch.cuda.empty_cache()
"""),
])


SAM2_NOTEBOOK = notebook([
    md("""
    # SAM2 Parameter Sweep

    SAM 2 is box-prompted here, so the meaningful knobs are the ones that create the box:
    the raw RGB-difference threshold and the amount of dilation applied before the largest
    changed component is boxed.

    | Param | What it does |
    |---|---|
    | `diff_thresh` | Pixel-difference cutoff used to find changed regions before boxing. Lower = bigger candidate box. |
    | `dilate` | Binary dilation applied before boxing. Higher = box expands around the changed region. |
    """),
    code(COMMON_SETUP),
    code("""
from params import SAM2_BASELINE
from sweep_utils import load_dataset_pair, show_image_pair, show_sweep
from test_change_detection import sam2_mask

DATASET = 'new3'   # <- change dataset here

project_root, data_dir, original_img, edited_img, original_path, edited_path = load_dataset_pair(DATASET)
checkpoint_path = project_root / SAM2_BASELINE['checkpoint']
assert checkpoint_path.exists(), f'SAM2 checkpoint not found: {checkpoint_path}'

print(f'Dataset: {DATASET}')
print(f'Checkpoint: {checkpoint_path}')
print(SAM2_BASELINE)
"""),
    code("""
show_image_pair(original_img, edited_img, title=f'{DATASET}: original vs edited')
"""),
    md("""
    ## Sweep 1 - `diff_thresh`

    Other params held at the SAM2 baseline from `params.py`.
    """),
    code("""
diff_thresh_values = [5, 10, 15, 20, 30, 40, 60]
diff_results = {}

for value in diff_thresh_values:
    mask, diff_map = sam2_mask(
        original_img, edited_img,
        checkpoint=str(checkpoint_path),
        model_cfg=SAM2_BASELINE['model_cfg'],
        diff_thresh=value,
        dilate=SAM2_BASELINE['dilate'],
    )
    diff_results[f'diff={value}'] = {
        'mask': mask,
        'diff_map': diff_map,
        'subtitle': f"dilate={SAM2_BASELINE['dilate']}",
    }

show_sweep(
    diff_results,
    edited_img,
    title=f'SAM2 diff-threshold sweep - {DATASET}',
    baseline_key=f"diff={SAM2_BASELINE['diff_thresh']}",
)
"""),
    md("""
    ## Sweep 2 - `dilate`

    This controls how much the candidate diff region expands before the box prompt is built.
    """),
    code("""
dilate_values = [0, 2, 4, 8, 12, 16]
dilate_results = {}

for value in dilate_values:
    mask, diff_map = sam2_mask(
        original_img, edited_img,
        checkpoint=str(checkpoint_path),
        model_cfg=SAM2_BASELINE['model_cfg'],
        diff_thresh=SAM2_BASELINE['diff_thresh'],
        dilate=value,
    )
    dilate_results[f'dilate={value}'] = {
        'mask': mask,
        'diff_map': diff_map,
        'subtitle': f"diff={SAM2_BASELINE['diff_thresh']}",
    }

show_sweep(
    dilate_results,
    edited_img,
    title=f'SAM2 dilation sweep - {DATASET}',
    baseline_key=f"dilate={SAM2_BASELINE['dilate']}",
)
"""),
    md("""
    ## Combos

    Edit these combinations with the settings that gave the tightest useful box above.
    """),
    code("""
combo_configs = {
    'baseline': dict(**SAM2_BASELINE),
    'sensitive-box': dict(**SAM2_BASELINE, diff_thresh=10, dilate=12),
    'tighter-box': dict(**SAM2_BASELINE, diff_thresh=25, dilate=4),
    'very-loose-box': dict(**SAM2_BASELINE, diff_thresh=8, dilate=16),
}

combo_results = {}
for name, cfg in combo_configs.items():
    mask, diff_map = sam2_mask(
        original_img, edited_img,
        checkpoint=str(checkpoint_path),
        model_cfg=cfg['model_cfg'],
        diff_thresh=cfg['diff_thresh'],
        dilate=cfg['dilate'],
    )
    combo_results[name] = {
        'mask': mask,
        'diff_map': diff_map,
        'subtitle': f"diff={cfg['diff_thresh']}  dilate={cfg['dilate']}",
    }

show_sweep(combo_results, edited_img, title=f'SAM2 combo comparison - {DATASET}', baseline_key='baseline')
"""),
    code("""
gc.collect()
if device == 'cuda':
    torch.cuda.empty_cache()
"""),
])


SAM3_NOTEBOOK = notebook([
    md("""
    # SAM3 Parameter Sweep

    SAM 3 behaves more like prompt-conditioned segmentation than a thresholded detector.
    In this setup, the practical sweep axis is **which text prompt** you give it.

    `None` means "let the VLM generate the prompt first" using the model configured in `params.py`.
    Strings and lists are passed straight through to `sam3_mask(...)`.
    """),
    code(COMMON_SETUP),
    code("""
from params import SAM3_BASELINE
from sweep_utils import load_dataset_pair, prompt_label, show_image_pair, show_sweep
from test_change_detection import sam3_mask

DATASET = 'new3'   # <- change dataset here

project_root, data_dir, original_img, edited_img, original_path, edited_path = load_dataset_pair(DATASET)
checkpoint_path = project_root / SAM3_BASELINE['checkpoint']
assert checkpoint_path.exists(), f'SAM3 checkpoint not found: {checkpoint_path}'

baseline_prompt = SAM3_BASELINE['text_prompt']
baseline_parts = [part.strip() for part in str(baseline_prompt).split(',') if part.strip()] if baseline_prompt is not None else []

print(f'Dataset: {DATASET}')
print(f'Checkpoint: {checkpoint_path}')
print(SAM3_BASELINE)
print(f'Baseline prompt parts: {baseline_parts}')
"""),
    code("""
show_image_pair(original_img, edited_img, title=f'{DATASET}: original vs edited')
"""),
    md("""
    ## Sweep 1 - prompt source comparison

    This compares VLM-generated prompting against the baseline prompt string and a few
    variants derived from it automatically.
    """),
    code("""
prompt_candidates = [None]
if baseline_prompt is not None:
    prompt_candidates.append(baseline_prompt)
if len(baseline_parts) == 1:
    prompt_candidates.append(baseline_parts[0])
elif len(baseline_parts) > 1:
    prompt_candidates.extend(baseline_parts)
    prompt_candidates.append(baseline_parts)
    prompt_candidates.append(', '.join(reversed(baseline_parts)))

# preserve order while dropping duplicates
deduped = []
seen = set()
for prompt in prompt_candidates:
    key = repr(prompt)
    if key not in seen:
        seen.add(key)
        deduped.append(prompt)

prompt_results = {}
for prompt in deduped:
    mask, diff_map = sam3_mask(
        original_img, edited_img,
        checkpoint=str(checkpoint_path),
        text_prompt=prompt,
        vlm_model=SAM3_BASELINE['vlm_model'],
    )
    label = prompt_label(prompt)
    prompt_results[label] = {
        'mask': mask,
        'diff_map': diff_map,
        'subtitle': 'prompt source',
    }

show_sweep(
    prompt_results,
    edited_img,
    title=f'SAM3 prompt-source comparison - {DATASET}',
    baseline_key=prompt_label(SAM3_BASELINE['text_prompt']),
)
"""),
    md("""
    ## Sweep 2 - custom prompt bank

    Edit `CUSTOM_PROMPTS` with nouns or short phrases that match the type of edit in your dataset.
    """),
    code("""
CUSTOM_PROMPTS = [
    SAM3_BASELINE['text_prompt'],
    'changed object',
    'edited object',
    'new object, removed object',
]

custom_results = {}
for prompt in CUSTOM_PROMPTS:
    mask, diff_map = sam3_mask(
        original_img, edited_img,
        checkpoint=str(checkpoint_path),
        text_prompt=prompt,
        vlm_model=SAM3_BASELINE['vlm_model'],
    )
    label = prompt_label(prompt)
    custom_results[label] = {
        'mask': mask,
        'diff_map': diff_map,
        'subtitle': 'manual prompt',
    }

show_sweep(
    custom_results,
    edited_img,
    title=f'SAM3 custom prompt sweep - {DATASET}',
    baseline_key=prompt_label(SAM3_BASELINE['text_prompt']),
)
"""),
    md("""
    ## Combos

    Keep your best prompt candidates here for quick re-comparison after you tweak the custom bank.
    """),
    code("""
best_prompts = {
    'baseline': SAM3_BASELINE['text_prompt'],
    'auto-vlm': None,
}

if baseline_parts:
    best_prompts['parts-list'] = baseline_parts if len(baseline_parts) > 1 else baseline_parts[0]

combo_results = {}
for name, prompt in best_prompts.items():
    mask, diff_map = sam3_mask(
        original_img, edited_img,
        checkpoint=str(checkpoint_path),
        text_prompt=prompt,
        vlm_model=SAM3_BASELINE['vlm_model'],
    )
    combo_results[name] = {
        'mask': mask,
        'diff_map': diff_map,
        'subtitle': prompt_label(prompt),
    }

show_sweep(combo_results, edited_img, title=f'SAM3 combo comparison - {DATASET}', baseline_key='baseline')
"""),
    code("""
gc.collect()
if device == 'cuda':
    torch.cuda.empty_cache()
"""),
])


def main():
    outputs = {
        "dinov3_param_sweep.ipynb": DINO_NOTEBOOK,
        "sam2_param_sweep.ipynb": SAM2_NOTEBOOK,
        "sam3_param_sweep.ipynb": SAM3_NOTEBOOK,
    }

    for name, data in outputs.items():
        path = ROOT / name
        path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
