"""
SAM-based change mask generator
================================
Two models, each with a different prompting strategy:

  SAM 2  (default) -- pixel diff between the two images → bounding box of the
                       changed region → SAM 2 box-prompt segmentation.
                       Needs only the two RGB images; no text required.

  SAM 3.1          -- VLM first generates a precise text description of what
                       changed; that description is then passed as a text prompt
                       to SAM 3.1.  A VLM is needed because SAM 3.1's text
                       encoder is sensitive to prompt specificity — a vague box
                       location won't exploit its language backbone.
                       You can bypass the VLM by passing --text-prompt directly.

Usage
-----
  # SAM 2 — pixel diff → box prompt (no text needed):
  python change_detection_results/sam_mask_infer.py \\
      --orig-img data/new0/HighresScreenshot00000.png \\
      --edit-img data/new0/HighresScreenshot00001.png \\
      --out-mask /tmp/change_mask.npy \\
      --model sam2 \\
      --checkpoint /path/to/sam2.1_hiera_large.pt \\
      --model-cfg sam2.1_hiera_l.yaml

  # SAM 3.1 — auto Gemma 4 prompt → SAM 3 segmentation:
  python change_detection_results/sam_mask_infer.py \\
      --orig-img data/new0/HighresScreenshot00000.png \\
      --edit-img data/new0/HighresScreenshot00001.png \\
      --out-mask /tmp/change_mask.npy \\
      --model sam3 \\
      --checkpoint /path/to/sam3.1/weights \\
      --vlm-model google/gemma-4-E2B-it

  # SAM 3.1 — manual text prompt (skip VLM):
  python change_detection_results/sam_mask_infer.py \\
      --orig-img data/new0/HighresScreenshot00000.png \\
      --edit-img data/new0/HighresScreenshot00001.png \\
      --out-mask /tmp/change_mask.npy \\
      --model sam3 \\
      --checkpoint /path/to/sam3.1/weights \\
      --text-prompt "the new red sofa on the left"

Args
----
  --orig-img       path to original image (PNG/EXR)
  --edit-img       path to edited image   (PNG/EXR)
  --out-mask       output .npy path  (bool array, same H×W as edit-img)
  --model          sam2 | sam3  (default: sam2)
  --checkpoint     SAM 2 .pt file path  OR  SAM 3.1 local weights dir / HF repo id
  --model-cfg      SAM 2 yaml config name, e.g. sam2.1_hiera_l.yaml
  --text-prompt    explicit change description for SAM 3.1 (skips VLM if provided)
  --vlm-model      HuggingFace Gemma 4 model id for auto text generation
  --diff-thresh    max-channel diff threshold 0-255 for SAM 2 change detection (default: 15)
  --dilate         dilation iterations on raw diff mask for SAM 2 (default: 8)
  --min-area-frac  minimum changed area as fraction of image (default: 0.001)
"""

import argparse
import os
import sys
from contextlib import nullcontext
import numpy as np
from PIL import Image
from scipy import ndimage

_VLM_CACHE = {}


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_rgb(path):
    if path.lower().endswith('.exr'):
        import OpenEXR, Imath
        f = OpenEXR.InputFile(path)
        dw = f.header()['dataWindow']
        W = dw.max.x - dw.min.x + 1
        H = dw.max.y - dw.min.y + 1
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        ch = [np.frombuffer(f.channel(c, FLOAT), dtype=np.float32).reshape(H, W) for c in 'RGB']
        return Image.fromarray((np.clip(np.stack(ch, axis=-1), 0, 1) * 255).astype(np.uint8))
    return Image.open(path).convert('RGB')


# ---------------------------------------------------------------------------
# SAM 2: pixel diff → bounding box → SAM 2 box-prompt segmentation
# ---------------------------------------------------------------------------

def get_diff_bbox(orig_img, edit_img, thresh=15, dilate=8, min_area_frac=0.001):
    """Bounding box of the largest pixel-diff region. Returns (box, diff_mask)."""
    orig     = np.array(orig_img.convert('RGB'), dtype=np.float32)
    edit_arr = np.array(edit_img.resize(orig_img.size, Image.BILINEAR).convert('RGB'), dtype=np.float32)

    diff     = np.abs(orig - edit_arr).max(axis=-1)
    raw_mask = diff > thresh

    if dilate > 0:
        raw_mask = ndimage.binary_dilation(raw_mask, iterations=dilate)

    H, W    = orig.shape[:2]
    min_px  = int(H * W * min_area_frac)
    labeled, n = ndimage.label(raw_mask)

    if n == 0:
        print("[SAM] Warning: no changed region found — using full image as box.", file=sys.stderr)
        return [0, 0, W, H], raw_mask

    sizes    = ndimage.sum(raw_mask, labeled, range(1, n + 1))
    best_idx = int(np.argmax(sizes)) + 1
    if sizes[best_idx - 1] < min_px:
        print(f"[SAM] Warning: diff region too small ({sizes[best_idx-1]} px) — using full image.", file=sys.stderr)
        return [0, 0, W, H], raw_mask

    component = labeled == best_idx
    rows = np.where(np.any(component, axis=1))[0]
    cols = np.where(np.any(component, axis=0))[0]
    pad  = 12
    box  = [max(0, cols[0] - pad), max(0, rows[0] - pad),
            min(W, cols[-1] + pad), min(H, rows[-1] + pad)]
    print(f"[SAM] Diff bbox: x0={box[0]} y0={box[1]} x1={box[2]} y1={box[3]}", file=sys.stderr)
    return box, raw_mask


def run_sam2_box(edit_img, box, checkpoint, model_cfg):
    """SAM 2 image predictor with a box prompt. Returns boolean H×W mask."""
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        raise ImportError(
            "SAM 2 not installed. Run:\n"
            "  pip install git+https://github.com/facebookresearch/sam2.git\n"
            "Checkpoints: https://github.com/facebookresearch/sam2#model-description"
        )

    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))
    image_np  = np.array(edit_img.convert('RGB'))

    with torch.inference_mode():
        predictor.set_image(image_np)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array(box, dtype=np.float32),
            multimask_output=True,
        )
    return masks[int(np.argmax(scores))].astype(bool)


# ---------------------------------------------------------------------------
# VLM: generate a precise text description of what changed between two images
# ---------------------------------------------------------------------------

def _is_gemma4_vlm(vlm_model_id):
    return "gemma-4" in str(vlm_model_id).lower()


def _load_vlm(vlm_model_id):
    if vlm_model_id in _VLM_CACHE:
        return _VLM_CACHE[vlm_model_id]

    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if not _is_gemma4_vlm(vlm_model_id):
        raise ValueError(
            f"Unsupported SAM3 VLM '{vlm_model_id}'. "
            "Only Gemma 4 VLMs are supported in this repo now."
        )

    print(f"[VLM] Loading processor for {vlm_model_id} ...", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(vlm_model_id)
    print(f"[VLM] Loading {vlm_model_id} in float32 on CPU ...", file=sys.stderr)
    model = AutoModelForMultimodalLM.from_pretrained(
        vlm_model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    print(f"[VLM] Model loaded.", file=sys.stderr)

    _VLM_CACHE[vlm_model_id] = (processor, model)
    return processor, model


def _clean_vlm_description(description):
    description = str(description).strip().strip("'\"")
    if not description:
        return description

    lines = [line.strip(" -*\t") for line in description.splitlines() if line.strip()]
    if len(lines) > 1:
        description = ", ".join(lines)
    else:
        description = lines[0]

    for prefix in (
        "changed objects:",
        "objects changed:",
        "changed object:",
        "object changed:",
        "changes:",
        "changed:",
    ):
        if description.lower().startswith(prefix):
            description = description[len(prefix):].strip()
            break

    return description.replace(";", ",").strip(" ,:-")


def generate_vlm_description(orig_img, edit_img,
                              vlm_model_id='google/gemma-4-E2B-it'):
    """
    Concatenates the two images side-by-side and asks a VLM to describe
    exactly what object was added, removed, or changed.

    Returns the raw VLM output string, which is fed directly to SAM 3.1.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("transformers not installed: pip install transformers accelerate")

    W, H = orig_img.size
    canvas = Image.new('RGB', (W * 2, H))
    canvas.paste(orig_img.convert('RGB'), (0, 0))
    canvas.paste(edit_img.convert('RGB'), (W, 0))

    processor, model = _load_vlm(vlm_model_id)

    instruction = (
        "The left half is the original scene; the right half is the edited scene. "
        "Return only a comma-separated list of the objects that were added, removed, "
        "or visually changed in the right image. Do not explain your answer."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": canvas},
                {"type": "text", "text": instruction},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=80)

    input_len = inputs["input_ids"].shape[1]
    completion_ids = generated_ids[:, input_len:]
    raw_description = processor.batch_decode(completion_ids, skip_special_tokens=True)[0].strip()
    description = _clean_vlm_description(raw_description)

    print(f"[VLM] Raw output: '{raw_description}'", file=sys.stderr)
    print(f"[VLM] Prompt fed to SAM3: '{description}'", file=sys.stderr)
    return description


# ---------------------------------------------------------------------------
# SAM 3.1: text-prompted segmentation
# ---------------------------------------------------------------------------

def _resolve_sam3_checkpoint(checkpoint):
    """Accept either a direct .pt file path or a directory containing one."""
    if os.path.isfile(checkpoint):
        return checkpoint
    if os.path.isdir(checkpoint):
        candidates = sorted(
            os.path.join(checkpoint, name)
            for name in os.listdir(checkpoint)
            if name.lower().endswith(".pt")
        )
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"Could not resolve a SAM 3 checkpoint from: {checkpoint}")


def _patch_sam3_decoder_dtype(torch):
    """
    Patch the SAM 3 decoder FFN to handle mixed float32/bfloat16 tensors on CUDA.
    """
    from sam3.model.decoder import TransformerDecoderLayer

    if getattr(TransformerDecoderLayer.forward_ffn, "_ue_depth_patched", False):
        return

    def _patched_forward_ffn(self, tgt):
        weight_dtype = self.linear1.weight.dtype
        if tgt.dtype != weight_dtype:
            tgt = tgt.to(weight_dtype)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    _patched_forward_ffn._ue_depth_patched = True
    TransformerDecoderLayer.forward_ffn = _patched_forward_ffn


def _normalize_sam3_prompts(text_prompt):
    """
    Normalize SAM 3 prompts into a list of clean noun phrases.

    Accepted inputs:
      - "robot"
      - "robot, red box"
      - ["robot", "red box"]
    """
    if isinstance(text_prompt, (list, tuple)):
        raw_prompts = text_prompt
    else:
        raw_prompts = [part for part in str(text_prompt).split(",")]

    prompts = []
    for raw_prompt in raw_prompts:
        prompt = " ".join(str(raw_prompt).strip().split()).strip(" \t\r\n'\"")
        if prompt:
            prompts.append(prompt)

    return prompts


def run_sam3_text(edit_img, text_prompt, checkpoint):
    """
    SAM 3.1 text-prompted segmentation.
    text_prompt should be a specific description of the changed object,
    ideally generated by a VLM (see generate_vlm_description). A single string,
    comma-separated string, or list of strings is accepted.
    Returns boolean H×W mask.
    """
    try:
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as e:
        raise ImportError(
            f"SAM 3 import failed ({e}).\n"
            "  pip install git+https://github.com/facebookresearch/sam3.git\n"
            "Weights (gated): https://huggingface.co/facebook/sam3.1"
        ) from e

    checkpoint = _resolve_sam3_checkpoint(checkpoint)
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        _patch_sam3_decoder_dtype(torch)

    model     = build_sam3_image_model(checkpoint_path=checkpoint)
    model.to(device)
    processor = Sam3Processor(model, device=device)

    image = edit_img.convert('RGB')
    autocast_ctx = (
        torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        if device == 'cuda'
        else nullcontext()
    )
    with autocast_ctx:
        state = processor.set_image(image)

    prompts = _normalize_sam3_prompts(text_prompt)
    if not prompts:
        raise RuntimeError("SAM 3 received an empty text prompt.")
    thresholds = [0.5, 0.3]
    best_mask = None
    best_score = None
    best_prompt = None

    print(f"[SAM3] Trying prompts: {prompts}", file=sys.stderr)
    for prompt in prompts:
        for threshold in thresholds:
            processor.reset_all_prompts(state)
            processor.set_confidence_threshold(threshold)
            with autocast_ctx:
                output = processor.set_text_prompt(state=state, prompt=prompt)

            masks = output['masks']
            scores = output['scores']
            if len(masks) == 0:
                continue

            scores_np = np.asarray([
                float(score.detach().float().cpu()) if hasattr(score, "detach") else float(score)
                for score in scores
            ])
            best = int(np.argmax(scores_np))
            score = float(scores_np[best])
            print(
                f"[SAM3] prompt='{prompt}' threshold={threshold:.2f} "
                f"best_score={score:.3f}",
                file=sys.stderr,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_prompt = prompt
                best_mask = np.squeeze(masks[best].detach().cpu().numpy()).astype(bool)

    if best_mask is not None:
        print(
            f"[SAM3] Selected prompt='{best_prompt}' with score={best_score:.3f}",
            file=sys.stderr,
        )
        return best_mask

    raise RuntimeError(f"SAM 3 returned no masks for prompts: {prompts}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Generate change mask using SAM 2 or SAM 3.1')
    parser.add_argument('--orig-img',      required=True)
    parser.add_argument('--edit-img',      required=True)
    parser.add_argument('--out-mask',      required=True)
    parser.add_argument('--model',         default='sam2', choices=['sam2', 'sam3'],
                        help='sam2: box-prompted (no text needed). sam3: text-prompted (needs VLM or --text-prompt).')
    parser.add_argument('--checkpoint',    required=True,
                        help='SAM 2 .pt file path OR SAM 3.1 local weights dir / HF repo id')
    parser.add_argument('--model-cfg',     default='sam2.1_hiera_l.yaml',
                        help='SAM 2 yaml config name (ignored for sam3)')
    parser.add_argument('--text-prompt',   default=None,
                        help='Explicit change description for SAM 3.1. If omitted, --vlm-model is used to auto-generate it.')
    parser.add_argument('--vlm-model',     default='google/gemma-4-E2B-it',
                        help='HuggingFace Gemma 4 VLM used to auto-generate the SAM 3.1 text prompt '
                             '(e.g. google/gemma-4-E2B-it)')
    parser.add_argument('--diff-thresh',   type=float, default=15.0,
                        help='SAM 2 only: pixel diff threshold 0-255 (default: 15)')
    parser.add_argument('--dilate',        type=int,   default=8,
                        help='SAM 2 only: dilation iterations on diff mask (default: 8)')
    parser.add_argument('--min-area-frac', type=float, default=0.001)
    args = parser.parse_args()

    orig_img = _load_rgb(args.orig_img)
    edit_img = _load_rgb(args.edit_img)
    if edit_img.size != orig_img.size:
        edit_img = edit_img.resize(orig_img.size, Image.BILINEAR)

    if args.model == 'sam2':
        box, _ = get_diff_bbox(orig_img, edit_img,
                               thresh=args.diff_thresh,
                               dilate=args.dilate,
                               min_area_frac=args.min_area_frac)
        mask = run_sam2_box(edit_img, box, args.checkpoint, args.model_cfg)

    else:  # sam3
        text_prompt = args.text_prompt
        if text_prompt is None:
            print("[SAM3] No --text-prompt given — running VLM to generate one ...", file=sys.stderr)
            text_prompt = generate_vlm_description(orig_img, edit_img, args.vlm_model)
        else:
            print(f"[SAM3] Manual prompt input: '{text_prompt}'", file=sys.stderr)
        mask = run_sam3_text(edit_img, text_prompt, args.checkpoint)

    H_orig, W_orig = np.array(orig_img).shape[:2]
    if mask.shape != (H_orig, W_orig):
        mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
        mask = np.array(mask_pil.resize((W_orig, H_orig), Image.NEAREST)) > 127

    np.save(args.out_mask, mask)
    print(f"OK: mask shape={mask.shape}, changed={mask.sum():,} px ({mask.mean()*100:.1f}%)")


if __name__ == '__main__':
    main()
