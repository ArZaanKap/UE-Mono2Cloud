"""
Standalone DepthLab inference script — called as subprocess by compare_edit_depth2.py.

Args:
    --in-image        path to RGB image (PNG)
    --in-depth        path to sparse metric depth .npy  (0 = unknown, metric_m = known)
    --in-mask         path to inpaint mask .npy         (1 = predict/changed, 0 = known/unchanged)
    --out-depth       path to write dense metric depth .npy
    --depthlab-dir    path to cloned DepthLab repo root (for src/ imports)
    --denoise-steps   diffusion steps (default 20; paper uses 50 for best accuracy)
    --processing-res  longest-edge cap in px (0 = use input resolution)

Checkpoints are resolved via HuggingFace hub (uses local cache automatically):
    prs-eth/marigold-depth-v1-0            — already cached from Marigold-DC
    laion/CLIP-ViT-H-14-laion2B-s32B-b79K — downloaded on first run (~3.5 GB)
    Johanan0528/DepthLab                   — three .pth files downloaded on first run (~600 MB)
"""
import argparse
import os
import sys
import numpy as np
import torch
from PIL import Image


HF_MARIGOLD = 'prs-eth/marigold-depth-v1-0'
HF_CLIP     = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
HF_DEPTHLAB = 'Johanan0528/DepthLab'


def _from_pretrained(cls, model_id_or_path, **kwargs):
    try:
        return cls.from_pretrained(model_id_or_path, local_files_only=True, **kwargs)
    except Exception:
        return cls.from_pretrained(model_id_or_path, **kwargs)


def _hf_hub_download(repo_id, filename):
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo_id, filename=filename, local_files_only=True)
    except Exception:
        return hf_hub_download(repo_id, filename=filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-image',       required=True)
    parser.add_argument('--in-depth',       required=True)
    parser.add_argument('--in-mask',        required=True)
    parser.add_argument('--out-depth',      required=True)
    parser.add_argument('--depthlab-dir',   required=True)
    parser.add_argument('--denoise-steps',  type=int, default=20)
    parser.add_argument('--processing-res', type=int, default=0)
    parser.add_argument('--strength',       type=float, default=1.0)
    parser.add_argument('--normalize-scale', type=float, default=1.0)
    parser.add_argument('--no-blend',       action='store_true')
    args = parser.parse_args()

    depthlab_dir = os.path.abspath(args.depthlab_dir)
    if depthlab_dir not in sys.path:
        sys.path.insert(0, depthlab_dir)

    from diffusers import DDIMScheduler, AutoencoderKL
    from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
    from src.models.unet_2d_condition import UNet2DConditionModel
    from src.models.unet_2d_condition_main import UNet2DConditionModel_main
    from src.models.projection import My_proj
    from inference.depthlab_pipeline import DepthLabPipeline
    from utils.image_util import get_filled_for_latents

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype  = torch.float16 if device.type == 'cuda' else torch.float32

    marigold_path = os.path.join(depthlab_dir, 'checkpoints', 'marigold-depth-v1-0')
    marigold_src = marigold_path if os.path.exists(os.path.join(marigold_path, 'model_index.json')) else HF_MARIGOLD

    # Prefer local checkpoints/cache; fall back to HF hub when files are missing.
    print('Loading VAE / text encoder / tokenizer / scheduler from Marigold...')
    vae          = _from_pretrained(AutoencoderKL,  marigold_src, subfolder='vae',          torch_dtype=dtype)
    text_encoder = _from_pretrained(CLIPTextModel,  marigold_src, subfolder='text_encoder', torch_dtype=dtype)
    tokenizer    = _from_pretrained(CLIPTokenizer,  marigold_src, subfolder='tokenizer')
    scheduler    = _from_pretrained(DDIMScheduler,  marigold_src, subfolder='scheduler')

    print('Loading CLIP image encoder...')
    image_enc = _from_pretrained(CLIPVisionModelWithProjection, HF_CLIP, torch_dtype=dtype)

    print('Loading DepthLab UNets...')
    denoising_unet = _from_pretrained(
        UNet2DConditionModel_main, marigold_src, subfolder='unet',
        in_channels=12, sample_size=96,
        low_cpu_mem_usage=False, ignore_mismatched_sizes=True,
        torch_dtype=dtype,
    )
    reference_unet = _from_pretrained(
        UNet2DConditionModel, marigold_src, subfolder='unet',
        in_channels=4, sample_size=96,
        low_cpu_mem_usage=False, ignore_mismatched_sizes=True,
        torch_dtype=dtype,
    )

    print('Loading DepthLab weights (.pth)...')
    mapping_path   = _hf_hub_download(HF_DEPTHLAB, 'mapping_layer.pth')
    reference_path = _hf_hub_download(HF_DEPTHLAB, 'reference_unet.pth')
    denoising_path = _hf_hub_download(HF_DEPTHLAB, 'denoising_unet.pth')

    mapping_layer = My_proj()
    mapping_layer.load_state_dict(torch.load(mapping_path,   map_location='cpu'), strict=False)
    reference_unet.load_state_dict(torch.load(reference_path, map_location='cpu'))
    denoising_unet.load_state_dict(torch.load(denoising_path, map_location='cpu'), strict=False)
    # .pth files load in FP32; cast back to match the rest of the pipeline
    mapping_layer  = mapping_layer.to(dtype)
    reference_unet = reference_unet.to(dtype)
    denoising_unet = denoising_unet.to(dtype)

    pipe = DepthLabPipeline(
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        mapping_layer=mapping_layer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        image_enc=image_enc,
        scheduler=scheduler,
    ).to(device)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        pass

    input_image = Image.open(args.in_image)
    depth_numpy = np.load(args.in_depth).astype(np.float32)
    mask        = np.load(args.in_mask).astype(np.float32)
    mask[mask > 0.5]  = 1.0
    mask[mask <= 0.5] = 0.0

    # Fill unknown regions via NN from known depths before VAE encoding
    depth_numpy = get_filled_for_latents(mask, depth_numpy)

    with torch.no_grad():
        pipe_out = pipe(
            input_image,
            denosing_steps     = args.denoise_steps,
            processing_res     = args.processing_res,
            match_input_res    = True,
            batch_size         = 1,
            show_progress_bar  = True,
            depth_numpy_origin = depth_numpy,
            mask_origin        = mask,
            guidance_scale     = 1,
            normalize_scale    = args.normalize_scale,
            strength           = args.strength,
            blend              = not args.no_blend,
        )

    depth_pred = pipe_out.depth_np
    np.save(args.out_depth, depth_pred)
    print(f"OK: shape={depth_pred.shape}, range={depth_pred.min():.3f}-{depth_pred.max():.3f}")


if __name__ == '__main__':
    main()
