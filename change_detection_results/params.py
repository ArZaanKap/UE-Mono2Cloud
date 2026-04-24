# Single source of truth for change detection parameters.
# Edit these dicts - both sweep notebooks and test_change_detection.py import from here.
# Change a value here, re-run the affected notebook/script, and the new params take effect.

DINO_BASELINE = dict(
    threshold   = 0.30,
    sigma       = 4,
    min_area    = 500,
    dilate_iter = 2,
)

DINOV3_BASELINE = dict(
    threshold   = 0.12,
    sigma       = 4,
    min_area    = 500,
    dilate_iter = 2,
    model_name  = "facebook/dinov3-vitb16-pretrain-lvd1689m",
)

# Threshold is computed adaptively from the distance-map skewness (None = auto).
# All other keys are passed directly to SamAutomaticMaskGenerator.
GESCF_BASELINE = dict(
    points_per_side        = 48,
    pred_iou_thresh        = 0.55,
    stability_score_thresh = 0.65,
    min_mask_region_area   = 25,
    overlap_frac           = 0.15,
)

RGB_BASELINE = dict(
    threshold   = 25,
    min_area    = 100,
    dilate_iter = 2,
)

OFFICIAL_GESCF_BASELINE = dict(
    output_size            = 512,
    feature_facet          = 'key',
    feature_layer          = 17,
    embedding_layer        = 32,
    points_per_side        = 32,
    pred_iou_thresh        = 0.7,
    stability_score_thresh = 0.7,
)

VIEWDELTA_BASELINE = dict(
    text_prompt = "all changes",
    threshold   = 0.15,   # prob_map rarely exceeds 0.4 on UE renders; 0.5 argmax gives 0% mask
)

# model: one of dino_2Cross_CMU | dino_2Cross_PSCD | dino_2Cross_DiffCMU
# threshold: softmax probability cutoff (0-1). Lower = bigger mask.
CROSSATTN_BASELINE = dict(
    model     = 'dino_2Cross_PSCD',
    threshold = 0.50,
)

# SAM 2: box-prompted. diff_thresh/dilate control the pixel-diff bounding box.
# checkpoint: path to the .pt weights file you downloaded.
SAM2_BASELINE = dict(
    checkpoint  = 'mask_models/weights/sam2.1_hiera_large.pt',
    model_cfg   = 'configs/sam2.1/sam2.1_hiera_l.yaml',
    diff_thresh = 15.0,
    dilate      = 8,
)

# Per-dataset depth-diff threshold (metres) for GT change mask derivation.
# Used as the default when --change-threshold is not passed on the CLI.
# Set to 0.0 to treat every non-zero depth diff as a change.
DATASET_CHANGE_THRESHOLDS = {
    'new0':   0.06,
    'new1':   0.05,
    'new2':   0.0,
    'new3':   0.0,
    'new4':   0.02,
}

# DINO → SAM2: use DINO feature distance map to generate SAM2 box prompts
# (one box per significant changed region) instead of the raw pixel-diff box.
DINOV3_SAM2_BASELINE = dict(
    checkpoint     = 'mask_models/weights/sam2.1_hiera_large.pt',
    model_cfg      = 'configs/sam2.1/sam2.1_hiera_l.yaml',
    dino_threshold = 0.12,
    sigma          = 4,
    dilate         = 8,
    model_name     = 'facebook/dinov3-vitb16-pretrain-lvd1689m',
)

DINOV2_SAM2_BASELINE = dict(
    checkpoint     = 'mask_models/weights/sam2.1_hiera_large.pt',
    model_cfg      = 'configs/sam2.1/sam2.1_hiera_l.yaml',
    dino_threshold = 0.30,
    sigma          = 4,
    dilate         = 8,
    model_name     = 'facebook/dinov2-with-registers-base',
)

# SAM 3.1: text-prompted.
# checkpoint: local .pt checkpoint, or a directory containing it.
# vlm_model: HuggingFace id of the Gemma 4 VLM used to auto-generate the text prompt.
# text_prompt: set to a non-None string or list to skip the VLM entirely.
SAM3_BASELINE = dict(
    checkpoint  = 'mask_models/weights/sam3.1/sam3.1_multiplex.pt',
    vlm_model   = 'google/gemma-4-E2B-it',
    text_prompt = "hand, robot",   # None -> auto-generate via vlm_model
)
