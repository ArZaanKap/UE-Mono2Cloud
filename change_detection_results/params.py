# Single source of truth for change detection parameters.
# Edit these dicts — both sweep notebooks and test_change_detection.py import from here.
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
