# TODO — Next Steps

Current best: **Depth Pro + GeSCF mask + least-squares → MAE 4.2cm, RMSE 6.5cm** (`data/depth4`)

---

## Priority 1 — Replace calibration step with diffusion-based depth completion

The literature frames our problem as **"depth completion with partial GT"** rather than monocular depth + post-hoc scale/shift. The edited pixels have no GT anchor; diffusion-based models can propagate geometry from surrounding known-depth pixels in a structure-preserving way that least-squares cannot.

### Marigold-DC ← try this first
- **Paper:** arxiv:2412.13389 (ICCV 2025)
- **Code:** https://github.com/prs-eth/Marigold-DC
- **What it does:** Diffusion depth completion with sparse GT as test-time guidance. Input: RGB + sparse known depths. Output: dense metric depth consistent with known regions.
- **How to plug in:** Feed unchanged-pixel GT depths (from UE) as sparse guide + edited RGB image → get dense depth for all pixels. Replaces steps 4–5 of current pipeline entirely.
- **No retraining needed — zero-shot.**

### DepthLab ← try second
- **Paper:** arxiv:2412.18153
- **Site:** https://johanan528.github.io/depthlab_web/
- **What it does:** Dual-branch diffusion — one branch reads RGB, one reads the known-depth region. Trained on Hypersim (synthetic indoor, close to UE renders).
- **Same drop-in role as Marigold-DC.** Benchmark both.

### Depth Anything with Any Prior ← try third
- **Paper:** ICLR 2026 (OpenReview)
- **What it does:** Conditional Depth Anything variant that accepts any partial depth prior at test time, scale-invariant log loss. Designed for exactly "I have some GT depth, fill in the rest".

---

## Priority 2 — Prompting Depth Anything for metric depth

- **Paper:** CVPR 2025
- **What it does:** Integrates sparse depth (LiDAR in their case, UE GT in ours) at multiple scales into a DPT-based model to get metric output everywhere.
- More involved integration than P1 methods but potentially higher accuracy.

---

## Priority 3 — Better change detection = better calibration anchor

Even with current least-squares approach, more accurate unchanged-pixel masks give the calibration more signal. GeSCF is already best; consider:
- Tuning GeSCF adaptive threshold per dataset
- Ensembling GeSCF + DINOv2 masks (union of confident unchanged regions)

---

## Housekeeping (carry-over, not done yet)

- Delete: `data/depth5/`, `data/mrq2/` (unused datasets)
- Delete: `main.txt`, `data_analysis_report.txt` (old notes, superseded by `.md` versions)
- Rename: `pointclouds/` → `output/depth_pro/`, `pointclouds2/` → `output/depth_anything_3/`
- `compare_edit_depth` scripts: add `--mask-path` arg so they aren't hard-coded to `.npy` files (Option B)

---

## What we already ruled out

| Method | Why not |
|---|---|
| Metric3D v2 | Tested — much worse than Depth Pro on our data |
| UniDepth | Similar role to Metric3D v2, not tried but lower priority now |
| 3D Gaussian Splatting | Wrong tool — needs multi-view, solves novel-view synthesis not metric depth |
| MASt3R / DUSt3R | Needs image pair at inference; we only have the edited image |
