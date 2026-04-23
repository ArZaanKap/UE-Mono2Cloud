# TODO — Next Steps

**Current benchmark datasets:** `new0`–`new4` — UE-rendered pairs with GT depth for both original and edit. Results in `compare_edit_depth/v2/`.

**Best on changed regions (v2 benchmark):** **Depth Pro + least-squares** — consistently lowest MAE across most datasets (6.9–25.0 cm depending on scene).

**Best on unchanged regions:** DA3 Giant / DA3 Nested — lower MAE than Depth Pro on calibrated pixels.

**DepthLab finding:** MAE ~1–3 cm on unchanged (best of all models), but MAE ~75–128 cm on changed — it propagates wrong depth to pixels without guidance rather than generating new geometry for new objects. Not suitable as-is for our use case.

---

## Priority 1 — Depth Anything with Any Prior ← main next model

- **Paper:** ICLR 2026 (OpenReview)
- **What it does:** Conditional Depth Anything variant that accepts any partial depth prior at test time, scale-invariant log loss. Designed for exactly "I have some GT depth, fill in the rest".
- Same drop-in role as Marigold-DC. The key difference from DepthLab: designed to generalise to pixels without guidance, not just propagate known values.
- **Status:** Not yet tried.

---

## Priority 2 — Prompting Depth Anything for metric depth

- **Paper:** CVPR 2025
- **What it does:** Integrates sparse depth (LiDAR in their case, UE GT in ours) at multiple scales into a DPT-based model to get metric output everywhere.
- More involved integration than P1 methods but potentially higher accuracy.

---

## Already tried

### Marigold-DC ← tried
- **Paper:** arxiv:2412.13389 (ICCV 2025)
- **Code:** https://github.com/prs-eth/Marigold-DC
- **Status:** Integrated into `compare_edit_depth/compare_edit_depth2.py` as `--model marigold_dc` and `MAIN_TEST/img_to_pointcloud_marigold.ipynb`.
- **Results on new0 (changed):** MAE 10.6 cm, RMSE 18.1 cm — competitive with Depth Pro but not better.
- **Bring-up notes:** Needed RGB-only input, matched RGB/depth guide sizes, 768px long-edge cap (RTX 3070 Ti 8 GB).

### DepthLab ← tried
- **Paper:** arxiv:2412.18153
- **Site:** https://johanan528.github.io/depthlab_web/
- **Status:** Integrated into `compare_edit_depth/compare_edit_depth2.py` as `--model depthlab`. Results in `v2/{dataset}_results2/depthlab/`.
- **Results (new4, changed):** MAE 74.8 cm, d1 = 72.4% — fails on new objects.
- **Results (new4, unchanged):** MAE 2.6 cm — best of all models.
- **Why it fails on changed:** DepthLab uses GT depth for unchanged pixels as dense guidance; changed pixels receive no guidance and the model propagates surrounding depth rather than predicting correct depth for new geometry.

---

## What we already ruled out

| Method | Why not |
|---|---|
| Metric3D v2 | Tested — much worse than Depth Pro on our data |
| UniDepth | Similar role to Metric3D v2, not tried but lower priority now |
| 3D Gaussian Splatting | Wrong tool — needs multi-view, solves novel-view synthesis not metric depth |
| MASt3R / DUSt3R | Needs image pair at inference; we only have the edited image |
