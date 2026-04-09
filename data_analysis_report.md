# Data Analysis Notes

Reference for depth file formats, capture methods, and model selection decisions.

---

## Depth File Formats

| File | Depth type | Conversion to metres | Use? |
|---|---|---|---|
| `_SceneDepth.exr` | Z-buffer (perpendicular) | `raw * 10000 / 100` | **Yes — use this** |
| `_SceneDepthWorldUnits.exr` | Euclidean (true 3D distance) | `raw * 100 / 100` | **No — causes distortion** |
| MRQ `_WorldDepth.exr` (R channel) | Z-buffer in cm | `raw / 100` | Yes, if using MRQ |

**Why Z-buffer and not Euclidean?**
Backprojection uses `X = (px - cx) * Z / f`, where Z must be the perpendicular depth.
Euclidean distance increases at the edges (the camera is further from off-axis pixels), which causes barrel distortion in the point cloud.

Verified with flat wall test (`data/depth_gt2`):
- `SceneDepth`: 1.20m constant across all pixels — correct
- `WorldUnits`: 1.23m centre, 1.96m at edges — wrong for backprojection

---

## MRQ vs High-Res Screenshot

| | High-Res Screenshot | Movie Render Queue (MRQ) |
|---|---|---|
| Resolution | 1526×858 | 1920×1080 |
| RGB quality | Good | Better AA, sharper edges |
| Depth file | `_SceneDepth.exr` (Z-buffer) | `_WorldDepth.exr` R channel (Z-buffer in cm) |
| `GT_TO_CENTIMETERS` | 10000 | 1 |
| MRQ `WorldDepth` G/B channels | — | World Y and Z coordinates — **ignore these** |

MRQ R-channel vs High-Res SceneDepth correlation = 0.9998 (verified on same scene).
Both depth models resize images internally, so higher resolution mainly benefits edge sharpness.

Current datasets use High-Res Screenshot (Option A — simpler, RGB and depth guaranteed to align).

---

## Scaling Method Comparison (depth4, Depth Anything V2)

| Method | RMSE | MAE | RelErr |
|---|---|---|---|
| None (raw) | 1.64m | 1.52m | 79.0% |
| Median | 0.14m | 0.09m | 3.8% |
| Least-squares | **0.08m** | **0.06m** | **2.9%** |

**Use least-squares.** Median has no shift term so it can't correct for affine offset.

---

## Depth Pro vs Depth Anything V2 (depth4, least-squares)

| Model | RMSE | MAE | RelErr | Delta1 | Scale needed |
|---|---|---|---|---|---|
| Depth Anything V2 | 0.084m | 0.056m | 2.9% | 99.9% | 0.60× |
| **Depth Pro** | **0.055m** | **0.043m** | **3.0%** | **100%** | 0.77× |

Depth Pro wins: 35% lower RMSE, 100% delta1, and needs less correction (closer to true scale).
Note: these numbers are from early evaluation on all pixels — later v2 evaluation on unchanged pixels only gives slightly different values (see `compare_edit_depth/`).
---

## Depth Models Tested In This Repo

For clarity, the repo has tested these depth models at some point:

| Model | Scope |
|---|---|
| Depth Pro | Current notebook and evaluation workflows |
| Depth Anything V2 Metric | Early analysis and evaluation scripts |
| Depth Anything 3 Giant 1.1 | Current DA3 notebook/evaluation workflow |
| Depth Anything 3 Nested Giant 1.1 | Current DA3 notebook/evaluation workflow |
| Metric3D v2 | Older `compare_edit_depth.py` evaluation only |

Metric3D v2 has saved result artefacts in `compare_edit_depth/depth3_results/` and `compare_edit_depth/depth4_results/`, but it is not part of the current recommended `compare_edit_depth2.py` pipeline.
