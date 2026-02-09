"""Verify GT_TO_CM using depth_gt2 dataset (camera 90cm from flat plane)."""

import numpy as np
import OpenEXR
import Imath

FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
KNOWN_DISTANCE_CM = 90.0
DATA = "data/depth_gt2"


def load_exr(path):
    exr = OpenEXR.InputFile(path)
    dw = exr.header()['dataWindow']
    w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    arr = np.frombuffer(exr.channel('R', FLOAT), dtype=np.float32).reshape(h, w)
    return arr


sd = load_exr(f"{DATA}/HighresScreenshot00000_SceneDepth.exr")
wu = load_exr(f"{DATA}/HighresScreenshot00000_SceneDepthWorldUnits.exr")

sd_val = float(sd[sd > 0].min())  # constant for flat plane
wu_min = float(wu[wu > 0].min())  # min = optical axis = planar distance (meters)

# Method 1: WorldUnits (meters) -> cm, then divide by SceneDepth raw
gt_m1 = (wu_min * 100) / sd_val

# Method 2: known distance / SceneDepth raw
gt_m2 = KNOWN_DISTANCE_CM / sd_val

print(f"SceneDepth raw (constant) : {sd_val:.15f}")
print(f"WorldUnits min (meters)   : {wu_min:.15f}")
print(f"")
print(f"Method 1 (WorldUnits/SD)  : {gt_m1:.6f}")
print(f"Method 2 (known 90cm/SD)  : {gt_m2:.6f}")
print(f"")
print(f"GT_TO_CM = {(gt_m1 + gt_m2) / 2:.1f}")
