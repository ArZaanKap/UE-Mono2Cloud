from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_SCRIPTS = {
    "depth_pro": PROJECT_ROOT / "MAIN" / "pipeline_dinov2_depth_pro.py",
    "da3_giant": PROJECT_ROOT / "MAIN" / "pipeline_dinov2_da3_giant.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wrapper entrypoint for Unreal/editor automation"
    )
    parser.add_argument("--pipeline", required=True, choices=sorted(PIPELINE_SCRIPTS.keys()))
    parser.add_argument("--dataset", default=None, help="Legacy dataset name under data/")
    parser.add_argument("--input-dir", default=None, help="Consumer input folder")
    parser.add_argument("--output-dir", default=None, help="Destination output folder")
    parser.add_argument("--fov-deg", type=float, default=None)
    parser.add_argument("--mask-model", default="dinov2", choices=["dinov2", "dinov3", "dinox"])
    parser.add_argument("--dinox-text-prompt", default=None)
    parser.add_argument("--dinox-token-env", default="DINOX_API_TOKEN")
    parser.add_argument("--mask-sky", action="store_true", default=False)
    parser.add_argument("--no-mask-sky", action="store_true", default=False)
    parser.add_argument("--da3-process-res", type=int, default=None)
    args = parser.parse_args()

    script_path = PIPELINE_SCRIPTS[args.pipeline]
    cmd = [sys.executable, str(script_path)]

    if args.input_dir:
        cmd += ["--input-dir", args.input_dir]
    elif args.dataset:
        cmd += ["--dataset", args.dataset]

    if args.output_dir:
        cmd += ["--output-dir", args.output_dir]
    if args.fov_deg is not None:
        cmd += ["--fov-deg", str(args.fov_deg)]
    if args.mask_model:
        cmd += ["--mask-model", args.mask_model]
    if args.dinox_text_prompt:
        cmd += ["--dinox-text-prompt", args.dinox_text_prompt]
    if args.dinox_token_env:
        cmd += ["--dinox-token-env", args.dinox_token_env]
    if args.mask_sky and not args.no_mask_sky:
        cmd += ["--mask-sky"]
    if args.no_mask_sky:
        cmd += ["--no-mask-sky"]
    if args.pipeline == "da3_giant" and args.da3_process_res is not None:
        cmd += ["--da3-process-res", str(args.da3_process_res)]

    result = subprocess.run(cmd)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
