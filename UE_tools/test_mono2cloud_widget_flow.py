"""Smoke test for the widget-facing Mono2Cloud Unreal API.

Run inside the Unreal Editor Python console after adding this folder to `sys.path`.

Default behavior is safe:
- prints the camera dropdown model
- validates the configured PNG path
- does not start a capture unless `RUN_PIPELINE = True`
"""

from __future__ import annotations

import json

import mono2cloud_widget_api as widget_api


RUN_PIPELINE = False
CAMERA_LABEL = None
EDITED_IMAGE_PATH = r"e:\1ARZAANWORK\UNI\year2\UE_depth\data\new2\HighresScreenshot00001.png"


def _pick_first_camera_label() -> str:
    model = widget_api.get_widget_model()
    cameras = model["cameras"]
    if not cameras:
        raise RuntimeError("No CameraActor or CineCameraActor was found in the current level.")
    return str(cameras[0]["label"])


def main() -> None:
    model = widget_api.get_widget_model()
    print("=== Widget Model ===")
    print(json.dumps(model, indent=2))

    camera_label = CAMERA_LABEL or _pick_first_camera_label()
    validated = widget_api.validate_widget_inputs(
        camera_label=camera_label,
        edited_image_path=EDITED_IMAGE_PATH,
    )
    print("=== Validation ===")
    print(json.dumps(validated, indent=2))

    if not RUN_PIPELINE:
        print("RUN_PIPELINE is False, so this was a dry run only.")
        return

    run_handle = widget_api.run_from_widget(
        camera_label=camera_label,
        edited_image_path=EDITED_IMAGE_PATH,
    )
    print("=== Started Run ===")
    print(json.dumps(widget_api.summarize_run(run_handle), indent=2))
    print("Watch the Unreal Output Log for async progress updates.")


if __name__ == "__main__":
    main()
