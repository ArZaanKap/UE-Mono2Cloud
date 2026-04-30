# Unreal MVP hookup

This folder contains the current editor-only MVP for the UE flow:

1. select one camera in Unreal
2. provide an already-made edited image
3. capture original RGB + original `SceneDepth`
4. run the repo pipeline
5. import the resulting `.las` into the Content Browser

The image-generation/API step is intentionally skipped for now.

## Files

- `UE_tools/mono2cloud_editor_mvp.py`
  - Unreal Python helper for capture, session prep, external pipeline run, and LAS import
  - now also exposes camera-listing helpers plus `run_camera_label_pipeline(...)`
- `UE_tools/mono2cloud_widget_api.py`
  - thin widget-facing layer for camera dropdown data, validation, and launch requests
- `UE_tools/test_mono2cloud_widget_flow.py`
  - Unreal-side smoke test for the future widget flow without building the widget yet

## Current assumptions

- You run this inside the Unreal Editor, not in a packaged build.
- The Unreal project has these plugins enabled:
  - Python Editor Script Plugin
  - Editor Scripting Utilities
  - LiDAR Point Cloud
- The external Python environment already runs the repo pipelines successfully.

## One-time setup

Open [UE_tools/mono2cloud_editor_mvp.py](/e:/1ARZAANWORK/UNI/year2/UE_depth/UE_tools/mono2cloud_editor_mvp.py) and fill in `USER_CONFIG`:

- `edited_image_path`
  - absolute path to the edited PNG/JPG you want to use for this run
- `repo_root`
  - `e:\\1ARZAANWORK\\UNI\\year2\\UE_depth`
- `python_exe`
  - absolute path to the Python executable that has the repo dependencies installed
- `pipeline`
  - `depth_pro` or `da3_giant`

You can also change:

- `capture_width`
- `capture_height`
- `asset_destination`
- `spawn_actor`

## How to run the current MVP in Unreal

1. Open your level.
2. Select exactly one `CameraActor` or `CineCameraActor`.
3. Open the Output Log and switch to Python.
4. Run:

```python
import sys
sys.path.append(r"e:\1ARZAANWORK\UNI\year2\UE_depth\UE_tools")
import mono2cloud_editor_mvp as mono2cloud
run_handle = mono2cloud.run_user_config()
print(run_handle)
```

`run_user_config()` now returns immediately with a `PipelineRunHandle` while the screenshot + pipeline work continues asynchronously on editor ticks. Watch the Output Log for progress.

If you want to inspect the latest state later in the same Python session:

```python
print(run_handle.status)
print(run_handle.error)
print(run_handle.result)
```

## Transition to a proper in-editor tool

The clean next step is:

1. keep `mono2cloud_editor_mvp.py` as the backend
2. call `mono2cloud_widget_api.py` from an `Editor Utility Widget`
3. let the widget own the UI only:
   - camera dropdown
   - edited-image file picker / drag-drop target
   - run button
   - optional status text

### What the widget should call

On widget construct / refresh:

```python
import mono2cloud_widget_api as mono2cloud_widget
model = mono2cloud_widget.get_widget_model()
print(model["cameras"])
```

When the user presses Run:

```python
run_handle = mono2cloud_widget.run_from_widget(
    camera_label="CameraActor",
    edited_image_path=r"E:\path\to\edited.png",
)
print(mono2cloud_widget.summarize_run(run_handle))
```

This means the widget no longer depends on:

- manual Output Log pasting every run
- manual camera selection as the only way to target a camera
- editing `USER_CONFIG["edited_image_path"]` each time

### Recommended Editor Utility Widget layout

Create an `Editor Utility Widget` in your Unreal project with these controls:

- `ComboBoxString`
  - populated from `get_widget_model()["cameras"]`
- `EditableTextBox`
  - stores the edited image path
- `Button`
  - `Browse...`
- `Border` or drop target area
  - accepts dragged PNG/JPG/EXR paths
- `Button`
  - `Run Mono2Cloud`
- `MultiLineEditableTextBox` or `TextBlock`
  - status / latest session path

Suggested widget behavior:

- `OnInitialized`
  - call Python to fetch camera options
- `Browse` button
  - open a file dialog or paste a path into the text box
- drag-drop
  - normalize the dropped file path, then update the text box
- `Run` button
  - call `run_from_widget(...)`
  - show `session_dir`
  - poll `get_active_runs()` if you want live status

### Why this is the right bridge

You do not need to jump straight to a full C++ plugin yet.

An `Editor Utility Widget + Python backend` gets you almost everything you want now:

- a real menu
- camera selection by dropdown
- upload / drag-drop for the edited PNG
- one-click execution
- reuse of your existing tested pipeline path

If that works well, you can later move the same API behind a true Unreal plugin tab or C++ tool without changing the external pipeline contract.

## Smoke-test the widget flow

Before building the widget, you can test the same API from Unreal Python:

```python
import sys
sys.path.append(r"e:\1ARZAANWORK\UNI\year2\UE_depth\UE_tools")
import test_mono2cloud_widget_flow as mono2cloud_test
mono2cloud_test.main()
```

By default this is a dry run:

- lists available cameras
- validates the edited image path
- does not start a capture

If you want it to actually launch the pipeline, set:

```python
mono2cloud_test.RUN_PIPELINE = True
mono2cloud_test.main()
```

## What it writes

The script creates a session folder in your Unreal project here:

```text
<YourProject>/Saved/Mono2Cloud/<timestamp>_<camera_name>/
```

Inside that folder you will get:

```text
original.png or original.exr
original_SceneDepth.exr
scene_edit.png
params.json
dpro/...
or
da3g/...
```

The pipeline output folder contains the same artifacts as the repo scripts:

- `.las`
- preview PNG
- summary JSON
- debug PNG folder

## What gets imported into Unreal

The resulting `.las` is imported into:

```text
/Game/Mono2Cloud
```

unless you change `asset_destination` in `USER_CONFIG`.

If `spawn_actor=True`, the script also spawns the imported point cloud asset into the current level.

## Important behavior

- The pipeline uses:
  - original RGB
  - original `SceneDepth.exr`
  - edited RGB
- It does not use edited GT depth.
- The DINOv2 mask params still come from `change_detection_results/params.py`.
- The DA3 script still uses native `process_res=0`, matching the compare-edit setup.

## Likely next step

Build the first `Editor Utility Widget` around `mono2cloud_widget_api.py`, then use the smoke test script to confirm the widget sends the same `camera_label + edited_image_path` inputs successfully.
