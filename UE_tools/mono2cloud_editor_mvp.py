"""Editor-only Unreal helper for the current Mono2Cloud MVP flow.

This script is meant to run inside the Unreal Editor's Python environment.
Current MVP flow:
1. Select exactly one camera actor in the level.
2. Provide an existing edited PNG/JPG on disk.
3. Capture original RGB + original SceneDepth from the selected camera.
4. Run the external Python pipeline in this repo.
5. Import the resulting LAS into the Content Browser.

The image-generation step is intentionally skipped for now.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import unreal


DEFAULT_CAPTURE_WIDTH = 1526
DEFAULT_CAPTURE_HEIGHT = 858
DEFAULT_CAPTURE_WARMUP_TICKS = 24
DEFAULT_CAPTURE_SCREEN_PERCENTAGE = None
DEFAULT_CAPTURE_AA_QUALITY = None
DEFAULT_ASSET_DESTINATION = "/Game/Mono2Cloud"
CAPTURE_TIMEOUT_SECONDS = 300.0
PIPELINE_OUTPUT_FOLDERS = {
    "depth_pro": "dpro",
    "da3_giant": "da3g",
}
PIPELINE_LEGACY_OUTPUT_FOLDERS = {
    "depth_pro": ["depth_pro", "pointclouds_dinov2_depth_pro"],
    "da3_giant": ["da3_giant", "pointclouds_dinov2_da3_giant"],
}
PIPELINE_NAME_TAGS = {
    "depth_pro": "dpro",
    "da3_giant": "da3g",
}
VALID_EDIT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".exr"}
SCREENSHOT_STEM_RE = re.compile(r"^HighresScreenshot(\d+)$", re.IGNORECASE)
ACTIVE_PIPELINE_RUNS = set()
SCENE_COLOR_FORMAT_128BPP = 5


USER_CONFIG = {
    # Replace edited_image_path with your latest edited PNG/JPG when you have one.
    "edited_image_path": r"e:\1ARZAANWORK\UNI\year2\UE_depth\data\new2\HighresScreenshot00001.png", #edit
    "repo_root": r"e:\1ARZAANWORK\UNI\year2\UE_depth",
    "python_exe": r"C:\Users\parve\AppData\Local\Programs\Python\Python310\python.exe",
    "pipeline": "depth_pro",
    "mask_model": "dinov2",
    "capture_width": DEFAULT_CAPTURE_WIDTH,
    "capture_height": DEFAULT_CAPTURE_HEIGHT,
    "capture_warmup_ticks": DEFAULT_CAPTURE_WARMUP_TICKS,
    "capture_screen_percentage": DEFAULT_CAPTURE_SCREEN_PERCENTAGE,
    "capture_aa_quality": DEFAULT_CAPTURE_AA_QUALITY,
    "force_realtime_viewport": True,
    "asset_destination": DEFAULT_ASSET_DESTINATION,
    "replace_existing_asset": True,
    "spawn_actor": True,
}


class PipelineRunHandle:
    """Tracks an editor run that continues asynchronously across Slate ticks."""

    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self.status = "pending"
        self.error: str | None = None
        self.result: SessionResult | None = None
        self.output_dir: Path | None = None
        self.original_rgb_path: Path | None = None
        self.original_depth_path: Path | None = None
        self.edited_image_path: Path | None = None
        self.summary_path: Path | None = None
        self.las_path: Path | None = None
        self.imported_asset_paths: list[str] = []
        self.camera_actor = None
        self.repo_root_path: Path | None = None
        self.python_exe_path: Path | None = None
        self.pipeline = "depth_pro"
        self.mask_model = "dinov2"
        self.capture_width = DEFAULT_CAPTURE_WIDTH
        self.capture_height = DEFAULT_CAPTURE_HEIGHT
        self.capture_warmup_ticks = DEFAULT_CAPTURE_WARMUP_TICKS
        self.capture_screen_percentage: float | None = DEFAULT_CAPTURE_SCREEN_PERCENTAGE
        self.capture_aa_quality: int | None = DEFAULT_CAPTURE_AA_QUALITY
        self.force_realtime_viewport = True
        self.asset_destination = DEFAULT_ASSET_DESTINATION
        self.replace_existing_asset = True
        self.spawn_actor = True
        self._viewport_key = None
        self._tick_handle = None
        self._tick_callback = None
        self._screenshot_task = None
        self._capture_deadline = 0.0
        self._screenshots_root: Path | None = None
        self._before_screenshots: set[Path] = set()
        self._previous_scene_color_format: int | None = None
        self._previous_screen_percentage: float | None = None
        self._previous_aa_quality: int | None = None
        self._warmup_ticks_remaining = 0

    def __repr__(self) -> str:
        details = [f"status={self.status!r}", f"session_dir={str(self.session_dir)!r}"]
        if self.error:
            details.append(f"error={self.error!r}")
        if self.result is not None:
            details.append(f"las_path={str(self.result.las_path)!r}")
        return f"PipelineRunHandle({', '.join(details)})"

    def is_done(self) -> bool:
        return self.status in {"completed", "failed"}


@dataclass
class SessionResult:
    session_dir: Path
    output_dir: Path
    original_rgb_path: Path
    original_depth_path: Path
    edited_image_path: Path
    las_path: Path
    summary_path: Path
    imported_asset_paths: list[str]


def _log(message: str) -> None:
    unreal.log(f"[Mono2Cloud] {message}")


def _warn(message: str) -> None:
    unreal.log_warning(f"[Mono2Cloud] {message}")


def _sanitize_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return cleaned.strip("._-") or "session"


def _get_pipeline_tag(pipeline: str) -> str:
    return PIPELINE_NAME_TAGS.get(pipeline, pipeline)


def _is_camera_actor(actor) -> bool:
    return isinstance(actor, unreal.CameraActor) or isinstance(actor, unreal.CineCameraActor)


def _get_all_level_actors() -> list[object]:
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    getter = getattr(actor_subsystem, "get_all_level_actors", None)
    if getter is not None:
        return list(getter())
    return list(unreal.EditorLevelLibrary.get_all_level_actors())


def list_camera_actors() -> list[object]:
    cameras = [actor for actor in _get_all_level_actors() if _is_camera_actor(actor)]
    return sorted(cameras, key=lambda actor: (actor.get_actor_label().lower(), actor.get_name().lower()))


def list_camera_options() -> list[dict[str, str]]:
    options = []
    for actor in list_camera_actors():
        actor_path = ""
        try:
            actor_path = actor.get_path_name()
        except Exception:
            actor_path = actor.get_name()
        options.append(
            {
                "label": actor.get_actor_label(),
                "name": actor.get_name(),
                "path": actor_path,
                "class_name": actor.get_class().get_name(),
            }
        )
    return options


def _get_camera_actor_by_label(camera_label: str):
    if not camera_label:
        raise RuntimeError("camera_label is required.")

    exact_matches = [actor for actor in list_camera_actors() if actor.get_actor_label() == camera_label]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise RuntimeError(
            f"Multiple cameras share the label '{camera_label}'. Rename one of them or select the camera manually."
        )

    casefold_matches = [
        actor for actor in list_camera_actors()
        if actor.get_actor_label().casefold() == camera_label.casefold()
    ]
    if len(casefold_matches) == 1:
        return casefold_matches[0]
    if len(casefold_matches) > 1:
        raise RuntimeError(
            f"Multiple cameras match '{camera_label}' ignoring case. Rename one of them or select the camera manually."
        )

    available = ", ".join(option["label"] for option in list_camera_options()) or "<none>"
    raise RuntimeError(f"Camera '{camera_label}' was not found. Available cameras: {available}")


def _get_selected_camera_actor():
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    selected = list(actor_subsystem.get_selected_level_actors())
    cameras = [
        actor for actor in selected
        if _is_camera_actor(actor)
    ]
    if len(cameras) != 1:
        raise RuntimeError("Select exactly one CameraActor or CineCameraActor before running Mono2Cloud.")
    return cameras[0]


def _get_camera_component(camera_actor):
    component = camera_actor.get_editor_property("camera_component")
    if component is None:
        raise RuntimeError(f"Selected camera '{camera_actor.get_actor_label()}' has no camera component.")
    return component


def _resolve_repo_root(repo_root: str) -> Path:
    if not repo_root:
        raise RuntimeError("USER_CONFIG['repo_root'] is required.")
    path = Path(repo_root).expanduser().resolve()
    if not (path / "MAIN" / "run_pipeline.py").exists():
        raise RuntimeError(f"Repo root does not contain MAIN/run_pipeline.py: {path}")
    return path


def _resolve_python_exe(python_exe: str) -> Path:
    if not python_exe:
        raise RuntimeError("USER_CONFIG['python_exe'] is required.")
    path = Path(python_exe).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"Python executable not found: {path}")
    return path


def _resolve_edited_image(edited_image_path: str) -> Path:
    if not edited_image_path:
        raise RuntimeError("USER_CONFIG['edited_image_path'] is required.")
    path = Path(edited_image_path).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"Edited image not found: {path}")
    if path.suffix.lower() not in VALID_EDIT_EXTENSIONS:
        raise RuntimeError(f"Unsupported edited image format: {path.suffix}")
    return path


def _make_session_dir(camera_actor) -> Path:
    saved_root = Path(unreal.Paths.project_saved_dir()).resolve()
    mono_root = saved_root / "Mono2Cloud"
    mono_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    camera_name = _sanitize_name(camera_actor.get_actor_label())
    session_dir = mono_root / f"{timestamp}_{camera_name}"
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def _write_params_json(
    session_dir: Path,
    camera_actor,
    *,
    pipeline: str,
    mask_model: str,
    capture_width: int,
    capture_height: int,
    capture_warmup_ticks: int,
    capture_screen_percentage: float | None,
    capture_aa_quality: int | None,
    force_realtime_viewport: bool,
) -> None:
    camera_component = _get_camera_component(camera_actor)
    payload = {
        "fov_deg": float(camera_component.get_editor_property("field_of_view")),
        "camera_label": camera_actor.get_actor_label(),
        "camera_name": camera_actor.get_name(),
        "captured_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline": pipeline,
        "mask_model": mask_model,
        "capture_width": int(capture_width),
        "capture_height": int(capture_height),
        "capture_warmup_ticks": int(capture_warmup_ticks),
        "capture_screen_percentage": None if capture_screen_percentage is None else float(capture_screen_percentage),
        "capture_aa_quality": None if capture_aa_quality is None else int(capture_aa_quality),
        "force_realtime_viewport": bool(force_realtime_viewport),
    }
    (session_dir / "params.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _copy_edited_image(edited_image_path: Path, session_dir: Path) -> Path:
    target = session_dir / f"scene_edit{edited_image_path.suffix.lower()}"
    shutil.copy2(edited_image_path, target)
    return target


def _snapshot_files(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {path.resolve() for path in root.rglob("*") if path.is_file()}


def _wait_for_file(path: Path, timeout_seconds: float = 30.0) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if path.exists():
            return True
        time.sleep(0.25)
    return path.exists()


def _execute_console_command(command: str) -> None:
    world = unreal.EditorLevelLibrary.get_editor_world()
    unreal.SystemLibrary.execute_console_command(world, command)


def _get_console_variable_int(name: str) -> int | None:
    world = unreal.EditorLevelLibrary.get_editor_world()
    getter = getattr(unreal.SystemLibrary, "get_console_variable_int_value", None)
    if getter is None:
        return None
    try:
        return int(getter(world, name))
    except Exception:
        return None


def _get_console_variable_float(name: str) -> float | None:
    world = unreal.EditorLevelLibrary.get_editor_world()
    getter = getattr(unreal.SystemLibrary, "get_console_variable_float_value", None)
    if getter is None:
        return None
    try:
        return float(getter(world, name))
    except Exception:
        return None


def _delete_files(paths: set[Path]) -> None:
    for path in sorted(paths):
        try:
            if path.exists() and path.is_file():
                path.unlink()
        except Exception as exc:
            _warn(f"Could not delete temporary screenshot file '{path}': {exc}")


def _find_named_buffer_file(paths: set[Path], token: str) -> Path | None:
    matches = sorted(
        path
        for path in paths
        if path.suffix.lower() == ".exr" and token.lower() in path.name.lower()
    )
    return matches[-1] if matches else None


def _trigger_high_res_capture(
    camera_actor,
    width: int,
    height: int,
) -> object | None:
    try:
        if hasattr(unreal, "AutomationLibrary") and hasattr(unreal.AutomationLibrary, "finish_loading_before_screenshot"):
            unreal.AutomationLibrary.finish_loading_before_screenshot()
    except Exception as exc:
        _warn(f"Could not finish loading before screenshot: {exc}")

    # Do not pass `filename=...` here. In editor builds, Unreal can keep
    # reusing the last HighResShot filename/path for later manual captures,
    # which makes the High Resolution Screenshot Tool appear to "save into"
    # the previous Mono2Cloud session directory. We capture into the normal
    # Saved/Screenshots location, then copy the files we need into the
    # session folder after they are written.
    command = (
        f"HighResShot {width}x{height} 0 0 0 0 0 1 0 0"
    )
    _log("Started HighResShot capture with buffer visualization targets and HDR visualization outputs.")
    _execute_console_command(command)
    return None


def _configure_viewport_for_camera(camera_actor):
    level_editor = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    viewport_key = level_editor.get_active_viewport_config_key()

    # Some UE builds expose a realtime override path here that can raise an ensure
    # when called from Python, so keep this optional.
    try:
        if hasattr(level_editor, "editor_set_game_view"):
            level_editor.editor_set_game_view(True, viewport_key)
    except Exception as exc:
        _warn(f"Could not enable Game View: {exc}")

    try:
        if hasattr(level_editor, "set_allows_cinematic_control"):
            level_editor.set_allows_cinematic_control(True, viewport_key)
    except Exception as exc:
        _warn(f"Could not enable cinematic control: {exc}")

    # Match the editor viewport to the selected camera transform before piloting.
    try:
        unreal_editor = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
        if hasattr(unreal_editor, "set_level_viewport_camera_info"):
            unreal_editor.set_level_viewport_camera_info(
                camera_actor.get_actor_location(),
                camera_actor.get_actor_rotation(),
            )
    except Exception as exc:
        _warn(f"Could not set viewport camera info: {exc}")

    try:
        if hasattr(level_editor, "set_exact_camera_view"):
            level_editor.set_exact_camera_view(True, viewport_key)
    except Exception as exc:
        _warn(f"Could not enable exact camera view: {exc}")

    level_editor.pilot_level_actor(camera_actor, viewport_key)
    level_editor.editor_invalidate_viewports()
    return level_editor, viewport_key


def _set_viewport_realtime(level_editor, viewport_key, enabled: bool) -> None:
    try:
        if hasattr(level_editor, "editor_set_viewport_realtime"):
            level_editor.editor_set_viewport_realtime(enabled, viewport_key)
            return
    except Exception as exc:
        _warn(f"Could not set viewport realtime={enabled}: {exc}")


def _apply_capture_overrides(run_handle: PipelineRunHandle) -> None:
    run_handle._previous_scene_color_format = _get_console_variable_int("r.SceneColorFormat")
    run_handle._previous_screen_percentage = _get_console_variable_float("r.ScreenPercentage")
    run_handle._previous_aa_quality = _get_console_variable_int("sg.AntiAliasingQuality")

    if run_handle.capture_screen_percentage is not None:
        _execute_console_command(f"r.ScreenPercentage {float(run_handle.capture_screen_percentage)}")
    if run_handle.capture_aa_quality is not None:
        _execute_console_command(f"sg.AntiAliasingQuality {int(run_handle.capture_aa_quality)}")

    _log(
        "Applied capture overrides: "
        f"screen_percentage={run_handle.capture_screen_percentage}, "
        f"aa_quality={run_handle.capture_aa_quality}, "
        f"realtime_viewport={run_handle.force_realtime_viewport}"
    )


def _discover_capture_outputs(
    session_dir: Path,
    screenshots_root: Path,
    before_screenshots: set[Path],
) -> tuple[Path | None, Path | None, set[Path], set[Path]]:
    after_session = _snapshot_files(session_dir)
    after_screenshots = _snapshot_files(screenshots_root)
    new_screenshot_files = after_screenshots - before_screenshots

    beauty_candidates = [
        session_dir / "original.png",
        session_dir / "original.exr",
        session_dir / "original.jpg",
        session_dir / "original.jpeg",
    ]
    beauty_path = next((path for path in beauty_candidates if path.exists()), None)

    session_depth_candidates = sorted(
        path
        for path in after_session
        if path.suffix.lower() == ".exr"
        and "scenedepth" in path.name.lower()
        and "worldunits" not in path.name.lower()
    )
    depth_path = session_depth_candidates[-1] if session_depth_candidates else None

    latest_beauty, latest_depth = _find_latest_screenshot_pair(new_screenshot_files)
    if beauty_path is None:
        beauty_path = latest_beauty
    if depth_path is None:
        depth_path = latest_depth
    return beauty_path, depth_path, new_screenshot_files, (after_session | after_screenshots)


def _finalize_capture_outputs(
    session_dir: Path,
    beauty_path: Path,
    depth_path: Path,
    new_screenshot_files: set[Path],
) -> tuple[Path, Path]:
    session_files = _snapshot_files(session_dir)
    original_target = session_dir / f"original{beauty_path.suffix.lower()}"
    depth_target = session_dir / "original_SceneDepth.exr"
    worldnormal_source = _find_named_buffer_file(session_files | new_screenshot_files, "worldnormal")
    worldnormal_target = session_dir / "original_WorldNormal.exr"
    if beauty_path.resolve() != original_target.resolve():
        shutil.copy2(beauty_path, original_target)
    if depth_path.resolve() != depth_target.resolve():
        shutil.copy2(depth_path, depth_target)
    if worldnormal_source is not None and worldnormal_source.resolve() != worldnormal_target.resolve():
        shutil.copy2(worldnormal_source, worldnormal_target)

    if not _wait_for_file(original_target) or not _wait_for_file(depth_target):
        raise RuntimeError("Captured files were not fully written to disk.")
    if worldnormal_source is not None and not _wait_for_file(worldnormal_target):
        raise RuntimeError("Captured WorldNormal EXR was not fully written to disk.")
    if worldnormal_source is None:
        _warn("Capture did not produce a WorldNormal EXR; continuing with RGB + SceneDepth only.")

    keep_names = {
        original_target.name.lower(),
        depth_target.name.lower(),
        worldnormal_target.name.lower(),
        "params.json",
    }
    session_capture_extras = {
        path
        for path in _snapshot_files(session_dir)
        if path.is_file()
        and path.parent == session_dir
        and (
            path.name.lower().startswith("original_")
            or path.stem.lower().startswith("highresscreenshot")
        )
        and path.name.lower() not in keep_names
    }
    _delete_files(session_capture_extras)

    # Unreal writes a beauty PNG plus many buffer EXRs into Saved/Screenshots.
    # Once we've copied the two files we need into the session folder, clean up
    # only the files created by this run so the screenshot cache does not grow.
    _delete_files(new_screenshot_files)
    return original_target, depth_target


def _teardown_capture_state(run_handle: PipelineRunHandle) -> None:
    if run_handle._tick_handle is not None:
        try:
            unreal.unregister_slate_post_tick_callback(run_handle._tick_handle)
        except Exception:
            pass
        run_handle._tick_handle = None
    run_handle._tick_callback = None

    try:
        _execute_console_command("r.BufferVisualizationDumpFrames 0")
        _execute_console_command("r.BufferVisualizationDumpFramesAsHDR 0")
        previous_scene_color_format = run_handle._previous_scene_color_format
        if previous_scene_color_format is not None:
            _execute_console_command(f"r.SceneColorFormat {previous_scene_color_format}")
        previous_screen_percentage = run_handle._previous_screen_percentage
        if previous_screen_percentage is not None:
            _execute_console_command(f"r.ScreenPercentage {previous_screen_percentage}")
        previous_aa_quality = run_handle._previous_aa_quality
        if previous_aa_quality is not None:
            _execute_console_command(f"sg.AntiAliasingQuality {previous_aa_quality}")
    except Exception:
        pass
    run_handle._previous_scene_color_format = None
    run_handle._previous_screen_percentage = None
    run_handle._previous_aa_quality = None

    viewport_key = run_handle._viewport_key
    if viewport_key is not None:
        try:
            level_editor = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
            if level_editor is not None:
                if run_handle.force_realtime_viewport:
                    _set_viewport_realtime(level_editor, viewport_key, False)
                level_editor.eject_pilot_level_actor(viewport_key)
        except Exception as exc:
            _warn(f"Could not eject pilot camera: {exc}")
    run_handle._viewport_key = None
    run_handle._screenshot_task = None


def _complete_run(run_handle: PipelineRunHandle) -> None:
    run_handle.result = SessionResult(
        session_dir=run_handle.session_dir,
        output_dir=run_handle.output_dir,
        original_rgb_path=run_handle.original_rgb_path,
        original_depth_path=run_handle.original_depth_path,
        edited_image_path=run_handle.edited_image_path,
        las_path=run_handle.las_path,
        summary_path=run_handle.summary_path,
        imported_asset_paths=run_handle.imported_asset_paths,
    )
    run_handle.status = "completed"
    ACTIVE_PIPELINE_RUNS.discard(run_handle)


def _fail_run(run_handle: PipelineRunHandle, exc: Exception) -> None:
    _teardown_capture_state(run_handle)
    run_handle.status = "failed"
    run_handle.error = str(exc)
    ACTIVE_PIPELINE_RUNS.discard(run_handle)
    unreal.log_error(f"[Mono2Cloud] Run failed: {exc}")
    stack = traceback.format_exc().strip()
    if stack and stack != "NoneType: None":
        unreal.log_error(stack)


def _resume_pipeline_after_capture(run_handle: PipelineRunHandle) -> None:
    run_handle.status = "running_pipeline"
    las_path, summary_path = _run_pipeline(
        run_handle.repo_root_path,
        run_handle.python_exe_path,
        run_handle.session_dir,
        run_handle.pipeline,
        run_handle.mask_model,
    )

    pipeline_tag = _get_pipeline_tag(run_handle.pipeline)
    asset_name = _sanitize_name(
        f"{run_handle.camera_actor.get_actor_label()}_{pipeline_tag}_{run_handle.session_dir.name}"
    )
    imported_asset_paths = _import_las_asset(
        las_path,
        destination_path=run_handle.asset_destination,
        destination_name=asset_name,
        replace_existing=run_handle.replace_existing_asset,
    )
    if run_handle.spawn_actor:
        for asset_path in imported_asset_paths:
            _spawn_actor_for_asset(asset_path)

    run_handle.output_dir = las_path.parent
    run_handle.summary_path = summary_path
    run_handle.las_path = las_path
    run_handle.imported_asset_paths = imported_asset_paths
    _log(f"Session complete. Imported asset(s): {imported_asset_paths}")
    _complete_run(run_handle)


def _trigger_capture_now(run_handle: PipelineRunHandle) -> None:
    _log(f"Capturing original RGB + SceneDepth from '{run_handle.camera_actor.get_actor_label()}'...")
    # Mirror the HRSST checkbox behavior as closely as possible:
    # Include Buffer Visualization Targets, Write HDR format visualization targets,
    # and a best-effort 128-bit scene-color override during the capture.
    _execute_console_command("r.BufferVisualizationDumpFrames 1")
    _execute_console_command("r.BufferVisualizationDumpFramesAsHDR 1")
    _execute_console_command(f"r.SceneColorFormat {SCENE_COLOR_FORMAT_128BPP}")
    run_handle._screenshot_task = _trigger_high_res_capture(
        run_handle.camera_actor,
        int(run_handle.capture_width),
        int(run_handle.capture_height),
    )
    run_handle._capture_deadline = time.time() + CAPTURE_TIMEOUT_SECONDS
    run_handle.status = "capturing"


def _advance_warmup(run_handle: PipelineRunHandle) -> None:
    if run_handle.status != "warming_up":
        return

    try:
        level_editor = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
        if level_editor is not None:
            level_editor.editor_invalidate_viewports()
    except Exception:
        pass

    if run_handle._warmup_ticks_remaining > 0:
        run_handle._warmup_ticks_remaining -= 1
        return

    _log("Viewport warm-up complete. Starting capture.")
    _trigger_capture_now(run_handle)


def _poll_capture_until_ready(run_handle: PipelineRunHandle, _delta_seconds: float) -> None:
    if run_handle.status != "capturing":
        return

    try:
        beauty_path, depth_path, new_screenshot_files, discovered = _discover_capture_outputs(
            run_handle.session_dir,
            run_handle._screenshots_root,
            run_handle._before_screenshots,
        )
        if beauty_path is not None and depth_path is not None:
            run_handle.original_rgb_path, run_handle.original_depth_path = _finalize_capture_outputs(
                run_handle.session_dir,
                beauty_path,
                depth_path,
                new_screenshot_files,
            )
            _teardown_capture_state(run_handle)
            _log("Capture complete. Continuing with external pipeline.")
            _resume_pipeline_after_capture(run_handle)
            return

        task = run_handle._screenshot_task
        if task is not None and hasattr(task, "is_valid_task") and task.is_valid_task():
            if not task.is_task_done():
                if time.time() < run_handle._capture_deadline:
                    return

        if time.time() >= run_handle._capture_deadline:
            discovered_paths = sorted(str(path) for path in discovered)
            raise RuntimeError(
                "Unreal capture did not produce both an original RGB image and an original SceneDepth EXR. "
                f"Files seen after capture within {CAPTURE_TIMEOUT_SECONDS:.0f}s: {discovered_paths}"
            )
    except Exception as exc:
        _fail_run(run_handle, exc)


def _start_capture_async(run_handle: PipelineRunHandle) -> None:
    level_editor, viewport_key = _configure_viewport_for_camera(run_handle.camera_actor)
    run_handle._viewport_key = viewport_key
    if run_handle.force_realtime_viewport:
        _set_viewport_realtime(level_editor, viewport_key, True)
    project_saved = Path(unreal.Paths.project_saved_dir()).resolve()
    run_handle._screenshots_root = project_saved / "Screenshots"
    run_handle._before_screenshots = _snapshot_files(run_handle._screenshots_root)
    _apply_capture_overrides(run_handle)
    run_handle._warmup_ticks_remaining = max(0, int(run_handle.capture_warmup_ticks))
    if run_handle._warmup_ticks_remaining > 0:
        run_handle.status = "warming_up"
        _log(
            f"Warming up viewport for {run_handle._warmup_ticks_remaining} ticks "
            "so temporal AA, shadows, and lighting can settle before capture."
        )
    else:
        _trigger_capture_now(run_handle)

    def _tick(delta_seconds: float) -> None:
        if run_handle.status == "warming_up":
            _advance_warmup(run_handle)
        _poll_capture_until_ready(run_handle, delta_seconds)

    run_handle._tick_callback = _tick
    run_handle._tick_handle = unreal.register_slate_post_tick_callback(_tick)


def _pick_capture_files(paths: set[Path]) -> tuple[Path | None, Path | None]:
    beauty = None
    depth = None
    for path in sorted(paths):
        lower_name = path.name.lower()
        suffix = path.suffix.lower()
        if suffix not in VALID_EDIT_EXTENSIONS:
            continue
        if suffix == ".exr" and "scenedepth" in lower_name and "worldunits" not in lower_name:
            depth = path
            continue
        if "scenedepth" in lower_name or "worldnormal" in lower_name or "mask" in lower_name:
            continue
        if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".exr"}:
            beauty = path
    return beauty, depth


def _extract_screenshot_index(path: Path) -> int | None:
    match = SCREENSHOT_STEM_RE.match(path.stem)
    if not match:
        return None
    return int(match.group(1))


def _find_latest_screenshot_pair(paths: set[Path]) -> tuple[Path | None, Path | None]:
    pairs: dict[int, dict[str, Path]] = {}

    for path in paths:
        lower_name = path.name.lower()
        suffix = path.suffix.lower()

        if suffix == ".png":
            index = _extract_screenshot_index(path)
            if index is not None:
                pairs.setdefault(index, {})["beauty"] = path
            continue

        if suffix == ".exr" and "scenedepth" in lower_name and "worldunits" not in lower_name:
            stem = path.stem
            if stem.lower().endswith("_scenedepth"):
                stem = stem[: -len("_SceneDepth")]
            index_match = SCREENSHOT_STEM_RE.match(stem)
            if index_match is not None:
                index = int(index_match.group(1))
                pairs.setdefault(index, {})["depth"] = path

    valid_indices = [index for index, entry in pairs.items() if "beauty" in entry and "depth" in entry]
    if not valid_indices:
        return _pick_capture_files(paths)

    latest_index = max(valid_indices)
    return pairs[latest_index]["beauty"], pairs[latest_index]["depth"]


def _run_pipeline(
    repo_root: Path,
    python_exe: Path,
    session_dir: Path,
    pipeline: str,
    mask_model: str,
) -> tuple[Path, Path]:
    if pipeline not in PIPELINE_OUTPUT_FOLDERS:
        raise RuntimeError(f"Unsupported pipeline: {pipeline}")

    cmd = [
        str(python_exe),
        str(repo_root / "MAIN" / "run_pipeline.py"),
        "--pipeline",
        pipeline,
        "--input-dir",
        str(session_dir),
        "--mask-model",
        mask_model,
    ]
    _log("Running external pipeline...")
    popen_kwargs = {
        "cwd": str(repo_root),
        "capture_output": True,
        "text": True,
        "check": False,
    }
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    result = subprocess.run(
        cmd,
        **popen_kwargs,
    )
    if result.stdout.strip():
        unreal.log(result.stdout.strip())
    if result.stderr.strip():
        unreal.log_warning(result.stderr.strip())
    if result.returncode != 0:
        raise RuntimeError(f"Pipeline failed with exit code {result.returncode}.")

    output_dir, summary_path = _find_pipeline_summary(session_dir, pipeline)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    las_path = Path(summary["las_path"]).resolve()
    if not las_path.exists():
        raise RuntimeError(f"LAS output missing: {las_path}")
    return las_path, summary_path


def _find_pipeline_summary(session_dir: Path, pipeline: str) -> tuple[Path, Path]:
    candidate_dir_names = [PIPELINE_OUTPUT_FOLDERS[pipeline], *PIPELINE_LEGACY_OUTPUT_FOLDERS.get(pipeline, [])]
    checked_output_dirs: list[Path] = []

    for dir_name in candidate_dir_names:
        output_dir = session_dir / dir_name
        if output_dir in checked_output_dirs:
            continue
        checked_output_dirs.append(output_dir)
        if not output_dir.exists():
            continue

        result_summary = output_dir / "result_summary.json"
        if result_summary.exists():
            return output_dir, result_summary

        summaries = sorted(output_dir.glob("*_summary.json"))
        if summaries:
            return output_dir, summaries[-1]

    recursive_candidates = sorted(
        (
            path for path in session_dir.rglob("*_summary.json")
            if path.is_file()
        ),
        key=lambda path: (path.stat().st_mtime, str(path)),
    )
    if recursive_candidates:
        summary_path = recursive_candidates[-1]
        return summary_path.parent, summary_path

    checked = ", ".join(str(path) for path in checked_output_dirs)
    raise RuntimeError(f"No summary JSON found in session dir. Checked: {checked}")


def _import_las_asset(
    las_path: Path,
    destination_path: str,
    destination_name: str,
    replace_existing: bool,
) -> list[str]:
    if not hasattr(unreal, "LidarPointCloudFactory"):
        raise RuntimeError("LiDAR Point Cloud plugin does not appear to be enabled in this Unreal project.")

    if not unreal.EditorAssetLibrary.does_directory_exist(destination_path):
        unreal.EditorAssetLibrary.make_directory(destination_path)

    task = unreal.AssetImportTask()
    task.set_editor_property("filename", str(las_path))
    task.set_editor_property("destination_path", destination_path)
    task.set_editor_property("destination_name", destination_name)
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", replace_existing)
    task.set_editor_property("replace_existing_settings", replace_existing)
    task.set_editor_property("save", True)
    task.set_editor_property("async_", False)
    task.set_editor_property("factory", unreal.LidarPointCloudFactory())

    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    imported_paths = list(task.get_editor_property("imported_object_paths") or [])
    if not imported_paths:
        imported_objects = list(task.get_objects() or [])
        imported_paths = [obj.get_path_name() for obj in imported_objects if obj is not None]
    if not imported_paths:
        raise RuntimeError(f"LAS import completed but no asset paths were returned for {las_path}")
    _verify_imported_assets(imported_paths)
    return imported_paths


def _verify_imported_assets(imported_paths: list[str]) -> None:
    failed_paths: list[str] = []

    for asset_path in imported_paths:
        try:
            if not unreal.EditorAssetLibrary.does_asset_exist(asset_path):
                failed_paths.append(f"{asset_path} (asset does not exist after import)")
                continue

            asset = unreal.EditorAssetLibrary.load_asset(asset_path)
            if asset is None:
                failed_paths.append(f"{asset_path} (asset could not be loaded after import)")
                continue

            save_ok = unreal.EditorAssetLibrary.save_loaded_asset(asset, only_if_is_dirty=False)
            if not bool(save_ok):
                failed_paths.append(f"{asset_path} (save_loaded_asset returned false)")
                continue
        except Exception as exc:
            failed_paths.append(f"{asset_path} ({exc})")

    if failed_paths:
        raise RuntimeError(
            "Imported LAS asset failed verification/save: " + "; ".join(failed_paths)
        )


def _spawn_actor_for_asset(asset_path: str) -> None:
    asset = unreal.EditorAssetLibrary.load_asset(asset_path)
    if asset is None:
        _warn(f"Imported asset could not be loaded for spawning: {asset_path}")
        return
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actor = actor_subsystem.spawn_actor_from_object(asset, unreal.Vector(0.0, 0.0, 0.0))
    if actor is not None:
        actor.set_actor_label(Path(asset_path).name)


def run_camera_actor_pipeline(
    camera_actor,
    edited_image_path: str,
    repo_root: str,
    python_exe: str,
    pipeline: str = "depth_pro",
    mask_model: str = "dinov2",
    capture_width: int = DEFAULT_CAPTURE_WIDTH,
    capture_height: int = DEFAULT_CAPTURE_HEIGHT,
    capture_warmup_ticks: int = DEFAULT_CAPTURE_WARMUP_TICKS,
    capture_screen_percentage: float | None = DEFAULT_CAPTURE_SCREEN_PERCENTAGE,
    capture_aa_quality: int | None = DEFAULT_CAPTURE_AA_QUALITY,
    force_realtime_viewport: bool = True,
    asset_destination: str = DEFAULT_ASSET_DESTINATION,
    replace_existing_asset: bool = True,
    spawn_actor: bool = True,
) -> PipelineRunHandle:
    if camera_actor is None or not _is_camera_actor(camera_actor):
        raise RuntimeError("run_camera_actor_pipeline requires a CameraActor or CineCameraActor.")

    repo_root_path = _resolve_repo_root(repo_root)
    python_exe_path = _resolve_python_exe(python_exe)
    edited_image = _resolve_edited_image(edited_image_path)
    session_dir = _make_session_dir(camera_actor)

    run_handle = PipelineRunHandle(session_dir)
    run_handle.camera_actor = camera_actor
    run_handle.repo_root_path = repo_root_path
    run_handle.python_exe_path = python_exe_path
    run_handle.pipeline = pipeline
    run_handle.mask_model = mask_model
    run_handle.capture_width = int(capture_width)
    run_handle.capture_height = int(capture_height)
    run_handle.capture_warmup_ticks = max(0, int(capture_warmup_ticks))
    run_handle.capture_screen_percentage = None if capture_screen_percentage is None else float(capture_screen_percentage)
    run_handle.capture_aa_quality = None if capture_aa_quality is None else int(capture_aa_quality)
    run_handle.force_realtime_viewport = bool(force_realtime_viewport)
    run_handle.asset_destination = asset_destination
    run_handle.replace_existing_asset = bool(replace_existing_asset)
    run_handle.spawn_actor = bool(spawn_actor)

    _write_params_json(
        session_dir,
        camera_actor,
        pipeline=run_handle.pipeline,
        mask_model=run_handle.mask_model,
        capture_width=run_handle.capture_width,
        capture_height=run_handle.capture_height,
        capture_warmup_ticks=run_handle.capture_warmup_ticks,
        capture_screen_percentage=run_handle.capture_screen_percentage,
        capture_aa_quality=run_handle.capture_aa_quality,
        force_realtime_viewport=run_handle.force_realtime_viewport,
    )
    edited_session_path = _copy_edited_image(edited_image, session_dir)
    run_handle.session_dir = session_dir
    run_handle.edited_image_path = edited_session_path

    ACTIVE_PIPELINE_RUNS.add(run_handle)
    _start_capture_async(run_handle)
    _log(
        "Capture queued. The run continues asynchronously while the editor keeps ticking. "
        "Watch the Output Log or inspect the returned handle for completion."
    )
    return run_handle


def run_camera_label_pipeline(
    camera_label: str,
    edited_image_path: str,
    repo_root: str,
    python_exe: str,
    pipeline: str = "depth_pro",
    mask_model: str = "dinov2",
    capture_width: int = DEFAULT_CAPTURE_WIDTH,
    capture_height: int = DEFAULT_CAPTURE_HEIGHT,
    capture_warmup_ticks: int = DEFAULT_CAPTURE_WARMUP_TICKS,
    capture_screen_percentage: float | None = DEFAULT_CAPTURE_SCREEN_PERCENTAGE,
    capture_aa_quality: int | None = DEFAULT_CAPTURE_AA_QUALITY,
    force_realtime_viewport: bool = True,
    asset_destination: str = DEFAULT_ASSET_DESTINATION,
    replace_existing_asset: bool = True,
    spawn_actor: bool = True,
) -> PipelineRunHandle:
    return run_camera_actor_pipeline(
        _get_camera_actor_by_label(camera_label),
        edited_image_path=edited_image_path,
        repo_root=repo_root,
        python_exe=python_exe,
        pipeline=pipeline,
        mask_model=mask_model,
        capture_width=capture_width,
        capture_height=capture_height,
        capture_warmup_ticks=capture_warmup_ticks,
        capture_screen_percentage=capture_screen_percentage,
        capture_aa_quality=capture_aa_quality,
        force_realtime_viewport=force_realtime_viewport,
        asset_destination=asset_destination,
        replace_existing_asset=replace_existing_asset,
        spawn_actor=spawn_actor,
    )


def run_selected_camera_pipeline(
    edited_image_path: str,
    repo_root: str,
    python_exe: str,
    pipeline: str = "depth_pro",
    mask_model: str = "dinov2",
    capture_width: int = DEFAULT_CAPTURE_WIDTH,
    capture_height: int = DEFAULT_CAPTURE_HEIGHT,
    capture_warmup_ticks: int = DEFAULT_CAPTURE_WARMUP_TICKS,
    capture_screen_percentage: float | None = DEFAULT_CAPTURE_SCREEN_PERCENTAGE,
    capture_aa_quality: int | None = DEFAULT_CAPTURE_AA_QUALITY,
    force_realtime_viewport: bool = True,
    asset_destination: str = DEFAULT_ASSET_DESTINATION,
    replace_existing_asset: bool = True,
    spawn_actor: bool = True,
) -> PipelineRunHandle:
    return run_camera_actor_pipeline(
        _get_selected_camera_actor(),
        edited_image_path=edited_image_path,
        repo_root=repo_root,
        python_exe=python_exe,
        pipeline=pipeline,
        mask_model=mask_model,
        capture_width=capture_width,
        capture_height=capture_height,
        capture_warmup_ticks=capture_warmup_ticks,
        capture_screen_percentage=capture_screen_percentage,
        capture_aa_quality=capture_aa_quality,
        force_realtime_viewport=force_realtime_viewport,
        asset_destination=asset_destination,
        replace_existing_asset=replace_existing_asset,
        spawn_actor=spawn_actor,
    )


def get_active_runs() -> list[PipelineRunHandle]:
    return sorted(ACTIVE_PIPELINE_RUNS, key=lambda run: str(run.session_dir))


def run_user_config() -> PipelineRunHandle:
    return run_selected_camera_pipeline(
        edited_image_path=USER_CONFIG["edited_image_path"],
        repo_root=USER_CONFIG["repo_root"],
        python_exe=USER_CONFIG["python_exe"],
        pipeline=USER_CONFIG["pipeline"],
        mask_model=USER_CONFIG["mask_model"],
        capture_width=int(USER_CONFIG["capture_width"]),
        capture_height=int(USER_CONFIG["capture_height"]),
        capture_warmup_ticks=int(USER_CONFIG["capture_warmup_ticks"]),
        capture_screen_percentage=None if USER_CONFIG["capture_screen_percentage"] is None else float(USER_CONFIG["capture_screen_percentage"]),
        capture_aa_quality=None if USER_CONFIG["capture_aa_quality"] is None else int(USER_CONFIG["capture_aa_quality"]),
        force_realtime_viewport=bool(USER_CONFIG["force_realtime_viewport"]),
        asset_destination=USER_CONFIG["asset_destination"],
        replace_existing_asset=bool(USER_CONFIG["replace_existing_asset"]),
        spawn_actor=bool(USER_CONFIG["spawn_actor"]),
    )
