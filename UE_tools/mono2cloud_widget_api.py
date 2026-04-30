"""Widget-facing helpers for the Mono2Cloud Unreal editor flow.

Use this module from an Editor Utility Widget so the widget only needs to:
1. populate a camera dropdown from `get_widget_model()`
2. capture a file path from a file picker / drag-drop target
3. call `run_from_widget(...)`

The heavy lifting stays in `mono2cloud_editor_mvp.py`.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import mono2cloud_editor_mvp as mono2cloud


def _get_backend():
    global mono2cloud
    if not hasattr(mono2cloud, "list_camera_options"):
        mono2cloud = importlib.reload(mono2cloud)
    return mono2cloud


def _copy_default_config() -> dict[str, object]:
    return dict(_get_backend().USER_CONFIG)


def get_widget_model() -> dict[str, object]:
    """Return the data a simple widget needs to render its controls."""
    config = _copy_default_config()
    backend = _get_backend()
    return {
        "backend_file": getattr(backend, "__file__", "<unknown>"),
        "cameras": backend.list_camera_options(),
        "defaults": {
            "repo_root": config["repo_root"],
            "python_exe": config["python_exe"],
            "pipeline": config["pipeline"],
            "mask_model": config["mask_model"],
            "capture_width": int(config["capture_width"]),
            "capture_height": int(config["capture_height"]),
            "capture_warmup_ticks": int(config["capture_warmup_ticks"]),
            "capture_screen_percentage": config["capture_screen_percentage"],
            "capture_aa_quality": config["capture_aa_quality"],
            "force_realtime_viewport": bool(config["force_realtime_viewport"]),
            "asset_destination": config["asset_destination"],
            "replace_existing_asset": bool(config["replace_existing_asset"]),
            "spawn_actor": bool(config["spawn_actor"]),
        },
        "active_runs": [summarize_run(run_handle) for run_handle in backend.get_active_runs()],
    }


def validate_widget_inputs(
    camera_label: str,
    edited_image_path: str,
    *,
    repo_root: str | None = None,
    python_exe: str | None = None,
) -> dict[str, str]:
    """Validate the form inputs and return normalized values for the widget."""
    backend = _get_backend()
    camera_actor = backend._get_camera_actor_by_label(camera_label)
    edited_image = backend._resolve_edited_image(edited_image_path)
    repo_root_path = backend._resolve_repo_root(repo_root or backend.USER_CONFIG["repo_root"])
    python_exe_path = backend._resolve_python_exe(python_exe or backend.USER_CONFIG["python_exe"])
    return {
        "camera_label": camera_actor.get_actor_label(),
        "camera_name": camera_actor.get_name(),
        "edited_image_path": str(edited_image),
        "repo_root": str(repo_root_path),
        "python_exe": str(python_exe_path),
    }


def summarize_run(run_handle: mono2cloud.PipelineRunHandle) -> dict[str, object]:
    result = {
        "status": run_handle.status,
        "session_dir": str(run_handle.session_dir),
        "camera_label": run_handle.camera_actor.get_actor_label() if run_handle.camera_actor else None,
        "edited_image_path": str(run_handle.edited_image_path) if run_handle.edited_image_path else None,
        "error": run_handle.error,
        "output_dir": str(run_handle.output_dir) if run_handle.output_dir else None,
        "las_path": str(run_handle.las_path) if run_handle.las_path else None,
        "imported_asset_paths": list(run_handle.imported_asset_paths),
    }
    if run_handle.summary_path is not None:
        result["summary_path"] = str(run_handle.summary_path)
    return result


def run_from_widget(
    camera_label: str,
    edited_image_path: str,
    *,
    repo_root: str | None = None,
    python_exe: str | None = None,
    pipeline: str | None = None,
    mask_model: str | None = None,
    capture_width: int | None = None,
    capture_height: int | None = None,
    capture_warmup_ticks: int | None = None,
    capture_screen_percentage: float | None = None,
    capture_aa_quality: int | None = None,
    force_realtime_viewport: bool | None = None,
    asset_destination: str | None = None,
    replace_existing_asset: bool | None = None,
    spawn_actor: bool | None = None,
) -> mono2cloud.PipelineRunHandle:
    config = _copy_default_config()
    backend = _get_backend()
    normalized = validate_widget_inputs(
        camera_label,
        edited_image_path,
        repo_root=repo_root,
        python_exe=python_exe,
    )

    return backend.run_camera_label_pipeline(
        camera_label=normalized["camera_label"],
        edited_image_path=normalized["edited_image_path"],
        repo_root=normalized["repo_root"],
        python_exe=normalized["python_exe"],
        pipeline=pipeline or str(config["pipeline"]),
        mask_model=mask_model or str(config["mask_model"]),
        capture_width=int(capture_width if capture_width is not None else config["capture_width"]),
        capture_height=int(capture_height if capture_height is not None else config["capture_height"]),
        capture_warmup_ticks=int(
            capture_warmup_ticks if capture_warmup_ticks is not None else config["capture_warmup_ticks"]
        ),
        capture_screen_percentage=(
            capture_screen_percentage
            if capture_screen_percentage is not None
            else config["capture_screen_percentage"]
        ),
        capture_aa_quality=(
            capture_aa_quality
            if capture_aa_quality is not None
            else config["capture_aa_quality"]
        ),
        force_realtime_viewport=(
            bool(force_realtime_viewport)
            if force_realtime_viewport is not None
            else bool(config["force_realtime_viewport"])
        ),
        asset_destination=asset_destination or str(config["asset_destination"]),
        replace_existing_asset=(
            bool(replace_existing_asset)
            if replace_existing_asset is not None
            else bool(config["replace_existing_asset"])
        ),
        spawn_actor=bool(spawn_actor) if spawn_actor is not None else bool(config["spawn_actor"]),
    )


def normalize_drop_path(raw_path: str) -> str:
    """Normalize a drag-dropped file path before handing it to the pipeline."""
    if not raw_path:
        raise RuntimeError("A PNG/JPG/EXR path is required.")
    return str(Path(raw_path).expanduser().resolve())
