from __future__ import annotations

from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".exr"}


def choose_output_dir(
    project_root: Path,
    explicit_output_dir: str | None,
    input_dir: Path | None,
    default_folder_name: str,
) -> Path:
    if explicit_output_dir:
        return Path(explicit_output_dir)
    if input_dir is not None:
        return input_dir / default_folder_name
    return project_root / "MAIN" / default_folder_name


def discover_input_files(input_dir: Path) -> tuple[Path, Path, Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    files = sorted([p for p in input_dir.iterdir() if p.is_file()])
    image_files = [p for p in files if p.suffix.lower() in IMAGE_EXTENSIONS]

    def first_existing(candidates: list[Path]) -> Path | None:
        for path in candidates:
            if path.exists():
                return path
        return None

    depth = first_existing(
        [
            input_dir / "original_SceneDepth.exr",
            input_dir / "SceneDepth.exr",
            input_dir / "scene_depth.exr",
        ]
    )
    if depth is None:
        depth_candidates = sorted(
            [
                p
                for p in files
                if p.suffix.lower() == ".exr"
                and "scenedepth" in p.name.lower()
                and "worldunits" not in p.name.lower()
            ]
        )
        if depth_candidates:
            preferred = [p for p in depth_candidates if "00000" in p.stem or "original" in p.stem.lower()]
            depth = preferred[0] if preferred else depth_candidates[0]
    if depth is None:
        raise FileNotFoundError(f"No original SceneDepth EXR found in {input_dir}")

    original = first_existing(
        [
            input_dir / "original.png",
            input_dir / "original.jpg",
            input_dir / "original.jpeg",
            input_dir / "original.exr",
            input_dir / "scene_original.png",
            input_dir / "scene_original.jpg",
            input_dir / "scene_original.exr",
        ]
    )
    if original is None:
        depth_prefix = depth.stem.lower().replace("_scenedepth", "")
        matched_prefix = [
            p for p in image_files
            if p != depth and p.stem.lower() == depth_prefix
        ]
        if matched_prefix:
            original = matched_prefix[0]
    if original is None:
        original_candidates = [
            p for p in image_files
            if "edit" not in p.name.lower()
            and "mask" not in p.name.lower()
            and "depth" not in p.name.lower()
            and "worldnormal" not in p.name.lower()
            and p.name.lower() != "params.json"
        ]
        preferred = [p for p in original_candidates if "00000" in p.stem or "original" in p.stem.lower()]
        if preferred:
            original = preferred[0]
        elif original_candidates:
            original = original_candidates[0]
    if original is None:
        raise FileNotFoundError(f"No original RGB image found in {input_dir}")

    edited = first_existing(
        [
            input_dir / "scene_edit.png",
            input_dir / "scene_edit.jpg",
            input_dir / "scene_edit.jpeg",
            input_dir / "edited.png",
            input_dir / "edited.jpg",
            input_dir / "edit.png",
            input_dir / "edit.jpg",
        ]
    )
    if edited is None:
        edited_candidates = [
            p for p in image_files
            if p != original
            and "edit" in p.name.lower()
            and "depth" not in p.name.lower()
            and "mask" not in p.name.lower()
        ]
        if edited_candidates:
            preferred = [p for p in edited_candidates if "00001" in p.stem or "edited" in p.stem.lower()]
            edited = preferred[0] if preferred else edited_candidates[0]
    if edited is None:
        remaining_images = [
            p for p in image_files
            if p not in {original, depth}
            and "depth" not in p.name.lower()
            and "mask" not in p.name.lower()
            and "worldnormal" not in p.name.lower()
        ]
        preferred = [p for p in remaining_images if "00001" in p.stem]
        if preferred:
            edited = preferred[0]
        elif len(remaining_images) == 1:
            edited = remaining_images[0]
    if edited is None:
        raise FileNotFoundError(f"No edited RGB image found in {input_dir}")

    return original, edited, depth
