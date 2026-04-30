from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def _refine(mask: np.ndarray, min_area: int = 500, dilate_iter: int = 2) -> np.ndarray:
    mask = mask.copy()
    labeled, n = ndimage.label(mask)
    for i in range(1, n + 1):
        if int(np.sum(labeled == i)) < int(min_area):
            mask[labeled == i] = False
    if dilate_iter > 0:
        mask = ndimage.binary_dilation(mask, iterations=int(dilate_iter))
    return ndimage.binary_fill_holes(mask)


def _load_dotenv_token(token_env: str) -> str | None:
    if os.environ.get(token_env):
        return os.environ.get(token_env)

    current = Path(__file__).resolve()
    candidates = [
        current.parent / ".env",
        current.parent.parent / ".env",
    ]
    for env_path in candidates:
        if not env_path.is_file():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key != token_env:
                continue
            value = value.strip().strip('"').strip("'")
            if value:
                os.environ.setdefault(token_env, value)
                return value
    return None


def _save_temp_png(image: Image.Image) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    path = Path(tmp.name)
    tmp.close()
    image.save(path, format="PNG")
    return path


def _normalize_category(value: str | None) -> str:
    text = (value or "unknown").strip().lower()
    return text or "unknown"


def _bbox_to_mask(bbox: list[float], shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    x0, y0, x1, y1 = [float(v) for v in bbox]
    left = max(0, min(w, int(np.floor(min(x0, x1)))))
    right = max(0, min(w, int(np.ceil(max(x0, x1)))))
    top = max(0, min(h, int(np.floor(min(y0, y1)))))
    bottom = max(0, min(h, int(np.ceil(max(y0, y1)))))
    mask = np.zeros((h, w), dtype=bool)
    if right > left and bottom > top:
        mask[top:bottom, left:right] = True
    return mask


def _decode_mask(mask_payload: object, bbox: list[float], shape_hw: tuple[int, int]) -> np.ndarray:
    if mask_payload is not None:
        try:
            from pycocotools import mask as mask_utils

            decoded = mask_utils.decode(mask_payload)
            if decoded.ndim == 3:
                decoded = decoded[..., 0]
            return decoded.astype(bool)
        except Exception:
            pass
    return _bbox_to_mask(bbox, shape_hw)


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = int(np.logical_and(mask_a, mask_b).sum())
    if inter <= 0:
        return 0.0
    union = int(np.logical_or(mask_a, mask_b).sum())
    if union <= 0:
        return 0.0
    return float(inter / union)


def _build_prompt(prompt_text: str | None) -> dict[str, str]:
    if prompt_text and prompt_text.strip():
        return {"type": "text", "text": prompt_text.strip()}
    return {"type": "universal"}


def _dinox_detect_objects(
    image_path: Path,
    *,
    token: str,
    model_name: str,
    prompt_text: str | None,
    bbox_threshold: float,
    iou_threshold: float,
) -> list[dict]:
    try:
        from dds_cloudapi_sdk import Client, Config
        from dds_cloudapi_sdk.image_resizer import image_to_base64
        from dds_cloudapi_sdk.tasks.v2_task import V2Task
    except ImportError as exc:
        raise ImportError(
            "DINO-X requires dds-cloudapi-sdk. Install it with: pip install dds-cloudapi-sdk>=0.5.3"
        ) from exc

    client = Client(Config(token))
    task = V2Task(
        api_path="/v2/task/dinox/detection",
        api_body={
            "model": model_name,
            "image": image_to_base64(str(image_path)),
            "prompt": _build_prompt(prompt_text),
            "targets": ["bbox", "mask"],
            "mask_format": "coco_rle",
            "bbox_threshold": float(bbox_threshold),
            "iou_threshold": float(iou_threshold),
        },
    )
    client.run_task(task)
    result = task.result or {}
    return list(result.get("objects") or [])


def _prepare_objects(objects: list[dict], shape_hw: tuple[int, int]) -> list[dict]:
    prepared: list[dict] = []
    for obj in objects:
        bbox = obj.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        mask = _decode_mask(obj.get("mask"), bbox, shape_hw)
        if mask.shape != shape_hw:
            mask = np.array(
                Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    (shape_hw[1], shape_hw[0]), Image.NEAREST
                )
            ) > 127
        prepared.append(
            {
                "category": _normalize_category(obj.get("category")),
                "score": float(obj.get("score", 0.0) or 0.0),
                "bbox": [float(v) for v in bbox],
                "mask": mask.astype(bool),
            }
        )
    return prepared


def _greedy_match(
    original_objects: list[dict],
    edited_objects: list[dict],
    *,
    match_iou: float,
) -> tuple[set[int], set[int]]:
    candidates: list[tuple[float, int, int]] = []
    for orig_idx, orig_obj in enumerate(original_objects):
        for edit_idx, edit_obj in enumerate(edited_objects):
            if orig_obj["category"] != edit_obj["category"]:
                continue
            iou = _mask_iou(orig_obj["mask"], edit_obj["mask"])
            if iou >= match_iou:
                candidates.append((iou, orig_idx, edit_idx))

    matched_orig: set[int] = set()
    matched_edit: set[int] = set()
    for _, orig_idx, edit_idx in sorted(candidates, reverse=True):
        if orig_idx in matched_orig or edit_idx in matched_edit:
            continue
        matched_orig.add(orig_idx)
        matched_edit.add(edit_idx)
    return matched_orig, matched_edit


def dinox_change_mask(
    original_img: Image.Image,
    edited_img: Image.Image,
    *,
    token: str | None = None,
    token_env: str = "DINOX_API_TOKEN",
    model_name: str = "DINO-X-1.0",
    prompt_text: str | None = None,
    bbox_threshold: float = 0.25,
    iou_threshold: float = 0.8,
    match_iou: float = 0.3,
    min_area: int = 500,
    dilate_iter: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    token_value = token or os.environ.get(token_env) or _load_dotenv_token(token_env)
    if not token_value:
        raise RuntimeError(
            f"DINO-X requires an API token. Set {token_env} in your environment or repo .env file."
        )

    shape_hw = (original_img.height, original_img.width)
    orig_path = _save_temp_png(original_img)
    edit_path = _save_temp_png(edited_img)
    try:
        prompt_mode = "text" if prompt_text and prompt_text.strip() else "prompt-free"
        print(f"  Running DINO-X ({prompt_mode}) ...")
        original_objects = _prepare_objects(
            _dinox_detect_objects(
                orig_path,
                token=token_value,
                model_name=model_name,
                prompt_text=prompt_text,
                bbox_threshold=bbox_threshold,
                iou_threshold=iou_threshold,
            ),
            shape_hw,
        )
        edited_objects = _prepare_objects(
            _dinox_detect_objects(
                edit_path,
                token=token_value,
                model_name=model_name,
                prompt_text=prompt_text,
                bbox_threshold=bbox_threshold,
                iou_threshold=iou_threshold,
            ),
            shape_hw,
        )
    finally:
        for path in (orig_path, edit_path):
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass

    print(
        f"  DINO-X objects: original={len(original_objects)} edited={len(edited_objects)} "
        f"(match_iou={match_iou:.2f})"
    )

    matched_orig, matched_edit = _greedy_match(
        original_objects,
        edited_objects,
        match_iou=float(match_iou),
    )

    change_score = np.zeros(shape_hw, dtype=np.float32)
    changed_mask = np.zeros(shape_hw, dtype=bool)

    for idx, obj in enumerate(edited_objects):
        if idx in matched_edit:
            continue
        changed_mask |= obj["mask"]
        change_score[obj["mask"]] = np.maximum(change_score[obj["mask"]], obj["score"])

    for idx, obj in enumerate(original_objects):
        if idx in matched_orig:
            continue
        changed_mask |= obj["mask"]
        change_score[obj["mask"]] = np.maximum(change_score[obj["mask"]], obj["score"])

    changed_mask = _refine(changed_mask, min_area=int(min_area), dilate_iter=int(dilate_iter))
    change_score *= changed_mask.astype(np.float32)
    return changed_mask, change_score
