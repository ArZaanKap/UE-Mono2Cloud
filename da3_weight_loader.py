from __future__ import annotations

from os import PathLike
from typing import List, Tuple

import torch
from safetensors import safe_open


def load_model_streaming(
    model: torch.nn.Module,
    filename: str | PathLike[str],
    *,
    strict: bool = True,
    log_every: int = 250,
) -> Tuple[List[str], List[str]]:
    """Load a safetensors checkpoint into an existing module without building a full state_dict copy.

    This keeps peak host RAM much lower than `safetensors.torch.load_model`, which first
    materializes the entire checkpoint as a Python dict of tensors before calling
    `model.load_state_dict(...)`.
    """

    state = model.state_dict(keep_vars=True)
    expected_keys = set(state.keys())
    loaded_keys = set()
    unexpected: List[str] = []

    with safe_open(str(filename), framework="pt", device="cpu") as handle:
        file_keys = list(handle.keys())
        total = len(file_keys)

        for idx, name in enumerate(file_keys, start=1):
            if name not in state:
                unexpected.append(name)
                continue

            dst = state[name]
            src = handle.get_tensor(name)

            if src.shape != dst.shape:
                raise RuntimeError(
                    f"Shape mismatch for {name}: checkpoint has {tuple(src.shape)}, "
                    f"model expects {tuple(dst.shape)}"
                )

            if src.dtype != dst.dtype or src.device != dst.device:
                src = src.to(device=dst.device, dtype=dst.dtype)

            with torch.no_grad():
                dst.copy_(src)

            loaded_keys.add(name)

            if log_every and (idx % log_every == 0 or idx == total):
                print(f"  loaded {idx}/{total} tensors", flush=True)

            del src

    missing = sorted(expected_keys - loaded_keys)

    if strict and (missing or unexpected):
        parts = [f"Error(s) in loading state_dict for {model.__class__.__name__}:"]
        if missing:
            parts.append(
                "    Missing key(s) in state_dict: "
                + ", ".join(f'"{key}"' for key in missing)
            )
        if unexpected:
            parts.append(
                "    Unexpected key(s) in state_dict: "
                + ", ".join(f'"{key}"' for key in unexpected)
            )
        raise RuntimeError("\n".join(parts))

    return missing, unexpected
