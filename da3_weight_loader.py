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
    log_every: int = 50,
) -> Tuple[List[str], List[str]]:
    """Load a safetensors checkpoint into an existing on-device module.

    Loads tensors directly to the model's device (e.g. CUDA) via safetensors,
    then copies into the model parameters in-place.  This avoids allocating
    large CPU tensors, which can OOM the system RAM on Windows.
    """

    # Detect the device the model lives on (all params should share one device).
    model_device = "cpu"
    for p in model.parameters():
        model_device = str(p.device)
        break

    state = model.state_dict(keep_vars=True)
    expected_keys = set(state.keys())
    loaded_keys: set[str] = set()
    unexpected: List[str] = []

    # device=model_device → get_tensor returns tensors already on CUDA (or
    # wherever the model lives).  Safetensors does this via a direct DMA read
    # from the memory-mapped file into a device buffer, so peak *host* RAM
    # stays near zero.
    with safe_open(str(filename), framework="pt", device=model_device) as handle:
        file_keys = list(handle.keys())
        total = len(file_keys)

        for idx, name in enumerate(file_keys, start=1):
            if name not in state:
                unexpected.append(name)
                continue

            dst = state[name]
            src = handle.get_tensor(name)  # already on model_device

            if src.shape != dst.shape:
                raise RuntimeError(
                    f"Shape mismatch for {name}: checkpoint has {tuple(src.shape)}, "
                    f"model expects {tuple(dst.shape)}"
                )

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
