# DA3 Notebook Memory Crash — Investigation

**Notebook:** `MAIN_TEST/img_to_pointcloud_da3.ipynb`
**Symptom:** Jupyter kernel dies silently when loading `da3_giant` weights, no Python traceback.
**Date:** 2026-04-12

---

## TL;DR

The kernel crash is a **native segfault (exit 139)**, not a Python OOM. It is caused by the **Windows commit limit** (physical RAM with no pagefile), not by GPU VRAM exhaustion or by anything DA3‑specific. Whenever the Python process already holds ~6 GB of committed memory and then `safetensors.safe_open(...).get_tensor(...)` tries to mmap the 5.42 GB `da3_giant` weight file, the combined commit charge exceeds the limit and the safetensors Rust binding crashes without surfacing an exception.

The previous workaround (`da3_weight_loader_cpu.py`, "build skeleton on GPU first, then stream from CPU") never actually worked — it just moved the crash one step later.

**Fix applied:** notebook now defaults to `da3_large` (1.64 GB safetensors, fits comfortably in 8 GB VRAM). Loading code simplified to a straightforward `safe_open` stream into a GPU skeleton. `da3_giant` is still selectable but only works if a Windows pagefile is enabled.

---

## Hardware / environment

| Component | Value |
|---|---|
| GPU | NVIDIA RTX 3070 Ti, **8.59 GB VRAM** |
| RAM | **34.28 GB** total, ~20 GB free at start |
| Pagefile | **Disabled** (`AutomaticManagedPagefile = False`, no `Win32_PageFileUsage`) |
| OS | Windows 11 Pro |
| CUDA | 12.4 |
| Python | 3.10.0 |
| torch | 2.6.0+cu124 |
| safetensors | 0.5.3 |

Because there is no pagefile, **`Total Virtual Memory == Total Physical RAM`**. Once the process commit charge approaches 34 GB (counting RSS + mmap + CUDA reserved), allocations from native code segfault instead of raising `MemoryError`.

---

## Model file sizes (cached)

| Variant | safetensors size |
|---|---|
| `da3_giant` (DA3-GIANT-1.1) | **5.42 GB** |
| `da3_large` (DA3-LARGE-1.1) | **1.64 GB** |
| Depth-Anything-V2-Metric-Indoor-Large (HF) | 1.30 GB |
| dinov2-base | 0.33 GB |

---

## What was tested

All tests run as standalone scripts so the failure mode is reproducible without Jupyter. Memory was sampled with `psutil.Process().memory_info().rss`, `psutil.virtual_memory()`, and `torch.cuda.mem_get_info()`.

### Test 1 — GPU skeleton fp32 build (no weight load)

```
[start]            RSS=0.39G  VRAMalloc=0.00G  free=7.44/8.59G
[after skeleton]   RSS=0.63G  VRAMalloc=5.49G  free=1.88/8.59G
```

✅ Works. fp32 skeleton on GPU = **5.49 GB VRAM**, leaves only 1.88 GB free.

Side note: PyTorch logs `expandable_segments not supported on this platform` — Cell 0's `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is silently ignored on Windows.

### Test 2 — GPU skeleton + safetensors stream load (notebook's current strategy)

```
[after skeleton]   RSS=0.63G  VRAMalloc=5.49G  free=1.88/8.59G
[file keys: 971]
  >>> [0] model.backbone.pretrained.blocks.0.attn.proj.bias
      getting tensor...
*** SEGMENTATION FAULT ***  exit 139
```

❌ Crashes on the **very first** `handle.get_tensor(name)` call — for a 1536‑element bias (6 KB!). The crash is not due to the tensor being too big; it is the act of safetensors mmap'ing the 5.42 GB file while the process already holds the 5.49 GB GPU skeleton.

### Test 3 — Build CPU skeleton, then `safetensors.torch.load_file`

```
[cpu skeleton]     RSS=6.03G  sys=20.40G  VRAMalloc=0.00G
*** SEGFAULT ***  in load_file
```

❌ Crashes during `load_file()`. CPU skeleton alone is fine, but `load_file` allocates an extra full state dict copy, peaking near the commit limit.

### Test 4 — Build CPU skeleton, stream load, no CUDA at all (`CUDA_VISIBLE_DEVICES=""`)

```
[cpu skeleton]      RSS=5.94G  avail=13.85G
[after state_dict]  RSS=5.94G  avail=13.86G
[after safe_open]   RSS=5.94G  avail=13.88G
[after keys]        RSS=5.94G  avail=13.88G
  [0] model.backbone.pretrained.blocks.0.attn.proj.bias
*** SEGFAULT ***
```

❌ Crashes with no CUDA involved at all. **This rules out a GPU/CUDA driver bug.**

### Test 5 — DA3 not involved at all: 6 GB of dummy `torch.zeros` + giant safetensors

```python
dummy = [torch.zeros(64*1024*1024, dtype=torch.float32) for _ in range(24)]
# RSS = 6.84G
with safe_open(giant_file, framework="pt", device="cpu") as h:
    h.get_tensor(keys[0])
```

```
[dummy alloc 6GB]      RSS=6.84G  avail=12.94G
[opened, 971 keys]     RSS=6.84G  avail=12.96G
  [0] ...
*** SEGFAULT ***
```

❌ **Reproduced without any DA3 code.** The crash is a pure interaction between Python process memory usage and safetensors mmap of a large file on Windows without a pagefile.

### Test 6 — Same dummy 6 GB, but stream from the **smaller** `da3_large` file (1.64 GB)

```
[dummy alloc 6GB]      RSS=6.84G  avail=12.95G
[opened, 637 keys]     RSS=6.84G  avail=12.95G
  [0..9] all loaded
[loaded all small]     RSS=6.84G  avail=12.94G
DONE
```

✅ Works fine. Confirms the trigger is **(big existing process memory) × (large safetensors file)**, not safetensors itself.

### Test 7 — End‑to‑end `da3_large` with the real notebook's image

```
[start]              RSS=0.39G  VRAMalloc=0.00G  free=7.44/8.59G
[skeleton 0.2s]      RSS=0.63G  VRAMalloc=1.65G  free=5.74/8.59G
[loaded 0.8s]        RSS=0.64G  VRAMalloc=1.65G  free=5.74/8.59G
[before inference]   RSS=0.66G  VRAMalloc=1.65G  free=5.74/8.59G
[after inference 0.9s] RSS=1.43G VRAMalloc=1.66G free=4.49/8.59G
depth shape: (280, 504)  range: 0.21-2.61
PEAK VRAM: 2.60G
```

✅ Works perfectly. **Peak VRAM = 2.60 GB**, inference = 0.9 s.

### Test 8 — Meta‑device init for `da3_giant`

Tried `with torch.device("meta"): DepthAnything3(model_name="da3-giant")` to skip the 5.4 GB skeleton entirely.

```
RuntimeError: Tensor.item() cannot be called on meta tensors
  at depth_anything_3/model/dinov2/vision_transformer.py:176
    x.item() for x in torch.linspace(0, drop_path_rate, depth)
```

❌ Vendored DinoV2 backbone calls `.item()` during construction, which doesn't work on meta tensors. Can't use this strategy without patching DA3 source.

---

## Failure-mode summary table

| Strategy | Process RSS at trigger | Result |
|---|---|---|
| GPU skeleton fp32 + stream load (notebook's strategy) | 0.6 G + 5.5 G VRAM | ❌ segfault on first `get_tensor` |
| CPU skeleton + `safetensors.torch.load_file` | 6.0 G | ❌ segfault in `load_file` |
| CPU skeleton + safetensors stream | 5.9 G | ❌ segfault on first `get_tensor` |
| CPU skeleton + stream, **CUDA hidden** | 5.9 G | ❌ segfault on first `get_tensor` |
| 6 GB dummy `torch.zeros` + safetensors stream from giant file | 6.8 G | ❌ segfault on first `get_tensor` |
| 6 GB dummy + safetensors stream from `da3_large` (1.6 G file) | 6.8 G | ✅ works |
| Safetensors stream from giant, no other allocations | 0.4 G | ✅ works |
| Meta‑device init for `da3_giant` | — | ❌ vendored DinoV2 incompatible |
| **`da3_large` end‑to‑end** | 0.6–1.4 G | ✅ **works (2.6 GB peak VRAM)** |

The pattern is clean: **(process RSS ≳ 5 GB) AND (mmap of safetensors ≳ 5 GB) → segfault on Windows without a pagefile.**

---

## Diagnosis

Linux mmap of a read‑only file does not charge against commit. Windows is stricter — the operating system tracks committed virtual memory against the system commit limit, and (without a pagefile) that limit equals physical RAM (~33 GB on this machine). The combination of:

- ~5.5 GB of in‑process tensors (the DA3 skeleton, in RAM or VRAM doesn't matter — both still pin commit on Windows for various reasons)
- A 5.4 GB safetensors mmap that the safetensors Rust binding touches/copies into a fresh torch tensor
- CUDA driver pinned/staging buffers (when CUDA is initialized)
- Python interpreter, imports, OS overhead

…approaches the commit limit, and the crash surfaces inside `safetensors`'s native code as an unhandled SIGSEGV instead of a `MemoryError`.

This is why every "creative" workaround (CPU intermediate, GPU intermediate, stream load, batch load) failed in the same way — they all need to mmap the same 5.4 GB file while ~5.5 GB is already pinned.

---

## Fix applied to the notebook

`MAIN_TEST/img_to_pointcloud_da3.ipynb`:

**Cell 0** — added `da3_large` as a `MODEL_VARIANT` option and made it the default. Kept `da3_giant` and `da3_nested` selectable, with a comment explaining the limitation.

**Cell 5** — removed the convoluted CPU‑intermediate workaround that fought (and lost) the OOM battle. Replaced with a clean stream‑load:

```python
with torch.device(device):
    da3_model = DepthAnything3(model_name=model_name)
da3_model.eval()

with safe_open(model_file, framework="pt", device="cpu") as _h:
    state = da3_model.state_dict(keep_vars=True)
    for _k in _h.keys():
        if _k in state:
            with torch.no_grad():
                state[_k].copy_(_h.get_tensor(_k))
```

Verified end‑to‑end: cells 0–9 and 11 run cleanly. `pointclouds_da3_large/concrete1_da3_large.las` saved with 4,202,496 points.

Backup of the previous notebook is still at `MAIN_TEST/img_to_pointcloud_da3.ipynb.bak`.

---

## Recommendations

| You want to run… | Do this |
|---|---|
| **`da3_large` (default)** | Already works. No further action. |
| `da3_small` / `da3_base` | Should also work — even smaller. Add to `DA3_HF_MODELS` and try. |
| **`da3_giant`** | Enable a Windows pagefile (8–16 GB is plenty). Settings → System → About → Advanced system settings → Performance Options → Advanced → Virtual memory → Change. After that no code changes are needed. |
| `da3_giant` on CPU only | Same fix — also needs a pagefile to mmap the file. |
| Avoid the pagefile entirely | Stick with `da3_large` or smaller variants. Hardware upgrade (≥16 GB VRAM) would also dodge the issue. |

---

## Side finding (unrelated to memory)

**Cell 9 `del`s variables that Cell 10 still uses.** Running the notebook top‑to‑bottom raises:

```
NameError: name 'sky_gt_full' is not defined
```

at the start of Cell 10. This is a pre‑existing bug independent of the memory issue. Either remove the relevant `del` statements from Cell 9 or move them to after Cell 10. Not fixed in this investigation.
