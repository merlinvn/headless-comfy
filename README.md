# headless-comfy

Thin headless Python API over a pinned, vendored ComfyUI snapshot.

The release wheel contains ComfyUI source already, so Colab/RunPod does **not**
need to clone ComfyUI at runtime. Python dependencies are normal pip dependencies.

## Goal

```bash
pip install headless-comfy
```

Then:

```python
from headless_comfy import ComfyRuntime

hc = ComfyRuntime("/content/ai_cache")
model = hc.load_diffusion_model("model.safetensors")
clip = hc.load_clip("clip.safetensors", type="krea2")
vae = hc.load_vae("vae.safetensors")
```

## Development

Vendor the pinned upstream snapshot before building/installing from source:

```bash
python scripts/vendor_comfyui.py
pip install -e .
headless-comfy doctor
```

Build a wheel:

```bash
python scripts/vendor_comfyui.py
python -m pip install build
python -m build --wheel
```

The wheel in `dist/` contains the vendored ComfyUI tree.

## Pinning ComfyUI

`COMFYUI_REF` contains the upstream tag/commit. Update it intentionally, vendor again,
and run smoke tests before publishing a new `headless-comfy` release.

## Why preserve the upstream tree?

ComfyUI currently has top-level modules such as `nodes.py` and `folder_paths.py`,
and upstream code imports those modules using their original layout. The wrapper keeps
the tree intact inside `_vendor/ComfyUI` and adds that root to `sys.path` at runtime.
This minimizes patches and makes updating the snapshot easier.

## Current API

- register model directories
- load diffusion model
- load CLIP/text encoder
- load VAE
- apply model-only LoRA
- text encode / zero conditioning
- empty latent
- KSampler via `common_ksampler`
- latent upscale
- tiled VAE decode
- tensor -> PIL
- unload / soft cache cleanup

This matches the primitives used by the supplied `HeadlessComfy` notebook.
