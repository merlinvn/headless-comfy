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

## Pipelines and multi-LoRA (0.2.0)

Use `notebooks/HeadlessComfyPipelines.ipynb` in Colab. The setup cell installs
the 0.2.0 wheel from the GitHub release and upgrades the Colab CUDA PyTorch packages.
The supplied notebooks in Downloads were reviewed as source material and left intact.

```python
from headless_comfy import LoraSpec, Pipeline, SamplingConfig
from headless_comfy.batch import BatchRunner, LoraStack

stacks = [
    LoraStack("subject_style", (
        LoraSpec("subject.safetensors", strength=0.7, trigger="subject_word"),
        LoraSpec("style.safetensors", strength=0.3, trigger="style_word"),
    )),
    LoraStack("base"),
]
pipeline = Pipeline.two_pass(
    base=SamplingConfig(steps=8),
    hires=SamplingConfig(steps=4, denoise=0.39), scale=1.5,
)
runner = BatchRunner(hc, pipeline, model=model, clip=clip, vae=vae,
                     model_ids={"unet": "model-revision", "clip": "encoder-revision",
                                "vae": "vae-revision", "lora:subject": "subject-revision",
                                "lora:style": "style-revision"})
results = runner.run(["portrait photograph"], stacks, "/content/drive/MyDrive/outputs",
                     sizes=((800, 1200),), seed_mode="per_prompt")
```

A stack applies all its LoRAs in order, with independent strengths and explicit
triggers. The Colab notebook retains your `infer_lora_trigger()` and rsync
`cache_file()` helpers; `cache_lora()` combines caching, registration checks and
trigger inference. Omit the trigger to infer it, supply a string to override it,
or supply `""` to disable it. Every stack starts from the base model and CLIP, preventing accumulation
between jobs. `clip_strength` defaults to zero for compatibility with the existing
Krea2 model-only workflow; set it for LoRAs trained with text encoder weights.

`per_prompt` shares a prompt's seed across stacks; `all_random` assigns a seed to
each image. Both persist seeds and chosen dimensions before inference. A configuration
hash selects the manifest directory, and resume checks PNG job metadata and file integrity.
Changed prompts, strengths, models or pipeline settings create a new run. Provide
immutable model/LoRA identities (ideally SHA256); filenames alone cannot detect replaced
weights. To reroll the same configuration, select a new output directory. One process
should own an output directory; concurrent writers are not supported.

PNG metadata includes the entire ordered stack, sampling stages, models, seed,
resolution, package version and vendored commit. These record the recipe; results
can still vary with Torch/CUDA versions and hardware. Dependencies follow upstream
constraints and are not a full environment lock.

### Compose other pipelines

```python
from headless_comfy import DecodeStage, LatentUpscaleStage, SampleStage

single = Pipeline((SampleStage(SamplingConfig()), DecodeStage()))
upscale_refine = Pipeline((LatentUpscaleStage(2.0),
                          SampleStage(SamplingConfig(steps=4, denoise=0.3)),
                          DecodeStage()))
# upscale_refine.run(..., latent=existing_latent)
```

Each stage implements `run(runtime, state)` and JSON-serializable `metadata()`.
A future image-model upscale stage can operate on `state.images` after decode;
this release supplies latent upscale, not an ESRGAN/image-model backend.
`Pipeline.run(..., on_stage=callback)` supports previews/inspection without embedding
notebook display code in the runtime. Batch pipelines must produce decoded images.
Pipeline execution runs under `torch.no_grad()` so ComfyUI can create Parameters
during LoRA application and sampling without building gradient graphs. Existing low-level runtime APIs
remain available, including `apply_lora`.

## Upgrade the pinned engine and package

1. Set `COMFYUI_REF` to a release tag or full commit SHA. It stays at `v0.35.0`
   for this change; verify new pins against your GPU/models before upgrading.
2. Run `python scripts/vendor_comfyui.py --sync-dependencies`. This copies the
   selected upstream requirements into `pyproject.toml` and records its resolved
   commit in the wheel. Review dependency changes; Torch/CUDA still depends on your
   execution platform. Vendoring without the flag fails if dependencies drift.
3. Run `python scripts/bump_version.py 0.2.1` for the next package release.
4. Run `PYTHONPATH=src python -m unittest discover -s tests -v`, build the wheel,
   then run a Colab GPU smoke test with your actual model files and LoRA combination.
5. Publish your tested release and change the Colab installer version. Restart the
   runtime after upgrading; imported top-level ComfyUI modules cannot be safely
   swapped in a running Python process.

`headless-comfy doctor` prints both package and engine provenance. Updating pip
packages alone does not change the vendored ComfyUI source. The dependency sync
includes upstream UI packages because the intact upstream tree imports them;
removing them requires a separate compatibility audit.

## Findings from the supplied notebooks

- Single-image generation already chains multiple LoRAs; batch generation instead
  loops over individual files. Shared stacks now work for both flows.
- Two-pass sampling was duplicated across single and batch cells. Named stage
  configurations now share that logic and preserve the existing 8/4-step defaults.
- Resume searched by prompt index/LoRA filename. It could incorrectly skip outputs
  after changing prompt text, strength or sampling. Configuration-specific manifests
  and PNG verification address that, and persist random sizes/seeds on interruption.
- Your filename trigger inference remains the notebook default, including the
  Krea2 naming pattern and legacy fallback. Explicit overrides and an empty string
  for LoRAs without a trigger remain available.
- The seed helper was called twice, save cells referenced obsolete single-LoRA
  variables, and the final cell contained a stray `e`. The new notebook removes
  that duplicated/state-dependent code and embedded output images.
- `__version__` previously said 0.1.0 while project metadata said 0.1.1. Version
  bumping and tests now keep them synchronized. Three dependency pins also differed
  from the selected engine's requirements; they now match the engine snapshot.
- Avoid unconditional `pip install -U torch ...` in each Colab session. Keep the
  platform's compatible stack unless a tested engine requirement needs a change.
- In `00.DownloadModels2Gdrive.ipynb`, add an HTTP timeout, validate a redirect's
  Location explicitly, and verify/checksum completed downloads. File existence alone
  can mistake a partial download for a finished one. The ModelScope-to-Drive copy
  should create its destination parent and publish through a temporary file. Those
  downloader changes are recommendations; the original downloader is unchanged.

Local tests exercise composition, LoRA routing, resume after interruption, corrupt
PNG recovery, configuration changes, cache refresh and version/dependency consistency.
They use a fake sampler; real model compatibility and image quality require the Colab
GPU smoke test. This refactor improves workflow reliability, not model quality itself.

### Original batch configuration and output folders

The Colab notebook preserves the old `BATCH_LORAS` list (including commented
choices), all seven `INPUT_OUTPUT` mappings and `all_random` seed mode. Every
`BATCH_LORAS` entry runs on every prompt in each input file. Existing filename
entries remain single-character jobs; named combinations generate multi-LoRA images.

The notebook calls `runner.run(..., group_by_run=False)` and assigns each stack's
`output_subdir` to its inferred trigger. Images retain the original layout:
`OUTPUT_DIR / trigger / pXX_compactLoraName_seedNN.png`. Manifests still live in
configuration-specific subfolders. Multiple checkpoints with the same trigger share
the folder but have distinct LoRA tags in filenames. Package users retain the
default `group_by_run=True` layout. The notebook selects `resume_mode="filename"`, matching the old skip logic:
any `pXX_compactLoraName_seed*.png` in the subfolder skips that prompt/LoRA,
regardless of seed or metadata. `SKIP_EXISTING=False` disables skipping.
Filename resume ignores changed settings and does not check PNG integrity.
The package default `resume_mode="metadata"` retains configuration-aware PNG verification.

`MODEL_IDS` contains weight identities for resume checks only; it does not select
LoRAs, prompts, or output folders. Those are controlled by `BATCH_LORAS`,
`STACKS` and `INPUT_OUTPUT`.

### Configure the next batch: single characters and combinations

`BATCH_LORAS` accepts the original filename entries, single dictionaries for custom
strengths, and named dictionaries with a `loras` list for combined characters.
Omitted strengths default to `1.0`; no shared strength variable is needed:

```python
BATCH_LORAS = [
    "single_character.safetensors",  # defaults to strength 1.0
    {"name": "char_a.safetensors", "strength": 0.4},
    {"name": "char_a_b_c", "loras": [
        {"name": "char_a.safetensors", "strength": 0.4},
        {"name": "char_b.safetensors", "strength": 0.2},
        {"name": "char_c.safetensors", "strength": 0.3},
    ]},
]
```

The notebook's `build_lora_stack()` caches each member, infers its trigger (unless
explicitly overridden), and constructs one image variant per entry. Combinations
save under their name, such as `OUTPUT_DIR/char_a_b_c/`; single characters retain
their trigger folders. Member order and independent strengths are recorded in PNG
metadata. Each new entry starts from the original base model and CLIP.

`RANDOM_ASPECT_RATIO=True` selects a size per image from `ASPECT_RATIOS`; `False`
uses `(WIDTH, HEIGHT)`. This flag applies in both `all_random` and `per_prompt`
seed modes, independently of seed selection. Dimensions remain persisted for resume.
The old LoRA choices and seven input/output mappings remain in the notebook.
