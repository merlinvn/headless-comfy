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
the matching release wheel. Keep the provider's compatible PyTorch/CUDA stack.
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
triggers. The Colab notebook retains your `infer_lora_trigger()` helper and uses the package
`downloads.cache_file()`; `cache_lora()` combines caching, registration checks and
trigger inference. Omit the trigger to infer it, supply a string to override it,
or supply `""` to disable it. Every stack starts from the base model and CLIP, preventing accumulation
between jobs. `clip_strength` defaults to zero for compatibility with the existing
Krea2 model-only workflow; set it for LoRAs trained with text encoder weights.

`per_prompt` shares a prompt's seed and randomly assigned dimensions across stacks;
`all_random` assigns a seed and dimensions to each image. Size assignments are
shuffled and balanced across the batch, with counts differing by at most one.
Both persist seeds and chosen dimensions before inference. A configuration
hash selects the manifest directory, and resume checks PNG job metadata and file integrity.
Changing LoRA stacks reuses that manifest folder; each stack's jobs and image metadata
remain distinct. Changed prompts, base models, sizes or pipeline settings create a new
folder. Provide immutable model/LoRA identities (ideally SHA256); filenames alone cannot
detect replaced weights. To reroll the same configuration, select a new output directory.
One process should own an output directory; concurrent writers are not supported.

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

1. Set `COMFYUI_REF` to a release tag or full commit SHA. The current pin is
   `v0.39.0`; verify new pins against your GPU/models before upgrading.
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

`RANDOM_ASPECT_RATIO=True` selects sizes from `ASPECT_RATIOS`; `False` uses
`(WIDTH, HEIGHT)`. Size choices are shuffled and balanced across the batch, with
counts differing by at most one. In `per_prompt` mode, one size is shared by all
LoRA stacks for a prompt. In `all_random` mode, each image gets its own size.
Dimensions remain persisted for resume.
The old LoRA choices and seven input/output mappings remain in the notebook.

## Portable model downloads (0.3.1)

`headless_comfy.downloads` has no Colab/Drive/RunPod imports, GPU initialization,
provider detection, shell setup, or automatic OS package installation. Python
`requests` and `filelock` are already package dependencies. `rsync` is unnecessary;
local files use atomic `shutil.copy2`. `aria2c` is an optional backend installed by
the environment owner. The default backend is streaming `requests`.

```python
from headless_comfy.downloads import cache_root, resolve_model_files
from headless_comfy import ComfyRuntime

root = cache_root()  # HC_CACHE_ROOT or ~/.cache/headless_comfy
MODEL = {
    "unet": {
        "civitai": {
            "model_version_id": 3091481, "file_id": 2971089,
            "filename": "krea2_turbo_int8_convrot.safetensors",
        },
        "dst_dir": root / "diffusion_models",
    },
    "clip": {
        "hf": {"repo_id": "Comfy-Org/Krea-2",
               "filename": "text_encoders/qwen3vl_4b_fp8_scaled.safetensors"},
        "dst_dir": root / "text_encoders", "type": "krea2",
    },
    "vae": {
        "hf": {"repo_id": "Comfy-Org/Krea-2",
               "filename": "vae/qwen_image_vae.safetensors"},
        "dst_dir": root / "vae",
    },
}
files = resolve_model_files(MODEL, workers=3)
hc = ComfyRuntime(root)
model = hc.load_diffusion_model(files["unet"].name)
clip = hc.load_clip(files["clip"].name, type=MODEL["clip"]["type"])
vae = hc.load_vae(files["vae"].name)
```

These IDs are illustrative and availability/access can change. Replace them with
your accessible models. A source may instead be `{"src": "/volume/model.safetensors",
"dst_dir": root / "loras"}`. The resolver accepts one remote source (`civitai` or `hf`) with optional `src`
fallback, or `src` alone. It rejects
conflicting configurations targeting the same destination in a parallel batch.
When both remote and `src` are configured, a valid remote cache/download wins.
After remote retries fail with `DownloadError`, the resolver warns and copies
`src` from Drive/local storage. Invalid configuration and filesystem errors do
not trigger fallback. The returned path uses the local source basename, which
may differ from the remote filename. Explicit remote SHA256/size constraints also
apply to fallback unless overridden by top-level local constraints. With no
hash, you are responsible for selecting suitable fallback weights. Subsequent
calls try the remote source again; fallback does not permanently disable it.

Hugging Face filenames may include repo subdirectories; destination names use the
basename, so use separate destination directories to avoid collisions.

Tokens can be explicit (`civitai_token=...`, `hf_token=...`) or environment values:
`CIVITAI_TOKEN` (fallback `CIVITAI_KEY`) and `HF_TOKEN`. Private/gated HF downloads
use bearer authentication; your account must have repository access. Bearer tokens
are stripped on cross-host redirects by requests. Do not put tokens in URLs.
Only completion identity hashes and remote validators are written to sidecars.

Choose a cache location in the bootstrap, using the **same MODEL configuration**:

| Environment | HC_CACHE_ROOT | Storage lifetime |
| --- | --- | --- |
| Colab | `/content/ai_cache` | Temporary runtime disk |
| RunPod Pod | `/workspace/ai_cache` | Depends on the attached volume and mount |
| Local Linux | `~/.cache/headless_comfy` | Local disk |
| Other GPU containers | A writable mounted directory | Depends on storage configuration |

Colab bootstrap can mount Drive and read `google.colab.userdata` outside the
package. Put secrets into environment variables or pass them directly. On RunPod,
configure secrets and persistent storage in the Pod template. `/workspace` alone
does not guarantee persistence through Pod deletion. Network volumes and volume
disks have different lifecycles ([RunPod storage docs](https://docs.runpod.io/pods/storage/types)).
Colab VMs have limited lifetimes ([Colab FAQ](https://research.google.com/colaboratory/faq.html)).
The supplied notebook is still a Colab/Drive example; skip its Drive cell and
set `HC_SOURCE_ROOT` and replace source/output paths for other environments.

### Completion, resume and integrity

Each target has an OS-backed `filelock` shared by threads/processes. Downloads
write `.part` plus `.part.json`, then atomically replace the target and record
`.hc.json` only after validation. Keep these files together on persistent storage.
Do not remove `.lock` files while workers are active. Shared/network filesystems
must support consistent file locking and atomic rename across all writers;
test this on your actual volume. The lock does not protect unrelated downloaders.

Retries start from the original provider API/resolve URL to refresh signed CDN
URLs. HTTP resume uses a strong ETag or Last-Modified with Range/If-Range. If the
server ignores Range or changes the validator, the partial file is truncated.
Without a validator, an interrupted HTTP transfer starts over. Existing completed
files remain available if a replacement download fails.

Supply `sha256` and/or exact `expected_size` in the `hf`/`civitai` source config
(or top level for `src`). Available Civitai SHA256 metadata is used automatically;
rounded `sizeKB` is not treated as an exact byte count. Content-Length validates
transport size when available. **A completion marker and size cannot detect every
same-size corruption**; SHA256 validates content on download and cache reuse.
Files predating sidecars are copied/downloaded once to establish completion.

Pin HF `revision` to a commit and Civitai to a version/file ID. Cached mutable HF
branches are not revalidated remotely; set top-level `force=True` to refresh.
For local sources, identity includes resolved source path, size and mtime; use
SHA256 if files can change while keeping the same size and timestamp.

Downloads and Drive/local copies show a progress bar by default; set
`show_progress=False` on a model config or low-level download call to hide it.
Progress displays the filename, byte count, speed and ETA without printing signed URLs.

Top-level remote options: `backend="requests"` or `"aria2"`, `connections=16`
(1–16; aria2 only), `retries=3`, `timeout=(15, 120)`, `force=False`. `workers`
controls parallel files; requests uses one stream per file. Low-level public API:
`aria2_download(url, directory, filename, ...)`. Install aria2 explicitly in a
Debian/Ubuntu image if desired:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends aria2 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
```

### Compatibility verification

Downloader tests use real localhost HTTP transfers for interruption, cache
corruption, renewed redirects, changed sources, ignored ranges, authentication
redirect boundaries and competing processes. They do not certify provider uptime,
private repository permissions, distributed volume locking or GPU inference.
The downloader CI uses four configurations: Ubuntu/Python 3.10 and 3.13,
macOS/Python 3.13, and Windows/Python 3.13. It runs for related source, dependency,
test or workflow changes on branches/PRs, or manually; release tags do not rerun
the matrix. Ubuntu jobs exercise real aria2 transfers.

Inference still requires a compatible PyTorch/CUDA stack, sufficient GPU memory,
and models supported by the pinned ComfyUI engine. Avoid loading another ComfyUI
installation in the same Python process. `headless-comfy doctor --require-cuda`
returns failure when Torch/CUDA or the vendored snapshot is unavailable.
The downloader also works without CUDA; this does not imply model inference will.

For an actual GPU smoke test, save the MODEL config as JSON (relative `dst_dir`
values such as `diffusion_models`, `text_encoders`, `vae`), then run in each target:

```bash
export HC_CACHE_ROOT=/path/to/cache
headless-comfy doctor --require-cuda
python examples/portable_smoke.py models.json --output smoke.png
```

Inspect the PNG, run again to verify cache reuse, interrupt a download and restart,
then restart the Pod/runtime to check the configured storage lifetime. This script
uses the Krea2 two-pass defaults; adapt the pipeline for other model families.
The notebook installer uses the matching published wheel. For unreleased changes,
set `HC_WHEEL_URL` to a locally built wheel in the notebook.
