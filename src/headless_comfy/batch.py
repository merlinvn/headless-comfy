"""Single-process batch runner with persisted seeds and configuration-aware resume."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import tempfile
from typing import Sequence

from .pipeline import LoraSpec, Pipeline


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def atomic_write(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
        temporary = Path(f.name)
    try:
        write(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def cache_file(source: str | Path, directory: str | Path) -> Path:
    """Copy to local disk atomically. Size/mtime checks are not content hashes."""
    source, directory = Path(source), Path(directory)
    if not source.is_file():
        raise FileNotFoundError(source)
    target = directory / source.name
    if source.resolve() == target.resolve():
        return target
    if (not target.exists() or target.stat().st_size != source.stat().st_size
            or target.stat().st_mtime_ns != source.stat().st_mtime_ns):
        atomic_write(target, lambda tmp: shutil.copy2(source, tmp))
    return target


def load_prompts(source: str | Path, output_dir: str | Path) -> list[str]:
    source, output_dir = Path(source), Path(output_dir)
    archive = output_dir / source.name
    actual = source if source.is_file() else archive
    prompts = [line.strip() for line in actual.read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
    if not prompts:
        raise ValueError(f"No prompts in {actual}")
    if actual.resolve() != archive.resolve():
        atomic_write(archive, lambda tmp: shutil.copy2(actual, tmp))
    return prompts


@dataclass(frozen=True)
class LoraStack:
    name: str
    loras: tuple[LoraSpec, ...] = ()
    output_subdir: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "loras", tuple(self.loras))
        if not self.name or self.name in (".", "..") or any(c in self.name for c in "/\\"):
            raise ValueError("Stack name must be a single directory name")
        if self.output_subdir is not None and (
            self.output_subdir in (".", "..") or any(c in self.output_subdir for c in "/\\")
        ):
            raise ValueError("Output subfolder must be a single directory name")

    def prompt(self, base: str) -> str:
        triggers = dict.fromkeys(spec.trigger.strip() for spec in self.loras
                                 if spec.trigger.strip() and (spec.strength or spec.clip_strength))
        return ", ".join([*triggers, base])


class BatchRunner:
    def __init__(self, runtime, pipeline: Pipeline, *, model, clip, vae,
                 model_ids: dict[str, str]):
        self.runtime, self.pipeline = runtime, pipeline
        self.model, self.clip, self.vae = model, clip, vae
        # Include immutable asset hashes/revisions here if filenames can be overwritten.
        self.model_ids = dict(model_ids)

    def run(self, prompts: Sequence[str], stacks: Sequence[LoraStack], output_dir,
            *, seed_mode="per_prompt", sizes=((800, 1200),), skip_existing=True,
            negative_prompt="", on_result=None, group_by_run=True, resume_mode="metadata"):
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo
        from . import __version__
        from ._bootstrap import snapshot_info

        prompts, stacks, sizes = list(prompts), list(stacks), [tuple(s) for s in sizes]
        if resume_mode not in ("metadata", "filename"):
            raise ValueError("Unknown resume mode")
        if seed_mode not in ("per_prompt", "all_random"):
            raise ValueError("Unknown seed mode")
        if not prompts or not stacks or not sizes or len({s.name for s in stacks}) != len(stacks):
            raise ValueError("Need prompts, sizes and uniquely named stacks")
        if any(len(s) != 2 or any(v <= 0 or v % 8 for v in s) for s in sizes):
            raise ValueError("Dimensions must be positive multiples of 8")
        config = {"prompts": prompts, "stacks": [asdict(s) for s in stacks],
                  "pipeline": self.pipeline.metadata(), "models": self.model_ids,
                  "sizes": sizes, "seed_mode": seed_mode, "negative_prompt": negative_prompt,
                  "size_assignment": "shared_per_prompt_v1",
                  "package_version": __version__, "comfyui": snapshot_info(),
                  "group_by_run": group_by_run}
        run_id = fingerprint(config)
        root = Path(output_dir) / run_id[:16]
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if fingerprint(manifest["config"]) != run_id:
                raise ValueError("Manifest configuration mismatch")
        else:
            rng = random.SystemRandom()
            shared = [rng.randrange(2**32) for _ in prompts]
            # per_prompt keeps dimensions aligned across all stacks for a prompt.
            prompt_sizes = ([rng.choice(sizes) for _ in prompts]
                            if seed_mode == "per_prompt" else None)
            # Seed and size are written BEFORE sampling, even in all_random mode.
            jobs = [{"stack": stack.name, "prompt_index": i,
                     "seed": shared[i] if seed_mode == "per_prompt" else rng.randrange(2**32),
                     "size": prompt_sizes[i] if prompt_sizes is not None else rng.choice(sizes)}
                    for stack in stacks for i in range(len(prompts))]
            manifest = {"config": config, "jobs": jobs}
            atomic_write(manifest_path, lambda tmp: tmp.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"))
        results = []
        for stack in stacks:
            model = clip = None
            try:
                for job in (j for j in manifest["jobs"] if j["stack"] == stack.name):
                    if group_by_run:
                        path = root / stack.name / f"p{job['prompt_index'] + 1:04d}_seed{job['seed']}.png"
                    else:
                        subdir = stack.output_subdir if stack.output_subdir is not None else stack.name
                        path = Path(output_dir) / subdir / (
                            f"p{job['prompt_index'] + 1:02d}_{stack.name}_seed{job['seed']}.png")
                    job_id = fingerprint({"run": run_id, "job": job})
                    valid = False
                    if skip_existing and resume_mode == "filename":
                        # Legacy notebook: pXX_loraTag_seed*.png, regardless of seed/config.
                        # Literal matching avoids interpreting LoRA names as glob patterns.
                        prefix = path.name.rsplit("_seed", 1)[0] + "_seed"
                        matches = sorted(p for p in path.parent.iterdir()
                                         if p.is_file() and p.name.startswith(prefix)
                                         and p.name.endswith(".png")) if path.parent.is_dir() else []
                        if matches:
                            path, valid = matches[0], True
                    elif skip_existing and path.exists():
                        try:
                            with Image.open(path) as image:
                                valid = image.info.get("job_id") == job_id
                                image.verify()
                        except (OSError, SyntaxError):
                            pass
                    if valid:
                        result = {"path": str(path), "skipped": True, **job}
                        if resume_mode == "filename":
                            matched_seed = re.search(r"_seed(\d+)\.png$", path.name)
                            result.update(seed=int(matched_seed[1]) if matched_seed else None, size=None)
                    else:
                        if model is None:
                            model, clip = self.runtime.apply_loras(self.model, stack.loras, self.clip)
                        prompt = stack.prompt(prompts[job["prompt_index"]])
                        state = self.pipeline.run(
                            self.runtime, model=model, clip=clip, vae=self.vae, prompt=prompt,
                            seed=job["seed"], width=job["size"][0], height=job["size"][1],
                            negative_prompt=negative_prompt)
                        if state.images is None:
                            raise ValueError("Batch pipeline must produce images (add DecodeStage)")
                        image = self.runtime.tensor_to_pil(state.images)
                        metadata = {"job_id": job_id, "run_id": run_id, "prompt": prompt,
                                    "job": job, "config": config, "final_size": image.size}
                        info = PngInfo()
                        info.add_text("job_id", job_id)
                        info.add_text("generation_metadata", json.dumps(metadata, ensure_ascii=False))
                        atomic_write(path, lambda tmp: image.save(tmp, format="PNG", pnginfo=info))
                        result = {"path": str(path), "skipped": False, **job}
                        del state, image
                    results.append(result)
                    if on_result:
                        on_result(result)
            finally:
                del model, clip
                self.runtime.soft_empty_cache()
        return results
