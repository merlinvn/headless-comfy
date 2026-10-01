"""Composable latent pipelines; no GPU imports until execution."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class LoraSpec:
    name: str
    strength: float = 1.0
    trigger: str = ""
    clip_strength: float = 0.0

    def __post_init__(self):
        if not self.name or not all(math.isfinite(x) for x in (self.strength, self.clip_strength)):
            raise ValueError("LoRA needs a name and finite strengths")


@dataclass(frozen=True)
class SamplingConfig:
    steps: int = 8
    cfg: float = 1.0
    sampler: str = "euler_ancestral"
    scheduler: str = "beta"
    denoise: float = 1.0

    def __post_init__(self):
        if self.steps < 1 or not math.isfinite(self.cfg) or not 0 < self.denoise <= 1:
            raise ValueError("Invalid sampling steps, cfg or denoise")


@dataclass
class PipelineState:
    model: Any
    clip: Any
    vae: Any
    positive: Any
    negative: Any
    seed: int
    latent: Any = None
    images: Any = None


class Stage(Protocol):
    def run(self, runtime: Any, state: PipelineState) -> None: ...
    def metadata(self) -> dict: ...


@dataclass(frozen=True)
class SampleStage:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    def run(self, runtime, state):
        if state.latent is None:
            raise ValueError("Sampling requires a latent")
        state.latent = runtime.sample(state.model, state.positive, state.negative,
                                     state.latent, seed=state.seed, **asdict(self.sampling))
        state.images = None

    def metadata(self):
        return {"type": "sample", **asdict(self.sampling)}


@dataclass(frozen=True)
class LatentUpscaleStage:
    scale: float = 1.5
    method: str = "bicubic"

    def __post_init__(self):
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be positive and finite")

    def run(self, runtime, state):
        if state.latent is None:
            raise ValueError("Latent upscale requires a latent")
        state.latent = runtime.upscale_latent(state.latent, self.scale, self.method)
        state.images = None

    def metadata(self):
        return {"type": "latent_upscale", "scale": self.scale, "method": self.method}


@dataclass(frozen=True)
class DecodeStage:
    tile: int = 512
    overlap: int = 64

    def __post_init__(self):
        if self.tile <= 0 or not 0 <= self.overlap < self.tile:
            raise ValueError("Decode overlap must be smaller than tile")

    def run(self, runtime, state):
        if state.latent is None:
            raise ValueError("Decode requires a latent")
        state.images = runtime.decode_tiled(state.vae, state.latent, self.tile, self.overlap)

    def metadata(self):
        return {"type": "decode", "tile": self.tile, "overlap": self.overlap}


@dataclass(frozen=True)
class Pipeline:
    stages: tuple[Stage, ...]

    def __post_init__(self):
        object.__setattr__(self, "stages", tuple(self.stages))
        if not self.stages:
            raise ValueError("A pipeline needs at least one stage")

    @classmethod
    def two_pass(cls, base: SamplingConfig | None = None,
                 hires: SamplingConfig | None = None, scale: float = 1.5):
        return cls((SampleStage(base or SamplingConfig()), LatentUpscaleStage(scale),
                    SampleStage(hires or SamplingConfig(steps=4, denoise=0.39)), DecodeStage()))

    def metadata(self):
        return [stage.metadata() for stage in self.stages]

    def run(self, runtime, *, model, clip, vae, prompt: str, seed: int,
            width: int = 800, height: int = 1200, negative_prompt: str = "",
            latent=None, on_stage: Callable | None = None):
        import torch

        if not 0 <= seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        if latent is None and (width <= 0 or height <= 0 or width % 8 or height % 8):
            raise ValueError("Dimensions must be positive multiples of 8")
        with torch.inference_mode():
            positive = runtime.encode(clip, prompt)
            negative = (runtime.encode(clip, negative_prompt) if negative_prompt
                        else runtime.zero_conditioning(positive))
            state = PipelineState(model, clip, vae, positive, negative, seed,
                                  latent if latent is not None else runtime.empty_latent(width, height))
            for index, stage in enumerate(self.stages):
                stage.run(runtime, state)
                if on_stage:
                    on_stage(index, stage, state)
        return state
