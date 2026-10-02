from __future__ import annotations

from pathlib import Path
from typing import Any

from ._bootstrap import import_comfy_modules


class ComfyRuntime:
    """Small semantic API over the ComfyUI primitives used by HeadlessComfy."""

    def __init__(self, model_root: str | Path | None = None):
        self.folder_paths, self.nodes, self.model_management = import_comfy_modules()
        self.model_root = Path(model_root).expanduser().resolve() if model_root else None
        if self.model_root:
            self.register_model_root(self.model_root)

    def register_model_root(self, root: str | Path) -> None:
        root = Path(root).expanduser().resolve()
        mapping = {
            "diffusion_models": root / "diffusion_models",
            "text_encoders": root / "text_encoders",
            "vae": root / "vae",
            "loras": root / "loras",
            "checkpoints": root / "checkpoints",
        }
        for kind, path in mapping.items():
            path.mkdir(parents=True, exist_ok=True)
            self.folder_paths.add_model_folder_path(kind, str(path))

    def load_diffusion_model(self, name: str, dtype: str = "default") -> Any:
        return self.nodes.UNETLoader().load_unet(
            unet_name=name,
            weight_dtype=dtype,
        )[0]

    def load_clip(self, name: str, type: str = "stable_diffusion", device: str = "default") -> Any:
        return self.nodes.CLIPLoader().load_clip(
            clip_name=name,
            type=type,
            device=device,
        )[0]

    def load_vae(self, name: str, patch_headless_output: bool = True) -> Any:
        vae = self.nodes.VAELoader().load_vae(vae_name=name)[0]
        if patch_headless_output:
            self.patch_vae_for_headless(vae)
        return vae

    @staticmethod
    def patch_vae_for_headless(vae: Any) -> Any:
        def safe_process_output(image):
            return ((image.clone() + 1.0) / 2.0).clamp(0.0, 1.0)

        vae.process_output = safe_process_output
        return vae

    def apply_lora(self, model: Any, name: str, strength: float = 1.0) -> Any:
        return self.nodes.LoraLoaderModelOnly().load_lora_model_only(
            model=model,
            lora_name=name,
            strength_model=strength,
        )[0]

    def apply_loras(self, model: Any, loras, clip: Any = None):
        """Apply an ordered stack to the supplied base; return (model, clip).

        Each new job should call this with the original base model/CLIP.
        Trigger words are explicit LoraSpec fields, never guessed from filenames.
        """
        specs = tuple(loras)
        if clip is None and any(spec.clip_strength != 0 for spec in specs):
            raise ValueError("CLIP strength requires a CLIP model")
        for spec in specs:
            if spec.strength == 0 and spec.clip_strength == 0:
                continue
            if spec.clip_strength == 0:
                model = self.apply_lora(model, spec.name, spec.strength)
            else:
                model, clip = self.nodes.LoraLoader().load_lora(
                    model=model, clip=clip, lora_name=spec.name,
                    strength_model=spec.strength, strength_clip=spec.clip_strength,
                )
        return model, clip

    def encode(self, clip: Any, text: str) -> Any:
        return self.nodes.CLIPTextEncode().encode(clip=clip, text=text)[0]

    def zero_conditioning(self, conditioning: Any) -> Any:
        return self.nodes.ConditioningZeroOut().zero_out(conditioning)[0]

    def empty_latent(self, width: int, height: int, batch_size: int = 1) -> Any:
        return self.nodes.EmptyLatentImage().generate(
            width=width,
            height=height,
            batch_size=batch_size,
        )[0]

    def sample(
        self,
        model: Any,
        positive: Any,
        negative: Any,
        latent: Any,
        seed: int,
        steps: int,
        cfg: float = 1.0,
        sampler: str = "euler_ancestral",
        scheduler: str = "beta",
        denoise: float = 1.0,
    ) -> Any:
        return self.nodes.common_ksampler(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent=latent,
            denoise=denoise,
        )[0]

    def upscale_latent(self, latent: Any, scale: float = 1.5, method: str = "bicubic") -> Any:
        return self.nodes.LatentUpscaleBy().upscale(
            samples=latent,
            upscale_method=method,
            scale_by=scale,
        )[0]

    @staticmethod
    def decode_tiled(vae: Any, latent: Any, tile: int = 512, overlap: int = 64):
        import torch

        samples = latent["samples"].detach().clone()
        with torch.no_grad():
            return vae.decode_tiled(
                samples,
                tile_x=tile,
                tile_y=tile,
                overlap=overlap,
            )

    @staticmethod
    def tensor_to_pil(image):
        """Convert ComfyUI image output to PIL, removing singleton batch axes.

        VAE outputs may carry more than one leading batch dimension, e.g.
        [1, 1, height, width, channels]. The original Colab helper handled
        this by repeatedly squeezing leading dimensions of size one.
        """
        import numpy as np
        from PIL import Image

        if isinstance(image, (list, tuple)):
            if not image:
                raise ValueError("Expected a non-empty image batch")
            image = image[0]
        if hasattr(image, "detach"):
            image = image.detach().cpu().numpy()
        image = np.asarray(image)
        while image.ndim > 3 and image.shape[0] == 1:
            image = np.squeeze(image, axis=0)
        if image.ndim != 3:
            raise ValueError(f"Expected HxWxC image after removing singleton batch axes, got {image.shape}")
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255.0).round().astype(np.uint8)
        return Image.fromarray(image)

    def soft_empty_cache(self) -> None:
        self.model_management.soft_empty_cache()

    def unload_all_models(self) -> None:
        self.model_management.unload_all_models()
