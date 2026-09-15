from pathlib import Path
from headless_comfy import ComfyRuntime

MODEL_ROOT = Path("/content/ai_cache")
hc = ComfyRuntime(MODEL_ROOT)

model = hc.load_diffusion_model("moodyKrea2Mix_v70_fp8.safetensors")
clip = hc.load_clip("qwen3vl_4b_fp8_scaled.safetensors", type="krea2")
vae = hc.load_vae("qwen_image_vae.safetensors")

prompt = "portrait photograph"
positive = hc.encode(clip, prompt)
negative = hc.zero_conditioning(positive)
latent = hc.empty_latent(800, 1200)

latent = hc.sample(
    model, positive, negative, latent,
    seed=32454, steps=8, cfg=1.0,
    sampler="euler_ancestral", scheduler="beta",
)

latent = hc.upscale_latent(latent, scale=1.5)
latent = hc.sample(
    model, positive, negative, latent,
    seed=32454, steps=4, cfg=1.0,
    sampler="euler_ancestral", scheduler="beta", denoise=0.39,
)

image = hc.tensor_to_pil(hc.decode_tiled(vae, latent))
image.save("output.png")
