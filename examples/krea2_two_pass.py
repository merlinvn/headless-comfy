from headless_comfy import ComfyRuntime, LoraSpec, Pipeline

hc = ComfyRuntime('/content/ai_cache')
base_model = hc.load_diffusion_model('moodyKrea2Mix_v70_fp8.safetensors')
base_clip = hc.load_clip('qwen3vl_4b_fp8_scaled.safetensors', type='krea2')
vae = hc.load_vae('qwen_image_vae.safetensors')
# Empty tuple = base model; add multiple specs to compose LoRAs.
loras: tuple[LoraSpec, ...] = ()
model, clip = hc.apply_loras(base_model, loras, base_clip)
state = Pipeline.two_pass().run(hc, model=model, clip=clip, vae=vae,
                                prompt='portrait photograph', seed=32454,
                                width=800, height=1200)
hc.tensor_to_pil(state.images).save('output.png')
