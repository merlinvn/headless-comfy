"""Run the same model configuration on Colab, RunPod or local CUDA Linux.

python examples/portable_smoke.py models.json --output smoke.png
models.json maps unet/clip/vae to configurations accepted by resolve_model_files.
Relative dst_dir values are placed beneath HC_CACHE_ROOT; absolute paths are kept.
"""
import argparse
import json
from pathlib import Path

from headless_comfy import ComfyRuntime, Pipeline, SamplingConfig
from headless_comfy.cli import doctor
from headless_comfy.downloads import cache_root, resolve_model_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--output', type=Path, default=Path('smoke.png'))
    parser.add_argument('--prompt', default='a landscape photograph')
    args = parser.parse_args()
    if doctor(require_cuda=True):
        raise SystemExit('Fix the environment checks before downloading models')
    configs = json.loads(args.config.read_text())
    root = cache_root()
    for cfg in configs.values():
        cfg['dst_dir'] = root / Path(cfg['dst_dir']).expanduser()
    files = resolve_model_files(configs)
    hc = ComfyRuntime(root)
    # Register arbitrary destination directories as well as standard cache folders.
    for key, kind in [('unet', 'diffusion_models'), ('clip', 'text_encoders'), ('vae', 'vae')]:
        hc.folder_paths.add_model_folder_path(kind, str(files[key].parent))
    model = hc.load_diffusion_model(files['unet'].name)
    clip = hc.load_clip(files['clip'].name, type=configs['clip'].get('type', 'stable_diffusion'))
    vae = hc.load_vae(files['vae'].name)
    state = Pipeline.two_pass(base=SamplingConfig(steps=8),
                              hires=SamplingConfig(steps=4, denoise=.39)).run(
        hc, model=model, clip=clip, vae=vae, prompt=args.prompt,
        seed=42, width=512, height=512)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hc.tensor_to_pil(state.images).save(args.output)
    hc.soft_empty_cache()
    print(f'Smoke image: {args.output.resolve()}')


if __name__ == '__main__':
    main()
