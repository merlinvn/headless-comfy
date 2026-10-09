import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from headless_comfy import (ComfyRuntime, DecodeStage, LatentUpscaleStage, LoraSpec,
                            Pipeline, SamplingConfig)
from headless_comfy.batch import BatchRunner, LoraStack, cache_file


class FakeRuntime:
    def __init__(self):
        self.calls = []
        self.applied = []

    def apply_loras(self, model, loras, clip):
        self.applied.append((model, tuple(loras)))
        return model + ''.join(s.name for s in loras), clip

    def encode(self, clip, text):
        return text

    def zero_conditioning(self, positive):
        return 'zero'

    def empty_latent(self, width, height):
        return (width, height)

    def sample(self, model, positive, negative, latent, **kwargs):
        self.calls.append(('sample', model, positive, negative, kwargs))
        return latent

    def upscale_latent(self, latent, scale, method):
        self.calls.append(('upscale', scale))
        return tuple(round(v * scale) for v in latent)

    def decode_tiled(self, vae, latent, tile, overlap):
        from PIL import Image
        self.calls.append(('decode',))
        return Image.new('RGB', latent)

    def tensor_to_pil(self, images):
        return images

    def soft_empty_cache(self):
        pass


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.torch_patch = patch.dict(sys.modules, {'torch': SimpleNamespace(no_grad=contextlib.nullcontext, inference_mode=lambda: (_ for _ in ()).throw(AssertionError('Pipeline must use no_grad for ComfyUI compatibility')))})
        self.torch_patch.start()
        self.addCleanup(self.torch_patch.stop)

    def runner(self, runtime, pipeline=None):
        return BatchRunner(runtime, pipeline or Pipeline.two_pass(), model='base', clip='clip',
                           vae='vae', model_ids={'unet': 'model-v1'})

    def test_pipeline_order_seed_negative_and_initial_latent(self):
        runtime = FakeRuntime()
        state = Pipeline.two_pass().run(runtime, model='base', clip='clip', vae='vae',
                                       prompt='hello', negative_prompt='bad', seed=42,
                                       latent=(16, 24))
        self.assertEqual([c[0] for c in runtime.calls], ['sample', 'upscale', 'sample', 'decode'])
        self.assertEqual(state.images.size, (24, 36))
        samples = [c for c in runtime.calls if c[0] == 'sample']
        self.assertEqual([c[4]['seed'] for c in samples], [42, 42])
        self.assertEqual(samples[1][4]['denoise'], 0.39)
        self.assertEqual(samples[0][3], 'bad')

    def test_stack_order_clip_and_zero_strength(self):
        runtime = object.__new__(ComfyRuntime)
        calls = []
        def model_only(**kw):
            calls.append(kw); return (kw['model'] + kw['lora_name'],)
        def both(**kw):
            calls.append(kw); return kw['model'] + kw['lora_name'], kw['clip'] + 'patched'
        runtime.nodes = SimpleNamespace(
            LoraLoaderModelOnly=lambda: SimpleNamespace(load_lora_model_only=model_only),
            LoraLoader=lambda: SimpleNamespace(load_lora=both))
        specs = [LoraSpec('a', .3), LoraSpec('off', 0), LoraSpec('b', .7, clip_strength=.2)]
        self.assertEqual(runtime.apply_loras('base', specs, 'clip'), ('baseab', 'clippatched'))
        self.assertEqual([c['lora_name'] for c in calls], ['a', 'b'])
        with self.assertRaises(ValueError):
            runtime.apply_loras('base', specs)

    def test_batch_resume_corruption_and_config_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = FakeRuntime()
            stacks = [LoraStack('combo', (LoraSpec('a', .3, 'word'), LoraSpec('b', .7, 'word'))),
                      LoraStack('base')]
            runner = self.runner(runtime)
            first = runner.run(['hello'], stacks, directory, sizes=((16, 24),), seed_mode='all_random')
            second = runner.run(['hello'], stacks, directory, sizes=((16, 24),), seed_mode='all_random')
            self.assertTrue(all(r['skipped'] for r in second))
            self.assertEqual([r['seed'] for r in first], [r['seed'] for r in second])
            self.assertEqual([a[0] for a in runtime.applied], ['base', 'base'])
            self.assertEqual(stacks[0].prompt('hello'), 'word, hello')
            Path(first[0]['path']).write_bytes(b'broken')
            repaired = runner.run(['hello'], stacks, directory, sizes=((16, 24),), seed_mode='all_random')
            self.assertFalse(repaired[0]['skipped'])
            self.assertEqual(repaired[0]['seed'], first[0]['seed'])
            changed = runner.run(['changed'], stacks, directory, sizes=((16, 24),), seed_mode='all_random')
            self.assertNotEqual(first[0]['path'], changed[0]['path'])
            manifests_before_stack_change = list(Path(directory).rglob('manifest.json'))
            changed_stack = runner.run(['hello'], [LoraStack('combo', (LoraSpec('a', .4),))],
                                       directory, sizes=((16, 24),), seed_mode='all_random')
            self.assertEqual(Path(first[0]['path']).parents[1],
                             Path(changed_stack[0]['path']).parents[1])
            self.assertFalse(changed_stack[0]['skipped'])
            self.assertEqual(list(Path(directory).rglob('manifest.json')),
                             manifests_before_stack_change)
            from PIL import Image
            with Image.open(changed[0]['path']) as image:
                metadata = json.loads(image.info['generation_metadata'])
                self.assertEqual(len(metadata['stack']['loras']), 2)

    def test_manifest_folder_is_shared_when_lora_stacks_change(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = FakeRuntime()
            runner = self.runner(runtime)
            prompts = ['one', 'two', 'three', 'four']
            sizes = ((16, 24), (24, 16), (16, 16))
            first_stack = LoraStack('first', (LoraSpec('a'),))
            added_stack = LoraStack('second', (LoraSpec('b'),))

            first = runner.run(prompts, [first_stack], directory, sizes=sizes)
            manifest_paths = list(Path(directory).rglob('manifest.json'))
            shared = {(r['prompt_index']): (r['seed'], tuple(r['size'])) for r in first}

            both = runner.run(prompts, [first_stack, added_stack], directory, sizes=sizes)
            self.assertEqual(list(Path(directory).rglob('manifest.json')), manifest_paths)
            added = [r for r in both if r['stack'] == 'second']
            self.assertEqual({r['prompt_index']: (r['seed'], tuple(r['size'])) for r in added}, shared)

            remaining = runner.run(prompts, [added_stack], directory, sizes=sizes)
            self.assertEqual(list(Path(directory).rglob('manifest.json')), manifest_paths)
            self.assertTrue(all(r['skipped'] for r in remaining))

    def test_legacy_subfolders_all_loras_all_prompts_all_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = FakeRuntime()
            stacks = [
                LoraStack('subject_st4000', (LoraSpec('subject_4000', trigger='subject_x'),), 'subject_x'),
                LoraStack('subject_st5000', (LoraSpec('subject_5000', trigger='subject_x'),), 'subject_x'),
            ]
            for job_name in ('batch_01', 'batch_02'):
                output = Path(directory) / job_name
                result = self.runner(runtime).run(['portrait', 'landscape'], stacks, output,
                    sizes=((16, 16),), group_by_run=False)
                self.assertEqual(len(result), 4)
                self.assertEqual(len({r['path'] for r in result}), 4)
                for r in result:
                    self.assertEqual(Path(r['path']).parent, output / 'subject_x')
                    self.assertTrue(Path(r['path']).name.startswith(
                        f"p{r['prompt_index'] + 1:02d}_{r['stack']}_seed"))
                resumed = self.runner(runtime).run(['portrait', 'landscape'], stacks, output,
                    sizes=((16, 16),), group_by_run=False)
                self.assertTrue(all(r['skipped'] for r in resumed))
            self.assertEqual([model for model, _ in runtime.applied], ['base'] * 4)
        with self.assertRaises(ValueError):
            LoraStack('valid', output_subdir='../escape')

    def test_legacy_filename_skip_old_outputs_and_disable(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            folder = output / 'subject_x'
            folder.mkdir()
            # Legacy matching needs neither runner metadata nor a decodable PNG.
            old = folder / 'p01_subject_st4000_seed999.png'
            old.write_bytes(b'old notebook output')
            runtime = FakeRuntime()
            stacks = [LoraStack('subject_st4000', (LoraSpec('subject', trigger='subject_x'),), 'subject_x')]
            runner = self.runner(runtime)
            results = runner.run(['new prompt', 'second prompt'], stacks, output,
                sizes=((16, 16),), group_by_run=False, resume_mode='filename')
            self.assertTrue(results[0]['skipped'])
            self.assertEqual(results[0]['path'], str(old))
            self.assertEqual(results[0]['seed'], 999)
            self.assertIsNone(results[0]['size'])
            self.assertFalse(results[1]['skipped'])
            changed = runner.run(['changed prompt', 'second prompt'], stacks, output,
                sizes=((16, 16),), group_by_run=False, resume_mode='filename')
            self.assertTrue(all(r['skipped'] for r in changed))
            checkpoint = [LoraStack('subject_st5000', stacks[0].loras, 'subject_x')]
            other = runner.run(['new prompt'], checkpoint, output,
                sizes=((16, 16),), group_by_run=False, resume_mode='filename')
            self.assertFalse(other[0]['skipped'])
            regenerated = runner.run(['new prompt', 'second prompt'], stacks, output,
                sizes=((16, 16),), group_by_run=False, resume_mode='filename', skip_existing=False)
            self.assertTrue(all(not r['skipped'] for r in regenerated))
            # A fully skipped run never applies/loads the LoRA stack.
            runtime.applied.clear()
            runner.run(['new prompt', 'second prompt'], stacks, output,
                sizes=((16, 16),), group_by_run=False, resume_mode='filename')
            self.assertEqual(runtime.applied, [])

    def test_failed_batch_preserves_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = FakeRuntime()
            runner = self.runner(runtime)
            with patch.object(runtime, 'sample', side_effect=RuntimeError('interrupted')):
                with self.assertRaises(RuntimeError):
                    runner.run(['hello'], [LoraStack('base')], directory, sizes=((16, 16),))
            manifest = json.loads(next(Path(directory).rglob('manifest.json')).read_text())
            result = runner.run(['hello'], [LoraStack('base')], directory, sizes=((16, 16),))
            self.assertEqual(result[0]['seed'], manifest['jobs'][0]['seed'])

    def test_cache_refresh_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'model'
            source.write_bytes(b'abc')
            dest = cache_file(source, root / 'cache')
            source.write_bytes(b'abcd')
            self.assertEqual(cache_file(source, root / 'cache').read_bytes(), b'abcd')
            self.assertFalse(list(dest.parent.glob('*.tmp')))
        for constructor in [lambda: SamplingConfig(steps=0), lambda: SamplingConfig(denoise=0),
                            lambda: LatentUpscaleStage(float('nan')), lambda: LoraStack('../oops')]:
            with self.assertRaises(ValueError):
                constructor()

    def test_bootstrap_rejects_foreign_comfyui(self):
        from headless_comfy._bootstrap import bootstrap
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nodes.py").touch()
            (root / "folder_paths.py").touch()
            with patch('headless_comfy._bootstrap.vendored_root', return_value=root):
                with patch.dict(sys.modules, {'nodes': SimpleNamespace(__file__='/other/nodes.py')}):
                    with self.assertRaisesRegex(RuntimeError, 'Restart Python'):
                        bootstrap()

    def test_version_and_dependencies(self):
        import tomllib
        from headless_comfy import __version__
        root = Path(__file__).resolve().parents[1]
        project = tomllib.loads((root / 'pyproject.toml').read_text())
        self.assertEqual(project['project']['version'], __version__)
        spec = importlib.util.spec_from_file_location('vendor', root / 'scripts/vendor_comfyui.py')
        vendor = importlib.util.module_from_spec(spec); spec.loader.exec_module(vendor)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pyproject.toml'
            path.write_text('dependencies = [\n  "old",\n]\n')
            with self.assertRaises(RuntimeError):
                vendor.sync_dependencies(path, ['new==1'])
            vendor.sync_dependencies(path, ['new==1'], write=True)
            vendor.sync_dependencies(path, ['new==1'])


if __name__ == '__main__':
    unittest.main()
