"""Exercise the Colab configuration helpers without Drive or a GPU."""
import ast
import contextlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from headless_comfy import LoraSpec, Pipeline
from headless_comfy.batch import BatchRunner, LoraStack
from test_core import FakeRuntime


class NotebookTests(unittest.TestCase):
    def setUp(self):
        notebook_path = Path(__file__).resolve().parents[1] / 'notebooks/HeadlessComfyPipelines.ipynb'
        self.notebook = json.loads(notebook_path.read_text())
        self.helper_source = next(''.join(c['source']) for c in self.notebook['cells']
                                  if 'def build_lora_stack(' in ''.join(c['source']))
        self.config_source = next(''.join(c['source']) for c in self.notebook['cells']
                                  if 'BATCH_LORAS = ' in ''.join(c['source']))
        self.namespace = dict(LoraSpec=LoraSpec, LoraStack=LoraStack,
                              DRIVE_ROOT=Path('/drive'), CACHE_ROOT=Path('/cache'))
        exec(self.helper_source, self.namespace)
        self.cached, self.registered = [], []
        def cache(source, directory):
            self.cached.append((source, directory))
            return directory / source.name
        self.namespace['cache_file'] = cache
        self.namespace['assert_registered'] = lambda *args: self.registered.append(args)
        self.torch_patch = patch.dict(sys.modules, {'torch': SimpleNamespace(inference_mode=contextlib.nullcontext)})
        self.torch_patch.start()
        self.addCleanup(self.torch_patch.stop)

    def test_single_and_weighted_combinations_generate_separately(self):
        build = self.namespace['build_lora_stack']
        single = build('jun_vu_krea2_c1-st6000.safetensors')
        weighted_single = build({'name': 'solo.safetensors', 'strength': .6, 'trigger': 'solo_word'})
        combo = build({'name': 'char_a_b_c', 'loras': [
            {'name': 'a.safetensors', 'strength': .4, 'trigger': 'a_word'},
            {'name': 'b.safetensors', 'strength': .2, 'trigger': 'b_word'},
            {'name': 'c.safetensors', 'strength': .3, 'trigger': 'c_word'},
        ]})
        self.assertEqual(single.loras[0].strength, 1.0)
        self.assertEqual(single.output_subdir, 'jun_vu_x')
        self.assertEqual(weighted_single.loras[0].strength, .6)
        self.assertEqual(build({'name': 'no_strength.safetensors'}).loras[0].strength, 1.0)
        defaults = build({'name': 'mixed_defaults', 'loras': [
            'default_char.safetensors', {'name': 'weighted_char.safetensors', 'strength': .4}]})
        self.assertEqual([spec.strength for spec in defaults.loras], [1.0, .4])
        self.assertEqual(combo.output_subdir, 'char_a_b_c')
        self.assertEqual([s.strength for s in combo.loras], [.4, .2, .3])
        self.assertEqual(combo.prompt('portrait'), 'a_word, b_word, c_word, portrait')
        self.assertEqual(len(self.cached), 8)
        self.assertEqual(len(self.registered), 8)
        with tempfile.TemporaryDirectory() as directory:
            runtime = FakeRuntime()
            runner = BatchRunner(runtime, Pipeline.two_pass(), model='base', clip='clip', vae='vae', model_ids={})
            results = runner.run(['portrait', 'landscape'], [single, combo], directory,
                                 sizes=((16, 24),), group_by_run=False, resume_mode='filename')
            self.assertEqual(len(results), 4)
            self.assertEqual([m for m, _ in runtime.applied], ['base', 'base'])
            self.assertEqual([s.strength for s in runtime.applied[1][1]], [.4, .2, .3])
            combo_result = next(r for r in results if r['stack'] == 'char_a_b_c')
            self.assertEqual(Path(combo_result['path']).parent.name, 'char_a_b_c')
            from PIL import Image
            with Image.open(combo_result['path']) as image:
                metadata = json.loads(image.info['generation_metadata'])
            self.assertEqual(metadata['prompt'], 'a_word, b_word, c_word, portrait')
            self.assertEqual([s['strength'] for s in metadata['config']['stacks'][1]['loras']], [.4, .2, .3])
        with self.assertRaises(ValueError):
            build({'name': 'empty', 'loras': []})

    def test_random_aspect_flag_controls_both_seed_modes_and_resume(self):
        tree = ast.parse(self.config_source)
        fields = {'WIDTH', 'HEIGHT', 'ASPECT_RATIOS', 'SIZES'}
        dimensions_code = ast.Module(body=[node for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in fields for t in node.targets)], type_ignores=[])
        rng = SimpleNamespace(randrange=lambda stop: 123, choice=lambda sizes: sizes[-1])
        for mode in ('all_random', 'per_prompt'):
            for flag in (True, False):
                with self.subTest(seed_mode=mode, random_aspect_ratio=flag), tempfile.TemporaryDirectory() as directory:
                    namespace = {'RANDOM_ASPECT_RATIO': flag}
                    exec(compile(dimensions_code, '<notebook-dimensions>', 'exec'), namespace)
                    expected = (1024, 1024) if flag else (800, 1200)
                    runtime = FakeRuntime()
                    runner = BatchRunner(runtime, Pipeline.two_pass(), model='base', clip='clip', vae='vae', model_ids={})
                    with patch('headless_comfy.batch.random.SystemRandom', return_value=rng):
                        results = runner.run(['portrait'], [LoraStack('base')], directory,
                                             seed_mode=mode, sizes=namespace['SIZES'])
                    self.assertEqual(tuple(results[0]['size']), expected)
                    again = runner.run(['portrait'], [LoraStack('base')], directory,
                                       seed_mode=mode, sizes=namespace['SIZES'])
                    self.assertTrue(again[0]['skipped'])
                    self.assertEqual(tuple(again[0]['size']), expected)
