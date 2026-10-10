"""Portable download contracts, exercised against a local HTTP server."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import contextlib
import io
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from headless_comfy import downloads as d

DATA = b'model-weights-' * 200000
SHA = hashlib.sha256(DATA).hexdigest()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        server = self.server
        server.calls.append((self.path, dict(self.headers)))
        route = self.path.split('?')[0]
        if route == '/renew':
            server.redirects += 1
            self.send_response(302)
            self.send_header('Location', f'/signed?attempt={server.redirects}')
            self.end_headers()
            return
        if route == '/signed' and server.redirects == 1:
            self.send_response(403)
            self.end_headers()
            return
        if route == '/auth-redirect':
            self.send_response(302)
            self.send_header('Location', server.external)
            self.end_headers()
            return
        if route == '/404':
            self.send_response(404)
            self.end_headers()
            return
        start = int(self.headers.get('Range', 'bytes=0-').split('=')[1].split('-')[0])
        ignored = route == '/ignore' or (self.headers.get('If-Range') and self.headers['If-Range'] != server.etag)
        if ignored:
            start = 0
        self.send_response(206 if start else 200)
        self.send_header('ETag', server.etag)
        self.send_header('Content-Length', str(len(server.data) - start))
        if start:
            self.send_header('Content-Range', f'bytes {start}-{len(server.data)-1}/{len(server.data)}')
        self.end_headers()
        try:
            if route == '/interrupt' and not server.interrupted:
                server.interrupted = True
                self.wfile.write(server.data[:1500000])
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR)
            else:
                self.wfile.write(server.data[start:])
        except (BrokenPipeError, ConnectionResetError):
            pass


class DownloadsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.server.calls = []
        self.server.redirects = 0
        self.server.interrupted = False
        self.server.etag = '"v1"'
        self.server.data = DATA
        self.sleep = patch.object(d.time, 'sleep')
        self.sleep.start()
        self.addCleanup(self.sleep.stop)

    def download(self, route='/model', **options):
        return d._download(self.url + route, self.root, 'model.bin', **options)

    def test_cache_corruption_and_changed_source(self):
        target = self.download(sha256=SHA)
        self.assertEqual(self.download(sha256=SHA), target)
        self.assertEqual(len(self.server.calls), 1)
        target.write_bytes(b'bad')
        self.download(sha256=SHA)
        self.assertEqual(target.read_bytes(), DATA)
        target.write_bytes(b'x' * len(DATA))
        self.download(sha256=SHA)
        self.assertEqual(target.read_bytes(), DATA)
        self.server.data = b'new-model'
        self.download('/different')
        self.assertEqual(target.read_bytes(), b'new-model')
        marker = json.loads((self.root / 'model.bin.hc.json').read_text())
        self.assertNotIn(self.url, str(marker))

    def test_interruption_resume_across_calls(self):
        with self.assertRaises(d.DownloadError):
            self.download('/interrupt', retries=0)
        self.assertFalse((self.root / 'model.bin').exists())
        self.assertGreater((self.root / 'model.bin.part').stat().st_size, 0)
        self.download('/interrupt', sha256=SHA)
        self.assertIn('Range', self.server.calls[-1][1])
        self.assertEqual((self.root / 'model.bin').read_bytes(), DATA)
        self.assertFalse((self.root / 'model.bin.part.json').exists())

    def test_signed_url_is_resolved_again(self):
        self.download('/renew', sha256=SHA)
        self.assertEqual(self.server.redirects, 2)
        self.assertEqual([p for p, _ in self.server.calls],
                         ['/renew', '/signed?attempt=1', '/renew', '/signed?attempt=2'])

    def test_range_ignored_and_validator_changed_restart_cleanly(self):
        for route, changed in [('/ignore', False), ('/model', True)]:
            with self.subTest(route=route):
                target = self.root / 'model.bin'
                d._sidecar(target, '.hc.json').unlink(missing_ok=True)
                part = d._sidecar(target, '.part')
                part.write_bytes(b'old-content')
                d._write(d._sidecar(target, '.part.json'),
                         {'source': d._identity(self.url + route), 'validator': '"v1"'})
                self.server.etag = '"v2"' if changed else '"v1"'
                self.download(route, sha256=SHA)
                self.assertEqual(target.read_bytes(), DATA)

    def test_two_threads_and_two_processes_only_download_once(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            result = list(pool.map(lambda _: self.download(), range(2)))
        self.assertEqual(result[0], result[1])
        self.assertEqual(len(self.server.calls), 1)
        (self.root / 'model.bin.hc.json').unlink()
        script = ('from headless_comfy.downloads import _download; import sys; '
                  '_download(sys.argv[1], sys.argv[2], "model.bin")')
        processes = [subprocess.Popen([sys.executable, '-c', script, self.url + '/model', str(self.root)])
                     for _ in range(2)]
        for process in processes:
            self.assertEqual(process.wait(timeout=20), 0)
        self.assertEqual(len(self.server.calls), 2)

    def test_failed_refresh_preserves_existing_file_and_redacts_credentials(self):
        target = self.download()
        with self.assertRaises(d.DownloadError) as caught:
            self.download('/404?token=secret', token='secret', retries=0)
        self.assertNotIn('secret', str(caught.exception))
        self.assertEqual(target.read_bytes(), DATA)
        with self.assertRaises(d.DownloadError):
            self.download(sha256='a' * 64, force=True, retries=0)
        self.assertEqual(target.read_bytes(), DATA)

    def test_authorization_not_forwarded_across_hosts(self):
        other = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        other.calls, other.etag, other.data = [], '"v1"', DATA
        thread = threading.Thread(target=other.serve_forever, daemon=True)
        thread.start()
        try:
            self.server.external = f'http://localhost:{other.server_port}/model'
            self.download('/auth-redirect', token='private-token')
            self.assertEqual(self.server.calls[0][1]['Authorization'], 'Bearer private-token')
            self.assertNotIn('Authorization', other.calls[0][1])
        finally:
            other.shutdown(); other.server_close(); thread.join()

    def test_local_source_identity_and_same_file(self):
        a, b = self.root / 'a', self.root / 'b'
        a.mkdir(); b.mkdir()
        for folder, content in [(a, b'aaa'), (b, b'bbb')]:
            (folder / 'model').write_bytes(content)
            os.utime(folder / 'model', ns=(1000000000, 1000000000))
        target = d.cache_file(a / 'model', self.root / 'cache')
        self.assertEqual(d.cache_file(b / 'model', target.parent).read_bytes(), b'bbb')
        self.assertEqual(d.cache_file(target, target.parent), target)
        with self.assertRaises(d.DownloadError):
            d.cache_file(a / 'model', target.parent, sha256='0' * 64)
        self.assertEqual(target.read_bytes(), b'bbb')

    def test_provider_tokens_and_hf_revision_url(self):
        with patch.dict(os.environ, {'HF_TOKEN': 'hf-secret'}), patch.object(d, '_download') as download:
            d.download_huggingface(self.root, 'org/repo', 'weights/file.bin', revision='refs/pr/1')
            args, options = download.call_args
            self.assertIn('/resolve/refs%2Fpr%2F1/weights/file.bin', args[0])
            self.assertEqual(args[2], 'file.bin')
            self.assertEqual(options['token'], 'hf-secret')
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def raise_for_status(self): pass
            def json(self): return {'files': [{'id': 8, 'name': 'remote.bin', 'hashes': {'SHA256': SHA}}]}
        with patch.dict(os.environ, {'CIVITAI_KEY': 'c-secret', 'CIVITAI_TOKEN': 'c-secret'}), patch('requests.get', return_value=Response()), patch.object(d, '_download') as download:
            d.download_civitai(self.root, 'alias.bin', url='https://civitai.com/api/download/models/7?fileId=8')
            self.assertEqual(download.call_args.kwargs['sha256'], SHA)
            self.assertEqual(download.call_args.kwargs['token'], 'c-secret')

    @unittest.skipUnless(shutil.which('aria2c'), 'aria2c is optional; exercised in Linux CI')
    def test_real_aria2_download_and_cache(self):
        target = d.aria2_download(self.url + '/model', self.root, 'model.bin', sha256=SHA)
        self.assertEqual(target.read_bytes(), DATA)
        count = len(self.server.calls)
        d.aria2_download(self.url + '/model', self.root, 'model.bin', sha256=SHA)
        self.assertEqual(len(self.server.calls), count)

    def test_aria2_passes_credentials_via_stdin_and_renews_redirect(self):
        class Input:
            def __init__(self): self.value = ''
            def write(self, value): self.value += value
            def close(self): pass
        class Process:
            def __init__(self):
                self.stdin = Input()
                self.returncode = 0
            def poll(self): return self.returncode
            def wait(self): return self.returncode
            def kill(self): self.returncode = -1
        processes = []
        def spawn(args, **kwargs):
            self.assertNotIn('aria-secret', ' '.join(args))
            self.assertEqual(kwargs['stdout'], subprocess.DEVNULL)
            (self.root / 'model.bin.part').write_bytes(DATA)
            process = Process()
            processes.append(process)
            return process
        with patch.object(d.shutil, 'which', return_value='aria2c'), patch.object(d.subprocess, 'Popen', side_effect=spawn):
            target = d.aria2_download(self.url + '/renew', self.root, 'model.bin', token='aria-secret', sha256=SHA)
        spec = processes[0].stdin.value
        self.assertIn('Authorization: Bearer aria-secret', spec)
        self.assertIn('  out=model.bin.part\n', spec)
        self.assertIn(f'  dir={self.root.resolve()}\n', spec)
        self.assertEqual(target.read_bytes(), DATA)
        self.assertEqual(self.server.redirects, 2)
        for path in self.root.glob('*.json'):
            self.assertNotIn('aria-secret', path.read_text())
            self.assertNotIn('/signed', path.read_text())

    def test_doctor_reports_missing_cuda_without_loading_engine(self):
        from headless_comfy.cli import doctor
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {'torch': torch}), patch('headless_comfy._bootstrap.bootstrap', return_value=self.root), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(doctor(), 0)
            self.assertEqual(doctor(require_cuda=True), 1)
        with patch.dict(sys.modules, {'torch': None}), patch('headless_comfy._bootstrap.bootstrap', return_value=self.root), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(doctor(), 1)

    def test_parallel_destination_collision_rejected_before_copy(self):
        configs = {key: {'src': self.root / key / 'same.bin', 'dst_dir': self.root / 'cache'}
                   for key in ('a', 'b')}
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            d.resolve_model_files(configs)
        self.assertFalse((self.root / 'cache').exists())

    def test_remote_failure_falls_back_to_local_in_parallel_resolver(self):
        source = self.root / 'drive' / 'fallback.bin'
        source.parent.mkdir()
        source.write_bytes(DATA)
        cfg = {'hf': {'repo_id': 'org/repo', 'filename': 'remote.bin', 'sha256': SHA},
               'src': source, 'dst_dir': self.root / 'cache', 'retries': 0}
        def remote(directory, **options):
            return d._download(self.url + '/404', directory, 'remote.bin', retries=0)
        with patch.object(d, 'download_huggingface', side_effect=remote), self.assertWarnsRegex(RuntimeWarning, 'using local source'):
            target = d.resolve_model_files({'clip': cfg})['clip']
        self.assertEqual(target.name, 'fallback.bin')
        self.assertEqual(target.read_bytes(), DATA)
        self.assertFalse((target.parent / 'remote.bin').exists())
        source.write_bytes(b'wrong-weights')
        with patch.object(d, 'download_huggingface', side_effect=d.DownloadError('offline')), self.assertWarns(RuntimeWarning), self.assertRaises(d.DownloadError):
            d.resolve_model_file(cfg)
        self.assertEqual(target.read_bytes(), DATA)

    def test_remote_success_does_not_require_local_fallback(self):
        cfg = {'civitai': {'model_version_id': 1, 'filename': 'model.bin'},
               'src': self.root / 'missing.bin', 'dst_dir': self.root / 'cache'}
        def remote(directory, **options):
            return d._download(self.url + '/model', directory, 'model.bin', sha256=SHA)
        with patch.object(d, 'download_civitai', side_effect=remote), patch.object(d, 'cache_file') as local:
            target = d.resolve_model_file(cfg)
            self.assertEqual(target.read_bytes(), DATA)
            local.assert_not_called()

    def test_fallback_does_not_hide_config_errors_or_missing_sources(self):
        cfg = {'hf': {'repo_id': 'org/repo', 'filename': 'model.bin'},
               'src': self.root / 'missing.bin', 'dst_dir': self.root / 'cache'}
        with patch.object(d, 'download_huggingface', side_effect=ValueError('bad config')), patch.object(d, 'cache_file') as local:
            with self.assertRaisesRegex(ValueError, 'bad config'):
                d.resolve_model_file(cfg)
            local.assert_not_called()
        with patch.object(d, 'download_huggingface', side_effect=d.DownloadError('offline')), self.assertWarns(RuntimeWarning):
            with self.assertRaises(FileNotFoundError):
                d.resolve_model_file(cfg)
        cfg.pop('src')
        with patch.object(d, 'download_huggingface', side_effect=d.DownloadError('offline')):
            with self.assertRaises(d.DownloadError):
                d.resolve_model_file(cfg)
        cfg['civitai'] = {'model_version_id': 1, 'filename': 'model.bin'}
        with self.assertRaises(ValueError):
            d.resolve_model_file(cfg)

    def test_parallel_collision_in_fallback_destination(self):
        configs = {
            'a': {'hf': {'repo_id': 'org/repo', 'filename': 'remote.bin'},
                  'src': self.root / 'same.bin', 'dst_dir': self.root / 'cache'},
            'b': {'src': self.root / 'other' / 'same.bin', 'dst_dir': self.root / 'cache'},
        }
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            d.resolve_model_files(configs)

    def test_aria2_missing_and_config_validation(self):
        with patch.object(d.shutil, 'which', return_value=None), self.assertRaises(d.DownloadError):
            d.aria2_download(self.url + '/model', self.root, 'model.bin', retries=0)
        for cfg in ({}, {'hf': {}, 'civitai': {}, 'src': 'model', 'dst_dir': self.root}):
            with self.assertRaises((ValueError, TypeError)):
                d.resolve_model_file(cfg)
        with self.assertRaises(ValueError):
            d._download(self.url, self.root, '../model')
        with patch.dict(os.environ, {'HC_CACHE_ROOT': str(self.root)}):
            self.assertEqual(d.cache_root(), self.root.resolve())
        (self.root / 'local').write_bytes(b'weights')
        configs = {'a': {'src': self.root / 'local', 'dst_dir': self.root / 'cache'}}
        self.assertTrue(d.resolve_model_files(configs)['a'].exists())


if __name__ == '__main__':
    unittest.main()
