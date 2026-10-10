"""Explicit model downloads, independent of notebook and GPU provider.

Completed files are published atomically. Sidecars record source identity and
completion, never credentials or signed redirect URLs. File locks require a
filesystem with working OS locks (including across hosts if sharing a volume).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import warnings
from urllib.parse import parse_qs, quote, urlparse

__all__ = ["DownloadError", "cache_root", "cache_file", "aria2_download", "download_civitai",
           "download_huggingface", "resolve_model_file", "resolve_model_files"]


class DownloadError(RuntimeError):
    """Download failed; messages omit URLs and credentials."""


def cache_root() -> Path:
    """HC_CACHE_ROOT, or ~/.cache/headless_comfy; no provider detection."""
    return Path(os.environ.get("HC_CACHE_ROOT", "~/.cache/headless_comfy")).expanduser().resolve()


def _target(directory, filename):
    filename = str(filename)
    if not filename or filename in (".", "..") or any(c in str(filename) for c in "/\\\r\n"):
        raise ValueError("Destination filename must be a single filename")
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


def _sidecar(target, suffix):
    return target.with_name(target.name + suffix)


def _read(path):
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(path, data):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, suffix=".tmp") as f:
        temporary = Path(f.name)
        try:
            json.dump(data, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_options(sha256, expected_size):
    if sha256 is not None and not re.fullmatch(r"[a-fA-F0-9]{64}", sha256):
        raise ValueError("sha256 must contain 64 hexadecimal characters")
    if expected_size is not None and (not isinstance(expected_size, int) or expected_size <= 0):
        raise ValueError("expected_size must be a positive byte count")


def _valid(path, size=None, sha256=None):
    return (path.is_file() and path.stat().st_size > 0
            and (size is None or path.stat().st_size == size)
            and (sha256 is None or _hash(path) == sha256.lower()))


def cache_file(source, directory, *, sha256=None, expected_size=None):
    """Atomic local copy with source identity, size/mtime and optional SHA256."""
    from filelock import FileLock
    _validate_options(sha256, expected_size)
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    target = _target(directory, source.name)
    with FileLock(str(_sidecar(target, ".lock"))):
        stat = source.stat()
        identity = _identity([str(source), stat.st_size, stat.st_mtime_ns])
        marker = _sidecar(target, ".hc.json")
        if source == target:
            if not _valid(target, expected_size, sha256):
                raise DownloadError("Local file failed integrity validation")
            return target
        if (_read(marker).get("source") == identity
                and _valid(target, expected_size or stat.st_size, sha256)):
            return target
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
        try:
            shutil.copy2(source, temporary)
            if (source.stat().st_size, source.stat().st_mtime_ns) != (stat.st_size, stat.st_mtime_ns) or not _valid(temporary, expected_size or stat.st_size, sha256):
                raise DownloadError("Local source changed or failed integrity validation")
            temporary.replace(target)
            _write(marker, {"source": identity, "size": target.stat().st_size})
        finally:
            temporary.unlink(missing_ok=True)
        return target


def _headers(token):
    if token and any(c in token for c in "\r\n"):
        raise ValueError("Invalid token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _transfer_http(url, part, state_path, identity, headers, timeout):
    import requests
    state = _read(state_path)
    offset = part.stat().st_size if part.exists() else 0
    validator = state.get("validator")
    # Never append unvalidated bytes to an unknown or changed remote resource.
    if offset and (state.get("source") != identity or not validator):
        part.unlink()
        offset = 0
    request_headers = {**headers, "Accept-Encoding": "identity"}
    if offset:
        request_headers.update({"Range": f"bytes={offset}-", "If-Range": validator})
    # Start at the provider URL on every attempt to renew expiring signed URLs.
    with requests.get(url, headers=request_headers, stream=True, timeout=timeout) as response:
        if response.status_code == 416:
            part.unlink(missing_ok=True)
            state_path.unlink(missing_ok=True)
            raise DownloadError("Range no longer valid; restarting download")
        response.raise_for_status()
        if response.status_code not in (200, 206):
            raise DownloadError("Unexpected download response")
        content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
        if content_type in ("text/html", "application/json"):
            raise DownloadError("Provider returned a page instead of model bytes")
        if response.headers.get("Content-Encoding", "identity") != "identity":
            raise DownloadError("Encoded responses cannot be resumed safely")
        etag = response.headers.get("ETag")
        current = etag if etag and not etag.startswith("W/") else response.headers.get("Last-Modified")
        total = None
        if response.status_code == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if not match or int(match[1]) != offset or int(match[2]) != int(match[3]) - 1:
                part.unlink(missing_ok=True)
                raise DownloadError("Invalid resume response")
            if offset and current != validator:
                part.unlink(missing_ok=True)
                raise DownloadError("Remote resource changed during resume")
            total = int(match[3])
        else:
            offset = 0  # Range ignored or If-Range changed: truncate, never append.
            length = response.headers.get("Content-Length")
            if length is not None:
                total = int(length)
        _write(state_path, {"source": identity, "validator": current, "size": total})
        with part.open("ab" if offset else "wb") as f:
            for chunk in response.iter_content(1024 * 1024):
                f.write(chunk)
            f.flush()
            os.fsync(f.fileno())
        if total is not None and part.stat().st_size != total:
            raise DownloadError("Incomplete download")
        return total


def _transfer_aria2(url, part, state_path, identity, headers, timeout, connections):
    import requests
    binary = shutil.which("aria2c")
    if binary is None:
        raise DownloadError("aria2c is missing; install it or use backend='requests'")
    # Resolve anew each attempt; requests strips bearer auth on cross-host redirects.
    with requests.get(url, headers={**headers, "Accept-Encoding": "identity"},
                      stream=True, timeout=timeout) as response:
        response.raise_for_status()
        if response.status_code != 200 or response.headers.get("Content-Type", "").split(";", 1)[0].lower() in ("text/html", "application/json"):
            raise DownloadError("Provider returned an invalid model response")
        final_url = response.url
        final_headers = {k: v for k, v in response.request.headers.items()
                         if k.lower() in ("authorization", "accept-encoding")}
        etag = response.headers.get("ETag")
        validator = etag if etag and not etag.startswith("W/") else response.headers.get("Last-Modified")
        length = response.headers.get("Content-Length")
        total = int(length) if length is not None else None
    state = _read(state_path)
    if state.get("source") != identity or not validator or state.get("validator") != validator:
        part.unlink(missing_ok=True)
        _sidecar(part, ".aria2").unlink(missing_ok=True)
    _write(state_path, {"source": identity, "validator": validator, "size": total})
    if any(c in final_url + str(part.parent) for c in "\r\n"):
        raise ValueError("Invalid URL or download directory")
    # Credentials stay out of process arguments and captured diagnostic output.
    # With --input-file, aria2 requires out on the individual input entry.
    spec = (final_url + f"\n  dir={part.parent}\n  out={part.name}\n"
            + "".join(f"  header={k}: {v}\n" for k, v in final_headers.items()))
    result = subprocess.run([binary, "--input-file=-",
                             "--continue=true", "--allow-overwrite=true", "--auto-file-renaming=false",
                             "--max-tries=1", f"--connect-timeout={timeout[0] if isinstance(timeout, tuple) else timeout}",
                             f"--timeout={timeout[1] if isinstance(timeout, tuple) else timeout}", f"--max-connection-per-server={connections}",
                             f"--split={connections}", "--console-log-level=error",
                             "--download-result=hide", "--enable-color=false"],
                            input=spec, text=True, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    if result.returncode:
        raise DownloadError(f"aria2c failed (exit {result.returncode})")
    if not part.is_file():
        raise DownloadError("aria2c did not produce the requested file")
    if total is not None and part.stat().st_size != total:
        raise DownloadError("Incomplete aria2c download")
    return total


def _download(url, directory, filename, *, token=None, sha256=None, expected_size=None,
              backend="requests", connections=16, retries=3, timeout=(15, 120), force=False,
              identity=None):
    import requests
    from filelock import FileLock
    _validate_options(sha256, expected_size)
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or any(c in url for c in "\r\n"):
        raise ValueError("Expected an HTTP(S) URL")
    if backend not in ("requests", "aria2") or not 1 <= connections <= 16 or retries < 0:
        raise ValueError("Invalid backend, connections (1..16), or retries")
    headers = _headers(token)
    identity = _identity(identity if identity is not None else url)
    target = _target(directory, filename)
    marker, part, state = (_sidecar(target, suffix) for suffix in (".hc.json", ".part", ".part.json"))
    with FileLock(str(_sidecar(target, ".lock"))):
        cached = _read(marker)
        if (not force and cached.get("source") == identity
                and _valid(target, expected_size or cached.get("size"), sha256)):
            return target
        if force or _read(state).get("source") != identity:
            for path in (part, state, _sidecar(part, ".aria2")):
                path.unlink(missing_ok=True)
        for attempt in range(retries + 1):
            try:
                if backend == "requests":
                    # Backend switching cannot resume aria2's preallocated ranges.
                    if _sidecar(part, ".aria2").exists():
                        part.unlink(missing_ok=True)
                        _sidecar(part, ".aria2").unlink()
                    total = _transfer_http(url, part, state, identity, headers, timeout)
                else:
                    total = _transfer_aria2(url, part, state, identity, headers, timeout, connections)
                if not _valid(part, expected_size or total, sha256):
                    part.unlink(missing_ok=True)
                    _sidecar(part, ".aria2").unlink(missing_ok=True)
                    raise DownloadError("Downloaded file failed integrity validation")
                part.replace(target)
                _write(marker, {"source": identity, "size": target.stat().st_size})
                state.unlink(missing_ok=True)
                _sidecar(part, ".aria2").unlink(missing_ok=True)
                return target
            except (requests.RequestException, DownloadError) as exc:
                status = exc.response.status_code if isinstance(exc, requests.HTTPError) and exc.response is not None else None
                if status in (400, 401, 404) or attempt == retries:
                    reason = str(exc) if isinstance(exc, DownloadError) else (f"HTTP {status}" if status else type(exc).__name__)
                    raise DownloadError(f"Download of {target.name} failed ({reason})") from None
                time.sleep(min(2 ** attempt, 8))
    raise AssertionError("Unreachable")


def aria2_download(url, directory, filename, **kwargs):
    """Optional parallel backend; aria2c must be installed by the caller."""
    return _download(url, directory, filename, backend="aria2", **kwargs)


def download_huggingface(directory, repo_id, filename, *, revision="main", token=None,
                         sha256=None, expected_size=None, **kwargs):
    """Download public/private/gated model files; gated access must be granted.

    Output uses the basename. Pin revision to a commit for reproducible caching;
    mutable revisions are refreshed only with force=True.
    """
    if not repo_id or not filename or not revision:
        raise ValueError("repo_id, filename and revision are required")
    token = os.environ.get("HF_TOKEN") if token is None else token
    url = f"https://huggingface.co/{quote(repo_id, safe='/')}/resolve/{quote(revision, safe='')}/{quote(filename, safe='/')}"
    return _download(url, directory, Path(filename).name, token=token, sha256=sha256,
                     expected_size=expected_size, identity=["hf", repo_id, revision, filename], **kwargs)


def download_civitai(directory, filename, *, model_version_id=None, file_id=None, url=None,
                     token=None, sha256=None, expected_size=None, **kwargs):
    """Civitai API URL or version/file ID; metadata SHA256 used when available."""
    import requests
    token = os.environ.get("CIVITAI_TOKEN", os.environ.get("CIVITAI_KEY")) if token is None else token
    if url:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != "civitai.com":
            raise ValueError("Use a civitai.com HTTPS API download URL, not a signed CDN URL")
        match = re.fullmatch(r"/api/download/models/(\d+)/?", parsed.path)
        if not match:
            raise ValueError("Expected a Civitai API model download URL")
        url_version = int(match[1])
        url_file = parse_qs(parsed.query).get("fileId", [None])[0]
        if model_version_id is not None and str(model_version_id) != str(url_version):
            raise ValueError("Conflicting model version IDs")
        if file_id is not None and url_file is not None and str(file_id) != str(url_file):
            raise ValueError("Conflicting file IDs")
        model_version_id = url_version
        file_id = file_id if file_id is not None else url_file
    if model_version_id is None:
        raise ValueError("Specify model_version_id or url")
    model_version_id = int(model_version_id)
    if file_id is not None:
        file_id = int(file_id)
    if not url:
        url = f"https://civitai.com/api/download/models/{model_version_id}"
    if file_id is not None and "fileId" not in parse_qs(urlparse(url).query):
        url += ("&" if "?" in url else "?") + f"fileId={file_id}"
    # Metadata is optional for an explicit download URL/ID. Do not guess among
    # multiple files when selecting by version alone.
    try:
        with requests.get(f"https://civitai.com/api/v1/model-versions/{model_version_id}",
                          headers=_headers(token), timeout=kwargs.get("timeout", (15, 120))) as response:
            response.raise_for_status()
            files = response.json().get("files", [])
    except (requests.RequestException, ValueError):
        files = []
    selected = [f for f in files if str(f.get("id")) == str(file_id)] if file_id is not None else [f for f in files if f.get("name") == filename]
    if not selected and file_id is None and not urlparse(url).query:
        selected = [f for f in files if f.get("primary")]
    if files and len(selected) != 1:
        raise ValueError("Select an unambiguous Civitai file_id")
    if selected:
        item = selected[0]
        sha256 = sha256 or item.get("hashes", {}).get("SHA256")
        if file_id is None:
            file_id = int(item["id"])
            url += ("&" if "?" in url else "?") + f"fileId={file_id}"
    # Civitai sizeKB can be rounded; only exact caller byte counts are enforced.
    return _download(url, directory, filename, token=token, sha256=sha256,
                     expected_size=expected_size, **kwargs)


def _model_sources(cfg):
    remote = [key for key in ("civitai", "hf") if cfg.get(key) is not None]
    if len(remote) > 1 or (not remote and cfg.get("src") is None):
        raise ValueError("Specify one of 'civitai' or 'hf', optionally with 'src', or 'src' alone")
    return remote + (["src"] if cfg.get("src") is not None else [])


def resolve_model_file(cfg, *, civitai_token=None, hf_token=None):
    """Try the remote source/cache first; on DownloadError, fall back to src.

    Configuration and filesystem errors propagate. Explicit integrity constraints
    also apply to the local fallback. Its original basename is retained.
    """
    sources = _model_sources(cfg)
    directory = cfg["dst_dir"]
    options = {k: cfg[k] for k in ("backend", "connections", "retries", "timeout", "force") if k in cfg}
    if sources[0] != "src":
        try:
            if sources[0] == "civitai":
                return download_civitai(directory, token=civitai_token, **options, **cfg["civitai"])
            return download_huggingface(directory, token=hf_token, **options, **cfg["hf"])
        except DownloadError:
            if "src" not in sources:
                raise
            warnings.warn(
                f"Remote download failed; using local source {Path(cfg['src']).name}",
                RuntimeWarning, stacklevel=2,
            )
    remote = cfg[sources[0]] if sources[0] != "src" else {}
    return cache_file(
        cfg["src"], directory,
        sha256=cfg.get("sha256", remote.get("sha256")),
        expected_size=cfg.get("expected_size", remote.get("expected_size")),
    )


def resolve_model_files(configs, *, civitai_token=None, hf_token=None, workers=3):
    """Resolve a named model configuration with bounded parallelism."""
    def resolve(item):
        key, cfg = item
        return key, resolve_model_file(cfg, civitai_token=civitai_token, hf_token=hf_token)
    if workers < 1:
        raise ValueError("workers must be positive")
    targets = {}
    for cfg in configs.values():
        signature = json.dumps(cfg, sort_keys=True, default=str)
        for kind in _model_sources(cfg):
            filename = Path(cfg["src"]).name if kind == "src" else Path(cfg[kind]["filename"]).name
            target = Path(cfg["dst_dir"]).expanduser().resolve() / filename
            if target in targets and targets[target] != signature:
                raise ValueError(f"Conflicting configurations for destination {target.name}")
            targets[target] = signature
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(resolve, configs.items()))
