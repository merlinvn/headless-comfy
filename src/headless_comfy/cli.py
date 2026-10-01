from __future__ import annotations

import argparse
import importlib.metadata
import platform
import sys


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def doctor() -> int:
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"headless-comfy: {_version('headless-comfy')}")
    from ._bootstrap import snapshot_info
    print(f"ComfyUI snapshot: {snapshot_info()}")
    print(f"torch: {_version('torch')}")
    print(f"transformers: {_version('transformers')}")
    print(f"safetensors: {_version('safetensors')}")

    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA runtime: {torch.version.cuda}")
    except Exception as exc:
        print(f"Torch check failed: {exc}")

    try:
        from ._bootstrap import bootstrap
        root = bootstrap()
        print(f"Vendored ComfyUI: OK ({root})")
    except Exception as exc:
        print(f"Vendored ComfyUI: FAIL ({exc})")
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="headless-comfy")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check Python, PyTorch/CUDA, and vendored ComfyUI")
    args = parser.parse_args()
    if args.command == "doctor":
        raise SystemExit(doctor())
