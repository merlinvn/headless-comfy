from __future__ import annotations

import importlib
import sys
from pathlib import Path

_BOOTSTRAPPED = False


def vendored_root() -> Path:
    return Path(__file__).resolve().parent / "_vendor" / "ComfyUI"


def bootstrap() -> Path:
    """Expose the vendored ComfyUI snapshot as its original top-level modules.

    ComfyUI currently contains modules such as nodes.py and folder_paths.py that
    import each other as top-level modules. Keeping the upstream tree intact and
    adding only the vendored root to sys.path is deliberately less invasive than
    rewriting upstream imports.
    """
    global _BOOTSTRAPPED
    root = vendored_root()
    if not root.exists():
        raise RuntimeError(
            "Vendored ComfyUI snapshot is missing. "
            "For development run: python scripts/vendor_comfyui.py. "
            "Release wheels should already contain it."
        )
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    _BOOTSTRAPPED = True
    return root


def import_comfy_modules():
    bootstrap()
    folder_paths = importlib.import_module("folder_paths")
    nodes = importlib.import_module("nodes")
    model_management = importlib.import_module("comfy.model_management")
    return folder_paths, nodes, model_management
