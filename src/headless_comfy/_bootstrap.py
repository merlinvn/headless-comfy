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
    if not (root / "nodes.py").is_file() or not (root / "folder_paths.py").is_file():
        raise RuntimeError(
            "Vendored ComfyUI snapshot is missing. "
            "For development run: python scripts/vendor_comfyui.py. "
            "Release wheels should already contain it."
        )
    for name in ("folder_paths", "nodes", "comfy", "comfy.model_management"):
        module = sys.modules.get(name)
        if module is not None:
            origin = getattr(module, "__file__", None)
            if origin is None or not Path(origin).resolve().is_relative_to(root):
                raise RuntimeError(
                    f"{name} is already imported from another runtime. "
                    "Restart Python before loading the pinned ComfyUI snapshot."
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


def snapshot_info() -> dict:
    """Build provenance; reading this does not import ComfyUI or initialize CUDA."""
    import json
    manifest = Path(__file__).resolve().parent / "_vendor" / "snapshot.json"
    return json.loads(manifest.read_text()) if manifest.exists() else {"ref": "unknown", "commit": "unknown"}
