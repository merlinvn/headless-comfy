#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = (ROOT / "COMFYUI_REF").read_text().strip()
DEST = ROOT / "src" / "headless_comfy" / "_vendor" / "ComfyUI"
TMP = ROOT / ".vendor-tmp"
REPO = "https://github.com/Comfy-Org/ComfyUI.git"


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def main() -> None:
    shutil.rmtree(TMP, ignore_errors=True)
    shutil.rmtree(DEST, ignore_errors=True)
    run("git", "clone", "--filter=blob:none", "--no-checkout", REPO, str(TMP))
    run("git", "-C", str(TMP), "checkout", REF)
    shutil.rmtree(TMP / ".git", ignore_errors=True)

    # Keep upstream tree intact because ComfyUI uses top-level imports.
    # Remove only obvious development/runtime bulk that headless execution does not need.
    for name in ["tests", "tests-unit", ".github", "models", "output", "input", "temp"]:
        shutil.rmtree(TMP / name, ignore_errors=True)

    DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(TMP), str(DEST))
    print(f"Vendored ComfyUI {REF} -> {DEST}")


if __name__ == "__main__":
    main()
