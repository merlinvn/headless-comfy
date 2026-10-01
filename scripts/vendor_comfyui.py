#!/usr/bin/env python3
"""Vendor a pinned snapshot and check/sync its Python requirements."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/Comfy-Org/ComfyUI.git"


def upstream_requirements(path: Path) -> list[str]:
    requirements = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-") or not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*", line):
            raise ValueError(f"Unsupported upstream requirement: {line}")
        requirements.append(line)
    return requirements


def sync_dependencies(project: Path, requirements: list[str], *, write=False) -> None:
    source = project.read_text()
    block = "dependencies = [\n" + "".join(f"  {json.dumps(r)},\n" for r in requirements) + "]"
    match = re.search(r"^dependencies = \[.*?^\]", source, flags=re.M | re.S)
    if not match:
        raise ValueError("Cannot locate project dependencies")
    current = json.loads(match.group(0).split("=", 1)[1].replace(",\n]", "\n]"))
    if current != requirements:
        if not write:
            raise RuntimeError("Dependencies differ from pinned ComfyUI. Run with --sync-dependencies and review pyproject.toml.")
        project.write_text(source[:match.start()] + block + source[match.end():])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync-dependencies", action="store_true")
    args = parser.parse_args()
    ref = (ROOT / "COMFYUI_REF").read_text().strip()
    if not re.fullmatch(r"v\d+\.\d+\.\d+(?:[-.][A-Za-z0-9.]+)?|[0-9a-fA-F]{40}", ref):
        raise ValueError("COMFYUI_REF must be a release tag or full commit SHA")
    parent = ROOT / "src" / "headless_comfy" / "_vendor"
    parent.mkdir(parents=True, exist_ok=True)
    dest = parent / "ComfyUI"
    with tempfile.TemporaryDirectory(prefix=".vendor-", dir=ROOT) as directory:
        checkout = Path(directory) / "ComfyUI"
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", REPO, str(checkout)], check=True)
        subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", ref], check=True)
        commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        requirements = upstream_requirements(checkout / "requirements.txt")
        sync_dependencies(ROOT / "pyproject.toml", requirements, write=args.sync_dependencies)
        for name in [".git", "tests", "tests-unit", ".github", "models", "output", "input", "temp"]:
            shutil.rmtree(checkout / name, ignore_errors=True)
        # Preserve existing snapshot until all fetch/validation steps succeed.
        backup = Path(directory) / "previous"
        if dest.exists():
            dest.rename(backup)
        try:
            shutil.move(str(checkout), str(dest))
            (parent / "snapshot.json").write_text(json.dumps(
                {"ref": ref, "commit": commit, "repository": REPO}, indent=2) + "\n")
        except Exception:
            shutil.rmtree(dest, ignore_errors=True)
            if backup.exists():
                backup.rename(dest)
            raise
    print(f"Vendored ComfyUI {ref} ({commit}) -> {dest}")


if __name__ == "__main__":
    main()
