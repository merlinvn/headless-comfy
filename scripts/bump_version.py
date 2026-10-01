#!/usr/bin/env python3
"""Update package, Python API and example Colab installer version together."""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('version', help='Release version, e.g. 0.2.1')
    args = parser.parse_args()
    if not re.fullmatch(r'\d+\.\d+\.\d+', args.version):
        parser.error('Use a stable major.minor.patch version')
    replacements = [
        (ROOT / 'pyproject.toml', r'^version = "[^"]+"', f'version = "{args.version}"'),
        (ROOT / 'src/headless_comfy/__init__.py', r'^__version__ = "[^"]+"', f'__version__ = "{args.version}"'),
    ]
    pending = []
    for path, pattern, replacement in replacements:
        text, count = re.subn(pattern, replacement, path.read_text(), flags=re.M)
        if count != 1:
            raise ValueError(f'Expected one version field in {path}')
        pending.append((path, text))
    path = ROOT / 'notebooks/HeadlessComfyPipelines.ipynb'
    notebook = json.loads(path.read_text())
    old_version = re.search(r'^version = "([^"]+)"', (ROOT / 'pyproject.toml').read_text(), re.M)[1]
    for cell in notebook['cells']:
        cell['source'] = [line.replace(old_version, args.version) for line in cell['source']]
    pending.append((path, json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'))
    for path, text in pending:
        path.write_text(text)
    print(f'Updated to {args.version}; review diff, test and rebuild before release.')


if __name__ == '__main__':
    main()
