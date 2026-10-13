"""Run with the candidate CUDA Python. Never loads model weight tensors."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata as md
import json
import subprocess
import sys
from pathlib import Path
from common import ARCH

PACKAGES = ('torch', 'triton', 'exllamav3', 'tabbyAPI', 'transformers',
            'huggingface_hub', 'safetensors', 'tokenizers', 'flash-linear-attention')

def identity(path: Path):
    path = path.resolve()
    digest = hashlib.sha256()
    for f in sorted(path.rglob('*')):
        if not f.is_file() or f.suffix not in ('.py', '.cu', '.cuh', '.cpp', '.h', '.toml', '.jinja', '.jinja2'): continue
        if any(x in ('.git', 'venv', '.venv', '__pycache__', 'node_modules', 'build', 'dist') for x in f.relative_to(path).parts): continue
        digest.update(str(f.relative_to(path)).encode())
        digest.update(hashlib.sha256(f.read_bytes()).digest())
    def git(*args):
        r = subprocess.run(['git', '-C', str(path), *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    return {'path': str(path), 'source_digest': digest.hexdigest(),
            'git_commit': git('rev-parse', 'HEAD'),
            'git_dirty': bool(git('status', '--porcelain', '--untracked-files=no'))}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tabby-src', required=True, type=Path)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--vision', action='store_true')
    args = parser.parse_args()
    import torch
    import exllamav3
    from exllamav3.architecture.architectures import get_architectures
    registered = get_architectures()
    if ARCH not in registered:
        raise RuntimeError(f'{ARCH} is not registered. Use the actual MiniCPM-enabled runtime, not a generic Qwen fork.')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA-enabled PyTorch with a visible GPU is required')
    tabby = args.tabby_src.resolve()
    if not (tabby / 'main.py').is_file(): raise RuntimeError('TABBY_SRC lacks main.py')
    if args.model_dir:
        from exllamav3 import Config
        cfg = Config.from_directory(str(args.model_dir))
        if args.vision and 'vision' not in cfg.model_classes:
            raise RuntimeError('Vision requested but native MiniCPM vision component is absent')
    versions = {}
    for p in PACKAGES:
        try: versions[p] = md.version(p)
        except md.PackageNotFoundError: versions[p] = None
    result = {'python': sys.executable, 'python_version': sys.version.split()[0],
              'packages': versions, 'torch_cuda': torch.version.cuda,
              'gpu_name': torch.cuda.get_device_name(0),
              'gpu_capability': list(torch.cuda.get_device_capability(0)),
              'gpu_total_bytes': torch.cuda.get_device_properties(0).total_memory,
              'architecture_registered': ARCH,
              'exllamav3': identity(Path(exllamav3.__file__).parent),
              'tabbyapi': identity(tabby),
              'qualification': 'local compatibility check only; not publisher benchmark attestation'}
    print('RECIPE_PROBE_JSON=' + json.dumps(result))

if __name__ == '__main__': main()
