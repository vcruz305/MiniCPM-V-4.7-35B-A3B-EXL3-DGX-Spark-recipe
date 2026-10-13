"""Non-destructive deployment management for a separate MiniCPM service."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from common import ROOT, ARCH, read_json, atomic_json, config_from_env, dump_yaml, verify_model

STATE = Path(os.environ.get('STATE_DIR', str(ROOT / '.state'))).expanduser().resolve()
MODEL = Path(os.environ.get('MODEL_DIR', str(Path.home() / 'models/MiniCPM-V-4.7-35B-A3B-EXL3-SAGE-3.74bpw'))).expanduser().resolve()

def probe(python, tabby, model=None, vision=False):
    cmd = [str(python), str(ROOT / 'tools/runtime_probe.py'), '--tabby-src', str(tabby)]
    if model: cmd += ['--model-dir', str(model)]
    if vision: cmd.append('--vision')
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if proc.returncode:
        # Errors have package/source paths but never contain API credentials.
        raise RuntimeError(proc.stderr[-5000:] or proc.stdout[-5000:] or 'Runtime probe failed')
    lines = [x for x in proc.stdout.splitlines() if x.startswith('RECIPE_PROBE_JSON=')]
    if len(lines) != 1: raise RuntimeError('Runtime probe did not return exactly one identity record')
    return json.loads(lines[0].split('=', 1)[1])

def adopted(check=True):
    file = STATE / 'adoption.json'
    if not file.is_file(): raise RuntimeError('No adopted runtime. Run setup.sh --adopt or read docs/runtime-status.md.')
    old = read_json(file)
    if check:
        new = probe(old['python'], old['tabbyapi']['path'])
        for key in ('python_version', 'packages', 'torch_cuda'):
            if old[key] != new[key]: raise RuntimeError(f'Runtime {key} changed; review and deliberately re-adopt')
        for key in ('exllamav3', 'tabbyapi'):
            if old[key]['source_digest'] != new[key]['source_digest']:
                raise RuntimeError(f'{key} source changed; review and deliberately re-adopt')
    return old

def hold_lock(stack, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = stack.enter_context(path.open('a+'))
    os.chmod(path, 0o600)
    try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError(f'Busy lock: {path}. No process was stopped.')
    return handle

def build():
    lock = read_json(ROOT / 'runtime.lock.json')['qualified']
    ex_ref = os.environ.get('EXLLAMAV3_REF') or lock['exllamav3_commit']
    tab_ref = os.environ.get('TABBY_REF') or lock['tabbyapi_commit']
    if any(not isinstance(x, str) or not re.fullmatch(r'[0-9a-f]{40}', x) for x in (ex_ref, tab_ref)):
        raise RuntimeError('Fresh build needs explicit 40-character MiniCPM-capable EXLLAMAV3_REF and TABBY_REF. Public qualified pins are not yet supplied.')
    base = os.environ.get('BASE_PYTHON')
    if not base or not Path(base).is_file(): raise RuntimeError('BASE_PYTHON must be an existing CUDA Python executable')
    info = subprocess.run([base, '-c', 'import torch,json,importlib.metadata as m; assert torch.cuda.is_available(); print(json.dumps({"torch":torch.__version__,"triton":m.version("triton"),"arch":"%d.%d"%torch.cuda.get_device_capability(0)}))'], text=True, capture_output=True, check=True)
    info = json.loads(info.stdout.strip().splitlines()[-1])
    home = Path(os.environ.get('RECIPE_HOME', str(Path.home() / 'minicpm47-exl3'))).expanduser()
    destination = home / ('build-' + ex_ref[:12] + '-' + tab_ref[:12])
    if destination.exists(): raise RuntimeError(f'Build directory already exists: {destination}. Choose a fresh RECIPE_HOME; no checkout was changed.')
    destination.mkdir(parents=True)
    for name, url, ref in [('exllamav3', os.environ.get('EXLLAMAV3_REPO', 'https://github.com/vcruz305/exllamav3.git'), ex_ref),
                           ('tabbyAPI', os.environ.get('TABBY_REPO', 'https://github.com/vcruz305/tabbyAPI.git'), tab_ref)]:
        if not re.fullmatch(r'https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?', url):
            raise RuntimeError('Source URL must be an HTTPS GitHub repository without embedded credentials')
        dst = destination / name
        subprocess.run(['git', 'init', '-q', str(dst)], check=True)
        subprocess.run(['git', '-C', str(dst), 'remote', 'add', 'origin', url], check=True)
        subprocess.run(['git', '-C', str(dst), 'fetch', '--depth', '1', 'origin', ref], check=True)
        subprocess.run(['git', '-C', str(dst), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
    src = destination / 'exllamav3'
    if not any(ARCH in f.read_text(errors='replace') for f in (src / 'exllamav3').rglob('*.py')):
        raise RuntimeError('Selected source does not mention the native MiniCPM architecture. Build stopped before pip; check the missing runtime.')
    if not shutil.which('nvcc'): raise RuntimeError('nvcc is required; no system toolkit is installed by this recipe')
    venv = destination / 'venv'
    subprocess.run([base, '-m', 'venv', '--system-site-packages', str(venv)], check=True)
    py = venv / 'bin/python'
    constraints = destination / 'cuda-constraints.txt'
    constraints.write_text(f'torch=={info["torch"]}\ntriton=={info["triton"]}\n')
    env = os.environ.copy()
    env.setdefault('MAX_JOBS', '4')
    env.setdefault('TORCH_CUDA_ARCH_LIST', info['arch'])
    subprocess.run([str(py), '-m', 'pip', 'install', 'ninja', 'packaging', 'setuptools', 'wheel'], check=True, env=env)
    subprocess.run([str(py), '-m', 'pip', 'install', '-c', str(constraints), '--no-build-isolation', str(src), str(destination / 'tabbyAPI')], check=True, env=env)
    subprocess.run([str(py), '-m', 'pip', 'check'], check=True, env=env)
    return probe(py, destination / 'tabbyAPI')

def serve():
    cfg = config_from_env(os.environ, MODEL)
    with ExitStack() as stack:
        hold_lock(stack, STATE / 'server.lock')
        if os.environ.get('GPU_LOCK_FILE'):
            gpu_lock = Path(os.environ['GPU_LOCK_FILE']).expanduser()
            if not gpu_lock.is_absolute(): raise ValueError('GPU_LOCK_FILE must be absolute')
            hold_lock(stack, gpu_lock)
        net = cfg['network']
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((net['host'], net['port']))
        runtime = adopted()
        manifest = read_json(ROOT / 'model.lock.json')
        print('Verifying pinned target files. This reads the weight shards once before launch.', flush=True)
        verified = verify_model(MODEL, manifest)
        probe(runtime['python'], runtime['tabbyapi']['path'], MODEL, cfg['model']['vision'])
        run = STATE / ('run-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + str(os.getpid()))
        run.mkdir(parents=True, exist_ok=False)
        config_path = run / 'config.yml'
        config_path.write_text(dump_yaml(cfg), encoding='utf-8')
        os.chmod(config_path, 0o600)
        receipt = {'runtime': runtime, 'model_verification': verified, 'config': cfg,
                   'profile': os.environ.get('PROFILE', 'single'),
                   'qualification': {
                       'clean_machine_gpu_validation': False,
                       'context_above_small_published_tests': cfg['model']['max_seq_len'] > (4096 if cfg['model']['vision'] else 8192),
                       'kv_dtype_is_recipe_default_not_a_published_pin': True,
                       'api_vision_requires_validation': cfg['model']['vision'],
                       '24gb_or_companion_unvalidated': os.environ.get('PROFILE') in ('24gb', 'companion')},
                   'scope': 'new local deployment, not an original publisher benchmark reproduction'}
        atomic_json(run / 'deployment.json', receipt)
        print(f'Launching target-only on {net["host"]}:{net["port"]}; auth stays enabled.', flush=True)
        child = subprocess.Popen([runtime['python'], 'main.py', '--config', str(config_path)], cwd=runtime['tabbyapi']['path'])
        previous = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, lambda *_: child.terminate() if child.poll() is None else None)
        try:
            try: result = child.wait()
            except KeyboardInterrupt:
                child.terminate()
                try: result = child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill(); result = child.wait()
        finally: signal.signal(signal.SIGTERM, previous)
        raise SystemExit(result)

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    setup = sub.add_parser('setup')
    group = setup.add_mutually_exclusive_group(required=True)
    group.add_argument('--adopt', action='store_true')
    group.add_argument('--build', action='store_true')
    group.add_argument('--check', action='store_true')
    setup.add_argument('--python')
    setup.add_argument('--tabby-src', type=Path)
    sub.add_parser('preview')
    sub.add_parser('serve')
    dl = sub.add_parser('download')
    dl.add_argument('--verify-only', action='store_true')
    dl.add_argument('--include-drafter', action='store_true')
    args = parser.parse_args()
    if args.command == 'preview':
        cfg = config_from_env(os.environ, MODEL)
        path = ROOT / 'config.preview.yml'
        path.write_text(dump_yaml(cfg), encoding='utf-8')
        print(dump_yaml(cfg)); print(f'Preview only: {path}; no GPU or live state touched.')
    elif args.command == 'setup':
        if args.check:
            adopted(); print('Adopted runtime identity and architecture check passed. Model not loaded.')
            return
        with ExitStack() as stack:
            hold_lock(stack, STATE / 'server.lock')
            if args.adopt:
                if not args.python or not args.tabby_src: parser.error('--adopt requires --python and --tabby-src')
                py = str(Path(args.python).expanduser().absolute())
                if not Path(py).is_file(): raise ValueError('Python executable not found')
                data = probe(py, args.tabby_src.expanduser().resolve())
            else: data = build()
            atomic_json(STATE / 'adoption.json', data)
            print('Runtime adopted. Private identity: ' + str(STATE / 'adoption.json'))
    elif args.command == 'download':
        if args.verify_only: py = sys.executable
        elif os.environ.get('DOWNLOAD_PYTHON'): py = os.environ['DOWNLOAD_PYTHON']
        else: py = adopted(check=False)['python']
        cmd = [py, str(ROOT / 'tools/download_model.py'), '--model-dir', str(MODEL)]
        if args.verify_only: cmd.append('--verify-only')
        if args.include_drafter: cmd.append('--include-drafter')
        subprocess.run(cmd, check=True)
    else: serve()

if __name__ == '__main__':
    try: main()
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        raise SystemExit(1)
