"""Recipe management. Runtime updates are staged; existing workloads are not killed."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import secrets
from common import ROOT, read_json, atomic_json, config_from_env, dump_yaml, verify_model
from rolling import ensure_latest, runtime_probe

STATE = Path(os.environ.get('STATE_DIR', str(ROOT / '.state'))).expanduser().resolve()
HOME = Path(os.environ.get('RECIPE_HOME', str(Path.home() / 'minicpm47-exl3'))).expanduser().resolve()
MODEL = Path(os.environ.get('MODEL_DIR', str(Path.home() / 'models/MiniCPM-V-4.7-35B-A3B-EXL3-SAGE-3.74bpw'))).expanduser().resolve()


def hold_lock(stack, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = stack.enter_context(path.open('a+'))
    os.chmod(path, 0o600)
    try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError(f'Busy lock: {path}. No process was stopped.')
    return handle


def stable_keys(tabby: Path, state: Path):
    """Preserve one private key store across rolling runtime installations."""
    target = state / 'api_tokens.yml'
    link = tabby / 'api_tokens.yml'
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError('This installation belongs to another STATE_DIR. Use a separate RECIPE_HOME; no keys were replaced.')
    elif link.exists():
        raise RuntimeError('The runtime already has its own api_tokens.yml. Refusing to replace it.')
    if not target.exists():
        atomic_json(target, {'api_key': secrets.token_hex(32), 'admin_key': secrets.token_hex(32)})
    os.chmod(target, 0o600)
    if not link.is_symlink(): link.symlink_to(target)
    return target


def serve():
    cfg = config_from_env(os.environ, MODEL)
    with ExitStack() as stack:
        hold_lock(stack, STATE / 'server.lock')
        # Separate homes for independent live services; no shared source mutation.
        hold_lock(stack, HOME / 'runtime.lock')
        if os.environ.get('GPU_LOCK_FILE'):
            gpu_lock = Path(os.environ['GPU_LOCK_FILE']).expanduser()
            if not gpu_lock.is_absolute(): raise ValueError('GPU_LOCK_FILE must be absolute')
            hold_lock(stack, gpu_lock)
        net = cfg['network']
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((net['host'], net['port']))
        if not (MODEL / 'config.json').is_file():
            raise RuntimeError('Model directory is missing. Run download.sh first or set MODEL_DIR.')
        # Every start follows the current fork HEADs. A network/update failure is
        # an error, never a silent fallback to an old runtime.
        installation = ensure_latest(HOME, STATE)
        runtime = installation['runtime']
        manifest = read_json(ROOT / 'model.lock.json')
        print('Verifying target files before model load.', flush=True)
        drafter = cfg['draft_model']['draft_mode'] != 'disabled'
        verified = verify_model(MODEL, manifest, include_drafter=drafter)
        runtime_probe(runtime['python'], runtime['tabbyapi']['path'], MODEL, cfg['model']['vision'])
        run = STATE / ('run-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + str(os.getpid()))
        run.mkdir(parents=True, exist_ok=False)
        config_path = run / 'config.yml'
        config_path.write_text(dump_yaml(cfg), encoding='utf-8')
        os.chmod(config_path, 0o600)
        receipt = {'runtime': runtime, 'runtime_update_policy': installation['policy'],
                   'resolved_runtime_sources': installation['sources'],
                   'model_verification': verified, 'config': cfg,
                   'profile': os.environ.get('PROFILE', 'single'),
                   'qualification': {'new_recipe_gpu_validation': False,
                       'context_above_small_published_tests': cfg['model']['max_seq_len'] > (4096 if cfg['model']['vision'] else 8192),
                       'api_vision_requires_validation': cfg['model']['vision'],
                       '24gb_or_companion_unvalidated': os.environ.get('PROFILE') in ('companion',)},
                   'scope': 'new local deployment; published benchmark results remain separate'}
        atomic_json(run / 'deployment.json', receipt)
        keys = stable_keys(Path(runtime['tabbyapi']['path']), STATE)
        mode = 'dSparK width %d' % cfg['draft_model']['draft_num_tokens'] if drafter else 'target-only'
        print(f'Launching {mode} at {net["host"]}:{net["port"]}. Private API keys: {keys}', flush=True)
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
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    setup = sub.add_parser('setup')
    group = setup.add_mutually_exclusive_group()
    group.add_argument('--adopt', action='store_true', help='Record an existing CUDA/MiniCPM environment without updating it')
    group.add_argument('--build', action='store_true', help='Install/update latest forks (same as default)')
    group.add_argument('--update', action='store_true', help='Install/update latest forks (same as default)')
    group.add_argument('--check', action='store_true', help='Check remote HEADs and installed runtime without updating it')
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
        print(dump_yaml(cfg)); print(f'Preview only: {path}; no GPU, network or live state touched.')
    elif args.command == 'setup':
        with ExitStack() as stack:
            hold_lock(stack, STATE / 'server.lock')
            hold_lock(stack, HOME / 'runtime.lock')
            if args.adopt:
                if not args.python or not args.tabby_src: parser.error('--adopt requires --python and --tabby-src')
                py = str(Path(args.python).expanduser().absolute())
                if not Path(py).is_file(): raise ValueError('Python executable not found')
                data = runtime_probe(py, args.tabby_src.expanduser().resolve())
                atomic_json(STATE / 'adoption.json', data)
                print('Existing environment recorded, not modified. Normal setup/serve will build latest forks separately.')
            else:
                ensure_latest(HOME, STATE, check_only=args.check)
                print('Latest fork check passed.' if args.check else 'Latest fork runtime ready. Model not loaded.')
    elif args.command == 'download':
        if args.verify_only: py = sys.executable
        elif os.environ.get('DOWNLOAD_PYTHON'): py = os.environ['DOWNLOAD_PYTHON']
        elif (HOME / 'active.json').is_file(): py = read_json(HOME / 'active.json')['runtime']['python']
        elif (STATE / 'adoption.json').is_file(): py = read_json(STATE / 'adoption.json')['python']
        else: py = sys.executable
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
