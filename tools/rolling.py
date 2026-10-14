"""Install the latest default branch of both forks, without version/ref pins.

The saved commits are cache keys and diagnostic receipts, not selection inputs.
No network, package install or CUDA import occurs at module import time.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid
from common import ROOT, ARCH, read_json, atomic_json


def output(args, **kwargs):
    proc = subprocess.run(args, capture_output=True, text=True, timeout=120, **kwargs)
    if proc.returncode:
        raise RuntimeError(proc.stderr[-5000:] or 'Command failed: ' + str(args[0]))
    return proc.stdout.strip()


def settings(env=None):
    env = os.environ if env is None else env
    for name in ('EXLLAMAV3_REF', 'TABBY_REF'):
        if env.get(name):
            raise ValueError(f'{name} is no longer used. Unset it; the recipe follows the latest default branches.')
    cfg = read_json(ROOT / 'runtime.json')
    urls = {
        'exllamav3': env.get('EXLLAMAV3_REPO', cfg['repositories']['exllamav3']),
        'tabbyapi': env.get('TABBY_REPO', cfg['repositories']['tabbyapi']),
    }
    for url in urls.values():
        if not re.fullmatch(r'https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?', url):
            raise ValueError('Runtime source must be an HTTPS GitHub repository without credentials or a ref suffix')
    return urls


def parse_head(text):
    branch = sha = None
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[0] == 'ref:' and fields[2] == 'HEAD':
            if fields[1].startswith('refs/heads/'):
                branch = fields[1][len('refs/heads/'):]
        if len(fields) == 2 and fields[1] == 'HEAD' and re.fullmatch(r'[a-f0-9]{40}', fields[0]):
            sha = fields[0]
    if not branch or not sha:
        raise RuntimeError('Remote did not advertise a default branch and HEAD commit')
    return {'branch': branch, 'sha': sha}


def latest_sources():
    refs = {}
    for name, url in settings().items():
        info = parse_head(output(['git', 'ls-remote', '--symref', url, 'HEAD']))
        refs[name] = {'repository': url, **info}
    return refs


def runtime_probe(python, tabby, model=None, vision=False):
    cmd = [str(python), str(ROOT / 'tools/runtime_probe.py'), '--tabby-src', str(tabby)]
    if model: cmd += ['--model-dir', str(model)]
    if vision: cmd.append('--vision')
    # First import can build the extension on a new environment.
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode:
        raise RuntimeError(proc.stderr[-5000:] or proc.stdout[-5000:] or 'Runtime probe failed')
    records = [x for x in proc.stdout.splitlines() if x.startswith('RECIPE_PROBE_JSON=')]
    if len(records) != 1:
        raise RuntimeError('Runtime probe did not return exactly one identity record')
    return json.loads(records[0].split('=', 1)[1])


def base_environment(python):
    program = '''import torch,triton,sys,json,importlib.metadata as m
from pathlib import Path
assert torch.cuda.is_available(), 'A working CUDA PyTorch environment is required'
try: triton_version = m.version('triton')
except m.PackageNotFoundError: raise RuntimeError('Install compatible Triton in the base CUDA environment first')
# pip resolves against the distribution metadata version (e.g. 2.14.1), which can lack the
# local '+cu130' suffix that torch.__version__ reports; constrain on what pip actually sees
try: torch_dist_version = m.version('torch')
except m.PackageNotFoundError: torch_dist_version = torch.__version__
site_paths=sorted(set(str(Path(mod.__file__).resolve().parents[1]) for mod in (torch,triton)))
print('BASE_JSON='+json.dumps({'python':sys.executable,'python_version':sys.version.split()[0],
    'torch':torch_dist_version,'torch_runtime':torch.__version__,'triton':triton_version,'cuda':torch.version.cuda,
    'cuda_site_packages':site_paths,
    'arch':'%d.%d'%torch.cuda.get_device_capability(0)}))'''
    text = output([str(python), '-c', program])
    rows = [x for x in text.splitlines() if x.startswith('BASE_JSON=')]
    if len(rows) != 1: raise RuntimeError('Cannot identify the base CUDA environment')
    info = json.loads(rows[0].split('=', 1)[1])
    nvcc = shutil.which('nvcc')
    cxx = os.environ.get('CXX') or shutil.which('c++')
    if not nvcc or not cxx:
        raise RuntimeError('nvcc and a C++ compiler are required; the recipe does not install system drivers/toolkits')
    info.update({'nvcc': output([nvcc, '--version']), 'compiler': output([cxx, '--version']),
                 'build_arch': os.environ.get('TORCH_CUDA_ARCH_LIST') or info['arch'],
                 'cxx': str(cxx)})
    return info


def choose_base(home: Path, state: Path):
    # Retain the venv executable spelling; resolving its symlink loses the venv.
    if os.environ.get('BASE_PYTHON'):
        return str(Path(os.environ['BASE_PYTHON']).expanduser().absolute())
    active = home / 'active.json'
    if active.is_file(): return read_json(active)['base_python']
    adopted = state / 'adoption.json'
    if adopted.is_file(): return read_json(adopted)['python']
    return sys.executable


def install_key(sources, base):
    helper_hash = hashlib.sha256()
    for name in ('rolling.py', 'runtime_probe.py'):
        helper_hash.update((ROOT / 'tools' / name).read_bytes())
    data = {'sources': sources, 'base': base, 'installer': helper_hash.hexdigest()}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def check_source_paths(sources):
    needed = (
        'exllamav3/architecture/minicpmv4_7.py',
        'exllamav3/architecture/minicpmv4_7_vision.py',
        'exllamav3/architecture/mm_processing/minicpmv4_7.py',
    )
    for name in needed:
        if not (sources / name).is_file(): raise RuntimeError(f'Latest source lacks {name}')
    if ARCH not in (sources / needed[0]).read_text():
        raise RuntimeError('Latest source lacks the native MiniCPM architecture declaration')


def verify_runtime(record):
    runtime = record['runtime']
    fresh = runtime_probe(runtime['python'], runtime['tabbyapi']['path'])
    for field in ('python_version', 'packages', 'torch_cuda'):
        if fresh[field] != runtime[field]:
            raise RuntimeError(f'Managed runtime {field} changed outside setup. Use a fresh RECIPE_HOME; nothing was overwritten.')
    for name in ('exllamav3', 'tabbyapi'):
        if fresh[name]['source_digest'] != runtime[name]['source_digest'] or fresh[name]['git_dirty']:
            raise RuntimeError(f'Managed {name} has local changes. Use a fresh RECIPE_HOME; no local work was discarded.')
        if fresh[name]['git_commit'] != record['sources'][name]['sha']:
            raise RuntimeError(f'Managed {name} source is not the recorded installation')
    return fresh


def expose_base_cuda_packages(venv_python: str, base: dict):
    # Python -m venv --system-site-packages does not inherit packages installed
    # in a *parent virtualenv*. Add the package directories containing the CUDA
    # Torch/Triton installations so managed runtime can use the user's working
    # GPU stack without reinstalling or changing it.
    sites = base.get('cuda_site_packages')
    if not isinstance(sites, list) or not sites:
        raise RuntimeError('Cannot locate base CUDA PyTorch/Triton site-packages')
    destination = Path(output([venv_python, '-c',
                               'import sysconfig; print(sysconfig.get_paths()["purelib"])']))
    destination.mkdir(parents=True, exist_ok=True)
    validated = []
    for site in sites:
        if not isinstance(site, str) or '\n' in site or '\r' in site:
            raise RuntimeError('Invalid CUDA site-packages path')
        resolved = Path(site).resolve()
        if not resolved.is_dir(): raise RuntimeError(f'Base CUDA site-packages is missing: {resolved}')
        validated.append(str(resolved))
    (destination / 'base-cuda-environment.pth').write_text('\n'.join(validated) + '\n')
    # Fail before pip if Torch and Triton are not importable from this new venv.
    output([venv_python, '-c', 'import torch,triton; assert torch.cuda.is_available()'])



def pip_check_lines(py, env):
    r = subprocess.run([py, '-m', 'pip', 'check'], env=env, capture_output=True, text=True)
    if r.returncode == 0: return []
    lines = [l.strip() for l in r.stdout.splitlines() if l.strip() and not l.startswith('No broken requirements')]
    if not lines: raise RuntimeError('pip check failed without reporting a requirement: ' + r.stderr.strip())
    return lines


def default_build_jobs():
    """Parallel nvcc jobs: one per core, capped so each job has ~3 GiB of available memory."""
    cores = os.cpu_count() or 4
    try:
        avail_kib = next(int(l.split()[1]) for l in open('/proc/meminfo') if l.startswith('MemAvailable:'))
        by_mem = max(1, avail_kib // (3 * 1024 * 1024))
    except (OSError, StopIteration, ValueError):
        by_mem = 4
    return max(1, min(cores, by_mem, 32))

def build_candidate(home, sources, base_python, base, key):
    destination = home / 'builds' / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8])
    destination.mkdir(parents=True, exist_ok=False)
    for name, ref in sources.items():
        dst = destination / name
        subprocess.run(['git', 'clone', '--depth', '1', '--single-branch', '--branch', ref['branch'],
                        ref['repository'], str(dst)], check=True)
        head = output(['git', '-C', str(dst), 'rev-parse', 'HEAD'])
        if head != ref['sha']:
            raise RuntimeError('The fork advanced during download. Rerun setup to resolve its latest HEAD; the previous installation was not changed.')
    check_source_paths(destination / 'exllamav3')
    if not (destination / 'tabbyapi/main.py').is_file(): raise RuntimeError('TabbyAPI main.py is absent')
    venv = destination / 'venv'
    subprocess.run([base_python, '-m', 'venv', '--system-site-packages', str(venv)], check=True)
    py = str(venv / 'bin/python')
    expose_base_cuda_packages(py, base)
    # Preserve the user's existing hardware-specific CUDA stack, not a fixed recipe version.
    constraints = destination / 'local-cuda-constraints.txt'
    constraints.write_text(f'torch=={base["torch"]}\ntriton=={base["triton"]}\n')
    env = os.environ.copy()
    env.setdefault('MAX_JOBS', str(default_build_jobs()))
    env['TORCH_CUDA_ARCH_LIST'] = base['build_arch']
    subprocess.run([py, '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel', 'ninja', 'packaging'], check=True, env=env)
    # PyTorch builds extensions in parallel only when a `ninja` executable is on PATH; otherwise it
    # silently falls back to a serial distutils build (~214 CUDA/C++ files one at a time, 30-45 min).
    # pip may satisfy `ninja` from the base environment, whose bin/ is not on PATH, so put both the
    # new venv's bin/ and the ninja package's own BIN_DIR first, and refuse to build without it.
    # Candidate dirs: the new venv's bin/, the ninja package's BIN_DIR (empty in some releases), and
    # the bin/ of whichever environment actually provides the ninja package (often the base CUDA venv)
    probe = ('import ninja, os, sys; d = getattr(ninja, "BIN_DIR", "") or ""; '
             'site = os.path.dirname(os.path.dirname(ninja.__file__)); '
             'print("\\n".join(x for x in (d, os.path.join(site, "..", "..", "..", "bin")) if x))')
    extra = [os.path.realpath(x) for x in output([py, '-c', probe]).splitlines() if x.strip()]
    env['PATH'] = os.pathsep.join([str(venv / 'bin')] + extra + [env.get('PATH', '')])
    found = shutil.which('ninja', path=env['PATH'])
    if not found or subprocess.run([found, '--version'], capture_output=True).returncode != 0:
        raise RuntimeError('ninja is not runnable from the build PATH; refusing a serial extension build')
    print(f"Parallel extension build: ninja at {found}, MAX_JOBS={env['MAX_JOBS']}", flush=True)
    # Problems already present in the user's base CUDA environment are reported, not fatal;
    # only dependency conflicts introduced by this install fail the build.
    pre_existing = pip_check_lines(py, env)
    subprocess.run([py, '-m', 'pip', 'install', '--upgrade', '--no-build-isolation', '--no-cache-dir',
                    '-c', str(constraints), '-e', str(destination / 'exllamav3'), str(destination / 'tabbyapi')], check=True, env=env)
    introduced = sorted(set(pip_check_lines(py, env)) - set(pre_existing))
    if pre_existing:
        print('Pre-existing base-environment pip check findings (not caused by this install):', flush=True)
        for line in pre_existing: print('  ' + line, flush=True)
    if introduced:
        raise RuntimeError('pip check found conflicts introduced by this install:\n' + '\n'.join(introduced))
    runtime = runtime_probe(py, destination / 'tabbyapi')
    if runtime['packages']['torch'] != base['torch'] or runtime['packages']['triton'] != base['triton']:
        raise RuntimeError('The base CUDA package versions changed during installation')
    record = {'policy': 'latest-default-branches', 'sources': sources, 'base_python': base_python,
              'build_environment': base, 'install_key': key, 'runtime': runtime}
    # Do not promote an installation that imports a different checkout or dirty source.
    verify_runtime(record)
    atomic_json(home / 'active.json', record)
    return record


def ensure_latest(home: Path, state: Path, check_only=False):
    """Caller holds the recipe-home lock. Always contact the remote first."""
    sources = latest_sources()
    for name, ref in sources.items():
        print(f'{name}: latest {ref["branch"]} at {ref["sha"][:12]}', flush=True)
    active_path = home / 'active.json'
    existing = read_json(active_path) if active_path.is_file() else None
    if check_only:
        if not existing: raise RuntimeError('No managed installation. Run setup.sh.')
        if existing['sources'] != sources: raise RuntimeError('Fork updates are available. Run setup.sh or start serve.sh to install them.')
        verify_runtime(existing)
        return existing
    base_python = choose_base(home, state)
    if not Path(base_python).is_file(): raise RuntimeError('BASE_PYTHON does not exist')
    base = base_environment(base_python)
    key = install_key(sources, base)
    if existing and existing.get('install_key') == key:
        verify_runtime(existing)
        print('Latest fork code already installed; no rebuild needed.', flush=True)
        return existing
    return build_candidate(home, sources, base_python, base, key)
