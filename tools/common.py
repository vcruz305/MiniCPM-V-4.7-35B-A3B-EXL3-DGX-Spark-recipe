"""Recipe helpers. No CUDA imports or downloads at module import."""
from __future__ import annotations
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'vcruz305/MiniCPM-V-4.7-35B-A3B-EXL3-SAGE-3.74bpw'
ARCH = 'MiniCPMV4_7ForConditionalGeneration'
PROFILES = {
    'single': dict(context=8192, chunk=1024, vision=False, cache='FP16'),
    # Full 262,144-token context. Q4 KV is both smaller and faster at this length than FP16 KV
    # (measured 45.0 vs 27.3 tok/s decode on one DGX Spark at 262,144 tokens).
    'long': dict(context=262144, chunk=2048, vision=False, cache='Q4'),
    'vision': dict(context=4096, chunk=512, vision=True, cache='FP16'),
    '24gb': dict(context=4096, chunk=512, vision=False, cache='FP16'),
    'companion': dict(context=4096, chunk=512, vision=False, cache='FP16'),
}
# Optional official dSparK drafter (Qwen3DSparkModel, bundled under dspark/ in the model repo).
# Exact: output is identical to target-only greedy decoding. Faster only on short prompts; its
# acceptance collapses as context grows, so it is opt-in and capped at DSPARK_MAX_CONTEXT.
DRAFT_MODES = ('disabled', 'dspark')
DSPARK_DEFAULT_TOKENS = 7
DSPARK_MAX_CONTEXT = 8192

def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))

def atomic_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    try:
        with temp.open('w', encoding='utf-8') as f:
            os.chmod(temp, 0o600)
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write('\n')
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)

def bounded_path(root: Path, rel: str) -> Path:
    if not rel or Path(rel).is_absolute() or '..' in Path(rel).parts:
        raise ValueError(f'Unsafe relative path: {rel!r}')
    target = (root / rel).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes model directory: {rel!r}')
    return target

def bool_value(text: str) -> bool:
    if text.lower() in ('1', 'true', 'yes'): return True
    if text.lower() in ('0', 'false', 'no'): return False
    raise ValueError(f'Expected true/false, got {text!r}')

# Sampler fallbacks for requests that leave a value out (force: false, so a value the client sends still
# wins). These are the model's own recommendation from generation_config.json (temperature 1.0, top_k 20,
# top_p 0.95). TabbyAPI does not read generation_config, and with no sampling section it falls back to
# temperature 1.0 with top_k 0 / top_p 1.0: every token is drawn from the full ~248K vocabulary. Agent
# frameworks such as Hermes send no sampler values, so their subagents then sample from the raw tail and
# make careless mistakes (wrong sums, malformed JSON) that the same model does not make with these values.
SAMPLER_DEFAULTS = {'temperature': 1.0, 'top_k': 20, 'top_p': 0.95, 'min_p': 0.0}

def sampling_config():
    cfg = {'override_preset': None}
    for name, value in SAMPLER_DEFAULTS.items():
        cfg[name] = {'override': value, 'force': False}
    return cfg

def config_from_env(env, model_dir: Path):
    profile = env.get('PROFILE', 'single')
    if profile not in PROFILES: raise ValueError(f'Unknown PROFILE={profile}')
    p = PROFILES[profile]
    seq = int(env.get('MAX_SEQ_LEN', p['context']))
    cache = int(env.get('CACHE_SIZE', seq))
    batch = int(env.get('MAX_BATCH_SIZE', 1))
    chunk = int(env.get('CHUNK_SIZE', p['chunk']))
    port = int(env.get('PORT', 8898))
    vision = bool_value(env.get('VISION', str(p['vision'])))
    mode = env.get('CACHE_MODE', p['cache'])
    draft = env.get('DRAFT_MODE', 'disabled')
    draft_tokens = int(env.get('DRAFT_TOKENS', DSPARK_DEFAULT_TOKENS))
    if not 256 <= seq <= 262144: raise ValueError('MAX_SEQ_LEN must be 256..262144')
    if cache < seq or cache % 256: raise ValueError('CACHE_SIZE must cover MAX_SEQ_LEN and be a multiple of 256')
    if not 1 <= batch <= 32: raise ValueError('MAX_BATCH_SIZE must be 1..32')
    if not 1 <= chunk <= seq: raise ValueError('CHUNK_SIZE must be positive and <= MAX_SEQ_LEN')
    if not 1 <= port <= 65535: raise ValueError('Invalid TCP port')
    if mode not in ('FP16', 'Q8', 'Q6', 'Q4') and not re.fullmatch(r'[2-8],[2-8]', mode):
        raise ValueError('Unsupported CACHE_MODE')
    if draft not in DRAFT_MODES:
        raise ValueError(f'DRAFT_MODE must be one of {DRAFT_MODES}; this model has no MTP or n-gram path in the recipe')
    if draft == 'dspark':
        if seq > DSPARK_MAX_CONTEXT:
            raise ValueError(f'DRAFT_MODE=dspark is limited to MAX_SEQ_LEN <= {DSPARK_MAX_CONTEXT}: the drafter is '
                             'slower than target-only decoding on long prompts. Use PROFILE=long without a drafter.')
        if vision:
            raise ValueError('DRAFT_MODE=dspark does not support image inputs; use VISION=false')
        if batch != 1:
            raise ValueError('DRAFT_MODE=dspark was measured at MAX_BATCH_SIZE=1 only')
        if not 1 <= draft_tokens <= 7:
            raise ValueError('DRAFT_TOKENS must be 1..7 (one-pass exact verification is capped at 8 rows)')
        draft_cfg = {'draft_mode': 'model', 'draft_model_dir': str(model_dir), 'draft_model_name': 'dspark',
                     'draft_cache_mode': mode, 'draft_num_tokens': draft_tokens, 'dynamic_draft': False}
    else:
        draft_cfg = {'draft_mode': 'disabled'}
    return {
        'network': {'host': env.get('HOST', '127.0.0.1'), 'port': port,
                    'disable_auth': False, 'allowed_origins': [],
                    'disable_fetch_requests': True},
        'model': {'backend': 'exllamav3', 'model_dir': str(model_dir.parent),
                  'model_name': model_dir.name, 'max_seq_len': seq,
                  'cache_size': cache, 'cache_mode': mode, 'max_batch_size': batch,
                  'chunk_size': chunk, 'vision': vision, 'tensor_parallel': False,
                  'inline_model_loading': False, 'use_dummy_models': False,
                  'prompt_template': None, 'tool_format': 'auto'},
        'draft_model': draft_cfg,
        'sampling': sampling_config(),
        'logging': {'log_prompt': False, 'log_requests': False,
                    'log_generation_params': False, 'log_chat_completion_requests': False},
        'developer': {'unsafe_launch': False},
    }

def dump_yaml(config) -> str:
    # This schema has exactly two levels; JSON scalars are valid YAML scalars.
    lines = ['# Generated by MiniCPM recipe; edit environment, not this file.']
    for section, values in config.items():
        lines.append(section + ':')
        for key, value in values.items():
            lines.append('  ' + key + ': ' + json.dumps(value, ensure_ascii=False))
    return '\n'.join(lines) + '\n'

def file_digest(path: Path, algorithm: str, git_blob=False):
    h = hashlib.new(algorithm)
    if git_blob: h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()

def verify_model(model_dir: Path, manifest, include_drafter=False):
    checked = []
    for item in manifest['files']:
        if item['path'].startswith('dspark/') and not include_drafter: continue
        p = bounded_path(model_dir, item['path'])
        if not p.is_file(): raise ValueError(f'Missing model file: {item["path"]}')
        if p.stat().st_size != item['size']: raise ValueError(f'Size mismatch: {item["path"]}')
        digest = file_digest(p, item['hash_algorithm'], item['hash_algorithm'] == 'sha1')
        if digest != item['hash']: raise ValueError(f'Hash mismatch: {item["path"]}')
        checked.append(item['path'])
    cfg = read_json(model_dir / 'config.json')
    if cfg.get('architectures') != [ARCH]: raise ValueError('Wrong model architecture; never relabel as Qwen')
    index = read_json(model_dir / 'model.safetensors.index.json')
    shards = sorted(set(index['weight_map'].values()))
    expected = sorted(i['path'] for i in manifest['files'] if re.fullmatch(r'model-\d+-of-\d+\.safetensors', i['path']))
    if shards != expected: raise ValueError('Index shard list differs from pinned target')
    if len(index['weight_map']) != manifest['target_tensors']: raise ValueError('Wrong tensor count')
    if sorted(p.name for p in model_dir.glob('*.safetensors')) != expected:
        raise ValueError('Unexpected/stale root weight shards; use a clean model folder')
    return {'revision': manifest['revision'], 'files_checked': len(checked),
            'target_weight_bytes': sum((model_dir / s).stat().st_size for s in shards),
            'drafter_included': include_drafter}
