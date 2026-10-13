"""Measure level-wall output throughput without inventing a decode interval."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import time
import uuid
from client import Client, content_of

PROMPT = 'Explain DNS caching, TTL expiry and negative caching to a junior engineer. Include one simple example.'

def summarize(rows, wall):
    if wall <= 0: raise ValueError('Nonpositive measured wall')
    successful = [r for r in rows if r['success']]
    tokens = sum(r['completion_tokens'] for r in successful)
    return {'requests': len(rows), 'successful': len(successful),
            'output_tokens_successful': tokens, 'measured_wall_s': wall,
            'aggregate_output_tps_e2e': tokens / wall,
            'request_e2e_p50_s': statistics.median(r['e2e_s'] for r in successful) if successful else None,
            'ttft_s': None, 'prefill_tps': None, 'decode_tps': None,
            'scope': 'non-streamed level-wall measurement, includes prefill/queue/failure wall'}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-url', default='http://127.0.0.1:8898/v1')
    p.add_argument('--model')
    p.add_argument('--warmups', type=int, default=2)
    p.add_argument('--samples', type=int, default=7)
    p.add_argument('--concurrency', type=int, default=1)
    p.add_argument('--max-tokens', type=int, default=256)
    p.add_argument('--prompt-file', type=Path)
    p.add_argument('--timeout', type=float, default=180)
    p.add_argument('--output', type=Path, default=Path('results/api-benchmark.json'))
    a = p.parse_args()
    if a.warmups < 0 or a.samples < 1 or a.concurrency < 1 or a.max_tokens < 1: p.error('Invalid count')
    c = Client(a.base_url, a.model, a.timeout)
    prompt = a.prompt_file.read_text() if a.prompt_file else PROMPT
    def request(i):
        public_prompt = f'Request reference {uuid.uuid4().hex}.\n' + prompt
        start = time.perf_counter()
        try:
            r = c.chat(public_prompt, a.max_tokens)
            elapsed = time.perf_counter() - start
            usage = r.get('usage') or {}
            count = usage.get('completion_tokens')
            prompt_count = usage.get('prompt_tokens')
            if type(count) is not int or count <= 0 or type(prompt_count) is not int:
                raise ValueError('Missing trustworthy server token counts')
            output = content_of(r)
            return {'id': i, 'success': True, 'e2e_s': elapsed,
                    'prompt_tokens': prompt_count, 'completion_tokens': count,
                    'finish_reason': r['choices'][0].get('finish_reason'),
                    'response_sha256': hashlib.sha256(json.dumps(r['choices'][0]['message'], sort_keys=True).encode()).hexdigest(),
                    'prompt': public_prompt, 'output': output}
        except Exception as e:
            return {'id': i, 'success': False, 'e2e_s': time.perf_counter() - start, 'error': str(e)}
    warmups = [request(-i-1) for i in range(a.warmups)]
    if not all(r['success'] for r in warmups): raise SystemExit('Warmup failed; fix server before timing')
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        rows = list(pool.map(request, range(a.samples)))
    wall = time.perf_counter() - start
    summary = summarize(rows, wall)
    out = {'model': c.model, 'base_url': a.base_url, 'concurrency': a.concurrency,
           'max_tokens_is_a_cap': a.max_tokens, 'thinking_requested': False,
           'prompt_cache_reset': False, 'warmups': warmups,
           'summary': summary, 'requests': rows}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))
    if summary['successful'] != a.samples: raise SystemExit(1)

if __name__ == '__main__': main()
