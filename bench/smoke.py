"""Text/API smoke. Optional image route uses two in-memory solid PNGs."""
from pathlib import Path
import argparse
import base64
import json
import struct
import zlib
from client import Client, content_of

def png_data_url(rgb):
    def chunk(tag, body):
        return struct.pack('!I', len(body)) + tag + body + struct.pack('!I', zlib.crc32(tag + body) & 0xffffffff)
    size = 224
    raw = (b'\0' + bytes(rgb) * size) * size
    data = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!IIBBBBB', size, size, 8, 2, 0, 0, 0))
    data += chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b'')
    return 'data:image/png;base64,' + base64.b64encode(data).decode()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-url', default='http://127.0.0.1:8898/v1')
    p.add_argument('--model')
    p.add_argument('--vision', action='store_true')
    p.add_argument('--output', type=Path, default=Path('results/smoke.json'))
    a = p.parse_args()
    c = Client(a.base_url, a.model)
    cases = [('capital', 'What is the capital of France? Answer with the city only.', 'paris'),
             ('arithmetic', 'What is 17 times 19? Answer with the number only.', '323')]
    if a.vision:
        cases = [(name, [{'type': 'text', 'text': 'Name the dominant color in this image. One word only.'},
                        {'type': 'image_url', 'image_url': {'url': png_data_url(rgb)}}], name)
                 for name, rgb in [('red', (255, 0, 0)), ('blue', (0, 0, 255))]]
    rows = []
    for name, prompt, expected in cases:
        try:
            r = c.chat(prompt, max_tokens=128)
            text = content_of(r)
            stopped = r['choices'][0].get('finish_reason') == 'stop'
            normalized = text.strip().lower().strip('.! \n')
            ok = stopped and normalized == expected
            rows.append({'case': name, 'pass': ok, 'response': text,
                         'finish_reason': r['choices'][0].get('finish_reason')})
        except Exception as e: rows.append({'case': name, 'pass': False, 'error': str(e)})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({'scope': 'small API smoke, not a benchmark score', 'model': c.model, 'vision': a.vision, 'results': rows}, indent=2))
    print(json.dumps(rows, indent=2))
    if not all(r['pass'] for r in rows): raise SystemExit(1)

if __name__ == '__main__': main()
