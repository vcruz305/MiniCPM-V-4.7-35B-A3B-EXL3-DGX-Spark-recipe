"""Small authenticated OpenAI-compatible client; no external dependencies."""
from __future__ import annotations
import json
import os
import urllib.error
import urllib.request

class Client:
    def __init__(self, base_url, model=None, timeout=120):
        self.base = base_url.rstrip('/')
        if not self.base.startswith(('http://', 'https://')): raise ValueError('Invalid base URL')
        self.timeout = timeout
        self.key = os.environ.get('TABBY_API_KEY')
        if not self.key: raise ValueError('Set TABBY_API_KEY to the inference key, not the admin key')
        models = self.request('/models')['data']
        ids = [row['id'] for row in models]
        if model:
            if model not in ids: raise ValueError('Requested model is not advertised by /v1/models')
            self.model = model
        else:
            if len(ids) != 1: raise ValueError('Specify --model because the server advertises zero or multiple models')
            self.model = ids[0]
    def request(self, path, data=None):
        body = None if data is None else json.dumps(data).encode()
        req = urllib.request.Request(self.base + path, data=body,
            headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r: return json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f'HTTP {e.code} on {path}') from e
    def chat(self, content, max_tokens=256):
        return self.request('/chat/completions', {
            'model': self.model, 'messages': [{'role': 'user', 'content': content}],
            'temperature': 0, 'seed': 1, 'max_tokens': max_tokens, 'stream': False,
            'chat_template_kwargs': {'enable_thinking': False}})

def content_of(reply):
    text = reply['choices'][0]['message'].get('content')
    if not isinstance(text, str): raise ValueError('Missing text response')
    return text
