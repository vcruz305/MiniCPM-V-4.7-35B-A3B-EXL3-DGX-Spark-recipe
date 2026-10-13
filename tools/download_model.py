"""Pinned download; Hugging Face library is imported only when downloading."""
from pathlib import Path
import argparse
import json
from common import ROOT, read_json, verify_model

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True, type=Path)
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--include-drafter', action='store_true')
    args = p.parse_args()
    manifest = read_json(ROOT / 'model.lock.json')
    if not args.verify_only:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise SystemExit('huggingface_hub is missing. Use the documented isolated download venv or a qualified environment.')
        patterns = [f['path'] for f in manifest['files']
                    if args.include_drafter or not f['path'].startswith('dspark/')]
        snapshot_download(repo_id=manifest['repo_id'], revision=manifest['revision'],
                          local_dir=str(args.model_dir), allow_patterns=patterns)
    result = verify_model(args.model_dir, manifest, args.include_drafter)
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
