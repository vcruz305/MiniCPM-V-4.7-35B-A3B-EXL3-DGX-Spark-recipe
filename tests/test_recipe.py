from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'bench')]
from common import ARCH, PROFILES, bounded_path, config_from_env, dump_yaml, file_digest, read_json, atomic_json, verify_model
from api_benchmark import summarize
from smoke import png_data_url

class ConfigTests(unittest.TestCase):
    def cfg(self, **env): return config_from_env(env, Path('/models/mini'))
    def test_default_target_only(self): self.assertEqual(self.cfg()['draft_model']['draft_mode'], 'disabled')
    def test_default_memory(self): self.assertEqual(self.cfg()['model']['cache_size'], 8192)
    def test_default_no_vision(self): self.assertFalse(self.cfg()['model']['vision'])
    def test_auth_enabled(self): self.assertFalse(self.cfg()['network']['disable_auth'])
    def test_no_open_cors(self): self.assertEqual(self.cfg()['network']['allowed_origins'], [])
    def test_all_profiles(self):
        for name, spec in PROFILES.items():
            with self.subTest(name=name):
                c=self.cfg(PROFILE=name); self.assertEqual(c['model']['max_seq_len'],spec['context'])
    def test_vision_profile(self): self.assertTrue(self.cfg(PROFILE='vision')['model']['vision'])
    def test_bad_profile(self):
        with self.assertRaises(ValueError): self.cfg(PROFILE='qwen')
    def test_context_above_source(self):
        with self.assertRaises(ValueError): self.cfg(MAX_SEQ_LEN='1048576')
    def test_unaligned_pool(self):
        with self.assertRaises(ValueError): self.cfg(CACHE_SIZE='8193')
    def test_pool_too_small(self):
        with self.assertRaises(ValueError): self.cfg(CACHE_SIZE='4096')
    def test_no_fake_mtp(self):
        with self.assertRaises(ValueError): self.cfg(DRAFT_MODE='mtp')
    def test_invalid_bool(self):
        with self.assertRaises(ValueError): self.cfg(VISION='perhaps')
    def test_invalid_cache(self):
        with self.assertRaises(ValueError): self.cfg(CACHE_MODE='nvfp4')
    def test_invalid_port(self):
        with self.assertRaises(ValueError): self.cfg(PORT='70000')
    def test_invalid_batch(self):
        with self.assertRaises(ValueError): self.cfg(MAX_BATCH_SIZE='0')
    def test_original_template(self): self.assertIsNone(self.cfg()['model']['prompt_template'])
    def test_yaml_scalar_paths(self):
        t=dump_yaml(config_from_env({},Path('/models/with space/model')))
        self.assertIn('model_dir: "/models/with space"',t)
    def test_preview_no_runtime(self):
        with tempfile.TemporaryDirectory() as td:
            env=dict(os.environ, STATE_DIR=td, MODEL_DIR='/missing/model')
            r=subprocess.run([sys.executable,str(ROOT/'tools/manage.py'),'preview'],env=env,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertFalse((Path(td)/'adoption.json').exists())
    def test_setup_refuses_missing_pins(self):
        with tempfile.TemporaryDirectory() as td:
            env=dict(os.environ, STATE_DIR=td)
            for key in ('EXLLAMAV3_REF','TABBY_REF'):env.pop(key,None)
            r=subprocess.run([sys.executable,str(ROOT/'tools/manage.py'),'setup','--build'],env=env,capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0);self.assertIn('Public qualified pins',r.stderr)

class IntegrityTests(unittest.TestCase):
    def test_parent_escape(self):
        with self.assertRaises(ValueError):bounded_path(Path('/tmp/model'),'../private')
    def test_absolute_escape(self):
        with self.assertRaises(ValueError):bounded_path(Path('/tmp/model'),'/etc/passwd')
    def test_symlink_escape(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'outside').symlink_to('/tmp')
            with self.assertRaises(ValueError):bounded_path(root,'outside/secret')
    def test_git_blob_hash(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x';p.write_bytes(b'hello')
            self.assertEqual(file_digest(p,'sha1',True),hashlib.sha1(b'blob 5\0hello').hexdigest())
    def test_sha256(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x';p.write_bytes(b'hello')
            self.assertEqual(file_digest(p,'sha256'),hashlib.sha256(b'hello').hexdigest())
    def test_manifest_totals(self):
        d=read_json(ROOT/'model.lock.json');w=[x for x in d['files'] if re.fullmatch(r'model-\d+-of-\d+\.safetensors',x['path'])]
        self.assertEqual(len(w),19);self.assertEqual(sum(x['size'] for x in w),17999304996)
    def test_manifest_checksums(self):
        for x in read_json(ROOT/'model.lock.json')['files']:
            self.assertRegex(x['hash'],r'^[a-f0-9]{'+('64' if x['hash_algorithm']=='sha256' else '40')+r'}$')
    def test_drafter_scope(self):
        d=read_json(ROOT/'model.lock.json');w=next(x for x in d['files'] if x['path']=='dspark/model.safetensors')
        self.assertEqual(w['size'],778452858);self.assertEqual(w['hash'],'211eda7b962795ba727b0d7db072d8c474dc3b94075f85f123fe363caf5e9679')
    def test_model_revision_immutable(self): self.assertRegex(read_json(ROOT/'model.lock.json')['revision'],r'^[a-f0-9]{40}$')
    def test_runtime_gap_explicit(self): self.assertIsNone(read_json(ROOT/'runtime.lock.json')['qualified']['exllamav3_commit'])
    def test_atomic_private_record(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'record.json';atomic_json(p,{'ok':True})
            self.assertTrue(read_json(p)['ok']);self.assertEqual(p.stat().st_mode&0o777,0o600)
    def test_missing_model_fails(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError): verify_model(Path(td),read_json(ROOT/'model.lock.json'))
    def test_card_snapshot(self):
        t=(ROOT/'evidence/MODEL_CARD.md').read_text();self.assertIn('69.70',t);self.assertIn('0.031476575675059786',t)
    def test_results_not_new_measurement(self): self.assertEqual(read_json(ROOT/'bench/published-results.json')['provenance'],'publisher_model_card_transcription')
    def test_matching_dspark_pair(self):
        r=read_json(ROOT/'bench/published-results.json')['dspark_matched']
        self.assertEqual(r['target_only_decode_p50'],69.33);self.assertEqual(r['draft_decode_p50'],45.31)

class PackVerificationTests(unittest.TestCase):
    def make_pack(self, root):
        data={'config.json': json.dumps({'architectures':[ARCH]}).encode(),
              'model.safetensors.index.json': json.dumps({'weight_map':{'w':'model-00001-of-00001.safetensors'}}).encode(),
              'model-00001-of-00001.safetensors': b'mocked weight bytes'}
        manifest={'revision':'a'*40,'target_tensors':1,'files':[]}
        for name,content in data.items():
            (root/name).write_bytes(content)
            manifest['files'].append({'path':name,'size':len(content),'hash_algorithm':'sha256','hash':hashlib.sha256(content).hexdigest()})
        return manifest
    def test_successful_pack_verification(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); m=self.make_pack(root)
            self.assertEqual(verify_model(root,m)['files_checked'],3)
    def test_same_size_corruption_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); m=self.make_pack(root)
            f=root/'model-00001-of-00001.safetensors';f.write_bytes(b'x'*f.stat().st_size)
            with self.assertRaisesRegex(ValueError,'Hash mismatch'):verify_model(root,m)
    def test_wrong_tensor_count_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); m=self.make_pack(root);m['target_tensors']=2
            with self.assertRaisesRegex(ValueError,'tensor count'):verify_model(root,m)
    def test_stale_shard_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); m=self.make_pack(root);(root/'old.safetensors').write_bytes(b'old')
            with self.assertRaisesRegex(ValueError,'stale'):verify_model(root,m)
    def test_drafter_not_required_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); m=self.make_pack(root)
            m['files'].append({'path':'dspark/model.safetensors','size':3,'hash_algorithm':'sha256','hash':'a'*64})
            self.assertEqual(verify_model(root,m)['files_checked'],3)
            with self.assertRaisesRegex(ValueError,'Missing model file'):verify_model(root,m,True)
    def test_source_card_matches_manifest(self):
        manifest=read_json(ROOT/'model.lock.json')
        item=next(x for x in manifest['files'] if x['path']=='README.md')
        self.assertEqual(file_digest(ROOT/'evidence/MODEL_CARD.md','sha1',True),item['hash'])
    def test_source_config_matches_manifest(self):
        manifest=read_json(ROOT/'model.lock.json')
        item=next(x for x in manifest['files'] if x['path']=='config.json')
        self.assertEqual(file_digest(ROOT/'evidence/model-config.json','sha1',True),item['hash'])

class BenchmarkTests(unittest.TestCase):
    def test_aggregate_includes_failure_wall(self):
        r=summarize([{'success':True,'completion_tokens':100,'e2e_s':2},{'success':False}],10)
        self.assertEqual(r['aggregate_output_tps_e2e'],10)
    def test_no_fabricated_decode(self):
        r=summarize([],1);self.assertIsNone(r['decode_tps']);self.assertIsNone(r['ttft_s'])
    def test_zero_wall_rejected(self):
        with self.assertRaises(ValueError):summarize([],0)
    def test_color_data_url(self):
        a=png_data_url((255,0,0));b=png_data_url((0,0,255))
        self.assertTrue(a.startswith('data:image/png;base64,'));self.assertNotEqual(a,b)

class DocsTests(unittest.TestCase):
    def test_local_markdown_links(self):
        for f in ROOT.rglob('*.md'):
            if '.state' in f.parts or f.name=='MODEL_CARD.md':continue
            for dest in re.findall(r'\]\(([^)]+)\)',f.read_text()):
                if dest.startswith(('http:','https:','#','mailto:')):continue
                self.assertTrue((f.parent/dest.split('#')[0]).exists(),f'{f}: broken {dest}')
    def test_shells_parse(self):
        for f in (ROOT/'exllamav3-tabby').glob('*.sh'):
            self.assertEqual(subprocess.run(['bash','-n',str(f)]).returncode,0)
    def test_no_dangerous_process_commands(self):
        for f in (ROOT/'exllamav3-tabby').glob('*.sh'):
            t=f.read_text();self.assertNotIn('pkill',t);self.assertNotIn('killall',t)

if __name__=='__main__':unittest.main()
