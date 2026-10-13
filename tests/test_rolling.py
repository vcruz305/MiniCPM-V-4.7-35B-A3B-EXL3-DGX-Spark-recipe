"""Deterministic rolling-main policy tests; no GPU or network needed."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "tools"))
import rolling
import manage

A="a"*40
B="b"*40
C="c"*40
SOURCES={"exllamav3":{"repository":"https://github.com/vcruz305/exllamav3.git","branch":"master","sha":A},
         "tabbyapi":{"repository":"https://github.com/vcruz305/tabbyAPI.git","branch":"main","sha":B}}

class RollingTests(unittest.TestCase):
    def test_remote_default_branch_master(self):
        self.assertEqual(rolling.parse_head("ref: refs/heads/master\tHEAD\n"+A+"\tHEAD"),{"branch":"master","sha":A})
    def test_remote_default_branch_main(self):
        self.assertEqual(rolling.parse_head("ref: refs/heads/main HEAD\n"+B+" HEAD"),{"branch":"main","sha":B})
    def test_remote_missing_sha_fails(self):
        with self.assertRaises(RuntimeError): rolling.parse_head("ref: refs/heads/main HEAD")
    def test_remote_missing_symbolic_ref_fails(self):
        with self.assertRaises(RuntimeError): rolling.parse_head(A+" HEAD")
    def test_ref_environment_rejected(self):
        for name in ("EXLLAMAV3_REF","TABBY_REF"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    rolling.settings({name:A})
    def test_default_uses_two_forks(self):
        cfg=rolling.settings({})
        self.assertEqual(cfg["exllamav3"],"https://github.com/vcruz305/exllamav3.git")
        self.assertEqual(cfg["tabbyapi"],"https://github.com/vcruz305/tabbyAPI.git")
    def test_remote_rejects_credentials(self):
        with self.assertRaises(ValueError):
            rolling.settings({"EXLLAMAV3_REPO":"https://mytoken@github.com/owner/repo"})
    def test_cannot_choose_fixed_tag(self):
        with self.assertRaises(ValueError):
            rolling.settings({"EXLLAMAV3_REPO":"https://github.com/owner/repo.git@v1.6"})
    def test_source_commit_changes_cache_key(self):
        base={"torch":"2.1", "triton":"2.0", "build_arch":"12.1"}
        self.assertNotEqual(rolling.install_key(SOURCES,base),
          rolling.install_key({**SOURCES,"exllamav3":{**SOURCES["exllamav3"],"sha":C}},base))
    def test_gpu_environment_changes_cache_key(self):
        base={"torch":"2.1","triton":"2.0","build_arch":"12.1"}
        changed={**base,"build_arch":"8.9"}
        self.assertNotEqual(rolling.install_key(SOURCES,base),
          rolling.install_key(SOURCES,changed))
    def test_cache_key_stable(self):
        base={"torch":"2.1","triton":"2.0","build_arch":"12.1"}
        self.assertEqual(rolling.install_key(SOURCES,base),rolling.install_key(SOURCES,base))
    def test_compatible_paths_required(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(RuntimeError,"Latest source lacks"):
                rolling.check_source_paths(Path(t))
    def test_newest_build_staged(self):
        base={"torch":"2.1","triton":"2.0","build_arch":"12.1"}
        with tempfile.TemporaryDirectory() as t:
            home=Path(t)/"home";state=Path(t)/"state"
            with patch.object(rolling,"latest_sources",return_value=SOURCES), \
                 patch.object(rolling,"choose_base",return_value=sys.executable), \
                 patch.object(rolling,"base_environment",return_value=base), \
                 patch.object(rolling,"build_candidate",return_value={"created":"new"}) as build:
                self.assertEqual(rolling.ensure_latest(home,state),{"created":"new"})
                build.assert_called_once()
    def test_matching_install_reused_not_rebuilt(self):
        base={"torch":"2.1","triton":"2.0","build_arch":"12.1"}
        key=rolling.install_key(SOURCES,base)
        with tempfile.TemporaryDirectory() as t:
            home=Path(t)/"home";home.mkdir();state=Path(t)/"state"
            (home/"active.json").write_text(json.dumps({"sources":SOURCES,"install_key":key,"runtime":{"python":sys.executable}}))
            with patch.object(rolling,"latest_sources",return_value=SOURCES), \
                 patch.object(rolling,"choose_base",return_value=sys.executable), \
                 patch.object(rolling,"base_environment",return_value=base), \
                 patch.object(rolling,"verify_runtime",return_value={}) as verify, \
                 patch.object(rolling,"build_candidate") as build:
                rolling.ensure_latest(home,state)
                verify.assert_called_once();build.assert_not_called()
    def test_new_commit_stages_new_build(self):
        base={"torch":"2.1","triton":"2.0","build_arch":"12.1"}
        key=rolling.install_key(SOURCES,base)
        with tempfile.TemporaryDirectory() as t:
            home=Path(t)/"home";home.mkdir();state=Path(t)/"state"
            (home/"active.json").write_text(json.dumps({"sources":SOURCES,"install_key":key,"runtime":{}}))
            updated={**SOURCES,"exllamav3":{**SOURCES["exllamav3"],"sha":C}}
            with patch.object(rolling,"latest_sources",return_value=updated), \
                 patch.object(rolling,"choose_base",return_value=sys.executable), \
                 patch.object(rolling,"base_environment",return_value=base), \
                 patch.object(rolling,"build_candidate",return_value={"created":"fresh"}) as build:
                self.assertEqual(rolling.ensure_latest(home,state),{"created":"fresh"})
                build.assert_called_once()
    def test_check_mode_detects_remote_change_without_install(self):
        with tempfile.TemporaryDirectory() as t:
            home=Path(t)/"home";home.mkdir();state=Path(t)/"state"
            (home/"active.json").write_text(json.dumps({"sources":SOURCES}))
            updated={**SOURCES,"exllamav3":{**SOURCES["exllamav3"],"sha":C}}
            with patch.object(rolling,"latest_sources",return_value=updated), \
                 patch.object(rolling,"build_candidate") as build:
                with self.assertRaisesRegex(RuntimeError,"updates are available"):
                    rolling.ensure_latest(home,state,check_only=True)
                build.assert_not_called()
    def test_no_install_check_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as t:
            with patch.object(rolling,"latest_sources",return_value=SOURCES), \
                 patch.object(rolling,"build_candidate") as build:
                with self.assertRaisesRegex(RuntimeError,"No managed installation"):
                    rolling.ensure_latest(Path(t)/"home",Path(t)/"state",check_only=True)
                build.assert_not_called()
    def test_atomic_promotion_only_after_validation(self):
        with tempfile.TemporaryDirectory() as t:
            home=Path(t)/"home";home.mkdir();state=Path(t)/"state"
            data={"install_key":"old","sources":SOURCES}
            (home/"active.json").write_text(json.dumps(data))
            updated={**SOURCES,"exllamav3":{**SOURCES["exllamav3"],"sha":C}}
            with patch.object(rolling,"latest_sources",return_value=updated), \
                 patch.object(rolling,"choose_base",return_value=sys.executable), \
                 patch.object(rolling,"base_environment",return_value={"torch":"2.1"}), \
                 patch.object(rolling,"build_candidate",side_effect=RuntimeError("compile failed")):
                with self.assertRaisesRegex(RuntimeError,"compile failed"):
                    rolling.ensure_latest(home,state)
            self.assertEqual(json.loads((home/"active.json").read_text()),data)
    def test_base_cuda_virtualenv_packages_are_visible(self):
        # Simulate a CUDA environment in a separate site directory. The
        # managed venv must inherit it even if BASE_PYTHON was itself a venv.
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); site=root/'base-site';site.mkdir()
            (site/'torch.py').write_text('class cuda:\n    @staticmethod\n    def is_available(): return True\n')
            (site/'triton.py').write_text('\n')
            venv=root/'managed'
            subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(venv)],check=True,
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            py=str(venv/'bin/python')
            rolling.expose_base_cuda_packages(py,{'cuda_site_packages':[str(site)]})
            found=subprocess.check_output([py,'-c','import torch,triton;print(torch.__file__)'],text=True).strip()
            self.assertEqual(Path(found),site/'torch.py')
    def test_base_cuda_site_missing_fails(self):
        with self.assertRaisesRegex(RuntimeError,'Cannot locate'):
            rolling.expose_base_cuda_packages(sys.executable,{'cuda_site_packages':[]})
    def test_keys_stable_across_builds(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); state=root/"state";state.mkdir()
            build1=root/"build1";build2=root/"build2";build1.mkdir();build2.mkdir()
            a=manage.stable_keys(build1,state)
            first=a.read_bytes()
            self.assertEqual(first,manage.stable_keys(build2,state).read_bytes())
            self.assertEqual((build1/"api_tokens.yml").resolve(),(build2/"api_tokens.yml").resolve())
            self.assertEqual(a.stat().st_mode&0o777,0o600)
            saved=json.loads(first)
            self.assertNotEqual(saved["api_key"],saved["admin_key"])
    def test_keys_conflict_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);state=root/"state";state.mkdir();src=root/"source";src.mkdir()
            (src/"api_tokens.yml").write_text("do-not-overwrite")
            with self.assertRaisesRegex(RuntimeError,"already has"):
                manage.stable_keys(src,state)
            self.assertEqual((src/"api_tokens.yml").read_text(),"do-not-overwrite")

if __name__=="__main__":unittest.main()
