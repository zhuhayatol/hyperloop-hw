from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from hyperloop.backends.runtime import BackendError
from hyperloop.backends.sglang_overlay import loaded_sources, sandbox_launch, source_edit, stage_overlay
from scripts.diagnose_ascend_stock_attention import _prefill_attention, _prefill_workload


class SGLangSourceOverlayTest(unittest.TestCase):
    patch_path = "python/sglang/srt/model.py"

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.original = self.root / "pinned"
        self.workspace = self.root / "trial"
        self.workspace.mkdir()
        self.module = self.original / self.patch_path
        self.module.parent.mkdir(parents=True)
        (self.module.parents[1] / "__init__.py").touch()
        (self.module.parent / "__init__.py").touch()
        self.module.write_bytes(b"VALUE = 1\n")

    def edit(self, before="VALUE = 1", after="VALUE = 2", path=None, name="R02"):
        return {"candidate_id": name, "patch_path": path or self.patch_path,
                "source_edit": {"before": before, "after": after}}

    def test_ordered_edits_leave_the_pinned_source_untouched_and_record_an_actually_loaded_file(self):
        overlay = stage_overlay(self.original, self.workspace, [self.edit(), self.edit("VALUE = 2", "VALUE = 3")],
                                [self.patch_path], commit="0" * 40)
        command = [sys.executable, "-c", "from sglang.srt.model import VALUE; print(VALUE)"]
        wrapped, env, runtime = sandbox_launch(command, {"DEEPSEEK_API_KEY": "not-for-service"}, Path("bwrap"),
                                               self.workspace, overlay)
        self.assertNotIn("DEEPSEEK_API_KEY", env)
        self.assertEqual(wrapped[-len(command):], command)
        self.assertIn("--ro-bind", wrapped)
        self.assertEqual(wrapped[wrapped.index("--bind") + 1], str(runtime))
        log = self.workspace / "server.log"
        with log.open("wb") as output:
            subprocess.run(command, env=env, cwd=runtime, stdout=output, check=True)
        self.assertIn("\n3", log.read_text())
        self.assertEqual(loaded_sources(log, overlay)[0]["sha256"], overlay["changes"][-1]["after_sha256"])
        self.assertEqual(self.module.read_bytes(), b"VALUE = 1\n")
        self.assertIn("-VALUE = 1\n+VALUE = 3\n", overlay["unified_diff"])

    def test_patch_path_must_be_under_the_package_and_allow_listed(self):
        with self.assertRaisesRegex(BackendError, "Python file"):
            source_edit(self.edit(path="python/sglang/srt/../../pinned.py"), [self.patch_path])
        with self.assertRaisesRegex(BackendError, "outside.*allow-list"):
            source_edit(self.edit(path="python/sglang/srt/other.py"), [self.patch_path])

    def test_rejects_an_ambiguous_anchor_and_preserves_the_baseline(self):
        self.module.write_bytes(b"VALUE = 1\nVALUE = 1\n")
        with self.assertRaisesRegex(BackendError, "missing or ambiguous"):
            stage_overlay(self.original, self.workspace, [self.edit()], [self.patch_path], commit="0" * 40)
        self.assertEqual(self.module.read_bytes(), b"VALUE = 1\nVALUE = 1\n")

    def test_a_staged_file_without_activation_evidence_cannot_be_attributed_a_result(self):
        overlay = stage_overlay(self.original, self.workspace, [self.edit()], [self.patch_path], commit="0" * 40)
        log = self.workspace / "server.log"
        log.write_text("service ready\n", encoding="utf-8")
        with self.assertRaisesRegex(BackendError, "no activation evidence"):
            loaded_sources(log, overlay)

    def test_sandbox_masks_private_config_and_keeps_compile_caches_in_the_trial(self):
        home = self.root / "home"
        private = home / ".config" / "hyperloop"
        private.mkdir(parents=True)
        (private / "deepseek.key").write_text("test-value", encoding="utf-8")
        command, env, runtime = sandbox_launch(["python"], {"HOME": str(home), "TRITON_CACHE_DIR": str(private)},
                                                Path("bwrap"), self.workspace, None)
        self.assertIn("--unshare-pid", command)
        self.assertEqual(command[command.index(str(private)) - 1], "--tmpfs")
        self.assertEqual(Path(env["TRITON_CACHE_DIR"]), runtime / "cache" / "triton")


class StockAttentionDiagnosticTest(unittest.TestCase):
    def test_single_token_capture_is_separate_from_service_benchmark_workload(self):
        workload = {"id": "fixed", "requests": [{"phase": "prefill", "output_tokens": 16},
                                                 {"phase": "decode", "output_tokens": 32}]}
        capture = _prefill_workload(workload)
        self.assertEqual(capture["requests"], [{"phase": "prefill", "output_tokens": 1}])
        self.assertEqual([case["output_tokens"] for case in workload["requests"]], [16, 32])

    def test_shared_graph_decode_operator_is_not_a_prefill_switch(self):
        flash = {"phase": "prefill", "name": "UnpadFlashAttentionBF16PrefixCacheNdKernel"}
        decode = {"phase": "decode", "name": "FusedInferAttentionScore"}
        self.assertEqual(_prefill_attention({"hotspots": [flash, decode]}),
                         {"default_flash_attention": [flash], "fused_infer_attention": []})
