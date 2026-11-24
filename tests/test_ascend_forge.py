import json
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from hyperloop.backends.runtime import BackendError
from hyperloop.kernel.ascend_forge import SWIGLU_PATH, _WORKER, _operator_edit, _run_worker, run_forge


_ACTIVATION = b'''\
import torch
_is_npu = False

class SiluAndMul(MultiPlatformOp):
    def forward_npu(self, x: torch.Tensor) -> torch.Tensor:
        out = torch_npu.npu_swiglu(x)
        return out
'''
_HOTSPOT = {"callsite": "sglang/srt/layers/activation.py:6", "name": "SwiGlu", "phase": "decode"}


class AscendForgeTests(unittest.TestCase):
    def test_overlay_uses_benchmarked_kernel_and_preserves_stock_fallback(self):
        worker = _WORKER.read_text(encoding="utf-8")
        candidate = _operator_edit(_ACTIVATION, worker, 4864, [_HOTSPOT])
        edit = candidate["source_edit"]
        adapted = _ACTIVATION.decode().replace(edit["before"], edit["after"])
        compile(adapted, SWIGLU_PATH, "exec")
        self.assertIn("def _hyperloop_forge_swiglu_launch(packed):", edit["after"])
        self.assertIn("x.shape[1] == 9728", edit["after"])
        self.assertIn("out = torch_npu.npu_swiglu(x)", edit["after"])
        self.assertEqual(len(candidate["operator_source_sha256"]), 64)
        self.assertIn(candidate["invocation_marker"], edit["after"])
        with self.assertRaisesRegex(BackendError, "already contains"):
            _operator_edit(adapted.encode(), worker, 4864, [_HOTSPOT])

    def test_no_worker_launch_with_same_name_but_different_source_location(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / SWIGLU_PATH
            original.parent.mkdir(parents=True)
            original.write_bytes(_ACTIVATION)
            model = root / "config.json"
            model.write_text('{"model_type":"qwen2","intermediate_size":4864}', encoding="utf-8")
            workspace = root / "forge"
            outcome = run_forge(
                python=root / "missing-executable", model_config=model, source_root=root,
                hotspots=[{**_HOTSPOT, "callsite": "sglang/srt/layers/activation.py:60"}], workspace=workspace,
                environment={}, uid=61000, gid=1000,
            )
            self.assertEqual(outcome["status"], "blocked")
            self.assertEqual(outcome["reason"], "no_source_correlated_forge_swiglu_hotspot")
            self.assertFalse(workspace.exists())

    def test_worker_has_a_durable_nonce_lease_before_running(self):
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            lease_path = workspace / "sglang-lease.json"
            process = Mock(pid=7331, returncode=0)
            process.poll.return_value = 0

            def start(*_args, env, **_kwargs):
                staged = json.loads(lease_path.read_text(encoding="utf-8"))
                self.assertEqual(staged["state"], "launching")
                self.assertEqual(staged["run_uid"], 61000)
                self.assertIsNone(staged["pid"])
                self.assertEqual(env["HYPERLOOP_SGLANG_LEASE"], staged["nonce"])
                return process

            def wait(*_args, **_kwargs):
                running = json.loads(lease_path.read_text(encoding="utf-8"))
                self.assertEqual(running["state"], "running")
                self.assertEqual(running["pid"], 7331)
                self.assertEqual(running["start_ticks"], 457)
                return 0

            def write_lease(directory, value):
                self.assertEqual(directory, workspace)
                lease_path.write_text(json.dumps(value), encoding="utf-8")

            process.wait.side_effect = wait
            with (patch("hyperloop.kernel.ascend_forge.subprocess.Popen", side_effect=start),
                  patch("hyperloop.backends.sglang._write_lease", side_effect=write_lease),
                  patch("hyperloop.backends.sglang._process", return_value=SimpleNamespace(start_ticks=457)),
                  patch("hyperloop.backends.sglang._lease_processes", return_value={}),
                  patch("hyperloop.backends.sglang._live_group", return_value=[]),
                  patch.object(signal, "SIGKILL", 9, create=True)):
                result = _run_worker(["python"], {}, workspace, workspace, 61000, 5)
            self.assertEqual(result["remaining_pids"], [])
            self.assertTrue(result["process_group_clear"])
            self.assertEqual(json.loads(lease_path.read_text(encoding="utf-8"))["state"], "released")
