from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from hyperloop.backends.runtime import BackendError
from hyperloop.kernel.ascend_forge import SWIGLU_PATH, _WORKER, _operator_edit, run_forge


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
