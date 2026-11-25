from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from hyperloop.backends.runtime import BackendError
from hyperloop.backends.sglang_rope import _cases, _model_spec, rope_layout_samples, rope_layout_source_edit


_MODEL = {"model_type": "qwen2", "hidden_size": 896, "num_attention_heads": 14, "num_key_value_heads": 2}
_WORKLOAD = {"concurrency": 2, "requests": [
    {"id": "P", "phase": "prefill", "input_tokens": [3, 6, 2]},
    {"id": "D", "phase": "decode", "input_tokens": [5, 2]},
]}


def observation():
    def layout(width, row_stride, offset):
        return {"shape": [2, width], "stride": [row_stride, 1], "storage_offset": offset,
                "is_contiguous": row_stride == width, "dtype": "torch.bfloat16"}

    return {"window": "probe-7", "pid": 42, "forward_mode": "EXTEND", "tokens": 2, "device": "npu:0",
            "cache_dtype": "torch.bfloat16", "position_values": [7, 8],
            "layouts": {"packed_qkv": layout(1152, 1152, 0), "split_q": layout(896, 1152, 0),
                        "split_k": layout(128, 1152, 896), "positions": {"shape": [2], "stride": [1],
                        "storage_offset": 0, "is_contiguous": True, "dtype": "torch.int64"}}}


class RoPELayoutProbeTests(unittest.TestCase):
    def test_observation_preserves_model_calls_and_compiles(self):
        patch = rope_layout_source_edit()["source_edit"]
        self.assertEqual(patch["after"].count("q, k = self.rotary_emb(positions, q, k)"), 1)
        self.assertTrue(patch["after"].startswith(patch["before"].splitlines(keepends=True)[0]))
        compile("class Qwen2Attention:\n    def forward(self, positions, qkv, forward_batch):\n"
                + patch["after"], "qwen2.py", "exec")

    def test_observed_stride_and_provenance_do_not_leak_to_derived_shape(self):
        recorded = {**observation(), "phase": "prefill"}
        prefill, decode = _cases(_WORKLOAD, [recorded], _model_spec(_MODEL))
        self.assertEqual(prefill["layout_source"], "service_observed")
        self.assertEqual(prefill["positions"], [7, 8])
        self.assertEqual(prefill["observed_layouts"]["split_k"]["stride"], [1152, 1])
        self.assertEqual(prefill["observation_pid"], 42)
        self.assertEqual(decode["layout_source"], "derived_model_config_and_frozen_request")
        self.assertEqual(decode["positions"], [2])
        self.assertNotIn("observed_layouts", decode)

    def test_layout_capture_checks_the_owned_window_and_worker(self):
        with TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            line = "HYPERLOOP_ROPE_LAYOUT=" + json.dumps(observation()) + "\n"
            log.write_text("stale output\n" + line, encoding="utf-8")
            reports = rope_layout_samples(log, len(b"stale output\n"), "probe-7", [42])
            self.assertEqual(reports[0]["phase"], "prefill")
            self.assertEqual(reports[0]["layouts"]["split_k"]["storage_offset"], 896)
            with self.assertRaisesRegex(BackendError, "outside the owned"):
                rope_layout_samples(log, 0, "probe-7", [43])

    def test_refuses_rope_scaling_that_changes_the_stock_implementation(self):
        with self.assertRaisesRegex(BackendError, "stock unscaled Qwen2"):
            _model_spec({**_MODEL, "rope_scaling": {"factor": 2.0}})

    def test_worker_publishes_pinned_runtime_before_building_npu_rope(self):
        worker = Path(__file__).resolve().parents[1] / "src/hyperloop/backends/_sglang_rope_worker.py"
        source = Path(__file__).resolve()
        torch = Mock(bfloat16="torch.bfloat16", __version__="2.10.0")
        torch.npu.is_available.return_value = True
        torch.npu.device_count.return_value = 1
        torch.inference_mode.side_effect = nullcontext
        torch.device.side_effect = lambda _name: nullcontext()
        context = Mock()

        class ServerArgs:
            def __init__(self, *, model_path, device, dtype, tp_size, attention_backend, rl_on_policy_target):
                self.settings = (model_path, device, dtype, tp_size, attention_backend, rl_on_policy_target)

        class RotaryEmbedding:
            def __init__(self):
                self.cos_sin_cache = SimpleNamespace(dtype=torch.bfloat16, device="npu:0")
                self._forward_method = self.forward_npu

            def forward_npu(self):
                raise AssertionError("hardware-free test must not call RoPE")

        def get_rope(*_args, **kwargs):
            torch.npu.set_device.assert_called_once_with(0)
            torch.device.assert_called_once_with("npu:0")
            context.set_server_args.assert_called_once()
            self.assertEqual(context.set_server_args.call_args.args[0].settings,
                             ("dummy", "npu", "bfloat16", 1, "ascend", None))
            self.assertEqual(kwargs["dtype"], torch.bfloat16)
            return RotaryEmbedding()

        modules = {
            "torch": torch, "torch_npu": SimpleNamespace(__version__="2.10.0"),
            "sglang.srt.runtime_context": SimpleNamespace(get_context=lambda: context),
            "sglang.srt.server_args": SimpleNamespace(ServerArgs=ServerArgs),
            "sglang.srt.layers.rotary_embedding": SimpleNamespace(RotaryEmbedding=RotaryEmbedding, get_rope=get_rope),
        }
        with patch.dict(sys.modules, modules):
            run = runpy.run_path(str(worker))["_run"]
            result = {}
            with patch.dict(run.__globals__, {"version": lambda _name: "0.5.16"}):
                run({"rope_source_path": str(source), "rope_source_sha256": sha256(source.read_bytes()).hexdigest(),
                     "model_spec": _model_spec(_MODEL), "cases": []}, result)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["runtime_context"]["publisher"], "RuntimeContext.set_server_args")
