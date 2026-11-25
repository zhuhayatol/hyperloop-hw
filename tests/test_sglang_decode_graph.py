"""Check the opt-in decode graph parameter at the actual launch boundary."""

import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from hyperloop.backends.runtime import BackendError
from hyperloop.backends.sglang import SGLangBackend, _decode_graph_args


_BASE = ["--skip-server-warmup", "--disable-cuda-graph", "--cuda-graph-max-bs-decode", "32",
         "--chunked-prefill-size", "512"]
_GRAPH_FLAGS = {"disable_cuda_graph": False, "cuda_graph_backend_decode": "full",
                "cuda_graph_backend_prefill": "disabled"}
_CAPTURE = (b"[2026-11-25 00:15:53] Capture target decode NPU graph begin. backend=full, bs=[1, 2]\n"
            b"[2026-11-25 00:15:57] Capture target decode NPU graph end. elapsed=3.72 s\n")


class DecodeGraphTests(unittest.TestCase):
    def setUp(self):
        work = TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.root = Path(work.name)
        backend = object.__new__(SGLangBackend)
        backend._stacks = {}
        backend._server_args = list(_BASE)
        backend._ascend_use_fia = None
        backend._child = None
        backend._probe = {"environment": {"rank_map": {"0": "0"}, "software": {"sglang": "0.5.16"}},
                          "environment_fingerprint": "same-environment"}
        backend._source_sandbox = None
        backend._trial_user = None
        backend._python = Path("python")
        backend._model = Path("model")
        backend._tokenizer = Path("tokenizer")
        backend._env = {}
        backend._startup_timeout = 1
        self.backend = backend

    def test_only_changes_decode_and_rejects_conflicting_graph_arguments(self):
        args, values = _decode_graph_args(_BASE, True)
        self.assertEqual([arg for arg in args if not arg.startswith("--cuda-graph-backend-")],
                         [arg for arg in _BASE if arg != "--disable-cuda-graph"])
        self.assertEqual(args[-2:], ["--cuda-graph-backend-decode=full", "--cuda-graph-backend-prefill=disabled"])
        self.assertEqual(values, _GRAPH_FLAGS)
        self.assertEqual(_decode_graph_args(_BASE, False), (_BASE, {"disable_cuda_graph": True}))
        for flags in (_BASE + ["--disable-cuda-graph"], _BASE + ["--cuda-graph-backend-prefill=full"],
                      _BASE + ["--disable-decode-cuda-graph"], _BASE[0:1]):
            with self.subTest(flags=flags), self.assertRaisesRegex(BackendError, "graph-off target"):
                _decode_graph_args(flags, True)
        with self.assertRaisesRegex(BackendError, "must be a boolean"):
            _decode_graph_args(_BASE, 1)

    def _launch(self, parameters, *, info=None, capture=_CAPTURE, kind="framework"):
        stack = {"parameters": parameters, "framework": [], "kernels": []}
        self.backend.bind_revision({"id": "test"}, expected_stack=stack)
        command = []

        def popen(argv, *, stdout, **kwargs):
            command[:] = argv
            stdout.write(capture)
            return SimpleNamespace(pid=99999, poll=lambda: None)

        def endpoint(url, timeout):
            if url.endswith("/health"):
                return io.BytesIO(b"ok")
            record = {"served_model_name": command[command.index("--served-model-name") + 1],
                      "model_path": "model", "tokenizer_path": "tokenizer", "host": "127.0.0.1",
                      "port": int(command[command.index("--port") + 1]), "device": "npu",
                      "dtype": "bfloat16", "tp_size": 1, "attention_backend": "ascend",
                      "version": "0.5.16", "chunked_prefill_size": 512, **(info or {})}
            return io.BytesIO(json.dumps(record).encode("utf-8"))

        with (patch("hyperloop.backends.sglang._write_lease"),
              patch("hyperloop.backends.sglang.subprocess.Popen", side_effect=popen),
              patch("hyperloop.backends.sglang._LOOPBACK.open", side_effect=endpoint)):
            return self.backend.launch({"revision": "test", "workspace": str(self.root), "kind": kind},
                                       {"dtype": "bfloat16"})

    def test_original_stack_keeps_its_graph_off_command(self):
        launch = self._launch({}, capture=b"")
        args = launch["launch_command"]
        self.assertIn("--disable-cuda-graph", args)
        self.assertIn("--skip-server-warmup", args)
        self.assertFalse(any(arg.startswith("--cuda-graph-backend-") for arg in args))
        self.assertEqual(self.backend._server_args, _BASE)

    def test_enabled_candidate_uses_observed_settings_and_capture(self):
        launch = self._launch({"decode_graph_enabled": True}, info=_GRAPH_FLAGS)
        self.assertNotIn("--disable-cuda-graph", launch["launch_command"])
        self.assertEqual(launch["server_info"] | _GRAPH_FLAGS, launch["server_info"])
        self.assertEqual(launch["environment_fingerprint"], "same-environment")
        self.assertEqual(self.backend._server_args, _BASE)

    def test_enabled_candidate_rejects_a_different_observed_mode(self):
        with self.assertRaisesRegex(BackendError, "serving identity mismatch"):
            self._launch({"decode_graph_enabled": True}, info={**_GRAPH_FLAGS, "cuda_graph_backend_decode": "disabled"})

    def test_disabled_candidate_checks_the_original_graph_off_mode(self):
        launch = self._launch({"decode_graph_enabled": False}, info={"disable_cuda_graph": True}, capture=b"")
        self.assertIn("--disable-cuda-graph", launch["launch_command"])
        self.assertIs(launch["server_info"]["disable_cuda_graph"], True)

    def test_enabled_candidate_requires_observed_capture_completion(self):
        with self.assertRaisesRegex(BackendError, "completion of full decode NPU graph"):
            self._launch({"decode_graph_enabled": True}, info=_GRAPH_FLAGS, capture=b"")

    def test_graph_on_does_not_stand_in_for_forward_timing_or_forge_proof(self):
        with self.assertRaisesRegex(BackendError, "model-forward timing"):
            self._launch({"decode_graph_enabled": True}, kind="model_forward")
        with patch.object(self.backend, "_validate_kernel_stack"), self.assertRaisesRegex(BackendError, "Forge invocation proof"):
            self.backend.bind_revision({"id": "forge"}, expected_stack={
                "parameters": {"decode_graph_enabled": True}, "framework": [], "kernels": [{}],
            })
