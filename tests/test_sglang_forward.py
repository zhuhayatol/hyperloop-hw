from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from hyperloop.backends.runtime import BackendError
from hyperloop.backends.sglang_forward import FORWARD_MARKER, forward_repeats, forward_samples, measure_forward
from hyperloop.evaluation.evidence import EvidenceProblem
from hyperloop.evaluation.workflow import Evaluator


def row(window, phase="EXTEND", *, pid=77, elapsed=1.5):
    tokens = [128] if phase == "EXTEND" else [1]
    return {"window": window, "pid": pid, "forward_mode": phase, "batch_size": 1,
            "input_shape": tokens, "position_shape": tokens, "model_dtype": "torch.bfloat16",
            "input_dtype": "torch.int64", "device": "npu:0", "elapsed_ms": elapsed,
            "sequence_lengths": [128], "extend_lengths": [128] if phase == "EXTEND" else [],
            "prefix_lengths": [0] if phase == "EXTEND" else []}


class SGLangForwardTests(unittest.TestCase):
    def test_opt_in_sets_a_bounded_repeat_count(self):
        self.assertIsNone(forward_repeats({}))
        self.assertEqual(forward_repeats({"enabled": True}), 3)
        self.assertEqual(forward_repeats({"enabled": True, "repeats": 4}), 4)
        for value in ({"repeats": 1}, {"enabled": True, "repeats": 0}, {"enabled": True, "repeats": True}):
            with self.subTest(value=value), self.assertRaises(BackendError):
                forward_repeats(value)

    def test_only_accepts_rows_in_the_capture_window_from_the_owned_worker(self):
        with TemporaryDirectory() as temporary:
            log = Path(temporary) / "server.log"
            before = (FORWARD_MARKER + json.dumps(row("warmup")) + "\n").encode()
            current = (FORWARD_MARKER + json.dumps(row("one", "DECODE")) + "\n").encode()
            log.write_bytes(before + current)
            self.assertEqual(forward_samples(log, len(before), "one", [77])[0]["phase"], "decode")
            for offset, window, workers in ((0, "one", [77]), (len(before), "one", [88]),
                                            (len(before) + len(current), "one", [77])):
                with self.subTest(offset=offset, workers=workers), self.assertRaises(BackendError):
                    forward_samples(log, offset, window, workers)

    def test_replays_fixed_requests_and_separates_phases_and_windows(self):
        @contextmanager
        def flush(url, path, timeout):
            self.assertEqual((url, path, timeout), ("http://127.0.0.1:1", "/flush_cache", 1))
            yield type("Response", (), {"read": lambda self, limit: b"flushed"})()

        request = {"id": "p1", "input_tokens": [12, 13], "output_tokens": 2, "phase": "prefill"}
        work = {"requests": [request], "concurrency": 1,
                "sampling": {"temperature": 0, "ignore_eos": True}}
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            control = root / "control"
            log = root / "server.log"
            log.touch()

            def generate(url, payload, timeout):
                self.assertEqual(payload["input_ids"], [12, 13])
                nonce = control.read_text(encoding="utf-8").strip()
                with log.open("a", encoding="utf-8") as stream:
                    for phase, elapsed in (("EXTEND", 2.0), ("DECODE", 1.0)):
                        stream.write(FORWARD_MARKER + json.dumps(row(nonce, phase, elapsed=elapsed)) + "\n")
                return {"output_ids": [14, 15]}

            with patch("hyperloop.backends.sglang_forward._request", flush), \
                    patch("hyperloop.backends.sglang_forward._complete", generate):
                measured = measure_forward("http://127.0.0.1:1", work, repeats=2, control=control,
                                           log=log, workers={77}, timeout=1)
            self.assertFalse(control.exists())
        self.assertEqual(measured["samples_ms"], [2.0, 1.0, 2.0, 1.0])
        self.assertEqual(measured["phases"], {"prefill": {"count": 2, "median_ms": 2.0},
                                            "decode": {"count": 2, "median_ms": 1.0}})
        self.assertEqual(measured["repeats"][0]["requests"][0]["output"], [14, 15])
        self.assertNotEqual(measured["repeats"][0]["window"], measured["repeats"][1]["window"])
        scope = {"repeats": 2, "symbol": measured["model_symbol"]}
        self.assertEqual(Evaluator._verify_forward_replays(measured, scope, {"p1": [14, 15]}), measured["phases"])
        measured["repeats"][1]["requests"][0]["output"] = [14, 16]
        with self.assertRaisesRegex(EvidenceProblem, "forward_replay_output_mismatch"):
            Evaluator._verify_forward_replays(measured, scope, {"p1": [14, 15]})
