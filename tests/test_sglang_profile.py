import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hyperloop.backends.runtime import BackendError
from hyperloop.backends.sglang_profile import _trace_events, collect_profile
from hyperloop.evaluation.measurements import rank_hotspots


def trace():
    return [
        {"ph": "M", "name": "process_name", "pid": 4, "tid": 0, "args": {"name": "Ascend Hardware"}},
        {"ph": "X", "cat": "cpu_op", "name": "SelfAttentionOperation", "pid": 1, "tid": 2,
         "ts": "1795438649826982.130", "dur": 4,
         "args": {"Call stack": "/workspace/sglang/python/sglang/srt/attention.py(1479): forward_extend;",
                  "Input Dims": [[4096, 896]], "Input type": ["BFloat16"]}},
        {"ph": "s", "name": "torch_to_npu", "id": 8, "pid": 1, "tid": 2, "ts": "1795438649826982.13"},
        {"ph": "f", "name": "torch_to_npu", "id": 8, "pid": 4, "tid": 5, "ts": "1795438649827361.110"},
        {"ph": "X", "name": "FlashAttentionKernel", "pid": 4, "tid": 5, "ts": "1795438649827361.11", "dur": 12,
         "args": {"Task Type": "MIX_AIC"}},
        {"ph": "X", "name": "UnmappedKernel", "pid": 4, "tid": 5, "ts": "1795438649827361.12", "dur": 2},
        {"ph": "X", "name": "OtherCPUOp", "pid": 1, "tid": 2, "ts": "1795438649827361.12", "dur": 18},
    ]


class AscendProfileTests(unittest.TestCase):
    def test_trace_keeps_exact_ascend_events_and_correlates_real_flow(self):
        rows, window = _trace_events({"traceEvents": trace()}, "prefill")
        self.assertEqual([row["name"] for row in rows], ["FlashAttentionKernel", "UnmappedKernel"])
        self.assertEqual(rows[0]["callsite"], "sglang/srt/attention.py:1479")
        self.assertEqual(rows[0]["input_shape"], [4096, 896])
        self.assertEqual(rows[0]["dtype"], "BFloat16")
        self.assertEqual(rows[0]["operator"], "SelfAttentionOperation")
        self.assertEqual([row["event_kind"] for row in rows], ["kernel", "device_runtime"])
        self.assertEqual(rows[1]["callsite"], "")
        self.assertEqual(rows[1]["source_correlation"], "unresolved")
        self.assertLess(window[0], window[1])
        hotspots = rank_hotspots({"rank": 0, "events": [*rows, {**rows[0], "phase": "decode"}]})
        self.assertEqual([row["phase"] for row in hotspots], ["prefill", "decode", "prefill"])

    def test_trace_does_not_infer_callsite_from_ambiguous_flow_or_cpu_only_trace(self):
        events = trace()
        events.insert(2, {**events[1], "name": "SecondCPUOp"})
        self.assertEqual(_trace_events(events, "decode")[0][0]["callsite"], "")
        with self.assertRaisesRegex(BackendError, "no timed hardware"):
            _trace_events(events[1:], "decode")

    def test_capture_keeps_owned_distinct_raw_trace_and_request_evidence_per_group(self):
        workload = {"sampling": {"temperature": 0, "ignore_eos": True}, "concurrency": 1,
                    "requests": [{"id": "long", "phase": "prefill", "input_tokens": [1, 2], "output_tokens": 2},
                                 {"id": "short", "phase": "decode", "input_tokens": [3], "output_tokens": 3}]}
        calls = []
        directory = None

        def api(base_url, endpoint, timeout, body=None):
            nonlocal directory
            calls.append(endpoint)
            if endpoint == "/flush_cache":
                return "Cache flushed."
            if endpoint == "/start_profile":
                directory = Path(body["output_dir"])
                self.assertEqual(body["activities"], ["CPU", "GPU"])
                self.assertIs(body["with_stack"], True)
                self.assertIs(body["record_shapes"], True)
            if endpoint == "/stop_profile":
                (directory / "trace_view.json").write_text(json.dumps(trace()), encoding="utf-8")
            return "OK"

        def complete(base_url, payload, timeout):
            calls.append("generate")
            return {"meta_info": {"id": payload["rid"], "prompt_tokens": len(payload["input_ids"]),
                                  "completion_tokens": payload["sampling_params"]["max_new_tokens"]}}

        with tempfile.TemporaryDirectory() as work, patch("hyperloop.backends.sglang_profile._profile_api", api), \
                patch("hyperloop.backends.sglang_profile._complete", complete):
            root = Path(work).resolve()
            result = collect_profile("http://127.0.0.1:30000", workload, revision="base", tool_version="v0.5.16",
                                     directory=root / "profile", evidence_root=root, request_timeout=3, export_timeout=5)
            self.assertEqual(calls, ["generate", "/flush_cache", "/start_profile", "generate", "/stop_profile"] * 2)
            self.assertEqual(result["capture"]["phase_grouping"], "frozen_request_group")
            phases = result["capture"]["phases"]
            self.assertEqual([item["phase"] for item in phases], ["prefill", "decode"])
            self.assertEqual([item["requests"][0]["request_id"] for item in phases], ["long", "short"])
            self.assertEqual([item["trace"]["path"] for item in phases], ["profile/prefill/trace_view.json",
                                                                           "profile/decode/trace_view.json"])
            self.assertTrue(all(len(item["trace"]["sha256"]) == 64 for item in phases))
            self.assertEqual([item["trace"]["kernel_events"] for item in phases], [1, 1])
            self.assertEqual(result["ranks"][0]["revision"], "base")
            self.assertEqual([item["phase"] for item in result["ranks"][0]["events"]], ["prefill"] * 2 + ["decode"] * 2)

    def test_capture_stops_the_profiler_when_profile_request_fails(self):
        workload = {"sampling": {"temperature": 0, "ignore_eos": True}, "concurrency": 1,
                    "requests": [{"id": "long", "phase": "prefill", "input_tokens": [1, 2], "output_tokens": 2}]}
        with tempfile.TemporaryDirectory() as work, patch("hyperloop.backends.sglang_profile._profile_api") as api, \
                patch("hyperloop.backends.sglang_profile._complete", side_effect=[{}, BackendError("generate failed")]):
            api.return_value = "Cache flushed."
            root = Path(work).resolve()
            with self.assertRaisesRegex(BackendError, "generate failed"):
                collect_profile("http://127.0.0.1:30000", workload, revision="base", tool_version="v0.5.16",
                                directory=root / "profile", evidence_root=root, request_timeout=3, export_timeout=5)
            self.assertEqual([call.args[1] for call in api.call_args_list],
                             ["/flush_cache", "/start_profile", "/stop_profile"])
