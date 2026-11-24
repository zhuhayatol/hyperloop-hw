"""Runtime validation calls and receipts remain linked to immutable evidence."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from hyperloop.backends import BackendError, SimulatedBackend
from hyperloop.evaluation import Evaluator


WORKLOAD = {
    "id": "validation-trace", "concurrency": 1,
    "requests": [{"id": "request-0", "input_tokens": [2, 3], "output_tokens": 8}],
}
REFERENCE = {"request-0": [5.0]}
POLICY = {"repeats": 3, "warmup": 1, "min_gain_pct": 2}


class ValidationInstrumentationTests(TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.backend = SimulatedBackend(
            {"id": "target", "devices": ["card-0", "card-1"], "tp_size": 2},
            {"original": {"throughput": 100}, "candidate": {"throughput": 110}},
        )
        self.evaluator = Evaluator(self.backend, self.root)

    def flow(self, result, *, mode="simulated"):
        records = result["evidence"]
        ref = records["validation_flow"]
        path = self.root / ref["path"]
        raw = path.read_bytes()
        self.assertEqual(ref["producer"], "evaluation_trace")
        self.assertEqual(ref["sha256"], hashlib.sha256(raw).hexdigest())
        flow = json.loads(raw)
        self.assertEqual(flow["schema_version"], 1)
        self.assertEqual(flow["attempt_id"], result["attempt_id"])
        self.assertEqual(flow["mode"], mode)
        for row in flow["steps"]:
            self.assertIn(row["state"], {"requested", "succeeded", "failed", "skipped"})
            self.assertTrue({"sender", "receiver", "message", "operation", "scope"} <= row.keys())
            self.assertTrue(set(row.get("evidence_keys", ())) <= records.keys())
        self.assertIn("decision", flow["steps"][-1]["evidence_keys"])
        return flow["steps"]

    @staticmethod
    def position(steps, operation, scope, state="succeeded"):
        return next(i for i, row in enumerate(steps)
                    if (row["operation"], row["scope"], row["state"]) == (operation, scope, state))

    def candidate(self, *, baseline, kind="kernel"):
        return self.evaluator.candidate("candidate", "original", WORKLOAD, REFERENCE, POLICY,
                                        attempt_id="candidate", baseline=baseline, kind=kind)

    def test_kernel_candidate_trace_preserves_call_returns_and_boundary_order(self):
        baseline = self.evaluator.baseline("original", WORKLOAD, REFERENCE, POLICY)
        result = self.candidate(baseline=baseline)
        self.assertEqual(result["version_action"], "KEEP")
        steps = self.flow(result)
        stages = [
            ("probe", "evaluation"), ("probe_identity", "evaluation"), ("prepare", "candidate"),
            ("launch", "candidate"), ("instance_identity", "candidate"), ("correctness", "kernel"),
            ("device_timing_summary", "kernel"), ("correctness", "candidate"), ("measurement_summary", "candidate"),
            ("cleanup", "candidate"), ("prepare", "parent"), ("correctness", "parent"),
            ("measurement_summary", "parent"), ("cleanup", "parent"), ("paired_comparison", "evaluation"),
            ("decision", "evaluation"),
        ]
        positions = [self.position(steps, operation, scope) for operation, scope in stages]
        self.assertEqual(positions, sorted(positions))
        request, reply = [r for r in steps if r["operation"] == "launch" and r["scope"] == "candidate"]
        self.assertEqual((request["state"], reply["state"]), ("requested", "succeeded"))
        self.assertEqual((request["sender"], request["receiver"]), (reply["receiver"], reply["sender"]))
        self.assertEqual(reply["sender"], "RuntimeBackend")
        self.assertIn("candidate_launch", reply["evidence_keys"])
        samples = [r for r in steps if r["operation"] == "benchmark_sample" and r["scope"] == "candidate"]
        self.assertEqual([r["state"] for r in samples], ["requested", "succeeded"] * POLICY["repeats"])
        self.assertEqual([r["evidence_keys"][-1] for r in samples[1::2]], [f"candidate_repeat_{i}" for i in range(3)])

    def test_wrong_kernel_output_records_skips_and_cleanup(self):
        self.backend.register_revision("candidate", {"faults": ["wrong_output"]})
        result = self.candidate(baseline=None)
        steps = self.flow(result)
        self.assertEqual(result["version_action"], "REVERT")
        self.assertIn("operator_kernel_correctness", steps[self.position(steps, "correctness", "kernel", "failed")]["evidence_keys"])
        self.assertLess(self.position(steps, "device_timer", "kernel", "skipped"), self.position(steps, "cleanup", "candidate"))
        self.assertEqual([r["state"] for r in steps if r["operation"] == "benchmark_sample"], ["skipped"])
        self.assertEqual([r["state"] for r in steps if r["operation"] == "prepare" and r["scope"] == "parent"], ["skipped"])
        self.assertNotIn(("device_timer", "candidate"), self.backend.calls)
        self.assertNotIn(("benchmark", "candidate"), self.backend.calls)

    def test_invalid_identity_stops_and_cleans_up(self):
        self.backend.register_revision("candidate", {"faults": ["old_implementation"]})
        result = self.candidate(baseline=None, kind="framework")
        steps = self.flow(result)
        failed = self.position(steps, "instance_identity", "candidate", "failed")
        self.assertIn("candidate_launch", steps[failed]["evidence_keys"])
        self.assertLess(failed, self.position(steps, "cleanup", "candidate"))
        self.assertFalse(any(r["operation"] == "observe_outputs" for r in steps))
        self.assertEqual(result["version_action"], "HOLD")

    def test_failed_probe_records_backend_error_without_launch(self):
        with patch.object(self.backend, "probe", side_effect=BackendError("unavailable")):
            result = self.evaluator.baseline("original", WORKLOAD, REFERENCE, POLICY)
        steps = self.flow(result, mode=None)
        self.assertEqual([r["state"] for r in steps if r["operation"] == "probe"], ["requested", "failed"])
        self.assertIn("backend_failure", steps[-1]["evidence_keys"])
        self.assertFalse(any(r["operation"] in {"prepare", "launch"} for r in steps))

    def test_partial_parent_sample_remains_a_raw_receipt_with_failed_validation(self):
        baseline = self.evaluator.baseline("original", WORKLOAD, REFERENCE, POLICY)
        self.backend.register_revision("original", {"faults": ["partial_samples"]})
        result = self.candidate(baseline=baseline, kind="framework")
        steps = self.flow(result)
        failed = self.position(steps, "validate_sample", "parent", "failed")
        self.assertIn("parent_repeat_1", steps[failed]["evidence_keys"])
        self.assertEqual(
            [r["evidence_keys"][-1] for r in steps if r["operation"] == "benchmark_sample"
             and r["scope"] == "parent" and r["state"] == "succeeded"], ["parent_repeat_0", "parent_repeat_1"],
        )
        self.assertLess(failed, self.position(steps, "cleanup", "parent"))
        self.assertFalse(any(r["operation"] == "paired_comparison" for r in steps))
        self.assertEqual(result["version_action"], "HOLD")

    def test_cleanup_failure_retains_completed_measurement_without_parent_attempt(self):
        self.backend.register_revision("candidate", {"faults": ["cleanup"]})
        result = self.candidate(baseline=None, kind="framework")
        steps = self.flow(result)
        failed = self.position(steps, "cleanup", "candidate", "failed")
        self.assertLess(self.position(steps, "measurement_summary", "candidate"), failed)
        self.assertIn("candidate_cleanup", steps[failed]["evidence_keys"])
        self.assertFalse(any(r["scope"] == "parent" for r in steps))
        self.assertEqual(result["version_action"], "HOLD")

    def test_separate_forward_and_partial_profile_link_raw_evidence(self):
        forward = self.evaluator.forward("candidate", WORKLOAD, REFERENCE, POLICY, attempt_id="forward")
        steps = self.flow(forward)
        self.assertLess(self.position(steps, "correctness", "forward"), self.position(steps, "forward_timer", "forward"))
        self.assertIn("forward_timing", steps[self.position(steps, "forward_timing_summary", "forward")]["evidence_keys"])
        self.assertFalse(any(r["operation"] == "benchmark_sample" for r in steps))

        self.backend.register_revision("candidate", {"faults": ["profile_missing_rank"]})
        profile = self.evaluator.profile("candidate", WORKLOAD, attempt_id="profile")
        steps = self.flow(profile)
        failure = self.position(steps, "parse_profile", "profile", "failed")
        self.assertLess(self.position(steps, "collect_profile", "profile"), failure)
        self.assertLess(failure, self.position(steps, "cleanup", "profile"))
        self.assertIn("profile_rank_0", steps[failure]["evidence_keys"])
        self.assertNotIn("profile_rank_1", steps[failure]["evidence_keys"])
        self.assertIn("profile_analysis", steps[failure]["evidence_keys"])
