"""Independent evaluation uses the same evidence gates with two mock targets."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from hyperloop.backends import SimulatedBackend
from hyperloop.evaluation import Evaluator


WORKLOAD = {
    "id": "prefill", "concurrency": 1,
    "requests": [{"id": "request-0", "input_tokens": [2, 3], "output_tokens": 8}],
}
REFERENCE = {"request-0": [5.0]}
POLICY = {"repeats": 3, "warmup": 1, "min_gain_pct": 2}


class EvaluationTests(TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.backend = self.make_backend(1)
        self.evaluator = Evaluator(self.backend, self.root)

    @staticmethod
    def make_backend(ranks):
        return SimulatedBackend(
            {"id": f"target-{ranks}", "devices": [f"card-{i}" for i in range(ranks)], "tp_size": ranks},
            {"original": {"throughput": 100}, "R01": {"throughput": 110},
             "R02": {"throughput": 112, "framework": "R01", "kernels": ["attention-patch"]}},
        )

    def baseline(self):
        return self.evaluator.baseline("original", WORKLOAD, REFERENCE, POLICY)

    def evaluate(self, revision, parent, baseline, **kwargs):
        return self.evaluator.candidate(revision, parent, WORKLOAD, REFERENCE, POLICY,
                                        attempt_id=revision, baseline=baseline, **kwargs)

    def test_parent_and_original_compare_separately(self):
        baseline = self.baseline()
        kept = self.evaluate("R01", "original", baseline)
        self.assertEqual(kept["version_action"], "KEEP")
        reverted = self.evaluate("R02", "R01", baseline, kind="kernel")
        self.assertEqual(reverted["version_action"], "REVERT")
        self.assertEqual(reverted["reason"], "gain_below_threshold")
        self.assertEqual(reverted["evidence_state"], "sufficient")
        self.assertAlmostEqual(reverted["comparison"]["over_parent"]["throughput_pct"], 1.818181818)
        self.assertAlmostEqual(reverted["comparison"]["over_original"]["throughput_pct"], 12)
        self.assertEqual(reverted["measurement"]["loaded_stack"], {"framework": "R01", "kernels": ["attention-patch"]})
        self.assertEqual(reverted["device_timing"]["boundary"], "kernel_device")
        calls = [operation for operation, name in self.backend.calls if name == "R02"]
        self.assertLess(calls.index("verify_kernel"), calls.index("device_timer"))
        self.assertLess(calls.index("device_timer"), calls.index("verify_service"))
        self.assertLess(calls.index("verify_service"), calls.index("benchmark"))
        for entry in reverted["evidence"].values():
            path = self.root / entry["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"])
            self.assertNotIn("workspaces", path.parts)

    def test_invalid_output_ends_before_profiling_or_performance(self):
        self.backend.register_revision("bad", {"throughput": 600, "faults": ["wrong_output"]})
        result = self.evaluate("bad", "original", self.baseline(), kind="kernel")
        self.assertEqual(result["version_action"], "REVERT")
        self.assertEqual(result["reason"], "kernel_correctness_failed")
        self.assertNotIn(("benchmark", "bad"), self.backend.calls)
        self.assertNotIn(("device_timer", "bad"), self.backend.calls)
        self.assertFalse(result["kernel_correctness"]["pass"])

    def test_old_load_and_wrong_runtime_never_reach_timing(self):
        baseline = self.baseline()
        for fault, reason, state in (
            ("old_implementation", "loaded_revision_mismatch", "conflict"),
            ("wrong_environment", "runtime_environment_fingerprint_mismatch", "incomparable"),
        ):
            with self.subTest(fault=fault):
                self.backend.register_revision(fault, {"faults": [fault], "throughput": 500})
                result = self.evaluate(fault, "original", baseline)
                self.assertEqual(result["version_action"], "HOLD")
                self.assertEqual(result["reason"], reason)
                self.assertEqual(result["evidence_state"], state)
                self.assertIn("candidate_launch", result["evidence"])
                self.assertIn(("cleanup", fault), self.backend.calls)
                self.assertNotIn(("benchmark", fault), self.backend.calls)

    def test_missing_parent_pair_retains_candidate_observation(self):
        baseline = self.baseline()
        self.backend.register_revision("R01", {"throughput": 110, "faults": ["partial_samples"]})
        result = self.evaluate("R02", "R01", baseline)
        self.assertEqual(result["evidence_state"], "missing")
        self.assertEqual(result["acceptance"], "undetermined")
        self.assertEqual(result["version_action"], "HOLD")
        self.assertEqual(result["measurement"]["summary"]["throughput"], 112)
        raw = self.root / result["evidence"]["parent_repeat_1"]["path"]
        self.assertIsNone(json.loads(raw.read_text(encoding="utf-8")))

    def test_cleanup_failure_retains_raw_results_and_blocks_promotion(self):
        baseline = self.baseline()
        self.backend.register_revision("unsafe", {"throughput": 120, "faults": ["cleanup"]})
        result = self.evaluate("unsafe", "original", baseline)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["reason"], "cleanup_unconfirmed")
        self.assertEqual(result["version_action"], "HOLD")
        self.assertIn("measurement", result)
        later = self.evaluator.measure_revision("R01", WORKLOAD, REFERENCE, POLICY, attempt_id="next")
        self.assertEqual(later["acceptance"], "undetermined")

    def test_dual_rank_capture_saves_raw_per_rank(self):
        backend = self.make_backend(2)
        evaluator = Evaluator(backend, self.root)
        trace = evaluator.profile("R01", WORKLOAD, attempt_id="dual-profile")
        self.assertEqual(trace["ranks"], [0, 1])
        self.assertEqual(trace["missing_ranks"], [])
        self.assertEqual([row["rank"] for row in trace["hotspots"]], [0, 1])
        self.assertEqual([row["duration_us"] for row in trace["hotspots"]], [90, 91])
        self.assertIn("profile_rank_1", trace["evidence"])
        self.assertNotIn(("benchmark", "R01"), backend.calls)
        backend.register_revision("R02", {"fault": "profile_missing_rank"})
        partial = evaluator.profile("R02", WORKLOAD, attempt_id="dual-partial")
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(partial["missing_ranks"], [1])

    def test_baseline_from_other_target_does_not_promote(self):
        baseline = self.baseline()
        evaluator = Evaluator(self.make_backend(2), self.root)
        result = evaluator.candidate("R01", "original", WORKLOAD, REFERENCE, POLICY,
                                     attempt_id="wrong-comparison", baseline=baseline)
        self.assertEqual(result["reason"], "comparison_environment_fingerprint_mismatch")
        self.assertEqual(result["version_action"], "HOLD")

    def test_baseline_timing_protocol_must_also_match(self):
        baseline = self.evaluator.baseline("original", WORKLOAD, REFERENCE, {**POLICY, "warmup": 0})
        result = self.evaluate("R01", "original", baseline)
        self.assertEqual(result["reason"], "comparison_protocol_fingerprint_mismatch")
        self.assertEqual(result["version_action"], "HOLD")

    def test_expected_loaded_stack_is_external_to_the_adapter(self):
        baseline = self.baseline()
        result = self.evaluate("R02", "R01", baseline, kind="kernel",
                               expected_stack={"framework": "R01", "kernels": ["different-artifact"]})
        self.assertEqual(result["reason"], "loaded_stack_mismatch")
        self.assertEqual(result["version_action"], "HOLD")
        self.assertNotIn(("verify_kernel", "R02"), self.backend.calls)

    def test_dual_target_requires_observed_communication_capability(self):
        backend = SimulatedBackend({"id": "dual", "devices": ["card0", "card1"], "tp_size": 2,
                                    "capabilities": ["verify", "benchmark", "profile"]})
        blocked = Evaluator(backend, self.root).baseline("original", WORKLOAD, REFERENCE, POLICY)
        self.assertEqual(blocked["status"], "blocked")
        self.assertEqual(blocked["reason"], "capabilities_missing:hccl")
        self.assertEqual(backend.calls, [("probe", None)])

    def test_latency_guard_can_reject_higher_throughput(self):
        baseline = self.baseline()
        self.backend.register_revision("slow-decode", {"throughput": 130, "tpot_ms": 49})
        result = self.evaluate("slow-decode", "original", baseline)
        self.assertEqual(result["version_action"], "REVERT")
        self.assertEqual(result["reason"], "tpot_regression")

    def test_device_only_timing_stays_separate(self):
        result = self.evaluator.kernel("R02", WORKLOAD, REFERENCE, POLICY, attempt_id="device")
        self.assertEqual(result["acceptance"], "pass")
        self.assertEqual(result["version_action"], "HOLD")
        self.assertIn("device_timing", result)
        self.assertNotIn("measurement", result)

    def test_model_forward_timing_is_its_own_boundary(self):
        result = self.evaluator.forward("original", WORKLOAD, REFERENCE, POLICY, attempt_id="forward")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["forward_timing"]["boundary"], "model_forward")
        self.assertNotIn("measurement", result)
        self.assertIn(("forward_timer", "original"), self.backend.calls)
        self.assertFalse(any(name == "benchmark" for name, _ in self.backend.calls))

    def test_attempt_identifiers_and_evidence_cannot_be_overwritten(self):
        baseline = self.baseline()
        frozen_file = self.root / baseline["evidence"]["decision"]["path"]
        previous = frozen_file.read_bytes()
        with self.assertRaises(FileExistsError):
            self.baseline()
        with self.assertRaises(ValueError):
            self.evaluator.profile("original", WORKLOAD, attempt_id="../escape")
        self.assertEqual(frozen_file.read_bytes(), previous)
