"""Measured Critic outcomes derive from recorded evidence and locked rules."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from hyperloop.actions.candidates import candidate_revision, original_revision, stack_identity
from hyperloop.backends import SimulatedBackend
from hyperloop.evaluation import Evaluator
from hyperloop.evaluation.adjudication import adjudicate
from hyperloop.session import SessionStore

TARGET = {"id": "simulated", "devices": ["rank-card"], "tp_size": 1}
WORKLOAD = {
    "id": "prefill", "concurrency": 1,
    "requests": [{"id": "request-0", "input_tokens": [2, 3], "output_tokens": 8}],
}
REFERENCE = {"request-0": [5.0]}
RULES = {"repeats": 3, "warmup": 1, "min_gain_pct": 2, "noise_floor_pct": 0.5}


class AdjudicationTests(TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base_revision = original_revision("original")
        self.store = SessionStore.create(Path(directory.name), "simulated", "adjudication", {
            "task": {"workload": WORKLOAD, "rules": RULES}, "reference": REFERENCE,
        })
        self.backend = SimulatedBackend(TARGET, {"original": {"loaded_stack": stack_identity(self.base_revision)}})
        self.evaluator = Evaluator(self.backend, self.store.path)
        self.baseline = {
            **self.evaluator.baseline("original", WORKLOAD, REFERENCE, RULES),
            "revision": self.base_revision,
        }

    def candidate(self, revision_id: str = "R01", **fixture) -> dict:
        specification = {
            "id": revision_id,
            "reason": "test evidence",
            "expected_metric": "throughput",
            "risk": "low",
            "validation": "paired trials",
            "parameters": {"chunked_prefill_size": 2048},
        }
        revision = candidate_revision(specification, self.base_revision, "parameters")
        self.backend.register_revision(revision_id, {"throughput": 110, **fixture, "loaded_stack": stack_identity(revision)})
        observed = self.evaluator.candidate(
            revision_id, "original", WORKLOAD, REFERENCE, RULES,
            attempt_id=revision_id, baseline=self.baseline,
            expected_stack=stack_identity(revision), expected_parent_stack=stack_identity(self.base_revision),
        )
        return {
            **observed, "candidate_id": revision_id, "parent": "original", "parent_revision": self.base_revision,
            "revision": revision, "kind": "parameters",
        }

    def test_recomputes_measurements_and_gates_from_verified_raw_files(self) -> None:
        claimed = self.candidate()
        claimed.update(version_action="REVERT", acceptance="fail", reason="adapter_claim", comparison={})
        claimed["measurement"]["summary"]["throughput"] = 700.0
        baseline = deepcopy(self.baseline)
        baseline["measurement"]["summary"]["throughput"] = 1.0
        result = adjudicate(self.store, claimed, baseline, RULES)
        self.assertEqual(result["version_action"], "KEEP")
        self.assertEqual(result["reason"], "gates_passed")
        self.assertAlmostEqual(result["measurement"]["summary"]["throughput"], 110.0)
        self.assertAlmostEqual(result["comparison"]["over_parent"]["throughput_pct"], 10.0)
        self.assertAlmostEqual(result["comparison"]["over_original"]["throughput_pct"], 10.0)
        self.assertIn("candidate_service_correctness", result["adjudication"]["attempt_evidence"])
        self.assertIn("parent_repeat_2", result["adjudication"]["attempt_evidence"])
        self.assertIn("revision_repeat_2", result["adjudication"]["baseline_evidence"])

    def test_claimed_keep_cannot_bypass_seeded_latency_regression(self) -> None:
        claimed = self.candidate(throughput=130, tpot_ms=35)
        claimed.update(version_action="KEEP", acceptance="pass", reason="adapter_claim")
        result = adjudicate(self.store, claimed, self.baseline, RULES)
        self.assertEqual(result["version_action"], "REVERT")
        self.assertEqual(result["reason"], "tpot_regression")
        self.assertAlmostEqual(result["comparison"]["over_parent"]["tpot_pct"], 25.0)

    def test_rejects_content_change_and_cross_arm_reference_substitution(self) -> None:
        claimed = self.candidate()
        mismatched = deepcopy(claimed)
        mismatched["evidence"]["parent_repeat_0"] = claimed["evidence"]["candidate_repeat_0"]
        result = adjudicate(self.store, mismatched, self.baseline, RULES)
        self.assertEqual(result["version_action"], "HOLD")
        self.assertEqual(result["reason"], "adjudication_evidence_identity:parent_repeat_0")
        path = self.store.artifact_path(claimed["evidence"]["candidate_repeat_0"]["path"])
        path.write_text("{}\n", encoding="utf-8")
        result = adjudicate(self.store, claimed, self.baseline, RULES)
        self.assertEqual(result["evidence_state"], "conflict")
        self.assertEqual(result["reason"], "adjudication_evidence_digest:candidate_repeat_0")

    def test_same_digest_does_not_make_wrong_normalized_metrics_consistent(self) -> None:
        claimed = self.candidate()
        reference = claimed["evidence"]["candidate_measurement"]
        path = self.store.artifact_path(reference["path"])
        record = json.loads(path.read_bytes())
        record["summary"]["throughput"] = 999.0
        encoded = json.dumps(record).encode("utf-8")
        path.write_bytes(encoded)
        reference["sha256"] = hashlib.sha256(encoded).hexdigest()
        result = adjudicate(self.store, claimed, self.baseline, RULES)
        self.assertEqual(result["version_action"], "HOLD")
        self.assertEqual(result["reason"], "adjudication_measurement_mismatch:candidate")

    def test_requires_original_raw_evidence_and_frozen_policy(self) -> None:
        claimed = self.candidate()
        incomplete = deepcopy(self.baseline)
        del incomplete["evidence"]["revision_repeat_1"]
        result = adjudicate(self.store, claimed, incomplete, RULES)
        self.assertEqual(result["version_action"], "HOLD")
        self.assertEqual(result["reason"], "adjudication_evidence_missing:revision_repeat_1")
        result = adjudicate(self.store, claimed, self.baseline, {**RULES, "min_gain_pct": 0})
        self.assertEqual(result["acceptance"], "undetermined")
        self.assertEqual(result["reason"], "adjudication_locked_rules_mismatch")

    def test_correctness_only_revert_is_independent_of_performance_and_baseline(self) -> None:
        claimed = self.candidate(faults=["wrong_output"])
        self.assertNotIn("candidate_repeat_0", claimed["evidence"])
        result = adjudicate(self.store, claimed, {}, RULES)
        self.assertEqual(result["version_action"], "REVERT")
        self.assertEqual(result["reason"], "service_correctness_failed")
        self.assertFalse(result["correctness"]["pass"])

    def test_kernel_keep_requires_an_independently_verified_device_stage(self) -> None:
        specification = {
            "id": "K01", "reason": "test device evidence", "expected_metric": "throughput",
            "risk": "low", "validation": "paired trials", "operator": "attention",
            "profile_ref": "simulated-profile", "backend": "forge",
        }
        revision = candidate_revision(specification, self.base_revision, "kernel")
        self.backend.register_revision("K01", {"throughput": 110, "loaded_stack": stack_identity(revision)})
        observed = self.evaluator.candidate(
            "K01", "original", WORKLOAD, REFERENCE, RULES,
            attempt_id="kernel", baseline=self.baseline, kind="kernel",
            expected_stack=stack_identity(revision), expected_parent_stack=stack_identity(self.base_revision),
        )
        claimed = {
            **observed, "candidate_id": "K01", "parent": "original", "parent_revision": self.base_revision,
            "revision": revision, "kind": "kernel",
        }
        result = adjudicate(self.store, claimed, self.baseline, RULES)
        self.assertEqual(result["version_action"], "KEEP")
        self.assertEqual(result["device_timing"]["boundary"], "kernel_device")
        self.assertIn("operator_kernel_correctness", result["adjudication"]["attempt_evidence"])
        del claimed["evidence"]["device_timing_raw"]
        result = adjudicate(self.store, claimed, self.baseline, RULES)
        self.assertEqual(result["version_action"], "HOLD")
        self.assertEqual(result["reason"], "adjudication_evidence_missing:device_timing_raw")

    def test_preserves_an_existing_block_or_trial_selection(self) -> None:
        blocked = self.candidate("blocked", faults=["cleanup"])
        self.assertEqual(adjudicate(self.store, blocked, self.baseline, RULES)["reason"], "cleanup_unconfirmed")
        self.backend.register_revision("blocked", {})
        self.backend.cleanup({"revision": "blocked", "workspace": blocked["runtime"]["workspace"]}, None)
        lower = self.candidate("lower")
        lower.update(version_action="HOLD", reason="stronger_candidate_selected")
        result = adjudicate(self.store, lower, self.baseline, RULES)
        self.assertEqual(result["version_action"], "HOLD")
        self.assertEqual(result["acceptance"], "pass")
        self.assertEqual(result["reason"], "stronger_candidate_selected")
