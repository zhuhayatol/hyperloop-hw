from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from hyperloop.actions.candidates import candidate_revision, original_revision, stack_identity
from hyperloop.actions.runtime import RuntimeActions
from hyperloop.contracts import ContractError
from hyperloop.kernel.routes import plan_kernel_handoff
from hyperloop.session import SessionStore


def candidate(identifier, **kwargs):
    return {
        "id": identifier,
        "reason": "Attention hotspot",
        "expected_metric": "throughput",
        "risk": "output drift",
        "validation": "same workload and Reference",
        **kwargs,
    }


class CandidateCompositionTests(unittest.TestCase):
    def test_kernel_inherits_accepted_framework_parameters_and_patch(self):
        original = original_revision("base")
        tuned = candidate_revision(candidate("params", parameters={"tile": 4}), original, "parameters")
        patched = candidate_revision(candidate("patch", patch_path="sglang/attention.py"), tuned, "patch")
        final = candidate_revision(
            candidate("kernel", backend="forge", operator="attention", profile_ref={"path": "profile/p.json"}),
            patched,
            "kernel",
        )
        stack = stack_identity(final)
        self.assertEqual([item["candidate_id"] for item in stack["framework"]], ["params", "patch"])
        self.assertEqual(stack["parameters"], {"tile": 4})
        self.assertEqual(stack["kernels"][0]["candidate_id"], "kernel")
        self.assertEqual(original, original_revision("base"))

    def test_stale_and_no_change_candidates_are_refused(self):
        original = original_revision("base")
        with self.assertRaisesRegex(ContractError, "no longer"):
            candidate_revision(candidate("r1", parent="other"), original, "parameters")
        with self.assertRaisesRegex(ContractError, "does not change"):
            candidate_revision(candidate("r1", parameters={}), original, "parameters")

    def test_same_attention_evidence_routes_to_two_distinct_deliveries(self):
        profile = {
            "status": "ok", "report_ref": {"path": "profile/rank-summary.json"},
            "hotspots": [{"name": "flash_attention", "rank": 0}],
        }
        planned = [candidate("a1", operator="attention"), candidate("a2", operator="linear")]
        for backend, delivery in (("geak", "service_candidate"), ("forge", "operator_patch")):
            with self.subTest(backend=backend):
                handoff, items = plan_kernel_handoff(backend, original_revision("base"), {}, profile, planned)
                self.assertEqual(handoff["delivery"], delivery)
                self.assertEqual([item["id"] for item in items], ["a1"])
                self.assertEqual(items[0]["backend"], backend)
        self.assertEqual(plan_kernel_handoff("geak", original_revision("base"), {}, None, planned)[1], [])

    def test_real_kernel_routes_keep_verified_handoff_without_executing_simulated_tools(self):
        with TemporaryDirectory() as root:
            store = SessionStore.create(Path(root), "real", "native-route", {
                "task": {"workload": {"id": "fixed"}, "rules": {}}, "reference": {},
                "scenario": {"kernel_plans": {route: [candidate("mock-kernel", operator="swiglu")]
                                              for route in ("geak", "forge")}},
            })
            hotspot = {"name": "SwiGlu", "callsite": "sglang/srt/layers/activation.py:123"}
            reference = store.put_evidence("evidence/profile.json", {"hotspots": [hotspot]}, producer="trace_analysis")
            state = {"current_revision": original_revision("base"), "decisions": [], "profiles": [{
                "revision": "base", "status": "ok", "hotspots": [hotspot], "report_ref": reference,
            }]}
            backend = Mock()
            actions = RuntimeActions(store, backend)
            for route in ("geak", "forge"):
                with self.subTest(route=route):
                    result = actions.kernel({"task_id": route, "params": {"backend": route}}, state)
                    self.assertEqual(result["status"], "blocked")
                    self.assertEqual(result["reason"], "native_kernel_driver_unconfigured")
                    self.assertEqual(result["handoff"]["profile_ref"], reference)
                    self.assertEqual(result["handoff"]["source_correlated_hotspots"], [hotspot])
                    self.assertFalse(result["decisions"])
                    self.assertEqual(set(result["evidence"]), {"handoff"})
                    self.assertTrue(store.verify_evidence(result["evidence"]["handoff"]))
                    self.assertFalse((store.path / "evidence" / "kernel" / f"{route}-delivery.json").exists())
            self.assertFalse(backend.mock_calls)
