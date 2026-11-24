import unittest

from hyperloop.actions.candidates import candidate_revision, original_revision, stack_identity
from hyperloop.contracts import ContractError
from hyperloop.kernel.routes import plan_kernel_handoff


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
