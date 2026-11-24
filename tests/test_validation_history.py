"""Check that historical validation flow cites only persisted evidence."""

from __future__ import annotations

from unittest import TestCase

from hyperloop.validation_history import reconstruct_validation_flow


def references(*names: str) -> dict[str, dict]:
    return {name: {"path": f"evaluation/attempt/{name}.json"} for name in names}


def ordered_keys(rows: list[dict]) -> list[str]:
    return [key for row in rows for key in row["evidence_keys"]]


class ValidationHistoryTests(TestCase):
    def test_orders_framework_validation_checks_and_raw_samples_by_execution(self) -> None:
        records = references(
            "parent_repeat_1", "candidate_repeat_10", "critic_assessment", "workload",
            "candidate_service_correctness", "candidate_prepared", "candidate_cleanup",
            "candidate_measurement", "paired_comparison", "policy", "parent_prepared",
            "parent_cleanup", "candidate_launch", "reference", "candidate_repeat_2",
            "parent_service_correctness", "parent_service_outputs", "parent_launch",
            "candidate_service_outputs", "probe", "parent_measurement", "decision",
        )
        rows = reconstruct_validation_flow(
            records, result={"status": "ok", "correctness": {"pass": True}, "parent_correctness": {"pass": True}},
        )
        self.assertEqual(ordered_keys(rows), [
            "workload", "policy", "reference", "probe",
            "candidate_prepared", "candidate_launch", "candidate_service_outputs",
            "candidate_service_correctness", "candidate_repeat_2", "candidate_repeat_10",
            "candidate_measurement", "candidate_cleanup", "parent_prepared", "parent_launch",
            "parent_service_outputs", "parent_service_correctness", "parent_repeat_1",
            "parent_measurement", "parent_cleanup",
        ])
        self.assertTrue(
            all(row["state"] == "succeeded" for row in rows if row["operation"] == "correctness")
        )
        self.assertNotIn("requested", {row["state"] for row in rows})

    def test_kernel_device_stage_comes_before_service_and_supports_separate_tool_receipts(self) -> None:
        records = references(
            "device_timing", "candidate_service_outputs", "candidate_launch", "operator_kernel_outputs",
            "kernel_reference", "operator_kernel_correctness", "candidate_repeat_0", "device_timing_raw",
        )
        rows = reconstruct_validation_flow(records, result={"kernel_correctness": {"pass": True}})
        self.assertEqual(ordered_keys(rows), [
            "candidate_launch", "kernel_reference", "operator_kernel_outputs", "operator_kernel_correctness",
            "device_timing_raw", "device_timing", "candidate_service_outputs", "candidate_repeat_0",
        ])
        self.assertEqual(next(row for row in rows if row["operation"] == "device_timer")["scope"], "kernel")

        tool = references("device_timing_raw", "kernel_launch", "publication", "kernel_cleanup")
        rows = reconstruct_validation_flow(tool, result={})
        self.assertEqual(ordered_keys(rows), ["kernel_launch", "device_timing_raw", "kernel_cleanup", "publication"])
        self.assertTrue(all(row["scope"] == "kernel" for row in rows))

    def test_profile_displays_recorded_ranks_without_inventing_the_missing_rank(self) -> None:
        records = references("profile_analysis", "profile_rank_10", "profile_rank_2", "workload", "profile_raw")
        rows = reconstruct_validation_flow(records, result={"missing_ranks": [1]})
        self.assertEqual(
            ordered_keys(rows), ["workload", "profile_raw", "profile_rank_2", "profile_rank_10", "profile_analysis"],
        )
        self.assertEqual(rows[-1]["state"], "failed")
        self.assertNotIn("Reference", rows[0]["message"])

    def test_failed_candidate_shows_no_synthetic_timing_or_parent_stages(self) -> None:
        records = references(
            "candidate_prepared", "candidate_launch", "candidate_service_outputs",
            "candidate_service_correctness", "candidate_cleanup", "decision",
        )
        rows = reconstruct_validation_flow(
            records, result={"correctness": {"pass": False}, "reason": "service_correctness_failed"},
        )
        self.assertEqual(ordered_keys(rows), [name for name in records if name != "decision"])
        self.assertEqual(next(row for row in rows if row["operation"] == "correctness")["state"], "failed")
        self.assertTrue(all(row["scope"] == "candidate" for row in rows))

    def test_backend_failure_keeps_only_available_stage_and_cleanup(self) -> None:
        records = references("revision_prepared", "revision_cleanup", "backend_failure", "decision")
        rows = reconstruct_validation_flow(records, result={"reason": "cleanup_unconfirmed"})
        self.assertEqual(ordered_keys(rows), ["revision_prepared", "revision_cleanup", "backend_failure"])
        self.assertEqual(rows[1]["state"], "failed")
        self.assertEqual(rows[2]["state"], "failed")

    def test_forward_and_stack_keep_their_own_timing_scopes(self) -> None:
        rows = reconstruct_validation_flow(references("forward_timing", "forward_raw", "forward_service_correctness"),
                                           result={"correctness": {"pass": True}})
        self.assertEqual(ordered_keys(rows), ["forward_service_correctness", "forward_raw", "forward_timing"])
        self.assertTrue(all(row["scope"] == "forward" for row in rows))
        rows = reconstruct_validation_flow(references("revision_repeat_0", "revision_measurement"), result={})
        self.assertEqual(ordered_keys(rows), ["revision_repeat_0", "revision_measurement"])
        self.assertTrue(all(row["scope"] == "revision" for row in rows))

    def test_only_final_decision_refs_do_not_imply_a_validation_flow(self) -> None:
        self.assertEqual(
            reconstruct_validation_flow(references("decision", "critic_assessment", "proposal", "paired_comparison"),
                                        result={"status": "ok"}),
            [],
        )
