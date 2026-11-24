"""End-to-end checks for interrupted and partially observed Sessions."""

from __future__ import annotations

import json
from unittest import TestCase
from unittest.mock import patch

from hyperloop.backends import SimulatedBackend
from hyperloop.backends.runtime import BackendError
from hyperloop.coordinator import Coordinator

import test_pipeline as pipeline


class PipelineEdgeTests(TestCase):
    setUp = pipeline.PipelineTests.setUp
    coordinator = pipeline.PipelineTests.coordinator

    @staticmethod
    def events(coordinator: Coordinator, kind: str) -> list[dict]:
        return [event for event in coordinator.store.events() if event["type"] == kind]

    def test_interruption_before_receipt_retries_with_a_new_task(self) -> None:
        coordinator = self.coordinator("before-receipt")
        benchmark = SimulatedBackend.benchmark

        def interrupted_benchmark(adapter, instance, workload, *, repeat, warmup):
            if instance["revision"] == "R01":
                raise pipeline.ProcessInterrupted("simulated exit during first measurements")
            return benchmark(adapter, instance, workload, repeat=repeat, warmup=warmup)

        with patch.object(SimulatedBackend, "benchmark", interrupted_benchmark):
            with self.assertRaises(pipeline.ProcessInterrupted):
                coordinator.run()
        checkpoint = coordinator.store.state()
        old = next(task for task in checkpoint["tasks"].values() if task["action"] == "explore")
        old_id = old["task_id"]
        self.assertEqual(old["status"], "running")
        self.assertFalse(checkpoint["decisions"])
        directory = coordinator.store.path
        old_receipt = directory / "evidence" / "actions" / f"{old_id}.json"
        self.assertFalse(old_receipt.exists())
        partial = {
            path: path.read_bytes() for path in (directory / "evaluation").glob(f"{old_id}-*/candidate_*.json")
        }
        self.assertTrue(partial)

        reopened = Coordinator.open(directory)
        reopened.run()
        state = reopened.store.state()
        self.assertEqual(state["phase"], "DONE")
        self.assertEqual(state["baseline"], checkpoint["baseline"])
        self.assertFalse(old_receipt.exists())
        self.assertTrue(all(path.read_bytes() == contents for path, contents in partial.items()))
        old_settlements = [
            event["payload"] for event in self.events(reopened, "task_settled")
            if event["payload"]["task_id"] == old_id
        ]
        self.assertEqual(len(old_settlements), 1)
        self.assertEqual(old_settlements[0]["result"]["reason"], "interrupted_before_action_receipt")
        retries = [task for task in state["tasks"].values() if task["params"].get("retry_of") == old_id]
        self.assertEqual(len(retries), 1)
        trial = [row for row in state["decisions"] if row["candidate_id"] == "R01"]
        self.assertEqual([(row["task_id"], row["version_action"]) for row in trial], [(retries[0]["task_id"], "KEEP")])
        self.assertTrue(reopened.store.verify_evidence(trial[0]["receipt"]))

    def test_unconfirmed_cleanup_defers_settlement_and_promotion(self) -> None:
        scenario = pipeline.load_fixture("fixtures/scenario.simulated.json")
        scenario["revisions"]["R01"]["faults"] = ["cleanup"]
        coordinator = self.coordinator("pending-cleanup", scenario=scenario)
        coordinator.run()
        checkpoint = coordinator.store.state()
        old = next(task for task in checkpoint["tasks"].values() if task["action"] == "explore")
        old_id = old["task_id"]
        self.assertEqual(old["status"], "cleanup_pending")
        self.assertEqual(checkpoint["current_revision"]["id"], "base")
        self.assertFalse(checkpoint["decisions"])
        events_before = coordinator.store.events()

        coordinator.run(max_steps=3)
        self.assertEqual(coordinator.store.events(), events_before)
        reopened = Coordinator.open(coordinator.store.path)
        reopened.run(max_steps=1)
        recovered = reopened.store.state()
        self.assertEqual(recovered["current_revision"]["id"], "base")
        self.assertEqual(pipeline.decisions_signature(recovered), [("R01", "base", "HOLD")])
        self.assertEqual(recovered["decisions"][0]["reason"], "cleanup_unconfirmed")
        release = next(event["payload"] for event in self.events(reopened, "task_recovered"))
        self.assertEqual(release["task_id"], old_id)
        self.assertTrue(release["cleanup"]["released"])
        self.assertTrue(reopened.store.verify_evidence(release["receipt"]))
        self.assertEqual(
            sum(event["payload"]["task_id"] == old_id for event in self.events(reopened, "task_settled")), 1
        )

    def test_partial_sweep_preserves_first_pair_and_closes(self) -> None:
        coordinator = self.coordinator("partial-sweep")
        benchmark = SimulatedBackend.benchmark

        def fail_second_candidate_arm(adapter, instance, workload, *, repeat, warmup):
            if workload["id"] == "prefill-decode-c4" and instance["revision"] == "K-FORGE":
                raise BackendError("simulated failure in the candidate arm at c4")
            return benchmark(adapter, instance, workload, repeat=repeat, warmup=warmup)

        with patch.object(SimulatedBackend, "benchmark", fail_second_candidate_arm):
            coordinator.run()
        state = coordinator.store.state()
        self.assertEqual(state["phase"], "DONE")
        self.assertEqual(state["stop_reason"], "sweep_failed_or_incomplete")
        self.assertEqual(state["current_revision"]["id"], "K-FORGE")
        result = state["last_conc_sweep"]
        self.assertEqual(result, state["sweeps"][-1])
        self.assertEqual(result["status"], "failed")
        self.assertEqual([pair["concurrency"] for pair in result["pairs"]], [1, 4])
        first, second = result["pairs"]
        self.assertAlmostEqual(first["comparison"]["throughput_pct"], 22)
        self.assertEqual(first["original"]["acceptance"], "pass")
        self.assertEqual(first["candidate"]["acceptance"], "pass")
        self.assertEqual(second["original"]["acceptance"], "pass")
        self.assertEqual(second["candidate"]["status"], "failed")
        self.assertNotIn("comparison", second)
        self.assertTrue(coordinator.store.verify_evidence(result["evidence"]["paired_sweep"]))
        self.assertEqual(pipeline.task_signature(coordinator.store.events())[-2:], [("CLOSE", 1), ("DONE", 1)])

    def test_profile_retains_rank_metadata_and_blocks_kernel_on_missing_rank(self) -> None:
        target = pipeline.TARGETS[1][1]
        complete = self.coordinator("two-ranks", target=target)
        for _ in range(20):
            state = complete.run(max_steps=1)
            self.assertNotEqual(state["phase"], "DONE")
            if state["profiles"]:
                break
        self.assertTrue(state["profiles"])
        profile = state["profiles"][0]
        self.assertEqual(profile["status"], "ok")
        capture = profile["capture"]
        self.assertTrue(capture["tool_version"])
        self.assertEqual(capture["clock_domain"], "per_rank")
        self.assertLess(*capture["window_us"])
        self.assertEqual([item["rank"] for item in profile["hotspots"]], [0, 1])
        for rank in (0, 1):
            raw_ref = profile["evidence"][f"profile_rank_{rank}"]
            self.assertTrue(complete.store.verify_evidence(raw_ref))
            raw = json.loads(complete.store.artifact_path(raw_ref["path"]).read_text(encoding="utf-8"))
            self.assertEqual(raw["rank"], rank)
            self.assertTrue(raw["events"][0]["input_shape"])
            self.assertTrue(profile["hotspots"][rank]["input_shape"])

        scenario = pipeline.load_fixture("fixtures/scenario.simulated.json")
        for revision in scenario["revisions"].values():
            revision.setdefault("faults", []).append("profile_missing_rank")
        partial = self.coordinator("rank-missing", target=target, scenario=scenario)
        partial.run()
        state = partial.store.state()
        self.assertEqual(state["phase"], "DONE")
        self.assertTrue(state["profiles"])
        for capture in state["profiles"]:
            self.assertEqual(capture["status"], "partial")
            self.assertEqual(capture["missing_ranks"], [1])
            self.assertTrue(partial.store.verify_evidence(capture["evidence"]["profile_rank_0"]))
        self.assertEqual(state["current_revision"]["kernels"], [])
        self.assertFalse(any(row["kind"] == "kernel" for row in state["decisions"]))
        kernel = [
            event["payload"]["result"] for event in self.events(partial, "task_settled")
            if event["payload"]["action"] == "kernel_agent"
        ]
        self.assertTrue(kernel)
        self.assertEqual(kernel[0]["handoff"]["reason"], "no_usable_profile")

    def test_empty_critic_response_is_retried_for_pending_proposal(self) -> None:
        scenario = pipeline.load_fixture("fixtures/scenario.simulated.json")
        scenario["agents"]["critic"].insert(0, [])
        coordinator = self.coordinator("empty-critic-first", scenario=scenario)
        coordinator.run()
        turns = [
            event for event in self.events(coordinator, "agent_turn_completed") if event["payload"]["role"] == "critic"
        ]
        self.assertGreaterEqual(len(turns), 2)
        self.assertEqual([event["payload"]["response_count"] for event in turns[:2]], [0, 1])
        approval = next(
            event for event in self.events(coordinator, "review_verdict")
            if event["payload"]["proposal_id"] == "review-R02"
        )
        self.assertLess(turns[0]["seq"], approval["seq"])
        self.assertFalse(any(event["seq"] < approval["seq"] for event in self.events(coordinator, "critic_skipped")))
        revision = next(row for row in coordinator.store.state()["decisions"] if row["candidate_id"] == "R02")
        self.assertEqual(revision["version_action"], "KEEP")
