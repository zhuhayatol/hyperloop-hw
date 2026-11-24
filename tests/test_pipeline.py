"""Acceptance tests for the recorded local-to-device optimization flow."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from hyperloop.backends import SimulatedBackend
from hyperloop.coordinator import Coordinator


PROJECT = Path(__file__).resolve().parents[1]
TARGETS = ((1, "target.910b3.simulated.json"), (2, "target.910b2-tp2.simulated.json"))


def load_fixture(path: str) -> dict:
    return json.loads((PROJECT / path).read_text(encoding="utf-8"))


def task_signature(events: list[dict]) -> list[tuple[str, int]]:
    return [
        (event["payload"]["phase"], event["payload"]["cycle"])
        for event in events
        if event["type"] == "phase_changed"
    ]


def decisions_signature(state: dict) -> list[tuple[str, str, str]]:
    return [
        (row["candidate_id"], row["parent"], row["version_action"])
        for row in state["decisions"]
    ]


class ProcessInterrupted(BaseException):
    """Simulate a process exit between a durable action receipt and writeback."""


class PipelineTests(TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "sessions"

    def coordinator(
        self,
        name: str,
        *,
        route: str = "forge",
        target: str = TARGETS[0][1],
        initial: str = "base",
        enablement: bool = True,
        scenario: dict | None = None,
    ) -> Coordinator:
        task = load_fixture("configs/task.simulated.json")
        task["settings"]["kernel_backend"] = route
        task["settings"]["enablement"] = enablement
        task["initial_revision"] = initial
        return Coordinator.create(
            self.root,
            name,
            task,
            load_fixture(f"configs/{target}"),
            load_fixture("fixtures/reference.simulated.json"),
            scenario if scenario is not None else load_fixture("fixtures/scenario.simulated.json"),
        )

    def assert_trace_and_evidence(self, coordinator: Coordinator, ranks: int) -> None:
        state = coordinator.store.state()
        self.assertGreater(len(state["profiles"]), 0)
        for profile in state["profiles"]:
            self.assertEqual(profile["status"], "ok")
            self.assertEqual(profile["ranks"], list(range(ranks)))
            self.assertEqual(profile["missing_ranks"], [])
            for rank in range(ranks):
                self.assertTrue(coordinator.store.verify_evidence(profile["evidence"][f"profile_rank_{rank}"]))
        for result in [state["baseline"], *state["decisions"]]:
            self.assertTrue(result["evidence"])
            for reference in result["evidence"].values():
                self.assertTrue(coordinator.store.verify_evidence(reference), reference)

    def test_both_kernel_routes_use_the_same_upstream_flow_across_targets(self) -> None:
        for route in ("forge", "geak"):
            paths = []
            fingerprints = []
            for ranks, target in TARGETS:
                with self.subTest(route=route, ranks=ranks):
                    coordinator = self.coordinator(f"{route}-tp{ranks}", route=route, target=target)
                    coordinator.run()
                    state = coordinator.store.state()
                    events = coordinator.store.events()
                    kernel = f"K-{route.upper()}"
                    self.assertEqual(state["phase"], "DONE")
                    self.assertEqual(
                        decisions_signature(state),
                        [
                            ("R01", "base", "KEEP"),
                            ("R02", "R01", "KEEP"),
                            (kernel, "R02", "KEEP"),
                            ("R03", kernel, "REVERT"),
                        ],
                    )
                    self.assertEqual(
                        [row["candidate_id"] for row in state["current_revision"]["framework"]], ["R01", "R02"]
                    )
                    self.assertEqual(
                        [row["candidate_id"] for row in state["current_revision"]["kernels"]], [kernel]
                    )
                    self.assertEqual(state["current_revision"]["id"], kernel)
                    self.assertTrue(coordinator.store.verify_evidence(state["current_revision"]["validation_ref"]))
                    candidate = state["decisions"][-1]
                    self.assertEqual((candidate["evidence_state"], candidate["acceptance"]), ("sufficient", "fail"))
                    self.assertAlmostEqual(candidate["comparison"]["over_parent"]["throughput_pct"], (123 / 122 - 1) * 100)
                    self.assertAlmostEqual(candidate["comparison"]["over_original"]["throughput_pct"], 23)
                    measured_stack = candidate["measurement"]["loaded_stack"]
                    self.assertEqual([row["candidate_id"] for row in measured_stack["framework"]], ["R01", "R02"])
                    self.assertEqual(
                        [row["candidate_id"] for row in measured_stack["kernels"]], [kernel, "R03"]
                    )
                    self.assertEqual(coordinator.store.path.parent.name, "simulated")
                    self.assertEqual(coordinator.store.manifest["mode"], "simulated")
                    self.assertEqual(state["baseline"]["revision"]["id"], "base")
                    self.assertAlmostEqual(state["baseline"]["measurement"]["summary"]["throughput"], 100)
                    completed = next(sweep for sweep in state["sweeps"] if sweep["status"] == "ok")
                    self.assertEqual([pair["concurrency"] for pair in completed["pairs"]], [1, 4, 8])
                    for pair in completed["pairs"]:
                        original = pair["original"]["measurement"]
                        accepted = pair["candidate"]["measurement"]
                        self.assertEqual((original["revision"], accepted["revision"]), ("base", kernel))
                        self.assertEqual(original["workload_fingerprint"], accepted["workload_fingerprint"])
                        self.assertEqual(original["environment_fingerprint"], accepted["environment_fingerprint"])
                        self.assertAlmostEqual(pair["comparison"]["throughput_pct"], 22)
                    self.assertTrue(coordinator.store.verify_evidence(completed["evidence"]["paired_sweep"]))
                    self.assertEqual(state["sweeps"][-1]["reason"], "no_validated_change_since_sweep")
                    self.assertTrue(state["stop_reason"])
                    self.assert_trace_and_evidence(coordinator, ranks)
                    phase_path = task_signature(events)
                    phases = [phase for phase, _ in phase_path if phase != "PRELUDE"]
                    self.assertEqual(phases[:3], ["FRAMEWORK_AGENT", "KERNEL_AGENT", "SWEEP"])
                    self.assertEqual(phases[-2:], ["CLOSE", "DONE"])
                    self.assertIn(("FRAMEWORK_AGENT", 2), phase_path)
                    kinds = {event["type"] for event in events}
                    self.assertTrue({"task_started", "task_settled", "candidate_decision"}.issubset(kinds))
                    explore = next(
                        event for event in events
                        if event["type"] == "task_queued" and event["payload"]["action"] == "explore"
                    )
                    self.assertIsNone(explore["payload"]["proposal_id"])
                    approval = next(
                        event for event in events
                        if event["type"] == "review_verdict" and event["payload"]["proposal_id"] == "review-R02"
                    )
                    patch_task = next(
                        event for event in events
                        if event["type"] == "task_started" and event["payload"]["action"] == "integrate_patch"
                    )
                    self.assertEqual(patch_task["payload"]["proposal_id"], "review-R02")
                    self.assertLess(approval["seq"], patch_task["seq"])
                    fingerprints.append(state["baseline"]["measurement"]["environment_fingerprint"])
                    paths.append((coordinator.store.manifest["task"], phase_path, decisions_signature(state)))

                    report_path = coordinator.store.path
                    summary = json.loads((report_path / "summary.json").read_text(encoding="utf-8"))
                    self.assertEqual(summary["state_event_seq"], events[-1]["seq"])
                    self.assertEqual(summary["current_revision"]["id"], kernel)
                    self.assertEqual(summary["stop_reason"], state["stop_reason"])
                    html = (report_path / "report.html").read_text(encoding="utf-8")
                    markdown = (report_path / "report.md").read_text(encoding="utf-8")
                    self.assertIn('class="mode">simulated', html)
                    self.assertIn("相对原始基线", markdown)
                    self.assertIn(candidate["evidence"]["candidate_launch"]["path"], html)
            if len(paths) == len(TARGETS):
                self.assertEqual(paths[0], paths[1])
                self.assertNotEqual(fingerprints[0], fingerprints[1])

    def test_checkpoint_resume_does_not_replace_baseline_or_repeat_committed_keep(self) -> None:
        coordinator = self.coordinator("checkpoint")
        for _ in range(40):
            coordinator.run(max_steps=1)
            if coordinator.store.state()["decisions"]:
                break
        checkpoint = coordinator.store.state()
        self.assertEqual(decisions_signature(checkpoint), [("R01", "base", "KEEP")])
        self.assertEqual(checkpoint["current_revision"]["id"], "R01")
        baseline = checkpoint["baseline"]
        prefix = coordinator.store.events()
        reopened = Coordinator.open(coordinator.store.path)
        reopened.run()
        final = reopened.store.state()
        replayed = reopened.store.events()
        self.assertEqual(final["baseline"], baseline)
        self.assertEqual(replayed[: len(prefix)], prefix)
        self.assertEqual([event["seq"] for event in replayed], list(range(1, len(replayed) + 1)))

        uninterrupted = self.coordinator("uninterrupted")
        uninterrupted.run()
        control = uninterrupted.store.state()
        self.assertEqual(final["phase"], "DONE")
        self.assertEqual(final["stop_reason"], control["stop_reason"])
        self.assertEqual(decisions_signature(final), decisions_signature(control))
        self.assertEqual(final["current_revision"]["id"], control["current_revision"]["id"])

    def test_resume_consumes_receipt_without_rerunning_the_effect(self) -> None:
        coordinator = self.coordinator("receipt")
        finish = Coordinator._finish_task
        interrupted_receipts: list[Path] = []

        def interrupt_after_explore(instance, *args, **kwargs):
            receipts = coordinator.store.path / "evidence" / "actions"
            for path in receipts.glob("*.json"):
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload["action"] == "explore":
                    interrupted_receipts.append(path)
                    raise ProcessInterrupted(path.name)
            return finish(instance, *args, **kwargs)

        with patch.object(Coordinator, "_finish_task", interrupt_after_explore):
            with self.assertRaises(ProcessInterrupted):
                coordinator.run()
        self.assertEqual(len(interrupted_receipts), 1)
        receipt_path = interrupted_receipts[0]
        receipt_bytes = receipt_path.read_bytes()
        receipt = json.loads(receipt_bytes)
        checkpoint = coordinator.store.state()
        self.assertEqual(checkpoint["current_revision"]["id"], "base")
        self.assertFalse(checkpoint["decisions"])
        raw_before = {
            path.relative_to(coordinator.store.path): path.read_bytes()
            for path in (coordinator.store.path / "evaluation").rglob("*.json")
        }
        self.assertTrue(raw_before)

        reopened = Coordinator.open(coordinator.store.path)
        prepare = SimulatedBackend.prepare

        def guard_replayed_prepare(adapter, revision, workspace, *, kind):
            if revision == "R01" and workspace.name == "candidate":
                self.fail("receipt recovery attempted to evaluate R01 again")
            return prepare(adapter, revision, workspace, kind=kind)

        with patch.object(SimulatedBackend, "prepare", guard_replayed_prepare):
            reopened.run()
        state = reopened.store.state()
        self.assertEqual(receipt_path.read_bytes(), receipt_bytes)
        self.assertEqual(state["phase"], "DONE")
        self.assertEqual(state["baseline"], checkpoint["baseline"])
        self.assertEqual(
            [(row["candidate_id"], row["version_action"]) for row in state["decisions"] if row["candidate_id"] == "R01"],
            [("R01", "KEEP")],
        )
        self.assertEqual(
            sum(
                event["type"] == "task_settled" and event["payload"]["task_id"] == receipt["task_id"]
                for event in reopened.store.events()
            ),
            1,
        )
        for path, raw in raw_before.items():
            self.assertEqual((reopened.store.path / path).read_bytes(), raw, path)

    def test_enablement_preserves_distinction_between_failed_start_and_effective_baseline(self) -> None:
        repaired = self.coordinator("repaired", initial="base-unavailable")
        repaired.run()
        state = repaired.store.state()
        self.assertEqual(state["phase"], "DONE")
        self.assertIn(("ENABLEMENT", 1), task_signature(repaired.store.events()))
        self.assertEqual(repaired.store.manifest["task"]["initial_revision"], "base-unavailable")
        self.assertEqual(state["baseline"]["revision"]["id"], "base-fixed")
        self.assertEqual(state["decisions"][0]["parent"], "base-fixed")

        disabled = self.coordinator("enablement-off", initial="base-unavailable", enablement=False)
        disabled.run()
        state = disabled.store.state()
        self.assertEqual(state["phase"], "DONE")
        self.assertIsNone(state["baseline"])
        self.assertIsNone(state["current_revision"])
        self.assertFalse(state["decisions"])
        self.assertFalse(state["profiles"])
        self.assertTrue(state["stop_reason"])
        self.assertTrue((disabled.store.path / "report.html").is_file())

    def test_incorrect_kernel_never_reaches_performance_gate(self) -> None:
        scenario = load_fixture("fixtures/scenario.simulated.json")
        scenario["revisions"]["R03"]["faults"] = ["wrong_output"]
        coordinator = self.coordinator("incorrect-kernel", scenario=scenario)
        coordinator.run()
        state = coordinator.store.state()
        failed = next(decision for decision in state["decisions"] if decision["candidate_id"] == "R03")
        self.assertEqual(state["current_revision"]["id"], "K-FORGE")
        self.assertEqual(failed["reason"], "kernel_correctness_failed")
        self.assertEqual(failed["version_action"], "REVERT")
        self.assertFalse(failed["kernel_correctness"]["pass"])
        self.assertNotIn("measurement", failed)
        self.assertNotIn("device_timing", failed)
        self.assertTrue(coordinator.store.verify_evidence(failed["evidence"]["operator_kernel_correctness"]))

    def test_close_remeasures_the_full_accepted_stack_when_its_evidence_is_unavailable(self) -> None:
        coordinator = self.coordinator("close-stack")
        for _ in range(100):
            coordinator.run(max_steps=1)
            if coordinator.store.state()["phase"] == "CLOSE":
                break
        checkpoint = coordinator.store.state()
        self.assertEqual(checkpoint["phase"], "CLOSE")
        accepted = checkpoint["current_revision"]
        self.assertEqual(accepted["id"], "K-FORGE")
        self.assertFalse(checkpoint["stack_validations"])
        evidence = coordinator.store.artifact_path(accepted["validation_ref"]["path"])
        evidence.unlink()

        reopened = Coordinator.open(coordinator.store.path)
        reopened.run()
        state = reopened.store.state()
        self.assertEqual(state["phase"], "DONE")
        self.assertEqual(state["baseline"], checkpoint["baseline"])
        self.assertEqual(state["current_revision"]["id"], accepted["id"])
        self.assertEqual(len(state["stack_validations"]), 1)
        validation = state["stack_validations"][0]
        self.assertEqual(validation["reason_context"], "close_stack_unverified")
        self.assertEqual(validation["acceptance"], "pass")
        self.assertAlmostEqual(validation["comparison"]["throughput_pct"], 22)
        self.assertEqual(validation["measurement"]["loaded_stack"]["framework"], accepted["framework"])
        self.assertEqual(validation["measurement"]["loaded_stack"]["kernels"], accepted["kernels"])
        self.assertTrue(reopened.store.verify_evidence(validation["evidence"]["revision_measurement"]))
