"""Exercise agent handoffs without a model server or a device."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hyperloop.agents import ReplayAgentBackend
from hyperloop.dispatcher import Dispatcher
from hyperloop.policy import PolicyGate
from hyperloop.router import IntentRouter

FRAMEWORK = "FRAMEWORK_AGENT"


class AgentRoutingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.events = []
        self.dispatcher = Dispatcher(emit=self._record)
        self.policy = PolicyGate(self.workspace.name, {"chunk_size"})
        self.router = IntentRouter(self.policy, self.dispatcher, self._record)

    def _record(self, kind: str, payload: dict) -> None:
        self.events.append((kind, payload))

    def _propose(self, proposal_id: str, **params) -> dict:
        return self.router.route(
            "specialist",
            FRAMEWORK,
            {"kind": "propose_action", "action": "integrate_patch", "proposal_id": proposal_id, "params": params},
        )

    def _review(self, proposal_id: str, verdict: str, **extra) -> dict:
        return self.router.route(
            "critic",
            FRAMEWORK,
            {"kind": "review_verdict", "proposal_id": proposal_id, "verdict": verdict, **extra},
        )

    def test_direct_explore_and_specialist_bypass_pending_review(self) -> None:
        explore = {
            "type": "delegate",
            "payload": {"action_name": "explore", "params": {"parameters": {"chunk_size": 16}}},
        }
        first = self.router.route("orchestration", FRAMEWORK, explore)
        specialist = self.router.route("orchestration", FRAMEWORK, {"kind": "delegate", "action": "specialist"})
        self.assertEqual(self.router.route("orchestration", FRAMEWORK, explore), first)
        self.assertEqual(self.router.pending_proposals, {})
        self.assertEqual([first["action"], specialist["action"]], ["explore", "specialist"])
        self.assertEqual([name for name, _ in self.events], ["task_queued", "task_queued"])

    def test_formal_landing_waits_for_review_and_keeps_handoff_ids(self) -> None:
        proposal = self._propose("source-fix", patch_path="patches/source.diff")
        self.assertEqual(proposal["status"], "pending")
        self.assertEqual(self.dispatcher.tasks, {})
        self.assertEqual(self.router.inbox_for("critic")[0]["proposal_id"], "source-fix")

        review = self._review("source-fix", "approve", reason="Measured on the active path")
        task = self.dispatcher.get(review["task_id"])
        self.assertEqual(task["proposal_id"], "source-fix")
        self.assertEqual(task["source"], "specialist")
        self.assertEqual(self.router.pending_proposals, {})
        self.assertEqual(
            [name for name, _ in self.events],
            ["proposal_registered", "review_verdict", "task_queued"],
        )
        self.assertEqual(self._review("source-fix", "approve")["code"], "proposal_not_pending")
        self.assertEqual(len(self.dispatcher.tasks), 1)

    def test_advice_materializes_reject_and_needs_review_return_feedback(self) -> None:
        self._propose("bad")
        self._propose("incomplete")
        self._propose("recommended")
        self.assertEqual(self._review("bad", "reject", reason="same as baseline")["status"], "reject")
        self.assertEqual(self._review("incomplete", "needs_review", reason="missing evidence")["status"], "needs_review")
        advised = self._review("recommended", "advise", reason="watch decoder regression")
        self.assertIn("task_id", advised)
        self.assertEqual(len(self.dispatcher.tasks), 1)
        self.assertEqual(
            [message["reason"] for message in self.router.inbox_for("specialist")],
            ["same as baseline", "missing evidence", "watch decoder regression"],
        )

    def test_policy_denies_internal_action_wrong_phase_and_unreviewed_landing(self) -> None:
        attempts = [
            ("orchestration", "KERNEL_AGENT", "kernel_agent", "coordinator_owned"),
            ("orchestration", "PRELUDE", "baseline", "coordinator_owned"),
            ("orchestration", "SWEEP", "conc_sweep", "coordinator_owned"),
            ("orchestration", "CLOSE", "report", "coordinator_owned"),
            ("orchestration", "CLOSE", "session_breakdown", "coordinator_owned"),
            ("orchestration", "PRELUDE", "explore", "phase_incompatible"),
            ("orchestration", FRAMEWORK, "integrate_patch", "review_required"),
            ("orchestration", "ENABLEMENT", "specialist", "coordinator_owned"),
        ]
        for role, phase, action, expected in attempts:
            with self.subTest(action=action, phase=phase):
                result = self.router.route(role, phase, {"kind": "delegate", "action": action})
                self.assertEqual(result["code"], expected)
        self.assertEqual(self.dispatcher.tasks, {})
        self.assertEqual(self.router.register_internal("KERNEL_AGENT", "kernel_agent", {})["status"], "queued")
        self.assertEqual(
            self.router.register_internal("ENABLEMENT", "specialist", {"mode": "enablement"})["action"],
            "specialist",
        )

    def test_policy_denies_disallowed_parameters_and_patch_paths(self) -> None:
        unsafe = [
            ("../other.diff", "patch_path_outside_workspace"),
            ("..\\other.diff", "patch_path_outside_workspace"),
            ("/abspath/patch.diff", "patch_path_outside_workspace"),
            ("C:\\abspath\\patch.diff", "patch_path_invalid"),
            ("sessions/evidence.diff", "patch_path_protected"),
            ("patches/../reference.diff", "patch_path_outside_workspace"),
        ]
        for path, code in unsafe:
            with self.subTest(path=path):
                self.assertEqual(self._propose("path-check", patch_path=path)["code"], code)
        result = self.router.route(
            "orchestration", FRAMEWORK, {"kind": "delegate", "action": "explore", "params": {"parameters": {"no": 1}}}
        )
        self.assertEqual(result["code"], "unsupported_parameter")
        self.assertEqual(self.router.pending_proposals, {})

    def test_symlink_cannot_redirect_a_patch_outside_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as outside:
            link = Path(self.workspace.name) / "linked"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(str(exc))
            self.assertEqual(
                self._propose("link-check", patch_path="linked/change.diff")["code"],
                "patch_path_outside_workspace",
            )

    def test_critic_empty_inbox_does_not_advance_replay_cursor(self) -> None:
        proposed = {"kind": "review_verdict", "proposal_id": "patch", "verdict": "approve"}
        backend = ReplayAgentBackend({"critic": [[proposed]], "orchestration": [[{"kind": "delegate", "action": "explore"}]]})
        self.assertEqual(backend.run("critic", {}), [])
        saved = backend.snapshot()
        self.assertEqual(backend.run("critic", {"pending_proposals": {"patch": {}}}), [proposed])
        self.assertEqual(backend.run("critic", {"pending_proposals": {"patch": {}}}), [])
        backend.restore(json.loads(json.dumps(saved)))
        self.assertEqual(backend.run("critic", {"pending_proposals": {"patch": {}}}), [proposed])

    def test_tasks_respect_budget_and_hold_lane_until_verified_cleanup(self) -> None:
        expensive = self.router.register_internal("PRELUDE", "baseline", {"estimated_seconds": 8})
        self.assertEqual(self.dispatcher.start_ready(phase="PRELUDE", remaining_budget=7), [])
        self.assertEqual(self.dispatcher.start_ready(phase="PRELUDE", remaining_budget=8)[0]["task_id"], expensive["task_id"])
        next_baseline = self.router.register_internal("PRELUDE", "baseline", {})
        self.assertEqual(next_baseline["estimated_seconds"], 1)

        unconfirmed = self.dispatcher.settle(expensive["task_id"], {"status": "ok"}, {"confirmed": False})
        self.assertEqual(unconfirmed["status"], "cleanup_pending")
        self.assertEqual(self.dispatcher.start_ready(phase="PRELUDE"), [])
        self.assertTrue(self.dispatcher.has_inflight)

        checkpoint = json.loads(json.dumps(self.dispatcher.snapshot(), sort_keys=True))
        recovered = Dispatcher()
        recovered.restore(checkpoint)
        self.assertEqual(recovered.start_ready(phase="PRELUDE"), [])
        interrupted = recovered.settle(expensive["task_id"], {"status": "interrupted"}, {"confirmed": True})
        self.assertEqual(interrupted["status"], "interrupted")
        self.assertEqual(recovered.start_ready(phase="PRELUDE")[0]["task_id"], next_baseline["task_id"])

    def test_task_result_feedback_and_pending_proposals_survive_restore(self) -> None:
        task = self.router.register_internal("PRELUDE", "baseline", {})
        self.dispatcher.start_ready()
        settled = self.dispatcher.settle(task["task_id"], {"status": "ok", "baseline": 100}, True)
        self.router.record_task_result(settled)
        self._propose("pending", patch_path="patches/source.diff")
        checkpoint = json.loads(json.dumps(self.router.snapshot()))
        recovered = IntentRouter(self.policy, Dispatcher())
        recovered.restore(checkpoint)
        self.assertEqual(recovered.inbox_for("orchestration")[0]["task_id"], task["task_id"])
        self.assertIn("pending", recovered.pending_proposals)
        self.assertEqual(recovered.inbox_for("critic")[0]["proposal_id"], "pending")
        self.assertEqual(settled, self.dispatcher.settle(task["task_id"], {"status": "ok", "baseline": 100}, True))


if __name__ == "__main__":
    unittest.main()
