from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from hyperloop.evaluation.evidence import AttemptEvidence
from hyperloop.report import write_report
from hyperloop.session import SessionStore


class ValidationGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = SessionStore.create(Path(directory.name), "simulated", "graph", {})

    def report(self):
        return write_report(self.store.path, self.store.manifest, self.store.state(), self.store.events())

    def test_candidate_graph_connects_proposal_runtime_trace_and_writeback(self) -> None:
        timing = self.store.put_evidence("evidence/sample.json", {"raw": True}, producer="runner")
        steps = [
            {"sender": "BenchmarkRunner", "receiver": "RuntimeBackend", "operation": "benchmark_sample",
             "message": "发送冻结负载", "state": "requested"},
            {"sender": "RuntimeBackend", "receiver": "BenchmarkRunner", "operation": "benchmark_sample",
             "message": "返回逐请求样本", "state": "succeeded", "evidence_keys": ["candidate_repeat_0"]},
        ]
        flow = self.store.put_evidence(
            "evidence/flow.json", {"schema_version": 1, "attempt_id": "task-0001", "mode": "simulated", "steps": steps},
            producer="evaluation_trace",
        )
        run = {"attempt_id": "task-0001", "candidate_id": "R01", "status": "ok",
               "evidence": {"candidate_repeat_0": timing, "validation_flow": flow}}
        receipt = self.store.put_evidence(
            "evidence/actions/task-0001.json", {"task_id": "task-0001", "action": "integrate_patch",
                                               "result": {"status": "ok", "decisions": [run]}},
            producer="action_executor",
        )
        task = {"task_id": "task-0001", "action": "integrate_patch", "source": "specialist", "proposal_id": "P01"}
        self.store.commit([
            ("proposal_registered", {"proposal_id": "P01", "source": "specialist"}),
            ("review_verdict", {"proposal_id": "P01", "verdict": "approve"}),
            ("task_started", task),
            ("task_settled", {**task, "result": {"receipt": receipt}}),
            ("delegated_result", task),
            ("critic_measured_verdict", {"task_id": "task-0001", "candidate_id": "R01", "version_action": "REVERT"}),
            ("candidate_decision", {"decision": {**run, "task_id": "task-0001", "version_action": "REVERT"}}),
        ])

        graph, = self.report()["validation_graphs"]
        attempt, = graph["attempts"]
        self.assertEqual(attempt["origin"], "instrumented")
        self.assertEqual([row["sender"] for row in attempt["steps"]], ["BenchmarkRunner", "RuntimeBackend"])
        self.assertEqual(attempt["steps"][1]["evidence"], [timing])
        self.assertEqual([row.get("event_seq") for row in graph["first"]], [1, 2, 3])
        self.assertEqual(graph["last"][-1]["receiver"], "Session")
        html = (self.store.path / "report.html").read_text("utf-8")
        self.assertIn('href="#validation-task-1-attempt-1"', html)
        self.assertIn('href="evidence/sample.json"', html)
        self.assertNotIn("<script>", html)

    def test_historical_steps_have_artifacts_and_skip_has_no_attempt(self) -> None:
        probe = self.store.put_evidence("evidence/probe.json", {"mode": "simulated"}, producer="runtime_backend")
        task = {"task_id": "task-0001", "action": "profile"}
        self.store.commit([
            ("task_started", task),
            ("profile_recorded", {**task, "attempt_id": "task-0001", "evidence": {"probe": probe}}),
            ("sweep_recorded", {"status": "skipped", "reason": "disabled", "pairs": []}),
        ])
        profile, sweep = self.report()["validation_graphs"]
        attempt, = profile["attempts"]
        self.assertEqual(attempt["origin"], "reconstructed")
        self.assertTrue(attempt["steps"])
        self.assertTrue(all(row["evidence"] for row in attempt["steps"]))
        self.assertFalse(sweep["attempts"])
        self.assertEqual(sweep["last"][0]["state"], "skipped")

    def test_native_geak_delivery_links_tool_trials_before_independent_validation(self) -> None:
        self.store = SessionStore.create(self.store.path.parents[1], "real", "native-geak", {})

        def trial(attempt_id, revision, raw_key):
            evidence = AttemptEvidence(self.store.path, attempt_id)
            evidence.save(raw_key, {"revision": revision}, "evaluation")
            evidence.save("validation_flow", {"schema_version": 1, "attempt_id": attempt_id, "steps": [{
                "sender": "RuntimeBackend", "receiver": "Evaluator", "message": "返回实际样本",
                "operation": "read", "state": "succeeded", "evidence_keys": [raw_key],
            }]}, "evaluation_trace")
            return {"attempt_id": attempt_id, "revision": revision, "status": "ok", "evidence": evidence.records}

        parent = trial("task-0001-geak-parent", "base", "revision_measurement")
        parent["launch_control"] = {"ASCEND_USE_FIA": "0"}
        candidate = trial("task-0001-geak-candidate", "candidate", "revision_measurement")
        candidate["launch_control"] = {"ASCEND_USE_FIA": "1"}
        candidate["activation"] = {**trial("task-0001-geak-candidate-activation", "candidate", "profile_analysis"),
                                   "expected_operator": "fusedinferattentionscore"}
        independent = trial("task-0001-independent", "candidate", "candidate_measurement")
        self._record_native_delivery("geak", "service_candidate", {
            "status": "ok", "reason": "service_candidate_ready_for_review",
            "parent_trial": parent, "candidate_trial": candidate,
        }, decisions=[{**independent, "candidate_id": "candidate"}])

        graph, = self.report()["validation_graphs"]
        self.assertEqual([row["attempt_id"] for row in graph["attempts"]], [
            "task-0001-geak-parent", "task-0001-geak-candidate", "task-0001-geak-candidate-activation",
            "task-0001-independent",
        ])
        self.assertTrue(all(attempt["origin"] == "instrumented" for attempt in graph["attempts"]))
        self.assertEqual(graph["first"][-1]["receiver"], "昇腾 GEAK")
        self.assertEqual(graph["attempts"][1]["steps"][0]["message"], "candidate · ASCEND_USE_FIA=1")
        self.assertEqual(graph["attempts"][2]["after"][-1]["message"], "服务候选及工具测量")
        html = (self.store.path / "report.html").read_text("utf-8")
        self.assertIn('href="evaluation/task-0001-geak-candidate-activation/profile_analysis.json"', html)

    def test_native_forge_failure_renders_worker_and_publication_without_a_decision(self) -> None:
        self.store = SessionStore.create(self.store.path.parents[1], "real", "native-forge", {})
        self._record_native_delivery("forge", "operator_patch", {
            "status": "failed", "reason": "forge_worker_timeout",
            "evidence": {"worker_process": {"exit_code": -9, "timed_out": True, "process_group_clear": True}},
        }, decisions=[])

        graph, = self.report()["validation_graphs"]
        worker, = graph["attempts"]
        self.assertEqual(worker["steps"][0]["receiver"], "Triton-Ascend Worker")
        self.assertEqual(worker["steps"][1]["state"], "failed")
        self.assertIn("-9", worker["steps"][1]["message"])
        self.assertEqual(worker["after"][0]["message"], "工具失败回执")
        self.assertEqual(worker["after"][0]["receiver"], "算子执行流程")
        self.assertTrue(worker["after"][0]["evidence"])

    def _record_native_delivery(self, backend, delivery, result, *, decisions):
        task = {"task_id": "task-0001", "action": "kernel_agent", "source": "coordinator"}
        handoff = self.store.put_evidence("evidence/kernel/task-0001-handoff.json", {"backend": backend},
                                          producer="kernel_handoff")
        publication = self.store.put_evidence("evidence/kernel/task-0001-delivery.json", {
            **result, "mode": "real", "backend": backend, "delivery": delivery,
        }, producer=f"native_{backend}")
        receipt = self.store.put_evidence("evidence/actions/task-0001.json", {
            **task, "result": {"status": result["status"], "reason": result["reason"], "decisions": decisions,
                               "evidence": {"handoff": handoff, "delivery": publication}},
        }, producer="action_executor")
        self.store.commit([
            ("task_started", task),
            ("task_settled", {**task, "result": {"receipt": receipt, "evidence": {"delivery": publication}}}),
            *(("candidate_attempt", {**decision, "task_id": task["task_id"]}) for decision in decisions),
        ])

    def test_detached_reproduction_appears_without_modifying_events(self) -> None:
        data = {"candidate_id": "R03", "status": "ok", "version_action": "REVERT",
                "mode": "simulated", "attempt_id": "reproduce-0123456789ab"}
        self.store.put_evidence(
            "evidence/reproductions/reproduce-0123456789ab.json",
            {"session_fingerprint": self.store.manifest["fingerprint"], "result": data},
            producer="independent_reproduction",
        )
        self.store.put_evidence(
            "evidence/reproductions/reproduce-0123456789cd.json",
            {"session_fingerprint": "other", "result": {**data, "attempt_id": "reproduce-0123456789cd"}},
            producer="independent_reproduction",
        )
        events = self.store.path / "events.jsonl"
        before = events.read_bytes()
        graph, = self.report()["validation_graphs"]
        self.assertEqual(graph["action"], "reproduce")
        self.assertEqual(graph["last"][0]["receiver"], "复测回执")
        self.assertEqual(events.read_bytes(), before)
