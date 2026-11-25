"""The report can replay committed handoffs without a running service."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from hyperloop.report import write_report
from hyperloop.session import SessionStore
from hyperloop.signal_flow import build_signal_flow


class SignalFlowTests(unittest.TestCase):
    def test_report_links_verified_calls_and_records_the_real_order(self) -> None:
        with TemporaryDirectory() as folder:
            store = SessionStore.create(Path(folder), "real", "signal-test", {})
            role = {"provider": "deepseek", "role": "orchestration", "task_id": None}
            turn = {**role, "turn_id": "turn-0002"}
            response = '<script data-test="untrusted">alert(1)</script>'
            ref = store.put_evidence("evidence/agents/call #2.json", {
                **turn, "request_context": {"direction_change": {"alternatives": ["patch", "kernel"]}},
                "response_content": response,
            }, producer="agent_provider")
            store.commit([
                ("phase_changed", {"phase": "FRAMEWORK_AGENT", "cycle": 1}),
                ("agent_provider_started", {**role, "request_key": "turn-0001"}),
                ("agent_provider_call", {**role, "turn_id": "turn-0001", "status": "failed", "reason": "request_timeout"}),
                ("task_retry_scheduled", {"role": "orchestration", "retry_of": "turn-0001", "reason": "request_timeout"}),
                ("direction_changed", {"tested_direction": "parameters", "alternatives": ["patch", "kernel"]}),
                ("agent_provider_started", {**role, "request_key": "turn-0002"}),
                ("agent_provider_call", {**turn, "status": "ok", "http_status": 200, "evidence": ref}),
                ("agent_intent", {**turn, "intent": {"kind": "delegate", "action": "explore", "params": {
                    "candidates": [{"id": "prefill-768", "parameters": {"chunked_prefill_size": 768}}],
                }}}),
                ("task_cleanup_wait", {"task_id": "T1", "action": "explore", "cleanup": {"released": False}}),
                ("task_recovered", {"task_id": "T1", "cleanup": {"released": True}}),
                ("sweep_recorded", {"status": "skipped", "reason": "stack_unverified"}),
                ("goal_assessed", {"status": "incomplete", "reason": "goal_retest_unavailable"}),
            ])

            report = write_report(store.path, store.manifest, store.state(), store.events())
            rows = report["signal_flow"]
            self.assertEqual([row["seq"] for row in rows], [row["seq"] for row in store.events()])
            self.assertEqual(rows[1]["preview"], "")
            self.assertEqual(rows[2]["sender"], "Agent API 客户端")
            self.assertIn("turn-0001", rows[3]["message"])
            self.assertEqual(rows[4]["receiver"], "Session")
            self.assertIn("请求附有换向要求", rows[5]["message"])
            self.assertIn('"patch"', rows[5]["preview"])
            self.assertIn(["evidence/agents/call #2.json", "evidence/agents/call%20%232.json"], rows[5]["links"])
            self.assertEqual(rows[6]["preview"], response)
            self.assertEqual(rows[6]["sender"], "DeepSeek")
            self.assertIn("chunked_prefill_size=768", rows[7]["message"])
            self.assertIn("资源尚未释放", rows[8]["message"])
            self.assertIn("资源已释放", rows[9]["message"])
            self.assertEqual(rows[10]["sender"], "Coordinator")
            self.assertEqual(rows[11]["receiver"], "Session")
            html = (store.path / "report.html").read_text(encoding="utf-8")
            self.assertIn('data-signal="player"', html)
            self.assertIn('data-signal="arrow-label"', html)
            self.assertIn('id="validation"', html)
            self.assertIn("真机事件", html)
            self.assertNotIn(response, html)
            self.assertIn(r'\u003cscript data-test=', html)

    def test_missing_or_mismatched_proof_does_not_fill_in_a_request(self) -> None:
        reply = {"role": "orchestration", "provider": "deepseek", "turn_id": "turn-real", "status": "ok"}
        proof = {**reply, "request_context": {"direction_change": {"alternatives": ["kernel"]}}}
        reply["evidence"] = {"path": "evidence/agent.json", "sha256": "digest"}
        messages = [
            {"seq": 6, "type": "agent_provider_call", "payload": reply},
            {"seq": 5, "type": "agent_provider_started", "payload": {
                "role": "critic", "provider": "deepseek", "request_key": "turn-real",
            }},
            {"seq": 8, "type": "agent_provider_started", "payload": {
                "role": "orchestration", "provider": "deepseek", "request_key": "turn-real",
            }},
        ]
        rows = build_signal_flow(messages, lambda ref: proof, lambda ref: None, {})
        self.assertEqual([row["seq"] for row in rows], [5, 6, 8])
        for row in (rows[0], rows[2]):
            self.assertEqual(row["preview"], "")
            self.assertEqual(row["links"], [])
            self.assertIn("没有可核验的请求正文", row["operation"])

    def test_fixture_turn_is_marked_as_replay(self) -> None:
        identity = {"turn_id": "turn-0001", "role": "specialist"}
        messages = [
            {"seq": 1, "type": "agent_turn_started", "payload": {**identity, "task_id": "T1", "context": {}}},
            {"seq": 2, "type": "agent_intent", "payload": {**identity, "intent": {"kind": "delegate"}}},
        ]
        (intent,) = build_signal_flow(messages, lambda ref: None, lambda ref: None, {})
        self.assertTrue(intent["replay"])
        self.assertEqual(intent["receiver"], "Intent Router")

    def test_only_verified_deliveries_add_native_tool_nodes(self) -> None:
        forge = {"path": "evidence/kernel/T1-delivery.json", "producer": "native_forge", "sha256": "first"}
        geak = {"path": "evidence/kernel/T2-delivery.json", "producer": "native_geak", "sha256": "second"}
        blocked = {"path": "evidence/kernel/T3-delivery.json", "producer": "native_forge", "sha256": "third"}
        events = [{"seq": seq, "type": "task_settled", "payload": {"task_id": task, "action": "kernel_agent",
                    "result": {"evidence": {"delivery": ref}}}}
                  for seq, task, ref in [(10, "T1", forge), (11, "T2", geak), (15, "T3", blocked), (16, "T4", forge)]]
        proofs = {
            forge["path"]: {"mode": "real", "backend": "forge", "delivery": "operator_patch", "status": "ok",
                            "candidate": {"operator": "SwiGLU", "id": "T1-swiglu"}},
            geak["path"]: {"mode": "real", "backend": "geak", "delivery": "service_candidate", "status": "ok",
                           "service_candidate": {"spec": {"id": "T2-geak"},
                                                 "configuration": {"ascend_use_fia": True}}},
            blocked["path"]: {"mode": "real", "backend": "forge", "delivery": "operator_patch", "status": "blocked"},
        }
        steps = build_signal_flow(events, lambda ref: proofs.get(ref["path"]), lambda ref: ref["path"],
                                  {"T1": "#validation-task-1"})
        deliveries = [step for step in steps if step["kind"] == "kernel_delivery"]
        self.assertEqual([(step["seq"], step["sender"]) for step in deliveries], [(10, "昇腾 Forge"), (11, "昇腾 GEAK")])
        self.assertIn("SwiGLU", deliveries[0]["message"])
        self.assertIn("ascend_use_fia=True", deliveries[1]["message"])
        self.assertIn(["查看 T1 的验证交接", "#validation-task-1"], deliveries[0]["links"])
        self.assertEqual(deliveries[0]["receiver"], "Kernel Agent")
        self.assertTrue(all(step["kind"] != "kernel_delivery"
                            for step in build_signal_flow(events, lambda ref: proofs.get(ref["path"]), lambda ref: None, {})))
