"""Exercise live reporting from the same receipts an unattended run leaves behind."""

import io
import tempfile
import unittest
from pathlib import Path

from hyperloop.evaluation.workflow import _ValidationAttempt
from hyperloop.observability import RunObserver, describe_event
from hyperloop.report import write_report
from hyperloop.session import SessionStore


class ObservabilityTests(unittest.TestCase):
    def test_log_names_the_baseline_and_delegated_values_from_events(self) -> None:
        baseline = describe_event({"type": "baseline_recorded", "payload": {"result": {"revision": {"id": "r0"}}}})
        self.assertIn("原始基线 r0", baseline)
        intent = describe_event({"type": "agent_intent", "payload": {"role": "orchestration", "intent": {
            "kind": "delegate", "action": "explore", "params": {"candidates": [
                {"id": "c768", "parameters": {"chunked_prefill_size": 768}, "reason": "减少切块次数"},
            ]},
        }}})
        self.assertIn("c768 · chunked_prefill_size=768 · 减少切块次数", intent)

    def test_launch_and_live_exchange_are_preserved_when_resuming(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "demo-910b3-01"
            overview = root.parent / "最新说明.md"
            stream = io.StringIO()
            observer = RunObserver(root, latest_path=overview, stream=stream)
            observer.progress("参考采集", "启动原版服务")
            with observer.activate():
                attempt = _ValidationAttempt(root, "reference-capture")
                attempt.step("Evaluator", "RuntimeBackend", "launch", "reference", "启动隔离参考服务", "requested")
            store = SessionStore.create(root, "simulated", root.name, {"task": {}})
            state = store.commit([
                ("task_started", {"task_id": "task-0001", "action": "baseline"}),
                ("agent_provider_started", {"role": "orchestration", "phase": "FRAMEWORK_AGENT"}),
            ])
            observer.checkpoint(store, state)
            log = store.path / "console.log"
            for snippet in ("参考采集 · 启动原版服务", "Evaluator → RuntimeBackend", "启动隔离参考服务",
                            "#1 · Dispatcher 已派发", "#2 · Coordinator 已请求 Orchestration"):
                self.assertIn(snippet, log.read_text(encoding="utf-8"))
                self.assertIn(snippet, stream.getvalue())
            self.assertIn("事件 #2", overview.read_text(encoding="utf-8"))

            resumed_stream = io.StringIO()
            resumed = RunObserver(store.path, stream=resumed_stream)
            state = store.emit("task_recovered", {"task_id": "task-0001", "cleanup": {"released": True}})
            resumed.checkpoint(store, state)
            self.assertNotIn("#1", resumed_stream.getvalue())
            self.assertIn("#3 · Coordinator 已确认", resumed_stream.getvalue())
            self.assertEqual(log.read_text(encoding="utf-8").count(" · #1 · "), 1)

    def test_report_keeps_agent_proposal_and_goal_evidence_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "demo-910b3-01"
            store = SessionStore.create(root, "simulated", root.name, {
                "task": {"goals": {"throughput_gain_pct": 15, "ttft_reduction_pct": 15}},
                "scenario": {"startup": {"schema_version": 1}},
            })
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "任务.md").write_text("保留原始任务", encoding="utf-8")
            call = store.put_evidence("evidence/agents/first.json", {
                "response_content": "已存放真实回复", "normalized_intents": [],
            }, producer="agent_provider")
            appraisal = store.put_evidence("evidence/goals/test.json", {"status": "fail"}, producer="goal_evaluator")
            store.commit([
                ("agent_provider_call", {"turn_id": "turn-1", "role": "specialist", "status": "ok", "evidence": call}),
                ("agent_intent", {"turn_id": "turn-1", "role": "specialist", "intent": {
                    "kind": "propose_action", "action": "integrate_patch",
                    "params": {"candidate": {"id": "qwen", "reason": "降低输入处理开销"}},
                }}),
                ("intent_result", {"turn_id": "turn-1", "role": "specialist", "result": {"status": "pending"}}),
                ("goal_assessed", {"status": "fail", "revision": "qwen", "baseline_revision": "original",
                                   "comparison": {"over_original": {"throughput_pct": 16, "ttft_pct": -12}},
                                   "evidence": {"goal_assessment": appraisal}}),
            ])
            state = store.state()
            state["current_revision"] = {"id": "qwen"}
            observer = RunObserver(root, stream=io.StringIO())
            observer.checkpoint(store, state)
            write_report(store.path, store.manifest, state, store.events())
            observer.checkpoint(store, state, status="done")
            markdown = (store.path / "report.md").read_text(encoding="utf-8")
            for snippet in ("../../inputs/", "[终端日志](console.log)", "[#2](report.html#event-2)", "降低输入处理开销",
                            "Intent Router 已处理 Specialist", "evidence/agents/first.json",
                            "## 独立目标验收", "未达标 · qwen", "evidence/goals/test.json",
                            "框架补丁 | 已提案，未派发", "算子 | 未派发"):
                self.assertIn(snippet, markdown)
            self.assertIn("[Markdown 报告](report.md)", (store.path / "最新说明.md").read_text(encoding="utf-8"))

    def test_supplemental_review_is_visible_in_console_report_and_signal_flow(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            store = SessionStore.create(root, "simulated", "review-log", {})
            summary = {"mode": "simulated", "boundary": "single_rope_and_copy_npu_events", "cases": [{
                "phase": "prefill", "layout_source": "derived_model_config_and_frozen_request",
                "split_rope_us": 20, "contiguous_rope_us": 15, "copy_us": 6,
                "copy_plus_rope_us": 21, "net_gain_us": -1, "outputs_match": True,
            }]}
            ref = store.put_evidence("evidence/review/probe.json", summary, producer="rope_layout_probe")
            identity = {"proposal_id": "proposal-0001", "task_id": "task-0002"}
            review = {"proposal_id": identity["proposal_id"], "verdict": "needs_review", "reason": "缺布局对照"}
            critic = {"role": "critic", "provider": "deepseek", "task_id": None}
            call = store.put_evidence("evidence/agents/rereview.json", {**critic, "turn_id": "turn-0003", "request_context": {
                "pending_proposals": [{"proposal_id": identity["proposal_id"], "review_evidence": {"summary": summary}}],
            }, "response_content": "净收益不足"}, producer="agent_provider")
            orchestration = {**critic, "role": "orchestration"}
            replan_call = store.put_evidence("evidence/agents/replan.json", {
                **orchestration, "turn_id": "turn-0004", "request_context": {"review_feedback": [{**review,
                    "status": "reject", "reason": "净收益不足"}]},
            }, producer="agent_provider")
            state = store.commit([
                ("proposal_registered", {"proposal_id": identity["proposal_id"], "source": "specialist"}),
                ("review_verdict", review),
                ("review_evidence_requested", {**identity, "kind": "rope_layout", "reason": "缺布局对照"}),
                ("task_started", {"task_id": identity["task_id"], "action": "rope_layout_probe"}),
                ("review_evidence_recorded", {**identity, "status": "ok", "summary": summary,
                                              "evidence": {"rope_layout": ref}}),
                ("proposal_reopened", {"proposal_id": identity["proposal_id"], "evidence_ref": ref, "summary": summary}),
                ("agent_provider_started", {**critic, "request_key": "turn-0003"}),
                ("agent_provider_call", {**critic, "turn_id": "turn-0003", "status": "ok", "evidence": call}),
                ("review_verdict", {**review, "verdict": "reject", "reason": "净收益不足"}),
                ("review_replan", {"proposal_ids": [identity["proposal_id"]], "reason": "需要尝试其他方案"}),
                ("agent_provider_started", {**orchestration, "request_key": "turn-0004"}),
                ("agent_provider_call", {**orchestration, "turn_id": "turn-0004", "status": "ok", "evidence": replan_call}),
                ("stopped", {"reason": "cycle_limit"}),
            ])
            stream = io.StringIO()
            RunObserver(root, stream=stream).checkpoint(store, state)
            self.assertIn("单次 RoPE 与复制的 NPU 设备计时", stream.getvalue())
            self.assertIn("模拟；prefill · 模型与冻结请求推导布局", stream.getvalue())
            self.assertIn("复制 + RoPE 21.00 μs", stream.getvalue())
            self.assertIn("净节省 -1.00 μs", stream.getvalue())
            self.assertIn("输出一致", stream.getvalue())
            self.assertIn("最近补证交接", (store.path / "最新说明.md").read_text(encoding="utf-8"))
            report = write_report(store.path, store.manifest, state, store.events())
            markdown = (store.path / "report.md").read_text(encoding="utf-8")
            review_section = markdown.split("## 补证与复审", 1)[1].split("## 候选验收", 1)[0]
            self.assertIn("缺布局对照", review_section)
            self.assertIn("净收益不足", review_section)
            self.assertIn("evidence/review/probe.json", review_section)
            self.assertIn('id="review-evidence"', (store.path / "report.html").read_text(encoding="utf-8"))
            flow = {step["kind"]: step for step in report["signal_flow"]}
            self.assertEqual(flow["task_started"]["receiver"], "RoPE 布局探测")
            self.assertTrue(flow["review_evidence_recorded"]["replay"])
            requests = [step for step in report["signal_flow"] if step["kind"] == "agent_provider_started"]
            self.assertIn("复审请求附有模拟布局补证", requests[0]["message"])
            self.assertIn("net_gain_us", requests[0]["preview"])
            self.assertIn("请求附有前次审查反馈", requests[1]["message"])
            self.assertIn("净收益不足", requests[1]["preview"])
            self.assertEqual(flow["review_replan"]["receiver"], "Session")

    def test_probe_without_measurements_reports_its_status_only(self) -> None:
        message = describe_event({"type": "review_evidence_recorded", "payload": {
            "proposal_id": "p1", "task_id": "t1", "status": "unsupported", "reason": "layout_unavailable",
        }})
        self.assertIn("不支持", message)
        self.assertIn("layout_unavailable", message)
        self.assertNotIn("μs", message)


if __name__ == "__main__":
    unittest.main()
