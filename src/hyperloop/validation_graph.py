"""Project committed task events and evaluation evidence into handoff graphs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .validation_history import reconstruct_validation_flow

Record = dict[str, Any]
ReadEvidence = Callable[[Mapping[str, Any]], Record | None]

_ACTIONS = {
    "baseline": "原始基线", "enablement": "修复后验证", "profile": "Profile 采集",
    "explore": "框架参数候选", "integrate_patch": "框架补丁候选",
    "kernel_agent": "算子候选", "conc_sweep": "并发扫描", "stack_revalidate": "整栈补测",
}

_EXECUTORS = {
    "baseline": "Baseline", "enablement": "Enablement", "profile": "Profile 动作",
    "explore": "Explore", "integrate_patch": "Integrate Patch",
    "kernel_agent": "算子执行流程", "conc_sweep": "SWEEP", "stack_revalidate": "整栈补测",
}
_ROLES = {"orchestration": "Orchestration", "specialist": "Specialist", "critic": "Critic"}


def _record(value: Any) -> Record:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[Record]:
    return [_record(item) for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _edge(
    sender: str, receiver: str, message: str, operation: str, *,
    status: str = "recorded", event: Record | None = None, evidence: Any = None,
) -> Record:
    row = {"sender": sender, "receiver": receiver, "message": message, "operation": operation,
           "state": status, "origin": "event" if event else "reconstructed"}
    if event:
        row["event_seq"] = event["seq"]
    if isinstance(evidence, Mapping) and isinstance(evidence.get("path"), str):
        row["evidence"] = [dict(evidence)]
    return row


def _run_graph(run: Record, label: str, reader: ReadEvidence) -> Record | None:
    references = _record(run.get("evidence"))
    if not references:
        return None
    trace_ref = _record(references.get("validation_flow"))
    trace = reader(trace_ref) if trace_ref else None
    attempt_id = run.get("attempt_id")
    if trace and trace.get("schema_version") == 1:
        attempt_id = attempt_id or trace.get("attempt_id")
    if not attempt_id:
        for ref in references.values():
            parts = str(_record(ref).get("path", "")).split("/")
            if len(parts) == 3 and parts[0] == "evaluation":
                attempt_id = parts[1]
                break
    if not attempt_id:
        return None
    recorded = (
        trace is not None and trace.get("schema_version") == 1
        and trace.get("attempt_id") == attempt_id and isinstance(trace.get("steps"), list)
        and all(isinstance(row, Mapping) and all(isinstance(row.get(key), str) for key in
                ("sender", "receiver", "message", "operation", "state")) for row in trace["steps"])
    )
    if recorded:
        steps = _rows(trace["steps"])
        origin = "instrumented"
    else:
        decision_ref = _record(references.get("decision"))
        stored_decision = reader(decision_ref) if decision_ref else None
        facts = dict(references)
        for key, reference in references.items():
            if key.endswith("_correctness") and isinstance(reference, Mapping):
                facts[key] = reader(reference) or reference
        steps = reconstruct_validation_flow(facts, result={**run, **(stored_decision or {})})
        if references.get("paired_comparison"):
            steps.append({
                "sender": "Evaluator", "receiver": "DecisionGate",
                "message": "候选、父版和原始基线的测量",
                "operation": "核对比较口径，计算相对两份锚点的变化", "state": "recorded",
                "evidence_keys": ["paired_comparison", "policy"],
            })
        if stored_decision:
            steps.append({
                "sender": "Evaluator", "receiver": "EvidenceStore",
                "message": f"评测结果 · {stored_decision.get('status', '')}",
                "operation": f"保存本次尝试的状态与原因 · {stored_decision.get('reason', '')}",
                "state": "recorded", "evidence_keys": ["decision"],
            })
        origin = "reconstructed"
    for step in steps:
        keys = step.pop("evidence_keys", [])
        step["evidence"] = [
            references[key] for key in keys if isinstance(key, str) and isinstance(references.get(key), Mapping)
        ] if isinstance(keys, list) else []
        step["origin"] = origin
    return {
        "attempt_id": str(attempt_id), "label": label, "origin": origin, "steps": steps,
        "candidate_id": run.get("candidate_id"), "status": (run.get("status") or (trace or {}).get("status")),
        "trace": trace_ref or None,
    }


def _attempts(result: Record, action: str, reader: ReadEvidence) -> list[Record]:
    attempts: list[Record] = []
    seen: set[str] = set()

    def add(run: Record, label: str, candidate_id: str | None = None) -> Record | None:
        attempt = _run_graph(run, label, reader)
        if attempt and attempt["attempt_id"] not in seen:
            seen.add(attempt["attempt_id"])
            attempt["candidate_id"] = candidate_id or attempt["candidate_id"]
            attempt["after"] = []
            attempts.append(attempt)
            return attempt
        return None

    add(result, _ACTIONS[action])
    forward = _record(result.get("model_forward"))
    if forward:
        add(forward, "原始版本模型 forward")
    for row in _rows(result.get("decisions")):
        candidate = str(row.get("candidate_id") or "候选")
        tool = _record(row.get("tool_evidence"))
        tool_attempt = None
        if tool.get("operator"):
            tool_attempt = add({"evidence": tool["operator"]}, f"{candidate} · 算子工具内验证", candidate)
        if tool.get("service"):
            tool_attempt = add({"evidence": tool["service"]}, f"{candidate} · GEAK 式工具内服务测量", candidate)
        publication_ref = _record(tool.get("publication"))
        publication = reader(publication_ref) if publication_ref else None
        if tool_attempt and publication:
            delivery = publication.get("delivery")
            message = {"service_candidate": "服务方案及工具测量", "operator_patch": "算子候选及设备测量"}.get(
                delivery, "工具结果")
            tool_attempt["after"].append(_edge(
                "算子工具", "算子执行流程", message,
                f"记录工具交付与验收入口 · {publication.get('status', '')}", evidence=publication_ref,
            ))
        primary = add(row, f"{candidate} · 项目侧候选／父版验证", candidate)
        if primary:
            primary["primary"] = True
        if row.get("model_forward"):
            add(_record(row["model_forward"]), f"{candidate} · 模型 forward", candidate)
    for pair in _rows(result.get("pairs")):
        concurrency = pair.get("concurrency", "?")
        for arm, name in (("original", "原始版"), ("candidate", "当前接受版")):
            if pair.get(arm):
                attempt = add(_record(pair[arm]), f"并发 {concurrency} · {name}")
                if attempt and arm == "candidate" and pair.get("comparison"):
                    attempt["after"].append(_edge(
                        "BenchmarkRunner", "SWEEP", f"并发 {concurrency} · 原始版与接受版的样本",
                        "对齐这一个工作点并保存两组指标的比较", evidence=_record(result.get("evidence")).get("paired_sweep"),
                    ))
    return attempts


def build_validation_graphs(state: Record, events: list[Record], reader: ReadEvidence) -> list[Record]:
    tasks: dict[str, Record] = {}
    proposals: dict[str, list[Record]] = {}
    delegations: dict[str, Record] = {}
    intents: dict[str, Record] = {}
    for task_id, data in _record(state.get("tasks")).items():
        task = _record(data)
        if task.get("action") in _ACTIONS:
            tasks[task_id] = {"task": task, "events": [], "runs": [], "cases": []}

    for event in events:
        kind = event.get("type")
        data = _record(event.get("payload"))
        proposal_id = data.get("proposal_id")
        if kind in {"proposal_registered", "review_verdict"} and isinstance(proposal_id, str):
            proposals.setdefault(proposal_id, []).append(event)
        turn = data.get("turn_id")
        if kind == "agent_intent" and isinstance(turn, str):
            intents[turn] = event
        if kind == "intent_result":
            response = _record(data.get("result"))
            if isinstance(response.get("task_id"), str) and response.get("action") in _ACTIONS:
                delegations[response["task_id"]] = intents.get(turn, event) if isinstance(turn, str) else event
        if kind in {"task_queued", "task_started", "task_settled", "task_cleanup_pending", "task_cancelled",
                    "task_recovered", "delegated_result"}:
            task_id = data.get("task_id")
            if isinstance(task_id, str) and task_id in tasks:
                tasks[task_id]["events"].append(event)
            elif data.get("action") in _ACTIONS and isinstance(task_id, str):
                tasks[task_id] = {"task": data, "events": [event], "runs": [], "cases": []}
        if kind in {"critic_measured_verdict", "candidate_decision", "candidate_attempt"}:
            candidate = _record(data.get("decision")) if kind == "candidate_decision" else data
            candidate_task = candidate.get("task_id")
            group = tasks.get(candidate_task) if isinstance(candidate_task, str) else None
            if group:
                group["cases"].append(event)
                if kind == "candidate_attempt":
                    group["runs"].append(candidate)
        elif kind in {"baseline_recorded", "baseline_failure", "profile_recorded", "sweep_recorded", "stack_validated"}:
            run = _record(data.get("result")) if kind == "baseline_recorded" else data
            task_id = run.get("task_id") or run.get("attempt_id")
            if kind == "sweep_recorded" and not task_id:
                task_id = f"sweep-skip-{event['seq']}"
                tasks[task_id] = {"task": {"action": "conc_sweep", "status": run.get("status")},
                                  "events": [], "runs": [], "cases": []}
            if isinstance(task_id, str) and task_id in tasks:
                tasks[task_id]["runs"].append(run)
                tasks[task_id]["events"].append(event)

    return [
        _task_graph(
            task_id, group, reader,
            proposals.get(group["task"].get("proposal_id"), []), delegations.get(task_id),
        )
        for task_id, group in tasks.items()
    ]


def _task_graph(
    task_id: str, group: Record, reader: ReadEvidence,
    proposal_events: list[Record], delegation: Record | None,
) -> Record:
    task = group["task"]
    action = task["action"]
    executor = _EXECUTORS[action]
    events = {row["type"]: row for row in group["events"]}
    result = _record(task.get("result"))
    result_event = events.get("task_settled") or events.get("task_cleanup_pending")
    if not result and result_event:
        result = _record(result_event["payload"].get("result"))
    receipt_ref = _record(result.get("receipt"))
    receipt = reader(receipt_ref) if receipt_ref else None
    receipt_valid = receipt is not None and receipt.get("task_id") == task_id and receipt.get("action") == action
    if receipt_valid:
        run = _record(receipt.get("result"))
    elif action in {"explore", "integrate_patch", "kernel_agent"}:
        run = {**result, "decisions": group["runs"] or result.get("decisions", [])}
    else:
        run = group["runs"][-1] if group["runs"] else result
    attempts = _attempts(run, action, reader)

    first = _submission_edges(proposal_events, delegation, action)
    last: list[Record] = []
    if "task_queued" in events:
        sender = "Coordinator" if task.get("source") == "coordinator" else "Intent Router"
        first.append(_edge(sender, "Dispatcher", f"{task_id} · {action}", "登记任务和资源要求",
                           event=events["task_queued"]))
    if "task_started" in events:
        first.append(_edge("Dispatcher", executor, f"{task_id} · {action}", "取得资源后由 Coordinator 启动该验证任务",
                           event=events["task_started"]))
    if receipt_ref:
        last.append(_edge(executor, "Coordinator", "任务回执", "保存执行结果、尝试编号和证据引用",
                          evidence=receipt_ref, event=result_event))
    if "task_cleanup_pending" in events:
        last.append(_edge("Dispatcher", "Coordinator", "清理待确认", "保留资源租约，等待恢复或清理",
                          status="failed", event=events["task_cleanup_pending"]))
    if "task_recovered" in events:
        recovered = _record(events["task_recovered"].get("payload"))
        last.append(_edge("Coordinator", "Dispatcher", "任务恢复", "核对回执和清理后恢复任务状态",
                          event=events["task_recovered"], evidence=recovered.get("receipt")))
    if "task_cancelled" in events:
        last.append(_edge("Dispatcher", "Coordinator", "任务取消", "记录原因，该队列项停止派发",
                          status="skipped", event=events["task_cancelled"]))
    if "delegated_result" in events:
        last.append(_edge("Dispatcher", "Orchestration 收件箱", "delegated_result", "发布任务终态及回执供后续回合读取",
                          event=events["delegated_result"]))
    for event in group["cases"]:
        last.extend(_candidate_edges(event))
    last.extend(_state_edges(events, run))

    return {"task_id": task_id, "action": action, "title": _ACTIONS[action],
            "status": run.get("status") or task.get("status"), "reason": run.get("reason"),
            "first": first, "attempts": attempts, "last": last}


def _submission_edges(proposals: list[Record], delegation: Record | None, action: str) -> list[Record]:
    steps = []
    for event in proposals:
        data = event["payload"]
        proposal_id = data["proposal_id"]
        if event["type"] == "proposal_registered":
            sender = _ROLES.get(data.get("source"), "提案角色")
            steps.append(_edge(sender, "Intent Router", f"{proposal_id} · 正式提案", "登记待审并交给 Critic",
                               event=event))
        else:
            verdict = data.get("verdict", "")
            steps.append(_edge("Critic", "Intent Router", f"{proposal_id} · {verdict}",
                               "处理该提案的预审结果", event=event))
    if delegation and not steps:
        sender = _ROLES.get(delegation["payload"].get("role"), "请求角色")
        steps.append(_edge(sender, "Intent Router", f"直接委派 · {action}", "通过准入后登记任务",
                           event=delegation))
    return steps


def _candidate_edges(event: Record) -> list[Record]:
    data = event["payload"]
    if event["type"] == "critic_measured_verdict":
        candidate = data.get("candidate_id", "")
        return [
            _edge("Coordinator", "独立 Critic", f"{candidate} · 候选、父版与原始基线证据", "按冻结规则复算正确性与指标"),
            _edge("独立 Critic", "Coordinator", f"{candidate} · {data.get('version_action', '')}",
                  f"返回裁决及依据 · {data.get('reason', '')}", event=event, evidence=data.get("evidence")),
        ]
    if event["type"] == "candidate_decision":
        decision = _record(data.get("decision"))
        action = decision.get("version_action", "")
        operation = "核对父版与就绪证据，更新当前版本" if data.get("promote") else "保存最终动作及原因"
        return [_edge("Coordinator", "Session", f"{decision.get('candidate_id', '')} · {action}", operation, event=event)]
    return []


def _state_edges(events: dict[str, Record], run: Record) -> list[Record]:
    steps = []
    for name, message, operation in (
        ("baseline_recorded", "有效原始基线", "记录原始版本、工作点和服务测量"),
        ("baseline_failure", "基线失败", "保存失败原因和证据"),
        ("profile_recorded", "Profile 结果", "写入已采集的 rank 和热点记录"),
        ("sweep_recorded", "并发扫描结果", "写入配对结果或跳过原因"),
        ("stack_validated", "整栈补测结果", "保留最终版本的加载与测量证据"),
    ):
        if name in events:
            payload = _record(events[name].get("payload"))
            reason = payload.get("reason")
            steps.append(_edge("Coordinator", "Session", f"{message} · {reason}" if reason else message,
                               operation, status="skipped" if run.get("status") == "skipped" else "recorded",
                               event=events[name], evidence=payload.get("receipt")))
    return steps


def build_reproduction_graphs(reproductions: list[Record], reader: ReadEvidence) -> list[Record]:
    graphs = []
    for item in reproductions:
        result = item["result"]
        attempt_id = result["attempt_id"]
        attempts = []
        for run, label in ((result, "独立复测 · 候选与父版"), (_record(result.get("model_forward")), "模型 forward")):
            attempt = _run_graph(run, label, reader)
            if attempt:
                attempts.append({**attempt, "after": []})
        graphs.append({
            "task_id": attempt_id, "action": "reproduce", "title": f'独立复测 · {result.get("candidate_id", "")}',
            "status": result.get("status"), "reason": result.get("reason"), "first": [], "attempts": attempts,
            "last": [_edge("独立 Critic", "复测回执",
                           f"独立复核 · {result.get('version_action', '')}",
                           "单独保存复测结果，原 Session 的裁决仍可追溯", evidence=item["reference"])],
        })
    return graphs
