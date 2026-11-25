"""Human-readable views of actual Hyperloop execution and committed events."""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import quote

from .session import SessionStore, atomic_write, utc_now


_ACTIONS = {
    "baseline": "原始基线",
    "enablement": "启动修复验证",
    "profile": "Profile 采集",
    "explore": "调参",
    "specialist": "专项调查",
    "integrate_patch": "框架补丁集成",
    "kernel_agent": "算子验证",
    "conc_sweep": "并发扫描",
    "stack_revalidate": "整栈独立复测",
    "rope_layout_probe": "RoPE 布局探测",
}
_ROLES = {"orchestration": "Orchestration", "specialist": "Specialist", "critic": "Critic"}
_DIRECTIONS = {"parameters": "调参", "patch": "框架补丁", "kernel": "算子", "profile": "补充采集"}
_PHASES = {
    "PRELUDE": "基线与采集", "ENABLEMENT": "启动修复", "FRAMEWORK_AGENT": "框架探索",
    "KERNEL_AGENT": "算子验证", "SWEEP": "并发扫描", "CLOSE": "收尾复测", "DONE": "结束",
}
_STATUSES = {
    "running": "进行中", "paused": "已暂停", "blocked": "受阻", "failed": "失败",
    "done": "已结束", "pass": "通过", "fail": "未达标", "incomplete": "证据不足",
    "ok": "完成", "succeeded": "完成", "skipped": "跳过", "cancelled": "已取消",
    "interrupted": "中断", "approve": "批准", "advise": "附意见批准", "reject": "拒绝",
    "needs_review": "需复审", "denied": "拒绝", "queued": "已登记", "pending": "待审",
    "recorded": "已记录", "requested": "请求", "cleanup_pending": "等待清理",
    "unsupported": "不支持", "unavailable": "不可用",
}
_REASONS = {
    "prelude_complete": "已完成基线和采集阶段", "framework_round_complete": "框架阶段任务结算完毕",
    "kernel_tasks_settled": "算子任务结算完毕", "target_reached": "已达到任务目标",
    "budget_exhausted": "调度预算用完", "cycle_limit": "达到循环上限", "converged": "没有新的已验证改动",
    "continue_from_accepted_revision": "从当前接受版继续", "no_validated_change_since_sweep": "上次扫描后没有已验证的改动",
    "repeated_approach_without_accepted_result": "同一方向的连续试验均未达到接受条件",
    "cleanup_unconfirmed": "仍需核对是否释放旧资源", "empty_review_bundle": "没有待审提案",
    "gain_below_threshold": "吞吐收益低于门槛", "ttft_regression": "TTFT 回归超过上限",
    "tpot_regression": "TPOT 回归超过上限", "gates_passed": "正确性和性能门槛通过",
    "sweep_failed_or_incomplete": "并发扫描失败或缺少结果", "sweep_disabled": "未开启并发扫描",
    "interrupted_before_action_receipt": "中断前尚未收到执行回执",
    "framework_review_unresolved": "当前框架方案未通过审查",
    "rope_layout_evidence_ready": "布局探测证据已核对",
    "rope_probe_parent_or_profile_changed": "当前版本或 Profile 已变化",
    "rope_layout_probe_unavailable": "当前环境没有可用的 RoPE 布局探测",
    "rope_layout_probe_failed": "RoPE 布局探测失败",
    "invalid_rope_layout_evidence": "RoPE 布局证据校验失败",
    "review_evidence_identity_mismatch": "补证对应的提案、版本或 Profile 未核对一致",
}
_DURABLE_LINE = re.compile(r"^\S+ · #(\d+) · ")
_ACTIVE: ContextVar[RunObserver | None] = ContextVar("hyperloop_observer", default=None)
REVIEW_EVIDENCE_EVENTS = frozenset({
    "review_evidence_requested", "review_evidence_recorded", "proposal_reopened", "review_replan",
})


def _record(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any, default: str = "未记录", limit: int = 360) -> str:
    if not isinstance(value, str) or not value.strip():
        return default
    clean = " ".join("".join(char if char.isprintable() else " " for char in value).split())
    if not clean:
        return default
    return clean if len(clean) <= limit else clean[:limit] + "…"


def _name(names: Mapping[str, str], value: Any, default: str = "未记录") -> str:
    return names.get(value, _text(value, default)) if isinstance(value, str) else default


def status_label(value: Any) -> str:
    return _name(_STATUSES, value)


def _reason(value: Any) -> str:
    reason = _name(_REASONS, value, "")
    return f"（{reason}）" if reason else ""


def _metrics(row: Any) -> str:
    comparisons = _record(row)
    results = []
    for field, label in (("throughput_pct", "吞吐"), ("ttft_pct", "TTFT P95"), ("tpot_pct", "TPOT P95")):
        value = comparisons.get(field)
        if type(value) in (float, int):
            results.append(f"{label} {value:+.2f}%")
    return "，".join(results)


def _candidate_text(value: Any) -> str:
    row = _record(value)
    identifier = _text(row.get("id"), "")
    parameters = [f"{_text(key, limit=48)}={value}" for key, value in _record(row.get("parameters")).items()
                  if type(value) in (int, float, bool)]
    rationale = _text(row.get("reason"), "", limit=160)
    return " · ".join(part for part in (identifier, ", ".join(parameters), rationale) if part)


def _rope_probe_summary(value: Any) -> str:
    summary = _record(value)
    lines = []
    boundary = summary.get("boundary") or summary.get("scope")
    if boundary == "single_rope_and_copy_npu_events":
        lines.append("单次 RoPE 与复制的 NPU 设备计时")
    elif boundary:
        lines.append(f"计时边界 {_text(boundary)}")
    if summary.get("mode") in ("simulated", "replay"):
        lines.append("模拟回放" if summary["mode"] == "replay" else "模拟")
    cases = summary.get("cases")
    if isinstance(cases, list):
        for index, item in enumerate(cases[:4], 1):
            case = _record(item)
            numbers = []
            for key, label in (
                ("split_rope_us", "切分视图 RoPE"), ("contiguous_rope_us", "连续布局 RoPE"),
                ("copy_us", "复制"), ("copy_plus_rope_us", "复制 + RoPE"), ("net_gain_us", "净节省"),
            ):
                sample = case.get(key)
                if type(sample) in (int, float):
                    numbers.append(f"{label} {sample:.2f} μs")
            if numbers:
                label = _text(case.get("label") or case.get("phase"), f"工作点 {index}")
                tokens = case.get("tokens")
                if type(tokens) is int:
                    label = f"{label} · {tokens} tokens"
                source = _name({"service_observed": "服务实测布局",
                                "derived_model_config_and_frozen_request": "模型与冻结请求推导布局",
                                "simulated": "模拟布局"},
                               case.get("layout_source"), "")
                matches = case.get("outputs_match")
                outputs = "输出一致" if matches is True else "输出不一致" if matches is False else ""
                values = [label, source, "、".join(numbers), outputs]
                lines.append(" · ".join(part for part in values if part))
    return "；".join(lines)


def describe_event(event: Mapping[str, Any]) -> str | None:
    """Translate only facts this event establishes into a short operator message."""
    kind = event.get("type")
    data = _record(event.get("payload"))
    action = _name(_ACTIONS, data.get("action"), "操作")
    task = _text(data.get("task_id"), "当前任务")
    role = _name(_ROLES, data.get("role"), "Agent")
    reason = _reason(data.get("reason"))
    if kind == "runtime_probed":
        return "Coordinator 已核对实际设备和运行环境。"
    if kind == "baseline_recorded":
        result = _record(data.get("result"))
        revision = _text(_record(result.get("revision") or data.get("revision")).get("id"))
        if data.get("reused_from"):
            source = _text(_record(data["reused_from"]).get("session_id"))
            return f"Coordinator 已复用 {source} 的原始基线 {revision}，原始样本校验通过。"
        return f"Evaluator 已交回原始基线 {revision} 的校验和计时，Coordinator 已登记。"
    if kind == "baseline_failure":
        return f"Evaluator 未能建立基线{reason}。"
    if kind == "profile_recorded":
        if data.get("reused_from"):
            source = _text(_record(data["reused_from"]).get("session_id"))
            return f"Coordinator 已复用 {source} 的 Profile 和已验证的各 rank 证据。"
        status = status_label(data.get("status"))
        return f"Profiler 已向 Coordinator 返回采集结果 {task} · {status}{reason}。"
    if kind == "recipe_retrieved":
        status = _text(data.get("status"))
        revision = _text(data.get("revision"))
        return f"Coordinator 核对历史配方 {revision} · {status}{reason}。"
    if kind == "phase_changed":
        return f"Coordinator 进入{_name(_PHASES, data.get('phase'))}阶段{reason}。"
    if kind == "task_queued":
        retry = _text(_record(data.get("params")).get("retry_of"), "")
        note = f"，重试 {retry}" if retry else ""
        return f"Dispatcher 已登记 {task} · {action}{note}，等待可用资源。"
    if kind == "task_started":
        return f"Dispatcher 已派发 {task} · {action}。"
    if kind in ("task_settled", "task_cancelled", "task_cleanup_pending", "task_cleanup_wait"):
        result = _record(data.get("result"))
        status = status_label(data.get("status") or ("cleanup_pending" if kind == "task_cleanup_wait" else None))
        return f"Dispatcher 记录 {task} · {action} · {status}{_reason(result.get('reason')) or reason}。"
    if kind == "task_recovered":
        cleanup = _record(data.get("cleanup"))
        if cleanup.get("released") is True or cleanup.get("confirmed") is True:
            return f"Coordinator 已确认 {task} 的旧资源释放，恢复记录已保存。"
        return f"Coordinator 已记录 {task} 的恢复结果{reason}。"
    if kind == "task_retry_scheduled":
        if data.get("role"):
            return f"Coordinator 已安排重试 {role} 的调用{reason}。"
        return f"Coordinator 为 {task} 登记重试 {_text(data.get('retry_task_id'))}{reason}。"
    if kind == "agent_turn_started":
        return f"Coordinator 已启动 {role} 的回合 {_text(data.get('turn_id'))}。"
    if kind == "agent_provider_started":
        return f"Coordinator 已请求 {role}，等待 Agent 回复。"
    if kind == "agent_provider_call":
        status = data.get("status")
        if status == "ok":
            count = data.get("intent_count")
            amount = f"，收到 {count} 条意图" if type(count) is int else ""
            return f"Coordinator 已收到 {role} 的回复{amount}，完整内容已保存。"
        return f"Coordinator 记录 {role} 调用{status_label(status)}{reason}。"
    if kind == "agent_turn_failed":
        return f"Coordinator 记录 {role} 回合失败{reason}。"
    if kind == "agent_intent":
        intent = _record(data.get("intent"))
        detail = _record(intent.get("payload", intent))
        intent_kind = intent.get("kind", intent.get("type"))
        if intent_kind == "review_verdict":
            review = status_label(detail.get("verdict"))
            return f"{role} 向 Intent Router 反馈 {_text(detail.get('proposal_id'))} · {review}{_reason(detail.get('reason'))}。"
        if intent_kind == "repair_startup":
            return f"{role} 交回启动修复建议{_reason(detail.get('reason'))}。"
        name = _name(_ACTIONS, detail.get("action"), _text(intent_kind, "意图"))
        params = _record(detail.get("params"))
        candidates = params.get("candidates")
        if isinstance(candidates, list):
            descriptions = "，".join(text for row in candidates if (text := _candidate_text(row)))
        else:
            descriptions = _candidate_text(params.get("candidate"))
        rationale = _reason(params.get("focus") or detail.get("reason"))
        label = f" · {descriptions}" if descriptions else ""
        return f"{role} 向 Intent Router 提交 {name}{label}{rationale}。"
    if kind == "intent_result":
        result = _record(data.get("result"))
        status = status_label(result.get("status"))
        ref = _text(result.get("task_id") or result.get("proposal_id"), "")
        suffix = f" · {ref}" if ref else ""
        return f"Intent Router 已处理 {role} 的意图 · {status}{suffix}{_reason(result.get('reason'))}。"
    if kind == "proposal_registered":
        return f"Intent Router 已登记拟议操作 {_text(data.get('proposal_id'))}，交 Critic 审查。"
    if kind == "review_verdict":
        proposal = _text(data.get("proposal_id"))
        verdict = status_label(data.get("verdict") or data.get("status"))
        return f"Intent Router 已发布提案 {proposal} 的审查结果 · {verdict}{reason}。"
    if kind == "review_evidence_requested":
        proposal = _text(data.get("proposal_id"))
        return f"Coordinator 为提案 {proposal} 安排 RoPE 布局探测 {task}，补齐 Critic 要求的证据{reason}。"
    if kind == "review_evidence_recorded":
        proposal = _text(data.get("proposal_id"))
        status = status_label(data.get("status"))
        summary = _rope_probe_summary(data.get("summary"))
        suffix = f" · {summary}" if summary else ""
        return f"Coordinator 收到提案 {proposal} 的 RoPE 布局探测回执 {task} · {status}{reason}{suffix}。"
    if kind == "proposal_reopened":
        proposal = _text(data.get("proposal_id"))
        return f"Intent Router 已把提案 {proposal} 和补证返回 Critic 的待审列表，等待复审。"
    if kind == "review_replan":
        raw = data.get("proposal_ids")
        proposals = "、".join(part for item in raw if (part := _text(item, ""))) if isinstance(raw, list) else ""
        suffix = f" · {proposals}" if proposals else ""
        return f"Coordinator 已记录继续寻找其他改法{suffix}{reason}。"
    if kind == "delegated_result":
        return f"Intent Router 已记录 {task} · {action} 的回执，供后续回合读取。"
    if kind == "candidate_decision":
        decision = _record(data.get("decision"))
        candidate = _text(decision.get("candidate_id"))
        verdict = _text(decision.get("version_action"))
        metrics = _metrics(_record(decision.get("comparison")).get("over_original"))
        suffix = f" · 对原始基线 {metrics}" if metrics else ""
        return f"Coordinator 写回候选 {candidate} · {verdict}{_reason(decision.get('reason'))}{suffix}。"
    if kind == "direction_changed":
        direction = _name(_DIRECTIONS, data.get("tested_direction"), "已试方向")
        alternatives = data.get("alternatives")
        names = "、".join(_name(_DIRECTIONS, item) for item in alternatives) if isinstance(alternatives, list) else ""
        suffix = f"，建议比较 {names}" if names else "，重新判断下一轮方向"
        return f"Coordinator 已记录从{direction}换向{reason}{suffix}。"
    if kind == "goal_assessed":
        metrics = _metrics(_record(data.get("comparison")).get("over_original", data.get("comparison")))
        suffix = f" · {metrics}" if metrics else ""
        return f"独立复测核对任务目标 · {_text(data.get('revision'))} · {status_label(data.get('status'))}{reason}{suffix}。"
    if kind == "accepted_stack_rolled_back":
        restored = _text(_record(data.get("revision")).get("id"))
        return f"Coordinator 根据整栈复测回退 {_text(data.get('rejected_revision'))}，恢复版本 {restored}{reason}。"
    if kind in ("stack_validated", "sweep_recorded", "enablement_suggestion"):
        label = {"stack_validated": "整栈复测", "sweep_recorded": "并发扫描", "enablement_suggestion": "启动修复建议"}[kind]
        return f"Coordinator 已登记{label} · {status_label(data.get('status'))}{reason}。"
    if kind == "stopped":
        return f"Coordinator 已结束会话{reason}。"
    if kind == "run_blocked":
        return f"Coordinator 在等待 {role} 时受阻，已保存当前位置{reason}。"
    return None


def _link(path: Path, from_dir: Path) -> str:
    try:
        return quote(Path(os.path.relpath(path, from_dir)).as_posix(), safe="/")
    except ValueError:
        return quote(path.as_posix(), safe="/")


def _md_text(value: Any) -> str:
    return re.sub(r"([\\\[\]<>`*_])", r"\\\1", _text(value))


class RunObserver:
    """Persist visible progress and a replayable latest page for one run."""

    def __init__(self, output_dir: Path, *, latest_path: Path | None = None, stream: TextIO | None = None) -> None:
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.latest_path = (latest_path or self.output_dir / "最新说明.md").resolve()
        self.stream = sys.stderr if stream is None else stream
        self.store: SessionStore | None = None
        self.state: Mapping[str, Any] = {}
        self.last_seq = 0
        self.activity = "等待启动"
        self.review_activity: str | None = None
        self.updated_at = utc_now()

    @contextmanager
    def activate(self) -> Iterator[RunObserver]:
        token = _ACTIVE.set(self)
        try:
            yield self
        finally:
            _ACTIVE.reset(token)

    def progress(self, component: str, message: str, *, status: str = "running") -> None:
        self.updated_at = utc_now()
        self.activity = f"{_text(component)} · {_text(message)}"
        self._append([f"{self.updated_at} · {self.activity}"])
        self._latest(status)

    def checkpoint(self, store: SessionStore, state: Mapping[str, Any], *, status: str | None = None) -> None:
        events = store.events()
        if self.store is None:
            self._bind(store)
            self.review_activity = next((describe_event(event) for event in reversed(events[:self.last_seq])
                                         if event.get("type") in REVIEW_EVIDENCE_EVENTS), None)
        self.state = state
        rows = []
        for event in events[self.last_seq:]:
            message = describe_event(event)
            if message:
                self.updated_at = event["at"]
                self.activity = message
                if event.get("type") in REVIEW_EVIDENCE_EVENTS:
                    self.review_activity = message
                rows.append(f"{self.updated_at} · #{event['seq']} · {message}")
        self._append(rows)
        self.last_seq = int(state["last_seq"])
        inferred = "done" if state.get("phase") == "DONE" else "running"
        if state.get("blocker") and inferred != "done":
            inferred = "blocked"
        self._latest(status or inferred)

    def _bind(self, store: SessionStore) -> None:
        self.store = store
        launch_log = self.output_dir / "console.log"
        session_log = store.path / "console.log"
        if launch_log != session_log and launch_log.is_file() and not session_log.exists():
            atomic_write(session_log, launch_log.read_bytes())
        if session_log.is_file():
            with session_log.open(encoding="utf-8") as stream:
                for line in stream:
                    if match := _DURABLE_LINE.match(line):
                        self.last_seq = max(self.last_seq, int(match[1]))

    def _append(self, rows: list[str]) -> None:
        if not rows:
            return
        text = "\n".join(rows) + "\n"
        paths = {self.output_dir / "console.log"}
        if self.store:
            paths.add(self.store.path / "console.log")
        for path in paths:
            with path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
        try:
            self.stream.write(text)
            self.stream.flush()
        except OSError:
            # The durable log continues when the terminal reader disconnects.
            pass

    def _latest(self, status: str) -> None:
        state = self.state
        current = _record(state.get("current_revision")).get("id")
        revision = _md_text(current) if current else "等待基线"
        seq = state.get("last_seq")
        session_id = self.store.manifest.get("session_id") if self.store else self.output_dir.name
        lines = ["# 最新说明", "", f"- 会话 · {_md_text(session_id)}",
                 f"- 更新于 {_md_text(self.updated_at)}" + (f" · 事件 #{seq}" if seq else ""),
                 f"- 状态 · {status_label(status)}",
                 f"- 阶段 · {_name(_PHASES, state.get('phase'), '启动预检')}",
                 f"- 当前操作 · {_md_text(self.activity)}", f"- 最近接受版 · {revision}"]
        if self.review_activity and self.review_activity != self.activity:
            lines.append(f"- 最近补证交接 · {_md_text(self.review_activity)}")
        assessments = state.get("goal_assessments", [])
        goal = next((row for row in reversed(assessments) if row.get("revision") == current), {})
        if self.store and _record(self.store.manifest.get("task")).get("goals"):
            lines.append(f"- 双指标验收 · {status_label(goal.get('status')) if goal else '等待独立复测'}")
        decision = next((row for row in reversed(state.get("decisions", []))
                         if row.get("candidate_id") == current and row.get("version_action") == "KEEP"), {})
        comparison = _record((goal or decision).get("comparison"))
        if metrics := _metrics(comparison.get("over_original", comparison)):
            source = "独立复测" if goal else "接受时的配对评测"
            lines.append(f"- 对原始基线 · {metrics}（{source}）")
        if state.get("stop_reason"):
            lines.append(f"- 停止原因 · {_md_text(_name(_REASONS, state['stop_reason']))}")
        blocker = _record(state.get("blocker"))
        if blocker.get("reason"):
            lines.append(f"- 阻断原因 · {_md_text(_name(_REASONS, blocker['reason']))}")
        paths = {self.latest_path, self.output_dir / "最新说明.md"}
        if self.store:
            files = (("终端日志", self.store.path / "console.log"), ("会话事件", self.store.path / "events.jsonl"),
                     ("Markdown 报告", self.store.path / "report.md"), ("交接图", self.store.path / "report.html"))
            paths.add(self.store.path / "最新说明.md")
        else:
            files = (("启动状态", self.output_dir / "startup.json"), ("启动日志", self.output_dir / "console.log"),
                     ("参考采集回执", self.output_dir / "reference" / "capture.json"))
        for latest in paths:
            links = [f"- [{label}]({_link(path, latest.parent)})" for label, path in files if path.is_file()]
            atomic_write(latest, ("\n".join([*lines, *links]) + "\n").encode("utf-8"))


def observe_evaluation_step(attempt_id: str, row: Mapping[str, Any]) -> None:
    observer = _ACTIVE.get()
    if observer is not None:
        sender = _text(row.get("sender"))
        receiver = _text(row.get("receiver"))
        message = _text(row.get("message"))
        state = status_label(row.get("state"))
        observer.progress(f"{sender} → {receiver}", f"{attempt_id} · {message} · {state}")
