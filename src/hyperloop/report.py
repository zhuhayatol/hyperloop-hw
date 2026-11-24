"""Read-only projections of a replayed Session and its evidence."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import hashlib
from html import escape
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from .contracts import json_bytes
from .session import atomic_write
from .validation_graph import build_reproduction_graphs, build_validation_graphs


_STYLES = """
  :root { color-scheme: light dark; font: 16px/1.55 system-ui, sans-serif; }
  body { margin: 0 auto; max-width: 74rem; padding: 1.5rem; }
  h1, h2, h3 { line-height: 1.25; overflow-wrap: anywhere; }
  nav { display: flex; flex-wrap: wrap; gap: .5rem 1rem; }
  section { margin-block: 2rem; scroll-margin-top: 1rem; }
  .mode { display: inline-block; border: 2px solid currentColor; border-radius: .25rem; padding: .12rem .55rem; }
  .table-scroll { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; }
  th, td { text-align: left; vertical-align: top; border-bottom: 1px solid GrayText; padding: .55rem; }
  details { border: 1px solid GrayText; border-radius: .25rem; padding: .65rem; margin-block: .45rem; }
  summary { cursor: pointer; overflow-wrap: anywhere; }
  :focus-visible { outline: 2px solid currentColor; outline-offset: 3px; }
  code, pre { white-space: pre-wrap; overflow-wrap: anywhere; }
  pre { margin-block-end: 0; }
  dl { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: .35rem 1rem; }
  dd { margin: 0; overflow-wrap: anywhere; }
  li { overflow-wrap: anywhere; }
  .validation-task { border-color: color-mix(in srgb, CanvasText 30%, Canvas); scroll-margin-top: 1rem; }
  .validation-task:target, .timeline-event:target { outline: 2px solid Highlight; outline-offset: 3px; }
  .flow-attempt { margin-block: .6rem; background: color-mix(in srgb, CanvasText 4%, Canvas); }
  .flow-caption { margin-block: .5rem; font-size: .875rem; }
  .flow-anchor { display: block; scroll-margin-top: 1rem; }
  .flow-list { list-style: none; display: grid; gap: .45rem; padding: 0; margin: .6rem 0; }
  .flow-step { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 2fr) minmax(0, 1.6fr);
    gap: .25rem .65rem; align-items: stretch; padding: .45rem; border-bottom: 1px solid color-mix(in srgb, CanvasText 18%, Canvas); }
  .flow-actor { padding: .42rem .55rem; border-radius: .3rem; border: 1px solid color-mix(in srgb, CanvasText 25%, Canvas);
    background: Canvas; overflow-wrap: anywhere; }
  .flow-actor small { display: block; font-size: .75rem; }
  .flow-actor p { margin: .2rem 0 0; font-size: .875rem; }
  .flow-transfer { align-self: center; display: grid; gap: .25rem; text-align: center;
    font-size: .875rem; overflow-wrap: anywhere; }
  .flow-transfer::after { content: ""; width: 100%; height: 9px; background: LinkText;
    clip-path: polygon(0 40%, calc(100% - 8px) 40%, calc(100% - 8px) 0, 100% 50%,
      calc(100% - 8px) 100%, calc(100% - 8px) 60%, 0 60%); }
  .flow-proof { grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: .2rem .75rem; font-size: .78rem; }
  .flow-file { overflow-wrap: anywhere; }
  .flow-proof .flow-state { font-weight: 650; }
  .flow-step[data-state="failed"] .flow-actor { border-color: #d26052; }
  .flow-step[data-state="skipped"] .flow-actor { border-style: dashed; }
  @media (max-width: 36rem) {
    .flow-step { grid-template-columns: minmax(0, 1fr) minmax(0, 1.25fr) minmax(0, 1.25fr); gap: .3rem; }
    .flow-actor { padding: .3rem; }
    .flow-actor strong, .flow-actor p, .flow-transfer { font-size: .78rem; }
  }
  @media (max-width: 30rem) { body { padding: .85rem; } dl { display: block; } dd { margin-bottom: .45rem; } }
"""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _records(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _label(value: Any, default: str = "未记录") -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (str, int, float)) and value != "":
        return str(value)
    return default


def _pick(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def _identity(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _pick(value, "id", "revision_id", "revision", "name")
    return value


def _pretty(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def _local_file(root: Path, raw: str) -> tuple[Path | None, str]:
    if not raw or "\\" in raw or ":" in raw or any(ord(char) < 32 for char in raw):
        return None, "unsafe_path"
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return None, "unsafe_path"
    try:
        local = root.joinpath(*path.parts).resolve()
        if not local.is_relative_to(root):
            return None, "unsafe_path"
        return (local, "available") if local.is_file() else (None, "missing")
    except (OSError, RuntimeError, ValueError):
        return None, "unavailable"


def _references(value: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if isinstance(value.get("path"), str):
            yield value
        for child in value.values():
            yield from _references(child)
    elif isinstance(value, list):
        for child in value:
            yield from _references(child)


def _evidence_index(root: Path) -> tuple[list[Mapping[str, Any]], list[str]]:
    index_path, status = _local_file(root, "evidence/index.jsonl")
    if index_path is None:
        return [], [] if status == "missing" else [f"证据索引无法读取 ({status})"]
    records = []
    warnings = []
    try:
        with index_path.open(encoding="utf-8") as source:
            for position, line in enumerate(source, 1):
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("expected object")
                    json_bytes(record)
                except ValueError:
                    warnings.append(f"证据索引第 {position} 行无法解析")
                else:
                    records.append(record)
    except (OSError, UnicodeError):
        warnings.append("证据索引无法读取")
    return records, warnings


def _reference_key(reference: Mapping[str, Any]) -> tuple[str, str]:
    return reference["path"], _label(_pick(reference, "sha256", "evidence_id", "id"), "")


def _index_entries(root: Path, *values: Any) -> list[dict[str, Any]]:
    entries: dict[tuple[str, str], dict[str, Any]] = {}
    digests: dict[Path, str | None] = {}
    for value in values:
        for reference in _references(value):
            key = _reference_key(reference)
            if key in entries:
                continue
            raw, _ = key
            local, state = _local_file(root, raw)
            claimed = reference.get("sha256")
            if local is not None and isinstance(claimed, str):
                if local not in digests:
                    try:
                        with local.open("rb") as source:
                            digests[local] = hashlib.file_digest(source, "sha256").hexdigest()
                    except OSError:
                        digests[local] = None
                observed = digests[local]
                if observed is None:
                    state = "unavailable"
                elif observed != claimed:
                    state = "hash_mismatch"
            entry = dict(reference)
            entry["link_state"] = state
            entry["href"] = quote(raw, safe="/") if state in {"available", "hash_mismatch"} else None
            entries[key] = entry
    return list(entries.values())


def _evidence_for(record: Mapping[str, Any], entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = {_reference_key(item) for item in _references(record)}
    return [entry for entry in entries if _reference_key(entry) in keys]


def _number(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return f"{value:+.2f}%"
    return _label(value, "未计算")


def _h(value: Any, default: str = "未记录") -> str:
    return escape(_label(value, default))


def _md(value: Any, default: str = "未记录") -> str:
    plain = _label(value, default).replace("\r", " ").replace("\n", " ")
    plain = escape(plain, quote=False)
    for marker in ("\\", "`", "|", "*", "_", "[", "]"):
        plain = plain.replace(marker, "\\" + marker)
    return plain


def _comparison(value: Any) -> str:
    record = _mapping(value)
    terms = (
        ("throughput_pct", "吞吐"),
        ("ttft_pct", "TTFT"),
        ("tpot_pct", "TPOT"),
    )
    return "，".join(f"{name} {_number(record[key])}" for key, name in terms if key in record) or "未计算"


def _decision_fields(decision: Mapping[str, Any]) -> tuple[str, ...]:
    evidence = _label(decision.get("evidence_state"))
    acceptance = _label(decision.get("acceptance"))
    evidence_labels = {"sufficient": "充分", "missing": "缺失", "incomparable": "不可比", "conflict": "冲突"}
    acceptance_labels = {"pass": "达标", "fail": "未达标", "undetermined": "无法判定"}
    comparison = _mapping(decision.get("comparison"))
    return (
        _label(decision.get("candidate_id")),
        evidence_labels.get(evidence, evidence),
        acceptance_labels.get(acceptance, acceptance),
        _label(decision.get("version_action")),
        _comparison(comparison.get("over_parent")),
        _comparison(comparison.get("over_original")),
    )


def _measurement(record: Any) -> Mapping[str, Any]:
    measurement = _mapping(record)
    return _mapping(measurement.get("summary"))


def _metric_value(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return f"{value:.6g}"
    return _label(value)


def _measurement_rows(
    decision: Mapping[str, Any], baseline: Any
) -> list[tuple[str, str, str, str]]:
    baseline_record = _mapping(baseline)
    original = baseline_record.get("measurement") or baseline_record.get("measurements")
    series = (
        ("候选", decision.get("measurement")),
        ("父版", decision.get("parent_measurement")),
        ("原始基线", original),
    )
    rows = []
    for label, record in series:
        values = _measurement(record)
        if values:
            rows.append(
                (
                    label,
                    _metric_value(values.get("throughput")),
                    _metric_value(values.get("ttft_ms")),
                    _metric_value(values.get("tpot_ms")),
                )
            )
    return rows


def _quality_text(row: Mapping[str, Any]) -> list[str]:
    lines = []
    for key, label in (("correctness", "服务正确性"), ("kernel_correctness", "算子正确性")):
        result = _mapping(row.get(key))
        passed = result.get("pass")
        if not isinstance(passed, bool):
            continue
        status = "通过" if passed else "未通过"
        cases = _records(result.get("cases"))
        failed = sum(case.get("pass") is False for case in cases)
        description = f"{label} · {status}"
        if cases:
            description += f" · {len(cases)} 例，失败 {failed} 例"
        lines.append(description)
    return lines


def _rank_list(profile: Mapping[str, Any]) -> list[str]:
    ranks = profile.get("ranks")
    if isinstance(ranks, list):
        return [_label(_pick(rank, "rank", "id") if isinstance(rank, Mapping) else rank) for rank in ranks]
    if isinstance(ranks, Mapping):
        return [_label(rank) for rank in ranks]
    record = profile.get("rank")
    return [_label(record)] if record is not None else []


def _profile_title(profile: Mapping[str, Any]) -> str:
    label = _label(_pick(profile, "capture_id", "task_id", "revision", "id"), "Profile")
    ranks = "，".join(_rank_list(profile)) or "未记录"
    missing = profile.get("missing_ranks")
    missing_label = "，".join(_label(rank) for rank in missing) if isinstance(missing, list) else ""
    suffix = f" · 缺少 rank {missing_label}" if missing_label else ""
    return f"{label} · rank {ranks}{suffix}"


def _timeline_entry(event: Mapping[str, Any]) -> Mapping[str, Any]:
    if event.get("type") != "control_checkpoint":
        return event
    payload = _mapping(event.get("payload"))
    router = _mapping(payload.get("router"))
    inbox = _mapping(payload.get("inbox", router.get("inboxes")))
    return {
        "seq": event.get("seq"),
        "at": event.get("at"),
        "type": "control_checkpoint",
        "payload": {
            "task_count": len(_mapping(payload.get("tasks", _mapping(payload.get("dispatcher")).get("tasks")))),
            "proposal_count": len(_mapping(payload.get("proposals", router.get("proposals")))),
            "inbox_message_count": sum(len(messages) for messages in inbox.values() if isinstance(messages, list)),
        },
    }


def _event_title(event: Mapping[str, Any]) -> str:
    payload = _mapping(event.get("payload"))
    checkpoint = ""
    if event.get("type") == "control_checkpoint":
        checkpoint = (
            f"任务 {payload['task_count']} · 提案 {payload['proposal_count']}"
            f" · 收件 {payload['inbox_message_count']}"
        )
    pieces = (
        _label(event.get("seq"), ""),
        _label(event.get("at"), ""),
        _label(event.get("type"), ""),
        checkpoint or _label(_pick(payload, "role", "agent", "phase", "task_id", "candidate_id"), ""),
    )
    return " · ".join(piece for piece in pieces if piece) or "事件"


def _details(
    title: str, data: Any, links: list[dict[str, Any]] | None = None, body: str = "",
    *, anchor: str | None = None,
) -> str:
    attributes = f' id="{escape(anchor, quote=True)}" class="timeline-event"' if anchor else ""
    return (
        f"<details><summary>{escape(title)}</summary>{body}"
        f"{_evidence_list(links) if links else ''}<pre{attributes}>{escape(_pretty(data))}</pre></details>"
    )


def _evidence_list(entries: list[dict[str, Any]]) -> str:
    items = []
    for entry in entries:
        name = escape(_label(entry.get("path")))
        href = entry.get("href")
        if isinstance(href, str):
            tag = f'<a href="{escape(href, quote=True)}">{name}</a>'
        else:
            tag = name
        if entry["link_state"] != "available":
            tag += f' · {_h(entry["link_state"])}'
        source = _label(_pick(entry, "producer", "kind"), "")
        items.append(f"<li>{tag}{' · ' + escape(source) if source else ''}</li>")
    return "<ul>" + "".join(items) + "</ul>" if items else "<p>尚无记录</p>"


def _heading_data(summary: Mapping[str, Any]) -> list[tuple[str, Any]]:
    scope = _mapping(summary.get("scope"))
    target = _mapping(scope.get("target"))
    task = _mapping(scope.get("task"))
    model = _mapping(task.get("model"))
    workload = _mapping(task.get("workload"))
    return [
        ("停止原因", summary.get("stop_reason")),
        ("当前阶段", summary.get("phase")),
        ("宏循环", summary.get("cycle")),
        ("当前版本", _identity(summary.get("current_revision"))),
        ("原始基线", _identity(_mapping(summary.get("baseline")).get("revision"))),
        ("目标", _pick(target, "id", "name")),
        ("TP", target.get("tp_size")),
        ("模型", _pick(model, "id", "name") or task.get("model")),
        ("工作负载", _pick(workload, "id", "name") or task.get("workload")),
        ("事件水位", summary.get("state_event_seq")),
    ]


_DECISION_COLUMNS = ("候选", "证据充分度", "指标验收", "版本动作", "相对父版", "相对原始基线")
_MEASUREMENT_COLUMNS = ("比较对象", "吞吐 token/s", "TTFT ms", "TPOT ms")


def _html_table(columns: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    headings = "".join(f'<th scope="col">{escape(name)}</th>' for name in columns)
    content = "".join(
        "<tr>" + "".join(f"<td>{escape(item)}</td>" for item in row) + "</tr>" for row in rows
    )
    return f'<div class="table-scroll"><table><thead><tr>{headings}</tr></thead><tbody>{content}</tbody></table></div>'


def _html_decision(row: Mapping[str, Any], summary: dict[str, Any]) -> str:
    candidate, _, _, action, _, _ = _decision_fields(row)
    reason = f'<p>{_h(row.get("reason"), "未记录裁决原因")}</p>'
    quality = "".join(f"<p>{escape(line)}</p>" for line in _quality_text(row))
    values = _measurement_rows(row, summary["baseline"])
    table = _html_table(_MEASUREMENT_COLUMNS, values) if values else ""
    target = next((f"validation-task-{number}-attempt-{position}"
                   for number, graph in enumerate(summary["validation_graphs"], 1)
                   for position, attempt in enumerate(graph["attempts"], 1)
                   if attempt["candidate_id"] == candidate and attempt.get("primary")), None)
    jump = f'<p><a href="#{target}">查看该候选的验证交接图</a></p>' if target else ""
    return _details(f"候选 {candidate} · {action}", row, _evidence_for(row, summary["evidence"]), jump + reason + quality + table)


_FLOW_STATES = {
    "requested": "已发出", "succeeded": "已完成", "recorded": "有记录",
    "failed": "失败", "skipped": "跳过", "running": "正在执行", "ok": "完成",
    "blocked": "被阻断", "partial": "记录不完整", "interrupted": "中断",
    "cleanup_pending": "清理待确认", "cancelled": "已取消",
}
_FLOW_ORIGINS = {"event": "会话事件", "instrumented": "执行记录", "reconstructed": "证据还原"}
_FLOW_SCOPES = {"evaluation": "本次评测", "candidate": "候选", "parent": "父版", "revision": "被测版本",
                "kernel": "算子", "forward": "模型 forward", "profile": "采集目标"}
_FLOW_OPERATIONS = {
    "save_contract": "封存工作点、参考结果与评测规则",
    "probe": "探测目标环境与可用能力", "probe_identity": "核对设备、rank 与环境身份",
    "prepare": "准备指定版本的独立工作区", "launch": "启动实例并返回加载状态",
    "instance_identity": "核对实际加载的版本、组合与工作点", "reference": "取冻结参考输出",
    "observe_outputs": "运行固定样例并返回逐项输出", "correctness": "核对输出及容差",
    "device_timer": "记录算子设备计时", "device_timing_summary": "校验并汇总设备样本",
    "forward_timer": "记录模型 forward 计时", "forward_timing_summary": "校验并汇总 forward 样本",
    "benchmark_sample": "按固定请求执行一个计时轮次", "validate_sample": "校验样本归属、工作点和 token 数",
    "measurement_summary": "汇总吞吐、TTFT 和 TPOT", "collect_profile": "采集各 rank 的时间线",
    "parse_profile": "核对 rank 并归并热点", "cleanup": "关停实例并确认资源释放",
    "paired_comparison": "核对口径并比较候选、父版与原始基线", "decision": "保存本次评测状态与原因",
}
_FLOW_REPLIES = {
    "save_contract": "取得已保存评测约定的引用", "probe": "取得实际环境并准备核对身份",
    "probe_identity": "接收环境比对结论", "prepare": "取得已准备的工作区", "launch": "取得启动凭据",
    "instance_identity": "记录加载身份的核对结果", "observe_outputs": "取得逐项输出，交正确性校验",
    "correctness": "依据校验结论决定是否继续测量", "device_timer": "取得设备计时样本",
    "device_timing_summary": "收取设备计时汇总", "forward_timer": "取得 forward 计时样本",
    "forward_timing_summary": "收取 forward 计时汇总", "benchmark_sample": "取得逐请求记录，校验该轮样本",
    "validate_sample": "接收该轮样本的校验结论", "measurement_summary": "收取指标汇总",
    "collect_profile": "取得采集结果，交 trace 解析", "parse_profile": "收取各 rank 的解析结论",
    "cleanup": "根据回执确认实例和资源是否释放", "paired_comparison": "接收候选与比较版本的结果",
    "decision": "取得评测状态记录的引用",
}


def _flow_reference(ref: Mapping[str, Any], index: dict[tuple[str, str], dict[str, Any]]) -> str:
    path = _h(ref.get("path"))
    entry = index.get(_reference_key(ref)) if isinstance(ref.get("path"), str) else None
    if entry and entry.get("link_state") == "available" and entry.get("href"):
        return f'<a class="flow-file" href="{escape(entry["href"], quote=True)}">{path}</a>'
    state = entry.get("link_state") if entry else "missing"
    return f'<span class="flow-file">{path} · {_h(state)}</span>'


def _flow_steps(rows: list[dict[str, Any]], index: dict[tuple[str, str], dict[str, Any]]) -> str:
    items = []
    for row in rows:
        state_name = row["state"]
        state = _h(_FLOW_STATES.get(state_name, state_name))
        origin = _h(_FLOW_ORIGINS.get(row["origin"], row["origin"]))
        operation_key = row["operation"]
        operation = _FLOW_OPERATIONS.get(operation_key, operation_key)
        if row["origin"] == "instrumented" and state_name in {"succeeded", "failed"}:
            operation = _FLOW_REPLIES.get(operation_key, operation)
        if state_name == "skipped":
            operation = "该步骤未执行，按本次状态继续收尾"
        operation = _h(operation)
        event = row.get("event_seq")
        trace = f'<a href="#event-{event}">事件 #{event}</a>' if type(event) is int else ""
        evidence = "".join(_flow_reference(ref, index) for ref in row.get("evidence", []))
        scope_label = _FLOW_SCOPES.get(row.get("scope"))
        scope = f"<span>{escape(scope_label)}</span>" if scope_label else ""
        items.append(
            f'<li class="flow-step" data-state="{_h(state_name)}">'
            f'<div class="flow-actor"><small>发出</small><strong>{_h(row["sender"])}</strong></div>'
            f'<div class="flow-transfer">{_h(row["message"])}</div>'
            f'<div class="flow-actor"><small>接收</small><strong>{_h(row["receiver"])}</strong><p>{operation}</p></div>'
            f'<div class="flow-proof"><span class="flow-state">{state}</span><span>{origin}</span>{scope}{trace}{evidence}</div></li>'
        )
    return f'<ol class="flow-list" role="list">{"".join(items)}</ol>' if items else ""


def _validation_section(summary: dict[str, Any]) -> str:
    index = {_reference_key(ref): ref for ref in summary["evidence"]}
    tasks = []
    for number, graph in enumerate(summary["validation_graphs"], 1):
        title = f'{graph["title"]} · {graph["task_id"]}'
        if graph["status"]:
            title += f' · {_FLOW_STATES.get(graph["status"], graph["status"])}'
        attempts = []
        for position, attempt in enumerate(graph["attempts"], 1):
            origin = _FLOW_ORIGINS[attempt["origin"]]
            trace = (f'<p class="flow-caption">{_flow_reference(attempt["trace"], index)}</p>'
                     if attempt["trace"] else "")
            attempts.append(
                f'<details class="flow-attempt"><summary>{_h(attempt["label"])} · {_h(attempt["attempt_id"])}'
                f' · {_h(origin)}</summary><span class="flow-anchor" id="validation-task-{number}-attempt-{position}"></span>'
                f'{trace}'
                f'{_flow_steps(attempt["steps"] + attempt["after"], index) or "<p>暂无已记录的交接。</p>"}</details>'
            )
        reason = f'<p class="flow-caption">记录原因 · {_h(graph["reason"])}</p>' if graph["reason"] else ""
        tasks.append(
            f'<details class="validation-task" id="validation-task-{number}"><summary>{_h(title)}</summary>'
            f'{reason}{_flow_steps(graph["first"], index)}'
            f'{"".join(attempts) or "<p>本任务无可展示的评测尝试。</p>"}'
            f'{_flow_steps(graph["last"], index)}</details>'
        )
    return "".join(tasks) or "<p>尚无验证任务。</p>"


def _report_html(summary: dict[str, Any]) -> str:
    heading = _heading_data(summary)
    mode = _h(summary["mode"])
    labels = "".join(f"<dt>{escape(name)}</dt><dd>{_h(value)}</dd>" for name, value in heading)
    sections = {
        "overview": "运行范围",
        "validation": "验证交接图",
        "decisions": "候选验收",
        "profiles": "采集与扫描",
        "timeline": "执行时间线",
        "evidence": "原始证据",
    }
    navigation = "".join(f'<a href="#{key}">{title}</a>' for key, title in sections.items())
    rows = [_decision_fields(row) for row in summary["decisions"]]
    evidence = summary["evidence"]
    decisions = "".join(_html_decision(row, summary) for row in summary["decisions"])
    profiles = "".join(
        _details(_profile_title(row), row, _evidence_for(row, evidence)) for row in summary["profiles"]
    )
    sweeps = "".join(
        _details(f"扫描 {index}", row, _evidence_for(row, evidence))
        for index, row in enumerate(summary["sweeps"], 1)
    )
    stack = "".join(
        _details(f"整栈复测 {index}", row, _evidence_for(row, evidence))
        for index, row in enumerate(summary["stack_validations"], 1)
    )
    events = "".join(_details(_event_title(event), event, anchor=f"event-{event['seq']}")
                     for event in summary["timeline"])
    baseline = _details(
        "原始基线的测量与证据", summary["baseline"], _evidence_for(summary["baseline"], evidence)
    )
    warnings = "".join(f"<li>{escape(item)}</li>" for item in summary["report_warnings"])
    return f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hyperloop Session · {_h(summary["session_id"])}</title><style>{_STYLES}</style></head>
<body><header><h1>Hyperloop Session · {_h(summary["session_id"])}</h1>
<p><strong class="mode">{mode}</strong></p><nav aria-label="报告目录">{navigation}
<a href="report.md">Markdown</a><a href="summary.json">JSON</a></nav></header>
<main>
<section id="overview"><h2>运行范围</h2><dl>{labels}</dl>{baseline}
{_details("任务及评测规则", summary["scope"])}
{_details("贡献来源", summary["contributions"])}</section>
<section id="validation"><h2>验证交接图</h2>
<p>箭头上是发送的内容，接收侧写明执行操作。每条边标出会话事件、执行记录或证据还原。</p>
{_validation_section(summary)}</section>
<section id="decisions"><h2>候选验收</h2>
{_html_table(_DECISION_COLUMNS, rows) if rows else '<p>尚无候选裁决</p>'}
{decisions}</section>
<section id="profiles"><h2>采集与扫描</h2>{profiles or "<p>尚无 Profile 记录</p>"}
{sweeps or "<p>尚无 SWEEP 记录</p>"}{stack}</section>
<section id="timeline"><h2>执行时间线</h2>{events or "<p>尚无事件</p>"}</section>
<section id="evidence"><h2>原始证据</h2><p>裁决取自已提交事件，文件状态反映本次读取。</p>
{f"<ul>{warnings}</ul>" if warnings else ""}
{_evidence_list(summary["evidence"])}</section>
</main></body></html>
"""


def _md_details(title: str, data: Any) -> list[str]:
    return [
        "",
        f"### {_md(title)}",
        "",
        *("    " + line for line in _pretty(data).splitlines()),
        "",
    ]


def _md_table(columns: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    header = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    return [*header, *("| " + " | ".join(_md(item) for item in row) + " |" for row in rows)]


def _report_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Hyperloop Session · {_md(summary['session_id'])}",
        "",
        f"**运行模式 · {_md(summary['mode'])}**",
        "",
        "[离线 HTML](report.html) · [结构化摘要](summary.json)",
        "",
        "## 运行范围",
        "",
    ]
    lines.extend(f"- {name} · {_md(value)}" for name, value in _heading_data(summary))
    lines.extend(_md_details("原始基线的测量与证据", summary["baseline"]))
    lines.extend(_md_details("任务及评测规则", summary["scope"]))
    lines.extend(_md_details("贡献来源", summary["contributions"]))
    lines.extend(["## 验证交接图", "", "[在 HTML 中逐次查看发送方、接收方和证据](report.html#validation)", ""])
    lines.extend(["## 候选验收", ""])
    decisions = summary["decisions"]
    if decisions:
        lines.extend(_md_table(_DECISION_COLUMNS, [_decision_fields(row) for row in decisions]))
    else:
        lines.append("尚无候选裁决")
    for row in decisions:
        fields = _decision_fields(row)
        lines.extend(["", f"### 候选 {_md(fields[0])} · {_md(fields[3])}", "", _md(row.get("reason")), ""])
        lines.extend(f"- {_md(line)}" for line in _quality_text(row))
        values = _measurement_rows(row, summary["baseline"])
        if values:
            lines.extend(["", *_md_table(_MEASUREMENT_COLUMNS, values)])
        lines.extend(_md_details("裁决记录", row))
    lines.extend(["", "## 采集与扫描", ""])
    for row in summary["profiles"]:
        lines.extend(_md_details(_profile_title(row), row))
    for index, row in enumerate(summary["sweeps"], 1):
        lines.extend(_md_details(f"扫描 {index}", row))
    for index, row in enumerate(summary["stack_validations"], 1):
        lines.extend(_md_details(f"整栈复测 {index}", row))
    if not (summary["profiles"] or summary["sweeps"] or summary["stack_validations"]):
        lines.append("尚无采集或扫描记录")
    lines.extend(["", "## 执行时间线", ""])
    lines.extend(f"- {_md(_event_title(row))}" for row in summary["timeline"])
    lines.extend(["", "## 原始证据", "", "裁决取自已提交事件，文件状态反映本次读取。", ""])
    lines.extend(f"- {_md(warning)}" for warning in summary["report_warnings"])
    for entry in summary["evidence"]:
        name = _md(entry["path"])
        if entry["href"] is None:
            lines.append(f"- {name} · {_md(entry['link_state'])}")
        else:
            state = f" · {_md(entry['link_state'])}" if entry["link_state"] != "available" else ""
            lines.append(f"- [{name}]({entry['href']}){state}")
    if not summary["evidence"]:
        lines.append("尚无记录")
    return "\n".join(lines) + "\n"


def write_report(
    session_dir: Path, manifest: dict[str, Any], state: dict[str, Any], events: list[dict[str, Any]]
) -> dict[str, Any]:
    """Publish report.md, report.html and summary.json from supplied replay state."""
    root = session_dir.resolve(strict=True)
    index, warnings = _evidence_index(root)
    raw_events = _records(events)
    timeline = [_timeline_entry(event) for event in raw_events]
    scope = {key: manifest[key] for key in ("task", "target", "created_at", "fingerprint") if key in manifest}
    scope["runtime"] = state.get("runtime") or manifest.get("runtime")
    reproductions = _reproduction_receipts(root, manifest)
    evidence = _index_entries(root, index, state, raw_events, reproductions)
    entries = {_reference_key(ref): ref for ref in evidence}

    def read_graph_evidence(ref: Mapping[str, Any]) -> dict[str, Any] | None:
        return _read_graph_evidence(root, entries, ref)

    summary = {
        "schema_version": 1,
        "session_id": manifest.get("session_id", root.name),
        "mode": manifest.get("mode", _mapping(manifest.get("target")).get("mode")),
        "phase": state.get("phase"),
        "cycle": state.get("cycle"),
        "state_event_seq": state.get("last_seq"),
        "scope": scope,
        "contributions": manifest.get("contributions", []),
        "baseline": state.get("baseline"),
        "current_revision": state.get("current_revision"),
        "decisions": _records(state.get("decisions")),
        "profiles": _records(state.get("profiles")),
        "sweeps": _records(state.get("sweeps")),
        "stack_validations": _records(state.get("stack_validations")),
        "timeline": timeline,
        "evidence": evidence,
        "validation_graphs": build_validation_graphs(state, raw_events, read_graph_evidence)
        + build_reproduction_graphs(reproductions, read_graph_evidence),
        "stop_reason": state.get("stop_reason"),
        "report_warnings": warnings,
    }
    atomic_write(root / "report.md", _report_markdown(summary).encode("utf-8"))
    atomic_write(root / "report.html", _report_html(summary).encode("utf-8"))
    atomic_write(root / "summary.json", json_bytes(summary))
    return summary


def _reproduction_receipts(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    folder = root / "evidence" / "reproductions"
    if not manifest.get("fingerprint") or not folder.resolve().is_relative_to(root):
        return []
    receipts = []
    for item in sorted(folder.glob("reproduce-*.json"), key=lambda path: path.name):
        if not re.fullmatch(r"reproduce-[0-9a-f]{12}\.json", item.name):
            continue
        relative = f"evidence/reproductions/{item.name}"
        local, _ = _local_file(root, relative)
        if local is None:
            continue
        try:
            raw = local.read_bytes()
            data = _mapping(json.loads(raw))
            json_bytes(data)
        except (OSError, UnicodeError, ValueError):
            continue
        result = _mapping(data.get("result"))
        if (
            data.get("session_fingerprint") == manifest["fingerprint"]
            and result.get("attempt_id") == item.stem and result.get("mode") == manifest.get("mode")
        ):
            digest = hashlib.sha256(raw).hexdigest()
            receipts.append({"result": result, "reference": {
                "path": relative, "sha256": digest, "producer": "independent_reproduction",
            }})
    return receipts


def _read_graph_evidence(
    root: Path, entries: dict[tuple[str, str], dict[str, Any]], reference: Mapping[str, Any],
) -> dict[str, Any] | None:
    path = reference.get("path")
    digest = reference.get("sha256")
    if not isinstance(path, str) or not isinstance(digest, str):
        return None
    entry = entries.get(_reference_key(reference))
    if not entry or entry["link_state"] != "available":
        return None
    local, _ = _local_file(root, path)
    if not local:
        return None
    try:
        raw = local.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            return None
        data = json.loads(raw)
        json_bytes(data)
    except (OSError, UnicodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None
